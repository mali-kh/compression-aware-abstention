"""Block 4 judge runner — calls GPT-5 (temp 0) per sampled example.

For each example in artifacts/block4_sample.jsonl:
  - Reconstruct the truncated context (decode kept tokens)
  - Build a judge prompt with: truncated_context, query, gold_answer
  - Ask: "Given ONLY the context above, can the question be answered
    with the reference answer? YES / NO / PARTIAL + brief reason."
  - Save prompt + verdict + reasoning to artifacts/block4_judge_log.jsonl

The judge does NOT see our tight-span label, the original ctx_len, the
mask, or the full original context.

OPENAI_API_KEY must be set in env. The script never writes the key to
disk and never echoes it.

Async with a bounded concurrency limit to respect rate limits.

Run via .venv-kvpress (has openai SDK + transformers tokenizer).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# offline mode for the local Qwen tokenizer (used only to decode kept
# tokens back to text); openai SDK does not need this
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSONL = ROOT / "artifacts" / "block4_sample.jsonl"
LOG_JSONL = ROOT / "artifacts" / "block4_judge_log.jsonl"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"


JUDGE_SYSTEM = (
    "You are an evaluator deciding whether a question can be answered "
    "from a partial context. Be strict and literal. Only answer based on "
    "what is explicitly stated or directly inferable from the provided "
    "context. Do NOT use outside knowledge."
)


def build_prompt(truncated_context: str, query: str, gold_answer: str) -> str:
    return (
        "I will show you (1) a context that has been partially truncated "
        "(some content removed), (2) a question, and (3) a reference "
        "answer. Decide whether the truncated context contains enough "
        "information to derive the reference answer.\n\n"
        "Note: the truncated context may have abrupt cuts because tokens "
        "were dropped from the middle. Read the surviving content as-is.\n\n"
        "=== CONTEXT (truncated) ===\n"
        f"{truncated_context}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {query}\n\n"
        f"Reference answer: {gold_answer}\n\n"
        "Given ONLY the context above (no outside knowledge), can the "
        "question be answered with the reference answer?\n\n"
        "Respond with EXACTLY one of these three labels on the FIRST line:\n"
        "  YES  — the context contains enough information to derive the reference answer\n"
        "  NO   — the context does not contain enough information\n"
        "  PARTIAL — the context contains some but not all of the necessary information\n\n"
        "Then on the SECOND line, give a brief one-sentence justification "
        "(why the context does or does not support the reference answer)."
    )


def parse_verdict(text: str) -> tuple[str, str]:
    """Return (verdict, reasoning). Verdict ∈ {YES, NO, PARTIAL, INVALID}."""
    text = (text or "").strip()
    lines = text.split("\n", 1)
    first = lines[0].strip().upper().rstrip(".:,")
    # tolerate "Answer: YES" or "**YES**" etc.
    for v in ("YES", "NO", "PARTIAL"):
        if v in first:
            verdict = v
            break
    else:
        verdict = "INVALID"
    reasoning = lines[1].strip() if len(lines) > 1 else ""
    if not reasoning and verdict != "INVALID":
        # may have everything on one line; strip the verdict word
        rest = text.lstrip()[len(first):].strip(" .:-—")
        reasoning = rest
    return verdict, reasoning


async def call_one(client, model: str, sample: dict, tokenizer, sem: asyncio.Semaphore,
                   *, max_tokens: int = 200, attempts: int = 3) -> dict:
    async with sem:
        # decode kept tokens to text
        ctx_ids = tokenizer(sample["context"], add_special_tokens=False).input_ids
        if len(ctx_ids) != len(sample["kept_mask"]):
            return {
                "id": sample["id"],
                "ratio": sample["ratio"],
                "label_type": sample["label_type"],
                "error": "tokenizer_length_mismatch",
            }
        kept = [tid for tid, k in zip(ctx_ids, sample["kept_mask"]) if k]
        truncated_ctx = tokenizer.decode(kept, skip_special_tokens=True)

        gold = sample["metadata"]["answer"]
        query = sample["query"]
        prompt = build_prompt(truncated_ctx, query, gold)

        last_err = None
        for attempt in range(attempts):
            try:
                # NOTE: GPT-5 (reasoning model) does not support temp != 1.
                # Spec asked for temp=0; we use seed=42 for best-effort
                # reproducibility and reasoning_effort='low' to limit the
                # reasoning-token budget while preserving judgment quality
                # (the task is read-context-and-classify, not deep reasoning).
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    seed=42,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="low",
                )
                text = resp.choices[0].message.content or ""
                verdict, reasoning = parse_verdict(text)
                usage = resp.usage
                return {
                    "id": sample["id"],
                    "ratio": sample["ratio"],
                    "label_type": sample["label_type"],
                    "stratum": sample.get("_block4_stratum"),
                    "ctx_len_kept_tokens": len(kept),
                    "judge_model": model,
                    "verdict": verdict,
                    "reasoning": reasoning,
                    "raw_text": text,
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                # rate-limit / transient: backoff
                await asyncio.sleep(2 ** attempt)
        return {
            "id": sample["id"],
            "ratio": sample["ratio"],
            "label_type": sample["label_type"],
            "error": last_err,
        }


async def amain(args):
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in env. Aborting.", file=sys.stderr)
        return 2

    print(f"[setup] loading tokenizer {TOKENIZER_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"[setup] loading sample from {SAMPLE_JSONL.relative_to(ROOT)} ...", flush=True)
    samples = [json.loads(l) for l in SAMPLE_JSONL.read_text().splitlines() if l.strip()]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[setup] {len(samples)} examples to judge with model={args.model}", flush=True)

    # resume support: skip ids already in log
    done_ids = set()
    if args.resume and LOG_JSONL.exists():
        for line in LOG_JSONL.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "verdict" in r and r.get("verdict") in ("YES", "NO", "PARTIAL"):
                done_ids.add((r["id"], r["ratio"]))
        print(f"[resume] {len(done_ids)} already judged in log; skipping those", flush=True)
    todo = [s for s in samples if (s["id"], s["ratio"]) not in done_ids]
    print(f"[run] {len(todo)} calls to make", flush=True)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_done = 0
    n_err = 0

    # open log in append mode
    LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    f = LOG_JSONL.open("a")
    try:
        # batch the asyncio.gather to print progress
        batch_size = 25
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            results = await asyncio.gather(*[
                call_one(client, args.model, s, tokenizer, sem,
                         max_tokens=args.max_tokens, attempts=args.attempts)
                for s in batch
            ])
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += len(batch)
            n_err += sum(1 for r in results if "error" in r)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - n_done) / rate if rate > 0 else 0.0
            print(
                f"  progress {n_done}/{len(todo)}  err={n_err}  "
                f"rate={rate:.1f}/s  eta={eta:.0f}s  ({elapsed:.0f}s elapsed)",
                flush=True,
            )
    finally:
        f.close()

    elapsed = time.time() - t0
    print(f"\n[done] {n_done} calls, {n_err} errors in {elapsed:.0f}s", flush=True)
    return 1 if n_err > n_done * 0.05 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5",
                    help="OpenAI model id; spec says GPT-5.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit calls (for smoke test).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--resume", action="store_true",
                    help="Skip ids already in the judge log.")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
