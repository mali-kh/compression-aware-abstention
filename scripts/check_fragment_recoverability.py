"""D5 — Fragment-recoverability sanity check.

GATING: if a clean Qwen2.5-7B-Instruct can recover the gold answer from the
decoded kept-tokens of an example we labelled ABSTAIN, our ABSTAIN labels
are too aggressive and training on them will teach the model to abstain
when it shouldn't.

Pipeline:
  1. Collect 30 ABSTAIN examples at ratio=0.30 with tight spans, no
     FLAGGED_PARTIAL (matches the v1 binary commitment).
  2. For each: extract the kept tokens via the cache mask, decode back
     to text. The result looks like Swiss cheese — that *is* the
     post-compression representation, semantically.
  3. Free the wrapper's small model. Load a fresh Qwen2.5-7B-Instruct
     (ungated, ~15 GB) — clean prompt, no compressed cache — and ask
     it the question with the decoded fragment as context.
  4. Score each response: case-insensitive substring match of gold
     answer or any alias counts as recovery; first-line exact match
     is also recorded for stricter accounting.

Verdict rule:
   <= 10% recovery (<=  3/30): labels clean, proceed to D6.
   10–30%  recovery: labels noisy. Tighten tau_low 0.3 -> 0.2 and re-run.
    > 30%  recovery: STOP. Labels wrong. Surface options.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compressors.kvzip import KVzipCompressor  # noqa: E402
from src.data.label_construction import (  # noqa: E402
    CacheState,
    LabelConfig,
    LabelType,
    label_example,
)
from src.data.musique import normalize_musique_example  # noqa: E402

OUT_JSONL = ROOT / "artifacts" / "fragment_recoverability.jsonl"
OUT_MD = ROOT / "artifacts" / "fragment_recoverability_inspect.md"


def topk_mask(frac_kept: torch.Tensor, ratio: float) -> torch.Tensor:
    """Same reduction the wrapper uses with reduction='topk'."""
    ctx_len = frac_kept.size(0)
    k = int(round(ratio * ctx_len))
    if k <= 0:
        return torch.zeros(ctx_len, dtype=torch.bool)
    if k >= ctx_len:
        return torch.ones(ctx_len, dtype=torch.bool)
    _, top_idx = torch.topk(frac_kept, k)
    mask = torch.zeros(ctx_len, dtype=torch.bool)
    mask.scatter_(0, top_idx, True)
    return mask


def collect_abstain_examples(
    *,
    target: int,
    ratio: float,
    max_tokens: int,
    max_scan: int,
):
    """Iterate Musique val until we have `target` ABSTAIN examples.

    Returns a list of dicts with everything we need for the judging step,
    so we can free the KVzip-wrapped model before loading the 7B judge.
    """
    print(f"  loading KVzip wrapper (qwen3-0.6b)...", flush=True)
    wrapper = KVzipCompressor(model_name="qwen3-0.6b")
    print(f"  loading Musique validation split...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    cfg = LabelConfig()  # binary; enable_flagged_partial=False default
    out: list[dict] = []
    scanned = 0
    for row in ds:
        if scanned >= max_scan:
            break
        scanned += 1
        if not row.get("answerable", True):
            continue
        ex = normalize_musique_example(row, wrapper.tokenizer, span_granularity="tight")
        if not ex.supporting_spans:
            continue
        if any(s.end > max_tokens for s in ex.supporting_spans):
            continue
        if ex.context_ids.size(0) > max_tokens:
            continue

        ids_dev = ex.context_ids.to(wrapper._impl.device).unsqueeze(0)
        kv = wrapper._impl.prefill(ids_dev, load_score=False)
        kv.prune(ratio=ratio, level="pair")
        slots = kv.valid.reshape(-1, kv.valid.size(-1))
        frac_kept = slots.float().mean(dim=0).detach().cpu()
        mask = topk_mask(frac_kept, ratio)

        cs = CacheState(kept_mask=mask, compressor_name="kvzip", ratio=ratio)
        label = label_example(
            context=ex.context,
            query=ex.question,
            answer=ex.answer,
            supporting_spans=ex.supporting_spans,
            cache_state=cs,
            cfg=cfg,
        )
        if label.type is not LabelType.ABSTAIN:
            continue

        kept_ids = ex.context_ids[mask]
        decoded = wrapper.tokenizer.decode(kept_ids, skip_special_tokens=True)
        out.append(
            {
                "id": ex.id,
                "question": ex.question,
                "answer": ex.answer,
                "answer_aliases": ex.answer_aliases,
                "ctx_len": int(ex.context_ids.size(0)),
                "n_kept": int(mask.sum().item()),
                "ratio": ratio,
                "span_retentions": [round(sr.fraction_kept, 3) for sr in label.span_retentions],
                "decoded_fragment": decoded,
            }
        )
        if len(out) >= target:
            break

    # free the small model before the 7B judge loads
    print(f"  collected {len(out)} ABSTAIN examples after scanning {scanned}", flush=True)
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def is_recovered(response: str, gold: str, aliases: list[str]) -> tuple[bool, str]:
    candidates = [gold] + (aliases or [])
    response_clean = response.strip().lower()
    first_line = response_clean.split("\n", 1)[0].strip()
    for cand in candidates:
        c = cand.strip().lower()
        if not c:
            continue
        if first_line == c:
            return True, f"exact_first_line:{cand}"
    for cand in candidates:
        c = cand.strip().lower()
        if not c:
            continue
        if c in response_clean:
            return True, f"substring:{cand}"
    return False, ""


@torch.inference_mode()
def judge_examples(examples: list[dict], judge_model_id: str) -> list[dict]:
    print(f"  loading judge model: {judge_model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # sdpa avoids any cross-talk with KVzip's monkey-patched flash path
        # (only Qwen3 was patched, but sdpa is the safe choice)
        attn_implementation="sdpa",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: list[dict] = []
    for i, ex in enumerate(examples, 1):
        prompt = (
            f"Context: {ex['decoded_fragment']}\n\n"
            f"Question: {ex['question']}\n\n"
            f"Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in = inputs.input_ids.size(1)
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()
        recovered, match = is_recovered(response, ex["answer"], ex["answer_aliases"])
        results.append({**ex, "judge_response": response, "recovered": recovered, "match_type": match})
        marker = "REC" if recovered else "abst"
        print(
            f"    [{i:2d}/{len(examples)}] {ex['id']}  {marker}  "
            f"gold={ex['answer']!r}  resp={response[:60]!r}",
            flush=True,
        )
    return results


def write_inspection_md(results: list[dict], rate: float, judge_model_id: str, ratio: float) -> None:
    md: list[str] = []
    md.append("# Fragment recoverability — inspection set\n")
    md.append("_Auto-generated by `scripts/check_fragment_recoverability.py`._\n")
    md.append("## Setup\n")
    md.append(f"- judge model (clean prompt, no compressed cache): `{judge_model_id}`")
    md.append(f"- compression ratio: {ratio}")
    md.append(f"- span granularity: tight (sub-answer via question_decomposition)")
    md.append(f"- labeler: binary (FLAGGED_PARTIAL disabled), tau_low=0.3, tau_high=0.8")
    md.append(f"- fragment_recovery_rate: **{rate:.1%}** "
              f"({sum(1 for r in results if r['recovered'])}/{len(results)})")
    md.append("")
    md.append("Each example below was labelled ABSTAIN by our pipeline. The decoded")
    md.append("fragment is the kept-token text — Swiss cheese, but representative of")
    md.append("what the post-compression KV cache contains semantically. If the judge")
    md.append("model can answer correctly from the fragment alone, our ABSTAIN label")
    md.append("is too aggressive on this example.\n")
    md.append("---\n")
    for i, r in enumerate(results[:10], 1):
        verdict = "❌ recovered (label may be wrong)" if r["recovered"] else "✅ judge also abstains/fails"
        md.append(f"## {i}. `{r['id']}` — {verdict}")
        md.append(f"**Question**: {r['question']}")
        md.append(f"**Gold answer**: `{r['answer']}`  (aliases: {r['answer_aliases']})")
        md.append(f"**Span retentions**: {r['span_retentions']}")
        md.append(f"**Kept tokens**: {r['n_kept']} / {r['ctx_len']}")
        md.append(f"**Match type**: `{r['match_type'] or '—'}`")
        md.append(f"**Judge response**:")
        md.append("```")
        md.append(r["judge_response"][:600])
        md.append("```")
        md.append(f"**Decoded fragment** (first 1500 chars):")
        md.append("```")
        md.append(r["decoded_fragment"][:1500])
        md.append("```\n---\n")
    OUT_MD.write_text("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-scan", type=int, default=300,
                    help="hard cap on Musique rows scanned to find n ABSTAIN examples")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    print(f"[D5] target_n={args.n} ratio={args.ratio} judge={args.judge_model}", flush=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    print("[step 1/3] collecting ABSTAIN examples...", flush=True)
    t0 = time.time()
    examples = collect_abstain_examples(
        target=args.n,
        ratio=args.ratio,
        max_tokens=args.max_tokens,
        max_scan=args.max_scan,
    )
    if len(examples) < args.n:
        print(
            f"WARNING: only {len(examples)} ABSTAIN examples found in first "
            f"{args.max_scan} rows (target {args.n}). Continuing with what we have."
        )
    print(f"  ({time.time()-t0:.1f}s)", flush=True)

    print("\n[step 2/3] judging fragments with clean Qwen2.5-7B-Instruct...", flush=True)
    t0 = time.time()
    results = judge_examples(examples, args.judge_model)
    print(f"  ({time.time()-t0:.1f}s)", flush=True)

    n_recovered = sum(1 for r in results if r["recovered"])
    rate = n_recovered / len(results) if results else 0.0

    print("\n[step 3/3] writing artifacts...", flush=True)
    with OUT_JSONL.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    write_inspection_md(results, rate, args.judge_model, args.ratio)
    print(f"  wrote {OUT_JSONL.relative_to(ROOT)}", flush=True)
    print(f"  wrote {OUT_MD.relative_to(ROOT)}", flush=True)

    print(f"\n=== fragment_recovery_rate = {rate:.3f} ({n_recovered}/{len(results)}) ===")
    if rate <= 0.10:
        print("VERDICT: <=10% — labels clean, proceed to D6")
    elif rate <= 0.30:
        print("VERDICT: 10-30% — labels noisy, tighten tau_low to 0.2 and re-run")
    else:
        print("VERDICT: >30% — STOP. Labels wrong. Surface options.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
