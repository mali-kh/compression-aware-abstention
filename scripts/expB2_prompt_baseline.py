"""Exp B2 — Prompt-only abstention baseline.

Run base Qwen2.5-7B-Instruct (no LoRA) on a val_v1*.jsonl with a system
prompt prepended to the eval prompt, classify outputs under v1's existing
classifier. Output JSONL has the same shape as v1_eval_*.jsonl's `base`
column for downstream comparison.

Use the existing v1 eval prompt template and just prepend the system
prompt — same generation config as Phase 3 (greedy, max_new_tokens=64).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.honest_accuracy import classify_response, looks_like_abstention  # noqa: E402

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE = "\n\nQuestion: {query}\n\nAnswer:"

P2 = (
    'If the available context does not contain enough evidence to answer the '
    'question, respond with: "I cannot determine the answer to this question '
    'from the available context." Otherwise, answer the question.'
)
P3 = (
    'You may have only partial access to the original context due to memory '
    'compression. Some details may be missing. Answer only if the question is '
    'clearly supported by the text you can read; otherwise abstain with the '
    'exact phrase "I cannot determine the answer to this question from the '
    'available context."'
)
PROMPTS = {"P2": P2, "P3": P3}


def build_input_ids(record: dict, tokenizer, system_prompt: str, max_seq_length: int = 4096) -> list[int]:
    """Mirror of v1's build_eval_input_ids but with a system prompt prepended."""
    context_ids = tokenizer(record["context"], add_special_tokens=False).input_ids
    if len(context_ids) != len(record["kept_mask"]):
        raise ValueError("kept_mask length mismatch")
    kept = [tid for tid, k in zip(context_ids, record["kept_mask"]) if k]
    sys_ids = tokenizer(system_prompt + "\n\n", add_special_tokens=False).input_ids
    prefix = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
    middle = tokenizer(PROMPT_MIDDLE.format(query=record["query"]), add_special_tokens=False).input_ids
    input_ids = sys_ids + prefix + kept + middle
    if len(input_ids) > max_seq_length - 96:
        raise ValueError(f"eval input too long: {len(input_ids)}")
    return input_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--prompt", required=True, choices=list(PROMPTS),
                    help="P2 (light) or P3 (strong + memory-compression-aware)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val_jsonl) if l.strip()]
    rows = [r for r in rows if r["label_type"] in ("abstain", "confident_answer")]
    print(f"[setup] loaded {len(rows)} eval-able rows from {args.val_jsonl}", flush=True)
    print(f"[setup] system prompt: {args.prompt}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[setup] loading {args.base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    sys_prompt = PROMPTS[args.prompt]
    out_rows: list[dict] = []
    t0 = time.time()
    skipped_too_long = 0
    for i, r in enumerate(rows):
        try:
            input_ids = build_input_ids(r, tokenizer, sys_prompt, args.max_seq_length)
        except ValueError:
            skipped_too_long += 1
            continue
        ids = torch.tensor([input_ids], device=model.device)
        with torch.inference_mode():
            gen = model.generate(
                ids, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()
        cls = classify_response(
            response,
            gold_answer=r["metadata"]["answer"],
            answer_aliases=r["metadata"].get("answer_aliases", []),
            gold_label=r["label_type"],
        )
        out_rows.append({
            "id": r["id"], "ratio": r["ratio"],
            "gold_label": r["label_type"], "gold_answer": r["metadata"]["answer"],
            "response": response,
            "classification": cls,
            "model_abstained": looks_like_abstention(response),
            "prompt": args.prompt,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  eval {i+1}/{len(rows)}  ({elapsed:.0f}s)", flush=True)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] wrote {len(out_rows)} rows to {args.out_jsonl} "
          f"(skipped {skipped_too_long} too-long inputs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
