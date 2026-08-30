"""Block 10 — adversarial / controlled-deletion evaluations.

Three sub-experiments under prompt-style inference:

  10A: fixed-retained-length L=500
       For each val record with sum(kept_mask) >= 500, take the first
       L=500 True positions of the existing kept_mask (in context order).
       This is a length-controlled subset that preserves KVzip-rank
       priority within the subset (the existing kept set IS the top-by-
       KVzip selection at the example's ratio).

  10B: adversarial-tight-span eviction
       For each CONFIDENT-gold record, force-evict all tight-span tokens
       (from supporting_spans), then refill an equal number of random
       non-kept positions (seed 42) so retention ratio is preserved.

  10C: anti-priority paired (r=0.50 only)
       For each r=0.50 CONFIDENT-gold record, generate two masks at the
       same ~50% retention:
         Mask A: existing kept_mask (KVzip top-50%)
         Mask B: flipped (KVzip bottom-50% — "worst possible 50%")

For each sub-experiment we run mixture-ps and mixture-cc adapters. We also
need mixture-cc baseline (original-ratio) PS results on the 10A and 10B
subsets so that 10A/10B's deltas are apples-to-apples.

Output JSONLs (per adapter, per sub-experiment) follow the same schema as
v1_eval*.jsonl `adapter` blocks (flat). Filenames:
  artifacts/block10a_eval_{adapter}.jsonl              (fixed-L)
  artifacts/block10a_baseline_eval_{adapter}.jsonl     (original-ratio on same subset)
  artifacts/block10b_eval_{adapter}.jsonl              (adversarial)
  artifacts/block10b_baseline_eval_{adapter}.jsonl     (original-ratio on same subset)
  artifacts/block10c_eval_{adapter}_maskA.jsonl
  artifacts/block10c_eval_{adapter}_maskB.jsonl

Run via .venv-kvpress.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# import classify_response without triggering torch import in honest_accuracy
def _load_classifier():
    src = ROOT / "src" / "eval" / "honest_accuracy.py"
    spec = importlib.util.spec_from_file_location("_honest", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_response, mod.looks_like_abstention

classify_response, looks_like_abstention = _load_classifier()

VAL_JSONL = ROOT / "artifacts" / "val_v1.jsonl"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTERS = {
    "mixture-ps": ROOT / "artifacts" / "v1_lora_best_mixture_seed42",
    "mixture-cc": ROOT / "artifacts" / "v1_lora_mixture_cc_lr2e-4_seed42_best",
}

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE = "\n\nQuestion: {query}\n\nAnswer:"


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------- mask construction rules ----------------


def mask_10a_fixed_length(record: dict, L: int = 500) -> list[bool] | None:
    """First L True positions of existing kept_mask, in context order.
    Returns None if sum(existing) < L (skip)."""
    existing = record["kept_mask"]
    if sum(existing) < L:
        return None
    new_mask = [False] * len(existing)
    n_taken = 0
    for i, k in enumerate(existing):
        if k and n_taken < L:
            new_mask[i] = True
            n_taken += 1
    return new_mask


def mask_10b_adversarial(record: dict, seed: int = 42) -> list[bool] | None:
    """Existing kept_mask MINUS tight-span tokens, refilled with random
    non-kept-and-non-tight positions to preserve retention ratio."""
    existing = list(record["kept_mask"])
    spans = record["metadata"].get("supporting_spans", [])
    if not spans:
        return None
    tight: set[int] = set()
    for s in spans:
        # supporting_spans is INCLUSIVE on both ends per probe (start, end with end-start=2 → 3 tokens)
        for i in range(s["start"], s["end"] + 1):
            tight.add(i)
    n_evicted_from_tight = 0
    for i in tight:
        if 0 <= i < len(existing) and existing[i]:
            existing[i] = False
            n_evicted_from_tight += 1
    if n_evicted_from_tight == 0:
        # tight spans were already not in the kept_mask; nothing to refill
        return existing
    candidates = [i for i in range(len(existing))
                  if not existing[i] and i not in tight]
    rng = random.Random((seed, record["id"], record["ratio"]))
    rng.shuffle(candidates)
    n_refill = min(n_evicted_from_tight, len(candidates))
    for i in candidates[:n_refill]:
        existing[i] = True
    return existing


def mask_10c_flipped(record: dict) -> list[bool]:
    """Bit-flipped existing kept_mask = KVzip bottom-50%."""
    return [not k for k in record["kept_mask"]]


# ---------------- prompt-style eval (reused from run_v1_full) ----------------


def build_eval_input_ids(kept_mask: list[bool], record: dict, tokenizer,
                         max_seq_length: int = 4096) -> list[int]:
    context_ids = tokenizer(record["context"], add_special_tokens=False).input_ids
    if len(context_ids) != len(kept_mask):
        raise ValueError(
            f"kept_mask length {len(kept_mask)} != context tokens {len(context_ids)}"
        )
    kept = [tid for tid, k in zip(context_ids, kept_mask) if k]
    prefix = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
    middle = tokenizer(PROMPT_MIDDLE.format(query=record["query"]),
                       add_special_tokens=False).input_ids
    input_ids = prefix + kept + middle
    if len(input_ids) > max_seq_length - 96:
        raise ValueError(f"eval input too long: {len(input_ids)}")
    return input_ids


@torch.no_grad()
def evaluate_with_mask(model, tokenizer, records, mask_fn, *,
                       extra_tag: str = "",
                       max_new_tokens: int = 64,
                       progress_every: int = 25) -> list[dict]:
    """Apply mask_fn(record) per record, then prompt-style generate."""
    model.eval()
    out: list[dict] = []
    t0 = time.time()
    n_skip = 0
    for i, r in enumerate(records):
        try:
            kept = mask_fn(r)
        except Exception as e:
            n_skip += 1
            continue
        if kept is None:
            n_skip += 1
            continue
        try:
            input_ids = build_eval_input_ids(kept, r, tokenizer)
        except ValueError:
            n_skip += 1
            continue
        ids = torch.tensor([input_ids], device=model.device)
        gen = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()
        cls = classify_response(
            response,
            gold_answer=r["metadata"]["answer"],
            answer_aliases=r["metadata"].get("answer_aliases", []),
            gold_label=r["label_type"],
        )
        out.append({
            "id": r["id"],
            "ratio": r["ratio"],
            "gold_label": r["label_type"],
            "gold_answer": r["metadata"]["answer"],
            "response": response,
            "classification": cls,
            "model_abstained": looks_like_abstention(response),
            "kept_count_actual": int(sum(kept)),
            "extra_tag": extra_tag,
        })
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"     {extra_tag}  {i+1}/{len(records)}  "
                  f"elapsed={elapsed:.0f}s  skipped={n_skip}", flush=True)
    elapsed = time.time() - t0
    print(f"  {extra_tag} done: {len(out)} ok, {n_skip} skipped, "
          f"{elapsed:.0f}s", flush=True)
    return out


# ---------------- per-adapter orchestration ----------------


def run_for_adapter(adapter_name: str, val_records: list[dict],
                    base_model_name: str = BASE_MODEL):
    adapter_dir = ADAPTERS[adapter_name]
    print(f"\n{'='*72}\n  ADAPTER = {adapter_name}  ({adapter_dir.name})\n{'='*72}",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    base.eval()
    adapted = PeftModel.from_pretrained(base, str(adapter_dir))
    adapted.eval()

    # ---- Subsets ----
    records_10a = [r for r in val_records if sum(r["kept_mask"]) >= 500]
    records_10b = [r for r in val_records
                   if r["label_type"] == "confident_answer"
                   and r["metadata"].get("supporting_spans")]
    records_10c = [r for r in records_10b if abs(r["ratio"] - 0.50) < 1e-9]

    print(f"  subsets: 10A n={len(records_10a)}, 10B n={len(records_10b)}, 10C n={len(records_10c)}",
          flush=True)

    out_root = ROOT / "artifacts"

    def write(out_path: Path, rows: list[dict]):
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out_path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)

    # 10A: fixed-L=500 mask
    rows = evaluate_with_mask(
        adapted, tokenizer, records_10a, mask_10a_fixed_length,
        extra_tag=f"10A {adapter_name}",
    )
    write(out_root / f"block10a_eval_{adapter_name}.jsonl", rows)

    # 10A baseline: original kept_mask on same subset
    rows = evaluate_with_mask(
        adapted, tokenizer, records_10a, lambda r: r["kept_mask"],
        extra_tag=f"10A-base {adapter_name}",
    )
    write(out_root / f"block10a_baseline_eval_{adapter_name}.jsonl", rows)

    # 10B: adversarial mask
    rows = evaluate_with_mask(
        adapted, tokenizer, records_10b, mask_10b_adversarial,
        extra_tag=f"10B {adapter_name}",
    )
    write(out_root / f"block10b_eval_{adapter_name}.jsonl", rows)

    # 10B baseline: original kept_mask on same subset
    rows = evaluate_with_mask(
        adapted, tokenizer, records_10b, lambda r: r["kept_mask"],
        extra_tag=f"10B-base {adapter_name}",
    )
    write(out_root / f"block10b_baseline_eval_{adapter_name}.jsonl", rows)

    # 10C: paired masks at r=0.50 CONFIDENT-gold
    rows = evaluate_with_mask(
        adapted, tokenizer, records_10c, lambda r: r["kept_mask"],
        extra_tag=f"10C-A {adapter_name}",
    )
    write(out_root / f"block10c_eval_{adapter_name}_maskA.jsonl", rows)
    rows = evaluate_with_mask(
        adapted, tokenizer, records_10c, mask_10c_flipped,
        extra_tag=f"10C-B {adapter_name}",
    )
    write(out_root / f"block10c_eval_{adapter_name}_maskB.jsonl", rows)

    # cleanup
    del adapted, base
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="mixture-ps,mixture-cc",
                    help="Comma-separated adapter names from ADAPTERS map.")
    ap.add_argument("--smoke", action="store_true",
                    help="Limit each subset to first 5 records (for smoke test).")
    args = ap.parse_args()

    val = load_jsonl(VAL_JSONL)
    print(f"[setup] val_records: {len(val)}", flush=True)

    if args.smoke:
        val = val[:50]
        print(f"[smoke] truncated to first {len(val)} records", flush=True)

    for name in args.adapters.split(","):
        name = name.strip()
        run_for_adapter(name, val)

    print("\n[done] all sub-experiments complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
