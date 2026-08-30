"""Exp 2a — KVzip phase of the mixture dataset build.

Selects the same 1000 train + 100 val MuSiQue ids as v1 (deterministic via
seed=42 + scan order in build_training_set.py's select_examples), runs KVzip
prefill+score+prune at each of the six ratios, labels via the v1 LabelConfig
defaults, writes ALL post-DROP-filter rows (no per-class balancing — that's
done by build_mixture_merge.py per stratum).

Output: `artifacts/_mixture_raw_kvzip.jsonl`. Schema matches train_v1.jsonl
plus a per-row `_pool` field so the merge step knows train vs val.

Runs in `.venv` (the project venv with transformers==4.51.3 for KVzip).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compressors.kvzip import KVzipCompressor  # noqa: E402
from src.data.label_construction import CacheState, LabelConfig, label_example  # noqa: E402
from src.data.musique import normalize_musique_example  # noqa: E402
# Reuse the v1 build script's pure helpers
sys.path.insert(0, str(ROOT / "scripts"))
from build_training_set import (  # noqa: E402
    package_row,
    select_examples,
    topk_mask,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--ratios", type=str, default="0.05,0.10,0.20,0.30,0.50,0.80")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "_mixture_raw_kvzip.jsonl"))
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    print(f"[setup] model={args.model_name} ratios={ratios}", flush=True)
    print("[setup] loading KVzip wrapper...", flush=True)
    wrapper = KVzipCompressor(model_name=args.model_name)
    cfg = LabelConfig()  # defaults: tau_high=0.8, tau_low=0.3, drop_uncertain=True, FP=False

    print("[setup] loading Musique train split + selecting examples...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    train_pool, val_pool, scanned = select_examples(
        ds, n_train=args.n_train, n_val=args.n_val, max_tokens=args.max_tokens, tokenizer=wrapper.tokenizer,
    )
    print(f"[setup] selected {len(train_pool)} train + {len(val_pool)} val (scanned {scanned})",
          flush=True)

    raw: list[dict] = []
    t0 = time.time()
    for pool_name, pool in [("train", train_pool), ("val", val_pool)]:
        for i, ex in enumerate(pool, 1):
            ids_dev = ex.context_ids.to(wrapper._impl.device).unsqueeze(0)
            kv = wrapper._impl.prefill(ids_dev, load_score=False)
            for ratio in ratios:
                kv.prune(ratio=ratio, level="pair")
                slots = kv.valid.reshape(-1, kv.valid.size(-1))
                frac_kept = slots.float().mean(dim=0).detach().cpu()
                mask = topk_mask(frac_kept, ratio)
                cs = CacheState(kept_mask=mask, compressor_name="kvzip", ratio=ratio)
                label = label_example(
                    context=ex.context, query=ex.question, answer=ex.answer,
                    supporting_spans=ex.supporting_spans, cache_state=cs, cfg=cfg,
                )
                if label.type.value == "drop":
                    continue
                row = {
                    "id": ex.id, "ratio": ratio,
                    "label_type": label.type.value,
                    "target_output": label.target_output,
                    "kept_mask": mask.tolist(),
                    "context": ex.context, "query": ex.question,
                    "label_obj": label, "ex_obj": ex,
                    "n_kept": int(mask.sum().item()),
                }
                packaged = package_row(row, model_name=args.model_name, split=f"mixture_{pool_name}")
                packaged["_pool"] = pool_name
                raw.append(packaged)
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{pool_name} {i}/{len(pool)}] elapsed={elapsed:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] wrote {len(raw)} post-DROP rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
