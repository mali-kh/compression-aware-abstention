"""Exp 2a — kvpress phase of the mixture dataset build.

Selects the same 1000 train + 100 val MuSiQue ids as v1, runs the chosen
kvpress press at each of the six ratios on the production base model
(Qwen2.5-7B-Instruct), labels via the v1 LabelConfig defaults, writes ALL
post-DROP-filter rows.

Output: `artifacts/_mixture_raw_<press>.jsonl`.

Runs in `.venv-kvpress` (transformers >=4.56 for kvpress).
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

from src.compressors.kvpress_wrapper import KVPressCompressor  # noqa: E402
from src.data.label_construction import (  # noqa: E402
    CacheState, LabelConfig, SupportingSpan, label_example,
)
from src.data.musique import normalize_musique_example  # noqa: E402


def select_pool_ids(model_name: str, n_train: int, n_val: int, max_tokens: int):
    """Re-derive the same 1000+100 ids as v1, deterministically.

    Uses transformers (any modern version) to instantiate the tokenizer; the
    selection logic only depends on the tokenizer producing identical token IDs
    for the context strings, which it does (Qwen2.5-7B's tokenizer is
    consistent across transformers versions).
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    train_pool, val_pool, scanned = [], [], 0
    for row in ds:
        scanned += 1
        if not row.get("answerable", True):
            continue
        ex = normalize_musique_example(row, tokenizer, span_granularity="tight")
        if not ex.supporting_spans or ex.n_paragraph_fallbacks > 0:
            continue
        if any(s.end > max_tokens for s in ex.supporting_spans):
            continue
        if ex.context_ids.size(0) > max_tokens:
            continue
        if len(train_pool) < n_train:
            train_pool.append(ex)
        elif len(val_pool) < n_val:
            val_pool.append(ex)
        else:
            break
    return train_pool, val_pool, scanned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--press", required=True, choices=["expected_attention", "snapkv"])
    ap.add_argument("--include-question", action="store_true",
                    help="SnapKV-native mode (mixture build does NOT use this — left for parity)")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--ratios", type=str, default="0.05,0.10,0.20,0.30,0.50,0.80")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = str(ROOT / "artifacts" / f"_mixture_raw_{args.press}.jsonl")

    ratios = [float(x) for x in args.ratios.split(",")]
    print(f"[setup] press={args.press} model={args.model_name} ratios={ratios}", flush=True)
    print("[setup] selecting same 1000+100 ids as v1...", flush=True)
    train_pool, val_pool, scanned = select_pool_ids(args.model_name, args.n_train, args.n_val, args.max_tokens)
    print(f"[setup] selected {len(train_pool)} train + {len(val_pool)} val (scanned {scanned})",
          flush=True)

    print(f"[setup] loading KVPressCompressor (press={args.press})...", flush=True)
    wrapper = KVPressCompressor(model_name=args.model_name, press=args.press)
    cfg = LabelConfig()

    raw: list[dict] = []
    t0 = time.time()
    for pool_name, pool in [("train", train_pool), ("val", val_pool)]:
        for i, ex in enumerate(pool, 1):
            ctx_ids = ex.context_ids
            if args.include_question:
                suffix = wrapper.tokenize(f"\n\nQuestion: {ex.question}\n\nAnswer:")
                prefill_ids = torch.cat([ctx_ids, suffix], dim=0)
            else:
                prefill_ids = ctx_ids
            for ratio in ratios:
                mask = wrapper.get_kept_token_mask(
                    ctx_ids, ratio,
                    prefill_ids=prefill_ids if args.include_question else None,
                )
                cs = CacheState(kept_mask=mask, compressor_name=args.press, ratio=ratio)
                label = label_example(
                    context=ex.context, query=ex.question, answer=ex.answer,
                    supporting_spans=ex.supporting_spans, cache_state=cs, cfg=cfg,
                )
                if label.type.value == "drop":
                    continue
                packaged = {
                    "id": ex.id, "ratio": ratio,
                    "label_type": label.type.value,
                    "target_output": label.target_output,
                    "kept_mask": mask.tolist(),
                    "context": ex.context, "query": ex.question,
                    "metadata": {
                        "compressor": args.press,
                        "model_name": args.model_name,
                        "ctx_len": int(ctx_ids.size(0)),
                        "n_kept": int(mask.sum().item()),
                        "answer": ex.answer,
                        "answer_aliases": ex.answer_aliases,
                        "supporting_spans": [
                            {"start": s.start, "end": s.end, "span_id": s.span_id, "source": s.source}
                            for s in ex.supporting_spans
                        ],
                        "span_retentions": [round(sr.fraction_kept, 3) for sr in label.span_retentions],
                        "rationale": label.rationale,
                        "split": f"mixture_{pool_name}",
                    },
                    "_pool": pool_name,
                }
                raw.append(packaged)
            if i % 50 == 0:
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
