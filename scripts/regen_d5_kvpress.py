"""Regenerate D5 wrong-but-specific fragments under a kvpress press.

Reads the 10 wrong-but-specific D5 picks (those in v1_d5_qualitative.md),
loads their Musique-val records, runs prefill + score + prune at ratio=0.30
under the chosen press, decodes the kept tokens to text. Outputs a small
JSONL keyed by id.

Runs in .venv-kvpress. Output is consumed by `scripts/d5_cross_compressor_md.py`
in the main venv (which loads the v1 LoRA adapter).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compressors.kvpress_wrapper import KVPressCompressor  # noqa: E402
from src.data.musique import normalize_musique_example  # noqa: E402


def load_d5_picks(qual_md: Path) -> list[str]:
    md = qual_md.read_text()
    ids = re.findall(r"`(2hop__\d+_\d+)`", md)
    return list(dict.fromkeys(ids))[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--press", required=True, choices=["expected_attention", "snapkv"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-question", action="store_true",
                    help="SnapKV-native mode (feed question to the press)")
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--qual-md", default=str(ROOT / "artifacts" / "v1_d5_qualitative.md"))
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    picks = load_d5_picks(Path(args.qual_md))
    print(f"[setup] {len(picks)} D5 picks: {picks[:3]}...", flush=True)

    print("[setup] loading Musique val + locating D5 picks...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    by_id: dict[str, dict] = {}
    for row in ds:
        if row["id"] in picks:
            by_id[row["id"]] = row
            if len(by_id) == len(picks):
                break
    print(f"[setup] located {len(by_id)}/{len(picks)} D5 picks in Musique val", flush=True)

    print(f"[setup] loading {args.model_name} + kvpress.{args.press}...", flush=True)
    wrapper = KVPressCompressor(model_name=args.model_name, press=args.press)

    out_rows: list[dict] = []
    for ex_id in picks:
        if ex_id not in by_id:
            print(f"  {ex_id}: not in Musique val, skipping", flush=True)
            continue
        row = by_id[ex_id]
        norm = normalize_musique_example(row, wrapper.tokenizer, span_granularity="tight")
        context = norm.context
        question = norm.question
        answer = norm.answer

        ctx_ids = wrapper.tokenize(context)
        if ctx_ids.shape[0] > args.max_tokens:
            print(f"  {ex_id}: ctx too long ({ctx_ids.shape[0]}), skipping", flush=True)
            continue
        if args.include_question:
            suffix_ids = wrapper.tokenize(f"\n\nQuestion: {question}\n\nAnswer:")
            prefill_ids = torch.cat([ctx_ids, suffix_ids], dim=0)
        else:
            prefill_ids = ctx_ids

        mask = wrapper.get_kept_token_mask(
            ctx_ids, args.ratio,
            prefill_ids=prefill_ids if args.include_question else None,
        )
        kept_ids = ctx_ids[mask]
        decoded_fragment = wrapper.tokenizer.decode(kept_ids.tolist(), skip_special_tokens=True)

        out_rows.append({
            "id": ex_id, "ratio": args.ratio,
            "press": args.press,
            "include_question": bool(args.include_question),
            "question": question, "answer": answer,
            "ctx_len": int(ctx_ids.shape[0]),
            "n_kept": int(mask.sum().item()),
            "decoded_fragment": decoded_fragment,
        })
        print(f"  {ex_id}: ctx_len={ctx_ids.shape[0]} n_kept={mask.sum().item()}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] wrote {len(out_rows)} fragments to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
