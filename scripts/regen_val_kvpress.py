"""Regenerate val_v1 under a kvpress press (Expected Attention or SnapKV).

Runs in `.venv-kvpress/`. Produces `artifacts/val_v1_<press>.jsonl` with the
same schema as `val_v1.jsonl` but with `kept_mask`, `n_kept`, `span_retentions`,
`label_type`, `target_output`, and `rationale` re-derived under the new
compressor at the same retention ratios.

Design notes:
  * Tokenizes context with the production base tokenizer (Qwen2.5-7B-Instruct).
    The original supporting_spans in metadata are token indices into THIS
    tokenization, so they remain valid 1:1.
  * Prefill is run on context-only (no question concatenated). For SnapKV
    this neutralises its query-aware advantage by construction — see the
    cross-compressor memo addendum for the reasoning. To run SnapKV in
    its query-aware mode instead, prefill on context+question (one-line
    change in `_compose_prefill_text`).
  * Three sanity outputs printed at the end:
      (a) realized vs target retention rate per ratio (calibration check),
      (b) label flip rate vs the original KVzip-labelled val_v1 (informativeness),
      (c) a single decoded fragment for one example (alignment eyeball).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compressors.kvpress_wrapper import KVPressCompressor  # noqa: E402
from src.data.label_construction import (  # noqa: E402
    CacheState,
    LabelConfig,
    SupportingSpan,
    label_example,
)


def _question_suffix_text(row: dict) -> str:
    """The text appended after the context in SnapKV-native prefill mode.

    Matches the eval prompt's MIDDLE format so the question is presented to
    the press identically to how it appears at inference time.
    """
    return f"\n\nQuestion: {row['query']}\n\nAnswer:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", default=str(ROOT / "artifacts" / "val_v1.jsonl"))
    ap.add_argument("--press", required=True, choices=["expected_attention", "snapkv"])
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap unique example ids (sanity-test runs)")
    ap.add_argument("--include-question", action="store_true",
                    help="feed context+question to the press (SnapKV-native mode)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val_jsonl) if l.strip()]
    print(f"[setup] loaded {len(rows)} val rows from {args.val_jsonl}", flush=True)

    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    ids = list(by_id.keys())
    if args.limit:
        ids = ids[: args.limit]
    print(f"[setup] {len(ids)} unique example ids; press={args.press}", flush=True)

    print(f"[setup] loading {args.model_name} + kvpress.{args.press}...", flush=True)
    wrapper = KVPressCompressor(model_name=args.model_name, press=args.press)
    cfg = LabelConfig()  # defaults match v1 (tau_high=0.8, tau_low=0.3, drop=True, FP=False)

    out_rows: list[dict] = []
    sanity_calib: dict[float, list[float]] = defaultdict(list)
    label_flips: list[tuple[str, float, str, str]] = []
    skipped: list[tuple[str, str]] = []
    t0 = time.time()

    for i, ex_id in enumerate(ids, 1):
        rows_for_id = by_id[ex_id]
        ctx_ids = wrapper.tokenize(rows_for_id[0]["context"])
        if args.include_question:
            # Tokenize the suffix separately and concat IDs. Tokenizing the
            # full string would risk BPE merges across the context/suffix
            # boundary that change the leading token IDs.
            suffix_ids = wrapper.tokenize(_question_suffix_text(rows_for_id[0]))
            prefill_ids = torch.cat([ctx_ids, suffix_ids], dim=0)
        else:
            prefill_ids = ctx_ids
        if prefill_ids.shape[0] > args.max_tokens:
            skipped.append((ex_id, f"prefill_len={prefill_ids.shape[0]} > {args.max_tokens}"))
            continue
        # Sanity: context tokenization MUST match the original kept_mask length.
        original_ctx_len = len(rows_for_id[0]["kept_mask"])
        if ctx_ids.shape[0] != original_ctx_len:
            skipped.append(
                (ex_id, f"tok_len {ctx_ids.shape[0]} != original {original_ctx_len}")
            )
            continue

        first_meta = rows_for_id[0]["metadata"]
        spans = [
            SupportingSpan(
                start=s["start"], end=s["end"],
                span_id=s.get("span_id"), source=s.get("source", "annotation"),
            )
            for s in first_meta["supporting_spans"]
        ]

        for r in rows_for_id:
            ratio = r["ratio"]
            mask = wrapper.get_kept_token_mask(
                ctx_ids, ratio,
                prefill_ids=prefill_ids if args.include_question else None,
            )
            sanity_calib[ratio].append(float(mask.float().mean().item()))

            cs = CacheState(kept_mask=mask, compressor_name=args.press, ratio=ratio)
            label = label_example(
                context=r["context"], query=r["query"],
                answer=r["metadata"]["answer"],
                supporting_spans=spans, cache_state=cs, cfg=cfg,
            )

            old_label = r["label_type"]
            new_label = label.type.value
            if old_label != new_label:
                label_flips.append((ex_id, ratio, old_label, new_label))

            new_metadata = dict(r["metadata"])
            new_metadata["compressor"] = args.press
            new_metadata["model_name"] = args.model_name
            new_metadata["n_kept"] = int(mask.sum().item())
            new_metadata["span_retentions"] = [
                {"span_id": sr.span.span_id, "fraction_kept": sr.fraction_kept}
                for sr in label.span_retentions
            ]
            new_metadata["rationale"] = label.rationale
            out_rows.append({
                "id": r["id"], "ratio": ratio,
                "label_type": new_label,
                "target_output": label.target_output,
                "kept_mask": mask.tolist(),
                "context": r["context"], "query": r["query"],
                "metadata": new_metadata,
            })

        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = (len(ids) - i) * elapsed / i
            print(f"  [{i}/{len(ids)}] elapsed={elapsed:.0f}s  ~{eta:.0f}s remaining",
                  flush=True)

    # write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] wrote {len(out_rows)} rows to {args.out}", flush=True)
    if skipped:
        print(f"[done] skipped {len(skipped)} ids:", flush=True)
        for ex_id, why in skipped[:10]:
            print(f"    {ex_id}: {why}", flush=True)

    # sanity (a): ratio calibration
    print("\n=== sanity (a): ratio calibration (target vs realized mask.mean()) ===")
    for ratio in sorted(sanity_calib.keys()):
        vals = sanity_calib[ratio]
        mean = sum(vals) / len(vals)
        print(f"  ratio={ratio:.2f}  realized={mean:.4f}  Δ={(mean - ratio) * 100:+.2f}pp  (n={len(vals)})")

    # sanity (b): label flips vs original KVzip-labelled val_v1
    print(f"\n=== sanity (b): label flips vs original KVzip val_v1 ===")
    print(f"  total flips: {len(label_flips)} / {len(out_rows)} = "
          f"{len(label_flips) / max(len(out_rows), 1):.1%}")
    flip_counter = Counter((old, new) for _, _, old, new in label_flips)
    for (old, new), cnt in flip_counter.most_common():
        print(f"    {old:>20} -> {new:<20}  ×{cnt}")
    flips_by_ratio: dict[float, int] = defaultdict(int)
    rows_by_ratio: dict[float, int] = defaultdict(int)
    for r in out_rows:
        rows_by_ratio[r["ratio"]] += 1
    for _, ratio, _, _ in label_flips:
        flips_by_ratio[ratio] += 1
    print("  flips per ratio:")
    for ratio in sorted(rows_by_ratio.keys()):
        n = rows_by_ratio[ratio]
        f = flips_by_ratio[ratio]
        print(f"    ratio={ratio:.2f}  flips={f}/{n} = {f/max(n,1):.1%}")

    # sanity (c): decoded fragment for one example at ratio=0.30
    print(f"\n=== sanity (c): decoded kept fragment (alignment eyeball) ===")
    sample = next((r for r in out_rows if abs(r["ratio"] - 0.30) < 1e-6), None)
    if sample is not None:
        ctx_ids_full = wrapper.tokenize(sample["context"])
        kept_ids = ctx_ids_full[torch.tensor(sample["kept_mask"], dtype=torch.bool)]
        decoded = wrapper.tokenizer.decode(kept_ids.tolist(), skip_special_tokens=True)
        print(f"  id={sample['id']}  ratio={sample['ratio']}  n_kept={sample['metadata']['n_kept']}/{len(sample['kept_mask'])}")
        print(f"  decoded fragment (first 400 chars): {decoded[:400]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
