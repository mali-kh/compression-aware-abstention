"""End-to-end dry run on 5 Musique examples.

Pipeline:
  Musique row
   -> normalize_musique_example (render context + token-level supporting spans)
   -> KVzipCompressor.get_kept_token_mask (ratio=0.20)
   -> label_construction.label_example -> Label
   -> emit a JSONL row with: id, context, query, kept_mask, target_output,
      label_type, metadata.

Then print all 5 records (compactly) to stdout for inspection.

Run:
  .venv/bin/python -B scripts/dry_run_musique.py [--ratio 0.2] [--n 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compressors.kvzip import KVzipCompressor  # noqa: E402
from src.data.label_construction import (  # noqa: E402
    CacheState,
    LabelConfig,
    label_example,
)
from src.data.musique import MusiqueExample, normalize_musique_example  # noqa: E402

OUT_PATH = ROOT / "artifacts" / "dry_run_musique.jsonl"


def _truncate_context_to_fit(
    ex: MusiqueExample, max_tokens: int
) -> MusiqueExample:
    """Drop trailing tokens beyond max_tokens and any spans that fall out.

    Musique contexts tokenize to ~1.5K tokens typically; this is a guard
    rather than the common case. If a supporting span gets truncated mid-
    span we drop the whole example by raising — better to skip than to
    score a half-span.
    """
    n = ex.context_ids.size(0)
    if n <= max_tokens:
        return ex
    keep = ex.context_ids[:max_tokens]
    surviving = []
    for s in ex.supporting_spans:
        if s.end <= max_tokens:
            surviving.append(s)
        else:
            raise ValueError(
                f"{ex.id}: supporting span {s.span_id} ends at token {s.end} "
                f"but context truncated to {max_tokens}; skipping example"
            )
    ex.context_ids = keep
    ex.supporting_spans = surviving
    return ex


def _select_examples(ds, n: int, max_tokens: int, tokenizer) -> list[MusiqueExample]:
    """Pick the first `n` answerable, in-budget examples."""
    out: list[MusiqueExample] = []
    for row in ds:
        if not row.get("answerable", True):
            continue
        ex = normalize_musique_example(row, tokenizer)
        if not ex.supporting_spans:
            continue
        try:
            ex = _truncate_context_to_fit(ex, max_tokens)
        except ValueError as e:
            print(f"  skip: {e}", file=sys.stderr)
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def _render_kept_mask_compact(mask: torch.Tensor, max_repr: int = 80) -> str:
    """Compact preview: 'TFFTTT...' for the first max_repr positions."""
    n = mask.size(0)
    head = "".join("T" if v else "F" for v in mask[:max_repr].tolist())
    suffix = "" if n <= max_repr else f"...({n - max_repr} more)"
    return head + suffix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="hard cap on context length (KVzip + A10 budget)")
    ap.add_argument("--model-name", default="qwen3-0.6b")
    args = ap.parse_args()

    print(f"Loading KVzip wrapper ({args.model_name})...")
    wrapper = KVzipCompressor(model_name=args.model_name)

    print("Loading Musique validation split...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    print(f"Selecting first {args.n} answerable examples (max {args.max_tokens} tokens)...")
    examples = _select_examples(ds, args.n, args.max_tokens, wrapper.tokenizer)
    print(f"Selected {len(examples)}.")

    cfg = LabelConfig()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    label_counts: Counter[str] = Counter()

    with OUT_PATH.open("w") as f:
        for ex in examples:
            mask = wrapper.get_kept_token_mask(ex.context_ids, ratio=args.ratio)
            assert mask.shape == (ex.context_ids.size(0),), (
                f"{ex.id}: mask len {mask.size(0)} != ctx len {ex.context_ids.size(0)}"
            )
            cs = CacheState(
                kept_mask=mask,
                compressor_name=wrapper.name,
                ratio=args.ratio,
            )
            label = label_example(
                context=ex.context,
                query=ex.question,
                answer=ex.answer,
                supporting_spans=ex.supporting_spans,
                cache_state=cs,
                cfg=cfg,
            )
            label_counts[label.type.value] += 1

            row = {
                "id": ex.id,
                "context": ex.context,
                "query": ex.question,
                "kept_mask": mask.tolist(),
                "target_output": label.target_output,
                "label_type": label.type.value,
                "metadata": {
                    "compressor": wrapper.name,
                    "ratio": args.ratio,
                    "model_name": args.model_name,
                    "ctx_len": int(ex.context_ids.size(0)),
                    "answer": ex.answer,
                    "answer_aliases": ex.answer_aliases,
                    "n_paragraphs": ex.n_paragraphs,
                    "supporting_paragraph_titles": ex.supporting_paragraph_titles,
                    "supporting_spans": [
                        {"start": s.start, "end": s.end, "span_id": s.span_id}
                        for s in ex.supporting_spans
                    ],
                    "span_retentions": [
                        round(sr.fraction_kept, 3) for sr in label.span_retentions
                    ],
                    "rationale": label.rationale,
                },
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} rows to {OUT_PATH.relative_to(ROOT)}")
    print(f"Label distribution: {dict(label_counts)}")
    print("\n" + "=" * 78)
    print(f"DRY-RUN INSPECTION  (compressor={wrapper.name}, ratio={args.ratio})")
    print("=" * 78)

    for i, row in enumerate(rows, 1):
        meta = row["metadata"]
        ctx = row["context"]
        ctx_preview = ctx if len(ctx) <= 280 else ctx[:280].rstrip() + " …"
        print(f"\n--- example {i}/{len(rows)}  id={row['id']} -----------------------------")
        print(f"  ctx_len      : {meta['ctx_len']} tokens")
        print(f"  query        : {row['query']}")
        print(f"  gold answer  : {meta['answer']!r}")
        print(f"  aliases      : {meta['answer_aliases']}")
        print(f"  n_paragraphs : {meta['n_paragraphs']}  (supporting: {len(meta['supporting_spans'])})")
        for span_info, supp_title, frac in zip(
            meta["supporting_spans"],
            meta["supporting_paragraph_titles"],
            meta["span_retentions"],
        ):
            tag = (
                "CONFIDENT-zone" if frac >= cfg.tau_high
                else "ABSTAIN-zone" if frac < cfg.tau_low
                else "uncertain-zone"
            )
            print(
                f"    [{span_info['start']:5d}, {span_info['end']:5d})  "
                f"\"{supp_title}\"  retention={frac:.3f}  ({tag})"
            )
        print(f"  → label      : {meta['rationale']}")
        print(f"  → label_type : {row['label_type']}")
        print(f"  → target_out : {row['target_output']!r}")
        print(f"  context (first 280 chars):")
        print(f"      {ctx_preview}")
        print(f"  kept_mask preview: {_render_kept_mask_compact(torch.tensor(row['kept_mask']))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
