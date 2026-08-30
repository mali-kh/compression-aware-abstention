"""Ratio-sweep diagnostic on Musique.

For each of N answerable Musique examples and each retention ratio R, we:

  1. prefill + score the context with KVzip (once per example — amortized
     across ratios so we don't pay the prefill cost six times)
  2. prune at ratio R and extract the per-token kept_mask under the
     wrapper's "topk" reduction
  3. compute per-supporting-span retention
  4. classify with label_construction.label_example
  5. record (id, ratio, label, per-span retention, min/max/mean, etc.)

Outputs:
  artifacts/ratio_sweep.jsonl       — one row per (example, ratio)
  artifacts/ratio_sweep_report.md   — summary tables, histograms, framing

The headline question this answers: at intermediate ratios, is the
distribution of per-example labels {CONFIDENT, ABSTAIN, DROP} multi-class
(=> cache-content-conditional signal exists) or single-class (=>
cache-size-conditional only)?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
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
from src.data.musique import normalize_musique_example  # noqa: E402



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


def select_examples(ds, n: int, max_tokens: int, tokenizer, *, span_granularity: str):
    out = []
    for row in ds:
        if not row.get("answerable", True):
            continue
        ex = normalize_musique_example(row, tokenizer, span_granularity=span_granularity)
        if not ex.supporting_spans:
            continue
        if any(s.end > max_tokens for s in ex.supporting_spans):
            continue
        if ex.context_ids.size(0) > max_tokens:
            ex.context_ids = ex.context_ids[:max_tokens]
        out.append(ex)
        if len(out) >= n:
            break
    return out


def make_ascii_hist(
    values: list[float],
    *,
    bins: int = 10,
    lo: float = 0.0,
    hi: float = 1.0,
    width: int = 36,
) -> str:
    if not values:
        return "  (no data)"
    counts = [0] * bins
    for v in values:
        b = min(int((v - lo) / (hi - lo) * bins), bins - 1)
        b = max(b, 0)
        counts[b] += 1
    max_c = max(counts) or 1
    lines = []
    for i in range(bins):
        edge_lo = lo + (hi - lo) * i / bins
        edge_hi = lo + (hi - lo) * (i + 1) / bins
        bar = "#" * int(round(width * counts[i] / max_c))
        lines.append(f"  [{edge_lo:.2f}, {edge_hi:.2f})  {bar:<{width}}  {counts[i]:3d}")
    return "\n".join(lines)


def class_entropy(counts: list[int]) -> float:
    """Shannon entropy of label distribution, in nats. 0 if pure, log(K) if uniform."""
    n = sum(counts)
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--ratios", type=str, default="0.05,0.10,0.20,0.30,0.50,0.80")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model-name", default="qwen3-0.6b")
    ap.add_argument(
        "--spans",
        choices=["paragraph", "tight"],
        default="paragraph",
        help="span granularity (paragraph: whole supporting paragraph; "
        "tight: per-hop sub-answer string with paragraph fallback)",
    )
    ap.add_argument(
        "--flagged-partial",
        action="store_true",
        help="enable Option A (FLAGGED_PARTIAL) instead of v1 collapse to ABSTAIN",
    )
    ap.add_argument(
        "--out-tag",
        default="",
        help="suffix appended to output filenames so multiple sweeps can coexist",
    )
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_jsonl = ROOT / "artifacts" / f"ratio_sweep{tag}.jsonl"
    out_md = ROOT / "artifacts" / f"ratio_sweep_report{tag}.md"

    print(
        f"[setup] model={args.model_name} ratios={ratios} n={args.n} "
        f"spans={args.spans} flagged_partial={args.flagged_partial} tag={tag!r}",
        flush=True,
    )

    print("[setup] loading wrapper...", flush=True)
    wrapper = KVzipCompressor(model_name=args.model_name)

    print("[setup] loading Musique validation split...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    examples = select_examples(
        ds, args.n, args.max_tokens, wrapper.tokenizer,
        span_granularity=args.spans,
    )
    print(f"[setup] selected {len(examples)} examples", flush=True)

    if args.spans == "tight":
        n_tight = sum(ex.n_tight_spans_located for ex in examples)
        n_fb = sum(ex.n_paragraph_fallbacks for ex in examples)
        print(f"[setup] tight spans: {n_tight} located, {n_fb} fell back to paragraph", flush=True)

    cfg = LabelConfig(enable_flagged_partial=args.flagged_partial)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    with out_jsonl.open("w") as f:
        for i, ex in enumerate(examples):
            t_ex = time.time()
            ctx_ids_dev = ex.context_ids.to(wrapper._impl.device).unsqueeze(0)
            kv = wrapper._impl.prefill(ctx_ids_dev, load_score=False)
            for ratio in ratios:
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
                span_rets = [round(sr.fraction_kept, 4) for sr in label.span_retentions]
                row = {
                    "id": ex.id,
                    "ratio": ratio,
                    "ctx_len": int(ex.context_ids.size(0)),
                    "n_supporting_spans": len(ex.supporting_spans),
                    "span_retentions": span_rets,
                    "min_span_retention": min(span_rets),
                    "max_span_retention": max(span_rets),
                    "mean_span_retention": sum(span_rets) / len(span_rets),
                    "label_type": label.type.value,
                    "answer": ex.answer,
                    "question": ex.question,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            print(
                f"  [{i+1:3d}/{len(examples)}] {ex.id} "
                f"ctx={ex.context_ids.size(0):4d}tok "
                f"spans={len(ex.supporting_spans)} "
                f"({time.time()-t_ex:.2f}s)",
                flush=True,
            )

    print(f"\n[done] {len(rows)} measurements in {time.time()-t_start:.1f}s", flush=True)
    print(f"[done] wrote {out_jsonl.relative_to(ROOT)}")

    # ============================ Markdown report ============================
    by_ratio = {r: [row for row in rows if row["ratio"] == r] for r in ratios}
    label_types = ["confident_answer", "abstain", "drop", "flagged_partial"]

    md: list[str] = []
    md.append("# Ratio-sweep diagnostic — Musique\n")
    md.append("_Auto-generated by `scripts/analyze_ratio_sweep.py`._\n")
    md.append("## Configuration\n")
    md.append(f"- compressor: **KVzip** (reduction=`topk`, model=`{args.model_name}`)")
    md.append(f"- examples: {len(examples)} answerable Musique validation rows, ctx_len <= {args.max_tokens}")
    md.append(f"- ratios swept: {ratios}")
    md.append(
        f"- labeler defaults: tau_high={cfg.tau_high}, tau_low={cfg.tau_low}, "
        f"drop_uncertain_zone={cfg.drop_uncertain_zone}, "
        f"enable_flagged_partial={cfg.enable_flagged_partial}"
    )
    md.append("")

    md.append("## Headline: label distribution per ratio\n")
    md.append("Each row sums to N (number of examples). Higher entropy = more class")
    md.append("diversity at that ratio = stronger cache-content-conditional signal")
    md.append("(max entropy with 3 active classes = 1.099 nats).\n")
    md.append("| ratio | confident | abstain | drop | flagged_partial | entropy (nats) |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for r in ratios:
        c = Counter(row["label_type"] for row in by_ratio[r])
        counts = [c.get(lt, 0) for lt in label_types]
        h = class_entropy(counts)
        md.append(f"| {r:.2f} | " + " | ".join(str(x) for x in counts) + f" | {h:.3f} |")
    md.append("")

    md.append("## Distribution of min-span retention per ratio\n")
    md.append("`min_span_retention` is the bottleneck the labeler sees: if any")
    md.append("required span falls below tau_low=0.30, the example becomes ABSTAIN.")
    md.append("If the distribution is *bimodal* (mass at low + high) at a fixed ratio,")
    md.append("different examples land in different label classes at the same ratio")
    md.append("=> cache-content-conditional signal exists.\n")
    md.append("```")
    for r in ratios:
        vals = [row["min_span_retention"] for row in by_ratio[r]]
        mu = statistics.mean(vals) if vals else 0
        sd = statistics.stdev(vals) if len(vals) >= 2 else 0
        md.append(f"\nratio = {r:.2f}    n={len(vals)}    mean={mu:.3f}    sd={sd:.3f}")
        md.append(make_ascii_hist(vals, bins=10, lo=0.0, hi=1.0))
    md.append("```")
    md.append("")

    md.append("## Distribution of mean-span retention per ratio\n")
    md.append(
        "Span-averaged (less conservative than min). Useful for examples with "
        "more than one supporting span.\n"
    )
    md.append("```")
    for r in ratios:
        vals = [row["mean_span_retention"] for row in by_ratio[r]]
        mu = statistics.mean(vals) if vals else 0
        sd = statistics.stdev(vals) if len(vals) >= 2 else 0
        md.append(f"\nratio = {r:.2f}    n={len(vals)}    mean={mu:.3f}    sd={sd:.3f}")
        md.append(make_ascii_hist(vals, bins=10, lo=0.0, hi=1.0))
    md.append("```")
    md.append("")

    # Class-mix sanity table — at each ratio, what fraction of examples end up in
    # CONFIDENT vs ABSTAIN vs DROP based on min_span_retention crossing thresholds?
    md.append("## Per-ratio quartile structure\n")
    md.append("Quartiles of min_span_retention. If Q1 and Q3 straddle a threshold")
    md.append("(tau_low=0.30 or tau_high=0.80), that ratio is on the decision boundary")
    md.append("for at least 25% of examples — exactly where training-set diversity comes from.\n")
    md.append("| ratio | Q1 | median | Q3 | range vs tau_low | range vs tau_high |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for r in ratios:
        vals = sorted(row["min_span_retention"] for row in by_ratio[r])
        if not vals:
            continue
        n = len(vals)
        q1 = vals[n // 4]
        med = vals[n // 2]
        q3 = vals[(3 * n) // 4]
        crosses_low = "✓ crosses" if q1 < cfg.tau_low <= q3 else "—"
        crosses_high = "✓ crosses" if q1 < cfg.tau_high <= q3 else "—"
        md.append(
            f"| {r:.2f} | {q1:.3f} | {med:.3f} | {q3:.3f} | {crosses_low} | {crosses_high} |"
        )
    md.append("")

    out_md.write_text("\n".join(md))
    print(f"[done] wrote {out_md.relative_to(ROOT)}", flush=True)

    print("\n=== quick label-distribution preview ===")
    for r in ratios:
        c = Counter(row["label_type"] for row in by_ratio[r])
        h = class_entropy([c.get(lt, 0) for lt in label_types])
        print(f"  ratio={r:.2f}  {dict(c)}  H={h:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
