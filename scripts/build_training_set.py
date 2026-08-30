"""D7 — Scaled labeled training-set construction.

Pipeline:
  1. Iterate Musique train split, take the first N_train + N_val examples
     that fit the token budget and have all sub-answers verbatim-locatable
     (tight spans, no fallback).
  2. Disjoint by example id: first N_train go into the train pool, the
     next N_val into the val pool. Same labelling treatment for both.
  3. For each example: prefill+score once on the configured KVzip model,
     prune at each ratio in {0.05, 0.10, 0.20, 0.30, 0.50, 0.80},
     reduce kept_mask via topk to match the labeler's calibration, run
     label_example with the binary labeler (FLAGGED_PARTIAL off).
  4. Drop instances labelled DROP. Keep CONFIDENT and ABSTAIN.
  5. Class-balance the TRAIN set only by undersampling the majority
     class (the user's "ratio reweighting" — we keep all minority and
     drop a random subset of majority; no oversampling/duplication).
     The VAL set is kept at natural distribution so eval reflects reality.
  6. Emit two JSONL files. Report counts, ratio distribution, and the
     critical ctx-length-parity check (CONFIDENT vs ABSTAIN means must be
     within 10% — otherwise context length becomes a shortcut feature).

Output schema (one JSON object per line):
  id, ratio, label_type, target_output, kept_mask (list[bool]),
  context, query, metadata{
      compressor, model_name, ctx_len, n_kept, answer, answer_aliases,
      supporting_spans, span_retentions, rationale, split,
  }
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

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

OUT_TRAIN = ROOT / "artifacts" / "train_v1.jsonl"
OUT_VAL = ROOT / "artifacts" / "val_v1.jsonl"
OUT_REPORT = ROOT / "artifacts" / "train_v1_report.md"


def topk_mask(frac_kept: torch.Tensor, ratio: float) -> torch.Tensor:
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


def select_examples(ds, *, n_train: int, n_val: int, max_tokens: int, tokenizer):
    """Iterate ds, return (train_pool, val_pool) disjoint by example id.

    Only keep examples where all sub-answers were located verbatim (tight)
    so the dataset is uniformly tight-labeled — no mixed-fallback pollution.
    """
    train_pool, val_pool = [], []
    scanned = 0
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


def label_one_example_at_ratios(
    ex,
    *,
    wrapper: KVzipCompressor,
    ratios: list[float],
    cfg: LabelConfig,
) -> list[dict]:
    """Prefill+score once, prune at each ratio, classify, return rows."""
    rows: list[dict] = []
    ids_dev = ex.context_ids.to(wrapper._impl.device).unsqueeze(0)
    kv = wrapper._impl.prefill(ids_dev, load_score=False)
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
        rows.append(
            {
                "id": ex.id,
                "ratio": ratio,
                "label_type": label.type.value,
                "target_output": label.target_output,
                "kept_mask": mask.tolist(),
                "context": ex.context,
                "query": ex.question,
                "label_obj": label,
                "ex_obj": ex,
                "n_kept": int(mask.sum().item()),
            }
        )
    return rows


def package_row(r: dict, *, model_name: str, split: str) -> dict:
    """Strip transient objects, keep only what's serialized to JSONL."""
    label = r["label_obj"]
    ex = r["ex_obj"]
    return {
        "id": r["id"],
        "ratio": r["ratio"],
        "label_type": r["label_type"],
        "target_output": r["target_output"],
        "kept_mask": r["kept_mask"],
        "context": r["context"],
        "query": r["query"],
        "metadata": {
            "compressor": "kvzip",
            "model_name": model_name,
            "ctx_len": int(ex.context_ids.size(0)),
            "n_kept": r["n_kept"],
            "answer": ex.answer,
            "answer_aliases": ex.answer_aliases,
            "supporting_spans": [
                {"start": s.start, "end": s.end, "span_id": s.span_id, "source": s.source}
                for s in ex.supporting_spans
            ],
            "span_retentions": [round(sr.fraction_kept, 3) for sr in label.span_retentions],
            "rationale": label.rationale,
            "split": split,
        },
    }


def class_balance_undersample(
    rows: list[dict],
    *,
    target_minority_frac: float,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Drop random majority-class instances until minority frac >= target.

    No oversampling. We keep every minority example, pick a random subset
    of the majority. Documented dropouts go into the report.
    """
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label_type"]].append(r)

    counts = {k: len(v) for k, v in by_class.items()}
    if not counts or len(counts) < 2:
        return rows, {"action": "no_balance_needed", "counts_before": counts}

    minority = min(counts, key=counts.get)
    majority = max(counts, key=counts.get)
    n_min = counts[minority]
    n_maj = counts[majority]
    cur_min_frac = n_min / (n_min + n_maj)

    if cur_min_frac >= target_minority_frac:
        return rows, {
            "action": "natural_balance_already_meets_target",
            "counts_before": counts,
            "counts_after": counts,
            "minority": minority,
            "minority_frac_before": cur_min_frac,
            "minority_frac_after": cur_min_frac,
        }

    # we want n_min / (n_min + n_maj') = target_minority_frac
    # => n_maj' = n_min * (1 - target) / target
    target_maj = int(n_min * (1 - target_minority_frac) / target_minority_frac)
    target_maj = min(target_maj, n_maj)
    sampled_maj = rng.sample(by_class[majority], target_maj)
    final = by_class[minority] + sampled_maj
    rng.shuffle(final)
    final_counts = Counter(r["label_type"] for r in final)
    return final, {
        "action": "undersampled_majority",
        "counts_before": counts,
        "counts_after": dict(final_counts),
        "minority": minority,
        "majority": majority,
        "minority_frac_before": cur_min_frac,
        "minority_frac_after": final_counts[minority] / sum(final_counts.values()),
        "majority_dropped": n_maj - target_maj,
    }


def class_ctx_len_parity(rows: list[dict]) -> dict:
    """Return mean/median ctx_len per class and the |diff|/min ratio."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label_type"]].append(r["ex_obj"].context_ids.size(0))
    out = {
        cls: {
            "mean": statistics.mean(lens),
            "median": statistics.median(lens),
            "n": len(lens),
        }
        for cls, lens in by_class.items()
    }
    if "confident_answer" in out and "abstain" in out:
        m_conf = out["confident_answer"]["mean"]
        m_abs = out["abstain"]["mean"]
        diff_pct = abs(m_conf - m_abs) / min(m_conf, m_abs)
        out["mean_diff_pct"] = diff_pct
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--ratios", type=str, default="0.05,0.10,0.20,0.30,0.50,0.80")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--target-minority-frac",
        type=float,
        default=0.30,
        help="balance target for the minority class in train (>=0.30 per spec)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default=None,
                    help="Override output train jsonl path. Default: artifacts/train_v1.jsonl")
    ap.add_argument("--out-val", default=None,
                    help="Override output val jsonl path. Default: artifacts/val_v1.jsonl")
    ap.add_argument("--out-report", default=None,
                    help="Override output report md path. Default: artifacts/train_v1_report.md")
    args = ap.parse_args()
    out_train = Path(args.out_train) if args.out_train else OUT_TRAIN
    out_val = Path(args.out_val) if args.out_val else OUT_VAL
    out_report = Path(args.out_report) if args.out_report else OUT_REPORT

    if args.target_minority_frac < 0.30:
        ap.error("--target-minority-frac must be >= 0.30 per D7 spec")

    ratios = [float(r) for r in args.ratios.split(",")]
    rng = random.Random(args.seed)
    out_train.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] model_name={args.model_name} ratios={ratios}", flush=True)
    print("[setup] loading KVzip wrapper...", flush=True)
    wrapper = KVzipCompressor(model_name=args.model_name)

    print("[setup] loading Musique train split...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="train")

    print(
        f"[setup] selecting {args.n_train} train + {args.n_val} val examples...",
        flush=True,
    )
    train_pool, val_pool, scanned = select_examples(
        ds,
        n_train=args.n_train,
        n_val=args.n_val,
        max_tokens=args.max_tokens,
        tokenizer=wrapper.tokenizer,
    )
    print(
        f"  train_pool={len(train_pool)} val_pool={len(val_pool)} "
        f"(scanned {scanned} rows)",
        flush=True,
    )

    cfg = LabelConfig()  # binary labeler, FP off
    print(f"[run] labelling train pool at {len(ratios)} ratios...", flush=True)
    t0 = time.time()
    train_rows: list[dict] = []
    for i, ex in enumerate(train_pool):
        train_rows.extend(
            label_one_example_at_ratios(ex, wrapper=wrapper, ratios=ratios, cfg=cfg)
        )
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1:4d}/{len(train_pool)}]  {elapsed:.1f}s  ({rate:.1f} ex/s; "
                f"~{(len(train_pool)-i-1)/rate:.0f}s remaining)",
                flush=True,
            )

    print(f"\n[run] labelling val pool ({len(val_pool)} examples)...", flush=True)
    val_rows: list[dict] = []
    for ex in val_pool:
        val_rows.extend(
            label_one_example_at_ratios(ex, wrapper=wrapper, ratios=ratios, cfg=cfg)
        )

    elapsed = time.time() - t0
    print(f"[run] done labelling in {elapsed:.1f}s "
          f"({len(train_rows)+len(val_rows)} raw instances)", flush=True)

    # Filter DROP
    train_kept = [r for r in train_rows if r["label_type"] != LabelType.DROP.value]
    val_kept = [r for r in val_rows if r["label_type"] != LabelType.DROP.value]
    train_dropped = len(train_rows) - len(train_kept)
    val_dropped = len(val_rows) - len(val_kept)

    train_natural_counts = Counter(r["label_type"] for r in train_kept)
    val_counts = Counter(r["label_type"] for r in val_kept)
    print(
        f"\n[filter] train: kept {len(train_kept)} dropped {train_dropped} "
        f"(natural balance: {dict(train_natural_counts)})",
        flush=True,
    )
    print(
        f"[filter] val: kept {len(val_kept)} dropped {val_dropped} "
        f"(natural balance: {dict(val_counts)})",
        flush=True,
    )

    train_balanced, balance_info = class_balance_undersample(
        train_kept,
        target_minority_frac=args.target_minority_frac,
        rng=rng,
    )
    print(f"[balance] {balance_info}", flush=True)

    # Write JSONL
    with out_train.open("w") as f:
        for r in train_balanced:
            f.write(json.dumps(package_row(r, model_name=args.model_name, split="train_v1")) + "\n")
    with out_val.open("w") as f:
        for r in val_kept:
            f.write(json.dumps(package_row(r, model_name=args.model_name, split="val_v1")) + "\n")

    train_parity = class_ctx_len_parity(train_balanced)
    val_parity = class_ctx_len_parity(val_kept)
    train_per_ratio = Counter(r["ratio"] for r in train_balanced)
    val_per_ratio = Counter(r["ratio"] for r in val_kept)

    # Markdown report
    md: list[str] = []
    md.append("# D7 — Training-set construction report\n")
    md.append("_Auto-generated by `scripts/build_training_set.py`._\n")
    md.append("## Setup\n")
    md.append(f"- KVzip target / training base: `{args.model_name}`")
    md.append(f"- ratios swept: {ratios}")
    md.append(f"- max ctx tokens: {args.max_tokens}")
    md.append(f"- labeler: binary (FLAGGED_PARTIAL off), tau_low=0.3, tau_high=0.8")
    md.append(f"- spans: tight (sub-answer via question_decomposition; no fallbacks)")
    md.append(f"- target minority class fraction: {args.target_minority_frac}")
    md.append(f"- random seed: {args.seed}")
    md.append("")
    md.append("## Pool sizes\n")
    md.append(f"- train pool examples: {len(train_pool)}  (scanned {scanned} rows)")
    md.append(f"- val pool examples:   {len(val_pool)}  (disjoint by example id)")
    md.append(f"- raw instances: train {len(train_rows)}  val {len(val_rows)}")
    md.append(f"- DROP filtered: train {train_dropped}  val {val_dropped}")
    md.append("")
    md.append("## Class balance — train\n")
    md.append(f"- before balancing: {dict(train_natural_counts)}")
    md.append(f"- after balancing:  {balance_info.get('counts_after', dict(train_natural_counts))}")
    md.append(f"- action: `{balance_info['action']}`")
    md.append(f"- final size: **{len(train_balanced)}** instances")
    md.append("")
    md.append("## Class balance — val (natural distribution, no balancing)\n")
    md.append(f"- counts: {dict(val_counts)}")
    md.append(f"- final size: **{len(val_kept)}** instances")
    md.append("")
    md.append("## Per-ratio counts\n")
    md.append("| ratio | train | val |")
    md.append("| --- | --- | --- |")
    for r in ratios:
        md.append(f"| {r:.2f} | {train_per_ratio.get(r, 0)} | {val_per_ratio.get(r, 0)} |")
    md.append("")

    md.append("## Critical: context-length parity (train)\n")
    md.append("If CONFIDENT and ABSTAIN means differ by more than 10%, context")
    md.append("length is a shortcut feature and the model can game the task.\n")
    md.append("| class | n | mean ctx_len | median ctx_len |")
    md.append("| --- | --- | --- | --- |")
    for cls in ("confident_answer", "abstain"):
        if cls in train_parity:
            stats = train_parity[cls]
            md.append(f"| `{cls}` | {stats['n']} | {stats['mean']:.1f} | {stats['median']:.1f} |")
    if "mean_diff_pct" in train_parity:
        diff = train_parity["mean_diff_pct"]
        verdict = "✅ within 10%" if diff < 0.10 else "❌ EXCEEDS 10% — confound"
        md.append(f"\n**|mean(CONF) - mean(ABST)| / min = {diff:.1%}**  {verdict}")
    md.append("")

    out_report.write_text("\n".join(md))

    print(f"\n[done] wrote:")
    def _disp(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT))
        except ValueError:
            return str(p)
    print(f"  {_disp(out_train)}  ({len(train_balanced)} instances)")
    print(f"  {_disp(out_val)}    ({len(val_kept)} instances)")
    print(f"  {_disp(out_report)}")
    if "mean_diff_pct" in train_parity:
        diff = train_parity["mean_diff_pct"]
        verdict = "OK" if diff < 0.10 else "BLOCKER"
        print(f"\nctx_len parity (train): |CONF-ABST|/min = {diff:.1%}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
