"""Block 5 (PILOT) — random-deletion training-set construction.

Drop-in counterpart to `scripts/build_training_set.py`: identical pipeline
except the per-example kept_mask is drawn UNIFORMLY AT RANDOM at the same
n_kept count that topk_mask would produce. Mask seed per example is
`numpy.random.default_rng(42 + example_index)` per the Block-5 spec.

No GPU needed (no KVzip prefill+score). Loads only the tokenizer.

Schema matches `train_v1.jsonl` byte-for-byte except metadata.compressor =
"uniform_random". Output paths default to artifacts/train_v1_random.jsonl
and artifacts/val_v1_random.jsonl; overridable via the same --out-train /
--out-val / --out-report flags as the v1 script.
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

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.label_construction import (  # noqa: E402
    CacheState,
    LabelConfig,
    LabelType,
    label_example,
)
from src.data.musique import normalize_musique_example  # noqa: E402

OUT_TRAIN = ROOT / "artifacts" / "train_v1_random.jsonl"
OUT_VAL = ROOT / "artifacts" / "val_v1_random.jsonl"
OUT_REPORT = ROOT / "artifacts" / "train_v1_random_report.md"


def random_mask(ctx_len: int, ratio: float, rng: np.random.Generator) -> torch.Tensor:
    """Boolean mask of length ctx_len with k = round(ratio * ctx_len) True
    entries chosen uniformly without replacement. Matches topk_mask in n_kept
    so the only difference between random and KVzip pipelines is *which*
    tokens are kept, not how many.
    """
    k = int(round(ratio * ctx_len))
    if k <= 0:
        return torch.zeros(ctx_len, dtype=torch.bool)
    if k >= ctx_len:
        return torch.ones(ctx_len, dtype=torch.bool)
    keep_indices = rng.choice(ctx_len, size=k, replace=False)
    mask = torch.zeros(ctx_len, dtype=torch.bool)
    mask[keep_indices] = True
    return mask


def select_examples(ds, *, n_train: int, n_val: int, max_tokens: int, tokenizer):
    """Iterate ds, return (train_pool, val_pool) disjoint by example id.

    Identical to build_training_set.py.select_examples. Iteration order is
    deterministic given a fixed dataset version, so the example pool is
    byte-identical to v1's.
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
    example_index: int,
    ratios: list[float],
    cfg: LabelConfig,
) -> list[dict]:
    """Per-(example, ratio) random kept_mask + label_example."""
    rng = np.random.default_rng(42 + example_index)
    ctx_len = ex.context_ids.size(0)
    rows: list[dict] = []
    for ratio in ratios:
        mask = random_mask(ctx_len, ratio, rng)
        cs = CacheState(kept_mask=mask, compressor_name="uniform_random", ratio=ratio)
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
    """Strip transient objects, keep only what's serialized to JSONL.
    metadata.compressor = "uniform_random" — the one schema deviation from v1.
    """
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
            "compressor": "uniform_random",
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
    """Identical to build_training_set.py — undersample majority to hit target."""
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


def per_ratio_class_breakdown(rows: list[dict]) -> dict:
    """{ratio: {label_type: count}} including DROP."""
    out: dict[float, Counter] = defaultdict(Counter)
    for r in rows:
        out[r["ratio"]][r["label_type"]] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--ratios", type=str, default="0.05,0.10,0.20,0.30,0.50,0.80")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--target-minority-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the class-balance RNG. Per-example mask seed is "
                         "always 42 + example_index per the Block-5 spec.")
    ap.add_argument("--out-train", default=None)
    ap.add_argument("--out-val", default=None)
    ap.add_argument("--out-report", default=None)
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
    print("[setup] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("[setup] loading Musique train split...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="train")

    print(f"[setup] selecting {args.n_train} train + {args.n_val} val examples...", flush=True)
    train_pool, val_pool, scanned = select_examples(
        ds, n_train=args.n_train, n_val=args.n_val,
        max_tokens=args.max_tokens, tokenizer=tokenizer,
    )
    print(f"  train_pool={len(train_pool)} val_pool={len(val_pool)} (scanned {scanned} rows)",
          flush=True)

    cfg = LabelConfig()
    print(f"[run] labeling train pool with random masks at {len(ratios)} ratios...", flush=True)
    t0 = time.time()
    train_rows: list[dict] = []
    # example_index runs 0..n_train-1 over train_pool then n_train..n_train+n_val-1 over val_pool.
    # This guarantees each example has a unique seed and train/val don't share mask seeds.
    for i, ex in enumerate(train_pool):
        train_rows.extend(
            label_one_example_at_ratios(ex, example_index=i, ratios=ratios, cfg=cfg)
        )
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:4d}/{len(train_pool)}]  {elapsed:.1f}s  ({rate:.1f} ex/s)",
                  flush=True)

    print(f"\n[run] labeling val pool ({len(val_pool)} examples)...", flush=True)
    val_rows: list[dict] = []
    for j, ex in enumerate(val_pool):
        global_idx = len(train_pool) + j
        val_rows.extend(
            label_one_example_at_ratios(ex, example_index=global_idx, ratios=ratios, cfg=cfg)
        )

    elapsed = time.time() - t0
    print(f"[run] done labeling in {elapsed:.1f}s "
          f"({len(train_rows)+len(val_rows)} raw instances)", flush=True)

    # per-ratio breakdown BEFORE filtering DROP — reportable
    train_per_ratio_full = per_ratio_class_breakdown(train_rows)
    val_per_ratio_full = per_ratio_class_breakdown(val_rows)

    # Filter DROP
    train_kept = [r for r in train_rows if r["label_type"] != LabelType.DROP.value]
    val_kept = [r for r in val_rows if r["label_type"] != LabelType.DROP.value]
    train_dropped = len(train_rows) - len(train_kept)
    val_dropped = len(val_rows) - len(val_kept)

    train_natural_counts = Counter(r["label_type"] for r in train_kept)
    val_counts = Counter(r["label_type"] for r in val_kept)
    print(f"\n[filter] train: kept {len(train_kept)} dropped {train_dropped} "
          f"(natural balance: {dict(train_natural_counts)})", flush=True)
    print(f"[filter] val: kept {len(val_kept)} dropped {val_dropped} "
          f"(natural balance: {dict(val_counts)})", flush=True)

    train_balanced, balance_info = class_balance_undersample(
        train_kept, target_minority_frac=args.target_minority_frac, rng=rng,
    )
    print(f"[balance] {balance_info}", flush=True)

    # Write JSONL
    with out_train.open("w") as f:
        for r in train_balanced:
            f.write(json.dumps(package_row(r, model_name=args.model_name, split="train_v1_random")) + "\n")
    with out_val.open("w") as f:
        for r in val_kept:
            f.write(json.dumps(package_row(r, model_name=args.model_name, split="val_v1_random")) + "\n")

    train_parity = class_ctx_len_parity(train_balanced)
    val_parity = class_ctx_len_parity(val_kept)
    train_per_ratio = Counter(r["ratio"] for r in train_balanced)
    val_per_ratio = Counter(r["ratio"] for r in val_kept)
    val_per_ratio_class = per_ratio_class_breakdown(val_kept)

    md: list[str] = []
    md.append("# Block 5 — Random-mask training-set construction report\n")
    md.append("_Auto-generated by `scripts/build_training_set_random.py`. "
              "Same pipeline as v1 but kept_mask is uniform-random at matched n_kept._\n")
    md.append("## Setup\n")
    md.append(f"- tokenizer / model_name: `{args.model_name}`")
    md.append(f"- ratios swept: {ratios}")
    md.append(f"- max ctx tokens: {args.max_tokens}")
    md.append(f"- labeler: binary (FLAGGED_PARTIAL off), tau_low=0.3, tau_high=0.8")
    md.append(f"- spans: tight (sub-answer via question_decomposition; no fallbacks)")
    md.append(f"- target minority class fraction: {args.target_minority_frac}")
    md.append(f"- per-example mask seed: 42 + example_index (numpy.default_rng)")
    md.append(f"- balance RNG seed: {args.seed}")
    md.append("")
    md.append("## Pool sizes\n")
    md.append(f"- train pool examples: {len(train_pool)}  (scanned {scanned} rows)")
    md.append(f"- val pool examples:   {len(val_pool)}  (disjoint by example id)")
    md.append(f"- raw instances: train {len(train_rows)}  val {len(val_rows)}")
    md.append(f"- DROP filtered: train {train_dropped}  val {val_dropped}")
    md.append("")
    md.append("## Train per-ratio class breakdown (before DROP filter)\n")
    md.append("| ratio | confident_answer | abstain | drop | total |")
    md.append("| --- | --- | --- | --- | --- |")
    for r in ratios:
        c = train_per_ratio_full.get(r, Counter())
        n_conf = c.get("confident_answer", 0)
        n_abst = c.get("abstain", 0)
        n_drop = c.get("drop", 0)
        md.append(f"| {r:.2f} | {n_conf} | {n_abst} | {n_drop} | {n_conf+n_abst+n_drop} |")
    md.append("")
    md.append("## Val per-ratio class breakdown (after DROP filter — what eval uses)\n")
    md.append("| ratio | confident_answer | abstain | total |")
    md.append("| --- | --- | --- | --- |")
    for r in ratios:
        c = val_per_ratio_class.get(r, Counter())
        n_conf = c.get("confident_answer", 0)
        n_abst = c.get("abstain", 0)
        md.append(f"| {r:.2f} | {n_conf} | {n_abst} | {n_conf+n_abst} |")
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
    md.append("## Per-ratio counts after balancing\n")
    md.append("| ratio | train (balanced) | val (natural) |")
    md.append("| --- | --- | --- |")
    for r in ratios:
        md.append(f"| {r:.2f} | {train_per_ratio.get(r, 0)} | {val_per_ratio.get(r, 0)} |")
    md.append("")
    md.append("## Critical: context-length parity (train, balanced)\n")
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
    md.append("## Comparison to v1 reference (committed numbers)\n")
    md.append("- v1 train final: 2669 instances (1868 ABSTAIN + 801 CONFIDENT, 30/70)")
    md.append("- v1 val final:   463 instances (366 ABSTAIN + 97 CONFIDENT, natural)")
    md.append(f"- v1 ctx-len parity: 1.5% (well under 10%)")
    md.append(f"- random run train final: {len(train_balanced)} instances")
    md.append(f"- random run val final:   {len(val_kept)} instances")
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
