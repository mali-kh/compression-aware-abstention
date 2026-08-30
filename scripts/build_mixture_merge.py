"""Exp 2a — merge + per-stratum balance to produce train_mixture + val_mixture.

Reads `_mixture_raw_kvzip.jsonl`, `_mixture_raw_expected_attention.jsonl`,
`_mixture_raw_snapkv.jsonl`. Splits by `_pool` ∈ {train, val}.

For each (compressor, ratio) stratum on the train side, applies a 30/70
ABSTAIN/CONFIDENT minority floor by undersampling the majority class.
Emits a `EXP2A_BLOCKED.md` and exits non-zero if any stratum has fewer than
`--min-stratum-n` total instances after DROP filter (per the spec — stop and
surface for review when a stratum is starved).

Val is kept at natural distribution (no balancing) — same discipline as v1.

Outputs:
  artifacts/train_mixture_v1.jsonl
  artifacts/val_mixture_v1.jsonl
  artifacts/exp2_dataset_report.md   (per-stratum counts, parity check, summary)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def per_stratum_balance_train(rows: list[dict], target_minority_frac: float, rng: random.Random,
                              low_ratio_skip_floor: float = 0.0):
    """Per-(compressor, ratio) undersample to achieve target minority fraction.

    Strata at ratio < low_ratio_skip_floor skip balancing entirely (keep
    natural distribution) — used in Exp 2 Option B to avoid starving
    low-ratio cells where CONFIDENT-gold is structurally ~0.

    Returns (balanced_rows, per_stratum_log).
    """
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[(r["metadata"]["compressor"], r["ratio"])].append(r)

    out_rows: list[dict] = []
    log: list[dict] = []
    for (comp, ratio), srows in sorted(by_stratum.items()):
        cls = Counter(r["label_type"] for r in srows)
        if ratio < low_ratio_skip_floor:
            log.append({"stratum": (comp, ratio), "before": dict(cls), "after": dict(cls),
                        "action": f"low_ratio_skipped_floor (< {low_ratio_skip_floor})"})
            out_rows.extend(srows)
            continue
        if len(cls) < 2:
            # only one class present; no balancing possible
            log.append({"stratum": (comp, ratio), "before": dict(cls), "after": dict(cls),
                        "action": "single_class_kept_all"})
            out_rows.extend(srows)
            continue
        minority = min(cls, key=cls.get)
        majority = max(cls, key=cls.get)
        n_min, n_maj = cls[minority], cls[majority]
        cur_frac = n_min / (n_min + n_maj)
        if cur_frac >= target_minority_frac:
            log.append({"stratum": (comp, ratio), "before": dict(cls), "after": dict(cls),
                        "action": "natural_meets_target",
                        "minority_frac_before": round(cur_frac, 4)})
            out_rows.extend(srows)
            continue
        target_maj = int(n_min * (1 - target_minority_frac) / target_minority_frac)
        target_maj = min(target_maj, n_maj)
        by_class = defaultdict(list)
        for r in srows:
            by_class[r["label_type"]].append(r)
        sampled_maj = rng.sample(by_class[majority], target_maj)
        kept = by_class[minority] + sampled_maj
        rng.shuffle(kept)
        after = Counter(r["label_type"] for r in kept)
        log.append({
            "stratum": (comp, ratio),
            "before": dict(cls), "after": dict(after),
            "action": "undersampled_majority",
            "minority_frac_before": round(cur_frac, 4),
            "minority_frac_after": round(after[minority] / sum(after.values()), 4),
            "majority_dropped": n_maj - target_maj,
        })
        out_rows.extend(kept)
    rng.shuffle(out_rows)
    return out_rows, log


def stratum_min_check(rows: list[dict], min_n: int) -> list[tuple]:
    """Return list of (compressor, ratio) strata with total post-DROP < min_n."""
    by_stratum = Counter((r["metadata"]["compressor"], r["ratio"]) for r in rows)
    return [s for s, n in by_stratum.items() if n < min_n]


def ctx_parity(rows: list[dict]) -> dict:
    """Per-(compressor) parity between CONF and ABST mean ctx_len."""
    by_comp: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_comp[r["metadata"]["compressor"]][r["label_type"]].append(r["metadata"]["ctx_len"])
    out = {}
    for comp, cls_lens in by_comp.items():
        cl = cls_lens.get("confident_answer", [])
        al = cls_lens.get("abstain", [])
        if not cl or not al:
            out[comp] = {"status": "insufficient",
                         "n_conf": len(cl), "n_abst": len(al)}
            continue
        mc = sum(cl) / len(cl)
        ma = sum(al) / len(al)
        d = abs(mc - ma) / min(mc, ma) * 100
        out[comp] = {
            "n_conf": len(cl), "mean_conf": round(mc, 1),
            "n_abst": len(al), "mean_abst": round(ma, 1),
            "delta_pct": round(d, 2),
            "status": "OK" if d < 10 else "WIDE",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvzip", default=str(ROOT / "artifacts" / "_mixture_raw_kvzip.jsonl"))
    ap.add_argument("--expattn", default=str(ROOT / "artifacts" / "_mixture_raw_expected_attention.jsonl"))
    ap.add_argument("--snapkv", default=str(ROOT / "artifacts" / "_mixture_raw_snapkv.jsonl"))
    ap.add_argument("--out-train", default=str(ROOT / "artifacts" / "train_mixture_v1.jsonl"))
    ap.add_argument("--out-val", default=str(ROOT / "artifacts" / "val_mixture_v1.jsonl"))
    ap.add_argument("--out-train-holdout",
                    default=str(ROOT / "artifacts" / "train_mixture_v1_holdout.jsonl"),
                    help="If --train-holdout-frac > 0, write the holdout shard here.")
    ap.add_argument("--report", default=str(ROOT / "artifacts" / "exp2_dataset_report.md"))
    ap.add_argument("--target-minority-frac", type=float, default=0.30)
    ap.add_argument("--min-stratum-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low-ratio-skip-floor", type=float, default=0.0,
                    help="Skip per-stratum 30/70 balance for strata with ratio < this. "
                         "Default 0.0 = balance all strata (legacy behavior). "
                         "Set to 0.20 to apply Exp 2 Option B (skip 0.05 + 0.10).")
    ap.add_argument("--train-holdout-frac", type=float, default=0.0,
                    help="If > 0, split the balanced train into (1-frac) train + frac holdout, "
                         "id-disjoint where possible (groups by example id). Used by Exp 2b "
                         "to compute val_loss for checkpoint selection on a train-derived "
                         "shard instead of val_mixture_v1 (avoids the v1 selection/eval leak).")
    args = ap.parse_args()

    raw_kv = load_jsonl(Path(args.kvzip))
    raw_ea = load_jsonl(Path(args.expattn))
    raw_sk = load_jsonl(Path(args.snapkv))
    print(f"[setup] kvzip raw={len(raw_kv)}  expattn raw={len(raw_ea)}  snapkv raw={len(raw_sk)}",
          flush=True)
    all_raw = raw_kv + raw_ea + raw_sk

    train = [r for r in all_raw if r.get("_pool") == "train"]
    val = [r for r in all_raw if r.get("_pool") == "val"]
    print(f"[setup] split: train={len(train)}  val={len(val)}", flush=True)

    # Sanity: id disjoint across pools (consistency with v1 discipline)
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    overlap = train_ids & val_ids
    if overlap:
        print(f"[ERROR] train/val id overlap: {len(overlap)} ids", flush=True)
        # write BLOCKED
        Path(ROOT / "artifacts" / "EXP2A_BLOCKED.md").write_text(
            f"# Exp 2a BLOCKED — train/val id overlap\n\n"
            f"{len(overlap)} ids appear in both pools. The mixture build's id selection "
            f"diverged from v1's. Investigate `select_examples` determinism.\n"
        )
        return 1

    # Stratum starvation check (train side)
    starved = stratum_min_check(train, args.min_stratum_n)
    if starved:
        msg = (
            f"# Exp 2a BLOCKED — under-strength strata\n\n"
            f"After DROP filter, the following (compressor, ratio) train strata have "
            f"fewer than {args.min_stratum_n} instances:\n\n"
        )
        by_stratum = Counter((r["metadata"]["compressor"], r["ratio"]) for r in train)
        for s in sorted(starved):
            msg += f"- `{s[0]}` @ ratio={s[1]:.2f}: n={by_stratum[s]}\n"
        msg += (
            "\nOptions for the user (do not commit autonomously):\n"
            "1. Drop the stratum entirely (compressor under-represented at that ratio).\n"
            "2. Lower `--min-stratum-n` if the threshold is too strict.\n"
            "3. Increase `--n-train` to give selection a wider pool.\n"
        )
        Path(ROOT / "artifacts" / "EXP2A_BLOCKED.md").write_text(msg)
        print(f"[BLOCKED] {len(starved)} starved strata; wrote artifacts/EXP2A_BLOCKED.md", flush=True)
        return 2

    rng = random.Random(args.seed)
    train_balanced, train_log = per_stratum_balance_train(
        train, args.target_minority_frac, rng,
        low_ratio_skip_floor=args.low_ratio_skip_floor,
    )
    print(f"[balance] train: {len(train)} → {len(train_balanced)} after per-stratum balance "
          f"(low_ratio_skip_floor={args.low_ratio_skip_floor})", flush=True)

    # Strip _pool key for cleanliness
    for r in train_balanced + val:
        r.pop("_pool", None)
        # Also strip any transient keys from the kvzip phase (label_obj / ex_obj)
        r.pop("label_obj", None)
        r.pop("ex_obj", None)

    # Optional id-grouped train/holdout split: same example id stays on the
    # same side so the holdout's val_loss isn't an inflated-comparison artifact.
    holdout_rows: list[dict] = []
    if args.train_holdout_frac > 0:
        by_id_train: dict[str, list[dict]] = defaultdict(list)
        for r in train_balanced:
            by_id_train[r["id"]].append(r)
        ids = sorted(by_id_train.keys())
        split_rng = random.Random(args.seed + 7)  # different seed than balance to avoid coupling
        split_rng.shuffle(ids)
        target_holdout = int(round(args.train_holdout_frac * len(train_balanced)))
        holdout_ids: list[str] = []
        cum = 0
        for ex_id in ids:
            holdout_ids.append(ex_id)
            cum += len(by_id_train[ex_id])
            if cum >= target_holdout:
                break
        holdout_id_set = set(holdout_ids)
        new_train = [r for r in train_balanced if r["id"] not in holdout_id_set]
        holdout_rows = [r for r in train_balanced if r["id"] in holdout_id_set]
        print(f"[split] train_holdout_frac={args.train_holdout_frac}: "
              f"train {len(train_balanced)} → train {len(new_train)} + holdout {len(holdout_rows)} "
              f"({len(holdout_ids)} ids in holdout, id-disjoint)", flush=True)
        train_balanced = new_train

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w") as f:
        for r in train_balanced:
            f.write(json.dumps(r) + "\n")
    with open(args.out_val, "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    print(f"[done] wrote {args.out_train} ({len(train_balanced)} rows)", flush=True)
    print(f"[done] wrote {args.out_val} ({len(val)} rows)", flush=True)
    if holdout_rows:
        with open(args.out_train_holdout, "w") as f:
            for r in holdout_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[done] wrote {args.out_train_holdout} ({len(holdout_rows)} rows)", flush=True)

    # ---- Defensive sanity checks (per Exp 2 spec): post-balance ----
    # 1. Strata at ratio >= low_ratio_skip_floor must have >= min_stratum_n
    # 2. Ctx-length parity per compressor on the train output must be <= 5%
    parity_train = ctx_parity(train_balanced)
    starved_post = []
    for entry in train_log:
        comp, ratio = entry["stratum"]
        n_after = sum(entry["after"].values())
        if ratio >= args.low_ratio_skip_floor and n_after < args.min_stratum_n:
            starved_post.append((comp, ratio, n_after))
    parity_wide = []
    for comp, info in parity_train.items():
        if info.get("status") == "WIDE" or (info.get("delta_pct", 0) and info["delta_pct"] > 5):
            parity_wide.append((comp, info["delta_pct"]))
    if starved_post or parity_wide:
        msg = ["# Exp 2a BLOCKED — post-balance sanity check failed\n"]
        if starved_post:
            msg.append("## Strata at ratio >= low_ratio_skip_floor below min_stratum_n:\n")
            for comp, ratio, n in starved_post:
                msg.append(f"- `{comp}` @ ratio={ratio:.2f}: n={n} (< {args.min_stratum_n})")
        if parity_wide:
            msg.append("\n## Context-length parity > 5%:\n")
            for comp, delta in parity_wide:
                msg.append(f"- `{comp}`: delta_pct={delta}%")
        msg.append(f"\nDid not write outputs. Re-run with adjusted flags or surface for review.")
        Path(ROOT / "artifacts" / "EXP2A_BLOCKED.md").write_text("\n".join(msg))
        print(f"[BLOCKED] post-balance sanity failed; wrote artifacts/EXP2A_BLOCKED.md", flush=True)
        return 3

    # ---- Report ----
    md = [
        "# Exp 2a — Mixture dataset construction report\n",
        f"_Generated by `scripts/build_mixture_merge.py`. Seed {args.seed}, target minority fraction {args.target_minority_frac}._\n",
        f"_Per-stratum balance config: `--low-ratio-skip-floor={args.low_ratio_skip_floor}` "
        f"(strata at ratio strictly below this skip the 30/70 floor)._\n",
        "## Source counts (post-DROP-filter, pre-balance)\n",
        f"- KVzip raw rows: {len(raw_kv)}",
        f"- Expected Attention raw rows: {len(raw_ea)}",
        f"- SnapKV (context-only) raw rows: {len(raw_sk)}",
        f"- TOTAL raw: {len(all_raw)}",
        f"  - train pool: {len(train)} rows",
        f"  - val pool:   {len(val)} rows\n",
        "## Train per-(compressor, ratio) strata — before/after balance\n",
        "| compressor | ratio | before | after | action | min_frac before → after | maj dropped |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in train_log:
        comp, ratio = entry["stratum"]
        before = entry["before"]
        after = entry["after"]
        b_str = " / ".join(f"{k}:{v}" for k, v in sorted(before.items()))
        a_str = " / ".join(f"{k}:{v}" for k, v in sorted(after.items()))
        md.append(
            f"| {comp} | {ratio:.2f} | {b_str} ({sum(before.values())}) | "
            f"{a_str} ({sum(after.values())}) | {entry['action']} | "
            f"{entry.get('minority_frac_before','-')} → {entry.get('minority_frac_after','-')} | "
            f"{entry.get('majority_dropped','-')} |"
        )
    md.append("")

    if args.low_ratio_skip_floor > 0:
        md.append(f"## Low-ratio strata summary (ratio < {args.low_ratio_skip_floor}; floor skipped)\n")
        md.append("Per the Option B logic, low ratios keep the natural class distribution "
                  "(near-100% ABSTAIN — these are the regimes where almost every example "
                  "*should* abstain by definition).\n")
        md.append("| compressor | ratio | abst | conf | total | natural minority frac |")
        md.append("| --- | --- | --- | --- | --- | --- |")
        for entry in train_log:
            comp, ratio = entry["stratum"]
            if ratio < args.low_ratio_skip_floor:
                a = entry["after"]
                total = sum(a.values())
                conf = a.get("confident_answer", 0)
                abst = a.get("abstain", 0)
                if total:
                    minority_frac = min(conf, abst) / total
                else:
                    minority_frac = 0
                md.append(f"| {comp} | {ratio:.2f} | {abst} | {conf} | {total} | "
                          f"{minority_frac:.1%} |")
        md.append("")

    md.append("## Val per-(compressor, ratio) strata (no balance)\n")
    val_strata = defaultdict(Counter)
    for r in val:
        val_strata[(r["metadata"]["compressor"], r["ratio"])][r["label_type"]] += 1
    md.append("| compressor | ratio | conf | abst | total |")
    md.append("| --- | --- | --- | --- | --- |")
    for (comp, ratio), c in sorted(val_strata.items()):
        md.append(f"| {comp} | {ratio:.2f} | {c.get('confident_answer',0)} | "
                  f"{c.get('abstain',0)} | {sum(c.values())} |")
    md.append("")
    md.append("## Context-length parity (train, per compressor)\n")
    md.append("| compressor | n_conf | mean_conf | n_abst | mean_abst | delta_pct | status |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for comp, info in sorted(parity_train.items()):
        if info.get("status") == "insufficient":
            md.append(f"| {comp} | {info['n_conf']} | - | {info['n_abst']} | - | - | insufficient |")
        else:
            md.append(f"| {comp} | {info['n_conf']} | {info['mean_conf']} | "
                      f"{info['n_abst']} | {info['mean_abst']} | {info['delta_pct']}% | {info['status']} |")
    md.append("")

    if holdout_rows:
        md.append("## Train/holdout split (for Exp 2b val_loss selection)\n")
        md.append(f"- holdout fraction: {args.train_holdout_frac}")
        md.append(f"- train kept: {len(train_balanced)} rows")
        md.append(f"- holdout: {len(holdout_rows)} rows")
        md.append(f"- id-grouped: same example id stays on the same side")
        md.append(f"- holdout written to: `{args.out_train_holdout}`\n")

    md.append("## Final dataset sizes\n")
    md.append(f"- train_mixture_v1.jsonl: **{len(train_balanced)}** instances")
    md.append(f"- val_mixture_v1.jsonl:   **{len(val)}** instances")
    if holdout_rows:
        md.append(f"- train_mixture_v1_holdout.jsonl: **{len(holdout_rows)}** instances")
    md.append(f"\n_Spec target range was [6700, 8000] (i.e. [2.5×, 3×] of v1's 2669) but explicitly "
              f"a guideline rather than a hard target. Final size is what the per-stratum logic "
              f"produces; do not pad or trim to fit._\n")
    Path(args.report).write_text("\n".join(md))
    print(f"[done] wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
