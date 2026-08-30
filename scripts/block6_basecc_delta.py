"""Block 6 follow-up — recompute KVzip cc cells against base under KVzip cc.

Block 6 originally compared cc-inference adapters to base under PROMPT-STYLE
inference (no base-cc on disk at the time). This script recomputes the two
KVzip cc cells using v1_eval_base_compressed_kvzip.jsonl as the base
comparator.

Cells affected:
  mixture-ps cc, KVzip val   (was 0.532 vs base-ps;  recompute vs base-cc)
  mixture-cc cc, KVzip val   (was 0.511 vs base-ps;  recompute vs base-cc)
The SnapKV-ctx cc cell stays as-is (no base-SnapKV-cc eval; left noted).

Output: a small delta-report comparing old vs new preservation ratios.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ART = Path("artifacts")
CONFIDENT_GOLDS = ("confident", "confident_answer")


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - spread) / denom
    hi = (center + spread) / denom
    return p, max(0.0, lo), min(1.0, hi)


def is_confident(g): return g in CONFIDENT_GOLDS
def is_correct_answer(d): return d["classification"] == "correct" and d["model_abstained"] is False


def aggregate_pair(base_file: Path, adapter_file: Path, ratio_filter=None):
    """Pair base-cc and adapter-cc rows by (id, ratio); return:
       n_C, base_correct, adapter_correct."""
    base_rows = [r for r in load_jsonl(base_file) if r.get("classification") != "error"]
    adapter_rows = [r for r in load_jsonl(adapter_file) if r.get("classification") != "error"]
    base_ix = {(r["id"], r["ratio"]): r for r in base_rows}
    n = bc = ac = 0
    for ar in adapter_rows:
        key = (ar["id"], ar["ratio"])
        br = base_ix.get(key)
        if br is None:
            continue
        if not is_confident(ar.get("gold_label", "")):
            continue
        if ratio_filter is not None and not math.isclose(ar["ratio"], ratio_filter):
            continue
        n += 1
        if is_correct_answer(br):
            bc += 1
        if is_correct_answer(ar):
            ac += 1
    return n, bc, ac


# Reference numbers from the original Block 6 (vs base-ps)
ORIG_AGG = {
    "mixture-ps cc, KVzip val": (97, 47, 25, 0.485, 0.258, 0.532),  # n_C, b cor, a cor, b rate, a rate, pres
    "mixture-cc cc, KVzip val": (97, 47, 24, 0.485, 0.247, 0.511),
}
ORIG_R080 = {
    "mixture-ps cc, KVzip val": (69, 36, 20, 0.522, 0.290, 0.556),
    "mixture-cc cc, KVzip val": (69, 36, 22, 0.522, 0.319, 0.611),
}


def fmt_ci(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    base_cc = ART / "v1_eval_base_compressed_kvzip.jsonl"
    if not base_cc.exists():
        raise SystemExit(f"missing {base_cc} — run the base-cc eval first")

    print("# Block 6 follow-up — KVzip cc cells with base-cc comparator\n")

    # Section 1 — sanity check on base-cc aggregate
    rows = [r for r in load_jsonl(base_cc) if r.get("classification") != "error"]
    n_total = len(rows)
    # honest accuracy: includes correct + correct-abstain
    def hc(r):
        gl = r.get("gold_label", "")
        if gl == "abstain":
            return r["model_abstained"] or r["classification"] == "correct"
        return r["classification"] == "correct"
    agg_hon = sum(1 for r in rows if hc(r)) / max(n_total, 1)
    print(f"## Section 1: Sanity check\n")
    print(f"Base under KVzip cc inference, n={n_total}")
    print(f"  Aggregate honest accuracy: {agg_hon:.3f}")
    print(f"  (Reference: base under prompt-style inference = 0.317 from existing tables)\n")

    # base-cc CONFIDENT-gold answer-correct counts (aggregate + r=0.80)
    conf_rows = [r for r in rows if is_confident(r.get("gold_label", ""))]
    n_C_agg = len(conf_rows)
    base_cc_correct_agg = sum(1 for r in conf_rows if is_correct_answer(r))
    r080_rows = [r for r in conf_rows if math.isclose(r["ratio"], 0.80)]
    n_C_r080 = len(r080_rows)
    base_cc_correct_r080 = sum(1 for r in r080_rows if is_correct_answer(r))
    print(f"Base-cc CONFIDENT-gold answer-correct counts:")
    print(f"  Aggregate (n_C={n_C_agg}): {base_cc_correct_agg} correct  "
          f"({base_cc_correct_agg/n_C_agg:.3f})")
    print(f"  r=0.80   (n_C={n_C_r080}): {base_cc_correct_r080} correct  "
          f"({base_cc_correct_r080/n_C_r080:.3f})\n")

    # Section 2 — recomputed cells
    print("## Section 2: Updated Block 6 cc-inference cells\n")
    targets = [
        ("mixture-ps cc, KVzip val", "v1_eval_mixture_compressed_kvzip.jsonl"),
        ("mixture-cc cc, KVzip val", "v1_eval_mixture_cc_kvzip.jsonl"),
    ]
    for view_name, view_filter in [("Aggregate", None), ("r=0.80", 0.80)]:
        print(f"### {view_name}\n")
        print("```")
        hdr = (f"{'condition':<28s} | {'n_C':>4s} | {'b cc':>4s} | {'a':>4s} | "
               f"{'base-cc rate (Wilson 95% CI)':<30s} | {'adapter rate (Wilson 95% CI)':<30s} | "
               f"{'pres NEW':>8s} | {'pres OLD':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for label, adapter_file in targets:
            n_C, bc, ac = aggregate_pair(base_cc, ART / adapter_file, view_filter)
            b_rate, b_lo, b_hi = wilson_ci(bc, n_C)
            a_rate, a_lo, a_hi = wilson_ci(ac, n_C)
            pres_new = ac / bc if bc > 0 else float("nan")
            orig = ORIG_AGG[label] if view_filter is None else ORIG_R080[label]
            pres_old = orig[5]
            print(f"{label:<28s} | {n_C:>4d} | {bc:>4d} | {ac:>4d} | "
                  f"{fmt_ci(b_rate, b_lo, b_hi):<30s} | {fmt_ci(a_rate, a_lo, a_hi):<30s} | "
                  f"{pres_new:>8.3f} | {pres_old:>8.3f}")
        print("```\n")

    # Section 3 — interpretation
    print("## Section 3: Interpretation\n")
    # compute the four key preservation ratios for the bullet
    def pres(view_filter):
        out = {}
        for label, adapter_file in targets:
            n_C, bc, ac = aggregate_pair(base_cc, ART / adapter_file, view_filter)
            out[label] = (n_C, bc, ac, ac/bc if bc > 0 else float("nan"))
        return out
    agg = pres(None)
    r080 = pres(0.80)
    msg = []
    for label in [t[0] for t in targets]:
        a_n, a_bc, a_ac, a_p = agg[label]
        r_n, r_bc, r_ac, r_p = r080[label]
        msg.append(f"{label}: agg pres = {a_p:.3f} (a={a_ac}/{a_bc}); r=0.80 pres = {r_p:.3f} (a={r_ac}/{r_bc})")
    print("- Recomputed preservation ratios (vs base-cc, KVzip cc inference):")
    for m in msg:
        print(f"    {m}")
    above_08 = [label for label in [t[0] for t in targets]
                if agg[label][3] >= 0.8]
    if len(above_08) == 2:
        print(f"- Both KVzip cc cells now preserve ≥ 0.8 of base-cc's correct answers, "
              f"clearing the 0.8 threshold.")
    else:
        print(f"- Cells clearing 0.8 threshold (aggregate): {above_08 or 'none'}.")


if __name__ == "__main__":
    main()
