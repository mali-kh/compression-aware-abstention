"""Block 1 follow-up: pooled McNemar at r=0.50 + high-fill over-abstention at r=0.80."""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

# Reuse Block 1 helpers
sys.path.insert(0, str(Path(__file__).parent))
from block1_stats import (
    COMPRESSORS, ART, RATIO, N_BOOT, SEED,
    honest_correct, load_jsonl, at_ratio, index_by_id,
    bootstrap_ci, paired_bootstrap_diff_ci, mcnemar_exact,
)

CONFIDENT_GOLDS = ("confident", "confident_answer")


def build_at_ratio(name, v1_file, mix_file, p2_file, ratio):
    """Same as block1_stats.build_compressor but for arbitrary ratio.
    Returns dict with arrays parallel by joined id, plus gold_label and model_abstained per condition."""
    v1 = at_ratio(load_jsonl(ART / v1_file), r=ratio)
    mx = at_ratio(load_jsonl(ART / mix_file), r=ratio)
    p2 = at_ratio(load_jsonl(ART / p2_file), r=ratio)

    v1_ix = index_by_id(v1)
    mx_ix = index_by_id(mx)
    p2_ix = index_by_id(p2)

    ids = sorted(set(v1_ix) & set(mx_ix) & set(p2_ix))

    out = {
        "ids": ids,
        "gold_label": [],
        "base_hc": [], "v1_hc": [], "mix_hc": [], "p2_hc": [],
        "base_ab": [], "v1_ab": [], "mix_ab": [], "p2_ab": [],
        "n_v1": len(v1), "n_mx": len(mx), "n_p2": len(p2),
    }
    for i in ids:
        v1r, mxr, p2r = v1_ix[i], mx_ix[i], p2_ix[i]
        gl = v1r["gold_label"]
        out["gold_label"].append(gl)
        b = v1r["base"]; a = v1r["adapter"]; am = mxr["adapter"]
        out["base_hc"].append(honest_correct(gl, b["classification"], b["model_abstained"]))
        out["v1_hc"].append(honest_correct(gl, a["classification"], a["model_abstained"]))
        out["mix_hc"].append(honest_correct(gl, am["classification"], am["model_abstained"]))
        out["p2_hc"].append(honest_correct(gl, p2r["classification"], p2r["model_abstained"]))
        out["base_ab"].append(b["model_abstained"])
        out["v1_ab"].append(a["model_abstained"])
        out["mix_ab"].append(am["model_abstained"])
        out["p2_ab"].append(p2r["model_abstained"])

    for k in ("gold_label",):
        out[k] = np.array(out[k])
    for k in ("base_hc", "v1_hc", "mix_hc", "p2_hc",
              "base_ab", "v1_ab", "mix_ab", "p2_ab"):
        out[k] = np.array(out[k], dtype=bool)
    return out


def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return float("nan"), float("nan")
    r = binomtest(int(k), int(n), p=0.5)
    ci = r.proportion_ci(confidence_level=conf, method="wilson")
    return float(ci.low), float(ci.high)


def main():
    rng = np.random.default_rng(SEED)

    # ---- Load r=0.50 data for all four compressors ----
    data_05 = {name: build_at_ratio(name, v1f, mxf, p2f, 0.5)
               for (name, v1f, mxf, p2f) in COMPRESSORS}
    # ---- Load r=0.80 data ----
    data_08 = {name: build_at_ratio(name, v1f, mxf, p2f, 0.8)
               for (name, v1f, mxf, p2f) in COMPRESSORS}

    print("=== Sanity: n at r=0.5 (joined) and r=0.8 (joined) per compressor ===")
    for name in data_05:
        n5 = len(data_05[name]["ids"]); n8 = len(data_08[name]["ids"])
        print(f"  {name:14s}  r=0.5 n={n5}   r=0.8 n={n8}")
    print()

    print("=== Sanity: confident-gold counts at r=0.8 (vs report 69/65/59/67) ===")
    for name in data_08:
        gl = data_08[name]["gold_label"]
        nC = int(np.sum(np.isin(gl, CONFIDENT_GOLDS)))
        nA = int(np.sum(gl == "abstain"))
        print(f"  {name:14s}  n_total={len(gl)}  n_confident={nC}  n_abstain={nA}")
    print()

    # ===========================================================
    # SECTION 1: Pooled McNemar at r=0.50
    # ===========================================================
    print("### Section 1: Pooled McNemar at r=0.50 (n_pooled=166)")
    print()
    pooled = {"base": [], "v1": [], "mix": [], "p2": []}
    for name in data_05:
        d = data_05[name]
        pooled["base"].extend(d["base_hc"])
        pooled["v1"].extend(d["v1_hc"])
        pooled["mix"].extend(d["mix_hc"])
        pooled["p2"].extend(d["p2_hc"])
    for k in pooled:
        pooled[k] = np.array(pooled[k], dtype=bool)
    n_pool = len(pooled["mix"])
    print(f"(pooled n = {n_pool})")
    print()

    header = f"{'comparison':18s} | n   | b   | c   | stat | exact p | Δ      | 95% CI on Δ"
    print(header)
    print("-" * len(header))
    pooled_comparisons = [
        ("mix vs P2",       "mix",  "p2"),
        ("mix vs Fragment", "mix",  "base"),
        ("v1 vs P2",        "v1",   "p2"),
    ]
    section1_rows = []
    for label, A, B in pooled_comparisons:
        a, b_disc, c_disc, dd, stat, p = mcnemar_exact(pooled[A], pooled[B])
        pt, lo, hi = paired_bootstrap_diff_ci(pooled[A], pooled[B], rng)
        sign = "+" if pt >= 0 else ""
        s_lo = "+" if lo >= 0 else ""
        s_hi = "+" if hi >= 0 else ""
        print(f"{label:18s} | {n_pool:3d} | {b_disc:3d} | {c_disc:3d} | {stat:4d} | {p:7.4f} | {sign}{pt:5.2f}  | [{s_lo}{lo:.2f}, {s_hi}{hi:.2f}]")
        section1_rows.append((label, n_pool, b_disc, c_disc, stat, p, pt, lo, hi))
    print()

    # Sanity check: discordants of pooled mix-vs-P2 should equal sum of per-compressor (b+c)
    per_comp_disc = []
    for name in data_05:
        d = data_05[name]
        _, b_disc, c_disc, _, _, _ = mcnemar_exact(d["mix_hc"], d["p2_hc"])
        per_comp_disc.append((name, b_disc, c_disc, b_disc + c_disc))
    total_disc_per_comp = sum(x[3] for x in per_comp_disc)
    pooled_b, pooled_c = section1_rows[0][2], section1_rows[0][3]
    print(f"(sanity: per-compressor mix-vs-P2 discordants = {[(n, bd+cd) for (n,bd,cd,_) in per_comp_disc]} ; sum = {total_disc_per_comp})")
    print(f"(sanity: pooled mix-vs-P2 b+c = {pooled_b + pooled_c}; match = {pooled_b + pooled_c == total_disc_per_comp})")
    print()

    # ===========================================================
    # SECTION 2a: Over-abstention rate on CONFIDENT-gold at r=0.80
    # ===========================================================
    print("### Section 2a: Over-abstention rate (model_abstained=True) on CONFIDENT-gold at r=0.80")
    print()
    header = f"{'Compressor':14s} | n_C | {'base ovc (CI)':22s} | {'v1 ovc (CI)':22s} | {'mix ovc (CI)':22s} | P2 ovc (CI)"
    print(header)
    print("-" * len(header))
    section2a_rows = []
    for name in data_08:
        d = data_08[name]
        gl = d["gold_label"]
        mask_C = np.isin(gl, CONFIDENT_GOLDS)
        nC = int(mask_C.sum())
        cells = []
        ovc_dict = {}
        for cond in ("base", "v1", "mix", "p2"):
            ab = d[f"{cond}_ab"][mask_C]
            k = int(ab.sum())
            rate = k / nC if nC else float("nan")
            lo, hi = wilson_ci(k, nC)
            cells.append(f"{rate:.2f} [{lo:.2f}, {hi:.2f}] ({k}/{nC})")
            ovc_dict[cond] = (rate, lo, hi, k, nC)
        print(f"{name:14s} | {nC:3d} | {cells[0]:22s} | {cells[1]:22s} | {cells[2]:22s} | {cells[3]}")
        section2a_rows.append((name, nC, ovc_dict))
    print()

    # ===========================================================
    # SECTION 2b: Paired McNemar mix-vs-P2 on CONFIDENT-gold at r=0.80
    # "correct" here means honest_correct restricted to confident-gold
    # ===========================================================
    print("### Section 2b: Paired McNemar mix-vs-P2 on CONFIDENT-gold at r=0.80 (correct = honest_correct on this subset)")
    print()
    header = f"{'Compressor':14s} | n_C | b (mix>P2) | c (P2>mix) | stat | exact p | Δ      | 95% CI on Δ"
    print(header)
    print("-" * len(header))
    pooled_C_mix = []
    pooled_C_p2 = []
    for name in data_08:
        d = data_08[name]
        gl = d["gold_label"]
        mask_C = np.isin(gl, CONFIDENT_GOLDS)
        mix_C = d["mix_hc"][mask_C]
        p2_C  = d["p2_hc"][mask_C]
        nC = len(mix_C)
        a, b_disc, c_disc, dd, stat, p = mcnemar_exact(mix_C, p2_C)
        pt, lo, hi = paired_bootstrap_diff_ci(mix_C, p2_C, rng)
        sign = "+" if pt >= 0 else ""
        s_lo = "+" if lo >= 0 else ""
        s_hi = "+" if hi >= 0 else ""
        print(f"{name:14s} | {nC:3d} | {b_disc:10d} | {c_disc:10d} | {stat:4d} | {p:7.4f} | {sign}{pt:5.2f}  | [{s_lo}{lo:.2f}, {s_hi}{hi:.2f}]")
        pooled_C_mix.extend(mix_C.tolist())
        pooled_C_p2.extend(p2_C.tolist())

    # POOLED row
    pooled_C_mix = np.array(pooled_C_mix, dtype=bool)
    pooled_C_p2 = np.array(pooled_C_p2, dtype=bool)
    nC_pool = len(pooled_C_mix)
    a, b_disc, c_disc, dd, stat, p = mcnemar_exact(pooled_C_mix, pooled_C_p2)
    pt, lo, hi = paired_bootstrap_diff_ci(pooled_C_mix, pooled_C_p2, rng)
    sign = "+" if pt >= 0 else ""
    s_lo = "+" if lo >= 0 else ""
    s_hi = "+" if hi >= 0 else ""
    print(f"{'POOLED':14s} | {nC_pool:3d} | {b_disc:10d} | {c_disc:10d} | {stat:4d} | {p:7.4f} | {sign}{pt:5.2f}  | [{s_lo}{lo:.2f}, {s_hi}{hi:.2f}]")
    print()

    # ===========================================================
    # Bullet summary
    # ===========================================================
    print("### Bullet summary")
    print()
    # Mix-vs-P2 pooled at r=0.50
    label, n_pool, b_disc, c_disc, stat, p, pt, lo, hi = section1_rows[0]
    print(f"1. **Pooled mix vs P2 at r=0.50** (n={n_pool}): b={b_disc} mix-only correct, c={c_disc} P2-only correct, "
          f"Δ={pt:+.3f} [{lo:+.3f}, {hi:+.3f}], McNemar exact p={p:.4f}. "
          f"{'SIGNIFICANT — adapter beats prompt across the pooled slice.' if p < 0.05 else 'Not significant at α=0.05; adapter trends positive but the pooled effect is modest.'}")
    label, n_pool, b_disc, c_disc, stat, p, pt, lo, hi = section1_rows[1]
    print(f"   Pooled mix vs Fragment: Δ={pt:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.4g} ({'sig' if p < 0.05 else 'n.s.'}).")
    label, n_pool, b_disc, c_disc, stat, p, pt, lo, hi = section1_rows[2]
    print(f"   Pooled v1 vs P2: Δ={pt:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.4f} ({'sig' if p < 0.05 else 'n.s.'}).")
    print()
    print("2. **Over-abstention claim at r=0.80 (CONFIDENT-gold)**:")
    for (name, nC, ovc) in section2a_rows:
        b_rate, *_ = ovc["base"]; v_rate, *_ = ovc["v1"]; m_rate, *_ = ovc["mix"]; p_rate, *_ = ovc["p2"]
        print(f"   - {name}: base={b_rate:.2f}  v1={v_rate:.2f}  mix={m_rate:.2f}  P2={p_rate:.2f}  (n_C={nC})")
    print()


if __name__ == "__main__":
    main()
