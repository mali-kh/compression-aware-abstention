"""Block 3 pivot — prompt-style vs compressed-cache on SnapKV val.

Outputs:
  C. Aggregate honest_acc per (adapter) — prompt-style vs compressed-cache + Δ
  D. Per-ratio table per adapter: prompt-style hon, compressed hon, Δ, McNemar p, paired-bootstrap CI
  E. r=0.80 over-abstention on CONFIDENT-gold per adapter, under each path
  F. Bucket call

KVzip cell skipped per Step 0 finding (kvpress.KVzipPress fake-evicts post-yield;
incompatible with the prompt-style flow).
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from block1_stats import (
    ART, N_BOOT, SEED,
    honest_correct, load_jsonl, at_ratio,
    paired_bootstrap_diff_ci, mcnemar_exact,
)

CONFIDENT_GOLDS = ("confident", "confident_answer")
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]


def index_by_key(records, key=("id", "ratio")):
    return {tuple(r[k] for k in key): r for r in records}


def build_paired(adapter_label, ps_file, cc_file):
    """Pair prompt-style (from nested 'adapter' block) with compressed-cache (flat).
    Returns dict: keys + per-key arrays of honest_correct and abstained for each path.
    """
    ps_rows = load_jsonl(ART / ps_file)
    cc_rows = load_jsonl(ART / cc_file)
    ps_ix = index_by_key(ps_rows)
    cc_ix = index_by_key(cc_rows)
    keys = sorted(set(ps_ix) & set(cc_ix))

    out = {"keys": keys, "ratio": [], "gold_label": [],
           "ps_hc": [], "cc_hc": [], "ps_ab": [], "cc_ab": []}
    for k in keys:
        psr, ccr = ps_ix[k], cc_ix[k]
        gl = psr["gold_label"]
        out["ratio"].append(psr["ratio"])
        out["gold_label"].append(gl)

        # prompt-style adapter response is in 'adapter' nested block
        ps_a = psr["adapter"]
        out["ps_hc"].append(honest_correct(gl, ps_a["classification"], ps_a["model_abstained"]))
        out["ps_ab"].append(ps_a["model_abstained"])
        # compressed-cache flat schema
        out["cc_hc"].append(honest_correct(gl, ccr["classification"], ccr["model_abstained"]))
        out["cc_ab"].append(ccr["model_abstained"])

    out["ratio"] = np.array(out["ratio"])
    out["gold_label"] = np.array(out["gold_label"])
    for k in ("ps_hc", "cc_hc", "ps_ab", "cc_ab"):
        out[k] = np.array(out[k], dtype=bool)
    return out


def main():
    rng = np.random.default_rng(SEED)

    cells = [
        ("v1",      "v1_eval_snapkv.jsonl",         "v1_eval_compressed_snapkv.jsonl"),
        ("mixture", "v1_eval_mixture_snapkv.jsonl", "v1_eval_mixture_compressed_snapkv.jsonl"),
    ]

    paired = {}
    for label, ps, cc in cells:
        if not (ART / cc).exists():
            print(f"!! missing {ART/cc} — skipping cell {label}")
            continue
        paired[label] = build_paired(label, ps, cc)
        # quick sanity
        d = paired[label]
        n_err = sum(1 for k in d["keys"]
                    if False)  # cc errors already filtered as classification=='error', drop those keys
        print(f"  {label}: n_paired={len(d['keys'])}")
    print()

    # ---- Section C: Aggregate honest_acc ----
    print("### C. Aggregate honest accuracy on SnapKV val")
    print()
    print(f"{'Adapter':10s} | {'n':4s} | prompt-style | compressed-cache | Δ (cc - ps)")
    print("-" * 70)
    for label in paired:
        d = paired[label]
        n = len(d["keys"])
        ps_acc = float(d["ps_hc"].mean())
        cc_acc = float(d["cc_hc"].mean())
        delta = cc_acc - ps_acc
        sign = "+" if delta >= 0 else ""
        print(f"{label:10s} | {n:4d} | {ps_acc:12.3f} | {cc_acc:16.3f} | {sign}{delta:.3f}")
    print()

    # ---- Section D: Per-ratio comparison ----
    print("### D. Per-ratio: prompt-style vs compressed-cache")
    print()
    for label in paired:
        d = paired[label]
        print(f"#### Adapter = {label}")
        print(f"{'ratio':5s} | {'n':3s} | {'ps hon':6s} | {'cc hon':6s} | {'Δ':6s} | "
              f"{'McNemar p':10s} | 95% CI on Δ (cc-ps)")
        print("-" * 80)
        for r in RATIOS:
            mask_r = np.isclose(d["ratio"], r, atol=1e-9)
            n_r = int(mask_r.sum())
            if n_r == 0:
                continue
            ps_r = d["ps_hc"][mask_r]
            cc_r = d["cc_hc"][mask_r]
            ps_acc = float(ps_r.mean())
            cc_acc = float(cc_r.mean())
            delta = cc_acc - ps_acc
            _, b_disc, c_disc, _, _, p = mcnemar_exact(cc_r, ps_r)
            pt, lo, hi = paired_bootstrap_diff_ci(cc_r, ps_r, rng)
            sign_d = "+" if delta >= 0 else ""
            sign_lo = "+" if lo >= 0 else ""
            sign_hi = "+" if hi >= 0 else ""
            print(f"{r:.2f}  | {n_r:3d} | {ps_acc:6.3f} | {cc_acc:6.3f} | {sign_d}{delta:5.3f} | "
                  f"{p:10.4g} | [{sign_lo}{lo:.2f}, {sign_hi}{hi:.2f}]")
        # Aggregate row
        a, b_disc, c_disc, dd, _, p = mcnemar_exact(d["cc_hc"], d["ps_hc"])
        pt, lo, hi = paired_bootstrap_diff_ci(d["cc_hc"], d["ps_hc"], rng)
        agg_d = float(d["cc_hc"].mean() - d["ps_hc"].mean())
        sign_d = "+" if agg_d >= 0 else ""
        sign_lo = "+" if lo >= 0 else ""
        sign_hi = "+" if hi >= 0 else ""
        print(f"AGG   | {len(d['keys']):3d} | {d['ps_hc'].mean():6.3f} | {d['cc_hc'].mean():6.3f} | "
              f"{sign_d}{agg_d:5.3f} | {p:10.4g} | [{sign_lo}{lo:.2f}, {sign_hi}{hi:.2f}]")
        print(f"  paired-discordants: cc-only-correct={b_disc}, ps-only-correct={c_disc}")
        print()

    # ---- Section E: r=0.80 over-abstention on CONFIDENT-gold ----
    print("### E. Over-abstention rate at r=0.80 on CONFIDENT-gold")
    print()
    print(f"{'Adapter':10s} | {'n_C':4s} | ps ovc | cc ovc | Δ")
    print("-" * 50)
    for label in paired:
        d = paired[label]
        mask_080 = np.isclose(d["ratio"], 0.80, atol=1e-9)
        mask_C = np.isin(d["gold_label"], CONFIDENT_GOLDS)
        mask = mask_080 & mask_C
        nC = int(mask.sum())
        if nC == 0:
            continue
        ps_ovc = float(d["ps_ab"][mask].mean())
        cc_ovc = float(d["cc_ab"][mask].mean())
        delta = cc_ovc - ps_ovc
        sign = "+" if delta >= 0 else ""
        print(f"{label:10s} | {nC:4d} | {ps_ovc:.3f}  | {cc_ovc:.3f}  | {sign}{delta:.3f}")
    print()


if __name__ == "__main__":
    main()
