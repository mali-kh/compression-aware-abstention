"""Block 5 pivot comparison — random-mask adapter vs base/v1/mixture/P2.

Aggregate honest_acc + r=0.50 honest_acc + r=0.80 over-abstention (CONFIDENT-gold)
+ paired McNemar random-vs-mixture aggregate. Pilot scope: KVzip + SnapKV-ctx only.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from block1_stats import (
    ART, RATIO, N_BOOT, SEED,
    honest_correct, load_jsonl, at_ratio, index_by_id, mcnemar_exact,
)

CONFIDENT_GOLDS = ("confident", "confident_answer")

# (name, v1_file, mix_file, p2_file, random_file)
PILOT_COMPRESSORS = [
    ("KVzip",
     "v1_eval.jsonl",
     "v1_eval_mixture_kvzip.jsonl",
     "v1_eval_prompt_p2_kvzip.jsonl",
     "v1_eval_random_kvzip.jsonl"),
    ("SnapKV-ctx",
     "v1_eval_snapkv.jsonl",
     "v1_eval_mixture_snapkv.jsonl",
     "v1_eval_prompt_p2_snapkv.jsonl",
     "v1_eval_random_snapkv.jsonl"),
]


def build(name, v1f, mxf, p2f, rndf, ratio_filter=None):
    """Return per-(joined-id) arrays for all five conditions.
    ratio_filter=None means use all rows; pass a float to filter.
    """
    v1 = load_jsonl(ART / v1f)
    mx = load_jsonl(ART / mxf)
    p2 = load_jsonl(ART / p2f)
    rnd = load_jsonl(ART / rndf)
    if ratio_filter is not None:
        v1 = at_ratio(v1, r=ratio_filter)
        mx = at_ratio(mx, r=ratio_filter)
        p2 = at_ratio(p2, r=ratio_filter)
        rnd = at_ratio(rnd, r=ratio_filter)

    v1_ix = index_by_id(v1)
    mx_ix = index_by_id(mx)
    p2_ix = index_by_id(p2)
    rnd_ix = index_by_id(rnd)
    # For aggregate (no ratio filter), join on (id, ratio) instead of id alone.
    def keyer(rows, indexed):
        if ratio_filter is None:
            return {(r["id"], r["ratio"]): r for r in rows}
        return indexed
    v1_k = keyer(v1, v1_ix)
    mx_k = keyer(mx, mx_ix)
    p2_k = keyer(p2, p2_ix)
    rnd_k = keyer(rnd, rnd_ix)

    keys = sorted(set(v1_k) & set(mx_k) & set(p2_k) & set(rnd_k))

    out = {"keys": keys, "gold_label": [],
           "base_hc": [], "v1_hc": [], "mix_hc": [], "p2_hc": [], "rnd_hc": [],
           "base_ab": [], "v1_ab": [], "mix_ab": [], "p2_ab": [], "rnd_ab": []}
    for k in keys:
        v1r, mxr, p2r, rndr = v1_k[k], mx_k[k], p2_k[k], rnd_k[k]
        gl = v1r["gold_label"]
        out["gold_label"].append(gl)
        b = v1r["base"]; a = v1r["adapter"]; am = mxr["adapter"]; ar = rndr["adapter"]
        out["base_hc"].append(honest_correct(gl, b["classification"], b["model_abstained"]))
        out["v1_hc"].append(honest_correct(gl, a["classification"], a["model_abstained"]))
        out["mix_hc"].append(honest_correct(gl, am["classification"], am["model_abstained"]))
        out["p2_hc"].append(honest_correct(gl, p2r["classification"], p2r["model_abstained"]))
        out["rnd_hc"].append(honest_correct(gl, ar["classification"], ar["model_abstained"]))
        out["base_ab"].append(b["model_abstained"])
        out["v1_ab"].append(a["model_abstained"])
        out["mix_ab"].append(am["model_abstained"])
        out["p2_ab"].append(p2r["model_abstained"])
        out["rnd_ab"].append(ar["model_abstained"])
    out["gold_label"] = np.array(out["gold_label"])
    for k in ("base_hc", "v1_hc", "mix_hc", "p2_hc", "rnd_hc",
              "base_ab", "v1_ab", "mix_ab", "p2_ab", "rnd_ab"):
        out[k] = np.array(out[k], dtype=bool)
    return out


def fmt_pct(x, n=None):
    if n is None:
        return f"{x:.3f}"
    return f"{x:.3f} ({int(round(x*n))}/{n})"


def main():
    print("### Block 5 pivot — sanity counts (joined per compressor)")
    print()
    agg_data = {}
    r05_data = {}
    r08_data = {}
    for (name, v1f, mxf, p2f, rndf) in PILOT_COMPRESSORS:
        agg = build(name, v1f, mxf, p2f, rndf, ratio_filter=None)
        r05 = build(name, v1f, mxf, p2f, rndf, ratio_filter=0.5)
        r08 = build(name, v1f, mxf, p2f, rndf, ratio_filter=0.8)
        agg_data[name] = agg
        r05_data[name] = r05
        r08_data[name] = r08
        print(f"  {name:14s}  agg n={len(agg['keys'])}   r=0.5 n={len(r05['keys'])}   r=0.8 n={len(r08['keys'])}")
    print()

    # ---- Aggregate honest accuracy ----
    print("### Aggregate honest accuracy")
    print()
    print(f"{'Compressor':14s} | n   | base  | v1    | mixture | random")
    print("-" * 60)
    for name in agg_data:
        d = agg_data[name]
        n = len(d["keys"])
        print(f"{name:14s} | {n:3d} | {d['base_hc'].mean():.3f} | {d['v1_hc'].mean():.3f} | {d['mix_hc'].mean():.3f}   | {d['rnd_hc'].mean():.3f}")
    print()

    # ---- At r=0.50 ----
    print("### At r=0.50")
    print()
    print(f"{'Compressor':14s} | n  | base  | v1    | mixture | random | P2")
    print("-" * 70)
    for name in r05_data:
        d = r05_data[name]
        n = len(d["keys"])
        print(f"{name:14s} | {n:2d} | {d['base_hc'].mean():.3f} | {d['v1_hc'].mean():.3f} | {d['mix_hc'].mean():.3f}   | {d['rnd_hc'].mean():.3f}  | {d['p2_hc'].mean():.3f}")
    print()

    # ---- At r=0.80, over-abstention on CONFIDENT-gold ----
    print("### At r=0.80, over-abstention rate on CONFIDENT-gold")
    print()
    print(f"{'Compressor':14s} | n_C | base  | v1    | mixture | random | P2")
    print("-" * 70)
    for name in r08_data:
        d = r08_data[name]
        gl = d["gold_label"]
        mask_C = np.isin(gl, CONFIDENT_GOLDS)
        nC = int(mask_C.sum())
        if nC == 0:
            continue
        ovc_b = d["base_ab"][mask_C].mean()
        ovc_v = d["v1_ab"][mask_C].mean()
        ovc_m = d["mix_ab"][mask_C].mean()
        ovc_r = d["rnd_ab"][mask_C].mean()
        ovc_p = d["p2_ab"][mask_C].mean()
        print(f"{name:14s} | {nC:3d} | {ovc_b:.3f} | {ovc_v:.3f} | {ovc_m:.3f}   | {ovc_r:.3f}  | {ovc_p:.3f}")
    print()

    # ---- Random vs mixture paired McNemar (aggregate) ----
    print("### Random vs mixture paired McNemar (aggregate, paired by (id, ratio))")
    print()
    print(f"{'Compressor':14s} | n   | b (rnd>mix) | c (mix>rnd) | exact p   | Δ rnd−mix")
    print("-" * 75)
    mcnemar_rows = []
    for name in agg_data:
        d = agg_data[name]
        n = len(d["keys"])
        a, b_disc, c_disc, dd, stat, p = mcnemar_exact(d["rnd_hc"], d["mix_hc"])
        delta = float(d["rnd_hc"].mean() - d["mix_hc"].mean())
        sign = "+" if delta >= 0 else ""
        print(f"{name:14s} | {n:3d} | {b_disc:11d} | {c_disc:11d} | {p:9.4g} | {sign}{delta:.3f}")
        mcnemar_rows.append((name, n, b_disc, c_disc, p, delta))
    print()

    # ---- Random vs v1 paired McNemar (aggregate) — useful for the bucket call ----
    print("### Random vs v1 paired McNemar (aggregate, supplementary)")
    print()
    print(f"{'Compressor':14s} | n   | b (rnd>v1)  | c (v1>rnd)  | exact p   | Δ rnd−v1")
    print("-" * 75)
    for name in agg_data:
        d = agg_data[name]
        n = len(d["keys"])
        a, b_disc, c_disc, dd, stat, p = mcnemar_exact(d["rnd_hc"], d["v1_hc"])
        delta = float(d["rnd_hc"].mean() - d["v1_hc"].mean())
        sign = "+" if delta >= 0 else ""
        print(f"{name:14s} | {n:3d} | {b_disc:11d} | {c_disc:11d} | {p:9.4g} | {sign}{delta:.3f}")
    print()


if __name__ == "__main__":
    main()
