"""Block 3.5 pivot — cc-trained adapter under KVzip cc inference.

Outputs (deliverables C / D / E / F per the spec):
  C. Aggregate honest accuracy on val_v1.jsonl (KVzip val, n=463):
        v1 prompt-style, mixture prompt-style, v1 KVzip cc, mixture
        KVzip cc, NEW (cc-trained) KVzip cc.
  D. Per-ratio table for the new adapter under KVzip cc inference.
  E. Headline McNemar: NEW (cc-trained) vs mixture (ps-trained), both
     under KVzip cc, at r=0.80 on CONFIDENT-gold.
  F. Bucket call (3 sentences max).

Inputs:
  artifacts/v1_eval.jsonl                              (paired base/adapter prompt-style; v1 LoRA)
  artifacts/v1_eval_mixture_seed42.jsonl               (paired base/adapter prompt-style; mixture LoRA)
  artifacts/v1_eval_compressed_kvzip.jsonl             (v1 LoRA under KVzip cc)
  artifacts/v1_eval_mixture_compressed_kvzip.jsonl     (mixture LoRA under KVzip cc)
  artifacts/v1_eval_compressed_trained_kvzip.jsonl     (NEW cc-trained LoRA under KVzip cc)

Run from .venv-kvpress (the .py uses numpy + scipy from block1_stats only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from block1_stats import (  # noqa: E402
    ART, SEED,
    honest_correct, load_jsonl,
    paired_bootstrap_diff_ci, mcnemar_exact,
)

CONFIDENT_GOLDS = ("confident", "confident_answer")
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]

PS_FILES = {
    "v1": "v1_eval.jsonl",
    "mixture": "v1_eval_mixture_seed42.jsonl",
}
CC_FILES = {
    "v1":              "v1_eval_compressed_kvzip.jsonl",
    "mixture":         "v1_eval_mixture_compressed_kvzip.jsonl",
    "NEW (cc-train)":  "v1_eval_compressed_trained_kvzip.jsonl",
}


def index_by_key(records, key=("id", "ratio")):
    return {tuple(r[k] for k in key): r for r in records}


def load_ps_adapter(label):
    """Load paired adapter rows (only adapter block, classified)."""
    rows = load_jsonl(ART / PS_FILES[label])
    out = []
    for r in rows:
        if r.get("adapter") is None:
            continue
        a = r["adapter"]
        out.append({
            "id": r["id"], "ratio": r["ratio"],
            "gold_label": r["gold_label"],
            "classification": a["classification"],
            "model_abstained": a["model_abstained"],
        })
    return out


def load_cc(label):
    rows = load_jsonl(ART / CC_FILES[label])
    return [r for r in rows if r.get("classification") != "error"]


def metric_arrays(rows):
    n = len(rows)
    ratio = np.array([r["ratio"] for r in rows])
    gold = np.array([r["gold_label"] for r in rows])
    hc = np.array([
        honest_correct(r["gold_label"], r["classification"], r["model_abstained"])
        for r in rows
    ], dtype=bool)
    ab = np.array([r["model_abstained"] for r in rows], dtype=bool)
    return n, ratio, gold, hc, ab


def section_C():
    print("### C. Aggregate honest accuracy on val_v1.jsonl  (KVzip val, n=463)\n")
    print("| Adapter        | Inference path  |   n   | Aggregate hon | Δhon vs base/v1-ps |")
    print("| -------------- | --------------- | ----- | ------------- | ------------------ |")
    # Reference: v1 prompt-style aggregate (use existing PS file)
    ps_v1 = load_ps_adapter("v1")
    n_v1, _, _, hc_v1_ps, _ = metric_arrays(ps_v1)
    base_acc = None  # base eval is in v1_eval.jsonl 'base' block
    base_rows = load_jsonl(ART / PS_FILES["v1"])
    base_hc = []
    for r in base_rows:
        if r.get("base") is None:
            continue
        b = r["base"]
        base_hc.append(honest_correct(r["gold_label"], b["classification"], b["model_abstained"]))
    base_acc = float(np.mean(base_hc)) if base_hc else float("nan")
    print(f"| base           | prompt-style    | {len(base_hc):4d} | {base_acc:.3f}         |   —                |")

    for label, rows in [("v1", ps_v1), ("mixture", load_ps_adapter("mixture"))]:
        n, _, _, hc, _ = metric_arrays(rows)
        agg = float(hc.mean())
        d = agg - base_acc
        sign = "+" if d >= 0 else ""
        print(f"| {label:14s} | prompt-style    | {n:4d}  | {agg:.3f}         | {sign}{d:.3f}             |")

    for label in ("v1", "mixture", "NEW (cc-train)"):
        path = ART / CC_FILES[label]
        if not path.exists():
            print(f"| {label:14s} | KVzip cc        |  ?   | (missing {path.name})        |   ?                |")
            continue
        rows = load_cc(label)
        n, _, _, hc, _ = metric_arrays(rows)
        agg = float(hc.mean())
        d = agg - base_acc
        sign = "+" if d >= 0 else ""
        print(f"| {label:14s} | KVzip cc        | {n:4d}  | {agg:.3f}         | {sign}{d:.3f}             |")
    print()


def section_D():
    print("### D. Per-ratio table for the NEW (cc-trained) adapter under KVzip cc\n")
    path = ART / CC_FILES["NEW (cc-train)"]
    if not path.exists():
        print(f"   (missing {path.name})\n")
        return
    rows = load_cc("NEW (cc-train)")
    n_total, ratio, gold, hc, ab = metric_arrays(rows)

    # Hallucination = wrong answer on confident gold (gave specific wrong answer);
    # Wrong = wrong answer on abstain gold (made something up).
    cls_arr = np.array([r["classification"] for r in rows])

    print(f"| ratio | n   | hon | abst | hal | wrong | over-abstain on C-gold |")
    print(f"| ----- | --- | --- | ---- | --- | ----- | ---------------------- |")
    for r in RATIOS:
        m = np.isclose(ratio, r, atol=1e-9)
        n = int(m.sum())
        if n == 0:
            print(f"| {r:.2f} | 0   | —   | —    | —   | —     | —                      |")
            continue
        hon_r = float(hc[m].mean())
        ab_r = float(ab[m].mean())
        # halluc = confident_gold rows where the model gave a wrong specific answer (not abstaining and not honest_correct)
        m_C = m & np.isin(gold, CONFIDENT_GOLDS) & (~ab) & (~hc)
        m_A = m & (gold == "abstain") & (~ab) & (~hc)
        hal_r = int(m_C.sum())
        wrong_r = int(m_A.sum())
        m_C_all = m & np.isin(gold, CONFIDENT_GOLDS)
        nC = int(m_C_all.sum())
        if nC > 0:
            ovc = float(ab[m_C_all].mean())
            ovc_str = f"{ovc:.3f} (n_C={nC})"
        else:
            ovc_str = "n/a (no C-gold)"
        print(f"| {r:.2f} | {n:3d} | {hon_r:.3f} | {ab_r:.3f} | {hal_r:3d} | {wrong_r:5d} | {ovc_str:22s} |")
    print()


def section_E():
    print("### E. Headline McNemar — NEW (cc-trained) vs mixture (ps-trained), both at r=0.80 on CONFIDENT-gold under KVzip cc\n")
    path_new = ART / CC_FILES["NEW (cc-train)"]
    path_mix = ART / CC_FILES["mixture"]
    if not (path_new.exists() and path_mix.exists()):
        print(f"   (missing {path_new.name} or {path_mix.name})\n")
        return
    rows_new = load_cc("NEW (cc-train)")
    rows_mix = load_cc("mixture")
    ix_new = index_by_key(rows_new)
    ix_mix = index_by_key(rows_mix)
    keys = sorted(set(ix_new) & set(ix_mix))

    n_new_ab = []
    n_mix_ab = []
    n_new_hc = []
    n_mix_hc = []
    for k in keys:
        rn = ix_new[k]
        rm = ix_mix[k]
        if not np.isclose(rn["ratio"], 0.80):
            continue
        if rn["gold_label"] not in CONFIDENT_GOLDS:
            continue
        n_new_ab.append(rn["model_abstained"])
        n_mix_ab.append(rm["model_abstained"])
        n_new_hc.append(honest_correct(rn["gold_label"], rn["classification"], rn["model_abstained"]))
        n_mix_hc.append(honest_correct(rm["gold_label"], rm["classification"], rm["model_abstained"]))

    arr_new_hc = np.array(n_new_hc, dtype=bool)
    arr_mix_hc = np.array(n_mix_hc, dtype=bool)
    a, b, c, _, _, p = mcnemar_exact(arr_new_hc, arr_mix_hc)
    rng = np.random.default_rng(SEED)
    pt, lo, hi = paired_bootstrap_diff_ci(arr_new_hc, arr_mix_hc, rng)
    delta = float(arr_new_hc.mean() - arr_mix_hc.mean())
    sign = "+" if delta >= 0 else ""
    sign_lo = "+" if lo >= 0 else ""
    sign_hi = "+" if hi >= 0 else ""
    print(f"  n_C = {len(arr_new_hc)}")
    print(f"  b (NEW correct, mixture wrong) = {b}")
    print(f"  c (mixture correct, NEW wrong) = {c}")
    print(f"  exact p = {p:.4g}")
    print(f"  Δ = {sign}{delta:.3f}  (NEW − mixture)")
    print(f"  95% CI on Δ: [{sign_lo}{lo:.3f}, {sign_hi}{hi:.3f}]\n")

    print(f"  over-abstention rates on C-gold at r=0.80:")
    print(f"    NEW (cc-train) :  {np.mean(n_new_ab):.3f}")
    print(f"    mixture (ps)   :  {np.mean(n_mix_ab):.3f}")
    print()


def section_F():
    print("### F. Bucket call\n")
    path_new = ART / CC_FILES["NEW (cc-train)"]
    path_mix = ART / CC_FILES["mixture"]
    if not (path_new.exists() and path_mix.exists()):
        print("   (missing inputs — re-run after Step 3 + Step 4a complete)\n")
        return
    # Compute the three call-out numbers
    rows_new = load_cc("NEW (cc-train)")
    rows_mix = load_cc("mixture")
    ix_new = index_by_key(rows_new)
    ix_mix = index_by_key(rows_mix)
    # over-abstention at r=0.80 on C-gold
    new_ab080_C = []
    mix_ab080_C = []
    for k in sorted(set(ix_new) & set(ix_mix)):
        rn = ix_new[k]
        rm = ix_mix[k]
        if not np.isclose(rn["ratio"], 0.80):
            continue
        if rn["gold_label"] not in CONFIDENT_GOLDS:
            continue
        new_ab080_C.append(rn["model_abstained"])
        mix_ab080_C.append(rm["model_abstained"])
    new_ab_rate = np.mean(new_ab080_C) if new_ab080_C else float("nan")
    mix_ab_rate = np.mean(mix_ab080_C) if mix_ab080_C else float("nan")

    # aggregate honest acc under cc inference for new vs v1 prompt-style baseline
    new_agg = float(np.mean([
        honest_correct(r["gold_label"], r["classification"], r["model_abstained"])
        for r in rows_new
    ]))
    ps_v1 = load_ps_adapter("v1")
    v1_ps_agg = float(np.mean([
        honest_correct(r["gold_label"], r["classification"], r["model_abstained"])
        for r in ps_v1
    ]))

    print(f"  - over-abstention drop at r=0.80 on C-gold (cc inference): "
          f"mixture (ps-trained) = {mix_ab_rate:.3f} → NEW (cc-trained) = {new_ab_rate:.3f}  "
          f"(Δ = {new_ab_rate-mix_ab_rate:+.3f})")
    print(f"  - aggregate honest acc under cc inference: NEW = {new_agg:.3f}  "
          f"(v1 prompt-style reference = {v1_ps_agg:.3f}; "
          f"gap = {new_agg-v1_ps_agg:+.3f})")
    print(f"  - manual inspection required — fill in surprises after looking at "
          f"per-ratio breakdown above (low-retention failure modes? training "
          f"convergence pattern?).")
    print()


def main():
    print("# Block 3.5 pivot — cc-trained adapter under KVzip cc inference\n")
    section_C()
    section_D()
    section_E()
    section_F()


if __name__ == "__main__":
    main()
