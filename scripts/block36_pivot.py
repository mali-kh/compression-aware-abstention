"""Block 3.6 pivot — mixture-cc adapter vs prior baselines under KVzip cc inference.

Deliverables C/D/E/F per the spec:
  C. Per-ratio honest accuracy: base / mixture-ps / NEW / mixture-cc, all
     under KVzip cc inference on val_v1.jsonl (n=463).
  D. Over-abstention + hallucination tables at r=0.80 on CONFIDENT-gold.
  E. McNemar at r=0.80 on CONFIDENT-gold:
       - mixture-cc vs mixture-ps (cc inference both)
       - mixture-cc vs NEW (cc-trained, KVzip-only, Block 3.5)
  F. Qualitative bucket A/B/C/D on mixture-cc r=0.80 wrong_answer set.
     Compares directly to NEW's distribution (A=13, B=25, C=3, D=1).

Run via /usr/bin/python3 (system) — needs scipy. Tokenizer-only call uses
the kvpress venv's HF cache through HF_HUB_OFFLINE.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from block1_stats import (  # noqa: E402
    ART, SEED,
    honest_correct, load_jsonl,
    paired_bootstrap_diff_ci, mcnemar_exact,
)

CONFIDENT_GOLDS = ("confident", "confident_answer")
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]

# Inference results files (all under KVzip cc inference path, val_v1.jsonl)
CC_FILES = {
    "v1": "v1_eval_compressed_kvzip.jsonl",
    "mixture-ps": "v1_eval_mixture_compressed_kvzip.jsonl",
    "NEW": "v1_eval_compressed_trained_kvzip.jsonl",
    "mixture-cc": "v1_eval_mixture_cc_kvzip.jsonl",
}
PS_FILE = "v1_eval.jsonl"  # for base-prompt-style row only


def index_by_key(records, key=("id", "ratio")):
    return {tuple(r[k] for k in key): r for r in records}


def metric_rows(rows):
    """Return dict with arrays for ratio, gold, hc, ab, classification."""
    n = len(rows)
    ratio = np.array([r["ratio"] for r in rows])
    gold = np.array([r["gold_label"] for r in rows])
    hc = np.array([
        honest_correct(r["gold_label"], r["classification"], r["model_abstained"])
        for r in rows
    ], dtype=bool)
    ab = np.array([r["model_abstained"] for r in rows], dtype=bool)
    cls = np.array([r["classification"] for r in rows])
    return {"n": n, "ratio": ratio, "gold": gold, "hc": hc, "ab": ab, "cls": cls}


# ---------- Section C: per-ratio honest accuracy ----------


def section_C():
    print("### C. Per-ratio honest accuracy on val_v1.jsonl (KVzip cc inference)\n")

    # base prompt-style baseline (cc only adapters; base under cc is part of v1_eval)
    base_rows = load_jsonl(ART / PS_FILE)
    base_records = []
    for r in base_rows:
        if r.get("base"):
            base_records.append({
                "id": r["id"], "ratio": r["ratio"],
                "gold_label": r["gold_label"],
                "classification": r["base"]["classification"],
                "model_abstained": r["base"]["model_abstained"],
            })
    base_m = metric_rows(base_records)

    cc_data = {}
    for label, fp in CC_FILES.items():
        path = ART / fp
        if not path.exists():
            print(f"  (missing {fp})\n")
            continue
        rows = [r for r in load_jsonl(path) if r["classification"] != "error"]
        cc_data[label] = metric_rows(rows)

    print("```")
    header = f"{'ratio':<5s} | {'n':>3s} | "
    header += " | ".join(f"{lbl:>10s}" for lbl in
                          ["base (ps)", "v1 (cc)", "mixture-ps cc", "NEW cc", "mixture-cc cc"])
    print(header)
    print("-" * len(header))
    for r in RATIOS:
        cells = []
        # base prompt-style at this ratio
        m = np.isclose(base_m["ratio"], r, atol=1e-9)
        n_r = int(m.sum())
        if n_r == 0:
            cells.append(f"{'—':>10s}")
        else:
            cells.append(f"{base_m['hc'][m].mean():>10.3f}")
        # cc adapters
        for lbl in ("v1", "mixture-ps", "NEW", "mixture-cc"):
            d = cc_data.get(lbl)
            if d is None:
                cells.append(f"{'—':>10s}")
                continue
            m = np.isclose(d["ratio"], r, atol=1e-9)
            n_r = int(m.sum())
            if n_r == 0:
                cells.append(f"{'—':>10s}")
            else:
                cells.append(f"{d['hc'][m].mean():>10.3f}")
        # n is from val record set; use any cc data
        ref = cc_data.get("mixture-cc") or list(cc_data.values())[0]
        ref_n = int(np.isclose(ref["ratio"], r, atol=1e-9).sum())
        print(f"{r:<5.2f} | {ref_n:>3d} | {' | '.join(cells)}")
    # AGG row
    cells = [f"{base_m['hc'].mean():>10.3f}"]
    for lbl in ("v1", "mixture-ps", "NEW", "mixture-cc"):
        d = cc_data.get(lbl)
        if d is None:
            cells.append(f"{'—':>10s}")
        else:
            cells.append(f"{d['hc'].mean():>10.3f}")
    print(f"{'AGG':<5s} | {len(base_m['hc']):>3d} | {' | '.join(cells)}")
    print("```\n")
    return cc_data, base_m


# ---------- Section D: r=0.80 C-gold over-abstention + hallucination ----------


def section_D(cc_data):
    print("### D. r=0.80 on CONFIDENT-gold under KVzip cc inference\n")
    print("Over-abstention rate (model abstains while gold is confident_answer):\n")
    print("```")
    print(f"{'adapter':<14s} | {'n_C':>4s} | over-abst rate | 95% CI (paired bootstrap on subset)")
    print("-" * 90)
    rng = np.random.default_rng(SEED)
    for lbl in ("mixture-ps", "NEW", "mixture-cc"):
        d = cc_data.get(lbl)
        if d is None:
            continue
        m = np.isclose(d["ratio"], 0.80) & np.isin(d["gold"], CONFIDENT_GOLDS)
        nC = int(m.sum())
        if nC == 0:
            continue
        rate = float(d["ab"][m].mean())
        # bootstrap CI on the proportion (single-vec)
        from block1_stats import bootstrap_ci
        rate_pt, lo, hi = bootstrap_ci(d["ab"][m].astype(int), rng)
        print(f"{lbl:<14s} | {nC:>4d} | {rate:>14.3f} | [{lo:.3f}, {hi:.3f}]")
    print("```\n")

    print("Hallucination count (gold=confident_answer, classification=wrong_answer):\n")
    print("```")
    print(f"{'adapter':<14s} | {'n_C':>4s} | hallucinations | as fraction")
    print("-" * 60)
    for lbl in ("mixture-ps", "NEW", "mixture-cc"):
        d = cc_data.get(lbl)
        if d is None:
            continue
        m = np.isclose(d["ratio"], 0.80) & np.isin(d["gold"], CONFIDENT_GOLDS)
        nC = int(m.sum())
        if nC == 0:
            continue
        n_hal = int(((d["cls"] == "wrong_answer") & m).sum())
        print(f"{lbl:<14s} | {nC:>4d} | {n_hal:>14d} | {n_hal/nC:.3f}")
    print("```\n")


# ---------- Section E: McNemar comparisons ----------


def section_E(cc_data):
    print("### E. Paired McNemar at r=0.80 on CONFIDENT-gold under KVzip cc\n")
    rng = np.random.default_rng(SEED)
    for label_a, label_b in [("mixture-cc", "mixture-ps"), ("mixture-cc", "NEW")]:
        d_a = cc_data.get(label_a)
        d_b = cc_data.get(label_b)
        if d_a is None or d_b is None:
            continue
        # Build per-key paired arrays
        rows_a = [{"id": "?", "ratio": d_a["ratio"][i], "gold": d_a["gold"][i],
                   "hc": d_a["hc"][i], "ab": d_a["ab"][i], "cls": d_a["cls"][i]}
                  for i in range(d_a["n"])]
        rows_b = [{"id": "?", "ratio": d_b["ratio"][i], "gold": d_b["gold"][i],
                   "hc": d_b["hc"][i], "ab": d_b["ab"][i], "cls": d_b["cls"][i]}
                  for i in range(d_b["n"])]
        # Re-pair using the original eval files for proper id matching
        f_a = ART / CC_FILES[label_a]
        f_b = ART / CC_FILES[label_b]
        ra = [r for r in load_jsonl(f_a) if r["classification"] != "error"]
        rb = [r for r in load_jsonl(f_b) if r["classification"] != "error"]
        ix_a = index_by_key(ra)
        ix_b = index_by_key(rb)
        keys = sorted(set(ix_a) & set(ix_b))
        a_hc, b_hc, a_ab, b_ab = [], [], [], []
        for k in keys:
            ra_, rb_ = ix_a[k], ix_b[k]
            if not np.isclose(ra_["ratio"], 0.80):
                continue
            if ra_["gold_label"] not in CONFIDENT_GOLDS:
                continue
            a_hc.append(honest_correct(ra_["gold_label"], ra_["classification"], ra_["model_abstained"]))
            b_hc.append(honest_correct(rb_["gold_label"], rb_["classification"], rb_["model_abstained"]))
            a_ab.append(ra_["model_abstained"])
            b_ab.append(rb_["model_abstained"])
        a_hc = np.array(a_hc, dtype=bool)
        b_hc = np.array(b_hc, dtype=bool)
        a_ab = np.array(a_ab, dtype=bool)
        b_ab = np.array(b_ab, dtype=bool)

        a_, b_, c_, _, _, p = mcnemar_exact(a_hc, b_hc)
        pt, lo, hi = paired_bootstrap_diff_ci(a_hc, b_hc, rng)
        delta = float(a_hc.mean() - b_hc.mean())
        sd = "+" if delta >= 0 else ""
        slo = "+" if lo >= 0 else ""
        shi = "+" if hi >= 0 else ""
        print(f"#### {label_a} vs {label_b}\n")
        print(f"  n_C = {len(a_hc)}")
        print(f"  b ({label_a} correct, {label_b} wrong) = {b_}")
        print(f"  c ({label_b} correct, {label_a} wrong) = {c_}")
        print(f"  exact p = {p:.4g}")
        print(f"  Δ on honest-correct = {sd}{delta:.3f}  ({label_a} − {label_b})")
        print(f"  95% CI on Δ (paired bootstrap) = [{slo}{lo:.3f}, {shi}{hi:.3f}]")
        print(f"  over-abstain on C-gold:  {label_a}={a_ab.mean():.3f}  vs  {label_b}={b_ab.mean():.3f}")
        print()


# ---------- Section F: qualitative bucketing on mixture-cc r=0.80 wrong_answer ----------


def section_F():
    print("### F. Qualitative bucket distribution on mixture-cc r=0.80 hallucinations\n")
    # Use the same approach as scripts/block35_qual_r080_halluc.py but for mixture-cc.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    eval_rows = load_jsonl(ART / "v1_eval_mixture_cc_kvzip.jsonl")
    val_records = {(r["id"], r["ratio"]): r for r in load_jsonl(ART / "val_v1.jsonl")}

    hal = [r for r in eval_rows
           if abs(r["ratio"] - 0.80) < 1e-9
           and r["gold_label"] == "confident_answer"
           and r["classification"] == "wrong_answer"]
    print(f"  n_hallucinations on mixture-cc at r=0.80 (C-gold) = {len(hal)}\n")

    if not hal:
        print("  (no hallucinations at r=0.80 to bucket; mixture-cc avoided this failure entirely)")
        return

    # Need a tokenizer for kept_text reconstruction.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"  (tokenizer load failed: {type(e).__name__}: {e})")
        print(f"  Falling back to skipping kept-vs-evicted distinction; reporting "
              f"raw response/gold for all {len(hal)} cases.")
        for h in hal[:10]:
            print(f"    id={h['id']}  gold={h['gold_answer']!r}  resp={h['response'][:120]!r}")
        return

    def first_chunk(s):
        s = s.strip()
        if len(s.split()) <= 8:
            return s
        return re.split(r"[.!?\n:]+", s, maxsplit=1)[0].strip()

    def has(needle, hay):
        if not needle.strip():
            return False
        return needle.lower() in hay.lower()

    A = []; B = []; C = []; D = []
    skipped = 0
    for h in hal:
        v = val_records.get((h["id"], h["ratio"]))
        if v is None:
            skipped += 1
            continue
        ctx_full = v["context"]
        ctx_ids = tok(ctx_full, add_special_tokens=False).input_ids
        if len(ctx_ids) != len(v["kept_mask"]):
            skipped += 1
            continue
        kept_ids = [t for t, k in zip(ctx_ids, v["kept_mask"]) if k]
        kept_text = tok.decode(kept_ids, skip_special_tokens=True)
        gold = h["gold_answer"]
        resp = h["response"]
        rc = first_chunk(resp)
        gold_in_kept = has(gold, kept_text)
        resp_in_kept = has(rc, kept_text)
        rec = {"id": h["id"], "gold": gold, "resp": resp[:140], "rc": rc[:120]}
        if gold_in_kept:
            (A if resp_in_kept else B).append(rec)
        else:
            (C if resp_in_kept else D).append(rec)

    n_total = len(A) + len(B) + len(C) + len(D)
    print(f"  Bucketed {n_total}/{len(hal)} (skipped {skipped})\n")
    print(f"  | Bucket | Description                                | mixture-cc | Block 3.5 NEW |")
    print(f"  | ------ | ------------------------------------------ | ---------- | ------------- |")
    print(f"  |   A    | gold IN kept, response IN kept (distractor)| {len(A):>10d} | {13:>13d} |")
    print(f"  |   B    | gold IN kept, response NOT in full (fab)   | {len(B):>10d} | {25:>13d} |")
    print(f"  |   C    | gold EVICTED, response IN kept (substitute)| {len(C):>10d} | {3:>13d} |")
    print(f"  |   D    | gold EVICTED, response NOT in full (worst) | {len(D):>10d} | {1:>13d} |")
    print()

    def show(name, items, n_show=3):
        if not items:
            return
        print(f"  --- Bucket {name} ({len(items)}) up to {n_show} ---")
        for r in items[:n_show]:
            print(f"    id={r['id']}")
            print(f"    gold: {r['gold']!r}")
            print(f"    resp: {r['resp']!r}")

    show("A — distractor", A)
    show("B — fabrication despite gold present", B)
    show("C — substitute under eviction", C)
    show("D — worst-case fabrication", D)
    print()
    return {"A": len(A), "B": len(B), "C": len(C), "D": len(D)}


def main():
    print("# Block 3.6 pivot — mixture-cc adapter under KVzip cc inference\n")
    cc_data, base_m = section_C()
    section_D(cc_data)
    section_E(cc_data)
    section_F()


if __name__ == "__main__":
    main()
