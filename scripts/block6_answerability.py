"""Block 6 — answerability preservation metric.

Recomputes from existing eval JSONLs. No new evals. Pairs each
adapter eval with its base baseline at the matching (compressor val,
inference path), then reports two tables:

  Table 1: aggregate over all retention ratios
  Table 2: restricted to r=0.80 (high-fill slice; mostly C-gold)

For paired ps eval files (v1_eval*.jsonl, v1_eval_mixture_*.jsonl):
  base block ↔ adapter block in the same row.

For flat cc eval files (v1_eval_*compressed*.jsonl, v1_eval_mixture_cc_*.jsonl):
  there is no base-under-cc on disk. We pair each cc-adapter row to the
  same id+ratio's base-under-ps from the matching paired ps file. This
  measures "preservation vs base-prompt-style" — i.e. does the adapter
  under deployment-realistic cc inference still answer as well as base
  did under the easier ps regime. Flagged in the output.

Definitions:
  CONFIDENT-gold examples: gold_label in {"confident", "confident_answer"}
  answer-attempted: model_abstained == False
  answer-correct  : classification == "correct" AND model_abstained == False
  answerability preservation rate = answer-correct / n_C
  preservation ratio = adapter_correct / base_correct (vs base on same val
                       set / inference path, with the cc caveat above)

Run via /usr/bin/python3 (system); needs nothing beyond stdlib + math.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ART = Path("artifacts")
CONFIDENT_GOLDS = ("confident", "confident_answer")


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return p, max(0.0, lo), min(1.0, hi)


def is_confident(gold_label: str) -> bool:
    return gold_label in CONFIDENT_GOLDS


def is_correct_answer(cls_dict) -> bool:
    """answer-correct: classification == 'correct' AND model_abstained == False."""
    return (cls_dict["classification"] == "correct"
            and cls_dict["model_abstained"] is False)


def is_answer_attempted(cls_dict) -> bool:
    return cls_dict["model_abstained"] is False


def filter_records(records, ratio_filter=None):
    """Restrict to CONFIDENT-gold (and optionally a specific ratio)."""
    out = []
    for r in records:
        if not is_confident(r.get("gold_label", "")):
            continue
        if ratio_filter is not None and not math.isclose(r["ratio"], ratio_filter):
            continue
        out.append(r)
    return out


# ---------- pairing ----------


def paired_counts(paired_file: Path, ratio_filter=None):
    """For a paired ps eval (v1_eval*.jsonl), count base/adapter
    answer-correct on CONFIDENT-gold."""
    rows = load_jsonl(paired_file)
    rows = [r for r in rows
            if is_confident(r.get("gold_label", ""))
            and r.get("base") is not None
            and r.get("adapter") is not None]
    if ratio_filter is not None:
        rows = [r for r in rows if math.isclose(r["ratio"], ratio_filter)]
    n = len(rows)
    base_c = sum(1 for r in rows if is_correct_answer(r["base"]))
    adapt_c = sum(1 for r in rows if is_correct_answer(r["adapter"]))
    base_a = sum(1 for r in rows if is_answer_attempted(r["base"]))
    adapt_a = sum(1 for r in rows if is_answer_attempted(r["adapter"]))
    return n, base_c, adapt_c, base_a, adapt_a


def cc_vs_ps_base_counts(cc_file: Path, ps_paired_file: Path, ratio_filter=None):
    """For a flat cc eval, pair each cc-adapter row with the same
    (id, ratio) row's BASE block from the matching paired ps file.
    Returns (n, base_c, adapter_c, base_a, adapter_a). The base is
    under PROMPT-STYLE inference (no base-cc available on disk)."""
    cc_rows = [r for r in load_jsonl(cc_file)
               if r.get("classification") != "error"
               and is_confident(r.get("gold_label", ""))]
    if ratio_filter is not None:
        cc_rows = [r for r in cc_rows if math.isclose(r["ratio"], ratio_filter)]
    ps_rows = load_jsonl(ps_paired_file)
    ps_ix = {(r["id"], r["ratio"]): r for r in ps_rows
             if r.get("base") is not None}
    n = 0
    base_c = adapt_c = base_a = adapt_a = 0
    for cc in cc_rows:
        key = (cc["id"], cc["ratio"])
        ps = ps_ix.get(key)
        if ps is None:
            continue
        if not is_confident(ps["gold_label"]):
            continue
        n += 1
        if is_correct_answer(ps["base"]):
            base_c += 1
        if is_answer_attempted(ps["base"]):
            base_a += 1
        if is_correct_answer(cc):
            adapt_c += 1
        if is_answer_attempted(cc):
            adapt_a += 1
    return n, base_c, adapt_c, base_a, adapt_a


# ---------- conditions to cover ----------

CONDITIONS = [
    # (label, paired_file, cc_file_or_None, comparator_paired_file_for_cc)
    ("v1 ps,        KVzip val",         "v1_eval.jsonl",                      None, None),
    ("v1 ps,        ExpAttn val",       "v1_eval_expattn.jsonl",              None, None),
    ("v1 ps,        SnapKV-ctx val",    "v1_eval_snapkv.jsonl",               None, None),
    ("v1 ps,        SnapKV-nat val",    "v1_eval_snapkv_native.jsonl",        None, None),

    ("mixture-ps ps, KVzip val",        "v1_eval_mixture_kvzip.jsonl",        None, None),
    ("mixture-ps ps, ExpAttn val",      "v1_eval_mixture_expattn.jsonl",      None, None),
    ("mixture-ps ps, SnapKV-ctx val",   "v1_eval_mixture_snapkv.jsonl",       None, None),
    ("mixture-ps ps, SnapKV-nat val",   "v1_eval_mixture_snapkv_native.jsonl", None, None),

    ("mixture-ps cc, KVzip val",        None, "v1_eval_mixture_compressed_kvzip.jsonl",  "v1_eval_mixture_kvzip.jsonl"),
    ("mixture-ps cc, SnapKV-ctx val",   None, "v1_eval_mixture_compressed_snapkv.jsonl", "v1_eval_mixture_snapkv.jsonl"),
    ("mixture-cc cc, KVzip val",        None, "v1_eval_mixture_cc_kvzip.jsonl",          "v1_eval_mixture_kvzip.jsonl"),
]


def compute_table(ratio_filter=None):
    """Returns list of dicts with all metrics per condition."""
    rows = []
    for label, paired_f, cc_f, ps_pair_for_cc in CONDITIONS:
        if paired_f is not None:
            n, bc, ac, ba, aa = paired_counts(ART / paired_f, ratio_filter)
            base_path = "ps"  # base under prompt-style
        else:
            n, bc, ac, ba, aa = cc_vs_ps_base_counts(
                ART / cc_f, ART / ps_pair_for_cc, ratio_filter,
            )
            base_path = "ps (no base-cc)"

        b_rate, b_lo, b_hi = wilson_ci(bc, n)
        a_rate, a_lo, a_hi = wilson_ci(ac, n)
        ratio_pres = (ac / bc) if bc > 0 else float("nan")
        rows.append({
            "label": label,
            "n_C": n,
            "base_correct": bc,
            "adapter_correct": ac,
            "base_attempted": ba,
            "adapter_attempted": aa,
            "base_rate": b_rate, "base_lo": b_lo, "base_hi": b_hi,
            "adapter_rate": a_rate, "adapter_lo": a_lo, "adapter_hi": a_hi,
            "preservation_ratio": ratio_pres,
            "base_path": base_path,
        })
    return rows


def fmt_rate(p, lo, hi):
    return f"{p:.3f} [{lo:.3f},{hi:.3f}]"


def print_table(name, rows):
    print(f"### {name}\n")
    print("```")
    hdr = (
        f"{'condition':<32s} | {'n_C':>3s} | {'b cor':>5s} | {'a cor':>5s} | "
        f"{'base rate (Wilson 95% CI)':<28s} | {'adapter rate (Wilson 95% CI)':<31s} | "
        f"{'pres':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        b_str = fmt_rate(r["base_rate"], r["base_lo"], r["base_hi"])
        a_str = fmt_rate(r["adapter_rate"], r["adapter_lo"], r["adapter_hi"])
        pr = r["preservation_ratio"]
        pr_str = "—" if pr != pr else f"{pr:.3f}"  # nan check
        print(
            f"{r['label']:<32s} | {r['n_C']:>3d} | "
            f"{r['base_correct']:>5d} | {r['adapter_correct']:>5d} | "
            f"{b_str:<28s} | {a_str:<31s} | {pr_str:>5s}"
        )
    print("```")
    print()


def main():
    print("# Block 6 — Answerability preservation\n")
    print("CONFIDENT-gold = `gold_label ∈ {confident, confident_answer}`")
    print("answer-correct = `classification == 'correct' AND model_abstained == False`")
    print("preservation ratio = adapter answer-correct count / base answer-correct count\n")
    print("Caveat: cc rows compare against base under PROMPT-STYLE inference (no base-cc")
    print("on disk); preservation ratio for cc rows is 'cc adapter vs base-ps'.\n")

    aggregate = compute_table(ratio_filter=None)
    print_table("Table 1: Aggregate (over all retention ratios)", aggregate)

    r080 = compute_table(ratio_filter=0.80)
    print_table("Table 2: r=0.80 only", r080)

    print("### Bullet summary\n")
    # Q1: v1 adapter under prompt-style preservation
    v1_lines = [r for r in aggregate if r["label"].startswith("v1 ps,")]
    avg_pres = sum(r["preservation_ratio"] for r in v1_lines if r["preservation_ratio"] == r["preservation_ratio"]) / max(len(v1_lines), 1)
    pres_above_1 = sum(1 for r in v1_lines if r["preservation_ratio"] >= 1.0)
    print(f"- v1 adapter (ps inference) preservation ratio across the 4 compressors: "
          f"avg = {avg_pres:.3f}; {pres_above_1}/4 cells have ratio ≥ 1.0 "
          f"(adapter answers correctly at-least-as-often as base on CONFIDENT-gold). "
          f"Per-cell: " +
          ", ".join(f"{r['label'].split(',')[1].strip()}={r['preservation_ratio']:.2f}" for r in v1_lines))

    # Q2: mixture-cc cc preservation
    mxcc = next(r for r in aggregate if r["label"].startswith("mixture-cc cc,"))
    mxcc_080 = next(r for r in r080 if r["label"].startswith("mixture-cc cc,"))
    print(f"- mixture-cc under cc inference (KVzip val): aggregate "
          f"preservation ratio = {mxcc['preservation_ratio']:.3f} "
          f"(adapter {mxcc['adapter_correct']}/{mxcc['n_C']} vs base-ps "
          f"{mxcc['base_correct']}/{mxcc['n_C']}). At r=0.80: "
          f"{mxcc_080['preservation_ratio']:.3f} "
          f"(adapter {mxcc_080['adapter_correct']}/{mxcc_080['n_C']} vs "
          f"base-ps {mxcc_080['base_correct']}/{mxcc_080['n_C']}).")

    # Q3: cells below 0.8
    lows = [(r["label"], r["preservation_ratio"]) for r in aggregate
            if r["preservation_ratio"] == r["preservation_ratio"]
            and r["preservation_ratio"] < 0.8]
    lows_080 = [(r["label"], r["preservation_ratio"]) for r in r080
                if r["preservation_ratio"] == r["preservation_ratio"]
                and r["preservation_ratio"] < 0.8]
    if not lows and not lows_080:
        print(f"- No cells fall below preservation ratio 0.8 in either table — "
              f"adapters preserve at least 80% of base's correct answers on "
              f"CONFIDENT-gold across every (adapter, inference, compressor) cell.")
    else:
        print(f"- Cells with preservation ratio < 0.8 (substantial sacrifice of correct answers):")
        for lbl, pr in lows:
            print(f"    Aggregate: {lbl} → {pr:.3f}")
        for lbl, pr in lows_080:
            print(f"    r=0.80:    {lbl} → {pr:.3f}")


if __name__ == "__main__":
    main()
