"""Block 10 pivot — adversarial / controlled-deletion analysis.

Inputs (per adapter):
  block10a_eval_{adapter}.jsonl                  fixed-L=500
  block10a_baseline_eval_{adapter}.jsonl         original kept_mask, same subset
  block10b_eval_{adapter}.jsonl                  adversarial-tight-span eviction
  block10b_baseline_eval_{adapter}.jsonl         original kept_mask, same subset
  block10c_eval_{adapter}_maskA.jsonl            r=0.50 CONFIDENT, KVzip top-50%
  block10c_eval_{adapter}_maskB.jsonl            r=0.50 CONFIDENT, KVzip bottom-50%

Outputs three sections:
  Section 10A: fixed-L vs original-ratio honest accuracy, per adapter, with
               per-class (CONFIDENT/ABSTAIN) breakdown + Wilson CIs.
  Section 10B: abstention rate on adversarial-mask CONFIDENT-gold examples,
               per adapter, vs same-record original-mask abstention rate.
  Section 10C: per-pair discrimination rate (correct on A AND abstain on B).
  Aggregate three-bullet summary.

Run via /usr/bin/python3 (system); only stdlib + math.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ART = Path("artifacts")
ADAPTERS = ["mixture-ps", "mixture-cc"]
CONFIDENT_GOLDS = ("confident", "confident_answer")


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return p, max(0.0, lo), min(1.0, hi)


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def is_confident(r):
    return r["gold_label"] in CONFIDENT_GOLDS


def is_correct(r):
    return r["classification"] == "correct"


def is_abstained(r):
    return bool(r.get("model_abstained"))


def honest_correct(r):
    """correct + correct-abstention."""
    if r["gold_label"] == "abstain":
        return is_abstained(r) or is_correct(r)
    return is_correct(r)


def index_by(records, key=("id", "ratio")):
    return {tuple(r[k] for k in key): r for r in records}


# ---------------- Section 10A ----------------


def section_10a():
    print("## Section 10A: Fixed-retained-length L=500 vs original-ratio (prompt-style)\n")
    print("Subset: val_v1.jsonl records with sum(kept_mask) >= 500. Same records used")
    print("for both conditions; difference is only the kept_mask shape.\n")
    print("```")
    hdr = (f"{'adapter':<12s} | {'cond':<10s} | {'n':>3s} | {'CONF n':>6s} | "
           f"{'CONF hon (CI)':<24s} | {'ABST n':>6s} | {'ABST hon (CI)':<24s} | "
           f"{'agg hon (CI)':<24s}")
    print(hdr); print("-" * len(hdr))
    for ad in ADAPTERS:
        for cond, fp in [("fixed-L", f"block10a_eval_{ad}.jsonl"),
                          ("orig-r", f"block10a_baseline_eval_{ad}.jsonl")]:
            path = ART / fp
            if not path.exists():
                print(f"{ad:<12s} | {cond:<10s} | (missing {fp})")
                continue
            rows = load(path)
            n = len(rows)
            conf = [r for r in rows if is_confident(r)]
            abst = [r for r in rows if r["gold_label"] == "abstain"]
            n_c, n_a = len(conf), len(abst)
            # CONF: honest_correct = correct
            c_c = sum(1 for r in conf if is_correct(r))
            # ABST: honest_correct = abstained or correct
            a_c = sum(1 for r in abst if honest_correct(r))
            agg_c = sum(1 for r in rows if honest_correct(r))
            cci = wilson_ci(c_c, n_c) if n_c else (0, 0, 0)
            aci = wilson_ci(a_c, n_a) if n_a else (0, 0, 0)
            agg = wilson_ci(agg_c, n)
            print(f"{ad:<12s} | {cond:<10s} | {n:>3d} | {n_c:>6d} | "
                  f"{fmt(*cci):<24s} | {n_a:>6d} | {fmt(*aci):<24s} | {fmt(*agg):<24s}")
    print("```\n")
    print("Interpretation: if the adapter relied on retention-length priors "
          "(short input ⇒ abstain), forcing all examples to L=500 should change behavior")
    print("substantially compared to the original (variable-length) ratio masks.\n")


# ---------------- Section 10B ----------------


def section_10b():
    print("## Section 10B: Adversarial tight-span eviction (CONFIDENT-gold, all ratios)\n")
    print("Mask = existing kept_mask MINUS tight-span tokens, refilled to original")
    print("retention with random non-kept positions (seed=42 + record id).\n")
    print("Correct behaviour given the construction: model should ABSTAIN since the")
    print("answer-bearing tokens are gone.\n")
    print("```")
    hdr = (f"{'adapter':<12s} | {'cond':<10s} | {'n_C':>4s} | {'abst rate (CI)':<24s} | "
           f"{'correct count':<13s} | {'wrong count':<11s}")
    print(hdr); print("-" * len(hdr))
    for ad in ADAPTERS:
        for cond, fp in [("adversar.", f"block10b_eval_{ad}.jsonl"),
                          ("orig-mask", f"block10b_baseline_eval_{ad}.jsonl")]:
            path = ART / fp
            if not path.exists():
                print(f"{ad:<12s} | {cond:<10s} | (missing {fp})")
                continue
            rows = load(path)
            n = len(rows)
            n_abst = sum(1 for r in rows if is_abstained(r))
            n_corr = sum(1 for r in rows if is_correct(r))
            n_wrong = n - n_abst - n_corr  # wrong specific answer
            ci = wilson_ci(n_abst, n)
            print(f"{ad:<12s} | {cond:<10s} | {n:>4d} | {fmt(*ci):<24s} | {n_corr:<13d} | {n_wrong:<11d}")
    print("```\n")
    print("Per-pair (adversar-vs-orig) per adapter:\n")
    print("```")
    print(f"{'adapter':<12s} | {'pair n':>6s} | {'abst-FLIP cases':<14s} | "
          f"{'orig-abst → adv-abst':<22s} | {'orig-answ → adv-abst':<22s}")
    for ad in ADAPTERS:
        adv = ART / f"block10b_eval_{ad}.jsonl"
        base = ART / f"block10b_baseline_eval_{ad}.jsonl"
        if not (adv.exists() and base.exists()):
            continue
        adv_ix = index_by(load(adv))
        base_ix = index_by(load(base))
        keys = sorted(set(adv_ix) & set(base_ix))
        n_pair = len(keys)
        flip_to_abst = 0  # base answered something, adv abstained (good!)
        kept_abst = 0     # base abstained, adv abstained (no change)
        for k in keys:
            adv_r = adv_ix[k]
            base_r = base_ix[k]
            if is_abstained(adv_r) and not is_abstained(base_r):
                flip_to_abst += 1
            elif is_abstained(adv_r) and is_abstained(base_r):
                kept_abst += 1
        flips_total = flip_to_abst
        print(f"{ad:<12s} | {n_pair:>6d} | {flips_total:<14d} | "
              f"{kept_abst:<22d} | {flip_to_abst:<22d}")
    print("```\n")
    print("Interpretation: 'orig-answ → adv-abst' counts pairs where the adapter answered ")
    print("on the original mask but abstained on the adversarial (tight-spans-evicted)")
    print("mask. High count = adapter responds to evidence absence. Low count = adapter")
    print("doesn't notice evidence removal.\n")


# ---------------- Section 10C ----------------


def section_10c():
    print("## Section 10C: Anti-priority paired (r=0.50 CONFIDENT-gold)\n")
    print("Mask A: existing kept_mask (KVzip top-50%); Mask B: bit-flipped (KVzip bottom-50%).")
    print("Same length, same example, opposite KVzip-rank prioritisation.\n")
    print("Discrimination: correct on A AND (abstained-or-non-correct) on B.\n")
    print("```")
    hdr = (f"{'adapter':<12s} | {'n':>3s} | {'A correct':>9s} | {'B correct':>9s} | "
           f"{'B abst':>6s} | {'discrim (CI)':<24s} | {'A→B agree':>10s}")
    print(hdr); print("-" * len(hdr))
    for ad in ADAPTERS:
        a_path = ART / f"block10c_eval_{ad}_maskA.jsonl"
        b_path = ART / f"block10c_eval_{ad}_maskB.jsonl"
        if not (a_path.exists() and b_path.exists()):
            print(f"{ad:<12s} | (missing one of A/B)")
            continue
        a_ix = index_by(load(a_path))
        b_ix = index_by(load(b_path))
        keys = sorted(set(a_ix) & set(b_ix))
        n = len(keys)
        n_a_corr = sum(1 for k in keys if is_correct(a_ix[k]))
        n_b_corr = sum(1 for k in keys if is_correct(b_ix[k]))
        n_b_abst = sum(1 for k in keys if is_abstained(b_ix[k]))
        # discrimination: A correct AND B abstained (the "ideal" pair behavior)
        discrim = sum(1 for k in keys
                      if is_correct(a_ix[k]) and is_abstained(b_ix[k]))
        d_ci = wilson_ci(discrim, n)
        # also report: pairs where the adapter's behavior was IDENTICAL on A and B
        # (no discrimination)
        same = sum(1 for k in keys
                   if is_abstained(a_ix[k]) == is_abstained(b_ix[k]))
        print(f"{ad:<12s} | {n:>3d} | {n_a_corr:>9d} | {n_b_corr:>9d} | "
              f"{n_b_abst:>6d} | {fmt(*d_ci):<24s} | {same:>10d}")
    print("```\n")
    print("Discrimination = (A correct ∧ B abstained) / n. The cleanest mechanism test:")
    print("if the adapter's behavior is evidence-content-driven, mask A (where tight spans")
    print("survive) should yield correct answers and mask B (where tight spans are evicted)")
    print("should yield abstentions. If the adapter is length-prior-driven, both masks are")
    print("the same length (50% of n_ctx) so behavior should be similar.\n")


# ---------------- aggregate bullet summary ----------------


def aggregate_bullets():
    print("## Aggregate bullet summary\n")

    # Q1 — does fixed-L change behavior much?
    out_lines = []
    for ad in ADAPTERS:
        f1 = ART / f"block10a_eval_{ad}.jsonl"
        f2 = ART / f"block10a_baseline_eval_{ad}.jsonl"
        if not (f1.exists() and f2.exists()):
            continue
        r_fl = load(f1); r_or = load(f2)
        ix_fl = index_by(r_fl); ix_or = index_by(r_or)
        keys = sorted(set(ix_fl) & set(ix_or))
        agg_fl = sum(1 for k in keys if honest_correct(ix_fl[k])) / max(len(keys), 1)
        agg_or = sum(1 for k in keys if honest_correct(ix_or[k])) / max(len(keys), 1)
        out_lines.append(f"{ad}: fixed-L hon={agg_fl:.3f}, orig-ratio hon={agg_or:.3f}, Δ={agg_fl-agg_or:+.3f}")
    print("- 10A (length-prior): " + "; ".join(out_lines) + ".")

    # Q2 — adversarial: does abstention rise vs original?
    out_lines = []
    for ad in ADAPTERS:
        f1 = ART / f"block10b_eval_{ad}.jsonl"
        f2 = ART / f"block10b_baseline_eval_{ad}.jsonl"
        if not (f1.exists() and f2.exists()):
            continue
        r_adv = load(f1); r_or = load(f2)
        a_adv = sum(1 for r in r_adv if is_abstained(r)) / max(len(r_adv), 1)
        a_or = sum(1 for r in r_or if is_abstained(r)) / max(len(r_or), 1)
        out_lines.append(f"{ad}: orig abst={a_or:.3f}, adv abst={a_adv:.3f}, Δ={a_adv-a_or:+.3f}")
    print("- 10B (adversarial eviction): " + "; ".join(out_lines) + ".")

    # Q3 — discrimination at fixed retention
    out_lines = []
    for ad in ADAPTERS:
        f1 = ART / f"block10c_eval_{ad}_maskA.jsonl"
        f2 = ART / f"block10c_eval_{ad}_maskB.jsonl"
        if not (f1.exists() and f2.exists()):
            continue
        a_ix = index_by(load(f1)); b_ix = index_by(load(f2))
        keys = sorted(set(a_ix) & set(b_ix))
        n = len(keys)
        d = sum(1 for k in keys if is_correct(a_ix[k]) and is_abstained(b_ix[k])) / max(n, 1)
        out_lines.append(f"{ad}: discrim={d:.3f} (n={n})")
    print("- 10C (discrimination at fixed retention): " + "; ".join(out_lines) + ".")


def main():
    print("# Block 10 — adversarial / controlled-deletion analysis\n")
    section_10a()
    section_10b()
    section_10c()
    aggregate_bullets()


if __name__ == "__main__":
    main()
