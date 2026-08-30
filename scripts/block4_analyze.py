"""Block 4 analysis — agreement statistics + qualitative disagreement.

Inputs:
  artifacts/block4_sample.jsonl   — the 400 sampled records (with our labels)
  artifacts/block4_judge_log.jsonl — GPT-5 verdicts per id+ratio

Outputs to stdout:
  Section 1: headline κ (3 ways: strict, lenient-CONF, lenient-ABS)
  Section 2: confusion matrix (our 2 labels × judge 3 verdicts)
  Section 3: per-ratio agreement
  Section 4: qualitative disagreement (25 from each direction)
  Section 5: three-sentence interpretation

No new evals. Stdlib only.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ART = Path("artifacts")
SAMPLE = ART / "block4_sample.jsonl"
LOG = ART / "block4_judge_log.jsonl"
SEED = 42
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def cohen_kappa_binary(a: list[int], b: list[int]) -> float:
    """Cohen's κ for two binary 0/1 vectors."""
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - p_e) < 1e-12:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def main():
    sample = load_jsonl(SAMPLE)
    log = load_jsonl(LOG)
    log_ix = {(r["id"], r["ratio"]): r for r in log if "verdict" in r}

    # join sample with judge verdict on (id, ratio)
    rows = []
    for s in sample:
        key = (s["id"], s["ratio"])
        j = log_ix.get(key)
        if j is None or "verdict" not in j:
            continue
        ours = "CONFIDENT" if s["label_type"] == "confident_answer" else "ABSTAIN"
        rows.append({
            "id": s["id"],
            "ratio": s["ratio"],
            "ours": ours,
            "verdict": j["verdict"],
            "reasoning": j.get("reasoning", ""),
            "raw_text": j.get("raw_text", ""),
            "kept_tokens": j.get("ctx_len_kept_tokens"),
            "answer": s["metadata"]["answer"],
            "query": s["query"],
            "context": s["context"],
            "kept_mask": s["kept_mask"],
        })

    n_total = len(rows)
    n_partial = sum(1 for r in rows if r["verdict"] == "PARTIAL")
    n_invalid = sum(1 for r in rows if r["verdict"] == "INVALID")

    # ---- Section 1: headline κ ----
    print("# Block 4 — LLM-as-judge label validation\n")
    print(f"Judge model: gpt-5 (reasoning_effort=low, seed=42; temperature=0 not supported by GPT-5)")
    print(f"Total joined: {n_total}")
    print(f"Verdicts: {Counter(r['verdict'] for r in rows)}")
    print(f"Note: PARTIAL n={n_partial}; INVALID n={n_invalid}\n")

    print("## Section 1: Headline agreement statistics\n")
    # Map our label to binary 0/1 (CONFIDENT=1, ABSTAIN=0)
    # Map judge YES=1, NO=0
    # Strict (drop PARTIAL/INVALID)
    a_strict = []
    b_strict = []
    for r in rows:
        if r["verdict"] not in ("YES", "NO"):
            continue
        a_strict.append(1 if r["ours"] == "CONFIDENT" else 0)
        b_strict.append(1 if r["verdict"] == "YES" else 0)
    k_strict = cohen_kappa_binary(a_strict, b_strict)
    p_o_strict = sum(1 for x, y in zip(a_strict, b_strict) if x == y) / max(len(a_strict), 1)
    print(f"  κ (strict, excluding PARTIAL/INVALID, n={len(a_strict)}):  {k_strict:.3f}")
    print(f"      raw agreement: {p_o_strict:.3f}")

    # Lenient-CONF: PARTIAL → CONFIDENT (i.e., judge YES)
    a_lc = []; b_lc = []
    for r in rows:
        if r["verdict"] == "INVALID":
            continue
        a_lc.append(1 if r["ours"] == "CONFIDENT" else 0)
        b_lc.append(1 if r["verdict"] in ("YES", "PARTIAL") else 0)
    k_lc = cohen_kappa_binary(a_lc, b_lc)
    p_o_lc = sum(1 for x, y in zip(a_lc, b_lc) if x == y) / max(len(a_lc), 1)
    print(f"  κ (lenient, PARTIAL→YES, n={len(a_lc)}):  {k_lc:.3f}")
    print(f"      raw agreement: {p_o_lc:.3f}")

    # Lenient-ABS: PARTIAL → ABSTAIN (i.e., judge NO)
    a_la = []; b_la = []
    for r in rows:
        if r["verdict"] == "INVALID":
            continue
        a_la.append(1 if r["ours"] == "CONFIDENT" else 0)
        b_la.append(1 if r["verdict"] == "YES" else 0)  # PARTIAL becomes 0 (NO)
    k_la = cohen_kappa_binary(a_la, b_la)
    p_o_la = sum(1 for x, y in zip(a_la, b_la) if x == y) / max(len(a_la), 1)
    print(f"  κ (lenient, PARTIAL→NO, n={len(a_la)}):  {k_la:.3f}")
    print(f"      raw agreement: {p_o_la:.3f}")
    print()

    # ---- Section 2: Confusion matrix ----
    print("## Section 2: Confusion matrix\n")
    cm = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r["ours"]][r["verdict"]] += 1
    n_conf = sum(cm["CONFIDENT"].values())
    n_abst = sum(cm["ABSTAIN"].values())
    print("```")
    print(f"                  | judge YES | judge NO | judge PARTIAL | judge INVALID | row total")
    print(f"-------------------|-----------|----------|---------------|---------------|----------")
    print(f"our CONFIDENT      | {cm['CONFIDENT']['YES']:>9d} | {cm['CONFIDENT']['NO']:>8d} | "
          f"{cm['CONFIDENT']['PARTIAL']:>13d} | {cm['CONFIDENT']['INVALID']:>13d} | {n_conf:>9d}")
    print(f"our ABSTAIN        | {cm['ABSTAIN']['YES']:>9d} | {cm['ABSTAIN']['NO']:>8d} | "
          f"{cm['ABSTAIN']['PARTIAL']:>13d} | {cm['ABSTAIN']['INVALID']:>13d} | {n_abst:>9d}")
    cyes = cm['CONFIDENT']['YES'] + cm['ABSTAIN']['YES']
    cno = cm['CONFIDENT']['NO'] + cm['ABSTAIN']['NO']
    cpart = cm['CONFIDENT']['PARTIAL'] + cm['ABSTAIN']['PARTIAL']
    cinv = cm['CONFIDENT']['INVALID'] + cm['ABSTAIN']['INVALID']
    print(f"col total          | {cyes:>9d} | {cno:>8d} | {cpart:>13d} | {cinv:>13d} | {n_conf+n_abst:>9d}")
    print("```\n")

    # ---- Section 3: Per-ratio agreement ----
    print("## Section 3: Per-ratio agreement\n")
    print("Agreement = our CONFIDENT ↔ judge YES, our ABSTAIN ↔ judge NO. Excludes PARTIAL.\n")
    print("```")
    print(f"ratio | n_total | n_PARTIAL | n_excl_PARTIAL | agreement | by-class agreement (CONF / ABST)")
    print(f"------|---------|-----------|----------------|-----------|---------------------------------")
    for r_val in RATIOS:
        rsubset = [x for x in rows if math.isclose(x["ratio"], r_val)]
        if not rsubset:
            continue
        rpart = sum(1 for x in rsubset if x["verdict"] == "PARTIAL")
        rsubset_excl = [x for x in rsubset if x["verdict"] in ("YES", "NO")]
        agree = sum(1 for x in rsubset_excl
                    if (x["ours"] == "CONFIDENT" and x["verdict"] == "YES")
                    or (x["ours"] == "ABSTAIN" and x["verdict"] == "NO"))
        agree_rate = agree / len(rsubset_excl) if rsubset_excl else float("nan")

        conf_rs = [x for x in rsubset_excl if x["ours"] == "CONFIDENT"]
        abst_rs = [x for x in rsubset_excl if x["ours"] == "ABSTAIN"]
        c_agree = (sum(1 for x in conf_rs if x["verdict"] == "YES") / len(conf_rs)
                   if conf_rs else float("nan"))
        a_agree = (sum(1 for x in abst_rs if x["verdict"] == "NO") / len(abst_rs)
                   if abst_rs else float("nan"))
        c_str = "—" if c_agree != c_agree else f"{c_agree:.3f} ({len(conf_rs)})"
        a_str = "—" if a_agree != a_agree else f"{a_agree:.3f} ({len(abst_rs)})"

        print(f"{r_val:.2f}  | {len(rsubset):>7d} | {rpart:>9d} | {len(rsubset_excl):>14d} | "
              f"{agree_rate:>9.3f} | CONF={c_str:>15s}  ABST={a_str:>15s}")
    print("```\n")

    # ---- Section 4: Qualitative disagreement ----
    print("## Section 4: Qualitative disagreement analysis\n")
    rng = random.Random(SEED)
    # CONFIDENT but judge NO — our label says "answerable" but judge disagrees
    conf_no = [r for r in rows if r["ours"] == "CONFIDENT" and r["verdict"] == "NO"]
    rng.shuffle(conf_no)
    # ABSTAIN but judge YES — our label says "not answerable" but judge says it is
    abst_yes = [r for r in rows if r["ours"] == "ABSTAIN" and r["verdict"] == "YES"]
    rng.shuffle(abst_yes)

    print(f"### 4a. CONFIDENT (ours) → NO (judge): n={len(conf_no)}\n")
    print(f"  Pattern: our label says the truncated context supports a specific answer, "
          f"but the judge says the context lacks the evidence.\n")
    for i, r in enumerate(conf_no[:25], 1):
        print(f"#### Case {i}: id={r['id']}, ratio={r['ratio']}, kept_tokens={r['kept_tokens']}")
        print(f"  query: {r['query']}")
        print(f"  gold answer: {r['answer']!r}")
        print(f"  judge reasoning: {r['reasoning'][:250]}")
        print()

    print(f"\n### 4b. ABSTAIN (ours) → YES (judge): n={len(abst_yes)}\n")
    print(f"  Pattern: our label says the truncated context is insufficient (we expect the "
          f"model to abstain), but the judge says the context is enough to derive the answer.\n")
    for i, r in enumerate(abst_yes[:25], 1):
        print(f"#### Case {i}: id={r['id']}, ratio={r['ratio']}, kept_tokens={r['kept_tokens']}")
        print(f"  query: {r['query']}")
        print(f"  gold answer: {r['answer']!r}")
        print(f"  judge reasoning: {r['reasoning'][:250]}")
        print()

    # ---- Section 5: interpretation ----
    print("## Section 5: Interpretation (three sentences)\n")
    if k_strict >= 0.7:
        level = "high"
    elif k_strict >= 0.4:
        level = "moderate"
    else:
        level = "low"
    # by-ratio range
    ratio_agree = []
    for r_val in RATIOS:
        rsubset = [x for x in rows if math.isclose(x["ratio"], r_val) and x["verdict"] in ("YES", "NO")]
        if not rsubset:
            continue
        agree = sum(1 for x in rsubset
                    if (x["ours"] == "CONFIDENT" and x["verdict"] == "YES")
                    or (x["ours"] == "ABSTAIN" and x["verdict"] == "NO")) / len(rsubset)
        ratio_agree.append((r_val, agree, len(rsubset)))
    if ratio_agree:
        ratio_agree.sort(key=lambda x: x[1])
        print(f"- Overall agreement is **{level}** (strict κ = {k_strict:.3f}; raw agreement {p_o_strict:.3f}). "
              f"PARTIAL n={n_partial} ({100*n_partial/n_total:.1f}%); leniency choice barely shifts κ "
              f"(PARTIAL→YES = {k_lc:.3f}, PARTIAL→NO = {k_la:.3f}).")
        ratio_str = ", ".join(f"r={r:.2f}: {a:.3f} (n={n})" for r, a, n in ratio_agree)
        print(f"- Per-ratio agreement: {ratio_str}.")
    print(f"- Confusion: {cm['CONFIDENT']['YES']}/{n_conf} CONFIDENT examples got judge YES "
          f"({cm['CONFIDENT']['YES']/n_conf:.3f}), {cm['ABSTAIN']['NO']}/{n_abst} ABSTAIN examples "
          f"got judge NO ({cm['ABSTAIN']['NO']/n_abst:.3f}); {cm['CONFIDENT']['NO']} cells "
          f"are CONFIDENT→NO and {cm['ABSTAIN']['YES']} cells are ABSTAIN→YES (qualitative "
          f"patterns above).")


if __name__ == "__main__":
    main()
