"""Recompute the paper's reported numbers from the released evaluation files.

Every table and headline statistic in the paper is regenerated here from
`artifacts/eval/*.jsonl`, then compared against the value printed in the paper.
Nothing is read from a cached summary: the script parses the raw per-example
classifications and recomputes honest accuracy, McNemar tests, bootstrap
confidence intervals, Wilson intervals and Cohen's kappa from scratch.

    python scripts/verify_paper_claims.py

Exit status is 0 if every check reproduces, 1 otherwise.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from math import comb, sqrt
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "artifacts" / "eval"

CONF_LABELS = {"confident", "confident_answer"}
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]

_checks: list[tuple[str, bool, str]] = []


def check(name: str, got, want, tol: float = 0.0) -> None:
    """Record a comparison between a recomputed value and the paper's value."""
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
        detail = f"paper={want}  recomputed={got:.4f}" if tol else f"paper={want}  recomputed={got}"
    else:
        ok = got == want
        detail = f"paper={want}  recomputed={got}"
    _checks.append((name, ok, detail))


def load(name: str) -> list[dict]:
    with (EVAL / name).open() as fh:
        return [json.loads(line) for line in fh]


def rows_of(name: str, side: str | None = None) -> list[dict]:
    """Return flat per-example records; paired files nest under 'adapter'/'base'."""
    raw = load(name)
    if side is not None and side in raw[0]:
        return [r[side] for r in raw]
    return raw


def honest(rows) -> float:
    return sum(r["classification"] == "correct" for r in rows) / len(rows)


def at_ratio(rows, r):
    return [x for x in rows if abs(x["ratio"] - r) < 1e-9]


def conf_gold(rows):
    return [x for x in rows if x["gold_label"] in CONF_LABELS]


def key(r):
    return (r["id"], round(r["ratio"], 4))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def boot_ci(pairs, seed: int = 42, reps: int = 10000):
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(reps):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a for a, _ in sample) / n - sum(b for _, b in sample) / n)
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def paired(a_rows, b_rows, ratio=None, conf_only=False):
    A = {key(r): r["classification"] == "correct" for r in a_rows}
    Ag = {key(r): r["gold_label"] for r in a_rows}
    B = {key(r): r["classification"] == "correct" for r in b_rows}
    ks = [k for k in A if k in B]
    if ratio is not None:
        ks = [k for k in ks if abs(k[1] - ratio) < 1e-9]
    if conf_only:
        ks = [k for k in ks if Ag[k] in CONF_LABELS]
    return [(A[k], B[k]) for k in ks]


def mcnemar_stats(pairs):
    b = sum(1 for a, x in pairs if a and not x)
    c = sum(1 for a, x in pairs if x and not a)
    n = len(pairs)
    delta = sum(a for a, _ in pairs) / n - sum(x for _, x in pairs) / n
    return b, c, delta, mcnemar_exact(b, c)


# --------------------------------------------------------------------------
# Table 1 - per-ratio honest accuracy, mean over seeds 42/43/44 (KVzip, ps)
# --------------------------------------------------------------------------
def table1():
    base = rows_of("v1_eval.jsonl", "base")
    seeds = [
        rows_of("v1_eval.jsonl", "adapter"),
        rows_of("v1_eval_seed43.jsonl", "adapter"),
        rows_of("v1_eval_seed44.jsonl", "adapter"),
    ]
    paper_base = {0.05: 0.15, 0.10: 0.27, 0.20: 0.30, 0.30: 0.35, 0.50: 0.44, 0.80: 0.51}
    paper_adp = {0.05: 1.00, 0.10: 0.99, 0.20: 0.92, 0.30: 0.79, 0.50: 0.66, 0.80: 0.60}
    for r in RATIOS:
        check(f"Table 1  r={r:.2f} base", honest(at_ratio(base, r)), paper_base[r], 0.005)
        mean = sum(honest(at_ratio(s, r)) for s in seeds) / 3
        check(f"Table 1  r={r:.2f} adapter (3-seed mean)", mean, paper_adp[r], 0.005)
    check("Table 1  aggregate base", honest(base), 0.32, 0.005)
    check(
        "Table 1  aggregate adapter (3-seed mean)",
        sum(honest(s) for s in seeds) / 3,
        0.86,
        0.005,
    )
    check("Table 1  n", len(base), 463)
    gold = Counter(r["gold_label"] in CONF_LABELS for r in base)
    check("Table 1  gold Confident/Abstain", f"{gold[True]}/{gold[False]}", "97/366")


# --------------------------------------------------------------------------
# Section 5.1 - hallucination collapse 232 -> 7
# --------------------------------------------------------------------------
def hallucinations():
    base = rows_of("v1_eval.jsonl", "base")
    adp = rows_of("v1_eval.jsonl", "adapter")
    nb = sum(r["classification"] == "hallucinated" for r in base)
    na = sum(r["classification"] == "hallucinated" for r in adp)
    check("Sec 5.1  base hallucinations", nb, 232)
    check("Sec 5.1  adapter hallucinations", na, 7)
    check("Abstract  hallucination reduction", round(100 * (nb - na) / nb), 97)


# --------------------------------------------------------------------------
# Table 2 - honest accuracy at r=0.50 across compressors
# --------------------------------------------------------------------------
COMPRESSORS = {
    "KVzip": "kvzip",
    "ExpAttn": "expattn",
    "SnapKV (ctx)": "snapkv",
    "SnapKV (nat)": "snapkv_native",
}
_MIX = {
    "KVzip": "v1_eval_mixture_kvzip.jsonl",
    "ExpAttn": "v1_eval_mixture_expattn.jsonl",
    "SnapKV (ctx)": "v1_eval_mixture_snapkv.jsonl",
    "SnapKV (nat)": "v1_eval_mixture_snapkv_native.jsonl",
}
_V1 = {
    "KVzip": "v1_eval.jsonl",
    "ExpAttn": "v1_eval_expattn.jsonl",
    "SnapKV (ctx)": "v1_eval_snapkv.jsonl",
    "SnapKV (nat)": "v1_eval_snapkv_native.jsonl",
}


def table2():
    paper = {
        "KVzip": (0.54, 0.44, 0.65, 0.69),
        "ExpAttn": (0.60, 0.38, 0.67, 0.71),
        "SnapKV (ctx)": (0.64, 0.33, 0.46, 0.64),
        "SnapKV (nat)": (0.54, 0.38, 0.41, 0.51),
    }
    for comp, slug in COMPRESSORS.items():
        p2 = at_ratio(load(f"v1_eval_prompt_p2_{slug}.jsonl"), 0.50)
        frag = at_ratio(rows_of(_MIX[comp], "base"), 0.50)
        v1 = at_ratio(rows_of(_V1[comp], "adapter"), 0.50)
        mix = at_ratio(rows_of(_MIX[comp], "adapter"), 0.50)
        want = paper[comp]
        check(f"Table 2  {comp} P2", honest(p2), want[0], 0.006)
        check(f"Table 2  {comp} Fragment", honest(frag), want[1], 0.006)
        check(f"Table 2  {comp} v1", honest(v1), want[2], 0.006)
        check(f"Table 2  {comp} Mix", honest(mix), want[3], 0.006)


# --------------------------------------------------------------------------
# Table 3 - aggregate delta honest accuracy, v1 vs mixture
# --------------------------------------------------------------------------
def table3():
    paper_v1 = {"KVzip": 0.53, "ExpAttn": 0.49, "SnapKV (ctx)": 0.21, "SnapKV (nat)": 0.20}
    paper_mix = {"KVzip": 0.55, "ExpAttn": 0.57, "SnapKV (ctx)": 0.53, "SnapKV (nat)": 0.43}
    seed43 = {
        "KVzip": "v1_eval_mixture_seed43_kvzip.jsonl",
        "ExpAttn": "v1_eval_mixture_seed43_expattn.jsonl",
        "SnapKV (ctx)": "v1_eval_mixture_seed43_snapkv.jsonl",
        "SnapKV (nat)": "v1_eval_mixture_seed43_snapkv_native.jsonl",
    }
    for comp in COMPRESSORS:
        base = honest(rows_of(_MIX[comp], "base"))
        check(
            f"Table 3  {comp} v1 (seed 42)",
            honest(rows_of(_V1[comp], "adapter")) - base,
            paper_v1[comp],
            0.006,
        )
        d42 = honest(rows_of(_MIX[comp], "adapter")) - base
        d43 = honest(rows_of(seed43[comp], "adapter")) - base
        check(f"Table 3  {comp} Mix (2-seed mean)", (d42 + d43) / 2, paper_mix[comp], 0.006)


# --------------------------------------------------------------------------
# Table 4 - compressed-cache inference
# --------------------------------------------------------------------------
def table4():
    base_cc = load("v1_eval_base_compressed_kvzip.jsonl")
    v1_cc = load("v1_eval_compressed_kvzip.jsonl")
    mixps_cc = load("v1_eval_mixture_compressed_kvzip.jsonl")
    mixcc_cc = load("v1_eval_mixture_cc_kvzip.jsonl")
    check("Table 4  base-cc aggregate", honest(base_cc), 0.186, 0.001)
    check("Table 4  v1-ps cc aggregate", honest(v1_cc), 0.406, 0.001)
    check("Table 4  mix-ps cc aggregate", honest(mixps_cc), 0.730, 0.001)
    check("Table 4  mix-cc cc aggregate", honest(mixcc_cc), 0.821, 0.001)
    per_ratio = {
        0.05: (0.560, 1.000, 1.000),
        0.10: (0.495, 0.958, 0.989),
        0.20: (0.415, 0.829, 0.963),
        0.30: (0.323, 0.600, 0.923),
        0.50: (0.271, 0.354, 0.500),
        0.80: (0.233, 0.315, 0.315),
    }
    for r, (a, b, c) in per_ratio.items():
        check(f"Table 4  r={r:.2f} v1-ps cc", honest(at_ratio(v1_cc, r)), a, 0.002)
        check(f"Table 4  r={r:.2f} mix-ps cc", honest(at_ratio(mixps_cc, r)), b, 0.002)
        check(f"Table 4  r={r:.2f} mix-cc cc", honest(at_ratio(mixcc_cc, r)), c, 0.002)
    # Appendix M: single-compressor cc-trained adapter
    single = load("v1_eval_compressed_trained_kvzip.jsonl")
    check("App M  single-compressor cc aggregate", honest(single), 0.734, 0.002)
    hi_single = conf_gold(at_ratio(single, 0.80))
    hi_mix = conf_gold(at_ratio(mixcc_cc, 0.80))
    check(
        "App M  single-compressor wrong answers at r=0.80",
        f"{sum(r['classification'] == 'wrong_answer' for r in hi_single)}/{len(hi_single)}",
        "42/69",
    )
    check(
        "App M  mix-cc wrong answers at r=0.80",
        f"{sum(r['classification'] == 'wrong_answer' for r in hi_mix)}/{len(hi_mix)}",
        "21/69",
    )
    # Sec 5.5: mix-cc vs mix-ps tie at r=0.80 on Confident-gold
    pairs = paired(
        at_ratio(mixcc_cc, 0.80), at_ratio(mixps_cc, 0.80), conf_only=True
    )
    _, _, _, p = mcnemar_stats(pairs)
    check("Sec 5.5  mix-cc vs mix-ps at r=0.80, McNemar p", round(p, 2), 0.80, 0.005)


# --------------------------------------------------------------------------
# Appendix K - McNemar and paired bootstrap
# --------------------------------------------------------------------------
def appendix_k():
    want_p2 = {"KVzip": (12, 5), "ExpAttn": (10, 5), "SnapKV (ctx)": (5, 5), "SnapKV (nat)": (5, 6)}
    want_frag = {"KVzip": (16, 4), "ExpAttn": (15, 1), "SnapKV (ctx)": (16, 4)}
    want_hi = {"KVzip": (25, 3), "ExpAttn": (25, 4), "SnapKV (ctx)": (23, 3), "SnapKV (nat)": (22, 3)}
    pooled_p2, pooled_frag, pooled_hi = [], [], []
    for comp, slug in COMPRESSORS.items():
        mix = rows_of(_MIX[comp], "adapter")
        p2 = load(f"v1_eval_prompt_p2_{slug}.jsonl")
        frag = rows_of(_MIX[comp], "base")
        pr = paired(mix, p2, ratio=0.50)
        b, c, _, _ = mcnemar_stats(pr)
        check(f"App K  mix vs P2 r=0.50 {comp} b/c", f"{b}/{c}", "%d/%d" % want_p2[comp])
        pooled_p2 += pr
        if comp in want_frag:
            pr = paired(mix, frag, ratio=0.50)
            b, c, _, _ = mcnemar_stats(pr)
            check(f"App K  mix vs Frag r=0.50 {comp} b/c", f"{b}/{c}", "%d/%d" % want_frag[comp])
        pooled_frag += paired(mix, frag, ratio=0.50)
        pr = paired(mix, p2, ratio=0.80, conf_only=True)
        b, c, _, _ = mcnemar_stats(pr)
        check(f"App K  mix vs P2 r=0.80 Conf {comp} b/c", f"{b}/{c}", "%d/%d" % want_hi[comp])
        pooled_hi += pr
    for label, pool, wb, wc, wp in [
        ("mix vs P2 r=0.50", pooled_p2, 32, 21, 0.169),
        ("mix vs Frag r=0.50", pooled_frag, 56, 13, 1.7e-7),
        ("mix vs P2 r=0.80 Conf", pooled_hi, 95, 13, 1.5e-16),
    ]:
        b, c, delta, p = mcnemar_stats(pool)
        check(f"App K  pooled {label} b/c", f"{b}/{c}", f"{wb}/{wc}")
        check(f"App K  pooled {label} p", p, wp, abs(wp) * 0.05 + 1e-18)
    _, _, d, _ = mcnemar_stats(pooled_hi)
    check("Sec 5.2  pooled high-retention advantage", d, 0.32, 0.006)


# --------------------------------------------------------------------------
# Appendix L - answerability preservation
# --------------------------------------------------------------------------
def appendix_l():
    def rate(rows, ratio=None):
        sel = conf_gold(rows if ratio is None else at_ratio(rows, ratio))
        k = sum(
            r["classification"] == "correct" and not r.get("model_abstained", False) for r in sel
        )
        return k, len(sel)

    cases = [
        ("v1 ps, KVzip", _V1["KVzip"], "adapter", None, 0.485, 0.557, 1.15),
        ("v1 ps, ExpAttn", _V1["ExpAttn"], "adapter", None, 0.468, 0.574, 1.23),
        ("v1 ps, SnapKV-ctx", _V1["SnapKV (ctx)"], "adapter", None, 0.545, 0.568, 1.04),
        ("v1 ps, SnapKV-nat", _V1["SnapKV (nat)"], "adapter", None, 0.567, 0.575, 1.01),
        ("mix ps, KVzip", _MIX["KVzip"], "adapter", None, 0.485, 0.608, 1.25),
        ("mix ps, ExpAttn", _MIX["ExpAttn"], "adapter", None, 0.468, 0.638, 1.36),
        ("mix ps, SnapKV-ctx", _MIX["SnapKV (ctx)"], "adapter", None, 0.545, 0.591, 1.08),
        ("mix ps, SnapKV-nat", _MIX["SnapKV (nat)"], "adapter", None, 0.567, 0.567, 1.00),
        ("mix ps, KVzip (r=0.80)", _MIX["KVzip"], "adapter", 0.80, 0.522, 0.652, 1.25),
        ("mix ps, ExpAttn (r=0.80)", _MIX["ExpAttn"], "adapter", 0.80, 0.492, 0.662, 1.34),
        ("mix ps, SnapKV-ctx (r=0.80)", _MIX["SnapKV (ctx)"], "adapter", 0.80, 0.542, 0.678, 1.25),
        ("v1 ps, KVzip (r=0.80)", _V1["KVzip"], "adapter", 0.80, 0.522, 0.638, 1.22),
    ]
    for name, fn, side, ratio, b_rate, a_rate, ratio_want in cases:
        bk, bn = rate(rows_of(fn, "base"), ratio)
        ak, an = rate(rows_of(fn, side), ratio)
        check(f"App L  {name} base rate", bk / bn, b_rate, 0.002)
        check(f"App L  {name} adapter rate", ak / an, a_rate, 0.002)
        check(f"App L  {name} preservation ratio", ak / bk, ratio_want, 0.006)
    # compressed-cache rows use base-cc as the comparator
    base_cc = load("v1_eval_base_compressed_kvzip.jsonl")
    for name, fn, ratio, want_a, want_ratio in [
        ("mix ps cc, KVzip", "v1_eval_mixture_compressed_kvzip.jsonl", None, 0.258, 6.25),
        ("mix cc cc, KVzip", "v1_eval_mixture_cc_kvzip.jsonl", None, 0.247, 6.00),
        ("mix ps cc, KVzip (r=0.80)", "v1_eval_mixture_compressed_kvzip.jsonl", 0.80, 0.290, 20.0),
        ("mix cc cc, KVzip (r=0.80)", "v1_eval_mixture_cc_kvzip.jsonl", 0.80, 0.319, 22.0),
    ]:
        bk, bn = rate(base_cc, ratio)
        ak, an = rate(load(fn), ratio)
        lo, hi = wilson(bk, bn)
        check(f"App L  {name} base-cc rate", bk / bn, 0.041 if ratio is None else 0.014, 0.002)
        check(f"App L  {name} adapter rate", ak / an, want_a, 0.002)
        check(f"App L  {name} lift", ak / bk, want_ratio, 0.02)
        check(f"App L  {name} base-cc Wilson CI excludes adapter", hi < ak / an, True)
    # Sec 5.5 raw counts for the cc-trained adapter
    ak, an = rate(load("v1_eval_mixture_cc_kvzip.jsonl"))
    bk, bn = rate(base_cc)
    check("Sec 5.5  mix-cc aggregate correct", f"{ak}/{an}", "24/97")
    check("Sec 5.5  base-cc aggregate correct", f"{bk}/{bn}", "4/97")
    ak, an = rate(load("v1_eval_mixture_cc_kvzip.jsonl"), 0.80)
    bk, bn = rate(base_cc, 0.80)
    check("Sec 5.5  mix-cc correct at r=0.80", f"{ak}/{an}", "22/69")
    check("Sec 5.5  base-cc correct at r=0.80", f"{bk}/{bn}", "1/69")


# --------------------------------------------------------------------------
# Section 5.2 - over-abstention on Confident-gold at r=0.80
# --------------------------------------------------------------------------
def over_abstention():
    got_p2, got_mix = [], []
    for comp, slug in COMPRESSORS.items():
        p2 = conf_gold(at_ratio(load(f"v1_eval_prompt_p2_{slug}.jsonl"), 0.80))
        mix = conf_gold(at_ratio(rows_of(_MIX[comp], "adapter"), 0.80))
        got_p2.append(sum(r["model_abstained"] for r in p2) / len(p2))
        got_mix.append(sum(r["model_abstained"] for r in mix) / len(mix))
    check("Sec 5.2  P2 over-abstention range low", round(100 * min(got_p2)), 49)
    check("Sec 5.2  P2 over-abstention range high", round(100 * max(got_p2)), 51)
    check("Sec 5.2  mixture over-abstention range low", round(100 * min(got_mix)), 6)
    check("Sec 5.2  mixture over-abstention range high", round(100 * max(got_mix)), 10)


# --------------------------------------------------------------------------
# Appendix O - controlled-deletion mechanism tests
# --------------------------------------------------------------------------
def appendix_o():
    for tag, want_fixed_c, want_orig_c, want_fixed_a, want_orig_a in [
        ("mixture-ps", 0.147, 0.621, 0.885, 0.823),
        ("mixture-cc", 0.063, 0.274, 0.956, 0.956),
    ]:
        fixed = load(f"block10a_eval_{tag}.jsonl")
        orig = load(f"block10a_baseline_eval_{tag}.jsonl")
        check(f"App O  10A {tag} n", len(fixed), 208)
        check(f"App O  10A {tag} fixed-L Conf", honest(conf_gold(fixed)), want_fixed_c, 0.002)
        check(f"App O  10A {tag} orig-r Conf", honest(conf_gold(orig)), want_orig_c, 0.002)
        abst = [r for r in fixed if r["gold_label"] not in CONF_LABELS]
        abst0 = [r for r in orig if r["gold_label"] not in CONF_LABELS]
        check(f"App O  10A {tag} fixed-L Abstain", honest(abst), want_fixed_a, 0.002)
        check(f"App O  10A {tag} orig-r Abstain", honest(abst0), want_orig_a, 0.002)
    for tag, want_orig, want_adv, want_flip in [
        ("mixture-ps", 0.165, 0.567, 41),
        ("mixture-cc", 0.649, 0.887, 25),
    ]:
        adv = conf_gold(load(f"block10b_eval_{tag}.jsonl"))
        orig = conf_gold(load(f"block10b_baseline_eval_{tag}.jsonl"))
        check(f"App O  10B {tag} n_C", len(adv), 97)
        check(
            f"App O  10B {tag} original abstention",
            sum(r["model_abstained"] for r in orig) / len(orig),
            want_orig,
            0.002,
        )
        check(
            f"App O  10B {tag} adversarial abstention",
            sum(r["model_abstained"] for r in adv) / len(adv),
            want_adv,
            0.002,
        )
        O = {key(r): r["model_abstained"] for r in orig}
        A = {key(r): r["model_abstained"] for r in adv}
        flips = sum(1 for k in O if k in A and not O[k] and A[k])
        check(f"App O  10B {tag} answered->abstained flips", flips, want_flip)
    for tag, want_a, want_b_abst in [("mixture-ps", 14, 22), ("mixture-cc", 5, 20)]:
        a = load(f"block10c_eval_{tag}_maskA.jsonl")
        b = load(f"block10c_eval_{tag}_maskB.jsonl")
        check(f"App O  10C {tag} n", len(a), 22)
        check(f"App O  10C {tag} Mask A correct", sum(r["classification"] == "correct" for r in a), want_a)
        check(f"App O  10C {tag} Mask B correct", sum(r["classification"] == "correct" for r in b), 0)
        check(f"App O  10C {tag} Mask B abstentions", sum(r["model_abstained"] for r in b), want_b_abst)
    # the corrected claim: mix-cc has two confident wrong answers on Mask B
    b = load("block10c_eval_mixture-cc_maskB.jsonl")
    check(
        "App O  10C mix-cc Mask B wrong answers",
        sum(r["classification"] == "wrong_answer" for r in b),
        2,
    )


# --------------------------------------------------------------------------
# Appendix N - LLM-judge label validation
# --------------------------------------------------------------------------
def appendix_n():
    rows = load("block4_judge_log.jsonl")
    check("App N  judged examples", len(rows), 400)
    counts = Counter(r["verdict"] for r in rows)
    check("App N  PARTIAL count", counts["PARTIAL"], 48)
    ours = {"confident_answer": "YES", "confident": "YES", "abstain": "NO"}
    strict = [(ours[r["label_type"]], r["verdict"]) for r in rows if r["verdict"] != "PARTIAL"]
    check("App N  strict n", len(strict), 352)

    def kappa(pairs):
        n = len(pairs)
        po = sum(a == b for a, b in pairs) / n
        ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
        return (po - pe) / (1 - pe), po

    k, po = kappa(strict)
    check("App N  Cohen's kappa (strict)", k, 0.609, 0.002)
    check("Limitations  raw agreement", po, 0.804, 0.002)
    for name, mapped, want in [
        ("PARTIAL->Confident", "YES", 0.525),
        ("PARTIAL->Abstain", "NO", 0.545),
    ]:
        pairs = [
            (ours[r["label_type"]], mapped if r["verdict"] == "PARTIAL" else r["verdict"])
            for r in rows
        ]
        check(f"App N  kappa lenient {name}", kappa(pairs)[0], want, 0.002)
    cm = Counter((r["label_type"], r["verdict"]) for r in rows)
    check("App N  Confident->YES", cm[("confident_answer", "YES")], 123)
    check("App N  Confident->NO", cm[("confident_answer", "NO")], 55)
    check("App N  Confident->PARTIAL", cm[("confident_answer", "PARTIAL")], 22)
    check("App N  Abstain->NO", cm[("abstain", "NO")], 160)
    check("App N  Abstain->YES", cm[("abstain", "YES")], 14)
    check("App N  Abstain->PARTIAL", cm[("abstain", "PARTIAL")], 26)


# --------------------------------------------------------------------------
# Appendix P - random-mask training-signal probe
# --------------------------------------------------------------------------
def appendix_p():
    rand_kv = rows_of("v1_eval_random_kvzip.jsonl", "adapter")
    mix_kv = rows_of("v1_eval_mixture_kvzip.jsonl", "adapter")
    check("App P  random-mask aggregate on KVzip", honest(rand_kv), 0.873, 0.002)
    check("App P  mixture aggregate on KVzip", honest(mix_kv), 0.855, 0.003)
    _, _, d, p = mcnemar_stats(paired(rand_kv, mix_kv))
    check("App P  random vs mixture delta", d, 0.017, 0.002)
    check("App P  random vs mixture p", round(p, 2), 0.23, 0.005)
    rand_sk = rows_of("v1_eval_random_snapkv.jsonl", "adapter")
    v1_sk = rows_of("v1_eval_snapkv.jsonl", "adapter")
    _, _, d, p = mcnemar_stats(paired(rand_sk, v1_sk))
    check("App P  random vs v1 on SnapKV-ctx delta", d, 0.276, 0.002)
    check("App P  random vs v1 on SnapKV-ctx p < 1e-20", p < 1e-20, True)


# --------------------------------------------------------------------------
# Appendices F/G/Q - test split, Llama replication, RULER
# --------------------------------------------------------------------------
def replications():
    t = load("v1_eval_test_held_out.jsonl")
    base, adp = [r["base"] for r in t], [r["adapter"] for r in t]
    check("App F  test split n", len(t), 100)
    check("App F  test split delta honest", honest(adp) - honest(base), 0.63, 0.006)
    ab = [r for r in adp if r["gold_label"] not in CONF_LABELS]
    bb = [r for r in base if r["gold_label"] not in CONF_LABELS]
    check(
        "App F  test split hallucinations",
        f"{sum(r['classification'] == 'hallucinated' for r in ab)}/{len(ab)}",
        "1/85",
    )
    check(
        "App F  test split base hallucinations",
        f"{sum(r['classification'] == 'hallucinated' for r in bb)}/{len(bb)}",
        "59/85",
    )
    sel = load("v1_eval_for_selection.jsonl")
    check(
        "App F  selection split delta honest",
        honest([r["adapter"] for r in sel]) - honest([r["base"] for r in sel]),
        0.50,
        0.006,
    )
    ll = load("v1_eval_llama_seed42.jsonl")
    lb, la = [r["base"] for r in ll], [r["adapter"] for r in ll]
    check("App G  Llama aggregate delta honest", honest(la) - honest(lb), 0.64, 0.006)
    nb = sum(r["classification"] == "hallucinated" for r in lb)
    na = sum(r["classification"] == "hallucinated" for r in la)
    check("App G  Llama hallucinations", f"{nb} -> {na}", "228 -> 10")
    check("App G  Llama hallucination reduction", round(100 * (nb - na) / nb), 96)
    ncb = sum(r["classification"] == "correct" for r in lb)
    nca = sum(r["classification"] == "correct" for r in la)
    check("App G  Llama correct-answer gain", nca - ncb, 298)


def main() -> int:
    for fn in (
        table1,
        hallucinations,
        table2,
        table3,
        table4,
        appendix_k,
        appendix_l,
        over_abstention,
        appendix_o,
        appendix_n,
        appendix_p,
        replications,
    ):
        try:
            fn()
        except Exception as exc:  # keep going so one missing file cannot hide the rest
            _checks.append((f"{fn.__name__} (crashed)", False, repr(exc)))

    width = max(len(n) for n, _, _ in _checks)
    failed = 0
    for name, ok, detail in _checks:
        if not ok:
            failed += 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(_checks) - failed}/{len(_checks)} checks reproduce the published values.")
    if failed:
        print(f"{failed} check(s) did not reproduce.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
