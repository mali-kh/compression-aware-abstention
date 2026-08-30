"""Block 1: bootstrap CIs and paired McNemar tests at r=0.50."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ART = Path("artifacts")
RATIO = 0.5
N_BOOT = 10_000
SEED = 42

COMPRESSORS = [
    ("KVzip",         "v1_eval.jsonl",              "v1_eval_mixture_kvzip.jsonl",         "v1_eval_prompt_p2_kvzip.jsonl"),
    ("ExpAttn",       "v1_eval_expattn.jsonl",      "v1_eval_mixture_expattn.jsonl",       "v1_eval_prompt_p2_expattn.jsonl"),
    ("SnapKV-ctx",    "v1_eval_snapkv.jsonl",       "v1_eval_mixture_snapkv.jsonl",        "v1_eval_prompt_p2_snapkv.jsonl"),
    ("SnapKV-native", "v1_eval_snapkv_native.jsonl","v1_eval_mixture_snapkv_native.jsonl", "v1_eval_prompt_p2_snapkv_native.jsonl"),
]


def honest_correct(gold_label, classification, model_abstained):
    if classification != "correct":
        return False
    if gold_label in ("confident", "confident_answer"):
        return model_abstained is False
    if gold_label == "abstain":
        return model_abstained is True
    raise ValueError(f"unexpected gold_label: {gold_label}")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def at_ratio(records, r=RATIO):
    return [x for x in records if abs(x.get("ratio", -1) - r) < 1e-9]


def index_by_id(records):
    d = {}
    for r in records:
        d[r["id"]] = r
    return d


def build_compressor(name, v1_file, mix_file, p2_file):
    """Return (ids, base_vec, v1_vec, mix_vec, p2_vec) as numpy bool arrays at r=0.5, joined by id."""
    v1 = at_ratio(load_jsonl(ART / v1_file))
    mx = at_ratio(load_jsonl(ART / mix_file))
    p2 = at_ratio(load_jsonl(ART / p2_file))

    v1_ix = index_by_id(v1)
    mx_ix = index_by_id(mx)
    p2_ix = index_by_id(p2)

    ids = sorted(set(v1_ix) & set(mx_ix) & set(p2_ix))

    base, v1v, mxv, p2v = [], [], [], []
    for i in ids:
        v1r, mxr, p2r = v1_ix[i], mx_ix[i], p2_ix[i]
        gl = v1r["gold_label"]
        # base / Fragment from v1 file's base block
        b = v1r["base"]
        base.append(honest_correct(gl, b["classification"], b["model_abstained"]))
        # v1 from v1 file's adapter block
        a = v1r["adapter"]
        v1v.append(honest_correct(gl, a["classification"], a["model_abstained"]))
        # mixture from mixture file's adapter block
        am = mxr["adapter"]
        mxv.append(honest_correct(gl, am["classification"], am["model_abstained"]))
        # P2 from top-level fields of P2 file
        p2v.append(honest_correct(gl, p2r["classification"], p2r["model_abstained"]))

    return ids, np.array(base), np.array(v1v), np.array(mxv), np.array(p2v), len(v1), len(mx), len(p2)


def bootstrap_ci(vec, rng, n=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI for the mean."""
    N = len(vec)
    idx = rng.integers(0, N, size=(n, N))
    means = vec[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vec.mean()), float(lo), float(hi)


def paired_bootstrap_diff_ci(a, b, rng, n=N_BOOT, alpha=0.05):
    """Paired bootstrap CI for mean(a) - mean(b)."""
    N = len(a)
    assert len(b) == N
    idx = rng.integers(0, N, size=(n, N))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def mcnemar_exact(a_correct, b_correct):
    """
    Exact McNemar via two-sided binomial test on discordants.
    Returns (a, b_disc, c_disc, d, stat, p) where the 2x2 is
        rows = A, cols = B, indexing (correct, wrong).
    Test statistic = min(b_disc, c_disc).  p-value from BinomialTest(min(b,c), b+c, 0.5).
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    a = int(np.sum(a_correct & b_correct))
    b_disc = int(np.sum(a_correct & ~b_correct))   # A correct, B wrong
    c_disc = int(np.sum(~a_correct & b_correct))   # A wrong,   B correct
    d = int(np.sum(~a_correct & ~b_correct))
    n = b_disc + c_disc
    if n == 0:
        return a, b_disc, c_disc, d, 0, 1.0
    k = min(b_disc, c_disc)
    p = binomtest(k, n, p=0.5, alternative="two-sided").pvalue
    stat = k
    return a, b_disc, c_disc, d, stat, p


def fmt_ci(point, lo, hi):
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def main():
    rng = np.random.default_rng(SEED)

    # Compute per-compressor data
    data = {}
    print("=== Sanity check: n at r=0.5 per file ===")
    for name, v1f, mxf, p2f in COMPRESSORS:
        ids, base, v1v, mxv, p2v, nv1, nmx, np2 = build_compressor(name, v1f, mxf, p2f)
        data[name] = {"ids": ids, "base": base, "v1": v1v, "mix": mxv, "p2": p2v}
        print(f"  {name:14s}  v1_file n={nv1}  mix_file n={nmx}  p2_file n={np2}  joined n={len(ids)}")
    print()

    print("=== Sanity check: point estimates vs report Table 2 ===")
    print("(report claims KVzip: base=0.44 v1=0.65 mix=0.69 P2=0.54)")
    for name in data:
        d = data[name]
        print(f"  {name:14s}  base={d['base'].mean():.3f}  v1={d['v1'].mean():.3f}  mix={d['mix'].mean():.3f}  P2={d['p2'].mean():.3f}  (n={len(d['ids'])})")
    print()

    # ----- Table A: honest accuracy with 95% bootstrap CIs -----
    print("### Table A: Honest accuracy with 95% bootstrap CIs (10,000 resamples, seed=42)")
    print()
    header = f"{'Compressor':14s} | n  | {'base (CI)':22s} | {'v1 (CI)':22s} | {'mix (CI)':22s} | P2 (CI)"
    print(header)
    print("-" * len(header))
    for name in data:
        d = data[name]
        n = len(d["ids"])
        cells = []
        for cond in ["base", "v1", "mix", "p2"]:
            point, lo, hi = bootstrap_ci(d[cond], rng)
            cells.append(fmt_ci(point, lo, hi))
        print(f"{name:14s} | {n:2d} | {cells[0]:22s} | {cells[1]:22s} | {cells[2]:22s} | {cells[3]}")
    print()

    # ----- Table B: paired McNemar tests -----
    print("### Table B: Paired McNemar tests at r=0.50 (exact two-sided)")
    print()
    header = f"{'Compressor':14s} | {'comparison':15s} | b (A>B) | c (B>A) | stat | exact p | sig α=0.05"
    print(header)
    print("-" * len(header))
    comparisons = [
        ("mix vs P2",       "mix",  "p2"),
        ("mix vs Fragment", "mix",  "base"),
        ("v1 vs P2",        "v1",   "p2"),
    ]
    mcnemar_rows = []
    for name in data:
        d = data[name]
        for label, A, B in comparisons:
            a, b_disc, c_disc, dd, stat, p = mcnemar_exact(d[A], d[B])
            sig = "yes" if p < 0.05 else "no"
            mcnemar_rows.append((name, label, a, b_disc, c_disc, dd, stat, p, sig))
            print(f"{name:14s} | {label:15s} | {b_disc:7d} | {c_disc:7d} | {stat:4d} | {p:7.4f} | {sig}")
    print()

    # ----- Table C: paired bootstrap CI on differences -----
    print("### Table C: Paired bootstrap CIs on the difference (A − B)")
    print()
    header = f"{'Compressor':14s} | {'comparison':15s} | point Δ | 95% CI on Δ"
    print(header)
    print("-" * len(header))
    diff_rows = []
    for name in data:
        d = data[name]
        for label, A, B in comparisons:
            point, lo, hi = paired_bootstrap_diff_ci(d[A], d[B], rng)
            diff_rows.append((name, label, point, lo, hi))
            sign = "+" if point >= 0 else ""
            sign_lo = "+" if lo >= 0 else ""
            sign_hi = "+" if hi >= 0 else ""
            print(f"{name:14s} | {label:15s} | {sign}{point:5.2f}  | [{sign_lo}{lo:.2f}, {sign_hi}{hi:.2f}]")
    print()

    # ----- Spot-check: KVzip mix vs P2, dump tuples and recount -----
    print("### Spot-check: KVzip mix vs P2 — manual recount")
    d = data["KVzip"]
    pairs = list(zip(d["mix"].tolist(), d["p2"].tolist()))
    print(f"  n={len(pairs)} paired (mix, P2) tuples")
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    dd = sum(1 for x, y in pairs if not x and not y)
    print(f"  manually-counted 2x2: a(both correct)={a}  b(mix only)={b}  c(P2 only)={c}  d(both wrong)={dd}")
    print(f"  discordants total = {b + c}; min(b,c) = {min(b, c)}")
    # Compare with mcnemar_exact
    aa, bb, cc, dd2, stat, p = mcnemar_exact(d["mix"], d["p2"])
    print(f"  mcnemar_exact returned: a={aa} b={bb} c={cc} d={dd2} stat={stat} p={p:.4f}")
    print(f"  match: {(a, b, c, dd) == (aa, bb, cc, dd2)}")
    print()

    # ----- Bullet summary -----
    print("### Bullet summary")
    print()
    by_comp = {}
    for (name, label, a, b_disc, c_disc, dd, stat, p, sig) in mcnemar_rows:
        by_comp.setdefault(name, {})[label] = (b_disc, c_disc, p, sig)
    diff_lookup = {(n, l): (pt, lo, hi) for (n, l, pt, lo, hi) in diff_rows}
    for name in data:
        for label, _, _ in comparisons:
            b_disc, c_disc, p, sig = by_comp[name][label]
            pt, lo, hi = diff_lookup[(name, label)]
            ci_includes_zero = (lo <= 0 <= hi)
            tag = "SIGNIFICANT" if sig == "yes" else "n.s."
            zero_note = "CI includes 0" if ci_includes_zero else "CI excludes 0"
            print(f"- **{name} / {label}**: Δ={pt:+.2f} [{lo:+.2f}, {hi:+.2f}], McNemar exact p={p:.3f} → {tag} (α=0.05); {zero_note}.")


if __name__ == "__main__":
    main()
