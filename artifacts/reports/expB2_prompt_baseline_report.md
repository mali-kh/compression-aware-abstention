# Exp B2 — Prompt-only abstention baseline (P2 only)

_P2 is the "light abstention" prompt. P3 was skipped by design after the gate tripped on P2 KVzip (Δhon vs base = +0.45, well above the +0.15 threshold) — P3 is a stronger version that would amplify the same pattern (more aggressive abstain → more over-abstention at high-fill) without changing the framing._

_All four compressors, base Qwen2.5-7B-Instruct, no LoRA, P2 prepended to the v1 eval prompt. Same eval harness as v1 Phase 3 (greedy, max_new_tokens=64). Source files: `artifacts/v1_eval_prompt_p2_<compressor>.jsonl`._

## TL;DR

- **P2 closes most of the v1 aggregate gap** on KVzip and ExpAttn (within-family). P2 actually **beats v1** on SnapKV aggregates (v1 never learned SnapKV's eviction signature; P2 blanket-abstains).
- **Mixture maintains its margin over P2** on aggregate (every compressor +5 to +13pp). The mixture's SnapKV contribution survives, just on a narrower basis than originally framed.
- **At ratio=0.50 (calibration-net slice)**: v1 over P2 = +10pp on KVzip / +7pp on ExpAttn / −18pp on SnapKV-ctx / −13pp on SnapKV-native. Mixture over P2 = +15pp / +12pp / +0pp / −3pp. **Adapter wins clearly at the within-family slice; ties on cross-family.**
- **At ratio=0.80 (high-fill)**: P2 honest_acc is *worse than base* on every compressor (over-abstains when the cache is intact). Adapter over P2 = +17pp to +32pp. **This is where the training contribution is most clear: the adapter learns when NOT to abstain.**

The paper's framing must shift slightly: training adds modest value at the calibration-net mid-slice (only on within-family compressors); the bigger training contribution is **calibrated non-abstention at high-fill ratios**, where prompt-only over-abstains.

## Aggregate per compressor

| compressor | n | base hon | base hal | P2 hon | P2 hal | v1 hon | v1 hal | mix hon | mix hal | P2 − base | v1 − P2 | mix − P2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **KVzip** | 463 | 0.317 | 0.570 | 0.767 | 0.106 | 0.844 | 0.076 | 0.855 | 0.058 | **+0.449** | +0.078 | +0.089 |
| **ExpAttn** | 414 | 0.312 | 0.577 | 0.749 | 0.130 | 0.802 | 0.130 | 0.879 | 0.068 | **+0.437** | +0.053 | +0.130 |
| **SnapKV (ctx)** | 431 | 0.339 | 0.517 | 0.819 | 0.065 | 0.550 | 0.313 | 0.872 | 0.084 | **+0.480** | **−0.269** | +0.053 |
| **SnapKV (native)** | 419 | 0.360 | 0.506 | 0.728 | 0.119 | 0.556 | 0.303 | 0.788 | 0.126 | **+0.368** | **−0.172** | +0.060 |

The two cells that should reshape the paper:

1. **v1 LOSES to P2 on SnapKV aggregate** (−27pp on ctx-only, −17pp on native). The v1 §10 "v1 transfers across query-agnostic but not cross-family" finding is correct, but the *quantitative* picture was inflated by the implicit comparison to base; against P2, v1 is much weaker on SnapKV.
2. **Mixture beats P2 on every compressor** (+0.05 to +0.13 aggregate). This is the v2-paper-relevant finding: the mixture training contribution is actually *more* meaningful when measured against a strong prompt baseline than against base.

## Per-ratio honest_accuracy by compressor

### KVzip (training compressor, within-family within v1)

| ratio | n | gC/gA | base | P2 | v1 | mix | P2 − base | v1 − P2 | mix − P2 |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 0/100 | 0.15 | **1.00** | 1.00 | 1.00 | +0.85 | +0.00 | +0.00 |
| 0.10 | 95 | 1/94 | 0.27 | 0.95 | 0.99 | 0.99 | +0.67 | +0.04 | +0.04 |
| 0.20 | 82 | 2/80 | 0.30 | 0.77 | 0.87 | 0.87 | +0.46 | +0.10 | +0.10 |
| 0.30 | 65 | 3/62 | 0.35 | 0.78 | 0.78 | 0.77 | +0.43 | +0.00 | −0.02 |
| **0.50** | **48** | **22/26** | **0.44** | **0.54** | **0.65** | **0.69** | **+0.10** | **+0.10** | **+0.15** |
| 0.80 | 73 | 69/4 | 0.51 | **0.34** | 0.60 | 0.66 | **−0.16** | **+0.26** | **+0.32** |

### ExpAttn (within-family transfer)

| ratio | n | gC/gA | base | P2 | v1 | mix | P2 − base | v1 − P2 | mix − P2 |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 98 | 0/98 | 0.21 | 0.92 | 0.92 | 1.00 | +0.70 | +0.00 | +0.08 |
| 0.10 | 89 | 1/88 | 0.22 | 0.87 | 0.89 | 0.98 | +0.64 | +0.02 | +0.11 |
| 0.20 | 68 | 1/67 | 0.31 | 0.82 | 0.90 | 0.96 | +0.52 | +0.07 | +0.13 |
| 0.30 | 51 | 4/47 | 0.37 | 0.76 | 0.73 | 0.78 | +0.39 | −0.04 | +0.02 |
| **0.50** | **42** | **23/19** | **0.38** | **0.60** | **0.67** | **0.71** | **+0.21** | **+0.07** | **+0.12** |
| 0.80 | 66 | 65/1 | 0.48 | **0.35** | 0.56 | 0.67 | **−0.14** | **+0.21** | **+0.32** |

### SnapKV (context-only) — cross-family

| ratio | n | gC/gA | base | P2 | v1 | mix | P2 − base | v1 − P2 | mix − P2 |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 0/100 | 0.25 | 0.99 | 0.64 | 1.00 | +0.74 | **−0.35** | +0.01 |
| 0.10 | 93 | 0/93 | 0.30 | 0.96 | 0.56 | 1.00 | +0.66 | **−0.40** | +0.04 |
| 0.20 | 75 | 2/73 | 0.37 | 0.89 | 0.55 | 0.93 | +0.52 | **−0.35** | +0.04 |
| 0.30 | 58 | 9/49 | 0.33 | 0.83 | 0.45 | 0.76 | +0.50 | **−0.38** | −0.07 |
| **0.50** | **39** | **18/21** | **0.33** | **0.64** | **0.46** | **0.64** | **+0.31** | **−0.18** | **+0.00** |
| 0.80 | 66 | 59/7 | 0.50 | **0.38** | 0.55 | 0.67 | **−0.12** | **+0.17** | **+0.29** |

### SnapKV (native) — cross-family

| ratio | n | gC/gA | base | P2 | v1 | mix | P2 − base | v1 − P2 | mix − P2 |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 0/100 | 0.27 | 0.96 | 0.67 | 1.00 | +0.69 | −0.29 | +0.04 |
| 0.10 | 92 | 1/91 | 0.32 | 0.87 | 0.62 | 0.97 | +0.55 | −0.25 | +0.10 |
| 0.20 | 71 | 13/58 | 0.37 | 0.72 | 0.48 | 0.73 | +0.35 | −0.24 | +0.01 |
| 0.30 | 48 | 18/30 | 0.38 | 0.65 | 0.42 | 0.50 | +0.27 | −0.23 | −0.15 |
| **0.50** | **37** | **28/9** | **0.38** | **0.54** | **0.41** | **0.51** | **+0.16** | **−0.13** | **−0.03** |
| 0.80 | 71 | 67/4 | 0.52 | **0.38** | 0.56 | 0.65 | **−0.14** | **+0.18** | **+0.27** |

## The high-fill over-abstention pattern

P2 honest_acc < base honest_acc at ratio=0.80 on every compressor:

| compressor | base hon @ 0.80 | P2 hon @ 0.80 | P2 over-abstention rate (CONF gold abstained / total gC) |
|---|---|---|---|
| KVzip | 0.51 | **0.34** | abst=49% on slice with 95% CONF gold → answers correctly 0.34, abstains incorrectly 0.49 |
| ExpAttn | 0.48 | **0.35** | abst=50%, gold 98% CONF |
| SnapKV (ctx) | 0.50 | **0.38** | abst=53%, gold 89% CONF |
| SnapKV (native) | 0.52 | **0.38** | abst=51%, gold 94% CONF |

P2 abstains on roughly half the high-fill slice — a fixed prior the prompt induces irrespective of cache content. The adapter (v1: 0.60-0.55; mix: 0.65-0.67 honest_acc) learns to *not* abstain when the cache is intact.

This is the cleanest mechanism evidence: the prompt cannot induce calibrated answer behavior; it can only induce a fixed abstain-prior. Calibrated abstention requires per-example sensitivity to cache content, which only the LoRA training delivers.

## Headline cells with target framing

For the paper, the operational test of the cache-state-introspection claim is at ratio=0.50 (the calibration-net slice where ratio-only and prompt-only policies are forced into a single prediction across the gold mix):

| comparison | KVzip @ 0.50 | SnapKV (ctx) @ 0.50 |
|---|---|---|
| base (P1) | 0.44 | 0.33 |
| base + P2 | 0.54 | 0.64 |
| v1 adapter | 0.65 (+10pp over P2) ✓ | 0.46 (−18pp under P2) ✗ |
| **mixture adapter** | **0.69 (+15pp over P2) ✓** | **0.64 (+0pp vs P2) ⚠** |

And at ratio=0.80 (calibrated non-abstention test):

| comparison | KVzip @ 0.80 | SnapKV (ctx) @ 0.80 |
|---|---|---|
| base (P1) | 0.51 | 0.50 |
| base + P2 | **0.34** | **0.38** |
| v1 adapter | 0.60 (+26pp over P2) | 0.55 (+17pp over P2) |
| **mixture adapter** | **0.66 (+32pp over P2)** | **0.67 (+29pp over P2)** |

## Interpretive paragraph (by design)

P2 wins decisively at low retention ratios (0.05–0.30) by giving the model a strong fixed abstention prior — at those ratios the gold distribution is ~99% ABSTAIN, so any policy that abstains aggressively scores near-perfect by construction. At ratio=0.50 (the calibration-net slice with balanced gold), the prompt-only policy is forced into a per-example trade-off it cannot resolve from text alone; the within-family adapter beats P2 by 10pp, the mixture by 15pp, and these are the headline numbers for the cache-state-introspection claim. At ratio=0.80 (high-fill), P2's behavior reveals its mechanism: a fixed abstention prior over-fires on the dominantly CONFIDENT-gold slice, and P2's honest_acc actually drops below base. The adapter does not have this failure mode — it learns calibrated non-abstention at high-fill — and that's where the training contribution is most cleanly visible (+17pp to +32pp over P2). The cross-family SnapKV cells at ratio=0.50 are the asterisk: v1 loses to P2 there because v1 never learned SnapKV's eviction signature (the §10 v1-memo finding); the mixture matches but does not clearly beat P2 on SnapKV at the calibration-net slice. The mixture's SnapKV contribution is most visible on aggregate and at high-fill ratios, where it both matches the prompt baseline's low-ratio gains and prevents the high-fill over-abstention failure mode.

## Reading against B2 success criteria

| criterion | result |
|---|---|
| v1 beats best prompt baseline by ≥15pp aggregate KVzip | **FAIL** (v1 over P2 = +7.8pp aggregate) |
| v1 beats best prompt baseline by ≥10pp at ratio=0.50 KVzip | **PASS marginally** (v1 over P2 = +10.4pp) |
| Mixture maintains SnapKV advantage | **PASS on aggregate** (mix +5pp), **PASS on high-fill** (mix +29pp), **TIES at ratio=0.50** (mix vs P2 = +0.0pp) |

This is your "partial" framing: training adds modest value at the calibration-net slice and substantial value at high-fill; the aggregate framing is weakened because the prompt baseline catches the easy-low-ratio cases for free.

## What this does NOT show

- **P3 not run.** P3 is stronger than P2 and per direction would amplify the same pattern (more aggressive abstain → bigger high-fill failure → closer aggregate match). Skipping P3 is a deliberate choice, not a methodological gap.
- **Other prompt designs not tested.** Per spec, no preemptive iteration on prompts. P2 is the defensible baseline.
- **Conditional answer quality not separately analyzed under P2.** A standalone calibration sub-check on P2 (overcaution + cw rate at ratios 0.50, 0.80) would tighten the mechanism narrative; out of scope for this report.

## Files

- `artifacts/v1_eval_prompt_p2_kvzip.jsonl`, `_expattn.jsonl`, `_snapkv.jsonl`, `_snapkv_native.jsonl` — P2 eval data
- `artifacts/v1_eval.jsonl`, `v1_eval_expattn.jsonl`, `v1_eval_snapkv.jsonl`, `v1_eval_snapkv_native.jsonl` — v1 + base
- `artifacts/v1_eval_mixture_*.jsonl` — mixture
- `scripts/expB2_prompt_baseline.py` — eval script
- This file
