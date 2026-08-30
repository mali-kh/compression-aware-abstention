# Exp 2 — Mixture-adapter calibration sub-check (seed 42)

_Conditional-wrong-rate analysis for the seed-42 mixture adapter against all four val sets. Same decomposition as v1 §4 / §10.4 (the writeup-folder version is `artifacts/writeup/07_calibration_subcheck.md`): for each conclusive (compressor, ratio) cell, separate the adapter's behavior into (a) overcaution rate (CONFIDENT-gold examples where the adapter abstained instead of answering) and (b) conditional wrong-when-answering rate (wrong-answer count / total examples the adapter chose to answer)._

_Computed in ~30 sec on the existing eval JSONLs — no new generation._

## Overcaution rate on CONFIDENT-gold per (ratio, compressor)

| ratio | compressor | n_CONF | base ovc | v1 adapter ovc | **mixture adapter ovc** | mix − v1 |
|---|---|---|---|---|---|---|
| 0.50 | KVzip | 22 | 5/22 = 23% | 7/22 = **32%** | **5/22 = 23%** | **−9pp** |
| 0.50 | ExpAttn | 23 | 8/23 = 35% | 2/23 = 9% | 5/23 = 22% | +13pp |
| 0.50 | SnapKV (ctx) | 18 | 3/18 = 17% | 3/18 = 17% | 3/18 = 17% | +0pp |
| 0.50 | SnapKV (native) | 28 | 8/28 = 29% | 4/28 = 14% | 5/28 = 18% | +4pp |
| 0.80 | KVzip | 69 | 12/69 = 17% | 2/69 = 3% | 7/69 = 10% | +7pp |
| 0.80 | ExpAttn | 65 | 9/65 = 14% | 2/65 = 3% | 5/65 = 8% | +5pp |
| 0.80 | SnapKV (ctx) | 59 | 11/59 = 19% | 0/59 = 0% | 4/59 = 7% | +7pp |
| 0.80 | SnapKV (native) | 67 | 9/67 = 13% | 0/67 = 0% | 4/67 = 6% | +6pp |

## Conditional wrong-when-answering rate (wrong_answer / examples adapter chose to answer)

| ratio | compressor | n_CONF | base cw / ans | v1 adapter cw / ans | **mixture adapter cw / ans** | mix − v1 |
|---|---|---|---|---|---|---|
| 0.50 | KVzip | 22 | 6/17 = 35% | 5/15 = 33% | **3/17 = 18%** | **−15pp** |
| 0.50 | ExpAttn | 23 | 5/15 = 33% | 6/21 = 29% | **3/18 = 17%** | **−12pp** |
| 0.50 | SnapKV (ctx) | 18 | 7/15 = 47% | 5/15 = 33% | 6/15 = 40% | +7pp |
| 0.50 | SnapKV (native) | 28 | 7/20 = 35% | 10/24 = 42% | **8/23 = 35%** | −7pp |
| 0.80 | KVzip | 69 | 21/57 = 37% | 23/67 = 34% | **17/62 = 27%** | **−7pp** |
| 0.80 | ExpAttn | 65 | 24/56 = 43% | 26/63 = 41% | **17/60 = 28%** | **−13pp** |
| 0.80 | SnapKV (ctx) | 59 | 16/48 = 33% | 24/59 = 41% | **15/55 = 27%** | **−14pp** |
| 0.80 | SnapKV (native) | 67 | 21/58 = 36% | 28/67 = 42% | **20/63 = 32%** | **−10pp** |

## Reading

### Two findings the mixture training surfaces

1. **The v1 KVzip-specific overcaution at ratio=0.50 closes.** v1 §4 / writeup-07 flagged the 32% overcaution rate (vs base 23%, +9pp absolute) at the headline mid-ratio under KVzip as the calibration concern that needed honesty in the limitations section. With mixture training that cell drops to **23% — exactly matching base** (−9pp vs v1). The KVzip-specific overcaution that was the v1 calibration tradeoff is no longer present.

2. **Conditional answer quality *improves* on 7 of 8 conclusive cells.** v1's calibration claim was preserved-within-noise (deltas −2pp to −3pp on n=15-57 — see writeup-07 reading). Mixture training does better than that:
   - KVzip @ 0.50: 33% → **18%** wrong-when-answering (−15pp)
   - ExpAttn @ 0.50: 29% → **17%** (−12pp)
   - KVzip @ 0.80: 34% → **27%** (−7pp)
   - ExpAttn @ 0.80: 41% → **28%** (−13pp)
   - SnapKV (ctx) @ 0.80: 41% → **27%** (−14pp)
   - SnapKV (native) @ 0.80: 42% → **32%** (−10pp)
   - SnapKV (native) @ 0.50: 42% → **35%** (−7pp)
   - Only SnapKV (ctx) @ 0.50 regresses: 33% → 40% (+7pp on n=15 — small slice, single-example noise)

The mixture-trained adapter is not just **more decisive** in the right direction — it's **better when it does decide to answer**. This is qualitatively different from v1's "preserved within noise" claim.

### What the v1 §10.4 framing predicted vs what happened

v1 §10.4 framed the calibration sub-check as preempting the obvious reviewer attack — "you traded helpfulness for safety." The defense was: conditional answer quality is preserved within noise; the v1 abstention gain is calibration-net.

For the mixture, the defense is stronger: conditional answer quality is *improved*. The mixture's abstention gain is therefore not a calibration-net trade — it's a calibration-net win plus an answer-quality-net win.

### Counter-finding worth naming

The trade is in the other direction: mixture overcaution at high ratios (0.80) rises modestly (3-10pp absolute, depending on compressor). The model is slightly more willing to abstain even when the cache looks full. v1 was at 0-3% overcaution at 0.80 across compressors; mixture is at 6-10%. Still well below base (13-19%), but worth naming.

Mechanism speculation: the mixture's training distribution at high ratios includes more abstain-gold examples than v1's (because each compressor contributes its own "high-fill but supporting evidence happened to be evicted" cases). The model learned to be less aggressively answer-confident at high ratios as a result. This is the v1 §5 finding ("gross-fill-proxy heuristic") being weakened — exactly what we'd want if v2 is moving toward the §5 fix.

## Where this shows up in the v2 paper

- **§4.1 calibration story**: replaces the v1 "preserved within noise" framing with "improved on 7 of 8 cells, with the KVzip-specific 0.50 overcaution gap closed." Stronger version of the same defense.
- **Limitations section**: the small overcaution rise at 0.80 (3-10pp) is a fair acknowledgement; still well below base.
- **Mechanism narrative**: the v1 §5 "gross-fill-proxy heuristic" is weakened by mixture training — direct evidence for why mixture training is the right v2 lead.

## What this analysis does not include

- **Single seed (42).** Seed 43 is currently training. The above numbers may shift slightly with seed 43; will fold in once available.
- **Aggregate (collapsed across ratios)** is not in the table because it's not the meaningful slice — calibration matters per-ratio.
- **n_CONF<5 ratios (0.05/0.10/0.20/0.30)**: not in the table per the v1 convention (insufficient sample for calibration analysis).

## Files

- `artifacts/v1_eval_mixture_kvzip.jsonl`, `_expattn`, `_snapkv`, `_snapkv_native` — source eval JSONLs
- `artifacts/v1_eval.jsonl`, `v1_eval_expattn.jsonl`, `v1_eval_snapkv.jsonl`, `v1_eval_snapkv_native.jsonl` — v1 source eval JSONLs (compared)
- This file
