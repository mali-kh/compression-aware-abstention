# Exp 2 — Mixture seed variance (seeds 42 + 43)

_Pairwise spread across two mixture-training seeds on per-compressor aggregate Δhon. Lock criterion (per spec): max spread ≤ 0.05 across all four compressors → "v2 finding locked."_

## TL;DR

**v2 finding LOCKED.** Max aggregate spread across all four compressors is **0.0173** (KVzip), well under the 0.05 threshold. The mixture-training-closes-SnapKV-gap result is robust to seed 42 vs 43.

## Aggregate Δhon spread per compressor

| compressor | n | base | seed 42 adapter hon | seed 42 Δhon | seed 43 adapter hon | seed 43 Δhon | aggregate spread |
|---|---|---|---|---|---|---|---|
| KVzip | 463 | 0.318 | 0.855 | +0.5378 | 0.873 | +0.5551 | **0.0173** |
| ExpAttn | 414 | 0.312 | 0.879 | +0.5676 | 0.879 | +0.5676 | **0.0000** |
| SnapKV (ctx) | 431 | 0.339 | 0.872 | +0.5336 | 0.865 | +0.5267 | **0.0070** |
| SnapKV (native) | 419 | 0.360 | 0.788 | +0.4272 | 0.790 | +0.4296 | **0.0024** |

| | max spread | locked? |
|---|---|---|
| **across 4 compressors** | **0.0173** | **PASS (≤ 0.05)** |

Notable: ExpAttn aggregate Δhon is **identical to 4 decimal places** across seeds (+0.5676 each). KVzip has the highest aggregate spread at 0.017pp.

## Training trajectory differences (informational)

The two seeds reach the same aggregate destination via different paths:

| | seed 42 | seed 43 |
|---|---|---|
| epoch 1 val_loss (holdout) | 0.1272 | **0.1029** |
| epoch 2 val_loss | 0.1242 | 0.1072 |
| epoch 3 val_loss | **0.1224** | 0.1291 |
| **best epoch** | **3** | **1** |
| best val_loss | 0.1224 | 0.1029 |

Different overfit shape: seed 42 monotone-improving across all 3 epochs (best=3); seed 43 already at minimum at epoch 1, overfits thereafter (best=1). Same observation v1 had with seeds 42/43/44 — sweep val_loss is noisy enough at 1-epoch granularity that the picked checkpoint can vary, but the resulting per-ratio metrics are stable. The "trajectory varies, destination is stable" pattern from v1 §7.1 reproduces under mixture training.

## Per-(compressor, ratio) spread — where the cell-level variance lives

Aggregate is locked, but individual cells show non-trivial spread, mostly at high ratios with small abstain-gold n:

| compressor | ratio | s42 Δhon | s43 Δhon | spread | flag |
|---|---|---|---|---|---|
| KVzip | 0.20 | +0.561 | +0.671 | 0.110 | * |
| KVzip | 0.30 | +0.415 | +0.600 | 0.185 | * |
| KVzip | 0.80 | +0.151 | +0.000 | 0.151 | * |
| ExpAttn | 0.30 | +0.412 | +0.569 | 0.157 | * |
| ExpAttn | 0.80 | +0.182 | +0.015 | 0.167 | * |
| SnapKV (ctx) | 0.30 | +0.431 | +0.552 | 0.121 | * |
| SnapKV (ctx) | 0.50 | +0.308 | +0.385 | 0.077 | * |
| **SnapKV (ctx)** | **0.80** | **+0.167** | **−0.061** | **0.227** | **\*\*** |
| SnapKV (native) | 0.20 | +0.366 | +0.451 | 0.084 | * |
| SnapKV (native) | 0.30 | +0.125 | +0.292 | 0.167 | * |
| **SnapKV (native)** | **0.80** | **+0.127** | **−0.085** | **0.211** | **\*\*** |

Cells without `*` (spread ≤ 0.05) are: all four compressors at ratios 0.05/0.10; KVzip 0.50; ExpAttn 0.05/0.10/0.20/0.50; SnapKV (ctx) 0.05/0.10/0.20; SnapKV (native) 0.05/0.10/0.50.

The two cells with extreme spread (`**`) are both at ratio=0.80 on SnapKV — Δhon **flips sign** across seeds. Mechanism: at ratio=0.80 the val has very few abstain-gold rows (4 for KVzip, 1 for ExpAttn, 10 for SnapKV-ctx, 10 for SnapKV-native — see v1 cross-compressor numbers), so a few example classifications flipping creates large percentage swings on a small denominator. This is exactly the v1 §5 "rare counter-instance" regime: high apparent fill but supporting evidence happens to be evicted. The cells are noise-dominated.

If we report per-ratio spread in a paper-style table, the 0.80 cells should be flagged as small-n. The aggregate metric integrates over the full val and is robust to per-cell noise.

## What this resolves vs leaves open

**Resolved:**
- The mixture-training-closes-SnapKV-gap result is not seed-specific.
- The aggregate v2 finding is reportable from a single seed without quantitative qualification.

**Still open (not part of the spread question):**
- B1's interaction with the v2 lock — B1's headline-test framing applies equally to both seeds.
- The 0.80 high-fill counter-instance regime is per-seed noisy; v2 paper can either (a) report aggregate only, (b) report per-ratio with the small-n flag at 0.80, or (c) enrich the high-fill abstain-gold subset before the next training run (the v1 §8 v2 priority that's been pending). I'd recommend (b) for the v2 paper as written — honest about noise without skipping the per-ratio framing the project standard requires.

## Files

- `artifacts/v1_eval_mixture_*.jsonl` — seed 42 paired evals (4 compressors)
- `artifacts/v1_eval_mixture_seed43_*.jsonl` — seed 43 paired evals (4 compressors)
- `artifacts/v1_lora_best_mixture_seed42/`, `v1_lora_best_mixture_seed43/` — adapters (epoch 3 vs epoch 1 — different best-by-val choices per seed)
- This file
