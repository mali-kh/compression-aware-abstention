# 05 — Cross-compressor kept-mask overlap

This is the version to cite. Computed on the 48 examples present in the eval set of all three compressors (KVzip, ExpAttn, SnapKV-context-only) at ratio=0.50, and similarly per-ratio at each row count below.

Each compressor is calibrated to keep an **identical number of context tokens** at each ratio (the wrapper's topk reduction enforces this). So differences in the table below are pure agreement-on-which-tokens, not differences in retention budget.

## Mean Jaccard (|A ∩ B| / |A ∪ B|) per pair

| ratio | n | KVzip ↔ ExpAttn | KVzip ↔ SnapKV (ctx) | ExpAttn ↔ SnapKV (ctx) |
|---|---|---|---|---|
| 0.05 | 100 | 0.030 | 0.031 | 0.147 |
| 0.10 | 95 | 0.083 | 0.073 | 0.223 |
| 0.20 | 82 | 0.205 | 0.156 | 0.343 |
| 0.30 | 65 | 0.340 | 0.254 | 0.422 |
| **0.50** | **48** | **0.597** | **0.453** | **0.558** |
| 0.80 | 73 | 0.833 | 0.741 | 0.779 |

## Same data, intersection-over-min (|A ∩ B| / min(|A|, |B|)) — often more interpretable

| ratio | KVzip ↔ ExpAttn | KVzip ↔ SnapKV (ctx) | ExpAttn ↔ SnapKV (ctx) |
|---|---|---|---|
| 0.05 | 0.058 | 0.060 | 0.251 |
| 0.10 | 0.152 | 0.136 | 0.360 |
| 0.20 | 0.339 | 0.269 | 0.508 |
| 0.30 | 0.507 | 0.404 | 0.592 |
| **0.50** | **0.747** | **0.623** | **0.715** |
| 0.80 | 0.909 | 0.851 | 0.876 |

## Sanity: identical kept-set sizes per compressor per ratio

(confirms calibration; differences in Jaccard are not due to budget mismatches)

| ratio | KVzip mean kept | ExpAttn mean kept | SnapKV (ctx) mean kept |
|---|---|---|---|
| 0.05 | 116.5 | 116.5 | 116.5 |
| 0.10 | 231.0 | 231.0 | 231.0 |
| 0.20 | 461.7 | 461.7 | 461.7 |
| 0.30 | 694.0 | 694.0 | 694.0 |
| 0.50 | 1175.6 | 1175.6 | 1175.6 |
| 0.80 | 1852.7 | 1852.7 | 1852.7 |

## Reading for the paper

At ratio=0.50 (the most-mixed compression regime), KVzip and ExpAttn agree on **75%** of kept tokens (intersection-over-min). KVzip and SnapKV (context-only) agree on **62%**. The 17% relative gap in cache-state overlap maps to a ~2.5x gap in adapter transfer at the same slice (+0.29 ExpAttn vs +0.13 SnapKV; see `04_cross_compressor_per_ratio.md`). Cache-state overlap is a *necessary but not sufficient* explanatory variable for the differential transfer — the relationship is monotonic in the predicted direction but non-linear.

SnapKV-native (with the question fed to the press) is not in this table because the prefill operates on a longer sequence (context + question), so a like-for-like mask comparison would need to slice to the context portion only — already done in the wrapper but not reported here. The SnapKV-native eval result is essentially identical to SnapKV-context-only (`04_cross_compressor_per_ratio.md` aggregate Δhon: +0.20 vs +0.21), so the missing overlap row is unlikely to change the story.
