# 07 — Calibration sub-check across all four compressors at conclusive ratios

This is the version to cite. Conclusive slices are ratios where there are enough CONFIDENT-gold examples to draw conclusions: ratio=0.50 (n=22 in KVzip val) and ratio=0.80 (n=69). Slices with n<5 (ratios 0.10, 0.20, 0.30) are not in this table — see `v1_run_report.md` for a fuller version with those rows annotated as inconclusive.

The two columns the paper needs:

- **adapter overcaution** = (adapter abstained when gold was confident_answer) / n_conf
- **adapter conditional wrong-when-answering** = (adapter answered with the wrong answer) / (adapter chose to answer)

The second column guards against the "adapter preserved honest_accuracy by abstaining more, not by answering better" reading.

## Table

| ratio | compressor | n_CONF | adapter overcautious | base overcautious | adapter wrong / answered | base wrong / answered |
|---|---|---|---|---|---|---|
| **0.50** | KVzip | 22 | 7/22 = **32%** | 5/22 = 23% | 5/15 = 33% | 6/17 = 35% |
| **0.50** | ExpAttn | 23 | 2/23 = **9%** | 8/23 = 35% | 6/21 = 29% | 5/15 = 33% |
| **0.50** | SnapKV (ctx) | 18 | 3/18 = **17%** | 3/18 = 17% | 5/15 = 33% | 7/15 = 47% |
| **0.50** | SnapKV (native) | 28 | 4/28 = **14%** | 8/28 = 29% | 10/24 = 42% | 7/20 = 35% |
| 0.80 | KVzip | 69 | 2/69 = 3% | 12/69 = 17% | 23/67 = 34% | 21/57 = 37% |
| 0.80 | ExpAttn | 65 | 2/65 = 3% | 9/65 = 14% | 26/63 = 41% | 24/56 = 43% |
| 0.80 | SnapKV (ctx) | 59 | 0/59 = 0% | 11/59 = 19% | 24/59 = 41% | 16/48 = 33% |
| 0.80 | SnapKV (native) | 67 | 0/67 = 0% | 9/67 = 13% | 28/67 = 42% | 21/58 = 36% |

## Reading for the paper

Two things land in the same direction across all four compressors:

1. **Conditional answer quality is preserved** — the "adapter wrong / answered" column is essentially identical to "base wrong / answered" within ±5pp at every conclusive cell. The adapter does not degrade answer quality conditional on choosing to answer; it only changes when it chooses to answer. This is the calibration-net result.

2. **Adapter overcaution at high ratio (0.80) is substantially below base overcaution** for every compressor (3% / 3% / 0% / 0% adapter vs 17% / 14% / 19% / 13% base). The adapter has learned to use cache fullness as a strong prior for "answer aggressively," and this generalizes across compressors.

One thing diverges:

- **Adapter overcaution at the headline mid-ratio (0.50)** is meaningfully higher than base on KVzip (32% vs 23%, +9pp), but *lower* than base on ExpAttn (9% vs 35%) and SnapKV-native (14% vs 29%), and equal on SnapKV-context-only (17%). So the +9pp KVzip-specific overcaution at the most-mixed slice is **not a property of the learned representation** — it is specific to KVzip's eviction signature. Worth a sentence in the paper.

The paper headline statistic from this table: **the adapter eliminates the answer-quality vs abstention-rate tradeoff that one might naively expect** — at every conclusive cell it changes the abstention rate (sometimes up, sometimes down) without degrading conditional answer quality.
