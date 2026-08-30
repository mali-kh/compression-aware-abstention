# 06 — SnapKV correct-abstentions are nearly a subset of KVzip's

Source: `artifacts/v1_eval.jsonl`, `v1_eval_snapkv.jsonl`, `v1_eval_snapkv_native.jsonl`. Computed on the full eval set (all ratios), filtering for adapter `classification == "correct"` on `gold_label == "abstain"` rows.

## Headline (paper-ready)

| set | size | of which also in KVzip's correct-abstain set |
|---|---|---|
| KVzip adapter correct-abstentions on ABSTAIN-gold | 337 | — |
| ExpAttn adapter correct-abstentions on ABSTAIN-gold | 278 | **267 / 278 = 96%** |
| SnapKV (context-only) adapter correct-abstentions | 187 | **182 / 187 = 97%** |
| SnapKV (native) adapter correct-abstentions | 160 | **156 / 160 = 98%** |

So the adapter's correct-abstentions on ExpAttn / SnapKV-ctx / SnapKV-native are nearly a strict subset of its correct-abstentions on KVzip. Other compressors do not enable the adapter to recognize abstain cases that KVzip didn't already enable; they only cause the adapter to *fail to recognize* abstain cases that KVzip did enable.

## Per-ratio breakdown of the SnapKV-context-only correct-abstentions

| ratio | n_abst_gold (SnapKV val) | SnapKV correct-abst | of those, also-correct under KVzip |
|---|---|---|---|
| 0.05 | 100 | 64 (64%) | 64 / 64 = 100% |
| 0.10 | 93 | 52 (56%) | 51 / 52 = 98% |
| 0.20 | 73 | 40 (55%) | 40 / 40 = 100% |
| 0.30 | 49 | 22 (45%) | 20 / 22 = 91% |
| 0.50 | 21 | 8 (38%) | 7 / 8 = 88% |
| 0.80 | 7 | 1 (14%) | 0 / 1 = 0% |

The 88% and 0% rows have small denominators (n=8 and n=1 respectively); the headline 97% is dominated by the lower-ratio bins where the count is large.

## Per-ratio breakdown of the gap — KVzip-correct-abstain examples that the SnapKV adapter hallucinated on

| ratio | KVzip correct-abst → SnapKV (ctx) hallucinated |
|---|---|
| 0.05 | 24 |
| 0.10 | 25 |
| 0.20 | 19 |
| 0.30 | 17 |
| 0.50 | 4 |
| 0.80 | 0 |
| **total** | **89** |

89 examples that KVzip's adapter correctly abstained on flip to adapter hallucination under SnapKV's eviction. Concentrated at low and mid ratios (24 + 25 + 19 + 17 = 85 of the 89 are at ratio ≤ 0.30) — exactly the regime where the kept-mask overlap between KVzip and SnapKV is lowest (see `05_kept_mask_overlap.md`: Jaccard 0.03 / 0.07 / 0.16 / 0.25 at those four ratios).

## Reading for the paper

This is the central evidence for the §10.7 decomposition (memo): the adapter has learned *two* behaviours and they transfer differently:

1. **Read the cache and answer if the answer is there** — transfers cleanly across compressors. Evidence: D5 cross-compressor table (ExpAttn answers Orleans/Emilia correctly, SnapKV-native answers Orleans/Emilia/conga correctly).
2. **Recognize that the cache is broken and abstain** — transfers within compressor family but not across. Evidence: the subset finding above. SnapKV does not enable abstain-recognition on examples KVzip didn't enable; it only causes failure-to-recognize on examples KVzip did enable.

The 89 figure is the cleanest single statistic for "how much of KVzip's recognition behaviour fails to generalize to SnapKV."
