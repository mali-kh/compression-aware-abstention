# Writeup pack — paste-ready facts and tables for the v1 paper

Compiled 2026-05-05 from the v1 + cross-compressor + seed-variance runs. All numbers double-checked against the source JSONLs in `artifacts/`.

| file | what's in it | section in v1_memo.md it draws from |
|---|---|---|
| `01_v1_yaml_and_actual_hparams.md` | The full `configs/train/v1.yaml` text + a table of YAML-vs-actual (one real discrepancy: LR scheduler) | §2 |
| `02_facts_check.md` | Each claimed number verified against source JSONLs | §2, §3 |
| `03_d5_three_examples_full_prose.md` | Three D5 wrong-but-specific examples with full question + gold + base response + adapter response, paper-ready | §6 (memo §3, §10.6) |
| `04_cross_compressor_per_ratio.md` | Per-ratio honest_acc + halluc + abst, four columns (KVzip, ExpAttn, SnapKV-ctx, SnapKV-native) | §10.2 |
| `05_kept_mask_overlap.md` | Jaccard + intersection-over-min cross-compressor agreement on kept tokens | §10.4 |
| `06_snapkv_subset_finding.md` | The 97/98/96% subset finding, plus per-ratio breakdown of the 89 "KVzip-correct → SnapKV-hallucinated" examples | §10.5 |
| `07_calibration_subcheck.md` | Per-compressor overcaution + conditional wrong rate at ratios 0.50 and 0.80 | §4, §10 |
| `08_seed_variance.md` | The seed-{42,43,44} side-by-side from this morning, paste-ready | §7.1 |
| `09_replicated_hallucinations.md` | Hallucination-set intersection across seeds (4 of 7 replicate exactly), with the 4 stable cases' questions | new (compute on demand) |

Reading order for paper-writing: 01 → 02 → 03 (Figure 1 candidates) → 04 (Table 1) → 05/07 → 06/08/09 (discussion-section material).
