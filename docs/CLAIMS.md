# Paper claim to artifact map

Each numbered result in the paper and the file it is computed from. Every entry
in the "verified" column is recomputed from raw per-example outputs by
`scripts/verify_paper_claims.py`, which reproduces all 202 checks.

Evaluation files use one of two shapes. Prompt-style paired files nest the two
conditions under `base` and `adapter` keys; compressed-cache and controlled-
deletion files are flat, one record per example. Both carry `id`, `ratio`,
`gold_label`, `classification`, and `model_abstained`.

Honest accuracy is `classification == "correct"`, which covers both a
`CONFIDENT`-gold example answered correctly and an `ABSTAIN`-gold example
refused correctly.

## Main body

| Paper location | Claim | Source file(s) | Verified |
|---|---|---|---|
| Abstract, C3 | 97% hallucination reduction (232 -> 7) | `v1_eval.jsonl` | yes |
| Abstract, C5 | 6-22x lift under compressed-cache decoding | `v1_eval_base_compressed_kvzip.jsonl`, `v1_eval_mixture_cc_kvzip.jsonl` | yes |
| Table 1 | Per-ratio honest accuracy, 3-seed mean | `v1_eval.jsonl`, `v1_eval_seed43.jsonl`, `v1_eval_seed44.jsonl` | yes |
| Sec 5.1 | Answerability preservation, ratio 1.00-1.36 | `logs/block6_answerability.log` + eval files | yes |
| Sec 5.2 | Best ratio-only threshold reaches 0.93 aggregate | `reports/expB1_ratio_baseline_report.md` | — |
| Sec 5.2 | P2 over-abstains on 49-51% of `CONFIDENT`-gold at r=0.80 | `v1_eval_prompt_p2_*.jsonl` | yes |
| Sec 5.2 | Fragment baseline catches 13/89 (15%) | `reports/expB3_fragment_baseline_report.md` | — |
| Table 2 | Honest accuracy at r=0.50, four compressors | `v1_eval_*.jsonl`, `v1_eval_mixture_*.jsonl`, `v1_eval_prompt_p2_*.jsonl` | yes |
| Sec 5.3 | Kept-token overlap 75% vs 62% at r=0.50 | `reports/writeup/05_kept_mask_overlap.md` | — |
| Sec 5.3 | Rescue analysis, 82 of 89 | `reports/exp2_mixture_training_report.md` | — |
| Table 3 | v1 (seed 42) vs mixture (2-seed mean) | `v1_eval_*.jsonl`, `v1_eval_mixture_seed43_*.jsonl` | yes |
| Sec 5.4 | Random-mask probe | `v1_eval_random_kvzip.jsonl`, `v1_eval_random_snapkv.jsonl` | yes |
| Table 4 | Compressed-cache honest accuracy | `v1_eval_base_compressed_kvzip.jsonl`, `v1_eval_compressed_kvzip.jsonl`, `v1_eval_mixture_compressed_kvzip.jsonl`, `v1_eval_mixture_cc_kvzip.jsonl` | yes |
| Sec 5.5 | mix-cc vs mix-ps tie at r=0.80 (p = 0.80) | same as Table 4 | yes |
| Sec 5.6 | Controlled-deletion mechanism tests | `block10a_*.jsonl`, `block10b_*.jsonl`, `block10c_*.jsonl` | yes |
| Sec 5.7 | RULER, 36-43pp hallucination drop | `exp4_ruler_eval.jsonl`, `reports/exp4_ruler_long_context_report.md` | — |
| Sec 5.7 | Llama replication, +0.64 and 96% reduction | `v1_eval_llama_seed42.jsonl` | yes |

## Appendices

| Appendix | Claim | Source file(s) | Verified |
|---|---|---|---|
| A | Label entropy, paragraph vs tight-span | `reports/ratio_sweep_report.md`, `reports/ratio_sweep_report_tight_only.md` | — |
| C | Calibration sub-checks | `reports/writeup/07_calibration_subcheck.md` | — |
| D | Per-ratio cross-compressor transfer (seed 42) | `reports/writeup/04_cross_compressor_per_ratio.md` | — |
| E | Mixture dataset sizes and seed robustness | `reports/exp2_dataset_report.md`, `reports/exp2_seed_spread_report.md` | — |
| F | Held-out test split | `v1_eval_test_held_out.jsonl`, `v1_eval_for_selection.jsonl` | yes |
| G | Llama-3.1-8B replication | `v1_eval_llama_seed42.jsonl` | yes |
| I | Qualitative examples | `reports/writeup/03_d5_three_examples_full_prose.md` | — |
| J | Pre-training fragment-recoverability check (0/30) | `fragment_recoverability.jsonl` | — |
| K | McNemar and paired bootstrap tables | `v1_eval_mixture_*.jsonl`, `v1_eval_prompt_p2_*.jsonl` | yes |
| L | Answerability preservation and Wilson intervals | eval files + `logs/block6_answerability.log` | yes |
| M | Compressed-cache training details | `logs/block36_pivot.log`, `v1_eval_compressed_trained_kvzip.jsonl` | yes |
| N | LLM-judge label validation (kappa = 0.61) | `block4_judge_log.jsonl`, `block4_sample.jsonl` | yes |
| O | Controlled-deletion experiments 10A/10B/10C | `block10a_*.jsonl`, `block10b_*.jsonl`, `block10c_*.jsonl` | yes |
| P | Random-mask training-signal probe | `v1_eval_random_*.jsonl` | yes |
| Q | RULER long-context generalization | `exp4_ruler_eval.jsonl` | — |

Rows marked "—" are backed by a committed report or log rather than by an
automated recomputation. Two of them warrant a note:

- **Appendix A** is a pre-training diagnostic run on a 50-example pool with a
  small mask generator (Qwen3-0.6B), before the production pipeline was built.
  The appendix caption states this. The production pipeline generates masks with
  the target base model itself.
- **Appendix D** reports seed 42 only, which is why its KVzip aggregate (+0.53)
  differs slightly from Table 1's three-seed mean (+0.54).

## Adapter to result map

| Adapter | Paper name | Used for |
|---|---|---|
| `v1_lora_best` | v1 / v1-ps (seed 42) | Table 1, Table 2, Table 3, Appendices D, F, K, L |
| `v1_lora_best_seed43`, `v1_lora_best_seed44` | v1 seeds 43, 44 | Table 1 three-seed mean |
| `v1_lora_best_mixture_seed42` | mix / mix-ps (seed 42) | Table 2, Table 3, Table 4, Appendices K, L, O |
| `v1_lora_best_mixture_seed43` | mix (seed 43) | Table 3 two-seed mean, Appendix E |
| `v1_lora_mixture_cc_lr2e-4_seed42_best` | mix-cc (seed 42) | Table 4, Appendices L, M, O |
| `v1_lora_mixture_cc_lr2e-4_seed43_best` | mix-cc (seed 43) | Appendix M per-seed results |
| `v1_lora_best_compressed_seed42` | single-compressor cc-trained | Appendix M ablation |
| `v1_lora_best_random_seed42` | random-mask adapter | Section 5.4, Appendix P |
| `v1_lora_best_llama_seed42` | Llama replication | Section 5.7, Appendix G |
