# Overnight summary — 2026-05-04 → 2026-05-05

_One-page index for your morning review. Read this first; then dive into the linked reports._

## TL;DR by experiment

| Exp | Status | Headline | Report |
|---|---|---|---|
| 1 — Held-out test split | ✅ done | Headline replicates: aggregate Δhon **+0.63** on test_held_out (n=100) vs v1_full's +0.53; 232 → 1 hallucinations on 85 ABSTAIN-gold rows in test. | `artifacts/exp1_test_split_report.md` |
| 2a — Mixture dataset construction | ⚠️ **BLOCKED** | Per-(compressor, ratio) 30/70 floor starves low-ratio strata; KVzip/ExpAttn at 0.05/0.10 collapse to **3-23 instances**. Six options surfaced; my recommendation is Option B. | `artifacts/EXP2A_BLOCKED.md` |
| 2b/c — Mixture training + eval | ⏸️ blocked on 2a | Not run — waiting on your option pick. | — |
| 3 — Llama-3.1-8B replication | ✅ done | Clean positive: **aggregate Δhon +0.64** (vs Qwen +0.53), monotone abstention curve preserved, 228 → 10 hallucinations. v1 result is not Qwen-specific. | `artifacts/exp3_llama_replication_report.md` |
| 4 — RULER multi-key NIAH | ✅ done (with caveat) | At 16K (4× v1's training max): abstention curve **monotone** (67% → 0%), conditional answer quality **86–94% correct** on confident-gold, aggregate hallucination halves at every length. **32K/64K source unavailable** on HF; reported lengths are 4K/8K/16K. | `artifacts/exp4_ruler_long_context_report.md` |
| 2 seed 43 | ⏸️ deferred | Only triggered if Exp 2 seed 42 produces meaningful improvement. Currently blocked. | — |

## Decisions waiting for you

1. **Exp 2a path forward.** Six options in `EXP2A_BLOCKED.md`. My recommendation is **Option B** (skip per-stratum balance at low ratios; mid/high keep 30/70). Whichever you pick, I can re-run Stage 4 (`build_mixture_merge.py`) without re-running the GPU stages — the three raw JSONLs (`_mixture_raw_kvzip.jsonl`, `_mixture_raw_expected_attention.jsonl`, `_mixture_raw_snapkv.jsonl`) are on disk.
2. **Exp 2 seed 43**. Conditional on Exp 2 seed 42 result. Skip until 2a is unblocked + 2b/c land.

## Things to read in priority order

1. `exp1_test_split_report.md` — does the headline replicate on truly held-out data? Yes.
2. `exp3_llama_replication_report.md` — is the result Qwen-specific? No — Llama replicates cleanly.
3. `exp4_ruler_long_context_report.md` — does the method generalize to long context retrieval? Yes at 16K, monotone abstention curve preserved.
4. `EXP2A_BLOCKED.md` — the blocker for the v2-defining experiment. Surface for your decision.

## Infrastructure issues encountered (and fixed)

Three small bugs surfaced while running the experiments. All committed as separate commits before the experimental commits, so the diagnostic boundaries are clean.

1. **`build_training_set.py` `Path.relative_to(ROOT)` crash** when `--out-train`/`--out-val` are relative paths not under ROOT. Fixed by wrapping the print-time `relative_to` in try/except + `resolve()`. Caught by Exp 3's first launch attempt; data was written before the crash, just the trailing print failed. Commit `3b3da53`.
2. **`run_v1_full.py` Phase 4/5 paths missed `--output-suffix`** — wrote `artifacts/v1_d5_qualitative.md` and `artifacts/v1_run_report.md` without the suffix, **clobbering the original v1 artifacts** during Exp 3. Restored via `finalize_v1_report.py` (all inputs intact). Suffix now propagates to both Phase 4/5 paths. Commit `b06522b`.
3. **Exp 4 RULER `answer` field is a list**, not a string — initial smoke test had all `gold_state="unknown"`. Fixed by unpacking `answer[0]` and switching the needle-locator from token-id substring to decoded-text substring. Re-smoked clean before launching the full run.

The two run_v1_full.py extensions earlier today (`--seed`/`--skip-base-eval` in `f65fe20`, `--output-suffix` in `67c6a1b`) plus the build_training_set extension (`--out-train` flags in `3c21ecf`) were prerequisite plumbing, all committed before the experimental commits.

## Compute used

- Exp 1: ~13 min (363 + 100 base eval; 363 + 100 adapter eval)
- Exp 2a: ~25 min (KVzip + ExpAttn + SnapKV-ctx mask gen for 1100 ids × 6 ratios)
- Exp 3: ~85 min (regen Llama train+val + LR sweep + 3-epoch + base+adapter eval)
- Exp 4: ~52 min (3 lengths × 100 examples × 6 ratios × 2 models)
- **Total: ~3 hours active compute**, plus ~30 min of model-load/setup time spread across stages.

## Files added by overnight work

Scripts (all committed):
- `scripts/exp1_test_split.py` — id-grouped val split
- `scripts/build_mixture_kvzip.py` — KVzip phase of mixture build
- `scripts/build_mixture_kvpress.py` — kvpress phase (parameterized by `--press`)
- `scripts/build_mixture_merge.py` — per-stratum balance + report
- `scripts/exp4_ruler_eval.py` — RULER NIAH eval

Reports (all committed):
- `artifacts/exp1_test_split_report.md`
- `artifacts/exp2_dataset_report.md`
- `artifacts/EXP2A_BLOCKED.md`
- `artifacts/exp3_llama_replication_report.md`
- `artifacts/exp4_ruler_long_context_report.md`
- `artifacts/train_v1_llama_report.md` (auto-generated by build_training_set.py)
- `artifacts/OVERNIGHT_SUMMARY.md` — this file

Data (gitignored, on disk):
- `artifacts/val_v1_for_selection.jsonl`, `test_held_out.jsonl` (Exp 1)
- `artifacts/_mixture_raw_kvzip.jsonl`, `_mixture_raw_expected_attention.jsonl`, `_mixture_raw_snapkv.jsonl`, `train_mixture_v1.jsonl`, `val_mixture_v1.jsonl` (Exp 2a — train_mixture is the under-target 5035-row version; resumable)
- `artifacts/v1_eval_for_selection.jsonl`, `v1_eval_test_held_out.jsonl` (Exp 1)
- `artifacts/train_v1_llama.jsonl`, `val_v1_llama.jsonl`, `v1_eval_llama_seed42.jsonl` (Exp 3)
- `artifacts/v1_lora_best_llama_seed42/`, `v1_lora_epoch{1,2,3}_llama_seed42/`, `v1_lora_sweep_lr*_llama_seed42/` (Exp 3 checkpoints)
- `artifacts/exp4_ruler_eval.jsonl` (Exp 4)
