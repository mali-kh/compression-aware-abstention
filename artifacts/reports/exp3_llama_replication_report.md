# Exp 3 — Llama-3.1-8B-Instruct replication

_v1 was trained on Qwen2.5-7B-Instruct. This experiment retrains the same v1 pipeline (LoRA r=16/α=32 on attention-only, KVzip mask provenance, 30/70 class balance, 3-LR sweep + 3-epoch best-by-val) using `NousResearch/Meta-Llama-3.1-8B-Instruct` as the base model. Goal: confirm the result is not Qwen-specific._

## Setup

- **Base model**: NousResearch/Meta-Llama-3.1-8B-Instruct (open mirror of Meta-Llama-3.1-8B-Instruct, identical tokenizer; D6 confirmed 100% tight-span localization on this tokenizer)
- **Compressor**: KVzip vendored at commit `5d8472975`. Smoke-tested on Llama-3.1 before launching the full experiment — KVzip's "Replace llama attention with KVzip" path activates correctly; mask alignment confirmed (`mask.mean()=0.2996` at requested 0.30 on a 751-token synthetic context).
- **Dataset**: regenerated `train_v1_llama.jsonl` and `val_v1_llama.jsonl` from the same Musique train pool (1000 train + 100 val ids, deterministic with seed=42), using Llama-3.1 as the KVzip mask-generator. Same labeler config as v1 (τ_high=0.8, τ_low=0.3, FLAGGED_PARTIAL off).
- **Training**: same hyperparameters as v1 (`scripts/run_v1_full.py --base-model Llama-3.1 --seed 42 --output-suffix _llama_seed42`).

## Llama dataset vs Qwen v1 dataset

| | Qwen v1 | Llama |
|---|---|---|
| train rows (post-DROP, post-balance) | 2669 | 3640 |
| train ABST / CONF | 1868 / 801 | 2548 / 1092 |
| val rows (post-DROP, no balance) | 463 | 465 |
| val natural CONF fraction | 21.0% | **24.3%** |
| train minority-frac after balance | 30.0% | 30.0% |

Llama's KVzip retains slightly more answer-bearing tokens than Qwen's at the same ratios — natural CONFIDENT-gold rate on val is 24% vs Qwen's 21%. Same model architecture family, different attention patterns, different KV importance scoring → different label distribution. Rebalancing to 30/70 yields a larger train set (3640 vs 2669) because there are more CONFIDENT instances to set the floor against.

## Training trajectory

| stage | LR / epoch | train_loss | val_loss | wall |
|---|---|---|---|---|
| sweep | 5e-5 | 1.131 | **0.1015 ← best** | 814s |
| sweep | 1e-4 | 0.733 | 0.1175 | 813s |
| sweep | 2e-4 | 0.470 | 0.1390 | 815s |
| epoch 1 (lr=5e-5) | 1 of 3 | 0.3648 | 0.0977 | 807s |
| **epoch 2 (lr=5e-5)** | **2 of 3 ← best** | **0.0495** | **0.0945** | **817s** |
| epoch 3 (lr=5e-5) | 3 of 3 | 0.0185 | 0.1225 | 814s |

Best LR = 5e-5 (Qwen seeds 42 and 43 picked 2e-4; Qwen seed 44 also picked 5e-5). Best epoch = 2 (lower LR overfits later — same pattern as Qwen seed 44). Llama's val loss plateaus much lower (~0.09) than Qwen's (~0.22), partly due to the different label distribution — different baseline loss is expected.

## Headline result — does v1 replicate on Llama?

### Per-ratio Δhonest_accuracy (adapter − base) side-by-side

| ratio | n (Qwen / Llama) | gold C/A (Qwen) | gold C/A (Llama) | Qwen Δhon | Llama Δhon |
|---|---|---|---|---|---|
| 0.05 | 100 / 99 | 0/100 | 0/99 | +0.85 | +0.84 |
| 0.10 | 95 / 94 | 1/94 | 0/94 | +0.72 | +0.82 |
| 0.20 | 82 / 84 | 2/80 | 0/84 | +0.56 | +0.83 |
| 0.30 | 65 / 59 | 3/62 | 1/58 | +0.43 | +0.73 |
| **0.50** | **48 / 50** | **22/26** | **35/15** | **+0.21** | **+0.20** |
| 0.80 | 73 / 79 | 69/4 | 77/2 | +0.10 | +0.19 |
| **AGG** | **463 / 465** | **97/366** | **113/352** | **+0.53** | **+0.64** |

### Adapter abstention rate per ratio (the load-bearing monotone claim)

| ratio | Qwen | Llama |
|---|---|---|
| 0.05 | 1.00 | 1.00 |
| 0.10 | 1.00 | 1.00 |
| 0.20 | 0.89 | 1.00 |
| 0.30 | 0.83 | 0.85 |
| 0.50 | 0.58 | 0.20 |
| 0.80 | 0.03 | 0.05 |

**Both monotone decreasing.** Llama and Qwen both learn cache-state-conditional abstention behavior. The qualitative *shape* is preserved.

The notable difference is at ratio=0.50: Llama abstains far less (0.20 vs Qwen's 0.58). This shifts the decision boundary toward "answer" earlier than Qwen does. Combined with Llama's gold-mix at 0.50 being CONF-heavy (35 CONF / 15 ABST vs Qwen's 22/26), Llama at this slice is in a different regime — most examples are CONF-gold and the model correctly chooses to answer most of them. The Δhon at 0.50 ends up identical (+0.20 vs +0.21).

### Aggregate hallucination drop

| | base hallucinated | adapter hallucinated | Δ |
|---|---|---|---|
| Qwen | 232 | 7 | **−225** |
| Llama | 228 | 10 | **−218** |

Near-identical. Both ~97% hallucination reduction.

### Aggregate correct-answer gain

| | base correct | adapter correct | Δ |
|---|---|---|---|
| Qwen | 147 | 391 | +244 |
| Llama | 111 | 409 | **+298** |

Llama's adapter actually produces *more* correct answers in absolute terms.

## Reading

The v1 result is not Qwen-specific. The cache-state-conditional behavior — monotonic abstention curve, ~97% hallucination drop on val, large positive Δhon at every ratio — replicates cleanly on Llama-3.1-8B-Instruct with the same training pipeline. Llama's abstention boundary sits at a slightly different place than Qwen's (sharper transition between "always abstain" and "always answer" at ratio 0.50), but the underlying behavior is the same.

The result is robust to (a) base model family (Qwen → Llama), (b) base model size (7B → 8B), (c) tokenizer (Qwen 151k → Llama 128k vocab). Three of the four "single-X" limitations the v1 memo listed are now partially addressed: single-base-model is no longer load-bearing.

## Files

- `artifacts/train_v1_llama.jsonl` — Llama-regenerated train (3640 instances)
- `artifacts/val_v1_llama.jsonl` — Llama-regenerated val (465 instances)
- `artifacts/train_v1_llama_report.md` — D7-style dataset report
- `artifacts/v1_lora_best_llama_seed42/` — best Llama adapter (epoch 2, lr=5e-5)
- `artifacts/v1_lora_epoch{1,2,3}_llama_seed42/` — per-epoch checkpoints
- `artifacts/v1_lora_sweep_lr*_llama_seed42/` — LR-sweep adapters
- `artifacts/v1_eval_llama_seed42.jsonl` — paired base+adapter classifications
- `artifacts/v1_d5_qualitative_llama_seed42.md` — Phase 4 output (will be empty for this run; ran with `--d5-jsonl artifacts/_empty_d5.jsonl` since D5 fragments are Qwen-derived)
- `artifacts/v1_run_report_llama_seed42.md` — Phase 5 auto-report
- This file

## Footnote on infrastructure issue

During Phase 5, `run_v1_full.py` wrote to hardcoded paths `artifacts/v1_d5_qualitative.md` and `artifacts/v1_run_report.md` without the `--output-suffix`, **clobbering the original Qwen v1 artifacts**. The clobbering was reversed by re-running `scripts/finalize_v1_report.py` (all the inputs — `fragment_recoverability.jsonl`, `v1_eval.jsonl`, `v1_lora_best/`, `/tmp/v1_run.log` — were intact). A patch (committed separately) propagates `out_suffix` into both Phase 4 and Phase 5 file paths so this can't happen again.
