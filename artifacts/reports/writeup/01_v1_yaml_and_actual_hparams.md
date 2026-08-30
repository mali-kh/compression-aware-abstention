# 01 — `configs/train/v1.yaml` (full paste) + YAML-vs-actual reconciliation

## Full file

```yaml
# LoRA training config — cache-state-conditional generation v1
#
# DRY-RUN values are tuned for the 0.6B base. Production columns below
# document recommended values for Qwen2.5-7B-Instruct on a single H100;
# they are NOT auto-applied — train_dry_run.py uses the dry-run section.
#
# These hyperparameters are starting points, justified inline. The first
# real training run should sweep LR (1e-4 / 5e-5 / 2e-4) before
# committing.

base_model: Qwen/Qwen3-0.6B   # dry-run only; production switches to Qwen2.5-7B-Instruct

lora:
  r: 16                         # standard LoRA rank for 7B-class instruct fine-tuning
  lora_alpha: 32                # alpha = 2*r is a common default for stable scaling
  lora_dropout: 0.05            # mild — dataset is 2.6K instances, regularization helps
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  task_type: CAUSAL_LM

optim:
  learning_rate: 1.0e-4         # LoRA-typical (10x what full fine-tune would use)
  weight_decay: 0.0
  warmup_ratio: 0.03            # ~30 steps over a 1000-step schedule
  lr_scheduler_type: cosine

training:
  per_device_train_batch_size: 2     # dry-run: small batch is fine on 0.6B
  gradient_accumulation_steps: 4     # effective batch 8
  num_train_epochs: 3
  bf16: true                          # A10 / H100 native; matches model load dtype
  gradient_checkpointing: true        # mandatory for 7B; cheap on 0.6B
  max_seq_length: 4096                # matches dataset ctx cap
  logging_steps: 1
  save_steps: 9999                    # dry-run: don't checkpoint mid-train
  seed: 42

dry_run:
  n_examples: 10                # only first 10 of train_v1.jsonl
  max_steps: 5                  # 5 optimizer steps
  output_dir: artifacts/dry_run_adapter
```

## What the v1 production run actually used

`scripts/run_v1_full.py` reads the `lora` section from this YAML (so r/alpha/dropout/bias/target_modules are honest), but everything in the `optim` and `training` sections is either overridden by CLI flags or simply not read. Confirmed by reading the script directly.

| Hyperparameter | YAML | Actual in v1 run | Source-of-truth |
|---|---|---|---|
| `lora.r` | 16 | **16** | YAML, read by `make_lora_config` |
| `lora.lora_alpha` | 32 | **32** | YAML |
| `lora.lora_dropout` | 0.05 | **0.05** | YAML |
| `lora.bias` | none | **none** | YAML |
| `lora.target_modules` | q/k/v/o_proj | **q/k/v/o_proj** | YAML |
| `optim.learning_rate` | 1e-4 | **2e-4** (sweep choice for seed 42) | CLI sweep `--lrs 5e-5,1e-4,2e-4`; per-seed: 42→2e-4, 43→2e-4, 44→5e-5 |
| `optim.weight_decay` | 0.0 | **0.0** | YAML, passed to AdamW |
| `optim.warmup_ratio` | 0.03 | **0.0 (no warmup)** | YAML value is **NOT used** — see discrepancy below |
| `optim.lr_scheduler_type` | cosine | **constant (no scheduler)** | YAML value is **NOT used** — see discrepancy below |
| `training.per_device_train_batch_size` | 2 | **1** | DataLoader hardcodes `batch_size=1` |
| `training.gradient_accumulation_steps` | 4 | **16** | CLI `--accum 16` (default) |
| Effective batch | YAML implies 8 | **16** | (1 × 16 grad accum) |
| `training.num_train_epochs` | 3 | **3** | CLI `--epochs 3` (default), matches YAML |
| `training.bf16` | true | **true** (bfloat16) | hardcoded in `fresh_base_model` |
| `training.gradient_checkpointing` | true | **true (use_reentrant=False)** | hardcoded after `get_peft_model` |
| `training.max_seq_length` | 4096 | **4096** | CLI `--max-seq-length 4096` (default), matches YAML |
| `training.seed` | 42 | **42 / 43 / 44** | CLI `--seed N` (added in commit `f65fe20`) |
| `attn_implementation` | (not in YAML) | **flash_attention_2** | hardcoded in `fresh_base_model` (HANDOFF §6.6 — Qwen2.5 + sdpa OOMs) |

### One discrepancy worth naming explicitly in the paper

The YAML declares `lr_scheduler_type: cosine` and `warmup_ratio: 0.03`, but `scripts/run_v1_full.py` constructs a plain `torch.optim.AdamW` and never wraps it in a scheduler. **The actual v1 run used a constant learning rate with no warmup** (whichever LR the sweep picked, applied flat for all 3 epochs). Do not write "cosine schedule with 3% warmup" in the methods section — that's what the YAML *says* but not what the code *did*.

The `per_device_train_batch_size: 2` field in the YAML is also unused; `DataLoader(batch_size=1)` is hardcoded. Effective batch is always 16 (1 microbatch × 16 grad-accum steps).

## Other configs referenced from the v1 pipeline

`configs/labeling.yaml`:
```yaml
label_construction:
  tau_high: 0.8
  tau_low: 0.3
  drop_uncertain_zone: True
  assume_all_spans_required: True
  enable_flagged_partial: True   # NOTE: yaml has this True, but code uses LabelConfig() defaults (False)
```

The v1 dataset construction (`scripts/build_training_set.py`) instantiates `LabelConfig()` with no kwargs and never reads `configs/labeling.yaml`. So the actual labeling used the dataclass defaults: `tau_high=0.8, tau_low=0.3, drop_uncertain_zone=True, assume_all_spans_required=True, enable_flagged_partial=False`. The empirical confirmation is that `train_v1.jsonl` and `val_v1.jsonl` contain zero `flagged_partial` rows.
