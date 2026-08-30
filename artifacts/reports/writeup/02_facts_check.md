# 02 — Facts check

Each claimed number, verified against the source JSONLs and configs on disk. ✓ = exact match. ⚠ = correct value differs / minor caveat.

## Dataset

| claim | verified value | source |
|---|---|---|
| 2669 train rows | ✓ 2669 | `wc -l artifacts/train_v1.jsonl` |
| 1868 ABSTAIN train | ✓ 1868 | `Counter(label_type)` over `train_v1.jsonl` |
| 801 CONFIDENT train | ✓ 801 | same |
| 463 val rows | ✓ 463 | `wc -l artifacts/val_v1.jsonl` |
| 366 ABSTAIN val | ✓ 366 | `Counter(label_type)` over `val_v1.jsonl` |
| 97 CONFIDENT val | ✓ 97 | same |

Note for the methods section: HANDOFF.md originally said 2653 train (that was the count from an earlier run). The regenerated dataset on the final machine has 2669 — same `--seed 42`, same script, but Musique upstream has been refreshed since the original D7 run, and a few additional examples now pass the filter pipeline. Both numbers are correct in their context; the actual training used 2669.

## Labeling thresholds

| claim | verified value | source |
|---|---|---|
| τ_high = 0.8 | ✓ 0.8 | `LabelConfig.__init__` default; confirmed by labeler trace |
| τ_low  = 0.3 | ✓ 0.3 | same |

## LoRA hyperparameters

| claim | verified value | source |
|---|---|---|
| r = 16 | ✓ 16 | `configs/train/v1.yaml` `lora.r`; loaded by `make_lora_config` |
| alpha = 32 | ✓ 32 | YAML `lora.lora_alpha` |
| target = q/k/v/o (attention-only) | ✓ q_proj / k_proj / v_proj / o_proj | YAML `lora.target_modules` |
| dropout = 0.05 | ✓ 0.05 | YAML `lora.lora_dropout` (not in the original claim list, but worth knowing) |
| bias = none | ✓ none | YAML `lora.bias` |

## Compression

| claim | verified value | source |
|---|---|---|
| 6 ratios: 0.05, 0.10, 0.20, 0.30, 0.50, 0.80 | ✓ exactly that set | `set(r['ratio'] for r in val_v1.jsonl)` |
| Compression method: KVzip | ✓ kvzip | `val_v1.jsonl` `metadata.compressor` |

## Base model

| claim | verified value | source |
|---|---|---|
| Qwen2.5-7B-Instruct | ✓ Qwen/Qwen2.5-7B-Instruct | `val_v1.jsonl` `metadata.model_name`; `scripts/run_v1_full.py` `--base-model` default |

## Best checkpoint

| claim | verified value | source |
|---|---|---|
| epoch 1 of 3 | ✓ epoch 1 | `/tmp/v1_run.log`: "best epoch by val loss: 1 (val_loss=0.2165)" |
| seed 42 | ✓ seed 42 | confirmed by absence of seed flag in original run + new runs at seeds 43/44 differ |
| lr = 2e-4 | ✓ 2e-4 | log: "best LR by val loss: 2e-04 (val_loss=0.2263)" |
| val_loss = 0.2165 | ✓ 0.2165 | log |

Across the three seeds the best checkpoint differs — see `08_seed_variance.md` for the full table:

| seed | best LR | best epoch | best val_loss |
|---|---|---|---|
| 42 | 2e-4 | 1 | 0.2165 |
| 43 | 2e-4 | 1 | 0.2234 |
| **44** | **5e-5** | **2** | 0.2192 |

If the paper cites only seed 42 as "the v1 run" that's defensible; if it cites multi-seed numbers, use the seed-variance table.
