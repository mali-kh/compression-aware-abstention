# D8 — LoRA Dry Run Report

_Auto-generated summary for review BEFORE authorizing a real training run._

## 1. Infrastructure checks — all PASS

| # | Check | Result | Notes |
|---|---|---|---|
| 1 | `tokenizer.padding_side='left'` set explicitly | ✅ | default was `'right'`; we override before any batching |
| 2 | Label mask covers exactly `target_output + eos`, nothing else | ✅ | 14 unmasked tokens decode to the literal abstention string + `<|im_end|>` (not just the final word — common silent bug ruled out) |
| 3 | `kept_mask` materially changes forward logits | ✅ | on a ratio=0.05 example (5% retention), `max\|logit(applied)−logit(passed)\|` at the first-target position = **9.66** — the mask is unambiguously affecting the forward pass, not silently dropped |
| 4 | 5 LoRA optim steps run to completion | ✅ | losses: 4.68 → 5.56 → 4.93 → 3.26 → 2.80; no NaN/Inf; gradient checkpointing enabled |
| 5 | Adapter save → reload → forward → finite logits | ✅ | round-trip clean; PEFT adapter dir at `artifacts/dry_run_adapter/` |

**No infrastructure issues found.** Loss curve trends down (1.88-nat drop in 5 steps on cycled data); training signal is alive.

## 2. Wall-clock estimate for real training (Qwen2.5-7B-Instruct)

Dataset: `train_v1.jsonl` = 2653 instances (1857 ABST + 796 CONF).

Effective config: `batch_size=1`, `grad_accum=16`, so 16 examples per optimizer step → ~166 steps/epoch.

| Hardware | Step time (est.) | 3 epochs (~498 steps) | Notes |
|---|---|---|---|
| 1× H100 80GB | ~1.5–2 s | **~12–17 min** | bf16 + gradient checkpointing; flash-attn 2 |
| 1× A10 23GB | ~5–7 s | **~45–60 min** | tight on VRAM; will need batch=1, grad_accum≥16 |
| 1× A100 40GB | ~3–4 s | **~25–35 min** | comfortable headroom |

Add ~2 min model load + ~5 min eval time per evaluation. If the real run sweeps LR (3 short 1-epoch runs at 5e-5 / 1e-4 / 2e-4) before the 3-epoch run, plan ~3× the per-epoch time for the sweep + ~3 epochs at the chosen LR.

**Total budget ask for v1 real run on H100: ~1 hour. On A10: ~3 hours.**

## 3. Hyperparameter recommendations with justification

### LoRA

| Param | Value | Justification |
|---|---|---|
| `r` | 16 | Standard for 7B instruct fine-tuning at this dataset scale. `r=8` often suffices for stylistic adaptation; `r=16` gives more headroom for the cache-state-conditional logic, which is a real behavioral shift not just a style nudge. `r=32+` risks overfitting on 2.6K examples. |
| `lora_alpha` | 32 | `alpha = 2r` — standard scaling, keeps effective LR consistent with `r=16`. |
| `lora_dropout` | 0.05 | Mild — dataset is small (2.6K), so some regularization helps; aggressive dropout (0.1+) starves the small minority class. |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` | Attention-only. The cache-state-conditional behavior is fundamentally an *attention-pattern decision* (does the model attend to surviving evidence to commit, or not). Adding MLP modules (`gate_proj, up_proj, down_proj`) doubles trainable params for marginal gain on this task; recommend trying only if v1 underfits. |
| `bias` | `none` | LoRA-standard; biases are hardly worth the extra memory. |

### Optimizer / schedule

| Param | Value | Justification |
|---|---|---|
| `learning_rate` | 1e-4 | LoRA-typical (~10× full fine-tune). **Sweep before committing**: 5e-5 / 1e-4 / 2e-4 in 1-epoch trials, pick by val loss. |
| `weight_decay` | 0.0 | Standard for LoRA; the adapter mass is small enough that L2 doesn't help. |
| `warmup_ratio` | 0.03 | ~15 steps at 498-step total — long enough to avoid early NaN, short enough to not waste budget. |
| `lr_scheduler_type` | `cosine` | Cosine is the standard for LoRA fine-tuning at this scale; linear is fine but cosine ends slightly softer. |

### Training

| Param | Value | Justification |
|---|---|---|
| `per_device_train_batch_size` | 1 | VRAM constraint at seq=4096 on A10/H100 with bf16. Batch>1 needs longer-context examples to be packed/truncated, complicating the kept_mask alignment. |
| `gradient_accumulation_steps` | 16 | Effective batch 16 — small enough for stable LoRA updates, large enough to smooth per-step loss. |
| `num_train_epochs` | 3 | 2.6K examples × 3 = 7.96K examples seen ≈ 498 optimizer steps. Empirically this is the sweet spot for LoRA on 1–10K-instance datasets. More epochs risk memorization. |
| `bf16` | true | A10 / H100 native; matches model load dtype; halves memory vs fp32. |
| `gradient_checkpointing` | true | Mandatory on A10 at 7B + seq=4096; cheap on H100 too. |
| `max_seq_length` | 4096 | Matches `train_v1.jsonl` cap. Most examples are 1500–3000 tokens. |
| `seed` | 42 | Deterministic |

### Eval gates BEFORE accepting v1 (per the project rule that per-ratio breakdowns are the headline)

Run on `val_v1.jsonl` (463 instances, natural distribution). Report:

1. **Per-ratio honest_accuracy** (correct CONFIDENT + correct ABSTAIN / total_at_ratio) at each of {0.05, 0.10, 0.20, 0.30, 0.50, 0.80}. The mid-ratios (0.20, 0.30, 0.50) are the headline.
2. **Confusion matrix per ratio**: confident-correct / confident-wrong / abstained-when-shouldn't / abstained-when-should.
3. **Calibration curve**: fraction abstained vs ratio. Trained model should abstain monotonically with decreasing ratio. A flat curve = the model isn't using the cache state.
4. **Span-level retention vs label**: scatter plot of mean_span_retention against the model's output (CONFIDENT/ABSTAIN). Healthy training puts the boundary near the τ_low / τ_high zone.

Aggregate honest_accuracy alone is misleading — see the per-ratio reporting rule above.

## 4. Recommendation

**All gates clear; v1 real training is technically authorized.** Final go/no-go was deferred; bring this back for review before kicking off the run.

If green-lit, suggest:

```
1. LR sweep (1 epoch each at 5e-5, 1e-4, 2e-4)        ~15 min on H100
2. Pick best by val loss
3. Full 3-epoch run on that LR                         ~17 min on H100
4. Eval on val_v1.jsonl with per-ratio breakdowns      ~5 min
                                                       ----------
                                                       ~40 min total
```

The artifact will be `artifacts/v1_lora/` containing the adapter + the eval report. Recommend NOT overwriting; tag this as v1 even if v2 trains immediately after.
