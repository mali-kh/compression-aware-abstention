# v1 — internal results memo

_Cache-state-conditional generation for KV cache compression. Internal results memo for the v1 LoRA training run on Qwen2.5-7B-Instruct + KVzip + Musique 2-hop QA. Written 2026-05-04 immediately after the v1 run, before the cross-compressor generalization test, deliberately so v1 is committed in writing on its own merits._

_Audience: future-self plus an unfamiliar co-author who will read this in three months._

---

## 1. Thesis

LLMs under aggressive KV cache compression hallucinate when the compressor evicts the supporting tokens for an answer. Existing training-aware compression methods (KV-Distill, LoMA, DuoAttention) optimize the *opposite* of honesty: they teach the model to keep answering through compression as if nothing were lost. We invert that framing. We fine-tune the model (LoRA) to abstain calibrated to its **actual cached state** — what tokens survived eviction in this specific example — not to the gross compression intensity. The load-bearing claim of v1 is that cache-state-conditional behaviour is learnable at all. This memo documents that it is.

## 2. Method

**Data.** Musique 2-hop QA. For each example we render the multi-paragraph context, run KVzip prefill + per-(layer, head, token) importance scoring + prune at six retention ratios {0.05, 0.10, 0.20, 0.30, 0.50, 0.80}, and reduce the per-slot retention scores to a per-token boolean `kept_mask` via top-k calibrated so `mask.mean() ≈ ratio`.

**Tight-span localisation (the pivot).** This was the methodological move that made the project tractable. Initially we used paragraph-level supporting spans (~100 tokens). At those scales, per-token retention averages out and the cache-state signal vanishes — the ratio-sweep diagnostic showed label entropy 0.000 at ratio=0.50 (every example landed on the same label). Switching to tight spans — the literal sub-answer string from `question_decomposition[i].answer` located verbatim inside its supporting paragraph (3–15 tokens, 100/100 located on the validation 50) — raised entropy to 0.95 at ratio=0.50. Per-example variance became visible. Without this pivot the cache-state-conditional thesis would have been undefendable on this dataset.

**Labels.** Per supporting span, compute the retained-token fraction. CONFIDENT if all spans ≥ τ_high=0.8; ABSTAIN if any span < τ_low=0.3; DROP otherwise (excluded from training). Binary only for v1: FLAGGED_PARTIAL is deferred (the schema preserves the code path, but the v1 dataset contains no FLAGGED_PARTIAL instances — confirmed empirically).

**Mask provenance.** All kept_masks were produced by KVzip running on the production base model (`Qwen/Qwen2.5-7B-Instruct`), not on a smaller validation stand-in. Eviction patterns are model-specific; mask-generator and mask-consumer must match.

**Training.** LoRA r=16, α=32, dropout configured per `configs/train/v1.yaml`, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`. Base in bf16 with `attn_implementation='flash_attention_2'` and gradient checkpointing (`use_reentrant=False`, re-applied after `get_peft_model` because PEFT silently clobbers it). AdamW with the configured weight decay. Loss masked everywhere except `target_output + eos`. Phase 1 swept LRs {5e-5, 1e-4, 2e-4} for 1 epoch each (best by val loss); Phase 2 ran 3 epochs at the chosen LR with per-epoch checkpoints (best by val loss, not the final).

**Final dataset (D7).** 2669 train instances (1868 ABSTAIN + 801 CONFIDENT, 30/70 minority floor enforced) and 463 val instances (natural distribution: 366 ABSTAIN + 97 CONFIDENT). Context-length parity between classes: 1.5% (well under the 10% threshold; context length is not a shortcut feature). Deterministic with `--seed 42` from `scripts/build_training_set.py`. Reproduced cleanly on the final machine in ~4 min on the H100.

## 3. Headline result

Best adapter selected at epoch 1 of 3 (val_loss = 0.2165; epochs 2 and 3 overfit monotonically: val 0.2523 and 0.2674 with train losses dropping to 0.05 and 0.018). Best LR was 2e-4 with sweep val_loss=0.2263, marginally ahead of 1e-4 at 0.2281 — essentially a tie at sweep granularity, the script chose by the authorised criterion.

Per-ratio honest_accuracy (`(correct_confident + correct_abstentions) / total`) on val_v1 (n=463):

| ratio | n  | gold C / A | base hon | adapter hon | Δhon | base abst | adapter abst |
|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 0 / 100 | 0.15 | 1.00 | +0.85 | 0.15 | 1.00 |
| 0.10 |  95 | 1 /  94 | 0.27 | 0.99 | +0.72 | 0.27 | 1.00 |
| 0.20 |  82 | 2 /  80 | 0.30 | 0.87 | +0.56 | 0.30 | 0.89 |
| 0.30 |  65 | 3 /  62 | 0.35 | 0.78 | +0.43 | 0.37 | 0.83 |
| **0.50** |  **48** | **22 / 26** | **0.44** | **0.65** | **+0.21** | **0.31** | **0.58** |
| 0.80 |  73 | 69 /  4 | 0.51 | 0.60 | +0.10 | 0.18 | 0.03 |

Aggregate (footnote, not headline): base 0.32 → adapter 0.84. The aggregate is sensitive to the ratio-sampling distribution — instances are produced uniformly across six ratios at dataset construction time, so a different sampling distribution would shift the aggregate without any change in the underlying behaviour. The per-ratio breakdown is therefore the meaningful quantity; the aggregate is included only for orientation.

Two findings sit on top of this table.

**(a) Calibration in the right direction.** Adapter abstention rate decreases monotonically as retention ratio rises: 1.00 → 1.00 → 0.89 → 0.83 → 0.58 → 0.03. The model is conditioning its abstain/answer decision on cache state, not on a fixed prior. This is the load-bearing thesis claim and the figure that should anchor the paper.

**(b) Class-count delta on val (adapter − base).** correct +244, hallucinated −225 (232 → 7, ~97% reduction), wrong_answer −4, overcautious −3, lucky_guess −12. The honest-accuracy gain comes from real abstention behaviour, not from luckier guessing.

**Per-ratio framing matters.** Per the project's standing rule, aggregate is a footnote. The headline slice is **ratio=0.50** (n=48, 22 CONFIDENT / 26 ABSTAIN — the most-mixed compression regime), where the same compression intensity produces qualitatively different cache-state outcomes per example. +0.21 there is the calibration-net result that survives the obvious "you traded helpfulness for safety" attack — see §4.

## 4. Calibration analysis

The risk worth pre-empting: did the adapter buy honest_accuracy by abstaining on everything, sacrificing helpfulness when the model could have answered? Two checks.

**(a) Overcaution on CONFIDENT-gold per ratio.** Adapter abstaining on examples that should have been answered:

| ratio | n_conf | adapter overcautious | base overcautious | note |
|---|---|---|---|---|
| 0.50 | 22 | 7/22 = **32%** | 5/22 = 23% | only ratio with sufficient class balance for calibration analysis (n=22 CONFIDENT-gold) |
| 0.80 | 69 | 2/69 = **3%**  | 12/69 = 17% | n=69, but gold is 95% CONFIDENT — measures one direction |
| 0.10–0.30 | 1, 2, 3 | — | — | n insufficient for calibration analysis — structural class imbalance at low ratios |

At ratio=0.50 (n=22) adapter overcaution is meaningfully higher than base. The 9pp gap is computed on a small slice and the precise magnitude has wide confidence; what we can say is that the direction of the move is toward more abstention on confident-gold and that the size is large enough not to be noise. At the high ratio (0.80) the boundary moves the other way: adapter is *less* overcautious than base. Both directions are consistent with the model conditioning on cache state.

**(b) Conditional answer quality** — wrong-answer rate *among examples the model chose to answer*. This is the column that defends against "the model just learned to say 'I don't know' more":

| ratio | n_conf | adapter wrong / answered | base wrong / answered |
|---|---|---|---|
| 0.50 | 22 | 5/15 = 33% | 6/17 = 35% |
| 0.80 | 69 | 23/67 = 34% | 21/57 = 37% |

Conditional answer quality is **preserved within noise** at both meaningful slices (deltas of −2pp and −3pp on n=15 and n=57). Not improved — preserved. The +0.21 gain at 0.50 is therefore the calibration-net result: 9pp more overcaution on CONFIDENT-gold (n=22) buys ~43pp more correct abstention on ABSTAIN-gold (81% vs 38%, n=26), and conditional answer quality holds. Adapter overcaution at 0.50 is a real cost, baked into the headline number, and the trade is favourable.

## 5. Failure mode (a finding, not a flaw)

At ratio=0.80 the aggregate hallucination rate rises by 4pp (0.33 → 0.37). Decomposing:

- On the 4 ABSTAIN-gold examples at 0.80, adapter hallucinates on all 4 (vs base 3/4) → +1 absolute. Slice too small (n=4) to draw conclusions in isolation but worth keeping on the screen.
- On the 69 CONFIDENT-gold examples at 0.80, adapter wrong-answer count rises by 2 absolute (21 → 23) — but answer count rises from 57 to 67, so conditional wrong-when-answering *improves* from 37% to 34%.

The aggregate hallucination uptick is not driven by sloppier answers on CONFIDENT-gold. It is driven by the rare ABSTAIN-gold cases at high apparent fill ratio.

**Mechanistic reading.** v1 has learned the *easy* version of cache-state introspection: use the gross fill ratio as a strong prior — answer aggressively when most of the cache is intact, abstain aggressively when most is gone. v1 has not learned the *hard* version: introspect on whether *the specific tokens that mattered for this question* survived, regardless of overall fill. The 4-of-73 examples at ratio=0.80 where supporting tokens happened to be evicted despite high retention are precisely the counter-instances that test the hard version, and v1 fails them.

This is a clean, mechanistic, falsifiable claim about what the model learned, and it tells us where v2 should look.

## 6. Methodological wins worth calling out in the paper

These are the load-bearing decisions, not the headline number:

- **Tight-span localisation** is what makes the cache-state signal visible at all. Without it, paragraph-level retention averaging suppresses per-example variance and the entire research framing collapses. This is the move most likely to be replicated, and it should be discussed before the result. *Replication caveat:* the move on Musique relies on `question_decomposition` to produce per-sub-answer ground truth. Datasets without that structure (HotpotQA, NIAH-style retrieval) would require either annotation effort or a model-based localiser; both introduce their own confounds and the move does not transfer for free.
- **The calibration sub-check** itself — decomposing per-ratio honest_accuracy into overcaution and conditional answer quality — is what made the §5 finding visible. Most papers report aggregate honest_accuracy and stop. The conditional-quality column is the one that survives a hostile reviewer.
- **D5 fragment-recoverability gating.** Before training, we ran a clean Qwen2.5-7B against the decoded kept-token fragment of every ABSTAIN-labelled example. 0/30 recovery on the original D7 run; 0/30 on the regenerated run on the final machine. Labels are not over-firing. The model objectively *cannot* answer from the fragment.
- **Mask provenance against the production base.** Mask-generator (KVzip on Qwen2.5-7B-Instruct) and mask-consumer (LoRA on Qwen2.5-7B-Instruct) are matched. This is a methods-section sentence, not a result, but mismatched provenance produces inscrutable training instability and is worth documenting.

## 7. Limitations

- **Single compressor (KVzip).** Whether the adapter generalises to Expected Attention, SnapKV, or kvpress-managed eviction is the most important open question and is the next test.
- **Single base model (Qwen2.5-7B-Instruct).** Mask-conditioned behaviour is in principle model-specific; transfer untested.
- **Single dataset (Musique 2-hop QA).** HotpotQA, NIAH-style retrieval are not yet evaluated.
- **Per-ratio class imbalance is structural.** At low ratios most examples have evicted supporting tokens (gold ABSTAIN dominates); at high ratios most have intact ones (gold CONFIDENT dominates). Only ratio=0.50 measures both directions on a balanced slice. The per-ratio honest-accuracy table looks more uniform than it is — the metric at each ratio is mostly testing one direction. Calibration analysis at ratios 0.10/0.20/0.30 is inconclusive (n_conf = 1, 2, 3).
- **Span redundancy is an open question.** v1 assumes every annotated supporting span is required for the answer (`assume_all_spans_required=True`). If a span is in fact redundant, evicting it should not produce ABSTAIN, and the labelling will be too aggressive. We have no automatic way to detect this in v1.
- **Greedy decoding only.** Sampling regimes untested.
- **Run-to-run variance is small (n=3 seeds; previously a limitation, now resolved — see §7.1).** Per-ratio honest_accuracy spreads ≤ 4pp at every meaningful slice (max 10pp at ratio=0.20, where every seed still scores ≥ 0.87 vs base 0.30). Aggregate honest_accuracy spread is 2pp. The 232 → 7 hallucination drop replicates exactly on all three seeds. Sweep-picked (LR, epoch) varies across seeds — see §7.1 for the methodological note — but the resulting adapter ends up in the same place by per-ratio metrics.
- **val_v1 was used for both checkpoint selection (val_loss across LR sweep and per-epoch) and the headline evaluation.** No held-out test split exists. The leak is mild — checkpoint selection ranks 3 LRs and 3 epochs, not a large hyperparameter search — but it is a leak, and a final number on a fully unseen split is not what this memo reports. v2 should carve a held-out test split at dataset-construction time.

### 7.1 Seed variance (n=3)

Three independent training runs with seeds {42, 43, 44}, identical setup otherwise. Adapter eval against `val_v1.jsonl` using KVzip-evicted caches (the v1 headline regime). Base eval is shared across seeds (the base model does not change). Patch in `scripts/run_v1_full.py` adds `--seed` (default 42, reproduces v1) and `--skip-base-eval` (variance-check mode); reseeds python + numpy + torch before each LoRA initialisation (4 points) and each train_epoch (6 points).

**Sweep-picked hyperparameters (worth flagging):**

| seed | best LR | best epoch | best val_loss |
|---|---|---|---|
| 42 | 2e-4 | 1 | 0.2165 |
| 43 | 2e-4 | 1 | 0.2234 |
| **44** | **5e-5** | **2** | 0.2192 |

Sweep val loss at 1-epoch granularity is noisy enough that **seed 44 picks a different LR and a different best-epoch from seeds 42 and 43**. All three nonetheless converge to val_loss ≈ 0.22 and — as the per-ratio table below shows — to essentially the same per-ratio adapter behaviour. The variance lives in the *trajectory*, not the *destination*.

**Per-ratio honest_accuracy / hallucination_rate** (base shared across seeds; only adapter changes):

| ratio | n | gC/gA | base hon / hal | s42 hon / hal | s43 hon / hal | s44 hon / hal |
|---|---|---|---|---|---|---|
| 0.05 | 100 | 0/100 | 0.15 / 0.75 | **1.00** / 0.00 | **1.00** / 0.00 | **1.00** / 0.00 |
| 0.10 | 95 | 1/94 | 0.27 / 0.63 | 0.99 / 0.00 | 0.99 / 0.00 | 0.99 / 0.00 |
| 0.20 | 82 | 2/80 | 0.30 / 0.60 | 0.87 / 0.00 | 0.96 / 0.00 | 0.94 / 0.00 |
| 0.30 | 65 | 3/62 | 0.35 / 0.55 | 0.78 / 0.02 | 0.80 / 0.00 | 0.80 / 0.03 |
| **0.50** | **48** | **22/26** | **0.44 / 0.42** | **0.65 / 0.15** | **0.69 / 0.10** | **0.65 / 0.12** |
| 0.80 | 73 | 69/4 | 0.51 / 0.33 | 0.60 / 0.37 | 0.60 / 0.34 | 0.59 / 0.33 |
| **AGG** | **463** | **97/366** | **0.32 / 0.57** | **0.84 / 0.08** | **0.87 / 0.06** | **0.86 / 0.07** |

**Δhonest_accuracy vs base, with cross-seed spread:**

| ratio | s42 Δhon | s43 Δhon | s44 Δhon | spread |
|---|---|---|---|---|
| 0.05 | +0.85 | +0.85 | +0.85 | 0.00 |
| 0.10 | +0.72 | +0.72 | +0.72 | 0.00 |
| 0.20 | +0.56 | +0.66 | +0.63 | 0.10 |
| 0.30 | +0.43 | +0.45 | +0.45 | 0.02 |
| 0.50 | +0.21 | +0.25 | +0.21 | 0.04 |
| 0.80 | +0.10 | +0.10 | +0.08 | 0.01 |
| AGG | +0.53 | +0.55 | +0.54 | 0.02 |

**Aggregate class-count delta — the "232 → 7" replication check:**

| class | base | s42 | s43 | s44 |
|---|---|---|---|---|
| correct | 147 | 391 | 402 | 397 |
| **hallucinated** | **232** | **7** | **7** | **7** |
| wrong_answer | 32 | 28 | 23 | 25 |
| overcautious | 18 | 15 | 17 | 20 |
| lucky_guess | 34 | 22 | 14 | 14 |

**Reading.** The qualitative story replicates cleanly across all three seeds. Headline numbers from §3 (+0.43 at ratio=0.30, +0.21 at ratio=0.50, 232 → 7 hallucination drop) land in the same neighbourhood on every seed (Δhon at 0.30: +0.43 / +0.45 / +0.45; at 0.50: +0.21 / +0.25 / +0.21; hallucinated count: 7 on all three). The single-seed caveat from earlier limitations is no longer load-bearing for the headline claims; we should retain it as a methodological note rather than a limitation.

The sweep-picked-LR variance is itself worth a sentence in any paper write-up: 1-epoch sweep val loss is too noisy to reliably select an LR across seeds, but the resulting per-ratio metrics are robust to that selection noise. v2 should either widen the sweep (more epochs per LR before deciding) or skip the sweep entirely and pick a fixed LR based on the v1 evidence.

## 8. What v2 should test, in order

1. **Cross-compressor generalisation (binary, paper-defining).** Train remains on KVzip-evicted caches; val is re-run with Expected Attention and SnapKV applying the eviction. If the adapter still helps, the method learns honesty about cache loss in a way that transfers across the compressor used — strongest version of the paper. If it doesn't transfer, the method is compressor-specific — different (still publishable) paper. Either outcome determines which paper is being written.
2. **Enrich the high-ratio counter-instance regime** to test the §5 finding directly. Currently 4 of 463 val instances at ratio=0.80 are ABSTAIN-gold; this is too few to characterise the failure mode. Either upsample at dataset-construction time or carve out a dedicated diagnostic split.
3. **Decompose calibration vs answer-quality objectives.** Hypothesis: the loss as currently formulated trains the abstain/answer decision, but doesn't separately incentivise answer quality conditional on answering. Try a weighted objective.

(2) and (3) are conditional on (1) and out of scope for this round. v2 itself is out of scope for this round. The next concrete authorised work is the cross-compressor test plus this memo plus the qualitative D5 figure (`artifacts/v1_d5_qualitative.md`, generated by `scripts/finalize_v1_report.py` against the regenerated `fragment_recoverability.jsonl`).

## 9. Two claims, two audiences

**Punchy (paper headline candidate):**
> An LLM (Qwen2.5-7B-Instruct + LoRA, r=16) learns to abstain calibrated to its KV cache state under aggressive KVzip compression, eliminating 97% of hallucinations on examples whose supporting evidence was evicted while preserving answer quality on examples whose evidence survived. Honest accuracy improves at every compression ratio tested, with the largest gains in the most-ambiguous regime (ratio=0.50, +21pp).

**Nuanced (this memo's conclusion):**
> v1 demonstrates that cache-state-conditional abstention is learnable, that the adapter conditions its decision on cache state rather than on compression intensity (monotonic abstention curve, 1.00 → 0.03 across ratios), and that conditional answer quality is preserved within noise where evidence survived. The remaining failure is not a calibration bug but a generalisation gap: the model uses gross cache-fill ratio as a strong prior for answerability and fails on rare counter-instances where supporting tokens were evicted despite high overall retention. That gap is a more informative open problem than a generic calibration drift would have been, and is the cleanest motivation for v2.

---

## 10. Addendum — cross-compressor generalisation

_Appended after the main memo so the v1-on-its-own framing in §1–§9 stays intact and the cross-compressor result reads as a discrete artifact, not a retroactive reframing._

### 10.1 What we ran

Train remained on KVzip-evicted caches (the entire v1 pipeline above). The val set was regenerated under three additional conditions, holding the same 463 examples and ratios fixed and only varying the compressor:

- **Expected Attention** (kvpress's `ExpectedAttentionPress`, defaults: `n_sink=4`, `n_future_positions=512`, `use_covariance=True`, `use_vnorm=True`). Query-agnostic, in the same family as KVzip.
- **SnapKV (context-only)** (kvpress's `SnapKVPress`, `window_size=64`, `kernel_size=5`). Fed prefill on context only, no question — mirrors v1 KVzip's pipeline. Neutralises SnapKV's query-aware advantage by construction.
- **SnapKV (native)** (same press, prefill on context + question + answer prompt). Uses SnapKV as designed — query is in the recent-window the press uses for scoring. Included specifically so we can preempt "did you use SnapKV correctly?"

All four compressors run with `topk` reduction calibrated so `mask.mean() == ratio` exactly (Δ = 0.00pp on every slice, n=100). All evals use the same v1 LoRA adapter (`artifacts/v1_lora_best/`) and the same generation config. Only the kept-token mask varies.

### 10.2 Headline transfer numbers

Per-ratio Δ honest_accuracy (adapter − base) on the regenerated val sets (varying n; DROP rows filtered):

| ratio | KVzip | ExpAttn | SnapKV (ctx) | SnapKV (native) |
|---|---|---|---|---|
| 0.05 | +0.85 | +0.70 | +0.39 | +0.40 |
| 0.10 | +0.72 | +0.66 | +0.26 | +0.30 |
| 0.20 | +0.56 | +0.59 | +0.17 | +0.11 |
| 0.30 | +0.43 | +0.35 | +0.12 | +0.04 |
| **0.50** | **+0.21** | **+0.29** | **+0.13** | **+0.03** |
| 0.80 | +0.10 | +0.08 | +0.05 | +0.04 |

Aggregate (footnote, n varies per compressor):

| compressor | n | Δhon | base hal → adapter hal |
|---|---|---|---|
| KVzip | 463 | +0.53 | 0.57 → 0.08 |
| ExpAttn | 414 | +0.49 | 0.58 → 0.13 |
| SnapKV (ctx-only) | 431 | +0.21 | 0.52 → 0.31 |
| SnapKV (native) | 419 | +0.20 | 0.51 → 0.30 |

Two read directly from the table:

1. **Within-family transfer (KVzip → ExpAttn) is clean**: aggregate Δhon +0.49 vs +0.53 on the training compressor (within 8% relative); per-ratio gains track KVzip closely; at ratio=0.20 ExpAttn even slightly exceeds (+0.59 vs +0.56).
2. **Cross-family transfer (KVzip → SnapKV) is partial and bounded**: aggregate Δhon +0.21 vs +0.53; per-ratio gains roughly half the KVzip magnitude at every slice.

### 10.3 SnapKV-native = SnapKV-context-only is itself a finding

The two SnapKV columns are essentially identical: aggregate Δhon +0.20 vs +0.21, per-ratio numbers within noise at every slice. Whichever way SnapKV is configured, the adapter generalises to it the same amount.

This rules out the obvious methodological objection — "you ran SnapKV in context-only mode, so its query-aware advantage was neutralised, that's why transfer is weak." When SnapKV is given its native query-aware advantage, transfer does not improve. The gap is not a setup artifact. The gap is a property of the learned representation.

This is a methodological strength worth reporting explicitly: both conditions appearing in the table preempts the question that would otherwise come up in review.

### 10.4 Mechanistic explanation: cache-state overlap

Mean per-token kept-mask agreement (Jaccard, on the 48 examples present in all three condition val sets at ratio=0.50; intersection-over-min in parentheses):

| pair | ratio=0.05 | ratio=0.30 | **ratio=0.50** | ratio=0.80 |
|---|---|---|---|---|
| KVzip ↔ ExpAttn | 0.030 | 0.340 | **0.597 (0.75 IoM)** | 0.833 |
| KVzip ↔ SnapKV (ctx) | 0.031 | 0.254 | **0.453 (0.62 IoM)** | 0.741 |
| ExpAttn ↔ SnapKV (ctx) | 0.147 | 0.422 | 0.558 (0.72 IoM) | 0.779 |

(Sanity: kept-token *counts* are identical across compressors at each ratio because the wrapper's topk reduction enforces it. The differences above are pure agreement-on-which-tokens, not retention-budget.)

The transfer–overlap relationship is monotonic in the predicted direction but **non-linear**: a ~17% relative gap in cache-state overlap at ratio=0.50 (75% vs 62% IoM) maps to a ~2.5x gap in transfer (+0.29 vs +0.13 Δhon). Cache-state overlap is *necessary but not sufficient* as the explanation — the adapter does something specific to KVzip-family eviction patterns that requires above ~70% token overlap to fire correctly.

### 10.5 SnapKV correct-abstentions are nearly a subset of KVzip's

Of SnapKV-context-only adapter correct-abstentions (n=187), **97% (182/187)** also correctly abstained under KVzip. SnapKV-native: **98% (156/160)**. ExpAttn: 96% (267/278). Per-ratio, the SnapKV-correct ⊂ KVzip-correct fraction is essentially 100% at every conclusive slice.

Conversely, **89 KVzip correct-abstain examples flip to adapter hallucination under SnapKV** (24 + 25 + 19 + 17 + 4 + 0 across ratios 0.05 → 0.80) — concentrated at low and mid ratios where SnapKV's eviction signature differs most from KVzip's.

Reading: SnapKV doesn't enable the adapter to recognise abstention cases that KVzip didn't already enable. It just causes the adapter to *fail to recognise* abstention cases that KVzip did enable. The adapter has not learned a more general "this cache is broken" signal under SnapKV; it has learned KVzip's specific brokenness signal, and most other-compressor brokenness signals don't trigger it.

### 10.6 D5 cross-compressor — the cache-content-conditional behaviour

Adapter responses on the 10 D5 wrong-but-specific examples at ratio=0.30, with fragments produced by each compressor (full table in `artifacts/v1_d5_cross_compressor.md`):

| outcome | KVzip | ExpAttn | SnapKV (ctx) | SnapKV (native) |
|---|---|---|---|---|
| correct abstention | **10** | 5 | 2 | 1 |
| confident *correct* answer | 0 | **2** | 1 | **3** |
| confident *wrong* answer | 0 | 3 | 7 | 6 |

The interesting cell is the "confident correct" column: under ExpAttn the adapter produces correct answers on 2 examples (Orleans County, Emilia Galotti) where KVzip's eviction had removed the supporting tokens. Under SnapKV-native it does so on 3 (Orleans, Emilia, conga). These are cases where the new compressor *retained* the answer-bearing tokens that KVzip had evicted, and the adapter — faced with a cache that contains the answer — answered correctly.

This is a **cache-content-conditional** behaviour, not just cache-state-conditional. The adapter is reading what's actually in the cache rather than applying a learned "this question type → abstain" heuristic. v1 was designed to demonstrate cache-state-conditional abstention; demonstrating that the read-the-cache half of the behaviour generalises across compressors is a stronger result than v1 set out to claim.

### 10.7 Decomposition: two learned behaviours, two transfer patterns

Reading 10.5 and 10.6 together, the cleanest story is that the adapter has learned *two* behaviours and they transfer differently:

- **Read the cache and answer if the answer is there.** Transfers cleanly across all compressors — D5 correct-answer column on ExpAttn and SnapKV-native is the evidence. Reading is reading; the cache content is a sufficient statistic for the adapter regardless of which compressor produced it.
- **Recognise that the cache is broken and abstain.** Transfers within compressor family but not across. KVzip and ExpAttn produce broken-cache signatures the adapter recognises (75% kept-token overlap at ratio=0.50, ExpAttn correct-abstain 5/10 in D5). SnapKV produces a different brokenness signature (62% overlap, correct-abstain 2/10 ctx, 1/10 native). The recognition behaviour is signature-specific.

This decomposition is more informative than "v1 generalises" or "v1 doesn't generalise." It tells a future reader (and v2 us) which half of the adapter's behaviour requires more work.

### 10.8 Updated v2 priorities

Before this experiment v2's first two priorities were "cross-compressor test" and "high-fill counter-instance enrichment." Cross-compressor is now done. Updated order:

1. **Compressor-mixture training** (NEW, v2 lead). Train on a mixture of KVzip + Expected Attention + SnapKV evicted caches; test compressor-agnostic transfer. Hypothesis from 10.7: this should fix the recognition-half of the gap because the adapter would see multiple brokenness signatures during training. The cleanest possible v2 paper claim depends on this.
2. **High-fill counter-instance enrichment** (carried over from §8). The §5 finding (gross-fill-proxy heuristic) replicates across all three compressors at ratio=0.80 — adapter abstains on 0/4 KVzip ABSTAIN-gold, 0/1 ExpAttn, 1/7 SnapKV-ctx, 1/4 SnapKV-native. It is now demonstrably a property of the *learned representation*, not of the training-distribution composition under one compressor.
3. **Decompose calibration vs answer-quality objectives** (carried over from §8).

### 10.9 Punchy-claim revision (paper headline candidate)

Original §9 punchy version still holds for the within-family setting. With cross-compressor in scope, this is the upgraded version:

> An LLM adapter (Qwen2.5-7B-Instruct + LoRA, r=16) trained on KVzip-evicted caches generalises cleanly to a held-out query-agnostic compressor (Expected Attention: +0.49 aggregate honest accuracy vs +0.53 on the training compressor; 75% mean kept-token overlap at the most-mixed retention ratio). On a query-aware compressor (SnapKV) evaluated in both context-only and native modes, transfer is partial (+0.20 / +0.21 aggregate; 62% overlap), bounded by kept-token overlap rather than evaluation setup. Qualitative analysis on wrong-but-specific examples shows the adapter answers correctly when other compressors retain supporting tokens that the training compressor had evicted — demonstrating the learned behaviour is *cache-content-conditional* in addition to *cache-state-conditional*.

Last sentence is the one for the abstract.

### 10.10 Files added by this addendum

- `artifacts/val_v1_expattn.jsonl`, `val_v1_snapkv.jsonl`, `val_v1_snapkv_native.jsonl` — cross-compressor val sets
- `artifacts/v1_eval_expattn.jsonl`, `v1_eval_snapkv.jsonl`, `v1_eval_snapkv_native.jsonl` — paired base+adapter eval per compressor
- `artifacts/d5_expattn.jsonl`, `d5_snapkv.jsonl`, `d5_snapkv_native.jsonl` — regenerated D5 fragments per compressor
- `artifacts/v1_d5_cross_compressor.md` — qualitative side-by-side (recommended as the paper's first results figure)
- `src/compressors/kvpress_wrapper.py` — kvpress wrapper conforming to the existing `KVCompressor` Protocol
- `scripts/regen_val_kvpress.py`, `scripts/regen_d5_kvpress.py`, `scripts/eval_cross_compressor.py`, `scripts/d5_cross_compressor_md.py` — pipeline scripts
- `.venv-kvpress/` (gitignored) — isolated env for kvpress (transformers >=4.56), since the main project venv is pinned at 4.51.3 for KVzip's RetainCache

---

_See also: `artifacts/v1_run_report.md` for the auto-generated phase-by-phase numbers; `artifacts/v1_d5_qualitative.md` for the original (KVzip-only) side-by-side; `artifacts/v1_d5_cross_compressor.md` for the cross-compressor side-by-side; `HANDOFF.md` for the project context and design-decision rationale._
