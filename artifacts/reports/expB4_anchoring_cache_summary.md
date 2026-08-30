# Exp B4 — Anchoring the Cache (HalluKV) — related-work brief

_OpenReview ID `K5YOyNaVN0`. Anonymous ACL submission. PDF fetched and parsed in full._

## Problem statement

The paper presents the first systematic study of how KV cache compression affects hallucination in **long-context summarization**. Two findings motivate the method:

1. Aggressive KV cache compression amplifies hallucination by up to **3.36×** the FullKV baseline (Multi_news with Llama-3-8B-Instruct, KV budget = 64 entries).
2. KV cache compression specifically causes **retrieval heads** (per Wu et al. 2024b's NIAH-based identification) to shift attention away from source context toward generated content during decoding — a "snowballing" effect where output drifts further from source as generation continues.

## Proposed method (HalluKV)

**Type**: inference-time intervention. **Decoding-phase only.** No retraining, no fine-tuning, no loss function changes, no data construction. Base model weights and the existing prefilling-phase compression are both untouched.

**Mechanism**: identify the top-α fraction of attention heads as "retrieval heads" (using existing NIAH-based identification from Fu et al. 2024). During decoding, for those selected heads only:
- **Step 1**: keep all compressed source-context KV pairs from the prefilling phase.
- **Step 2**: evict all generated KV pairs (pairs from positions after the prefill).
- **Step 3**: compute attention only over the kept source context.

For non-retrieval heads, attention computes normally over the full cache (source + generated). The intervention is a binary "retrieval head / not" classification with a single hyperparameter α (mask ratio) and is composable on top of any prefilling-phase compression scheme — they show it stacked on SnapKV, PyramidKV, and HeadKV.

## Evaluation

- **Models** (six): Llama-3-8B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, Mistral-7B-Instruct-v0.3, Qwen2-7B-Instruct, Qwen2.5-7B-Instruct.
- **Datasets**: Two long-context summarization datasets from LongBench — `Multi_news` (multi-document summarization) and `Gov_report` (long-document summarization). They mention an Appendix-D extension to QASPER (QA), but the main results are summarization only.
- **Compression budgets**: KV-64, KV-128, KV-256, KV-1024 — these are absolute KV cache sizes (number of retained tokens), not retention ratios. Note this is a different parameterization than ours (we use retention ratios on context length).
- **Baselines**: SnapKV, PyramidKV, HeadKV — three prefilling-phase compression methods. HalluKV is built on top of HeadKV (set α=0.20). FullKV baseline is also reported.
- **Metrics**: ROUGE for generation quality + **FineSurE** (Song et al. 2024, a Claude-3.7-powered fine-grained framework) for hallucination scores. Hallucination score = percentage of unfaithful statements.

## Headline result

Headline reduction: **5.48 points average hallucination score** on Llama-3-8B-Instruct vs HeadKV baseline. ROUGE simultaneously *improves* by 1.13 average points. Generalizes across all six models and four KV budgets.

Specific cells from Table 1 they cite:
- Multi_news, KV-64, Llama-3-8B: hallucination 60.33% → 55.81% (HeadKV → HalluKV; 7.5% relative reduction)
- Multi_news, KV-1024, Llama-3-8B: 22.50% → 19.95% (11.3% relative reduction)
- Effect is largest on the second half of generated summaries (the "hallucination at the last" phenomenon, which they explicitly target with the decoding-time intervention).

## How HalluKV differs from our work — the §1 positioning paragraph

The two papers share the surface motivation ("KV cache compression causes hallucination, propose a fix") but diverge on every meaningful axis:

| dimension | HalluKV | Our work (v1 + v2/mixture) |
|---|---|---|
| **intervention type** | Inference-time, decoding-phase eviction in retrieval heads | Train-time, LoRA fine-tuning on cache-state-conditional labels |
| **what's modified** | Attention computation in selected heads | Generation policy of the model itself |
| **what's left untouched** | Model weights, training procedure | Compressor, decoding algorithm, attention mechanism |
| **task family** | Summarization (Multi_news, Gov_report) | Multi-hop QA (MuSiQue) + retrieval (RULER NIAH) |
| **evaluation framing** | ROUGE + FineSurE hallucination percentage on summaries | Per-ratio honest_accuracy + hallucination rate + abstention rate on QA classifications |
| **hallucination definition** | Unfaithful generated statements (LLM-judged) | Confident wrong answers when supporting tokens were evicted (substring-match-or-abstain) |
| **abstention** | Not modeled — model still generates a summary, just one that drifts less | Calibrated abstention is the central trained behavior |
| **per-ratio breakdown** | KV-budget-by-budget (4 budgets) | Per-retention-ratio (6 ratios) with explicit headline at the most-mixed slice |
| **cross-compressor** | Stacks on top of any prefilling compressor | Trains for compressor mixture; generalizes across compressor families |
| **base model coverage** | 6 LLMs, all 7-8B | 2 LLMs (Qwen2.5-7B v1, Llama-3.1-8B replication) |
| **key claim** | Decoding-phase eviction of retrieval-head generated cache → less drift | LoRA-trained calibrated abstention → answer or abstain based on cache state |

The **single most important distinction**: HalluKV does not change *what* the model says — it changes *how* attention is computed during decoding to reduce drift. Our adapter changes *what* the model is willing to say (generates calibrated "I cannot determine" abstentions when supporting evidence is evicted). They are complementary, not competing — HalluKV could be applied on top of our adapter at inference time. Neither work tries this composition.

## Lessons for our experimental design

Worth borrowing from HalluKV (or worth flagging as differences they didn't have to confront):

1. **Six base models** — we have two (Qwen v1, Llama replication). v2 paper would benefit from at least Mistral as a third family. Cheap to add given v1's training pipeline.
2. **FineSurE as a hallucination metric** — they use an LLM-judged hallucination percentage; we use substring-match-or-abstain. Our metric is cheaper and more reproducible; theirs is more semantically rich. The tradeoff is well-understood; defensible to keep ours but worth a sentence in methods explaining why.
3. **First/second-segment hallucination split** — they observe hallucination is worse later in generation. We don't have multi-segment generation (our outputs are short answers), so this lens doesn't apply.
4. **No ratio-only baseline ablation** — they don't probe whether their method beats a trivial threshold policy that uses only the KV budget. Our Exp B1 (just done) does this — paper-defining for us. Worth noting in the lit-review paragraph: "unlike prior work [HalluKV] we explicitly probe ratio-only baselines to test whether the cache-state-introspection claim survives a simple threshold heuristic."
5. **No QA evaluation in main results** — they're summarization-only. Our QA + retrieval evaluation is a meaningful extension and probably the cleaner setting for testing calibrated abstention (because abstention has a well-defined "correct" target in QA, whereas in summarization "should the model omit this fact" is fuzzier).

## Honest weaknesses they acknowledge — worth not repeating

From their Limitations section:

1. **Binary retrieval/non-retrieval head classification oversimplifies** — heads have multiple roles (retrieval, ICL, safety) per Olsson et al. 2022. They classify based on NIAH-only.
2. **Evaluation primarily summarization** — they note "preliminary results on QASPER QA" in Appendix D suggest generalization but admit it's not the main result. This is a weakness for them; for us it's a strength (we eval on QA + NIAH directly).
3. **Mask ratio α=0.2 is a single hyperparameter** — they don't extensively sweep it.

## What we should NOT cite or claim

- They never claim to test cache-state introspection per se. Don't conflate.
- They claim "first systematic study of KV compression × hallucination in summarization." Be careful with our own "first systematic study of cache-state-conditional generation" framing — adjacent but distinct claim.
- They use FineSurE (LLM-judged) and we use substring-match. Not directly comparable; don't try.

## Files

- `/tmp/anchoring_cache.txt` — locally-extracted plaintext (1164 lines)
- This file — the brief
