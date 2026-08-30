# Exp B3 — Fragment-prompt baseline

_Tests whether the cache representation carries signal beyond the surviving text. Compares the v1 / mixture adapter to a model that sees only the **decoded surviving tokens** as a text prompt — no architectural-level cache compression at inference._

## Methodological note (read this first)

The v1 eval harness already constructs the inference prompt as `"Context: <decoded kept tokens>\n\nQuestion: {q}\n\nAnswer:"` for both base and adapter. Verified byte-by-byte on a sample row: the v1 base eval prompt and the prompt the B3 spec asks for are **identical**. The model never sees a compressed cache at inference; it sees decoded text. Cache compression is only used to derive *which* tokens to retain.

Therefore the v1 base eval column in `artifacts/v1_eval.jsonl` IS the fragment-prompt baseline. This report is a re-presentation + sub-analysis on that existing data; no new generation. Re-running with greedy decoding would be byte-identical.

## Aggregate on val_v1 (KVzip, n=463)

| policy | honest acc | hallucination | abstention | Δhon vs fragment |
|---|---|---|---|---|
| **fragment-prompt** (= v1 base) | **0.317** | **0.570** | 0.255 | +0.000 |
| v1 adapter | 0.844 | 0.076 | 0.760 | **+0.527** |
| mixture adapter | 0.855 | 0.058 | 0.762 | **+0.538** |

Both adapters dominate fragment-prompt by ~+0.53 aggregate. Hallucination rate drops from 0.570 to ~0.07. The "decode kept tokens and prompt" pipeline is **far weaker** than the LoRA-trained adapter on the same input.

## Per-ratio honest_accuracy

| ratio | n | gC/gA | fragment | v1 adp | mix adp | v1 − frag | mix − frag |
|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 0/100 | 0.150 | 1.000 | 1.000 | **+0.850** | +0.850 |
| 0.10 | 95 | 1/94 | 0.274 | 0.989 | 0.989 | +0.716 | +0.716 |
| 0.20 | 82 | 2/80 | 0.305 | 0.866 | 0.866 | +0.561 | +0.561 |
| 0.30 | 65 | 3/62 | 0.354 | 0.785 | 0.769 | +0.431 | +0.415 |
| **0.50** | **48** | **22/26** | **0.438** | **0.646** | **0.688** | **+0.208** | **+0.250** |
| 0.80 | 73 | 69/4 | 0.507 | 0.603 | 0.658 | +0.096 | +0.151 |

**Headline cell, ratio=0.50 KVzip**: fragment-prompt 0.438 → v1 adapter 0.646 (+21pp) → mixture 0.688 (+25pp). Both adapters beat the success criterion (≥10pp at ratio=0.50) by a comfortable margin.

## Per-ratio hallucination rate

| ratio | fragment | v1 adp | mix adp |
|---|---|---|---|
| 0.05 | 0.750 | 0.000 | 0.000 |
| 0.10 | 0.632 | 0.000 | 0.000 |
| 0.20 | 0.598 | 0.000 | 0.000 |
| 0.30 | 0.554 | 0.015 | 0.046 |
| 0.50 | 0.417 | 0.146 | 0.125 |
| 0.80 | 0.329 | 0.370 | 0.247 |

Fragment-prompt hallucinates aggressively at low ratios (~60-75%) — the adapter eliminates this almost entirely (0% at 0.05/0.10/0.20). The exception is ratio=0.80 where v1 adapter (0.37) is slightly *worse* than fragment-prompt (0.33) — the v1 high-ratio over-confident-answer pattern from §5 of the v1 memo. Mixture (0.247) beats fragment-prompt at every ratio including 0.80.

## §10.5 sub-analysis — does the surviving text carry the abstention signal?

The 89 (id, ratio) keys identified in v1 §10.5 are: examples where the v1 adapter **correctly abstained on KVzip but hallucinated on SnapKV**. They isolate v1's cross-compressor recognition gap.

For each of those 89 keys, on the **KVzip val row** (the side v1 adapter got right, by definition):

| classifier | correct (= correctly abstained) | hallucinated | lucky guess |
|---|---|---|---|
| **fragment-prompt** (base, same input as adapter) | **13 / 89 = 15%** | **71 / 89 = 80%** | 5 / 89 = 6% |
| v1 adapter (correct by definition) | 89 / 89 = 100% | 0 | 0 |

Fragment-prompt only catches **13 of the 89** abstention cases v1 catches. **The surviving text alone does NOT carry the abstention signal** — the LoRA training is doing the work. The base model, given identical input, hallucinates on 80% of these examples; the adapter abstains correctly on 100%.

This is the cleanest single piece of evidence in the project that the LoRA fine-tuning is producing behavior the base model does not exhibit on the same input. For the paper: the cache-state-conditional behavior is **a learned property of the trained model**, not an emergent property of presenting the model with a truncated context.

## Reading

The B3 success criterion was: v1 / mixture beat fragment-prompt by ≥10pp at ratio=0.50. **Both pass cleanly** (+21pp v1, +25pp mix). The §10.5 sub-analysis goes further: it shows that on the precise examples that motivated v1's cross-compressor finding, fragment-prompt cannot recover the adapter's behavior from the same input.

The relationship is one-directional: the cache representation does **not** carry signal beyond the surviving text in the architectural sense (we don't generate from a compressed cache). Rather, the surviving-text input alone is **insufficient** for the abstention decision; the LoRA training is what enables a model to recognize broken-cache fragments and respond appropriately. Fragment-prompt has access to identical information and does not learn to use it.

## Implications for B2

B2 (prompt-only abstention) is the natural follow-up: if a strong system prompt instructing the model to abstain when context is insufficient closes the gap, then the LoRA training contribution is partly "discoverable via prompting." If P3 (the strong abstention prompt) does not close the gap, the LoRA training contribution is locked.

B3 result alone does not tell us this — base + raw fragment prompt is weak, but base + raw fragment + abstention instruction might be much better. B2 follows next in the experiment queue.

## Files

- `artifacts/v1_eval.jsonl` — the source data; the `base` column IS the fragment-prompt baseline
- `artifacts/v1_eval_mixture_kvzip.jsonl` — mixture adapter on the same val
- `artifacts/v1_eval_snapkv.jsonl` — v1 SnapKV adapter eval (used for the 89-key intersection)
- This file
