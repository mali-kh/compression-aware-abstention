# Compression-Aware Abstention

Code, evaluation outputs, and reproduction instructions for **"Compression-Aware
Abstention: Teaching LLMs to Refuse When KV-Compression Masks Remove Answer
Evidence"** (GroundLM workshop, EMNLP 2026).

KV-cache compression evicts context tokens to save inference memory. When the
evicted tokens happen to carry the evidence a query depends on, the model is left
with a fluent, on-topic context that no longer determines the answer, and it
tends to fabricate one. This repository trains a small LoRA adapter to do the
opposite: answer when the supporting evidence survives compression, and refuse
when it does not.

## What is here

| Path | Contents |
|---|---|
| `src/` | Compressor wrappers (KVzip, kvpress), tight-span labeler, LoRA trainer, evaluation harness |
| `scripts/` | Dataset construction, training, and per-experiment analysis entry points |
| `configs/` | Training configurations (`configs/train/v1.yaml` is the headline run) |
| `tests/` | Unit tests for label construction and the KVzip wrapper |
| `artifacts/eval/` | Per-example model outputs and classifications for every experiment in the paper |
| `artifacts/logs/` | Analysis logs for the compressed-cache, mechanism, judge, and answerability experiments |
| `artifacts/reports/` | Internal experiment write-ups, included verbatim as provenance |

Trained adapters and the compression-labeled datasets are hosted separately
because of their size; see [Trained adapters and data](#trained-adapters-and-data).

## Verifying the paper's numbers

Every table and headline statistic in the paper is recomputed from the released
per-example outputs by a single script. It re-derives honest accuracy, exact
McNemar tests, paired bootstrap intervals, Wilson intervals, and Cohen's kappa
from the raw classifications rather than reading any cached summary:

```bash
python scripts/verify_paper_claims.py
```

This requires no GPU, no model weights, and no dependencies beyond the Python
standard library. It prints one line per check and exits non-zero if any value
fails to reproduce.

```
202/202 checks reproduce the published values.
```

`docs/CLAIMS.md` maps each table, figure, and numbered claim in the paper to the
specific file that backs it.

## Method in brief

1. **Tight-span localization.** For each MuSiQue 2-hop example, the
   `question_decomposition` field gives each single-hop sub-answer as a short
   atomic string. Locating it verbatim in its supporting paragraph yields a
   short span (median 3 tokens). Paragraph-level spans (~100 tokens) average out under
   compression and produce zero label variance at the mid retention ratio; tight
   spans are what make the learning problem well-posed.
2. **Labeling.** A compressor produces a per-token survival mask at retention
   ratio *r*. An example is `CONFIDENT` when every supporting span retains at
   least `tau_high = 0.8` of its tokens, `ABSTAIN` when any span falls below
   `tau_low = 0.3`, and dropped otherwise.
3. **Training.** A rank-16 LoRA adapter (10.1M parameters, attention projections
   only) on a frozen Qwen2.5-7B-Instruct, trained to emit the gold answer on
   `CONFIDENT` examples and a fixed refusal string on `ABSTAIN` examples.

Prompt-style KVzip masks come from the vendored reference implementation, not
from kvpress: kvpress's `KVzipPress` defers eviction in a way that is
incompatible with prompt-style mask extraction. Expected Attention and SnapKV
use kvpress, and the compressed-cache experiments use `KVzipPress`.

## Reproducing from scratch

Requires a single 80GB GPU (the paper's runs used an H100).

```bash
uv sync
uv pip install flash-attn==2.7.4.post1 --no-build-isolation   # after uv sync, not during
```

KVzip is vendored rather than installed; clone it to `third_party/KVzip` at
commit `5d8472975`. Note that `transformers` is pinned to 4.51.3 because
KVzip's `RetainCache`/`EvictCache` rely on the pre-4.55 `DynamicCache` API,
while kvpress needs a newer `transformers`; the two run in separate virtual
environments (`.venv` and `.venv-kvpress`).

```bash
# 1. Build the compression-labeled dataset (KVzip masks, six retention ratios)
python scripts/build_training_set.py --seed 42

# 2. Train and evaluate the headline adapter (LR sweep, 3 epochs, best by val loss)
python scripts/run_v1_full.py --seed 42

# 3. Cross-compressor evaluation on held-out compressors
python scripts/regen_val_kvpress.py --press expected_attention
python scripts/eval_cross_compressor.py

# 4. Multi-compressor mixture training
python scripts/build_mixture_merge.py
python scripts/run_v1_mixture_compressed.py

# 5. Compressed-cache inference and cc-training
python scripts/eval_compressed_cache.py     # run from .venv-kvpress
```

Individual experiments have their own entry points: `block10_eval.py` and
`block10_pivot.py` for the controlled-deletion mechanism tests, `block4_*.py`
for the LLM-judge label validation, `block6_*.py` for answerability
preservation, and `exp4_ruler_eval.py` for the long-context evaluation.

## Trained adapters and data

Adapter weights (~400MB) and the compression-labeled datasets (~1GB, the masks
dominate) are on the Hugging Face Hub:

- Adapters: <https://huggingface.co/moalikhoda/compression-aware-abstention-adapters>
- Datasets: <https://huggingface.co/datasets/moalikhoda/compression-aware-abstention-data>

The adapters released are the best checkpoints for each configuration reported
in the paper: v1 (KVzip-only) at seeds 42/43/44, the multi-compressor mixture at
seeds 42/43, the compressed-cache-trained mixture at seeds 42/43, the
single-compressor cc-trained ablation, the random-mask probe, and the
Llama-3.1-8B replication.

## Scope and known limits

These are stated at greater length in the paper's Limitations section, and are
repeated here because they bound what this code supports:

- Training data is MuSiQue 2-hop QA only. The tight-span construction depends on
  `question_decomposition` annotations, which many QA datasets do not have.
- Aggregate honest accuracy is exploitable: because the gold label distribution
  is U-shaped in the retention ratio, a policy that never inspects the context
  can beat the trained adapters on aggregate by thresholding on *r* alone. The
  per-ratio and class-conditional slices are the meaningful comparisons.
- Tight-span survival is a surrogate for contextual sufficiency, not ground
  truth. Against an LLM-judge sufficiency verdict it reaches Cohen's kappa 0.61;
  disagreements concentrate on multi-hop questions where the answer string
  survives but the bridging evidence does not.
- Compressed-cache evaluation uses kvpress's `KVzipPress` only. Other presses
  take different eviction and position-indexing paths, and cross-compressor
  generalization under compressed-cache inference is untested.

## Citation

```bibtex
@inproceedings{khodabandehlou2026compression,
  title     = {Compression-Aware Abstention: Teaching {LLM}s to Refuse When
               {KV}-Compression Masks Remove Answer Evidence},
  author    = {Khodabandehlou, Mohammadali and Krishnamachari, Bhaskar},
  booktitle = {Proceedings of the Workshop on Grounding and Reliability of
               Language Models (GroundLM) at EMNLP},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). MuSiQue is distributed
by its authors under CC BY 4.0; RULER, kvpress, and KVzip retain their upstream
licenses.
