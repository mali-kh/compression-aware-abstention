# 04 — Cross-compressor per-ratio table (four columns, paste-ready)

This is the version to cite. All four compressors run at the same six retention ratios; topk reduction calibrated so `mask.mean() ≈ ratio` exactly (calibration Δ = 0.00pp at every cell, n=100 examples).

The v1 adapter (`artifacts/v1_lora_best/`, seed 42, epoch 1, lr=2e-4) is the same in every column — only the kept-token mask the adapter sees changes.

## Headline: per-ratio Δhonest_accuracy (adapter − base)

| ratio | n_KVzip / n_ExpAttn / n_SnapKV-ctx / n_SnapKV-nat | KVzip | ExpAttn | SnapKV (ctx-only) | SnapKV (native) |
|---|---|---|---|---|---|
| 0.05 | 100 / 98 / 100 / 100 | +0.85 | +0.70 | +0.39 | +0.40 |
| 0.10 | 95 / 89 / 93 / 92 | +0.72 | +0.66 | +0.26 | +0.30 |
| 0.20 | 82 / 68 / 75 / 71 | +0.56 | +0.59 | +0.17 | +0.11 |
| 0.30 | 65 / 51 / 58 / 48 | +0.43 | +0.35 | +0.12 | +0.04 |
| **0.50** | **48 / 42 / 39 / 37** | **+0.21** | **+0.29** | **+0.13** | **+0.03** |
| 0.80 | 73 / 66 / 66 / 71 | +0.10 | +0.08 | +0.05 | +0.04 |

n varies per compressor because each compressor produces a different number of DROP rows after labeling (KVzip 0, ExpAttn 49, SnapKV-ctx 32, SnapKV-native 44 across all ratios; DROP rows are filtered before eval because there is no defined gold behavior).

## Full per-ratio rates (all of base+adapter, hon/hal/abst)

| ratio | compressor | n | base hon / hal / abst | adapter hon / hal / abst |
|---|---|---|---|---|
| 0.05 | KVzip | 100 | 0.15 / 0.75 / 0.15 | **1.00 / 0.00 / 1.00** |
| 0.05 | ExpAttn | 98 | 0.21 / 0.70 / 0.21 | 0.92 / 0.05 / 0.92 |
| 0.05 | SnapKV (ctx) | 100 | 0.25 / 0.65 / 0.25 | 0.64 / 0.24 / 0.64 |
| 0.05 | SnapKV (native) | 100 | 0.27 / 0.65 / 0.27 | 0.67 / 0.21 / 0.67 |
| 0.10 | KVzip | 95 | 0.27 / 0.63 / 0.27 | **0.99 / 0.00 / 1.00** |
| 0.10 | ExpAttn | 89 | 0.22 / 0.72 / 0.22 | 0.89 / 0.08 / 0.89 |
| 0.10 | SnapKV (ctx) | 93 | 0.30 / 0.55 / 0.30 | 0.56 / 0.27 / 0.56 |
| 0.10 | SnapKV (native) | 92 | 0.32 / 0.59 / 0.31 | 0.62 / 0.22 / 0.61 |
| 0.20 | KVzip | 82 | 0.30 / 0.60 / 0.30 | **0.87 / 0.00 / 0.89** |
| 0.20 | ExpAttn | 68 | 0.31 / 0.56 / 0.29 | 0.90 / 0.04 / 0.88 |
| 0.20 | SnapKV (ctx) | 75 | 0.37 / 0.49 / 0.35 | 0.55 / 0.29 / 0.53 |
| 0.20 | SnapKV (native) | 71 | 0.37 / 0.55 / 0.29 | 0.48 / 0.34 / 0.39 |
| 0.30 | KVzip | 65 | 0.35 / 0.55 / 0.37 | **0.78 / 0.02 / 0.83** |
| 0.30 | ExpAttn | 51 | 0.37 / 0.53 / 0.35 | 0.73 / 0.04 / 0.76 |
| 0.30 | SnapKV (ctx) | 58 | 0.33 / 0.55 / 0.26 | 0.45 / 0.36 / 0.45 |
| 0.30 | SnapKV (native) | 48 | 0.38 / 0.57 / 0.20 | 0.42 / 0.33 / 0.27 |
| **0.50** | **KVzip** | **48** | **0.44 / 0.42 / 0.31** | **0.65 / 0.15 / 0.58** |
| **0.50** | **ExpAttn** | **42** | **0.38 / 0.38 / 0.33** | **0.67 / 0.24 / 0.36** |
| **0.50** | **SnapKV (ctx)** | **39** | **0.33 / 0.51 / 0.21** | **0.46 / 0.38 / 0.28** |
| **0.50** | **SnapKV (native)** | **37** | **0.38 / 0.49 / 0.27** | **0.41 / 0.41 / 0.14** |
| 0.80 | KVzip | 73 | 0.51 / 0.33 / 0.18 | 0.60 / 0.37 / 0.03 |
| 0.80 | ExpAttn | 66 | 0.48 / 0.38 / 0.14 | 0.56 / 0.41 / 0.03 |
| 0.80 | SnapKV (ctx) | 66 | 0.50 / 0.27 / 0.18 | 0.55 / 0.42 / 0.02 |
| 0.80 | SnapKV (native) | 71 | 0.52 / 0.33 / 0.18 | 0.56 / 0.44 / 0.01 |

## Aggregate (footnote, sensitive to ratio sampling — see memo §3)

| compressor | n | base hon | adapter hon | Δhon | base hal → adapter hal |
|---|---|---|---|---|---|
| KVzip | 463 | 0.32 | 0.84 | +0.53 | 0.57 → 0.08 |
| ExpAttn | 414 | 0.31 | 0.80 | +0.49 | 0.58 → 0.13 |
| SnapKV (ctx-only) | 431 | 0.34 | 0.55 | +0.21 | 0.52 → 0.31 |
| SnapKV (native) | 419 | 0.36 | 0.56 | +0.20 | 0.51 → 0.30 |

## Compressor configurations used

- **KVzip**: vendored at commit `5d8472975` (`third_party/KVzip/`). Reduction = `topk` (top-k tokens by per-(layer, head) survival fraction). `kv_type="retain"`.
- **Expected Attention**: kvpress 0.5.3 `ExpectedAttentionPress` with defaults: `n_sink=4`, `n_future_positions=512`, `use_covariance=True`, `use_vnorm=True`, `epsilon=0.0`.
- **SnapKV (context-only)**: kvpress 0.5.3 `SnapKVPress` with defaults: `window_size=64`, `kernel_size=5`. Prefill input is context tokens only (no question), to mirror v1's KVzip pipeline.
- **SnapKV (native)**: same press, prefill input is `context + "\n\nQuestion: {q}\n\nAnswer:"`. Question becomes the recent-window the press scores against, restoring SnapKV's intended query-aware advantage.

The kvpress wrapper subclasses `ScorerPress` to record per-(layer, head, token) survival into a Python dict during the prefill hook, then reduces to a per-token mask via the same topk operation as the KVzip wrapper, so masks across all four compressors live in the same semantic space.
