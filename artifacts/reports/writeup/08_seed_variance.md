# 08 — Seed-variance comparison (n=3 seeds: 42, 43, 44)

Three independent training runs with seeds {42, 43, 44}. Identical setup otherwise. Adapter eval against `val_v1.jsonl` using KVzip-evicted caches. Base eval is shared across seeds (the base model does not change). Patch: `scripts/run_v1_full.py` commit `f65fe20` adds `--seed` and `--skip-base-eval`.

## Sweep-picked hyperparameters per seed

| seed | best LR | best epoch | best val_loss |
|---|---|---|---|
| 42 | 2e-4 | 1 | 0.2165 |
| 43 | 2e-4 | 1 | 0.2234 |
| **44** | **5e-5** | **2** | 0.2192 |

Seed 44 picks a different LR and a different best-epoch from seeds 42 and 43. All three converge to val_loss ≈ 0.22 nonetheless, and to essentially the same per-ratio adapter behavior (table below). The variance lives in the *trajectory*, not the *destination*.

## Per-ratio honest_accuracy / hallucination_rate

(base shared across seeds; only adapter changes; KVzip val)

| ratio | n | gC/gA | base hon / hal | seed 42 hon / hal | seed 43 hon / hal | seed 44 hon / hal |
|---|---|---|---|---|---|---|
| 0.05 | 100 | 0/100 | 0.15 / 0.75 | **1.00 / 0.00** | **1.00 / 0.00** | **1.00 / 0.00** |
| 0.10 | 95 | 1/94 | 0.27 / 0.63 | 0.99 / 0.00 | 0.99 / 0.00 | 0.99 / 0.00 |
| 0.20 | 82 | 2/80 | 0.30 / 0.60 | 0.87 / 0.00 | 0.96 / 0.00 | 0.94 / 0.00 |
| 0.30 | 65 | 3/62 | 0.35 / 0.55 | 0.78 / 0.02 | 0.80 / 0.00 | 0.80 / 0.03 |
| **0.50** | **48** | **22/26** | **0.44 / 0.42** | **0.65 / 0.15** | **0.69 / 0.10** | **0.65 / 0.12** |
| 0.80 | 73 | 69/4 | 0.51 / 0.33 | 0.60 / 0.37 | 0.60 / 0.34 | 0.59 / 0.33 |
| **AGG** | **463** | **97/366** | **0.32 / 0.57** | **0.84 / 0.08** | **0.87 / 0.06** | **0.86 / 0.07** |

## Δhonest_accuracy spread per ratio

| ratio | s42 Δhon | s43 Δhon | s44 Δhon | min | max | spread |
|---|---|---|---|---|---|---|
| 0.05 | +0.85 | +0.85 | +0.85 | +0.85 | +0.85 | 0.00 |
| 0.10 | +0.72 | +0.72 | +0.72 | +0.72 | +0.72 | 0.00 |
| 0.20 | +0.56 | +0.66 | +0.63 | +0.56 | +0.66 | **0.10** |
| 0.30 | +0.43 | +0.45 | +0.45 | +0.43 | +0.45 | 0.02 |
| 0.50 | +0.21 | +0.25 | +0.21 | +0.21 | +0.25 | 0.04 |
| 0.80 | +0.10 | +0.10 | +0.08 | +0.08 | +0.10 | 0.01 |
| AGG | +0.53 | +0.55 | +0.54 | +0.53 | +0.55 | 0.02 |

Per-ratio Δhon spread is ≤ 4pp at every meaningful slice except ratio=0.20, where it is 10pp (every seed still scores ≥ 0.87 honest_accuracy at that slice vs base 0.30). Aggregate spread is 2pp.

## Aggregate class-count delta — the "232 → 7" replication check

| class | base | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| correct | 147 | 391 | 402 | 397 |
| **hallucinated** | **232** | **7** | **7** | **7** |
| wrong_answer | 32 | 28 | 23 | 25 |
| overcautious | 18 | 15 | 17 | 20 |
| lucky_guess | 34 | 22 | 14 | 14 |

Hallucinated count lands at exactly 7 on all three seeds.

## Reading for the paper

Inline robustness paragraph in the methods section:

> Three independent training runs (seeds {42, 43, 44}, identical setup otherwise) confirm that the headline result is not seed-specific. Per-ratio honest accuracy spread across seeds is ≤ 4pp at every meaningful slice (max 10pp at ratio=0.20, where every seed still scores at least 0.87 vs base 0.30). The aggregate hallucination count drops to exactly 7 on all three seeds (from base 232). The sweep-picked learning rate varies across seeds (2e-4 on seeds 42 and 43, 5e-5 on seed 44) but the resulting per-ratio adapter behavior is essentially identical, indicating that 1-epoch sweep validation loss is too noisy to reliably select an LR while the resulting model converges to similar per-ratio metrics regardless.

The sweep-picked-LR variance is itself worth flagging — see also `02_facts_check.md`. v2 should either widen the sweep (more epochs per LR before deciding) or skip the sweep entirely and pick a fixed LR.
