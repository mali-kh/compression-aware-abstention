"""Block 4 sampler — stratified 200/class from train_v1.jsonl, seed 42.

Per-(label, ratio) cell counts in train_v1.jsonl are skewed:
  CONFIDENT: 0 / 1 / 11 / 31 / 142 / 616 across r ∈ {0.05..0.80}
  ABSTAIN:  508 / 497 / 382 / 311 / 153 /  17

Strict 33-per-cell stratification is impossible (CONFIDENT at r=0.05/0.10
is empty/near-empty; ABSTAIN at r=0.80 has only 17). Algorithm:
  1. Try base = target/6 per cell.
  2. For deficit cells (avail < base), take all available.
  3. For surplus cells, take `base + share-of-(target - taken-so-far)`,
     biased toward cells with the most remaining capacity.

Seed 42 (numpy random.Generator). Saves to artifacts/block4_sample.jsonl
with the original record + a `stratum` field for downstream grouping.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = ROOT / "artifacts" / "train_v1.jsonl"
OUT_JSONL = ROOT / "artifacts" / "block4_sample.jsonl"
SEED = 42
TARGET_PER_CLASS = 200
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]


def allocate(target: int, avail_per_ratio: list[int]) -> list[int]:
    n_cells = len(avail_per_ratio)
    base = target // n_cells
    plan = [min(base, a) for a in avail_per_ratio]
    while sum(plan) < target:
        deficit = target - sum(plan)
        # remaining capacity per cell
        cap = [avail_per_ratio[i] - plan[i] for i in range(n_cells)]
        if max(cap) == 0:
            break
        # add to the cell with most remaining capacity
        i = max(range(n_cells), key=lambda j: cap[j])
        take = min(deficit, cap[i])
        plan[i] += take
    return plan


def main():
    rng = np.random.default_rng(SEED)
    print(f"[setup] reading {TRAIN_JSONL.relative_to(ROOT)} ...", flush=True)
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for line in TRAIN_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # normalize ratio to canonical bucket if floating-point drift
        ratio = float(r["ratio"])
        ratio = min(RATIOS, key=lambda x: abs(x - ratio))
        key = (r["label_type"], ratio)
        pool[key].append(r)

    print("\n[stratification] per-(label, ratio) counts available:", flush=True)
    for label in ("confident_answer", "abstain"):
        for r in RATIOS:
            n = len(pool.get((label, r), []))
            print(f"  ({label}, {r:.2f}): {n}")

    sample: list[dict] = []
    print(f"\n[allocation] target = {TARGET_PER_CLASS} per class\n", flush=True)
    for label in ("confident_answer", "abstain"):
        avail = [len(pool.get((label, r), [])) for r in RATIOS]
        plan = allocate(TARGET_PER_CLASS, avail)
        print(f"  {label}: avail={avail}, plan={plan}, sum={sum(plan)}", flush=True)
        for r, n_take in zip(RATIOS, plan):
            cell = pool.get((label, r), [])
            if not cell or n_take == 0:
                continue
            idx = rng.choice(len(cell), size=n_take, replace=False)
            for i in sorted(idx):
                rec = cell[i]
                rec_with_stratum = dict(rec)
                rec_with_stratum["_block4_stratum"] = {
                    "label_type": label, "ratio": r,
                }
                sample.append(rec_with_stratum)

    print(f"\n[done] sampled {len(sample)} examples (seed={SEED})", flush=True)
    print(f"  saving -> {OUT_JSONL.relative_to(ROOT)}", flush=True)
    with OUT_JSONL.open("w") as f:
        for rec in sample:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())
