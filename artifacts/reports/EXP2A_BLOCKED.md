# Exp 2a — BLOCKED: post-balance stratum starvation

## What happened

Mixture dataset construction completed all four stages:

- KVzip raw: 5062 rows (post-DROP)
- ExpAttn raw: 5207 rows
- SnapKV (context-only) raw: 6098 rows
- TOTAL raw, train pool: 14856 rows

Per-(compressor, ratio) 30/70 minority-class balance reduced the train set to **5035 instances**, which is below the spec's [2.5×, 3×] target of [6700, 8000]. Files were written to `artifacts/train_mixture_v1.jsonl` and `artifacts/val_mixture_v1.jsonl`, but **I am surfacing this for review before proceeding to Exp 2b** because multiple per-(compressor, ratio) strata fall below the 30-instance threshold the spec set.

(`build_mixture_merge.py`'s BLOCKED check was on *pre-balance* raw counts, which all exceeded 30. The spec's intent — read more carefully — was post-balance counts. The check's threshold logic is wrong; the underlying problem it was meant to catch is real and is here.)

## Starved post-balance strata

| compressor | ratio | pre-balance (abst / conf) | post-balance | status |
|---|---|---|---|---|
| **kvzip** | 0.05 | 1000 / 0 | **1000 (single class)** | no balance possible — only abstain present |
| **kvzip** | 0.10 | 972 / 1 | **3** | starved |
| **kvzip** | 0.20 | 823 / 11 | 36 | borderline |
| **expected_attention** | 0.05 | 973 / 1 | **3** | starved |
| **expected_attention** | 0.10 | 915 / 7 | **23** | starved |
| **snapkv** | 0.05 | 991 / 6 | **19** | starved |

Healthy strata (post-balance ≥ 100): KVzip @ 0.30/0.50/0.80; ExpAttn @ 0.20/0.30/0.50/0.80; SnapKV @ 0.10/0.20/0.30/0.50/0.80. Eleven of eighteen strata.

## Mechanism

At low compression ratios (0.05, 0.10), almost every example *should* abstain by definition — the supporting tokens are mostly evicted regardless of compressor. So the natural CONFIDENT-gold count at those ratios is near zero (1, 7, 6 in the worst cells). The 30/70 minority-floor balance treats CONFIDENT as the minority and undersamples ABSTAIN to bring CONFIDENT to 30%, which means dropping ~970 ABSTAIN rows to keep ~13 total. This is the spec's stratification doing what it was told, but it converts a structural class-imbalance into a tiny-sample problem in the wrong direction.

This is exactly the v1 limitation §7 already named ("per-ratio class imbalance is structural; at low ratios most examples *should* abstain"). The mixture doesn't change that structure; it just exposes it three times over.

## Options for your review

### Option A — Drop low-ratio strata entirely
Drop the 5 starved strata + the single-class KVzip @ 0.05 cell. Train on remaining 12 strata (mid + high ratios only).

- **Pro**: clean per-stratum 30/70 balance; no tiny-sample noise; aggressive but defensible
- **Pro**: tracks the v1 thesis (mid-ratios are where the cache-state-conditional signal lives)
- **Con**: loses any training exposure to "what an extreme-compression cache looks like"
- **Estimated final train size**: ~3700 (rough; would need re-run to confirm)

### Option B — Lower the minority floor at low ratios only
At ratios 0.05 / 0.10, accept the natural class distribution (no balance). At 0.20+, keep the 30/70 floor.

- **Pro**: keeps the abstain-rich training signal at low ratios (which is the truth — the model SHOULD abstain there)
- **Pro**: no oversampling, no row duplication
- **Con**: low-ratio strata become abstain-dominated (likely 99%+); model sees little CONFIDENT signal at those compression levels (which is the truth)
- **Estimated final train size**: ~10000 (large; the abstain-floods get included)

### Option C — Lower the floor uniformly to 15% or 20%
Apply a single, lower minority-fraction floor across all strata.

- **Pro**: simple; respects the per-stratum framework
- **Con**: weakens the v1 commitment to 30/70 (you may not want this)
- **Con**: still gives small low-ratio cells, just slightly bigger
- **Estimated final train size at 15% floor**: ~6500

### Option D — Pool low-ratio strata across compressors for balancing
Treat "all compressors @ ratio=0.05" as one stratum for balancing purposes (so CONFIDENT count = sum across compressors). At ratio=0.05 that's 0+1+6 = 7 CONFIDENT total → 16 ABSTAIN to hit 30%. Still small, but more than per-compressor.

- **Pro**: respects per-ratio stratification, allows compressors to contribute proportionally
- **Con**: loses per-compressor signal at the pooled levels (the mixture's whole point)
- **Estimated final train size**: ~5500

### Option E — Oversample CONFIDENT minority at starved strata
Duplicate CONFIDENT rows to bring counts up.

- **Pro**: keeps stratum and minority-fraction commitments intact
- **Con**: train sees duplicated examples, risks memorisation; goes against your "no oversampling" framing in v1
- **Estimated final train size**: ~6700

### Option F — Proceed anyway with the 5035-row dataset
Acknowledge the starved strata as a known limitation; train on what we have.

- **Pro**: zero additional compute
- **Con**: tiny strata dominate per-stratum gradient noise; the 5035 size is below your target range; the result interpretation gets murky ("did the mixture training fail because the recipe doesn't work, or because half the strata had n=20?")

## My recommendation

**Option B** — it's the most honest reading of the data. At ratio=0.05 essentially every example *should* abstain regardless of compressor; we shouldn't be balancing that stratum to 30/70 because there's no "decision" to balance. The mid and high ratios — where the cache-state-conditional signal lives — keep the 30/70 floor.

Option B is also the closest match to the v1 dataset construction, which applied the 30/70 floor to the train set as a whole (not per-ratio), so most of the abstain-floods at low ratios *did* get into v1's training set. The mixture-with-Option-B would be a defensible analogue.

If you'd rather a cleaner per-stratum story (every cell at a fixed minority fraction), **Option A** is the next-best — drop the starved cells entirely, accept that the mixture only trains on mid+high ratios. The honest paper framing for that result would be "compressor-mixture training improves transfer in the regimes where multiple cache-state outcomes are possible; we did not train at extreme compression because the class distribution there is structurally one-sided."

I would NOT recommend C (weakens the project's standing 30/70 commitment), D (defeats the per-stratum design), E (the only oversampling decision in the project; deserves explicit re-authorization), or F (the worst-of-both-worlds in interpretability).

## What I'm doing while you decide

Halting Exp 2b/2c (no training on the 5035-row dataset). Proceeding with Exp 3 (Llama replication) and Exp 4 (RULER) in the meantime — both are independent of Exp 2.

When you return:
- If you choose Option A, B, or C: I'll re-run Stage 4 (`build_mixture_merge.py`) with the new logic and proceed to Exp 2b.
- If you choose D or E: I'll need a small code change to the merge script first.
- If you choose F: I'll proceed with the existing dataset and document the limitation in the Exp 2 report.

The raw stage outputs (`_mixture_raw_*.jsonl`) are kept on disk so any of the above can be applied without re-running the GPU stages (saves ~25 min per re-run).
