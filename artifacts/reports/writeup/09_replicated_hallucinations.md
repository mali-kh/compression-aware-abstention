# 09 — Hallucination intersection across the three seeds

Source: `artifacts/v1_eval.jsonl`, `v1_eval_seed43.jsonl`, `v1_eval_seed44.jsonl`. Filtered for adapter `classification == "hallucinated"` (i.e. ABSTAIN-gold examples where the adapter confidently emitted a wrong specific answer).

## Headline

| set | size |
|---|---|
| seed 42 hallucinated | 7 |
| seed 43 hallucinated | 7 |
| seed 44 hallucinated | 7 |
| pairwise overlap (s42 ∩ s43) | 5 / 7 |
| pairwise overlap (s42 ∩ s44) | 5 / 7 |
| pairwise overlap (s43 ∩ s44) | 5 / 7 |
| **3-way intersection** | **4 / 7** |
| union over all three seeds | 10 |

The hallucination *count* of 7 is robust to seed (see `08_seed_variance.md`), but the **identity** of those 7 examples is mostly seed-specific. Four examples replicate exactly across all three seeds — these are the "stable hard cases" — and the remaining 3 per seed are seed-specific failure modes.

## The 4 stable cases (replicate on every seed)

| id | ratio | n_kept / ctx_len | question | gold |
|---|---|---|---|---|
| `2hop__18025_34503` | 0.50 | 1873 / 3746 | Who did Swansea City play in its first game in the league whose trophy Arsenal won in the 2003-2004 season? | Wigan Athletic |
| `2hop__270350_10984` | 0.80 | 2036 / 2545 | What saw the start of the newspaper that employs David Gonzalez? | 1851 |
| `2hop__42152_20556` | 0.50 | 1520 / 3039 | What abolished slavery throughout the sovereignty imperialism is most often associated with? | The Slavery Abolition Act |
| `2hop__92103_29346` | 0.80 | 1782 / 2228 | What shifted attention from the controversy over the required service of the president who was the son of another president, who signed the Civil Rights Act of 1991? | a segment on 60 Minutes |

### What the adapter says on the stable-4 (per seed)

| id | seed 42 response | seed 43 response | seed 44 response |
|---|---|---|---|
| `2hop__18025_34503` | "Liberty Stadium" | "Liberty Stadium" | "Liberty Stadium" |
| `2hop__270350_10984` | "North Eugene High School" | "North American newspapers" | "North Eugene High School" |
| `2hop__42152_20556` | "the British Empire" | "Slave Trade Act" | "the British Empire" |
| `2hop__92103_29346` | "the Killian documents" | "Killian documents" | "a segment aired by CBS News" |

A few seeds give exactly the same wrong answer (UHF-style: "Liberty Stadium" three times; "the British Empire" twice; "Killian documents" twice). The 4th example is more interesting — seed 44 produces a near-paraphrase of the gold answer ("a segment aired by CBS News" vs gold "a segment on 60 Minutes") which the substring-match classifier scores as wrong. Worth flagging for the discussion.

## Reading for the paper

Discussion paragraph candidate:

> Of the 7 examples on which the adapter hallucinates per seed, only 4 are the same across all three runs. The remaining 3 are seed-specific — different runs fail on different examples. The 4 stable cases are concentrated at high retention (ratio=0.50 and 0.80), share a common structure (questions requiring multi-hop integration of facts that the kept-token fragment includes only partially), and elicit highly-specific wrong answers across seeds (e.g. "Liberty Stadium" for the Wigan Athletic question on all three seeds). One of the 4 stable cases (seed 44 on the Killian documents question) produces a near-paraphrase of the gold answer that the substring-match classifier scores as wrong; under a more permissive matcher the 3-way replicated count would be 3, not 4. The stable-4 are good candidates for inclusion in a v2 high-fill-counter-instance training set, since they are the cases the model demonstrably fails on regardless of training stochasticity.
