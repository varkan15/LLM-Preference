# Review and Fixes — Classical Bradley-Terry Baseline

Review of the Run 2 code (`bt_baseline.py`, `features.py`, `data.py`), the two
fixes applied, and the seed-0 rerun after the fixes.

## 1. Summary

- The model is sound: Bradley-Terry logistic regression on feature differences,
  no intercept, mirrored training pairs, prompt-disjoint split, TF-IDF fit on train
  only, paired bootstrap for model comparisons.
- Two sampling bugs were found and fixed. One of them changed a headline conclusion:
  **the HH-RLHF transfer result was measured on harmlessness pairs only.** After the
  fix, both models score at chance on HH-RLHF, and HH does *not* prefer shorter
  responses overall.
- UltraFeedback results barely moved (59.2% full, 56.5% → 56.8% length-only).
- Five smaller issues are listed in Section 4 and left unfixed.

## 2. Bugs fixed

### 2.1 HH-RLHF transfer sampled harmlessness pairs only

**Problem.** `load_hh(limit=2000)` took the first 2,000 rows of the HH-RLHF test
split. That split concatenates four subsets in order:

| Subset | Test rows | Row range |
|---|---|---|
| harmless-base | 2,312 | 0 – 2,311 |
| helpful-base | 2,354 | 2,312 – 4,665 |
| helpful-online | 1,137 | 4,666 – 5,802 |
| helpful-rejection-sampled | 2,749 | 5,803 – 8,551 |

So all 2,000 evaluated pairs came from harmless-base.

**Verified by:** loading the full test split and the `harmless-base` subset
separately and checking that rows 0–1,999 match exactly (`True`).

**Fix (`data.py`).** Shuffle the cleaned rows with a fixed seed before applying
`limit`. The seed is fixed at 0 and independent of `--seed`, so every run scores the
same HH pairs and differences between seeds come from the model, not the sample.
Shuffling happens after rows are built, so `hh-test-{i}` IDs keep their original
row index.

```python
rows = _clean(rows)
random.Random(seed).shuffle(rows)
return rows[:limit] if limit else rows
```

**Verified by:** mapping each sampled ID back to its subset. The 2,000-pair sample is
now 558 harmless-base, 529 helpful-base, 240 helpful-online, 673
helpful-rejection-sampled. Harmless-base is 28% of the sample vs 27% of the full split.

### 2.2 Frozen test set depended on `--max-train`

**Problem.** One random generator subsampled train, then val, then test. With
`--max-train N`, the train draw consumed random numbers first, so a different 4,000
test pairs were saved for the same seed. That breaks the promise that
`results/splits/uf_test_seed{N}.jsonl` is a fixed test set for comparing against DPO.

**Fix (`bt_baseline.py`).** One independent generator per split:

```python
def subsample(xs, n, k):
    rng = np.random.default_rng((args.seed, k))
    ...
train = subsample(train, args.max_train, 0)
val = subsample(val, args.max_val, 1)
test = subsample(test, args.max_test, 2)
```

The test sample now depends only on `--seed` and `--max-test`.

## 3. Rerun results (seed 0, default arguments)

Command: `python bt_baseline.py --seed 0` (CPU, ~40–75 s). Full HH split:
`--hh-test 100000`.

### 3.1 UltraFeedback (n = 4,000, prompt-disjoint)

| Model | Before | After |
|---|---|---|
| BT full | 59.2% [57.6, 60.6] | 59.2% [57.7, 60.8] |
| BT length-only | 56.5% [54.9, 58.0] | 56.8% [55.1, 58.3] |
| Full − length-only (paired) | +2.7 [+1.0, +4.5] | +2.4 [+0.7, +4.2] |
| Selected C | 1.0 | 0.01 |

The shifts come from fix 2.2, which draws a different 4,000 test pairs from the same
prompt-hash split. Validation accuracy is flat across the C grid (59.28–59.35%), so
the change in C is noise. Length buckets and learned weights moved by about 1 point.

### 3.2 HH-RLHF transfer

| Model | Before (first 2,000) | After (2,000 seeded sample) | After (all 8,505) |
|---|---|---|---|
| BT full | 48.2% [46.1, 50.3] | 51.9% [49.8, 54.3] | 51.0% [49.9, 52.0] |
| BT length-only | 43.5% [41.2, 45.8] | 52.5% [50.2, 54.8] | 51.9% [50.9, 53.0] |
| Chosen longer / shorter | 36% / 49% | 44% / 39% | 43% / 38% |

**What changed in the conclusions:**

- **Before:** "HH-RLHF has the opposite length skew; short harmless refusals are
  preferred, so the length rule scores below chance." This was true of harmless-base
  only.
- **After:** across all of HH-RLHF the chosen response is longer more often than
  shorter, the same direction as UltraFeedback but weaker. Length-only lands just
  above chance; the full model is at chance and no better than length alone.
  Surface features learned on UltraFeedback carry almost no signal on HH-RLHF.

`BT_Results.md` and `results/bt_seed0.json` were updated to the after numbers.

## 4. Issues found but not fixed

| Issue | Where | Why it matters | Suggested fix |
|---|---|---|---|
| `n_lines`, `n_bullets` are raw counts | `features.py` | Heavy-tailed; one long code dump or list dominates the margin. Several worst errors are list- or screenplay-formatted responses. | `np.log1p` on both |
| Margin of exactly 0 counts as wrong | `evaluate()` | Penalises length-only when token counts tie | Score ties as 0.5, or report how many there are |
| `jaccard` and `overlap_frac` nearly duplicate | `features.py` | Opposite-signed weights (−0.49 / +0.31) that are hard to interpret | Drop one |
| C grid has no effect | `run()` | Validation accuracy flat across all C; tuning adds nothing | Fix C, say so in one line |
| No tests; Windows-only setup instructions | repo | Symmetry and split properties unchecked | Port a small test for `margin(A,B) = −margin(B,A)` and split disjointness |

## 5. Cross-check against Run 1

The Run 1 implementation (different features, official HF train/test split,
895 non-tie test pairs) gives 60.4% [57.3, 63.7] full and 56.4% [53.2, 59.8]
length-only. Both implementations agree on about 60% for surface features with
length explaining most of it. They use different test sets; which one is the
headline for the Phase 2 report is still open.
