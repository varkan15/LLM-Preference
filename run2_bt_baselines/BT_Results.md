# Classical Baseline Results — Bradley-Terry over Surface Features

CS F429 NLP · Phase 2 · Classical baseline (baseline 1 of 4 in the Phase 1 plan)

## Summary

| Model | UltraFeedback test acc | 95% CI | HH-RLHF transfer acc | 95% CI |
|---|---|---|---|---|
| Random | 50.0% | — | 50.0% | — |
| BT length-only | 56.8% | [55.1, 58.3] | 52.5% | [50.2, 54.8] |
| **BT full (6 features)** | **59.2%** | **[57.7, 60.8]** | **51.9%** | **[49.8, 54.3]** |

- The full surface-feature model beats the length-only control by **+2.4 points**
  (paired bootstrap 95% CI [+0.7, +4.2]). The CI excludes 0, so the non-length
  features add a small but real signal.
- Most of the model's accuracy is still the **length confound**. When the chosen
  response is the shorter one, accuracy falls to 36.6%, well below chance.
- The model **does not transfer** to HH-RLHF: both variants are at chance level
  (51.9% / 52.5%). Surface features carry almost no preference signal there.

## Setup

| Item | Value |
|---|---|
| Training corpus | `trl-lib/ultrafeedback_binarized` (HF train + test pooled, then re-split) |
| Filtering | dropped tied-score, empty, identical-response and duplicate pairs |
| Split | prompt-disjoint by MD5 hash of the prompt: 80 / 10 / 10, salt = seed |
| Pairs used | train 43,973 · val 4,000 · test 4,000 (val/test subsampled to 4,000) |
| Transfer corpus | `Anthropic/hh-rlhf`, test split, 2,000 pairs sampled with a fixed shuffle (seed 0, independent of `--seed`; evaluation only, no refitting) |
| Seed | 0 |
| Hardware | CPU only; full run ≈ 40–75 s |

### Model

$$P(A \succ B \mid x) = \sigma\big(w^\top (f(x, A) - f(x, B))\big)$$

- Logistic regression on **feature differences**, with **no intercept** and features
  scaled without mean-centering, so `margin(A, B) = −margin(B, A)`.
- **Symmetric augmentation:** each pair is added both as `(c − r, 1)` and as `(r − c, 0)`.
- **Regularization:** `C` was chosen on the validation split from {0.01, 0.1, 1, 10}.
  Validation accuracy was flat (59.28–59.35%), and C = 0.01 was selected.
- **Length-only control:** the same model restricted to the `log_len` feature.

### Features

All features are computed per `(prompt, response)`. TF-IDF is fit on training text only.

| Feature | Definition |
|---|---|
| `log_len` | log(1 + #tokens), using the `Qwen/Qwen2.5-1.5B-Instruct` tokenizer |
| `n_lines` | number of lines in the response |
| `n_bullets` | lines starting with `-`, `*`, `•`, `1.` or `1)` |
| `jaccard` | word-set Jaccard similarity between prompt and response |
| `overlap_frac` | fraction of prompt words that appear in the response |
| `tfidf_cos` | TF-IDF cosine similarity between prompt and response |

## Results

### UltraFeedback held-out test (n = 4,000, prompt-disjoint)

| Model | Accuracy | 95% bootstrap CI |
|---|---|---|
| BT full | 59.2% | [57.7, 60.8] |
| BT length-only | 56.8% | [55.1, 58.3] |
| Full − length-only (paired) | +2.4 pts | [+0.7, +4.2] |

### Accuracy by length difference

Buckets are based on the log-token-length difference (chosen − rejected), with a threshold of ±0.2.

| Bucket | n | BT full | BT length-only |
|---|---|---|---|
| Chosen shorter | 1,319 | 36.6% | 0.0% |
| Similar length | 817 | 56.8% | 49.8% |
| Chosen longer | 1,864 | 76.2% | 100.0% |

- **Length-only model:** it is a pure "pick the longer response" rule, so it gets 0% / 100%
  in the two outer buckets.
- **Full model:** it gives up some accuracy on chosen-longer pairs to win back part of the
  chosen-shorter pairs. Its main gain is on the **similar-length** bucket, where length
  cannot help (56.8% vs 49.8%).
- **UltraFeedback length skew:** the chosen response is longer in 47% of test pairs and
  shorter in 33%.

### Cross-corpus transfer: HH-RLHF (n = 2,000, model trained on UltraFeedback only)

| Bucket | n | BT full | BT length-only |
|---|---|---|---|
| **Overall** | 2,000 | **51.9%** [49.8, 54.3] | **52.5%** [50.2, 54.8] |
| Chosen shorter | 775 | 54.8% | 0.0% |
| Similar length | 348 | 57.8% | 49.7% |
| Chosen longer | 877 | 47.1% | 100.0% |

The HH-RLHF test split is four subsets concatenated in order (harmless-base first), so the
2,000 pairs are drawn after a fixed shuffle to cover all four. On this mix the chosen response
is longer in 44% of pairs and shorter in 39%, the same direction as UltraFeedback but weaker,
so "pick the longer response" lands just above chance. The full model does no better overall:
it gains on similar-length pairs but loses on chosen-longer ones.

Scoring the whole test split (`--hh-test 100000`, n = 8,505, no sampling) gives the same picture:
BT full 51.0% [49.9, 52.0], length-only 51.9% [50.9, 53.0].

### Learned weights (full model, standardized feature differences)

| Feature | Weight |
|---|---|
| `jaccard` | −0.485 |
| `n_lines` | +0.329 |
| `overlap_frac` | +0.309 |
| `tfidf_cos` | +0.287 |
| `n_bullets` | +0.163 |
| `log_len` | −0.128 |

Length-only model: `log_len` = +0.174.

The features are strongly correlated (`log_len`, `n_lines`, `n_bullets` and the overlap
measures all scale with response length). Read the weights jointly, not as independent
effects. For example, the negative `log_len` weight does not mean shorter responses are
preferred: the length signal has moved into `n_lines` and the overlap features.

### Most confident errors (qualitative)

These are the five test pairs the full model gets most confidently wrong (most negative margin).

| ID | Margin | Pattern |
|---|---|---|
| uf-train-26524 | −2.21 | Rejected is a 144-line screenplay-format retelling; chosen is a one-paragraph noir story |
| uf-train-24849 | −1.91 | Chosen echoes the user's AppleScript (high prompt overlap); rejected is a verbose "I must point out…" reply |
| uf-train-32423 | −1.60 | Chosen is a one-line prompt rewrite; rejected is a 31-line, 21-bullet answer |
| uf-train-61820 | −1.59 | Chosen is the requested one-line headline; rejected adds a 480-word article |
| uf-train-49574 | −1.52 | Both long; rejected has 35 bullets vs 5 and opens with "As a respectful and helpful AI assistant…" |

The same failure shows up in every case: the model rewards **length and formatting**
(lines, bullets) without any notion of correctness or relevance. Concise correct answers
and prose answers lose to verbose or list-formatted ones. This is the gap a model-based
signal (DPO margin, frozen representations) is meant to close.

## Takeaways

1. Surface features reach **59.2%** on UltraFeedback, a meaningful bar the DPO
   baseline needs to clear.
2. About **6.8 of the 9.2 points above chance come from length alone**. The remaining
   2.4 points are statistically significant but small.
3. Surface-feature preferences are **corpus-specific**: on HH-RLHF both variants are at
   chance, and the extra features do not beat length alone.
   This supports using cross-corpus transfer as the harder test in Phase 3.
4. The "chosen shorter" bucket (36.6%) and the similar-length bucket (56.8%) are the
   right places to show whether DPO or the representation-based reranker go beyond
   length.

## Limitations and pending work

- **One seed so far.** Seeds 1 and 2 (different prompt-hash splits) are still to run. The CIs
  above cover test-set sampling only, not variation between splits.
- **HH-RLHF transfer** uses a 2,000-pair seeded sample of the test split by default; the full split (8,505 pairs) gives the same result.
- **Multi-turn prompts:** UltraFeedback prompts use only the first user turn. HH-RLHF uses the
  full dialogue before the final assistant turn as the prompt.
- **Test IDs** keep the original HF split name (`uf-train-…`/`uf-test-…`). The HF splits
  were pooled and re-split by prompt hash, so the ID prefix does not indicate our split.

## Reproducing

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe bt_baseline.py --seed 0     # this report
.venv\Scripts\python.exe bt_baseline.py --seed 1
.venv\Scripts\python.exe bt_baseline.py --seed 2
.venv\Scripts\python.exe bt_baseline.py --predict    # score a custom pair
```

Each run writes four outputs:
- `results/bt_seed{N}.json`: all metrics in this report
- `results/bt_seed{N}.joblib`: the saved model
- `results/splits/uf_test_seed{N}.jsonl`: the frozen test set, reused to score the DPO baseline
  on the same pairs
