# CS F429 NLP — Team 1 — Phase 2: Bradley-Terry Baseline

Pairwise preference prediction on UltraFeedback with a classical Bradley-Terry
(BT) logistic regression over surface features, plus a length-only control.
Full write-up: `report/Phase2_Report_TEAM_1.pdf`.

## Contents

```
team1_phase2_BT/
├── README.md                      this file
├── report/
│   ├── Phase2_Report_TEAM_1.pdf   the Phase 2 report
│   ├── Phase2_Report_TEAM_1.md    report source
│   └── REVIEW_AND_FIXES.md        code review + sampling fixes (report Section 6)
├── run1_official_split/           Run 1: official HF test split (895 pairs)
│   ├── data.py                    load UltraFeedback, ties, A/B randomization
│   ├── classical.py               train BT full + length-only, write predictions
│   ├── analyze.py                 CIs, confusion matrix, length buckets, worst errors
│   └── requirements.txt
└── run2_bt_baselines/             Run 2: prompt-hash re-split (4,000 pairs) + HH-RLHF transfer
    ├── bt_baseline.py             train, evaluate, save; --predict for a custom pair
    ├── features.py                surface features (Qwen token length, lines, bullets, overlap, TF-IDF)
    ├── data.py                    load UltraFeedback + HH-RLHF, prompt-disjoint split
    ├── requirements.txt
    └── BT_Results.md              detailed Run 2 results
```

No data files are included. Each run downloads its data from the Hugging Face Hub
on first use (no account or token needed) and caches it in `~/.cache/huggingface/`.
About 770 MB in total; no GPU needed.

## Run 1 — official test split (Python 3.12, ~3 min)

```bash
cd run1_official_split
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python data.py          # dataset statistics -> outputs/data_stats.json
.venv/bin/python classical.py     # train + test; prints accuracy and weights
.venv/bin/python analyze.py outputs/preds_length.csv outputs/preds_classical.csv
```

Expected: `classical test acc (non-tie, n=895): 0.6045`, `length test acc ...: 0.5642`,
`classical - length: +0.0402  95% CI [+0.0179, +0.0615]`.

## Run 2 — re-split + HH-RLHF transfer (Python 3.12, ~2.5 min)

```bash
cd run2_bt_baselines
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python bt_baseline.py --seed 0                    # main run -> results/bt_seed0.json
.venv/bin/python bt_baseline.py --seed 0 --hh-test 100000   # transfer on all 8,505 HH-RLHF pairs
.venv/bin/python bt_baseline.py --predict                   # demo: score a pair you type in
```

Expected: `BT full acc=0.592`, `BT length-only acc=0.568`, `diff=+0.024`; HH-RLHF
`BT full acc=0.519`, `BT length-only acc=0.525`.

## Notes

- **Windows:** use `py -3.12 -m venv .venv`, then
  `.venv\Scripts\pip` and `.venv\Scripts\python` in place of `.venv/bin/...`.
- **One virtual environment per run.** The two runs pin different scikit-learn versions.
- **Cache location:** set `HF_HOME=/path/to/cache` before running to keep downloads elsewhere.
- **Offline machine:** run the two commands below once with internet access, copy the
  cache folder over, and set `HF_HOME` to it plus `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

  ```bash
  python -c "from datasets import load_dataset; load_dataset('trl-lib/ultrafeedback_binarized'); load_dataset('Anthropic/hh-rlhf', split='test')"
  python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
  ```
