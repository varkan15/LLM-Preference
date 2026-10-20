"""UltraFeedback loading, tie handling, A/B flattening, shared preds CSV."""
import csv
import json
import os
import random
import statistics

from datasets import load_dataset

DATASET = "trl-lib/ultrafeedback_binarized"


def is_tie(row):
    return row["score_chosen"] == row["score_rejected"]


def load_pairs(split, n=None, drop_ties=False, seed=0):
    ds = load_dataset(DATASET, split=split)
    if drop_ties:
        ds = ds.filter(lambda r: not is_tie(r))
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


def prompt_text(msgs):
    return "\n".join(m["content"] for m in msgs if m["role"] != "assistant")


def to_ab(ds, seed=0):
    """One dict per pair with A/B order randomized; label=1 iff A is the chosen response."""
    rng = random.Random(seed)
    out = []
    for i, r in enumerate(ds):
        c, rj = r["chosen"][-1]["content"], r["rejected"][-1]["content"]
        a_chosen = rng.random() < 0.5
        out.append(dict(
            pair_id=i, prompt=r["chosen"][:-1],
            a=c if a_chosen else rj, b=rj if a_chosen else c,
            label=int(a_chosen), len_chosen=len(c.split()), len_rejected=len(rj.split()),
            is_tie=int(is_tie(r)),
        ))
    return out


PRED_FIELDS = ["pair_id", "margin", "label", "len_chosen", "len_rejected", "is_tie"]


def write_preds(pairs, margins, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        for p, m in zip(pairs, margins):
            w.writerow({k: p[k] for k in PRED_FIELDS if k != "margin"} | {"margin": float(m)})


def stats(ds):
    lc = [len(r["chosen"][-1]["content"].split()) for r in ds]
    lr = [len(r["rejected"][-1]["content"].split()) for r in ds]
    gaps = [r["score_chosen"] - r["score_rejected"] for r in ds]
    return dict(
        n=len(ds), ties=sum(g == 0 for g in gaps),
        median_words_chosen=statistics.median(lc), median_words_rejected=statistics.median(lr),
        frac_chosen_longer=round(sum(a > b for a, b in zip(lc, lr)) / len(ds), 4),
        frac_gap_le_0_5=round(sum(g <= 0.5 for g in gaps) / len(ds), 4),
        multi_turn=sum(len(r["chosen"]) > 2 for r in ds),
    )


if __name__ == "__main__":
    s = {split: stats(load_pairs(split)) for split in ["train", "test"]}
    print(json.dumps(s, indent=2))
    os.makedirs("outputs", exist_ok=True)
    json.dump(s, open("outputs/data_stats.json", "w"), indent=2)
