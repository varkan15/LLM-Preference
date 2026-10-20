"""Error analysis over any preds_*.csv (pair_id, margin, label, len_chosen, len_rejected, is_tie).

  python analyze.py outputs/preds_length.csv outputs/preds_classical.csv outputs/preds_dpo.csv [--failures 3]
First file is the reference for paired comparisons.
"""
import argparse
import os

import numpy as np
import pandas as pd


def prepare(df):
    df = df.copy()
    df["pred"] = (df.margin > 0).astype(int)  # zero margin -> B
    df["correct"] = df.pred == df.label
    df["chosen_longer"] = df.len_chosen > df.len_rejected
    df["abs_len_diff"] = (df.len_chosen - df.len_rejected).abs()
    return df


def accuracy(df):
    return float(df.correct.mean())


def boot(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def report(name, df):
    nt = df[df.is_tie == 0]
    lo, hi = boot(nt.correct)
    print(f"\n=== {name}")
    print(f"accuracy (non-tie, n={len(nt)}): {accuracy(nt):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    ties = df[df.is_tie == 1]
    if len(ties):
        print(f"ties slice (n={len(ties)}, label arbitrary): {accuracy(ties):.4f}")
    print("confusion (rows = true, cols = predicted):")
    print(pd.crosstab(nt.label.map({1: "A", 0: "B"}), nt.pred.map({1: "A", 0: "B"})).to_string())
    print("accuracy by length:")
    print(nt.groupby(nt.chosen_longer.map({True: "chosen longer", False: "chosen shorter/equal"}))
          .correct.agg(["mean", "size"]).round(4).to_string())
    q = pd.qcut(nt.abs_len_diff, 4, duplicates="drop")
    print(nt.groupby(q, observed=True).correct.agg(["mean", "size"]).round(4)
          .rename_axis("|len diff| words").to_string())


def paired(ref_name, ref, name, df):
    m = ref[ref.is_tie == 0].merge(df, on="pair_id", suffixes=("_r", ""))
    d = m.correct.astype(float) - m.correct_r.astype(float)
    lo, hi = boot(d)
    print(f"{name} - {ref_name}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


def failures(name, df, k):
    from data import load_pairs, prompt_text, to_ab
    pairs = {p["pair_id"]: p for p in to_ab(load_pairs("test"))}
    wrong = df[(df.is_tie == 0) & ~df.correct].reindex(
        df.margin.abs().sort_values(ascending=False).index).dropna().head(k)
    print(f"\n--- {name}: {k} most confident errors")
    for _, r in wrong.iterrows():
        p = pairs[int(r.pair_id)]
        cut = lambda s: s[:300].replace("\n", " ") + ("…" if len(s) > 300 else "")
        print(f"[pair {int(r.pair_id)}] margin {r.margin:+.3f}, chose {'A' if r.pred else 'B'}, "
              f"human chose {'A' if r.label else 'B'}")
        print(f"  PROMPT: {cut(prompt_text(p['prompt']))}\n  A ({len(p['a'].split())}w): {cut(p['a'])}"
              f"\n  B ({len(p['b'].split())}w): {cut(p['b'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--failures", type=int, default=3)
    a = ap.parse_args()
    runs = {os.path.basename(f).removeprefix("preds_").removesuffix(".csv"): prepare(pd.read_csv(f))
            for f in a.files}
    for name, df in runs.items():
        report(name, df)
    if len(runs) > 1:
        print("\n=== paired accuracy difference vs first file")
        (ref_name, ref), *rest = runs.items()
        for name, df in rest:
            paired(ref_name, ref, name, df)
    if a.failures:
        for name, df in runs.items():
            failures(name, df, a.failures)
