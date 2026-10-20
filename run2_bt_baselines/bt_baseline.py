"""Classical baseline: Bradley-Terry logistic regression over surface features.

    P(A > B | prompt) = sigmoid( w . (f(prompt, A) - f(prompt, B)) )

Usage:
    python bt_baseline.py                      # full run, seed 0
    python bt_baseline.py --seed 1 --max-train 5000 --hh-test 1000
    python bt_baseline.py --predict            # interactive pairwise prediction
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import load_hh, load_ultrafeedback, save_jsonl, split_rows
from features import Featurizer

LENGTH_COL = Featurizer.names.index("log_len")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def make_pairs(Fc, Fr):
    """Symmetric augmentation: (c - r, 1) and (r - c, 0)."""
    D = Fc - Fr
    X = np.vstack([D, -D])
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    return X, y


class BradleyTerry:
    """Bradley-Terry model with a linear utility over feature columns `cols`.

    No intercept and no mean-centering, so margin(A, B) == -margin(B, A).
    """

    def __init__(self, C=1.0, cols=None):
        self.C = C
        self.cols = cols

    def _sel(self, F):
        return F if self.cols is None else F[:, self.cols]

    def fit(self, Fc, Fr):
        X, y = make_pairs(self._sel(Fc), self._sel(Fr))
        self.scaler = StandardScaler(with_mean=False).fit(X)
        self.lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
        self.lr.fit(self.scaler.transform(X), y)
        return self

    def margin(self, Fa, Fb):
        """Log-odds that A is preferred over B (> 0 means A wins)."""
        D = self._sel(Fa) - self._sel(Fb)
        return self.lr.decision_function(self.scaler.transform(D))

    def prob(self, Fa, Fb):
        return 1.0 / (1.0 + np.exp(-self.margin(Fa, Fb)))

    def contributions(self, Fa, Fb):
        """Per-feature contribution to the margin (in standardized units)."""
        D = self.scaler.transform(self._sel(Fa) - self._sel(Fb))
        return D * self.lr.coef_[0]

    def weights(self, names):
        names = names if self.cols is None else [names[c] for c in self.cols]
        return dict(zip(names, self.lr.coef_[0].round(4).tolist()))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def bootstrap_ci(correct, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    accs = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(correct_a, correct_b, n_boot=1000, seed=0):
    """CI on acc(A) - acc(B) over the same test pairs."""
    diff = correct_a.astype(float) - correct_b.astype(float)
    lo, hi = bootstrap_ci(diff, n_boot, seed)
    return {"diff": float(diff.mean()), "ci95": [lo, hi]}


def length_buckets(Fc, Fr, correct, thresh=0.2):
    """Accuracy split by log-length difference (chosen - rejected)."""
    d = Fc[:, LENGTH_COL] - Fr[:, LENGTH_COL]
    masks = {
        "chosen_shorter": d < -thresh,
        "similar_length": np.abs(d) <= thresh,
        "chosen_longer": d > thresh,
    }
    return {
        k: {"acc": float(correct[m].mean()) if m.any() else None, "n": int(m.sum())}
        for k, m in masks.items()
    }


def evaluate(model, Fc, Fr, seed=0):
    m = model.margin(Fc, Fr)
    correct = (m > 0).astype(float)
    lo, hi = bootstrap_ci(correct, seed=seed)
    return correct, {
        "acc": float(correct.mean()),
        "ci95": [lo, hi],
        "n": int(len(correct)),
        "by_length": length_buckets(Fc, Fr, correct),
    }


def worst_errors(model, rows, Fc, Fr, k=5):
    """The k test pairs the model gets most confidently wrong."""
    m = model.margin(Fc, Fr)
    idx = np.argsort(m)[:k]
    return [{
        "id": rows[i]["id"],
        "margin": round(float(m[i]), 3),
        "prompt": rows[i]["prompt"][:200],
        "chosen": rows[i]["chosen"][:200],
        "rejected": rows[i]["rejected"][:200],
    } for i in idx if m[i] < 0]


def fmt(name, r):
    lo, hi = r["ci95"]
    return f"  {name:22s} acc={r['acc']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={r['n']}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(args):
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UltraFeedback ...")
    rows = load_ultrafeedback()
    train, val, test = split_rows(rows, seed=args.seed)

    def subsample(xs, n, k):
        rng = np.random.default_rng((args.seed, k))
        if n and len(xs) > n:
            xs = [xs[i] for i in sorted(rng.choice(len(xs), n, replace=False))]
        return xs

    train = subsample(train, args.max_train, 0)
    val = subsample(val, args.max_val, 1)
    test = subsample(test, args.max_test, 2)
    print(f"  pairs: train={len(train)} val={len(val)} test={len(test)} "
          f"(prompt-disjoint, seed={args.seed})")
    # Frozen test set, reused by the DPO baseline for a like-for-like comparison.
    save_jsonl(test, out_dir / "splits" / f"uf_test_seed{args.seed}.jsonl")

    print("Fitting featurizer and extracting features ...")
    fz = Featurizer().fit(train)
    Fc_tr, Fr_tr = fz.pair_features(train)
    Fc_va, Fr_va = fz.pair_features(val)
    Fc_te, Fr_te = fz.pair_features(test)

    print("Tuning C on validation ...")
    val_acc = {}
    for C in args.C_grid:
        m = BradleyTerry(C).fit(Fc_tr, Fr_tr)
        val_acc[C] = float((m.margin(Fc_va, Fr_va) > 0).mean())
        print(f"  C={C:<8g} val acc={val_acc[C]:.4f}")
    best_C = max(val_acc, key=val_acc.get)

    bt = BradleyTerry(best_C).fit(Fc_tr, Fr_tr)
    bt_len = BradleyTerry(best_C, cols=[LENGTH_COL]).fit(Fc_tr, Fr_tr)

    corr_full, res_full = evaluate(bt, Fc_te, Fr_te, args.seed)
    corr_len, res_len = evaluate(bt_len, Fc_te, Fr_te, args.seed)

    results = {
        "method": "bradley_terry_surface",
        "seed": args.seed,
        "best_C": best_C,
        "val_acc_by_C": {str(k): v for k, v in val_acc.items()},
        "features": Featurizer.names,
        "weights": bt.weights(Featurizer.names),
        "length_only_weights": bt_len.weights(Featurizer.names),
        "ultrafeedback_test": {
            "bt_full": res_full,
            "bt_length_only": res_len,
            "full_minus_length": paired_bootstrap(corr_full, corr_len, seed=args.seed),
        },
        "worst_errors": worst_errors(bt, test, Fc_te, Fr_te),
    }

    print("\nUltraFeedback held-out test (prompt-disjoint):")
    print(fmt("BT full", res_full))
    print(fmt("BT length-only", res_len))
    d = results["ultrafeedback_test"]["full_minus_length"]
    print(f"  full - length-only     diff={d['diff']:+.3f}  "
          f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")

    print("\nAccuracy by length difference (BT full | length-only):")
    for k in res_full["by_length"]:
        a, b = res_full["by_length"][k], res_len["by_length"][k]
        if a["n"]:
            print(f"  {k:16s} {a['acc']:.3f} | {b['acc']:.3f}   n={a['n']}")

    print("\nLearned weights (standardized feature differences):")
    for k, w in results["weights"].items():
        print(f"  {k:14s} {w:+.4f}")

    if args.hh_test:
        print(f"\nCross-corpus transfer: HH-RLHF test ({args.hh_test} seeded-sample pairs) ...")
        hh = load_hh("test", limit=args.hh_test)
        Fc_hh, Fr_hh = fz.pair_features(hh)
        _, hh_full = evaluate(bt, Fc_hh, Fr_hh, args.seed)
        _, hh_len = evaluate(bt_len, Fc_hh, Fr_hh, args.seed)
        results["hh_rlhf_transfer"] = {"bt_full": hh_full, "bt_length_only": hh_len}
        print(fmt("BT full", hh_full))
        print(fmt("BT length-only", hh_len))

    results["runtime_sec"] = round(time.time() - t0, 1)
    res_path = out_dir / f"bt_seed{args.seed}.json"
    res_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    model_path = out_dir / f"bt_seed{args.seed}.joblib"
    joblib.dump({"featurizer": fz, "bt": bt, "bt_len": bt_len}, model_path)
    print(f"\nSaved {res_path} and {model_path}  ({results['runtime_sec']}s)")


def predict_interactive(args):
    """Score a hand-entered pair with a saved model."""
    bundle = joblib.load(Path(args.out) / f"bt_seed{args.seed}.joblib")
    fz, bt, bt_len = bundle["featurizer"], bundle["bt"], bundle["bt_len"]
    prompt = input("Prompt: ")
    a = input("Response A: ")
    b = input("Response B: ")
    Fa = fz.transform([prompt], [a])
    Fb = fz.transform([prompt], [b])
    p = float(bt.prob(Fa, Fb)[0])
    p_len = float(bt_len.prob(Fa, Fb)[0])
    print(f"\nBT full:        P(A > B) = {p:.3f}  -> {'A' if p > 0.5 else 'B'}")
    print(f"BT length-only: P(A > B) = {p_len:.3f}  -> {'A' if p_len > 0.5 else 'B'}")
    print("Per-feature contribution to the margin (positive favors A):")
    for name, c in zip(Featurizer.names, bt.contributions(Fa, Fb)[0]):
        print(f"  {name:14s} {c:+.3f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="split salt and RNG seed")
    p.add_argument("--max-train", type=int, default=0, help="0 = use all")
    p.add_argument("--max-val", type=int, default=4000)
    p.add_argument("--max-test", type=int, default=4000)
    p.add_argument("--hh-test", type=int, default=2000,
                   help="HH-RLHF transfer pairs to evaluate (0 = skip)")
    p.add_argument("--C-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    p.add_argument("--out", default="results")
    p.add_argument("--predict", action="store_true",
                   help="score a pair interactively with a saved model")
    return p.parse_args()


if __name__ == "__main__":
    # Go through the importable module so pickled classes resolve to
    # bt_baseline.BradleyTerry (not __main__) when loaded from other scripts.
    import bt_baseline
    args = bt_baseline.parse_args()
    bt_baseline.predict_interactive(args) if args.predict else bt_baseline.run(args)
