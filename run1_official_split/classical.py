"""Classical baseline: Bradley-Terry logistic regression on hand-crafted feature differences.

  python classical.py   -> outputs/preds_classical.csv, outputs/preds_length.csv, outputs/classical.joblib
"""
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_pairs, prompt_text, to_ab, write_preds

FEATURES = ["n_words", "n_lines", "has_code", "prompt_coverage", "tfidf_cos"]
WORD = re.compile(r"\w+")


class Model:
    def __init__(self, features="all"):
        self.cols = list(range(len(FEATURES))) if features == "all" else [0]
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
        # Mirrored data has zero mean, so the scaler only rescales; no intercept => P(A>B) = 1 - P(B>A).
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(fit_intercept=False, max_iter=1000))

    def feats(self, prompts, resps):
        cos = np.asarray(self.vec.transform(prompts).multiply(self.vec.transform(resps)).sum(1)).ravel()
        rows = []
        for p, r in zip(prompts, resps):
            pw, rw = set(WORD.findall(p.lower())), set(WORD.findall(r.lower()))
            rows.append([len(r.split()), r.count("\n") + 1, int("```" in r), len(pw & rw) / max(1, len(pw))])
        return np.c_[np.array(rows, dtype=float), cos][:, self.cols]

    def fit(self, prompts, chosen, rejected):
        self.vec.fit(prompts + chosen + rejected)
        d = self.feats(prompts, chosen) - self.feats(prompts, rejected)
        self.clf.fit(np.r_[d, -d], np.r_[np.ones(len(d)), np.zeros(len(d))])
        return self

    def margins(self, prompts, a, b):
        return self.clf.decision_function(self.feats(prompts, a) - self.feats(prompts, b))

    def margin(self, prompt_msgs, a, b):
        return float(self.margins([prompt_text(prompt_msgs)], [a], [b])[0])

    def weights(self):
        return dict(zip([FEATURES[i] for i in self.cols], self.clf[-1].coef_[0].round(4)))


def fit(train_ds, features="all"):
    prompts = [prompt_text(r["chosen"][:-1]) for r in train_ds]
    return Model(features).fit(prompts, [r["chosen"][-1]["content"] for r in train_ds],
                               [r["rejected"][-1]["content"] for r in train_ds])


def load(path="outputs/classical.joblib"):
    return joblib.load(path)


def main():
    train = load_pairs("train", drop_ties=True)
    pairs = to_ab(load_pairs("test"))
    prompts = [prompt_text(p["prompt"]) for p in pairs]
    for name, features in [("classical", "all"), ("length", "length")]:
        m = fit(train, features)
        margins = m.margins(prompts, [p["a"] for p in pairs], [p["b"] for p in pairs])
        write_preds(pairs, margins, f"outputs/preds_{name}.csv")
        ok = [(g > 0) == bool(p["label"]) for p, g in zip(pairs, margins) if not p["is_tie"]]
        print(f"{name:9s} test acc (non-tie, n={len(ok)}): {sum(ok) / len(ok):.4f}  weights {m.weights()}")
        if name == "classical":
            joblib.dump(m, "outputs/classical.joblib")


if __name__ == "__main__":
    import classical  # pickle the model as classical.Model, not __main__.Model, so other scripts can load it
    classical.main()
