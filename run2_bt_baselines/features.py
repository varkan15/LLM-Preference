"""Hand-crafted surface features for the classical Bradley-Terry baseline."""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_WORD = re.compile(r"\w+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", flags=re.M)


def _words(text):
    return set(_WORD.findall(text.lower()))


class Featurizer:
    """Maps (prompt, response) -> fixed-length surface feature vector."""

    names = ["log_len", "n_lines", "n_bullets", "jaccard", "overlap_frac", "tfidf_cos"]

    def __init__(self, tokenizer_name=TOKENIZER_NAME, batch_size=512):
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.tok = None
        self.tfidf = None

    # The tokenizer is reloaded lazily instead of being pickled with the model.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["tok"] = None
        return state

    def _tokenizer(self):
        if self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self.tok

    def fit(self, train_rows):
        """Fit TF-IDF on training text only (no test leakage)."""
        texts = [x["prompt"] for x in train_rows]
        texts += [x["chosen"] for x in train_rows]
        texts += [x["rejected"] for x in train_rows]
        self.tfidf = TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=2)
        self.tfidf.fit(texts)
        return self

    def _token_lengths(self, texts):
        tok = self._tokenizer()
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = tok(texts[i:i + self.batch_size], add_special_tokens=False)
            out.extend(len(ids) for ids in enc["input_ids"])
        return np.array(out, dtype=float)

    def transform(self, prompts, responses):
        if self.tfidf is None:
            raise RuntimeError("Featurizer.fit must be called before transform")
        # TF-IDF rows are L2-normalized, so the row-wise dot product is the cosine.
        P = self.tfidf.transform(prompts)
        R = self.tfidf.transform(responses)
        cos = np.asarray(P.multiply(R).sum(axis=1)).ravel()

        n_tok = self._token_lengths(responses)
        feats = np.zeros((len(responses), len(self.names)))
        for i, (p, r) in enumerate(zip(prompts, responses)):
            pw, rw = _words(p), _words(r)
            inter = len(pw & rw)
            feats[i] = [
                np.log1p(n_tok[i]),
                r.count("\n") + 1,
                len(_BULLET.findall(r)),
                inter / max(1, len(pw | rw)),
                inter / max(1, len(pw)),
                cos[i],
            ]
        return feats

    def pair_features(self, rows):
        """Returns (F_chosen, F_rejected), each of shape (n, n_features)."""
        prompts = [x["prompt"] for x in rows]
        Fc = self.transform(prompts, [x["chosen"] for x in rows])
        Fr = self.transform(prompts, [x["rejected"] for x in rows])
        return Fc, Fr
