"""Data loading and prompt-disjoint splitting shared by all Phase 2 baselines.

Every example is normalized to a dict: {"id", "prompt", "chosen", "rejected"}.
"""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def _last_assistant(messages):
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def _first_user(messages):
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def _clean(rows):
    """Drop empty / identical pairs and exact duplicates."""
    seen, out = set(), []
    for x in rows:
        c, r = x["chosen"].strip(), x["rejected"].strip()
        if not c or not r or c == r or not x["prompt"].strip():
            continue
        key = (x["prompt"], c, r)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def load_ultrafeedback():
    """All of trl-lib/ultrafeedback_binarized (train + test pooled).

    The HF train/test splits are pooled and re-split by prompt hash so that
    no prompt appears in more than one of our splits. Tied-score pairs are
    dropped because they carry no preference.
    """
    ds = load_dataset("trl-lib/ultrafeedback_binarized")
    rows = []
    for split in ("train", "test"):
        for i, ex in enumerate(ds[split]):
            if ex["score_chosen"] == ex["score_rejected"]:
                continue
            rows.append({
                "id": f"uf-{split}-{i}",
                "prompt": _first_user(ex["chosen"]),
                "chosen": _last_assistant(ex["chosen"]),
                "rejected": _last_assistant(ex["rejected"]),
            })
    return _clean(rows)


def load_hh(split="test", limit=None, seed=0):
    """Anthropic/hh-rlhf, used only for cross-corpus transfer evaluation.

    Each side is a full transcript "\\n\\nHuman: ... \\n\\nAssistant: ...".
    The prompt is everything up to the final assistant turn, and the response
    is that final turn. The split is several subsets concatenated in order
    (harmless-base first), so rows are shuffled with a fixed seed before `limit`.
    """
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    tag = "\n\nAssistant:"
    rows = []
    for i, ex in enumerate(ds):
        c, r = ex["chosen"], ex["rejected"]
        ci, ri = c.rfind(tag), r.rfind(tag)
        if ci < 0 or ri < 0:
            continue
        rows.append({
            "id": f"hh-{split}-{i}",
            "prompt": c[:ci].strip(),
            "chosen": c[ci + len(tag):].strip(),
            "rejected": r[ri + len(tag):].strip(),
        })
    rows = _clean(rows)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def prompt_bucket(prompt, salt=0):
    """Deterministic bucket in [0, 100) from the prompt text."""
    h = hashlib.md5(f"{salt}::{prompt}".encode("utf-8")).hexdigest()
    return int(h, 16) % 100


def split_rows(rows, seed=0, val_pct=10, test_pct=10):
    """Prompt-disjoint train / val / test split (by prompt hash)."""
    train, val, test = [], [], []
    for x in rows:
        b = prompt_bucket(x["prompt"], salt=seed)
        if b < test_pct:
            test.append(x)
        elif b < test_pct + val_pct:
            val.append(x)
        else:
            train.append(x)
    return train, val, test


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
