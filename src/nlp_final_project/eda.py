"""First look at LEDGAR: imbalance, lengths, leakage, drift and slice coverage.

Run: uv run nlp-final-project eda
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon
from transformers import AutoTokenizer

from . import plotting
from .config import CLASSIFIER_MODEL, CLASSIFIER_REVISION, FIGURES, REPORTS, save_json
from .data import label_names, load_ledgar
from .slices import compute_slices


def _norm_hash(t: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest()


def run() -> dict:
    ds = load_ledgar()
    names = label_names()
    L = len(names)
    out: dict = {"n_labels": L, "splits": {s: len(ds[s]) for s in ds}}

    # --- class imbalance ---------------------------------------------------
    counts = {s: np.bincount(ds[s]["label"], minlength=L) for s in ds}
    tr = counts["train"]
    order = np.argsort(-tr)
    out["imbalance"] = {
        "most_frequent": [(names[i], int(tr[i])) for i in order[:10]],
        "least_frequent": [(names[i], int(tr[i])) for i in order[-10:]],
        "max_min_ratio": float(tr.max() / tr.min()),
        "top10_share": float(tr[order[:10]].sum() / tr.sum()),
        "classes_below_100_train": int((tr < 100).sum()),
        "test_classes_below_20": int((counts["test"] < 20).sum()),
    }
    # Chronological split -> does the label mix drift?
    p = {s: c / c.sum() for s, c in counts.items()}
    out["label_drift_js"] = {
        "train_vs_validation": float(jensenshannon(p["train"], p["validation"])),
        "train_vs_test": float(jensenshannon(p["train"], p["test"])),
    }
    shift = p["test"] - p["train"]
    out["largest_share_shifts_test_minus_train"] = [
        (names[i], round(float(shift[i]) * 100, 2)) for i in np.argsort(-np.abs(shift))[:8]
    ]

    # --- lengths -------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL, revision=CLASSIFIER_REVISION)
    sample = list(ds["train"].shuffle(seed=0).select(range(5000))["text"])
    n_tok = np.array([len(x) for x in tok(sample, add_special_tokens=True)["input_ids"]])
    n_words = np.array([len(t.split()) for t in ds["train"]["text"]])
    out["lengths"] = {
        "words_median": float(np.median(n_words)),
        "words_p95": float(np.percentile(n_words, 95)),
        "tokens_median": float(np.median(n_tok)),
        "tokens_p90": float(np.percentile(n_tok, 90)),
        "tokens_p95": float(np.percentile(n_tok, 95)),
        "share_over_256_tokens": float((n_tok > 256).mean()),
        "share_over_512_tokens": float((n_tok > 512).mean()),
    }

    # --- duplicates & leakage ---------------------------------------------
    h = {s: [_norm_hash(t) for t in ds[s]["text"]] for s in ds}
    train_set = set(h["train"])
    dup_train = len(h["train"]) - len(train_set)
    lab_by_hash: dict[str, set] = {}
    for hh, y in zip(h["train"], ds["train"]["label"]):
        lab_by_hash.setdefault(hh, set()).add(y)
    out["duplicates"] = {
        "train_internal_duplicates": dup_train,
        "train_duplicate_texts_with_conflicting_labels": sum(len(v) > 1 for v in lab_by_hash.values()),
        "test_texts_also_in_train": int(sum(x in train_set for x in h["test"])),
        "validation_texts_also_in_train": int(sum(x in train_set for x in h["validation"])),
    }

    # --- slice coverage (proxies for contract kinds / parties) -------------
    out["slices_train"] = {}
    out["slices_test"] = {}
    for split, key in (("train", "slices_train"), ("test", "slices_test")):
        for name, arr in compute_slices(list(ds[split]["text"])).items():
            out[key][name] = dict(Counter(arr.tolist()).most_common())

    save_json(out, REPORTS / "eda.json")

    # --- figures -------------------------------------------------------------
    fig, ax = plotting.plt.subplots(figsize=(9, 3.4))
    ax.bar(range(L), tr[order], color=plotting.SERIES[0], width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Clause type (sorted by training frequency)")
    ax.set_ylabel("Training clauses (log)")
    ax.set_title(f"LEDGAR is long-tailed: {tr.max()} vs {tr.min()} examples per class")
    for i in (0, L - 1):
        ax.annotate(names[order[i]], (i, tr[order[i]]), textcoords="offset points",
                    xytext=(4 if i == 0 else -4, 4), ha="left" if i == 0 else "right",
                    fontsize=8, color=plotting.INK_2)
    plotting.save(fig, FIGURES / "class_distribution.png")

    fig, ax = plotting.plt.subplots(figsize=(7, 3.2))
    ax.hist(n_tok, bins=60, range=(0, 700), color=plotting.SERIES[0])
    for v, lbl in ((256, "256"), (512, "512")):
        ax.axvline(v, color=plotting.INK_2, lw=1, ls="--")
        ax.text(v + 5, ax.get_ylim()[1] * 0.9, lbl, color=plotting.INK_2, fontsize=8)
    ax.set_xlabel("Legal-BERT tokens per clause (5k-clause sample)")
    ax.set_ylabel("Clauses")
    ax.set_title("Clause length")
    plotting.save(fig, FIGURES / "token_lengths.png")
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
