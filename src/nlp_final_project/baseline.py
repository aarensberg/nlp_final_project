"""TF-IDF + logistic regression baseline.

A strong classical baseline is the first thing a CTO will ask for: if a linear
model on n-grams is within a point of BERT, the transformer has to justify its
cost. Run: uv run nlp-final-project baseline
"""

from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import ARTIFACTS, REPORTS, SEED, save_json
from .data import split_arrays
from .metrics import core_metrics


def run() -> dict:
    Xtr, ytr = split_arrays("train")
    Xva, yva = split_arrays("validation")
    Xte, yte = split_arrays("test")

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                          sublinear_tf=True, lowercase=True)
    Atr = vec.fit_transform(Xtr)
    clf = LogisticRegression(C=20.0, max_iter=2000, random_state=SEED)
    clf.fit(Atr, ytr)
    train_s = time.time() - t0

    joblib.dump({"vectorizer": vec, "classifier": clf}, ARTIFACTS / "baseline.joblib")
    out = {"model": "tfidf(1-2gram)+logreg", "train_seconds": train_s}
    for split, X, y in (("validation", Xva, yva), ("test", Xte, yte)):
        logits = clf.decision_function(vec.transform(X))
        np.savez_compressed(ARTIFACTS / f"baseline_{split}_logits.npz", logits=logits, labels=y)
        out[split] = core_metrics(y, logits.argmax(1))
    save_json(out, REPORTS / "baseline_metrics.json")
    return out


if __name__ == "__main__":
    print(run())
