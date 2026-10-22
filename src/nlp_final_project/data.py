"""Loading LEDGAR (LexGLUE) and a few helpers shared by every stage."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from datasets import DatasetDict, load_dataset

from .config import DATASET, DATASET_REVISION


@lru_cache(maxsize=1)
def load_ledgar() -> DatasetDict:
    """LEDGAR from LexGLUE: 60k train / 10k validation / 10k test, 100 clause types.

    The official split is chronological (older filings in train, newer in test),
    so the test score already includes some temporal drift.
    """
    return load_dataset(*DATASET, revision=DATASET_REVISION)


def label_names() -> list[str]:
    return load_ledgar()["train"].features["label"].names


def split_arrays(split: str) -> tuple[list[str], np.ndarray]:
    ds = load_ledgar()[split]
    return list(ds["text"]), np.asarray(ds["label"])


def frequency_buckets(train_labels: np.ndarray, n_labels: int) -> dict[int, str]:
    """Tag each class as head / torso / tail by its training frequency (terciles)."""
    counts = np.bincount(train_labels, minlength=n_labels)
    lo, hi = np.quantile(counts, [1 / 3, 2 / 3])
    return {
        c: ("tail" if counts[c] <= lo else "head" if counts[c] > hi else "torso")
        for c in range(n_labels)
    }
