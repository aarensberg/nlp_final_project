"""Metrics shared by the baseline, the fine-tuned model and the audit.

Macro-F1 definition (used everywhere: point scores, bootstrap, paired bootstrap,
weight selection). The label universe is FIXED to the 100 LEDGAR labels:

    F1_c = 2 TP_c / (n_true_c + n_pred_c)        macro-F1 = mean over the 100 labels
    F1_c = 1 if label c has no true and no predicted example (no error is possible)

A fixed universe keeps every resample on the same denominator. scikit-learn's
default instead averages over the labels present in y_true or y_pred, so a class
that vanishes from a bootstrap resample silently leaves the average.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

N_LABELS = 100  # LEDGAR label universe
MACRO_F1_DEFINITION = ("mean over the fixed universe of 100 labels; F1_c = 2TP/(n_true+n_pred); "
                       "F1_c = 1 when label c has neither true nor predicted examples")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int = N_LABELS,
             absent_value: float = 1.0) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_labels)[:n_labels]
    denom = (np.bincount(y_true, minlength=n_labels)[:n_labels]
             + np.bincount(y_pred, minlength=n_labels)[:n_labels])
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), absent_value)
    return float(f1.mean())


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int = N_LABELS) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")) if len(y_true) else None,
        "macro_f1": macro_f1(y_true, y_pred, n_labels) if len(y_true) else None,
        "weighted_f1": (float(f1_score(y_true, y_pred, labels=range(n_labels), average="weighted",
                                       zero_division=0)) if len(y_true) else None),
        "n_labels_in_universe": n_labels,
        "n_labels_present_in_truth": int(len(np.unique(y_true))),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, n_labels: int = N_LABELS) -> dict:
    """95% percentile bootstrap CI for micro- and macro-F1, resampling CLAUSES.

    Clauses of the same contract are not independent, but LEDGAR has no contract
    identifier, so a contract-level (cluster) bootstrap is impossible: intervals
    are likely somewhat too narrow.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    mi, ma = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mi.append(float((y_true[idx] == y_pred[idx]).mean()))
        ma.append(macro_f1(y_true[idx], y_pred[idx], n_labels))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"micro_f1_95ci": q(mi), "macro_f1_95ci": q(ma)}


def per_class(y_true, y_pred, names: list[str]) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(names)), zero_division=0)
    return [
        {"label": names[i], "precision": float(p[i]), "recall": float(r[i]),
         "f1": float(f[i]), "support": int(s[i])}
        for i in range(len(names))
    ]


def top_confusions(y_true, y_pred, names, k: int = 15) -> list[dict]:
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    best = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [{"true": names[t], "predicted": names[p], "count": c} for (t, p), c in best]


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the top-1 confidence."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(probs, y_true, n_bins: int = 10):
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017), fitted on validation logits only."""
    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def coverage_curve(probs, y_true, steps: int = 21) -> list[dict]:
    """Share correct among answered clauses for a range of target coverages.

    Coverage is the share actually accepted by `conf >= threshold` (ties included),
    so it can differ slightly from the nominal step.
    """
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    rows = []
    for cov in np.linspace(1.0, 0.5, steps):
        thr = np.quantile(conf, 1 - cov)
        m = conf >= thr
        rows.append({"coverage": float(m.mean()), "threshold": float(thr),
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def select_threshold(probs: np.ndarray, y_true: np.ndarray, target: float = 0.95) -> dict:
    """Lowest threshold t such that the set {conf >= t} reaches `target` accuracy.

    Candidates are whole groups of identical confidences (a threshold accepts a
    tie group entirely), and the accuracy is that of the set really accepted by
    `>=`. If no non-empty set reaches the target, `threshold` is None: the
    variant must abstain on everything (a numeric 1.0 would still accept
    clauses whose confidence is exactly 1).
    """
    conf = np.asarray(probs).max(1)
    correct = (np.asarray(probs).argmax(1) == y_true).astype(np.int64)
    if len(conf) == 0:
        return {"target": target, "feasible": False, "threshold": None, "n_accepted": 0,
                "coverage": 0.0, "accuracy_accepted": None}
    order = np.argsort(-conf, kind="stable")
    c, k = conf[order], correct[order]
    ends = np.r_[np.nonzero(c[1:] != c[:-1])[0], len(c) - 1]  # last index of each tie group
    acc = np.cumsum(k)[ends] / (ends + 1)
    ok = np.nonzero(acc >= target)[0]
    if len(ok) == 0:
        return {"target": target, "feasible": False, "threshold": None, "n_accepted": 0,
                "coverage": 0.0, "accuracy_accepted": None}
    thr = float(c[ends[ok.max()]])
    accepted = conf >= thr
    real_acc = float(correct[accepted].mean())
    assert real_acc >= target and accepted.sum() == ends[ok.max()] + 1
    return {"target": target, "feasible": True, "threshold": thr, "n_accepted": int(accepted.sum()),
            "coverage": float(accepted.mean()), "accuracy_accepted": real_acc}


def threshold_for_precision(probs, y_true, target: float = 0.95) -> float | None:
    """Threshold of `select_threshold`; None means 'abstain on every input'."""
    return select_threshold(probs, y_true, target)["threshold"]


def answers(conf: np.ndarray, threshold: float | None) -> np.ndarray:
    """Boolean mask of the inputs a variant answers (None threshold -> answers none)."""
    conf = np.asarray(conf)
    return np.zeros(conf.shape, dtype=bool) if threshold is None else conf >= threshold
