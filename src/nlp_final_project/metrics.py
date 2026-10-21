"""Metrics shared by the baseline, the fine-tuned model and the audit."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI for micro- and macro-F1 (resampling clauses)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mi, ma = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mi.append(f1_score(y_true[idx], y_pred[idx], average="micro"))
        ma.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
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
    """Accuracy when the model only answers above a confidence threshold."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    rows = []
    for cov in np.linspace(1.0, 0.5, steps):
        thr = np.quantile(conf, 1 - cov)
        m = conf >= thr
        rows.append({"coverage": float(m.mean()), "threshold": float(thr),
                     "accuracy": float(correct[m].mean())})
    return rows


def threshold_for_precision(probs, y_true, target: float = 0.95) -> float:
    """Lowest confidence threshold whose answered set reaches `target` accuracy."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(cum_acc >= target)[0]
    if len(ok) == 0:
        return 1.0
    return float(conf[order][ok.max()])
