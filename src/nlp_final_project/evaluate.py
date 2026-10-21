"""Held-out evaluation of the fine-tuned classifier (and comparison with the baseline).

Produces: overall metrics with bootstrap CIs, per-class scores, top confusions,
calibration before/after temperature scaling, and a confidence threshold that
lets the assistant abstain ("ask a lawyer") when it is unsure.

Run: uv run nlp-final-project evaluate
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import plotting
from .config import ARTIFACTS, FIGURES, REPORTS, get_device, load_json, save_json
from .data import frequency_buckets, label_names, split_arrays
from .metrics import (bootstrap_ci, core_metrics, coverage_curve, ece, fit_temperature, per_class,
                      reliability, softmax, threshold_for_precision, top_confusions)
from .train import predict_logits, tokenize

MODEL_DIR = ARTIFACTS / "classifier"


def model_logits(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Logits of the fine-tuned model on a split (cached in artifacts/)."""
    cache = ARTIFACTS / f"model_{split}_logits.npz"
    if cache.exists():
        z = np.load(cache)
        return z["logits"], z["labels"]
    X, y = split_arrays(split)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    max_len = load_json(MODEL_DIR / "training_config.json")["max_length"]
    logits = predict_logits(model, tokenize(tok, X, max_len), tok.pad_token_id, device, desc=split)
    np.savez_compressed(cache, logits=logits, labels=y)
    return logits, y


def _hash(t: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest()


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test: do the two systems differ on the same test clauses?"""
    b01 = int((a_correct & ~b_correct).sum())
    b10 = int((~a_correct & b_correct).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"model_right_baseline_wrong": b01, "baseline_right_model_wrong": b10, "p_value": float(p)}


def run() -> dict:
    names = label_names()
    _, ytr = split_arrays("train")
    Xte, _ = split_arrays("test")
    lv, yv = model_logits("validation")
    lt, yt = model_logits("test")
    pred = lt.argmax(1)

    out: dict = {"model_dir": str(MODEL_DIR)}
    out["test"] = core_metrics(yt, pred) | bootstrap_ci(yt, pred)
    out["validation"] = core_metrics(yv, lv.argmax(1))

    # Leakage check: 282 test clauses appear verbatim in train (see EDA).
    train_hashes = {_hash(t) for t in split_arrays("train")[0]}
    unseen = np.array([_hash(t) not in train_hashes for t in Xte])
    out["test_unseen_only"] = core_metrics(yt[unseen], pred[unseen])

    # Head / torso / tail classes.
    buckets = frequency_buckets(ytr, len(names))
    pc = per_class(yt, pred, names)
    for row, c in zip(pc, range(len(names))):
        row["train_count"] = int((ytr == c).sum())
        row["bucket"] = buckets[c]
    out["by_frequency_bucket"] = {
        b: float(np.mean([r["f1"] for r in pc if r["bucket"] == b])) for b in ("head", "torso", "tail")
    }
    out["worst_classes"] = sorted(pc, key=lambda r: r["f1"])[:12]
    out["top_confusions"] = top_confusions(yt, pred, names)

    # Calibration + selective prediction (threshold chosen on validation only).
    T = fit_temperature(lv, yv)
    pv, pt_raw, pt = softmax(lv, T), softmax(lt), softmax(lt, T)
    thr = threshold_for_precision(pv, yv, 0.95)
    answered = pt.max(1) >= thr
    out["calibration"] = {
        "temperature": T,
        "ece_test_before": ece(pt_raw, yt),
        "ece_test_after": ece(pt, yt),
        "abstain_threshold_for_95pct_val_accuracy": thr,
        "test_coverage_at_threshold": float(answered.mean()),
        "test_accuracy_when_answering": float((pred[answered] == yt[answered]).mean()),
        "test_accuracy_on_abstained": float((pred[~answered] == yt[~answered]).mean()),
    }
    out["coverage_curve_test"] = coverage_curve(pt, yt)

    # Baseline comparison on the same clauses.
    bpath = ARTIFACTS / "baseline_test_logits.npz"
    if bpath.exists():
        bl = np.load(bpath)["logits"]
        bpred = bl.argmax(1)
        out["baseline_test"] = core_metrics(yt, bpred) | bootstrap_ci(yt, bpred)
        out["mcnemar_vs_baseline"] = mcnemar(pred == yt, bpred == yt)
        bpc = per_class(yt, bpred, names)
        for r, b in zip(pc, bpc):
            r["baseline_f1"] = b["f1"]

        # Ensemble: weighted mean of both calibrated probability vectors.
        # The weight is chosen on validation (macro-F1), then applied once to test.
        bv = np.load(ARTIFACTS / "baseline_validation_logits.npz")["logits"]
        Tb = fit_temperature(bv, yv)
        pbv, pbt = softmax(bv, Tb), softmax(bl, Tb)
        grid = np.linspace(0, 1, 21)
        val_f1 = [f1_score(yv, (w * pv + (1 - w) * pbv).argmax(1), average="macro") for w in grid]
        w = float(grid[int(np.argmax(val_f1))])
        pe = w * pt + (1 - w) * pbt
        epred = pe.argmax(1)
        out["ensemble_test"] = core_metrics(yt, epred) | bootstrap_ci(yt, epred) | {
            "weight_bert": w, "temperature_baseline": Tb, "val_macro_f1_by_weight": dict(zip(map(float, grid), map(float, val_f1)))}
        out["ensemble_test_unseen_only"] = core_metrics(yt[unseen], epred[unseen])
        out["mcnemar_ensemble_vs_baseline"] = mcnemar(epred == yt, bpred == yt)
        epc = per_class(yt, epred, names)
        for r, e in zip(pc, epc):
            r["ensemble_f1"] = e["f1"]
        out["ensemble_by_frequency_bucket"] = {
            b: float(np.mean([e["f1"] for e, r in zip(epc, pc) if r["bucket"] == b]))
            for b in ("head", "torso", "tail")}
        out["baseline_by_frequency_bucket"] = {
            b: float(np.mean([x["f1"] for x, r in zip(bpc, pc) if r["bucket"] == b]))
            for b in ("head", "torso", "tail")}
        np.savez_compressed(ARTIFACTS / "ensemble_test_probs.npz", probs=pe, weight=w)
        # Same abstention analysis for the ensemble.
        pev = w * pv + (1 - w) * pbv
        ethr = threshold_for_precision(pev, yv, 0.95)
        ans = pe.max(1) >= ethr
        out["ensemble_calibration"] = {
            "ece_test": ece(pe, yt), "abstain_threshold_for_95pct_val_accuracy": ethr,
            "test_coverage_at_threshold": float(ans.mean()),
            "test_accuracy_when_answering": float((epred[ans] == yt[ans]).mean())}

    save_json(out, REPORTS / "eval_metrics.json")
    import csv
    with open(REPORTS / "per_class_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pc[0].keys()))
        w.writeheader()
        w.writerows(sorted(pc, key=lambda r: -r["support"]))
    _figures(pc, pv, pt_raw, pt, yt, out)
    return out


def _figures(pc, pv, pt_raw, pt, yt, out):
    plt = plotting.plt
    # Per-class F1 vs training frequency.
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [r["train_count"] for r in pc]
    ax.scatter(x, [r["f1"] for r in pc], s=22, color=plotting.SERIES[0], label="Fine-tuned BERT",
               edgecolor=plotting.SURFACE, linewidth=1, zorder=3)
    if "baseline_f1" in pc[0]:
        ax.scatter(x, [r["baseline_f1"] for r in pc], s=14, color=plotting.SERIES[1], marker="s",
                   label="TF-IDF baseline", edgecolor=plotting.SURFACE, linewidth=1, zorder=2)
    for r in sorted(pc, key=lambda r: r["f1"])[:5]:
        ax.annotate(r["label"], (r["train_count"], r["f1"]), xytext=(5, -3),
                    textcoords="offset points", fontsize=7, color=plotting.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Training clauses for the class (log)")
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class F1 against training frequency")
    ax.legend(loc="lower right")
    plotting.save(fig, FIGURES / "per_class_f1_vs_frequency.png")

    # Reliability diagram.
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=plotting.INK_2, lw=1, ls="--")
    for probs, lbl, c in ((pt_raw, "Raw", plotting.SERIES[1]), (pt, "Temperature-scaled", plotting.SERIES[0])):
        rows = reliability(probs, yt)
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", ms=5, lw=2, color=c, label=lbl)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration (test)")
    ax.legend(loc="upper left")
    plotting.save(fig, FIGURES / "reliability.png")

    # Coverage vs accuracy.
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    cc = out["coverage_curve_test"]
    ax.plot([r["coverage"] for r in cc], [r["accuracy"] for r in cc], lw=2, color=plotting.SERIES[0],
            marker="o", ms=4)
    cal = out["calibration"]
    ax.axvline(cal["test_coverage_at_threshold"], color=plotting.INK_2, lw=1, ls="--")
    ax.text(cal["test_coverage_at_threshold"] - 0.01, min(r["accuracy"] for r in cc),
            f"95% target\n{cal['test_coverage_at_threshold']:.0%} answered", ha="right",
            fontsize=8, color=plotting.INK_2)
    ax.invert_xaxis()
    ax.set_xlabel("Share of clauses the model answers (coverage)")
    ax.set_ylabel("Accuracy on answered clauses")
    ax.set_title("Abstaining on low-confidence clauses")
    plotting.save(fig, FIGURES / "coverage_accuracy.png")


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2)[:4000])
