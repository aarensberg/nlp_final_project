"""Held-out evaluation of the three classifier variants on the LEDGAR test split.

Variants: `ensemble` (shipped: weighted BERT + TF-IDF), `bert` (fine-tuned
Legal-BERT alone), `tfidf` (TF-IDF + logistic regression baseline).

Every parameter (temperatures, ensemble weight, abstention thresholds) is fitted
on VALIDATION by `inference.fit_inference_config()` and only applied to test.

Outputs: reports/eval_metrics.json, reports/per_class_test.csv,
reports/inference_config.json and figures in reports/figures/.

Run: uv run nlp-final-project evaluate
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import binomtest

from . import plotting
from .config import FIGURES, REPORTS, SEED, save_json
from .data import frequency_buckets, label_names, split_arrays
from .inference import VARIANTS, Predictor, fit_inference_config, split_logits
from .metrics import (MACRO_F1_DEFINITION, answers, bootstrap_ci, core_metrics, coverage_curve, ece, macro_f1,
                      per_class, reliability, softmax, top_confusions)
from .retrieval import norm_hash

LABEL = {"ensemble": "Combination (BERT 0.2 + TF-IDF 0.8)", "bert": "Legal-BERT alone",
         "tfidf": "TF-IDF + logistic regression"}
COLOR = {"ensemble": plotting.SERIES[0], "bert": plotting.SERIES[1], "tfidf": plotting.SERIES[2]}


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray, a: str, b: str) -> dict:
    """Exact McNemar test on the clauses where exactly one system is correct.

    It tests whether the two error rates differ (accuracy = micro-F1 here);
    it is NOT a test of the macro-F1 difference.
    """
    ab = int((a_correct & ~b_correct).sum())
    ba = int((~a_correct & b_correct).sum())
    p = binomtest(ab, ab + ba, 0.5).pvalue if ab + ba else 1.0
    return {f"{a}_right_{b}_wrong": ab, f"{b}_right_{a}_wrong": ba, "p_value": float(p),
            "tests": "difference in error rate (accuracy), not macro-F1"}


def paired_bootstrap_delta(y, pa, pb, n_boot: int = 1000, seed: int = SEED) -> dict:
    """95% CI of (A - B) for micro- and macro-F1, resampling the same clauses for both.

    Same macro-F1 definition as the point scores (fixed 100-label universe).
    Resampling unit = clause (no contract identifier in LEDGAR)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_mi, d_ma = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d_mi.append((pa[i] == y[i]).mean() - (pb[i] == y[i]).mean())
        d_ma.append(macro_f1(y[i], pa[i]) - macro_f1(y[i], pb[i]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {
        "delta_micro_f1_points": float(((pa == y).mean() - (pb == y).mean()) * 100),
        "delta_micro_f1_95ci_points": [x * 100 for x in ci(d_mi)],
        "delta_macro_f1_points": float((macro_f1(y, pa) - macro_f1(y, pb)) * 100),
        "delta_macro_f1_95ci_points": [x * 100 for x in ci(d_ma)],
    }


def run() -> dict:
    names = label_names()
    Xtr, ytr = split_arrays("train")
    Xte, _ = split_arrays("test")

    cfg = fit_inference_config()  # validation only
    P = Predictor()
    lbv, yv = split_logits("validation", "bert")
    ltv, _ = split_logits("validation", "tfidf")
    lbt, yt = split_logits("test", "bert")
    ltt, _ = split_logits("test", "tfidf")
    pv = P.probs_from_logits(lbv, ltv)
    pt = P.probs_from_logits(lbt, ltt)
    raw_t = {"bert": softmax(lbt), "tfidf": softmax(ltt)}

    train_hashes = {norm_hash(t) for t in Xtr}
    unseen = np.array([norm_hash(t) not in train_hashes for t in Xte])
    buckets = frequency_buckets(ytr, len(names))
    train_count = np.bincount(ytr, minlength=len(names))

    out: dict = {
        "primary_variant": "ensemble",
        "variants_described": LABEL,
        "parameters": "fitted on validation, see reports/inference_config.json",
        "macro_f1_definition": MACRO_F1_DEFINITION,
        "bootstrap": "1000 resamples of test CLAUSES (no contract id available, so intervals ignore "
                     "within-contract correlation and are likely somewhat too narrow)",
        "test_n": int(len(yt)),
        "test_n_without_train_duplicates": int(unseen.sum()),
        "variants": {},
    }
    preds, per_cls = {}, {}
    for v in VARIANTS:
        pred = pt[v].argmax(1)
        preds[v] = pred
        pc = per_class(yt, pred, names)
        per_cls[v] = pc
        for r, c in zip(pc, range(len(names))):
            r["train_count"], r["bucket"] = int(train_count[c]), buckets[c]
        thr = P.threshold(v)
        ans = answers(pt[v].max(1), thr)
        out["variants"][v] = {
            "description": LABEL[v],
            "test": core_metrics(yt, pred) | bootstrap_ci(yt, pred),
            "validation": core_metrics(yv, pv[v].argmax(1)),
            "test_without_train_duplicates": core_metrics(yt[unseen], pred[unseen]),
            "macro_f1_by_frequency_bucket": {
                b: float(np.mean([r["f1"] for r in pc if r["bucket"] == b])) for b in ("head", "torso", "tail")},
            "worst_classes": [{k: r[k] for k in ("label", "precision", "recall", "f1", "support", "train_count")}
                              for r in sorted(pc, key=lambda r: r["f1"])[:12]],
            "top_confusions": top_confusions(yt, pred, names),
            "calibration": {
                "temperature": (cfg[v]["temperature"] if v != "ensemble" else None),
                "ece_test_before_temperature": (ece(raw_t[v], yt) if v in raw_t else None),
                "ece_test": ece(pt[v], yt),
            },
            "abstention": {
                "threshold_chosen_on_validation": thr,
                "validation_target_correct_rate": cfg["abstention"]["target_validation_accuracy"],
                "validation_coverage": cfg["abstention"]["validation_coverage"][v],
                "test_coverage": float(ans.mean()),
                "test_n_handled": int(ans.sum()),
                "test_correct_rate_among_handled": float((pred[ans] == yt[ans]).mean()) if ans.any() else None,
                "test_correct_rate_among_abstained": float((pred[~ans] == yt[~ans]).mean()) if (~ans).any() else None,
            },
            "coverage_curve_test": coverage_curve(pt[v], yt),
        }

    out["comparisons"] = {
        "ensemble_vs_tfidf": paired_bootstrap_delta(yt, preds["ensemble"], preds["tfidf"])
        | {"mcnemar": mcnemar(preds["ensemble"] == yt, preds["tfidf"] == yt, "ensemble", "tfidf")},
        "ensemble_vs_bert": paired_bootstrap_delta(yt, preds["ensemble"], preds["bert"])
        | {"mcnemar": mcnemar(preds["ensemble"] == yt, preds["bert"] == yt, "ensemble", "bert")},
        "bert_vs_tfidf": paired_bootstrap_delta(yt, preds["bert"], preds["tfidf"])
        | {"mcnemar": mcnemar(preds["bert"] == yt, preds["tfidf"] == yt, "bert", "tfidf")},
    }
    save_json(out, REPORTS / "eval_metrics.json")
    _write_csv(per_cls, names)
    _figures(per_cls, pt, raw_t, yt, out)
    return out


def _write_csv(per_cls, names) -> None:
    rows = []
    for c in range(len(names)):
        base = per_cls["ensemble"][c]
        row = {"label": base["label"], "test_support": base["support"], "train_count": base["train_count"],
               "frequency_bucket": base["bucket"]}
        for v in VARIANTS:
            r = per_cls[v][c]
            row |= {f"{v}_precision": round(r["precision"], 4), f"{v}_recall": round(r["recall"], 4),
                    f"{v}_f1": round(r["f1"], 4)}
        rows.append(row)
    with open(REPORTS / "per_class_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["test_support"]))


def _figures(per_cls, pt, raw_t, yt, out) -> None:
    plt = plotting.plt
    # Per-class F1 vs training frequency, three variants.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v, marker, size in (("tfidf", "s", 14), ("bert", "^", 16), ("ensemble", "o", 22)):
        pc = per_cls[v]
        ax.scatter([r["train_count"] for r in pc], [r["f1"] for r in pc], s=size, marker=marker,
                   color=COLOR[v], label=LABEL[v], edgecolor=plotting.SURFACE, linewidth=0.8,
                   zorder=3 if v == "ensemble" else 2)
    for r in sorted(per_cls["ensemble"], key=lambda r: r["f1"])[:5]:
        ax.annotate(r["label"], (r["train_count"], r["f1"]), xytext=(5, -3), textcoords="offset points",
                    fontsize=7, color=plotting.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Training clauses for the class (log)")
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class test F1 against training frequency")
    ax.legend(loc="lower right", fontsize=8)
    plotting.save(fig, FIGURES / "per_class_f1_vs_frequency.png")

    # Reliability diagram: calibrated scores of each variant (+ raw BERT for reference).
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=plotting.INK_2, lw=1, ls="--")
    series = [(pt[v], LABEL[v] + (" (temperature-scaled)" if v != "ensemble" else ""), COLOR[v], "-")
              for v in VARIANTS]
    series.append((raw_t["bert"], "Legal-BERT alone, before scaling", COLOR["bert"], ":"))
    for probs, lbl, c, ls in series:
        rows = reliability(probs, yt)
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", ms=4, lw=2, ls=ls, color=c, label=lbl)
    ax.set_xlabel("Mean confidence (test)")
    ax.set_ylabel("Share correct (test)")
    ax.set_title("Calibration on the test split")
    ax.legend(loc="upper left", fontsize=7)
    plotting.save(fig, FIGURES / "reliability.png")

    # Coverage vs share correct, with each variant's own validation threshold.
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v in VARIANTS:
        cc = out["variants"][v]["coverage_curve_test"]
        ab = out["variants"][v]["abstention"]
        cc = [r for r in cc if r["accuracy"] is not None]
        ax.plot([r["coverage"] for r in cc], [r["accuracy"] for r in cc], lw=2, color=COLOR[v], label=LABEL[v])
        if ab["test_correct_rate_among_handled"] is not None:
            ax.scatter([ab["test_coverage"]], [ab["test_correct_rate_among_handled"]], s=60, color=COLOR[v],
                       edgecolor=plotting.SURFACE, linewidth=1.5, zorder=4)
    ax.axhline(0.95, color=plotting.INK_2, lw=1, ls="--")
    ax.text(0.52, 0.952, "95% validation target (not a test guarantee)", ha="right", va="bottom",
            fontsize=7, color=plotting.INK_2)
    ax.invert_xaxis()
    ax.set_xlabel("Share of test clauses handled (coverage)")
    ax.set_ylabel("Share correct among handled clauses")
    ax.set_title("Abstention: dots = each variant's validation threshold")
    ax.legend(loc="lower left", fontsize=8)
    plotting.save(fig, FIGURES / "coverage_accuracy.png")


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2)[:4000])
