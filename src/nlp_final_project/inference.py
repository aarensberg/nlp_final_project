"""One inference path for the three classifier variants: BERT, TF-IDF, combination.

Used by `evaluate`, `audit` and the `classify` CLI, so the numbers in the
reports come from exactly the code a user runs.

    fit_inference_config()  - choose temperatures, the BERT weight and one abstention
                              threshold per variant, on the VALIDATION split only;
                              save them with fingerprints of every input.
    Predictor               - load the saved config, refuse it if a model file,
                              the tokenizer or the label order changed, then
                              predict / abstain on new clauses.

A confidence score is the model's calibrated estimate that the predicted clause
TYPE is right. It says nothing about the clause's legal validity or effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np

from .config import ARTIFACTS, REPORTS, get_device, load_json, save_json
from .data import label_names, load_ledgar, split_arrays
from .fingerprint import cached_npz, sha256_file, sha256_files, sha256_json, sha256_texts
from .metrics import MACRO_F1_DEFINITION, answers, fit_temperature, macro_f1, select_threshold, softmax

BERT_DIR = ARTIFACTS / "classifier"
TFIDF_PATH = ARTIFACTS / "baseline.joblib"
CONFIG_PATH = ARTIFACTS / "inference_config.json"
CONFIG_COPY = REPORTS / "inference_config.json"  # committed copy (artifacts/ is git-ignored)
VARIANTS = ("ensemble", "bert", "tfidf")
TARGET_ACCURACY = 0.95  # abstention target, met on validation
WEIGHT_GRID = [round(x, 2) for x in np.linspace(0, 1, 21)]
# Bump when the meaning of cached logits or of the saved config changes.
# 2: numeric context in cache keys; tie-aware thresholds (None = abstain on all);
#    macro-F1 over the fixed 100-label universe.
INFERENCE_SCHEMA = 2
CONFIDENCE_NOTE = ("Confidence = calibrated score that the clause TYPE is correct. It is not a "
                   "probability that the clause is legally valid, enforceable or risk-free.")


# ------------------------------------------------------------- fingerprints ----
def bert_fingerprint() -> dict:
    tcfg = load_json(BERT_DIR / "training_config.json")
    return {
        "weights_sha256": sha256_files([BERT_DIR / "model.safetensors", BERT_DIR / "config.json"]),
        "tokenizer_sha256": sha256_files([BERT_DIR / "tokenizer.json", BERT_DIR / "tokenizer_config.json"]),
        "max_length": tcfg["max_length"],
        "truncation": "keep first max_length tokens",
    }


def bert_numeric_context() -> dict:
    """How BERT logits are computed HERE. Same weights + different context = different numbers.

    Policy: cached logits are reused only if this context is identical; moving
    between CPU and MPS/CUDA (or upgrading torch/transformers) recomputes them.
    Measured effect of CPU (float32) vs M3 GPU (bfloat16): max |dp| 0.012 on BERT
    probabilities, no label change on 200 test clauses.
    """
    import torch
    import transformers

    dev = get_device().type
    return {"backend": dev, "precision": "float32" if dev == "cpu" else "bfloat16 autocast",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "batching": "length-sorted batches of 64, padded to a multiple of 64"}


def tfidf_fingerprint() -> dict:
    return {"joblib_sha256": sha256_file(TFIDF_PATH)}


def tfidf_numeric_context() -> dict:
    import sklearn

    return {"backend": "cpu", "precision": "float64", "sklearn": sklearn.__version__}


def split_fingerprint(split: str) -> dict:
    X, y = split_arrays(split)
    return {"split": split, "n": len(X), "sha256": sha256_texts(X, y),
            "hf_fingerprint": load_ledgar()[split]._fingerprint}


def labels_fingerprint(names: list[str]) -> str:
    return sha256_json(names)


# -------------------------------------------------------------- raw scores ----
class _Bert:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(BERT_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(self.device).eval()
        self.max_length = load_json(BERT_DIR / "training_config.json")["max_length"]
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]

    def logits(self, texts: list[str]) -> np.ndarray:
        from .train import predict_logits, tokenize

        return predict_logits(self.model, tokenize(self.tok, texts, self.max_length),
                              self.tok.pad_token_id, self.device, desc="bert")


class _Tfidf:
    def __init__(self):
        b = joblib.load(TFIDF_PATH)
        self.vec, self.clf = b["vectorizer"], b["classifier"]

    def logits(self, texts: list[str]) -> np.ndarray:
        return self.clf.decision_function(self.vec.transform(texts)).astype(np.float64)


def split_logits(split: str, which: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached logits of `which` ('bert' | 'tfidf') on a dataset split.

    The cache key includes the model and data fingerprints: if either changes
    the cache is recomputed instead of being reused.
    """
    fp = bert_fingerprint() if which == "bert" else tfidf_fingerprint()
    ctx = bert_numeric_context() if which == "bert" else tfidf_numeric_context()
    meta = {"schema": INFERENCE_SCHEMA, "model": which, **fp, "numeric_context": ctx,
            "data": split_fingerprint(split)}
    X, y = split_arrays(split)
    model_cls = _Bert if which == "bert" else _Tfidf
    z = cached_npz(ARTIFACTS / f"{which}_{split}_logits.npz", meta,
                   lambda: {"logits": model_cls().logits(X), "labels": y})
    return z["logits"], z["labels"]


# ----------------------------------------------------- validation-only fit ----
def fit_inference_config() -> dict:
    """Choose every inference parameter on the validation split, then save it."""
    names = label_names()
    lb, yv = split_logits("validation", "bert")
    lt, yv2 = split_logits("validation", "tfidf")
    assert np.array_equal(yv, yv2)

    T_bert, T_tfidf = fit_temperature(lb, yv), fit_temperature(lt, yv)
    pb, pt = softmax(lb, T_bert), softmax(lt, T_tfidf)
    val_f1 = {w: macro_f1(yv, (w * pb + (1 - w) * pt).argmax(1)) for w in WEIGHT_GRID}
    w = max(WEIGHT_GRID, key=lambda k: val_f1[k])  # first maximum, same rule as before (np.argmax)
    pe = w * pb + (1 - w) * pt
    probs = {"bert": pb, "tfidf": pt, "ensemble": pe}
    details = {v: select_threshold(p, yv, TARGET_ACCURACY) for v, p in probs.items()}
    thresholds = {v: d["threshold"] for v, d in details.items()}

    cfg = {
        "schema": INFERENCE_SCHEMA,
        "created_from": "validation split only (test never used to choose a parameter)",
        "macro_f1_definition": MACRO_F1_DEFINITION,
        "labels": names,
        "labels_sha256": labels_fingerprint(names),
        "data": {"dataset": "coastalcph/lex_glue/ledgar", "validation": split_fingerprint("validation")},
        "bert": {"dir": str(BERT_DIR.relative_to(ARTIFACTS.parent)), **bert_fingerprint(),
                 "numeric_context_of_fit": bert_numeric_context(),
                 "base_model": load_json(BERT_DIR / "training_config.json")["model_name"],
                 "temperature": T_bert},
        "tfidf": {"path": str(TFIDF_PATH.relative_to(ARTIFACTS.parent)), **tfidf_fingerprint(),
                  "numeric_context_of_fit": tfidf_numeric_context(), "temperature": T_tfidf},
        "ensemble": {"weight_bert": w, "weight_tfidf": round(1 - w, 2),
                     "rule": "p = w * softmax(bert/T_bert) + (1-w) * softmax(tfidf/T_tfidf)",
                     "weight_chosen_by": "validation macro-F1 over grid", "val_macro_f1_by_weight": val_f1},
        "abstention": {
            "rule": ("answer only if max calibrated score >= threshold of THAT variant; the threshold "
                     "is the lowest whole tie group whose accepted set reaches the target; "
                     "threshold null = no non-empty set reaches it -> always abstain"),
            "target_validation_accuracy": TARGET_ACCURACY,
            "thresholds": thresholds,
            "threshold_details_on_validation": details,
            "validation_coverage": {v: float(answers(p.max(1), thresholds[v]).mean()) for v, p in probs.items()},
        },
        "note": CONFIDENCE_NOTE,
    }
    save_json(cfg, CONFIG_PATH)
    save_json(cfg, CONFIG_COPY)
    return cfg


# ----------------------------------------------------------------- predict ----
@dataclass
class Prediction:
    variant: str
    label: str
    confidence: float
    abstain: bool
    top: list[tuple[str, float]]


class ConfigMismatchError(RuntimeError):
    pass


class Predictor:
    """Applies the saved, validation-chosen configuration. Refuses stale artifacts."""

    def __init__(self, config_path: Path = CONFIG_PATH, verify: bool = True):
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} missing: run `uv run nlp-final-project evaluate` first")
        self.cfg = json.loads(config_path.read_text())
        self.names = self.cfg["labels"]
        if verify:
            self.verify()

    def verify(self) -> None:
        problems = []
        if labels_fingerprint(label_names()) != self.cfg["labels_sha256"]:
            problems.append("dataset label order")
        fb = bert_fingerprint()
        for k in ("weights_sha256", "tokenizer_sha256", "max_length"):
            if fb[k] != self.cfg["bert"][k]:
                problems.append(f"bert {k}")
        if tfidf_fingerprint()["joblib_sha256"] != self.cfg["tfidf"]["joblib_sha256"]:
            problems.append("tfidf model file")
        cfg_labels = [load_json(BERT_DIR / "config.json")["id2label"][str(i)] for i in range(len(self.names))]
        if cfg_labels != self.names:
            problems.append("bert id2label order")
        if self.cfg.get("schema") != INFERENCE_SCHEMA:
            problems.append(f"config schema {self.cfg.get('schema')} != {INFERENCE_SCHEMA}")
        if problems:
            raise ConfigMismatchError("inference config does not match current artifacts: "
                                      + ", ".join(problems) + ". Re-run `evaluate` to refit it.")

    def numeric_context_warning(self) -> str | None:
        """Thresholds were fitted on logits computed in a given context (e.g. mps/bf16).
        Running elsewhere (e.g. CPU/float32) is allowed but probabilities differ slightly."""
        fit, now = self.cfg["bert"].get("numeric_context_of_fit", {}), bert_numeric_context()
        keys = ("backend", "precision")
        if any(fit.get(k) != now[k] for k in keys):
            return (f"note: thresholds were fitted with BERT on {fit.get('backend')}/{fit.get('precision')}; "
                    f"now running on {now['backend']}/{now['precision']} (measured CPU vs M3 GPU: "
                    f"max |dp| 0.012 on BERT probabilities). Decisions near a threshold may differ.")
        return None

    @cached_property
    def _bert(self) -> _Bert:
        return _Bert()

    @cached_property
    def _tfidf(self) -> _Tfidf:
        return _Tfidf()

    def threshold(self, variant: str) -> float | None:
        """None = no threshold reached the validation target: this variant always abstains."""
        return self.cfg["abstention"]["thresholds"][variant]

    def probs_from_logits(self, bert_logits: np.ndarray, tfidf_logits: np.ndarray) -> dict[str, np.ndarray]:
        pb = softmax(bert_logits, self.cfg["bert"]["temperature"])
        pt = softmax(tfidf_logits, self.cfg["tfidf"]["temperature"])
        w = self.cfg["ensemble"]["weight_bert"]
        return {"bert": pb, "tfidf": pt, "ensemble": w * pb + (1 - w) * pt}

    def probs(self, texts: list[str], variants=VARIANTS) -> dict[str, np.ndarray]:
        need_bert = any(v in ("bert", "ensemble") for v in variants)
        need_tfidf = any(v in ("tfidf", "ensemble") for v in variants)
        n = len(self.names)
        lb = self._bert.logits(texts) if need_bert else np.zeros((len(texts), n))
        lt = self._tfidf.logits(texts) if need_tfidf else np.zeros((len(texts), n))
        p = self.probs_from_logits(lb, lt)
        return {v: p[v] for v in variants}

    def decide(self, probs: np.ndarray, variant: str, k: int = 3) -> list[Prediction]:
        if not 1 <= k <= len(self.names):
            raise ValueError(f"k must be between 1 and {len(self.names)}, got {k}")
        thr = self.threshold(variant)
        out = []
        for row in probs:
            top = np.argsort(-row, kind="stable")[:k]
            conf = float(row[top[0]])
            out.append(Prediction(variant, self.names[top[0]], conf, not bool(answers(conf, thr)),
                                  [(self.names[i], float(row[i])) for i in top]))
        return out

    def predict(self, texts: list[str], variant: str = "ensemble", k: int = 3) -> list[Prediction]:
        return self.decide(self.probs(texts, (variant,))[variant], variant, k)


# --------------------------------------------------------------------- CLI ----
def cli_classify(argv=None) -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="nlp-final-project classify",
                                description="Predict the clause type of one or more clauses.")
    p.add_argument("clause", nargs="*", help="clause text (omit to read one clause from stdin)")
    p.add_argument("--variant", choices=VARIANTS, default="ensemble")
    p.add_argument("-k", type=int, default=3, help="number of candidate labels to show (1-100)")
    p.add_argument("--json", action="store_true", help="machine-readable output on stdout (logs on stderr)")
    a = p.parse_args(argv)
    if not 1 <= a.k <= 100:
        p.error("-k must be between 1 and 100")
    texts = [" ".join(a.clause).strip()] if a.clause else [sys.stdin.read().strip()]
    if not texts[0]:
        p.error("empty clause")
    pred = Predictor()
    warn = pred.numeric_context_warning()
    if warn:
        print(warn, file=sys.stderr)
    thr = pred.threshold(a.variant)
    for t, r in zip(texts, pred.predict(texts, a.variant, a.k)):
        if a.json:
            print(json.dumps({"clause": t, "variant": r.variant, "label": r.label, "confidence": r.confidence,
                              "abstain": r.abstain, "threshold": thr,
                              "threshold_note": (None if thr is not None else
                                                 "no threshold reached the validation target: always abstains"),
                              "top": r.top, "note": CONFIDENCE_NOTE}, indent=2))
            continue
        verdict = ("ABSTAIN - below the confidence threshold, send to a lawyer" if r.abstain
                   else "answered (confidence above threshold)")
        thr_txt = f"{thr:.3f}" if thr is not None else "none reached the target: always abstains"
        print(f"variant      : {r.variant}  (threshold {thr_txt}, chosen on validation)")
        print(f"predicted    : {r.label}  (confidence {r.confidence:.3f})")
        print(f"decision     : {verdict}")
        print("candidates   : " + " | ".join(f"{lbl} {s:.3f}" for lbl, s in r.top))
        print(f"note         : {CONFIDENCE_NOTE}")
