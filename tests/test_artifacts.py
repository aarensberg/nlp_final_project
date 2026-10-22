"""Consistency tests on the trained artifacts (skipped when artifacts/ is empty)."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import needs_models
from nlp_final_project.config import ARTIFACTS, REPORTS
from nlp_final_project.data import label_names, split_arrays
from nlp_final_project.metrics import core_metrics

pytestmark = needs_models

# Reference test scores recorded before the inference refactor (micro-F1, macro-F1, %).
REFERENCE = {"tfidf": (87.70, 82.19), "bert": (86.35, 76.27), "ensemble": (88.43, 82.84)}
# Tolerance between raw-text inference and the cached probabilities. Measured on
# 200 test clauses: 0.0 on the Apple-M3 GPU (same bf16 path, even one clause at a
# time); max 0.012 (BERT) / 0.0024 (combination) with 100% same labels on CPU,
# where BERT runs in float32 instead of bfloat16. 0.02 leaves margin for that.
BERT_PROB_ATOL = 0.02
MIN_ARGMAX_AGREEMENT = 0.98


@pytest.fixture(scope="module")
def predictor():
    from nlp_final_project.inference import Predictor
    return Predictor()


def test_label_order_is_identical_everywhere(predictor):
    import joblib
    names = label_names()
    cfg = json.loads((ARTIFACTS / "classifier" / "config.json").read_text())
    assert [cfg["id2label"][str(i)] for i in range(len(names))] == names
    assert predictor.names == names
    clf = joblib.load(ARTIFACTS / "baseline.joblib")["classifier"]
    assert np.array_equal(clf.classes_, np.arange(len(names)))


def test_parameters_were_fitted_on_validation(predictor):
    c = predictor.cfg
    assert c["created_from"].startswith("validation")
    assert c["ensemble"]["weight_bert"] == 0.2
    assert set(c["abstention"]["thresholds"]) == {"ensemble", "bert", "tfidf"}
    assert len(set(c["abstention"]["thresholds"].values())) == 3  # one per variant


def test_probabilities_are_normalised_and_ensemble_is_weighted(predictor):
    from nlp_final_project.inference import split_logits
    from nlp_final_project.metrics import softmax
    lb, _ = split_logits("validation", "bert")
    lt, _ = split_logits("validation", "tfidf")
    p = predictor.probs_from_logits(lb[:200], lt[:200])
    for v in p:
        np.testing.assert_allclose(p[v].sum(1), 1.0, atol=1e-6)
    c = predictor.cfg
    expected = 0.2 * softmax(lb[:200], c["bert"]["temperature"]) + 0.8 * softmax(lt[:200], c["tfidf"]["temperature"])
    np.testing.assert_allclose(p["ensemble"], expected, atol=1e-12)


def test_abstention_uses_each_variants_own_threshold(predictor):
    for v in ("ensemble", "bert", "tfidf"):
        thr = predictor.threshold(v)
        probs = np.full((2, 100), 0.0)
        probs[0, 3], probs[0, 4] = thr + 1e-6, 1 - thr - 1e-6
        probs[1, 3], probs[1, 4] = thr - 1e-3, 1 - thr + 1e-3
        d = predictor.decide(probs, v)
        assert d[0].abstain is (d[0].confidence < thr)
        assert [x.abstain for x in d] == [probs[0].max() < thr, probs[1].max() < thr]


def test_shared_inference_reproduces_reported_test_scores(predictor):
    from nlp_final_project.inference import split_logits
    lb, y = split_logits("test", "bert")
    lt, _ = split_logits("test", "tfidf")
    probs = predictor.probs_from_logits(lb, lt)
    report = json.loads((REPORTS / "eval_metrics.json").read_text())
    for v, (mi, ma) in REFERENCE.items():
        m = core_metrics(y, probs[v].argmax(1))
        assert round(m["micro_f1"] * 100, 2) == mi
        assert round(m["macro_f1"] * 100, 2) == ma
        assert m["micro_f1"] == report["variants"][v]["test"]["micro_f1"]


def test_stale_config_is_refused(tmp_path):
    from nlp_final_project.inference import CONFIG_PATH, ConfigMismatchError, Predictor
    bad = json.loads(CONFIG_PATH.read_text())
    bad["bert"]["weights_sha256"] = "0" * 64
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ConfigMismatchError, match="weights_sha256"):
        Predictor(p)


@pytest.mark.slow
def test_raw_text_inference_matches_cached_probabilities(predictor):
    """Running BERT on raw texts (other batches) stays within the stated tolerance."""
    from nlp_final_project.inference import split_logits
    X, y = split_arrays("test")
    idx = np.arange(0, 10000, 50)  # 200 clauses
    lb, _ = split_logits("test", "bert")
    lt, _ = split_logits("test", "tfidf")
    cached = predictor.probs_from_logits(lb[idx], lt[idx])
    fresh = predictor.probs([X[i] for i in idx])
    np.testing.assert_allclose(fresh["tfidf"], cached["tfidf"], atol=1e-9)  # TF-IDF is exact
    for v in ("bert", "ensemble"):
        assert np.abs(fresh[v] - cached[v]).max() <= BERT_PROB_ATOL
        assert (fresh[v].argmax(1) == cached[v].argmax(1)).mean() >= MIN_ARGMAX_AGREEMENT


# ----------------------------------------------------------------- CLI ----
def _cli(*args, timeout=600):
    return subprocess.run([sys.executable, "-m", "nlp_final_project", *args],
                          capture_output=True, text=True, timeout=timeout)


def test_cli_usage():
    r = _cli()
    assert r.returncode == 0 and "classify" in r.stdout and "never trains" in r.stdout


@pytest.mark.slow
def test_cli_classify_json():
    r = _cli("classify", "--json", "This Agreement shall be governed by and construed in accordance with "
             "the laws of the State of New York, without regard to its conflict of laws principles.")
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out["label"] == "Governing Laws" and out["abstain"] is False
    assert "not a probability that the clause is legally valid" in out["note"]
    assert len(out["top"]) == 3


@pytest.mark.slow
def test_cli_search_runs_and_states_filter_limit():
    if shutil.which("true") is None:  # pragma: no cover
        pytest.skip()
    r = _cli("search", "termination with a 30-day notice period", "-k", "3")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "role of the duration is not checked" in r.stdout
    assert r.stdout.count("[") >= 3
