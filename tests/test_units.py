"""Fast tests that need no trained model."""

import json

import numpy as np
import pytest

from nlp_final_project.fingerprint import cached_npz
from nlp_final_project.metrics import softmax, threshold_for_precision
from nlp_final_project.perturb import british, eu_jurisdiction, euro, gender_swap, has_us_state
from nlp_final_project.retrieval import ClauseIndex, SparseBM25, day_periods
from nlp_final_project.train import ResumeError, TrainConfig, check_resume_compatible


# ------------------------------------------------------------ day periods ----
@pytest.mark.parametrize("text, expected", [
    ("upon 30 days prior written notice", {30}),
    ("a 30-day notice period", {30}),
    ("thirty (30) days' notice", {30}),
    ("within thirty days", {30}),
    ("sixty (60) calendar days", {60}),
    ("five (5) Business Days", {5}),
    ("within 10 days of receipt and 30 days after termination", {10, 30}),
    ("no deadline is stated here", set()),
    ("payable within one year", set()),
])
def test_day_periods(text, expected):
    assert day_periods(text) == expected


def test_day_periods_known_limits():
    # Documented limits, asserted so a future change is noticed:
    assert day_periods("5 business days") == day_periods("5 calendar days")  # kind not kept
    assert day_periods("thirty-one days") == set()  # number word outside the list, no digits
    assert day_periods("pay within 30 days") == day_periods("30 days' notice of termination")  # role ignored


def _tiny_index():
    texts = ["Either party may terminate this Agreement upon thirty (30) days written notice.",
             "Payment is due within 30 days of invoice.",
             "Either party may terminate this Agreement upon ninety (90) days notice.",
             "This Agreement is governed by the laws of the State of New York."]
    idx = ClauseIndex(texts, np.array([0, 1, 0, 2]), np.zeros((4, 8), dtype=np.float32))
    idx.names = ["Terminations", "Payments", "Governing Laws"]
    return idx


def test_period_filter_keeps_only_matching_durations():
    hits = _tiny_index().search("termination with a 30-day notice period", k=10, mode="bm25")
    assert {h.text for h in hits} == {
        "Either party may terminate this Agreement upon thirty (30) days written notice.",
        "Payment is due within 30 days of invoice."}  # payment clause passes too: role is not checked


def test_period_filter_can_return_nothing():
    assert _tiny_index().search("termination with 45 days notice", k=10, mode="bm25") == []


def test_sparse_bm25_matches_textbook_formula():
    import math
    docs = ["rent payable within thirty days", "governing law of new york", "rent rent rent"]
    bm = SparseBM25(docs)
    toks = [d.split() for d in docs]
    avg = np.mean([len(t) for t in toks])
    ref = []
    for t in toks:
        s = 0.0
        for w in ["rent", "thirty"]:
            df = sum(w in d for d in toks)
            idf = math.log((3 - df + 0.5) / (df + 0.5) + 1)
            f = t.count(w)
            s += idf * f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * len(t) / avg))
        ref.append(s)
    np.testing.assert_allclose(bm.get_scores("rent thirty"), ref, rtol=1e-5)


# ---------------------------------------------------------- perturbations ----
def test_gender_swap_object_and_possessive():
    assert gender_swap("pay her within 30 days. Her salary. If he resigns, the shares are his.") == \
        "pay him within 30 days. His salary. If she resigns, the shares are hers."


def test_jurisdiction_adjective_and_exclusions():
    out = eu_jurisdiction("governed by the laws of the State of Delaware. Seller is a Delaware corporation.")
    assert "Delaware" not in out and " corporation" in out
    assert out.split("Seller is a ")[1].split()[0] in {"English", "German", "French", "Dutch", "Irish"}
    assert not has_us_state("the Southern District of New York")
    assert not has_us_state("courts located in New York County")


def test_currency_and_spelling():
    assert euro("$1,000 payable in U.S. dollars") == "EUR 1,000 payable in euros"
    assert british("the laboratory licensee fulfilled the judgment") == \
        "the laboratory licensee fulfilled the judgment"
    assert british("organization in favor of the center") == "organisation in favour of the centre"


# --------------------------------------------------------- probabilities ----
def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(50, 100)) * 5, t=0.9)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
    assert (p >= 0).all()


def test_threshold_reaches_target_on_its_own_data():
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(500, 10)) * 3)
    y = np.where(rng.random(500) < 0.8, probs.argmax(1), rng.integers(0, 10, 500))
    thr = threshold_for_precision(probs, y, 0.9)
    ans = probs.max(1) >= thr
    assert (probs.argmax(1)[ans] == y[ans]).mean() >= 0.9


# ----------------------------------------------------------------- caches ----
def test_cache_recomputes_when_meta_changes(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"x": np.arange(3)}

    path = tmp_path / "c.npz"
    cached_npz(path, {"model": "a"}, compute, log=lambda m: None)
    cached_npz(path, {"model": "a"}, compute, log=lambda m: None)
    assert len(calls) == 1  # reused
    cached_npz(path, {"model": "b"}, compute, log=lambda m: None)
    assert len(calls) == 2  # invalidated


def test_cache_without_meta_is_stale(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, x=np.arange(3))
    msgs = []
    cached_npz(path, {"model": "a"}, lambda: {"x": np.arange(4)}, log=msgs.append)
    assert msgs and "stale" in msgs[0]
    assert json.loads(str(np.load(path)["__meta__"])) == {"model": "a"}


# ----------------------------------------------------------------- resume ----
def test_resume_refuses_changed_epochs():
    saved = {k: v for k, v in TrainConfig(epochs=2).__dict__.items()}
    with pytest.raises(ResumeError, match="epochs"):
        check_resume_compatible(saved, TrainConfig(epochs=3))
    check_resume_compatible(saved, TrainConfig(epochs=2, resume=True, ckpt_every=50))  # same trajectory


def test_resume_refuses_checkpoint_without_config():
    with pytest.raises(ResumeError, match="no saved configuration"):
        check_resume_compatible(None, TrainConfig())
