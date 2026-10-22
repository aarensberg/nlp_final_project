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


# ------------------------------------------------- abstention thresholds ----
from nlp_final_project.metrics import answers, macro_f1, select_threshold  # noqa: E402


def _probs(conf):
    conf = np.asarray(conf, dtype=float)
    return np.stack([conf, 1 - conf], 1)  # argmax = class 0 whenever conf >= 0.5


def test_threshold_accepts_whole_tie_groups():
    # 0.95 correct; four clauses at 0.9 (two correct, two wrong); 0.6 wrong. Target 0.75.
    # Old code cut inside the 0.9 group and returned 0.9, whose accepted set is 3/5 = 0.6.
    probs = _probs([0.95, 0.9, 0.9, 0.9, 0.9, 0.6])
    y = np.array([0, 0, 0, 1, 1, 1])
    r = select_threshold(probs, y, 0.75)
    acc = answers(probs.max(1), r["threshold"])
    assert r["threshold"] == 0.95 and acc.sum() == 1 == r["n_accepted"]
    assert (probs.argmax(1)[acc] == y[acc]).mean() >= 0.75


def test_threshold_impossible_target_abstains_even_at_confidence_one():
    probs = np.array([[1.0, 0.0], [0.6, 0.4]])
    r = select_threshold(probs, np.array([1, 1]), 0.95)
    assert r == {"target": 0.95, "feasible": False, "threshold": None, "n_accepted": 0,
                 "coverage": 0.0, "accuracy_accepted": None}
    assert not answers(probs.max(1), r["threshold"]).any()  # 1.0 is NOT accepted


def test_threshold_confidence_one_group_can_be_accepted():
    probs = _probs([1.0, 1.0, 0.7])
    r = select_threshold(probs, np.array([0, 0, 1]), 0.95)
    assert r["threshold"] == 1.0 and r["n_accepted"] == 2 and r["accuracy_accepted"] == 1.0


def test_threshold_all_correct_answers_everything():
    probs = _probs([0.9, 0.8, 0.55])
    r = select_threshold(probs, np.array([0, 0, 0]), 0.95)
    assert r["threshold"] == 0.55 and r["coverage"] == 1.0


def test_threshold_empty_input_and_zero_coverage():
    r = select_threshold(np.zeros((0, 2)), np.zeros(0, dtype=int), 0.95)
    assert r["threshold"] is None and r["coverage"] == 0.0 and r["accuracy_accepted"] is None
    assert answers(np.array([0.99, 0.5]), None).tolist() == [False, False]


# ---------------------------------------------------------------- macro-F1 ----
from sklearn.metrics import f1_score  # noqa: E402

from nlp_final_project.metrics import bootstrap_ci, macro_f1_absent_one, macro_sensitivity  # noqa: E402


def _sk(y, p, n=100):
    return f1_score(y, p, labels=range(n), average="macro", zero_division=0)


def test_macro_f1_single_wrong_example_is_zero_not_098():
    y, p = np.array([3]), np.array([7])
    assert macro_f1(y, p) == 0.0 == _sk(y, p)
    assert macro_f1_absent_one(y, p) == pytest.approx(0.98)  # old convention: 98 absent labels count 1


@pytest.mark.parametrize("seed, n_labels, n, missing", [
    (0, 10, 50, []), (1, 10, 30, [3, 7]), (2, 100, 400, list(range(90, 100))), (3, 100, 10_000, []),
])
def test_macro_f1_equals_sklearn(seed, n_labels, n, missing):
    rng = np.random.default_rng(seed)
    allowed = np.setdiff1d(np.arange(n_labels), missing)
    y = rng.choice(allowed, n)
    p = np.where(rng.random(n) < 0.7, y, rng.choice(allowed, n))
    assert macro_f1(y, p, n_labels) == pytest.approx(_sk(y, p, n_labels), abs=1e-12)


def test_macro_f1_fixed_universe_when_a_class_disappears():
    y = np.array([0, 0, 1, 2])
    p = np.array([0, 0, 1, 1])
    assert macro_f1(y, p, n_labels=3) == pytest.approx((1 + 2 / 3 + 0) / 3)
    sub = macro_f1(y[:3], p[:3], n_labels=3)  # class 2 vanished from truth AND predictions
    assert sub == pytest.approx((1 + 1 + 0) / 3) == pytest.approx(_sk(y[:3], p[:3], 3))
    assert macro_f1_absent_one(y[:3], p[:3], n_labels=3) == pytest.approx(1.0)  # sensitivity only
    assert f1_score(y[:3], p[:3], average="macro") == pytest.approx(1.0)  # implicit universe: drops it


def test_point_bootstrap_and_sensitivity_use_consistent_definitions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 100, 2000)
    p = np.where(rng.random(2000) < 0.8, y, rng.integers(0, 100, 2000))
    point = macro_f1(y, p)
    ci = bootstrap_ci(y, p, n_boot=200)["macro_f1_95ci"]
    assert ci[0] <= point <= ci[1]
    # the bootstrap applies the same function: recompute the first resample by hand
    idx = np.random.default_rng(0).integers(0, 2000, 2000)
    assert macro_f1(y[idx], p[idx]) == pytest.approx(_sk(y[idx], p[idx]))
    sens = macro_sensitivity(y, p, n_boot=50)
    assert sens["macro_f1_absent_one"] >= point


# ------------------------------------------------------- filtered ranking ----
def _index(texts, emb, query_vec):
    idx = ClauseIndex(texts, np.zeros(len(texts), dtype=int), np.asarray(emb, dtype=np.float32))
    idx.names = ["X"]
    idx.embed = lambda qs, is_question: np.tile(np.asarray(query_vec, dtype=np.float32), (len(qs), 1))
    return idx


def test_filtered_docs_are_ranked_among_eligible_only():
    # 300 fillers beat the only eligible clause on dense and it shares no BM25 term with the query.
    # Old behaviour: RRF over the whole bank gave it 0 and it was returned with an uninformative 0.
    # New behaviour: rankings are restricted to eligible docs, so it gets a real (positive) rank.
    texts = [f"governing law clause number {i}" for i in range(300)] + ["Notice of forty-five days is required."]
    emb = np.zeros((301, 4)); emb[:300, 0] = 1; emb[300, 1] = 1
    idx = _index(texts, emb, [1, 0, 0, 0])
    d = idx.search_detailed("governing law with a 45-day period", k=5, mode="hybrid", depth=200)
    assert d["n_eligible"] == 1 and d["n_returned"] == 1
    assert d["hits"][0].score > 0


def test_unranked_eligible_docs_are_not_padded_in():
    # 3 eligible docs, no BM25 overlap with the query, depth 1: only the dense rank-1 doc is ranked.
    texts = ["alpha 30 days", "beta 30 days", "gamma 30 days", "other text"]
    emb = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0, 0, 0, 1]])
    idx = _index(texts, emb, [1, 0, 0, 0])
    d = idx.search_detailed("zeta 30-day", k=5, mode="hybrid", depth=1)
    assert d["n_eligible"] == 3 and d["n_ranked"] == 1
    assert [h.text for h in d["hits"]] == ["alpha 30 days"]


def test_corpus_smaller_than_rrf_depth():
    texts = ["terminate on notice", "governing law", "payment terms"]
    idx = _index(texts, np.eye(3, 4), [1, 0, 0, 0])
    hits = idx.search("terminate", k=10, mode="hybrid", depth=200)
    assert 1 <= len(hits) <= 3 and hits[0].text == "terminate on notice"


def test_fewer_than_k_candidates_and_no_candidate():
    texts = ["terminate on 30 days notice", "pay within 30 days", "terminate on 60 days notice", "law"]
    idx = _index(texts, np.eye(4, 4), [1, 0, 0, 0])
    d = idx.search_detailed("termination with 30 days notice", k=10, mode="hybrid")
    assert d["n_eligible"] == 2 and d["n_returned"] == 2
    d0 = idx.search_detailed("termination with 45 days notice", k=10, mode="hybrid")
    assert d0["n_eligible"] == 0 and d0["hits"] == []


def test_ties_are_broken_by_bank_position():
    texts = ["same text", "same text", "same text"]
    idx = _index(texts, np.ones((3, 4)) / 2, [0.5, 0.5, 0.5, 0.5])
    for mode in ("dense", "bm25", "hybrid"):
        assert [h.text for h in idx.search("same text", k=3, mode=mode)] == texts
        ids = idx.top(*idx.scored(mode, idx.dense_scores(np.ones(4) / 2), idx.bm25_scores("same text")), 3)
        assert ids == [0, 1, 2]


def test_multiple_durations_any_or_all():
    texts = ["notice of 30 days", "cure within 10 days", "10 days to cure and 30 days notice", "law"]
    idx = _index(texts, np.eye(4, 4), [1, 1, 1, 1])
    any_ = idx.search_detailed("10 days cure or 30 days notice", k=10, mode="dense", period_match="any")
    all_ = idx.search_detailed("10 days cure or 30 days notice", k=10, mode="dense", period_match="all")
    assert any_["n_eligible"] == 3 and all_["n_eligible"] == 1
    assert [h.text for h in all_["hits"]] == ["10 days to cure and 30 days notice"]


def test_search_rejects_bad_k():
    idx = _index(["a"], np.eye(1, 4), [1, 0, 0, 0])
    with pytest.raises(ValueError):
        idx.search("a", k=0)
