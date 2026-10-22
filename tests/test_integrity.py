"""Integrity checks on artifacts, without any heavy model."""

import json

import numpy as np
import pytest

from nlp_final_project.inference import ConfigMismatchError, check_label_outputs
from nlp_final_project.retrieval import IndexIntegrityError, ClauseIndex, validate_index_files, write_index_files


# ------------------------------------------------------------ TF-IDF outputs ----
def test_label_outputs_accepts_exact_order():
    check_label_outputs(np.arange(100), 100, 100, "tfidf")


@pytest.mark.parametrize("classes, n_out, msg", [
    (np.r_[1, 0, np.arange(2, 100)], 100, "wrong order"),
    (np.arange(99), 99, "99 outputs"),
    (np.r_[np.arange(99), 100], 100, "missing"),
])
def test_label_outputs_rejects_bad_models(classes, n_out, msg):
    with pytest.raises(ConfigMismatchError, match=msg):
        check_label_outputs(classes, n_out, 100, "tfidf")


def test_real_sklearn_model_with_a_missing_class_is_rejected():
    from sklearn.linear_model import LogisticRegression
    X = np.random.default_rng(0).normal(size=(30, 4))
    y = np.tile([0, 1, 3], 10)  # class 2 never seen -> 3 outputs, classes [0, 1, 3]
    clf = LogisticRegression(max_iter=200).fit(X, y)
    with pytest.raises(ConfigMismatchError):
        check_label_outputs(clf.classes_, clf.coef_.shape[0], 4, "tfidf")


# ---------------------------------------------------------------- index ----
SPEC = {"encoder": "test", "train_sha256": "x"}


@pytest.fixture
def index_dir(tmp_path):
    texts = ["clause a", "clause b", "clause c"]
    emb = np.eye(3, 4, dtype=np.float32)
    write_index_files(tmp_path, texts, np.array([0, 1, 2]), emb, {"spec": SPEC})
    return tmp_path


def test_valid_index_passes(index_dir):
    problems, meta = validate_index_files(index_dir, SPEC)
    assert problems == [] and meta["n_clauses"] == 3 and set(meta["files"]) == {
        "embeddings.npy", "labels.npy", "texts.json"}


@pytest.mark.parametrize("victim, corrupt", [
    ("embeddings.npy", lambda p: np.save(p, np.ones((3, 4), dtype=np.float32))),     # same size, new content
    ("labels.npy", lambda p: np.save(p, np.array([0, 1]))),                           # shorter
    ("texts.json", lambda p: p.write_text(json.dumps(["clause a", "clause b", "XXXXXX c"]))),
])
def test_corrupted_file_is_detected(index_dir, victim, corrupt):
    corrupt(index_dir / victim)
    problems, _ = validate_index_files(index_dir, SPEC)
    assert any(victim in p for p in problems), problems
    with pytest.raises(IndexIntegrityError, match=victim):
        ClauseIndex.load(index_dir, rebuild=False)


def test_missing_file_and_old_meta_format_are_detected(index_dir):
    meta = json.loads((index_dir / "meta.json").read_text())
    del meta["files"]
    (index_dir / "meta.json").write_text(json.dumps(meta))
    assert "older index format" in validate_index_files(index_dir, SPEC)[0][0]
    (index_dir / "labels.npy").unlink()
    assert "missing file" in validate_index_files(index_dir, SPEC)[0][0]


def test_spec_change_is_detected(index_dir):
    problems, _ = validate_index_files(index_dir, SPEC | {"encoder": "other"})
    assert problems == ["built with a different encoder/config/clause bank"]


def test_length_mismatch_with_consistent_hashes_is_detected(tmp_path):
    # meta written for 3 texts but 2 embedding rows: hashes match the files, lengths do not.
    write_index_files(tmp_path, ["a", "b", "c"], np.array([0, 1, 2]), np.eye(2, 4, dtype=np.float32),
                      {"spec": SPEC})
    problems, _ = validate_index_files(tmp_path, SPEC)
    assert any("length mismatch" in p for p in problems)


# ---------------------------------------------------------------- resume ----
def test_strict_resume_without_checkpoint_fails(tmp_path):
    from nlp_final_project.train import ResumeError
    from test_resume import _run
    with pytest.raises(ResumeError, match="no checkpoint"):
        _run(tmp_path / "run", resume=True)


def test_resume_if_present_starts_new_run_then_resumes(tmp_path):
    import torch
    from test_resume import _cfg, _data, _final_state, _tiny_model
    from nlp_final_project.train import fit
    etr, ytr, eva, yva = _data()
    cfg = _cfg(tmp_path / "run")
    cfg.resume_if_present = True
    torch.manual_seed(0)
    log = fit(cfg, _tiny_model(), etr, ytr, eva, yva, pad_id=0, device=torch.device("cpu"), stop_after_steps=4)
    assert log["interrupted_at_step"] == 4  # new run started although a resume was allowed
    assert _final_state(tmp_path / "run")["state"]["step"] == 3  # checkpoint every 3 steps
    log = fit(cfg, _tiny_model(), etr, ytr, eva, yva, pad_id=0, device=torch.device("cpu"))
    assert len(log["epochs"]) == 2


def test_both_resume_flags_are_rejected(tmp_path):
    import torch
    from test_resume import _data, _tiny_model
    import nlp_final_project.train as T
    etr, ytr, eva, yva = _data()
    cfg = T.TrainConfig(resume=True, resume_if_present=True, output_dir=str(tmp_path / "r"))
    with pytest.raises(T.ResumeError, match="not both"):
        T.fit(cfg, _tiny_model(), etr, ytr, eva, yva, pad_id=0, device=torch.device("cpu"))
