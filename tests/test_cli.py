"""CLI behaviour that needs no trained model, run against an EMPTY project root."""

import os
import subprocess
import sys

import pytest

COMMANDS = ["eda", "baseline", "train", "evaluate", "classify", "index", "search", "audit", "manifest", "all"]


def _run(root, *args, timeout=120):
    env = os.environ | {"NLP_PROJECT_ROOT": str(root)}
    return subprocess.run([sys.executable, "-m", "nlp_final_project", *args],
                          capture_output=True, text=True, timeout=timeout, env=env)


def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize("cmd", COMMANDS)
def test_help_does_nothing_else(tmp_path, cmd):
    r = _run(tmp_path, cmd, "--help")
    assert r.returncode == 0, r.stderr[-1500:]
    assert "usage:" in r.stdout
    assert _files(tmp_path) == []  # no report, cache, model or folder written


def test_top_level_help_and_unknown_command(tmp_path):
    r = _run(tmp_path, "--help")
    assert r.returncode == 0 and "never trains" in r.stdout
    r = _run(tmp_path, "nonsense")
    assert r.returncode == 2 and "unknown command" in r.stderr
    assert _files(tmp_path) == []


@pytest.mark.parametrize("args", [
    ("classify", "-k", "0", "some clause"),
    ("classify", "-k", "101", "some clause"),
    ("classify", "   "),
    ("search", "-k", "0", "termination"),
    ("search", "   "),
])
def test_invalid_arguments_are_rejected_before_any_loading(tmp_path, args):
    r = _run(tmp_path, *args)
    assert r.returncode == 2, (r.stdout, r.stderr[-800:])
    assert _files(tmp_path) == []


def test_all_refuses_to_run_without_models(tmp_path):
    r = _run(tmp_path, "all")
    assert r.returncode != 0 and "missing models" in r.stderr
    assert _files(tmp_path) == []


def test_numeric_context_changes_with_backend(monkeypatch):
    from nlp_final_project.inference import bert_numeric_context
    monkeypatch.setenv("NLP_DEVICE", "cpu")
    cpu = bert_numeric_context()
    monkeypatch.setenv("NLP_DEVICE", "mps")
    gpu = bert_numeric_context()
    assert cpu["precision"] == "float32" and gpu["precision"] == "bfloat16 autocast"
    assert cpu != gpu  # -> different cache key -> logits recomputed, never reused across backends


def test_cache_invalidated_by_numeric_context_only(tmp_path):
    import numpy as np
    from nlp_final_project.fingerprint import cached_npz
    calls = []
    meta = {"weights_sha256": "w", "numeric_context": {"backend": "mps", "precision": "bfloat16 autocast"}}
    f = lambda: calls.append(1) or {"x": np.arange(2)}  # noqa: E731
    cached_npz(tmp_path / "c.npz", meta, f, log=lambda m: None)
    same_weights_other_backend = meta | {"numeric_context": {"backend": "cpu", "precision": "float32"}}
    cached_npz(tmp_path / "c.npz", same_weights_other_backend, f, log=lambda m: None)
    assert len(calls) == 2


def test_manifest_schema_without_artifacts(tmp_path):
    """Manifest in an empty root: schema complete, artifacts marked absent, no absolute path."""
    r = _run(tmp_path, "manifest")
    assert r.returncode == 0, r.stderr[-1500:]
    import json
    m = json.loads((tmp_path / "reports" / "run_manifest.json").read_text())
    from nlp_final_project.manifest import REQUIRED_KEYS
    assert set(REQUIRED_KEYS) <= set(m)
    assert m["artifacts"]["bert_weights"]["present"] is False
    assert "T" in m["generated_at"] and ("+" in m["generated_at"] or "Z" in m["generated_at"])
    text = json.dumps(m)
    assert str(tmp_path) not in text and "/Users/" not in text and "/home/" not in text
