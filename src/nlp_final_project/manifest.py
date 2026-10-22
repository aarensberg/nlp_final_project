"""Run manifest: what produced the current reports, in one JSON file.

    uv run nlp-final-project manifest          # writes reports/run_manifest.json (~10 s)

Contains no absolute path and no personal information: paths are relative to
the project root, and the command log stores commands as typed in the project.
Command records come from reports/validation_logs/commands.jsonl, appended by
the validation runs (see reports/final_validation.md).
"""

from __future__ import annotations

import datetime as dt
import getpass
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from .config import (ARTIFACTS, CLASSIFIER_MODEL, CLASSIFIER_REVISION, DATASET, DATASET_REVISION, EMBEDDING_MODEL,
                     EMBEDDING_REVISION, REPORTS, ROOT, SEED, get_device, save_json)
from .fingerprint import sha256_file

MANIFEST = REPORTS / "run_manifest.json"
COMMAND_LOG = REPORTS / "validation_logs" / "commands.jsonl"
SCHEMA = 1
REQUIRED_KEYS = ("schema", "generated_at", "git", "environment", "hardware", "data_and_models", "training",
                 "artifacts", "reports", "commands")


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _run(cmd: list[str], strip: bool = True) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        return out.strip() if strip else out.rstrip("\n")
    except Exception:
        return None


def sanitize(text: str) -> str:
    """Replace machine-specific paths: project root, temporary scratch folders, home directory."""
    text = text.replace(str(ROOT), "<project>")
    text = re.sub(r"(?:file://)?(?:/private)?/(?:tmp|var/folders)/\S*?/scratchpad", "<scratch>", text)
    return text.replace(str(Path.home()), "~")


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def _hash(p: Path) -> dict:
    if not p.exists():
        return {"path": _rel(p), "present": False}
    return {"path": _rel(p), "present": True, "sha256": sha256_file(p), "bytes": p.stat().st_size}


def build_manifest() -> dict:
    import datasets
    import numpy
    import sentence_transformers
    import sklearn
    import torch
    import transformers

    status = _run(["git", "status", "--porcelain"], strip=False)  # leading spaces are significant
    dev = get_device().type
    tcfg_path = ARTIFACTS / "classifier" / "training_config.json"
    tcfg = json.loads(tcfg_path.read_text()) if tcfg_path.exists() else None
    if tcfg and "output_dir" in tcfg:
        tcfg["output_dir"] = Path(tcfg["output_dir"]).name  # never store an absolute path
    commands = ([json.loads(sanitize(line)) for line in COMMAND_LOG.read_text().splitlines() if line.strip()]
                if COMMAND_LOG.exists() else [])
    report_files = sorted([*REPORTS.glob("*.json"), *REPORTS.glob("*.csv"), *REPORTS.glob("*.md"),
                           *(REPORTS / "figures").glob("*.png")])
    # final_validation.* records this manifest's hash, so it cannot be hashed here (circular).
    report_files = [p for p in report_files if p.name != MANIFEST.name and not p.name.startswith("final_validation.")]
    return {
        "schema": SCHEMA,
        "generated_at": _now(),
        "git": {"commit": _run(["git", "rev-parse", "HEAD"]), "dirty": bool(status),
                "changed_paths": sorted({line[3:] for line in (status or "").splitlines()})},
        "environment": {
            "python": platform.python_version(), "uv": (_run(["uv", "--version"]) or "").split(" (")[0],
            "torch": torch.__version__, "transformers": transformers.__version__,
            "datasets": datasets.__version__, "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "sentence-transformers": sentence_transformers.__version__, "lockfile": _hash(ROOT / "uv.lock")},
        "hardware": {"os": platform.system(), "os_release": platform.mac_ver()[0] or platform.release(),
                     "machine": platform.machine(), "backend": dev,
                     "bert_inference_precision": "float32" if dev == "cpu" else "bfloat16 autocast",
                     "embedding_precision": "float32"},
        "data_and_models": {"dataset": "/".join(DATASET), "dataset_revision": DATASET_REVISION,
                            "classifier_base": CLASSIFIER_MODEL, "classifier_revision": CLASSIFIER_REVISION,
                            "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION},
        "training": {"seed": SEED, "recorded_training_config": tcfg,
                     "provenance": "reports/training_provenance.md"},
        "artifacts": {
            "bert_weights": _hash(ARTIFACTS / "classifier" / "model.safetensors"),
            "bert_config": _hash(ARTIFACTS / "classifier" / "config.json"),
            "bert_tokenizer": _hash(ARTIFACTS / "classifier" / "tokenizer.json"),
            "bert_tokenizer_config": _hash(ARTIFACTS / "classifier" / "tokenizer_config.json"),
            "tfidf_baseline": _hash(ARTIFACTS / "baseline.joblib"),
            "inference_config": _hash(ARTIFACTS / "inference_config.json"),
            "index_meta": _hash(ARTIFACTS / "clause_index" / "meta.json"),
            "index_embeddings": _hash(ARTIFACTS / "clause_index" / "embeddings.npy"),
            "index_labels": _hash(ARTIFACTS / "clause_index" / "labels.npy"),
            "index_texts": _hash(ARTIFACTS / "clause_index" / "texts.json"),
        },
        "reports": [_hash(p) for p in report_files],
        "commands": commands,
    }


def check_no_private_paths(manifest: dict) -> None:
    text = json.dumps(manifest)
    user = getpass.getuser()
    for needle in (str(ROOT), str(Path.home()), "/Users/", "/home/", f"-{user}-", f"/{user}/"):
        if needle and needle in text:
            raise ValueError(f"manifest would contain an absolute/personal path ({needle!r})")


def write_manifest() -> dict:
    m = build_manifest()
    check_no_private_paths(m)
    save_json(m, MANIFEST)
    return m


def cli(argv=None) -> None:
    import argparse

    argparse.ArgumentParser(prog="nlp-final-project manifest",
                            description="Write reports/run_manifest.json (versions, hashes, commands).").parse_args(argv)
    m = write_manifest()
    print(f"manifest written: {_rel(MANIFEST)} (commit {m['git']['commit']}, dirty={m['git']['dirty']}, "
          f"{len(m['reports'])} report files, {len(m['commands'])} commands)", file=sys.stdout)
