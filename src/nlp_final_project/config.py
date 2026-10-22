"""Shared paths, defaults and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

# NLP_PROJECT_ROOT lets tests run the CLI against an empty folder.
ROOT = Path(os.environ.get("NLP_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
ARTIFACTS = ROOT / "artifacts"  # model weights, embeddings, predictions (git-ignored)
REPORTS = ROOT / "reports"  # metrics JSON + markdown (committed)
FIGURES = REPORTS / "figures"

SEED = 42
DATASET = ("coastalcph/lex_glue", "ledgar")
# Hub revisions (commit SHAs) pinned so a rebuild downloads exactly what we used.
DATASET_REVISION = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"

# LEDGAR is English (US SEC filings), so we use an English legal-domain BERT.
CLASSIFIER_MODEL = "nlpaueb/legal-bert-base-uncased"
CLASSIFIER_REVISION = "15b570cbf88259610b082a167dacc190124f60f6"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we use so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """GPU if available. Set NLP_DEVICE=cpu to force CPU (e.g. while training uses the GPU)."""
    if os.environ.get("NLP_DEVICE"):
        return torch.device(os.environ["NLP_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path):
    return json.loads(Path(path).read_text())
