"""Content fingerprints, and caches that refuse to be reused when an input changed.

Every cached array (logits, embeddings) is stored with the fingerprints of what
produced it: model weights, tokenizer, configuration, data. On load the stored
fingerprints are compared with the current ones; any difference invalidates
the cache, which is then recomputed (never silently reused).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: p.name):
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def sha256_texts(texts: list[str], labels: np.ndarray | None = None) -> str:
    """Order-sensitive fingerprint of a list of clauses (and their labels)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x1f")
    if labels is not None:
        h.update(np.asarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class StaleCacheError(RuntimeError):
    pass


def cached_npz(path: Path, meta: dict, compute: Callable[[], dict[str, np.ndarray]],
               log: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    """Load `path` if its stored meta equals `meta`, else recompute and overwrite.

    A cache without meta (written by an older version) is treated as stale.
    Messages go to stderr so that machine-readable stdout stays clean.
    """
    log = log or (lambda m: print(m, file=sys.stderr))
    if path.exists():
        z = np.load(path, allow_pickle=False)
        stored = json.loads(str(z["__meta__"])) if "__meta__" in z.files else None
        if stored == meta:
            return {k: z[k] for k in z.files if k != "__meta__"}
        diff = sorted(k for k in set(meta) | set(stored or {}) if (stored or {}).get(k) != meta.get(k))
        log(f"[cache] {path.name} is stale (changed: {', '.join(diff) or 'no metadata'}); recomputing")
    arrays = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    return arrays
