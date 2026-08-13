"""Shared helpers + constants for issue #2224 (PV screening on real corpora).

Unit-1 scope (plan v3 §4 P0a + P0c/4b-1): the corpus pool builder
(``issue2224_build_pools.py``) and the predictor capture/score driver
(``issue2224_predictor_scores.py``) share the constants, atomic-file and
reproducibility-metadata helpers here.

Content hygiene: helpers here never print corpus text — digests only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ISSUE = 2224
POOL_SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 1

POOLS_DIR_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "pools"
SCREENING_SCORES_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "screening_scores"
SMOKE_ROOT_DEFAULT = Path("/tmp/issue2224_smoke")


def repro_meta(script: str) -> dict:
    """Reproducibility metadata block (git provenance + env versions + ts).

    Uses the canonical ``orchestrate.provenance.git_provenance`` helper
    (records ``git_dirty`` / ``git_dirty_paths`` per code-style.md) rather
    than a fresh ``subprocess`` git shellout.
    """
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    def _v(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "?"

    meta: dict = {
        "script": script,
        "issue": ISSUE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": {m: _v(m) for m in ("torch", "transformers", "datasets", "numpy")},
    }
    meta.update(as_metadata_dict(git_provenance()))
    return meta


def sha256_file(path: Path | str) -> str:
    """Streaming sha256 of a file (hex digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(obj: dict, path: Path) -> None:
    """Write JSON atomically (tmp in the same dir + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a whole JSONL file atomically (tmp + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict) -> None:
    """Append ONE row to a JSONL file as a single ``O_APPEND`` write."""
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def count_jsonl_lines(path: Path) -> int:
    """Number of newline-terminated lines in a JSONL file (0 when absent)."""
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def truncate_jsonl(path: Path, n_keep: int) -> None:
    """Keep only the first ``n_keep`` lines of a JSONL file (atomic rewrite).

    Used by the resume path: a crash between a candidates-append and its meta
    checkpoint can leave MORE rows on disk than the sidecar records; the
    resume re-scans from the checkpointed position, so the surplus rows must
    be dropped or they would duplicate.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        for i, line in enumerate(src):
            if i >= n_keep:
                break
            dst.write(line)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, path)


def load_jsonl(path: Path) -> list[dict]:
    """Load a whole JSONL file into a list of dicts."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_seed(*parts: object, base: int = 0) -> int:
    """Deterministic 32-bit seed from string parts (PYTHONHASHSEED-proof).

    ``hash()`` on strings is salted per process, so seeding RNGs off it is
    machine/process-dependent; this sha256-based derivation reproduces across
    machines (the #1946 argsort-tie / determinism lesson, applied to seeding).
    """
    key = "|".join(str(p) for p in parts)
    h = hashlib.sha256(f"{base}|{key}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def token_stats(values: list[int]) -> dict:
    """min/p50/p90/max/mean digest for a token-count list (never raw text)."""
    import numpy as np

    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }
