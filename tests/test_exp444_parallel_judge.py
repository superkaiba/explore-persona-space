"""Unit tests for ``scripts/run_experiment_444.py::_judge_rows_parallel``.

The helper is the parallel-fan-out used by both judge loops in #444's
experiment driver (``_judge_cell_completions`` for full-eval and the
fp-calibration loop inside ``phase_fp_calibration``). It must:

1. Return verdicts in the SAME order as input jobs — downstream
   aggregation zips verdicts back onto rows positionally, so an
   out-of-order verdict mis-attributes a judgement.
2. Turn a per-job exception into ``{"_error": str(e)}`` rather than
   propagating — mirrors the serial loops' broad try/except so one bad
   row never aborts a chunk.
3. Return exactly one verdict per input job.

These tests stub ``_haiku_judge_call`` so they run in <1s on a GPU-less
VM without an Anthropic API key. They are unit tests on the concurrency
contract, not integration tests on the judge model.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest


def _load_driver() -> Any:
    """Import ``scripts/run_experiment_444.py`` despite its sibling-bootstrap import.

    The driver does ``from _bootstrap import ...`` which only resolves when
    ``scripts/`` itself is on ``sys.path``. We inject it once (idempotent)
    then import via importlib. Same pattern as
    ``tests/test_exp444_answer_slot_entropy.py``.
    """
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("run_experiment_444")


# ── 1. Order preservation ─────────────────────────────────────────────────────


def test_judge_rows_parallel_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verdict i must map to job i even when jobs finish out of order.

    We give the stub variable sleeps inversely proportional to the job
    index (job 0 sleeps longest, job N-1 finishes first) so a naive
    implementation that yields verdicts in completion order would produce
    a reversed sequence. ThreadPoolExecutor.map preserves submission
    order regardless, so the verdict at position i carries job i's
    payload back.
    """
    driver = _load_driver()

    n_jobs = 12

    def _stub(system: str, user: str) -> dict[str, Any]:
        # Parse the job index back out of the user string we encoded below.
        idx = int(user.split(":")[1])
        # Earlier jobs sleep longer so completion order is reversed.
        time.sleep(0.02 * (n_jobs - idx))
        return {"idx": idx, "system": system, "user": user}

    monkeypatch.setattr(driver, "_haiku_judge_call", _stub)

    jobs = [(f"sys-{i}", f"user:{i}") for i in range(n_jobs)]
    verdicts = driver._judge_rows_parallel(jobs, max_workers=8)

    assert len(verdicts) == n_jobs
    for i, v in enumerate(verdicts):
        assert v["idx"] == i, f"verdict[{i}] mismatched: {v}"
        assert v["user"] == f"user:{i}"


# ── 2. Per-job exception → {"_error": ...} ────────────────────────────────────


def test_judge_rows_parallel_wraps_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising job becomes {"_error": str(e)}; other jobs still complete.

    Mirrors the serial loops' ``except Exception as e: verdict = {"_error":
    str(e)}`` shape so the downstream JSONL row schema is unchanged
    whether the call succeeded or failed.
    """
    driver = _load_driver()

    def _stub(system: str, user: str) -> dict[str, Any]:
        if user == "boom":
            raise RuntimeError("simulated haiku failure")
        return {"ok": True, "user": user}

    monkeypatch.setattr(driver, "_haiku_judge_call", _stub)

    jobs = [
        ("sys", "first"),
        ("sys", "boom"),
        ("sys", "third"),
    ]
    verdicts = driver._judge_rows_parallel(jobs, max_workers=4)

    assert len(verdicts) == 3
    assert verdicts[0] == {"ok": True, "user": "first"}
    assert verdicts[1] == {"_error": "simulated haiku failure"}
    assert verdicts[2] == {"ok": True, "user": "third"}


# ── 3. Cardinality (one verdict per job) ──────────────────────────────────────


def test_judge_rows_parallel_returns_one_verdict_per_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input → empty output; large input → exact count match.

    Cardinality matters because the two callers zip ``verdicts`` against
    the in-chunk row list with ``strict=True``. A missing or extra
    verdict would raise at the zip rather than silently mis-attribute,
    but the contract is still that ``len(out) == len(jobs)`` exactly.
    """
    driver = _load_driver()

    call_count = 0
    lock = threading.Lock()

    def _stub(system: str, user: str) -> dict[str, Any]:
        nonlocal call_count
        with lock:
            call_count += 1
        return {"user": user}

    monkeypatch.setattr(driver, "_haiku_judge_call", _stub)

    # Empty case.
    assert driver._judge_rows_parallel([], max_workers=4) == []
    assert call_count == 0

    # Non-empty case: exactly 50 jobs in, exactly 50 verdicts out.
    jobs = [("sys", f"u-{i}") for i in range(50)]
    verdicts = driver._judge_rows_parallel(jobs, max_workers=8)
    assert len(verdicts) == 50
    assert call_count == 50
    # Each verdict carries its own job's user payload (extra order check).
    for i, v in enumerate(verdicts):
        assert v["user"] == f"u-{i}"
