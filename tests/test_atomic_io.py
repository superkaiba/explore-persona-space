"""Tests for the shared process-safe atomic-write module (#2336, plan v3 batch 0a).

Ported from the 4-test donor shape at ``tests/test_issue2329_r2_fixes.py:596-717`` with
the round-1 A2 hardening: ONE shared fork-hammer harness (``_run_hammer``) serves the safe
arm (test 1) AND the deliberately-unsafe control arm (test 2), so no later tuning can
split their parameters; a ``multiprocessing.Barrier`` between the WRITE and the PUBLISH
makes the unsafe-arm collision deterministic rather than scheduling-dependent.

The deliberately-unsafe temp derivation in ``_unsafe_writer`` lives in ``tests/`` — outside
the (unit-2) ``--check-shared-tmp-name`` scan scope — so no waiver comment is needed.
"""

from __future__ import annotations

import datetime
import json
import logging
import multiprocessing
import os
import queue
import shutil
import tempfile
from pathlib import Path

import pytest

import explore_persona_space.atomic_io as atomic_io
from explore_persona_space.atomic_io import (
    atomic_replace,
    save_npy_atomic,
    save_pt_atomic,
    savez_atomic,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)

_PAYLOAD = {"scope": "hammer", "criterion": "atomic", "rows": [1, 2, 3]}


# ── shared fork-hammer harness (serves tests 1 AND 2 — one config, no split) ──


def _safe_writer(dest: Path, barrier, stage_box: dict) -> None:
    """One write via the shared ``atomic_replace`` primitive.

    The barrier sits between the WRITE and the PUBLISH (the ``os.replace`` fires at
    context exit, after the barrier), matching the unsafe control's ordering exactly.
    """
    stage_box["stage"] = "open"
    with atomic_replace(dest) as tmp:
        fh = tmp.open("w", encoding="utf-8")
        stage_box["stage"] = "write"
        with fh:
            fh.write(json.dumps(_PAYLOAD))
        barrier.wait(timeout=60)
        stage_box["stage"] = "replace"
    # context exit above ran os.replace(tmp, dest)


def _unsafe_writer(dest: Path, barrier, stage_box: dict) -> None:
    """Deliberately-UNSAFE control: the pre-#2329 shared deterministic temp name.

    Parent-dir creation is identical to ``atomic_replace``'s
    (``mkdir(parents=True, exist_ok=True)``) so a missing-parent open failure can never
    masquerade as the collision; all 8 workers derive the SAME ``<name>.tmp`` path, the
    barrier forces every write to complete before any publish, and the first
    ``os.replace`` consumes the shared temp — every later worker dies
    ``FileNotFoundError`` at the replace.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".tmp")  # unsafe: process-SHARED temp name
    stage_box["stage"] = "open"
    fh = tmp.open("w", encoding="utf-8")
    stage_box["stage"] = "write"
    with fh:
        fh.write(json.dumps(_PAYLOAD))
    barrier.wait(timeout=60)
    stage_box["stage"] = "replace"
    os.replace(tmp, dest)


def _hammer_worker(writer, dest_str: str, barrier, resq) -> None:
    """Forked child: one write; reports ``(ok, exc_type, exc_stage)`` on the queue."""
    stage_box: dict = {"stage": None}
    try:
        writer(Path(dest_str), barrier, stage_box)
    except BaseException as e:
        resq.put((False, type(e).__name__, stage_box["stage"]))
        raise SystemExit(1) from e
    resq.put((True, None, None))


def _run_hammer(writer, n_workers=8, rounds=5):
    """Hammer ONE shared destination with ``n_workers`` REAL forked processes per round.

    Harness mechanics (the donor shape, ``test_issue2329_r2_fixes.py:604``, plus the
    round-1 A2 hardening): the destination's parent does NOT pre-exist per round; a
    ``multiprocessing.Barrier(n_workers)`` between the WRITE and the PUBLISH forces all
    workers to complete their temp write before any replace runs; each worker returns
    ``(ok, exc_type, exc_stage)`` with ``exc_stage in {open, write, replace}``.

    Scratch lives in a ``tempfile.mkdtemp`` dir (not pytest ``tmp_path`` — concurrent
    pytest sessions prune numbered ``/tmp/pytest-of-*`` roots mid-test on this shared VM).

    Returns one dict per round: ``{"results": [(ok, exc_type, exc_stage)],
    "dest_content": str | None, "residue": [tmp-shaped names left in dest.parent]}``.
    """
    ctx = multiprocessing.get_context("fork")
    base = Path(tempfile.mkdtemp(prefix="atomic-io-hammer-"))
    out = []
    try:
        for round_idx in range(rounds):
            dest = base / f"round{round_idx}" / "manifests" / "out.json"
            barrier = ctx.Barrier(n_workers)
            resq = ctx.Queue()
            procs = [
                ctx.Process(target=_hammer_worker, args=(writer, str(dest), barrier, resq))
                for _ in range(n_workers)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=120)
            for p in procs:  # pragma: no cover - hang guard
                if p.is_alive():
                    p.terminate()
            results = []
            while True:
                try:
                    results.append(resq.get_nowait())
                except queue.Empty:
                    break
            assert len(results) == n_workers, (round_idx, results)
            residue = (
                [f.name for f in dest.parent.iterdir() if ".tmp" in f.name]
                if dest.parent.exists()
                else []
            )
            out.append(
                {
                    "results": results,
                    "dest_content": dest.read_text(encoding="utf-8") if dest.exists() else None,
                    "residue": residue,
                }
            )
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return out


# ── test 1: the safe arm — 8/8 workers succeed EVERY round ──


def test_concurrent_same_destination_all_workers_succeed():
    """Unique pid+uuid temp names => the barrier changes nothing for the safe arm —
    each worker replaces its OWN temp; 8/8 succeed in all 5 rounds, content intact,
    zero temp residue."""
    rounds = _run_hammer(_safe_writer)
    assert len(rounds) == 5
    for round_idx, r in enumerate(rounds):
        oks = [ok for ok, _, _ in r["results"]]
        assert oks == [True] * 8, (round_idx, r["results"])
        assert r["dest_content"] is not None, round_idx
        assert json.loads(r["dest_content"]) == _PAYLOAD, round_idx
        assert r["residue"] == [], (round_idx, r["residue"])


# ── test 2: the unsafe control — the SAME harness fails it deterministically ──


def test_unsafe_derivation_control_fails_deterministically():
    """Harness-sensitivity control on the SAME helper: (a) >=1 worker FAILS every
    round; (b) every failure is ``FileNotFoundError`` with ``exc_stage == "replace"``
    (the race signature: all 8 wrote the same temp; the first ``os.replace`` consumed
    it); (c) >=1 worker SUCCEEDS — the replace-winner. Requirement (c) is what makes a
    control broken at open/write (which fails 8/8) FAIL this test instead of silently
    certifying the harness."""
    rounds = _run_hammer(_unsafe_writer)
    assert len(rounds) == 5
    for round_idx, r in enumerate(rounds):
        failures = [(t, s) for ok, t, s in r["results"] if not ok]
        successes = [ok for ok, _, _ in r["results"] if ok]
        assert failures, (round_idx, r["results"])  # (a) >=1 FAILS
        assert all(t == "FileNotFoundError" and s == "replace" for t, s in failures), (
            round_idx,
            failures,
        )  # (b) the race signature
        assert successes, (round_idx, r["results"])  # (c) >=1 SUCCEEDS (the winner)
        assert r["dest_content"] is not None, round_idx
        assert json.loads(r["dest_content"]) == _PAYLOAD, round_idx


# ── test 3: unlink-on-failure leaves no orphan residue ──


def test_unlink_on_failure_no_orphan_residue(tmp_path):
    """A raising write body leaves zero ``*.tmp*`` residue and no destination."""
    dest = tmp_path / "out.json"
    with (
        pytest.raises(RuntimeError, match="simulated write failure"),
        atomic_replace(dest) as tmp,
    ):
        tmp.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated write failure")
    assert [f.name for f in tmp_path.iterdir() if ".tmp" in f.name] == []
    assert not dest.exists()
    # Success path leaves no residue either.
    with atomic_replace(dest) as tmp:
        tmp.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert json.loads(dest.read_text(encoding="utf-8")) == {"x": 1}
    assert [f.name for f in tmp_path.iterdir() if ".tmp" in f.name] == []


# ── test 4: a cleanup failure never masks the original exception (B1 contract) ──


@pytest.mark.parametrize("caller_logger_name", [None, "some.caller"])
def test_cleanup_failure_does_not_mask_original_exception(
    tmp_path, monkeypatch, caplog, caller_logger_name
):
    """When the replace fails AND the best-effort temp unlink ALSO fails, the ORIGINAL
    exception escapes unchanged with its traceback intact; the cleanup error is only
    logged. Two sub-cases pin the B1 ``logger=`` parameter contract: default path
    (warning under ``explore_persona_space.atomic_io``) and an explicit
    ``logger=logging.getLogger("some.caller")`` (warning under ``some.caller``)."""

    def _boom_replace(src, dst):
        raise OSError("simulated replace failure")

    def _boom_unlink(self, missing_ok=False):  # mirrors Path.unlink's signature
        raise PermissionError("simulated unlink failure")

    expected_logger = caller_logger_name or "explore_persona_space.atomic_io"
    kwargs = {} if caller_logger_name is None else {"logger": logging.getLogger(caller_logger_name)}
    with monkeypatch.context() as m:
        m.setattr(atomic_io.os, "replace", _boom_replace)
        m.setattr(Path, "unlink", _boom_unlink)
        with (
            caplog.at_level(logging.WARNING, logger=expected_logger),
            pytest.raises(OSError, match="simulated replace failure") as excinfo,
            atomic_replace(tmp_path / "a.json", **kwargs) as tmp,
        ):
            tmp.write_text("{}", encoding="utf-8")
    # The ORIGINAL exception type escapes — not the cleanup's PermissionError
    # (PermissionError IS an OSError subclass, so pin the exact type too).
    assert type(excinfo.value) is OSError, type(excinfo.value)
    assert any(t.name == "_boom_replace" for t in excinfo.traceback), [
        t.name for t in excinfo.traceback
    ]
    cleanup_warnings = [
        rec
        for rec in caplog.records
        if rec.name == expected_logger and "cleanup unlink of" in rec.getMessage()
    ]
    assert len(cleanup_warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "simulated unlink failure" in cleanup_warnings[0].getMessage()


# ── test 5: temp names are process-unique ──


def test_tmp_paths_are_process_unique(tmp_path):
    """Two derivations in the SAME pid differ (uuid fragment per call); the pid is
    embedded in the name; names keep the ``<dest-name>.<pid>.<uuid8>.tmp`` shape."""
    dest = tmp_path / "x.json"
    names = []
    for _ in range(2):
        with atomic_replace(dest) as tmp:
            names.append(tmp.name)
            tmp.write_text("{}", encoding="utf-8")
    assert names[0] != names[1], names
    for n in names:
        assert n.startswith(dest.name + f".{os.getpid()}."), n
        assert n.endswith(".tmp"), n


# ── test 6: all 7 typed wrappers, against real torch/numpy ──


def test_wrappers_roundtrip_real_torch_numpy(tmp_path):
    """Executes every wrapper (lazy torch/numpy imports included) with round-trip
    equality; ``savez_atomic`` additionally with ``compressed=True`` — exercising the
    numpy handle-form trap defense (a path-form ``np.savez(tmp)`` would write
    ``<tmp>.npz`` and the replace would die ``FileNotFoundError``)."""
    import numpy as np
    import torch

    write_json_atomic(tmp_path / "a.json", {"b": 2, "a": 1}, sort_keys=True)
    assert (tmp_path / "a.json").read_text(encoding="utf-8") == json.dumps(
        {"a": 1, "b": 2}, indent=2, sort_keys=True
    )

    rows = [{"i": 0}, {"i": 1}]
    write_jsonl_atomic(tmp_path / "a.jsonl", rows)
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln) for ln in lines] == rows

    write_text_atomic(tmp_path / "a.txt", "héllo\n")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "héllo\n"

    write_bytes_atomic(tmp_path / "a.bin", b"\x00\x01\x02")
    assert (tmp_path / "a.bin").read_bytes() == b"\x00\x01\x02"

    t = torch.tensor([1.0, 2.0])
    save_pt_atomic(tmp_path / "a.pt", t)
    assert torch.equal(torch.load(tmp_path / "a.pt", weights_only=True), t)

    arr = np.arange(6).reshape(2, 3)
    savez_atomic(tmp_path / "a.npz", x=arr)
    with np.load(tmp_path / "a.npz") as z:
        np.testing.assert_array_equal(z["x"], arr)
    savez_atomic(tmp_path / "b.npz", compressed=True, x=arr)
    with np.load(tmp_path / "b.npz") as z:
        np.testing.assert_array_equal(z["x"], arr)

    save_npy_atomic(tmp_path / "a.npy", arr)
    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), arr)

    assert [f.name for f in tmp_path.iterdir() if ".tmp" in f.name] == []


# ── test 7: the utils.save_json_atomic re-point (the step-6 gate) ──


def test_save_json_atomic_repoint(tmp_path, monkeypatch):
    """(a) Success path: output bytes identical to the pre-migration form for a fixture
    exercising ``default=str`` (a datetime key) + ``indent=2``. (b) Failure posture (the
    error-contract FIX made testable): a serialization failure propagates the ORIGINAL
    exception even when temp cleanup is made to fail — the pre-migration body's bare
    ``os.remove`` inside ``except Exception:`` would have displaced it."""
    from explore_persona_space.utils import save_json_atomic

    fixture = {"when": datetime.datetime(2026, 8, 23, 12, 0, 0), "n": 1}
    out = tmp_path / "out.json"
    save_json_atomic(out, fixture)
    # Pre-migration form: json.dump(data, tmp, indent=2, default=str) into a utf-8
    # text-mode temp — byte-identical to json.dumps with the same kwargs.
    assert out.read_bytes() == json.dumps(fixture, indent=2, default=str).encode("utf-8")

    class _Unserializable:
        def __repr__(self):
            raise ValueError("simulated serialization failure")

    def _boom_unlink(self, missing_ok=False):
        raise PermissionError("simulated unlink failure")

    def _boom_remove(p):
        raise PermissionError("simulated remove failure")

    # BOTH cleanup channels are made to fail: ``Path.unlink`` (the re-pointed body's,
    # via ``atomic_replace``) AND ``os.remove`` (the pre-migration body's bare handler),
    # so this test is RED pre-fix (the old handler's remove failure displaced the
    # ValueError) and GREEN post-fix.
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", _boom_unlink)
        m.setattr(os, "remove", _boom_remove)
        with pytest.raises(ValueError, match="simulated serialization failure") as excinfo:
            save_json_atomic(tmp_path / "fail.json", {"bad": _Unserializable()})
    assert type(excinfo.value) is ValueError, type(excinfo.value)
    assert not (tmp_path / "fail.json").exists()
