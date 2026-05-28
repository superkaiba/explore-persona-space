"""CPU smoke test for the CVD-aware orphan-PID check in eval_issue396_logprob.py.

The post-vLLM-teardown sanity check (``_check_orphan_pids_on_visible_gpus``)
must scope its ``nvidia-smi --query-compute-apps`` filter to the GPUs
visible to the current process via ``CUDA_VISIBLE_DEVICES``. Without
this, parallel eval subprocesses on a multi-GPU pod see each other's
legitimate vLLM-worker PIDs as "orphans" and abort the run with a
false-positive ``RuntimeError`` (task #396 incident, 2026-05-27: 3 of
4 Wave-1 subprocesses died here despite each one's own GPU being
clean).

These tests mock ``subprocess.check_output`` to return synthetic
``nvidia-smi`` output and exercise the three branches:

1. CVD restricted to GPU 0 — an orphan PID on GPU 0 fires RuntimeError.
2. CVD restricted to GPU 0 — a PID on GPU 1 (NOT in CVD) is correctly
   ignored; no RuntimeError.
3. CVD unset — fall back to the legacy ``--query-compute-apps=pid``
   path that aborts on ANY non-self PID (correct on single-GPU pods).

The tests do NOT touch GPUs, HF, or the network.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_eval_module():
    """Load scripts/eval_issue396_logprob.py without running its main()."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "eval_issue396_logprob", SCRIPTS_DIR / "eval_issue396_logprob.py"
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _build_smi_mock(uuid_map: dict[int, str], pid_uuid_pairs: list[tuple[str, str]]):
    """Return a fake ``subprocess.check_output`` that emits realistic nvidia-smi shapes.

    ``uuid_map``: physical GPU index -> UUID (the index-uuid table).
    ``pid_uuid_pairs``: list of (pid, gpu_uuid) holding-pairs, mirroring
    ``--query-compute-apps=pid,gpu_uuid`` output.
    """

    def _fake_check_output(cmd, *args, **kwargs):
        joined = " ".join(cmd)
        if "--query-gpu=index,uuid" in joined:
            return "\n".join(f"{idx}, {uuid}" for idx, uuid in sorted(uuid_map.items())) + "\n"
        if "--query-compute-apps=pid,gpu_uuid" in joined:
            return "\n".join(f"{pid}, {uuid}" for pid, uuid in pid_uuid_pairs) + "\n"
        if "--query-compute-apps=pid" in joined:
            return "\n".join(pid for pid, _ in pid_uuid_pairs) + "\n"
        raise AssertionError(f"unexpected nvidia-smi invocation: {cmd!r}")

    return _fake_check_output


# ── Case 1: CVD=0, orphan on our GPU → RuntimeError ──────────────────────────


def test_orphan_on_cvd_visible_gpu_raises(monkeypatch):
    """A PID holding a CVD-visible GPU (not ourselves) must trip the safety check."""
    mod = _import_eval_module()

    uuid_map = {0: "GPU-aaa", 1: "GPU-bbb", 2: "GPU-ccc", 3: "GPU-ddd"}
    our_pid = str(os.getpid())
    other_pid = "999999"

    # CVD=0 → visible UUID set = {GPU-aaa}.
    # Holding pairs: our_pid on GPU-aaa (us, OK), other_pid on GPU-aaa (orphan, BAD).
    pid_pairs = [(our_pid, "GPU-aaa"), (other_pid, "GPU-aaa")]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("subprocess.check_output", _build_smi_mock(uuid_map, pid_pairs))

    with pytest.raises(RuntimeError) as excinfo:
        mod._check_orphan_pids_on_visible_gpus()
    msg = str(excinfo.value)
    assert other_pid in msg, "orphan PID must appear in the error message"
    assert "CVD-visible GPU" in msg, "error message must explain the CVD scoping"


# ── Case 2: CVD=0, peer PID on a DIFFERENT GPU → ignored, no RuntimeError ───


def test_peer_pid_on_other_gpu_is_ignored(monkeypatch):
    """A peer subprocess holding GPU 1 must NOT fire the check when CVD=0.

    Regression guard for the canonical task #396 incident shape:
    4 subprocesses run in parallel on a 4-GPU pod, each pinned via CVD;
    the post-teardown check on subprocess A (CVD=0) must not abort
    because subprocess B (CVD=1) is legitimately still holding GPU 1.
    """
    mod = _import_eval_module()

    uuid_map = {0: "GPU-aaa", 1: "GPU-bbb", 2: "GPU-ccc", 3: "GPU-ddd"}
    our_pid = str(os.getpid())
    peer_pids = ["111111", "222222", "333333"]

    # CVD=0 → visible UUID set = {GPU-aaa}. Our PID holds GPU-aaa (cleanly).
    # Peer PIDs hold GPUs 1/2/3 — NOT visible to us; check must ignore them.
    pid_pairs = [
        (our_pid, "GPU-aaa"),
        (peer_pids[0], "GPU-bbb"),
        (peer_pids[1], "GPU-ccc"),
        (peer_pids[2], "GPU-ddd"),
    ]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("subprocess.check_output", _build_smi_mock(uuid_map, pid_pairs))

    # Must not raise.
    mod._check_orphan_pids_on_visible_gpus()


# ── Case 3: CVD unset → fall back to legacy pid-only path ───────────────────


def test_cvd_unset_falls_back_to_legacy_path(monkeypatch):
    """With CVD unset, ANY non-self PID is an orphan (correct on single-GPU pods)."""
    mod = _import_eval_module()

    uuid_map: dict[int, str] = {0: "GPU-aaa"}
    our_pid = str(os.getpid())
    other_pid = "888888"

    # Legacy path: ``--query-compute-apps=pid`` returns the pid list.
    pid_pairs = [(our_pid, "GPU-aaa"), (other_pid, "GPU-aaa")]

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("subprocess.check_output", _build_smi_mock(uuid_map, pid_pairs))

    with pytest.raises(RuntimeError) as excinfo:
        mod._check_orphan_pids_on_visible_gpus()
    msg = str(excinfo.value)
    assert other_pid in msg, "orphan PID must appear in the legacy-path error message too"


def test_cvd_unset_no_other_pids_passes(monkeypatch):
    """Legacy path: only-self PID on the box passes cleanly."""
    mod = _import_eval_module()

    uuid_map: dict[int, str] = {0: "GPU-aaa"}
    our_pid = str(os.getpid())

    pid_pairs: list[tuple[str, str]] = [(our_pid, "GPU-aaa")]

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("subprocess.check_output", _build_smi_mock(uuid_map, pid_pairs))

    # Must not raise.
    mod._check_orphan_pids_on_visible_gpus()


# ── Bonus: ``CUDA_VISIBLE_DEVICES`` with multiple indices ────────────────────


def test_cvd_multi_gpu_visible_set(monkeypatch):
    """CVD=0,2 → visible UUIDs are GPUs 0 and 2; PID on GPU 1 is ignored."""
    mod = _import_eval_module()

    uuid_map = {0: "GPU-aaa", 1: "GPU-bbb", 2: "GPU-ccc", 3: "GPU-ddd"}
    our_pid = str(os.getpid())

    # Peer on GPU 1 (not visible) — must be ignored.
    pid_pairs: list[tuple[str, Any]] = [(our_pid, "GPU-aaa"), ("777777", "GPU-bbb")]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    monkeypatch.setattr("subprocess.check_output", _build_smi_mock(uuid_map, pid_pairs))

    # Must not raise.
    mod._check_orphan_pids_on_visible_gpus()

    # Now add a peer on GPU 2 (which IS visible) — must raise.
    pid_pairs_with_visible_orphan = [*pid_pairs, ("555555", "GPU-ccc")]
    monkeypatch.setattr(
        "subprocess.check_output",
        _build_smi_mock(uuid_map, pid_pairs_with_visible_orphan),
    )
    with pytest.raises(RuntimeError) as excinfo:
        mod._check_orphan_pids_on_visible_gpus()
    assert "555555" in str(excinfo.value)
