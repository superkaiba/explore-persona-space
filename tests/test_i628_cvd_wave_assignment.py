"""CVD-clobber regression smoke for the #628 wave launcher (sibling of
tests/test_cvd_wave_assignment_smoke.py, per the #523/#545 gotcha).

``i628_dispatch._run_wave`` must pin a DISTINCT physical GPU id into each
worker subprocess's LAUNCHER env (``CUDA_VISIBLE_DEVICES=<gpu>``) — the
in-process ``gpu_id`` clobber alone is silently defeated by import-time
cuInit — and shard cells via ``--worker-shard k/n``. No GPU needed: Popen is
stubbed and the GPU pool monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class _FakeProc:
    def wait(self):
        return 0


def _args(**over):
    base = dict(
        phase="1",
        arms=["rig_O_sep_deadneg"],
        train_cids=None,
        seeds=(42, 1042),
        smoke=False,
        dry_run=True,
        skip_upload=True,
        enforce_gate=False,
        workers=0,
        worker_shard=None,
        step=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_wave_pins_distinct_cvd_per_worker(monkeypatch):
    import i628_dispatch as d

    launches = []

    def fake_popen(cmd, cwd=None, env=None):
        launches.append({"cmd": cmd, "env": env})
        return _FakeProc()

    monkeypatch.setattr(d.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(d, "_gpu_pool", lambda: ["0", "1", "2", "3"])
    d._run_wave(_args(), "1", "train", n_items=32)

    assert len(launches) == 4
    cvds = [launch["env"]["CUDA_VISIBLE_DEVICES"] for launch in launches]
    # The #523 regression signature is ALL workers on GPU 0 — every worker
    # must get a DISTINCT physical id from the pool.
    assert sorted(cvds) == ["0", "1", "2", "3"], cvds
    for k, launch in enumerate(launches):
        cmd = launch["cmd"]
        assert cmd[cmd.index("--worker-shard") + 1] == f"{k}/4"
        assert cmd[cmd.index("--phase") + 1] == "1"
        assert cmd[cmd.index("--step") + 1] == "train"


def test_wave_respects_parent_cvd_narrowing(monkeypatch):
    """A parent narrowed to GPUs 2,3 must hand workers physical ids 2 and 3
    (never a re-zeroed '0' that would remap to physical GPU 0)."""
    import i628_dispatch as d

    launches = []
    monkeypatch.setattr(
        d.subprocess,
        "Popen",
        lambda cmd, cwd=None, env=None: (launches.append(env), _FakeProc())[1],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    d._run_wave(_args(), "2", "cells", n_items=8)
    assert sorted(e["CUDA_VISIBLE_DEVICES"] for e in launches) == ["2", "3"]


def test_wave_never_overspawns_tiny_cell_counts(monkeypatch):
    import i628_dispatch as d

    launches = []
    monkeypatch.setattr(
        d.subprocess,
        "Popen",
        lambda cmd, cwd=None, env=None: (launches.append(cmd), _FakeProc())[1],
    )
    monkeypatch.setattr(d, "_gpu_pool", lambda: ["0", "1", "2", "3"])
    d._run_wave(_args(), "1", "train", n_items=1)  # the 1-cell smoke shape
    assert len(launches) == 1
    assert launches[0][launches[0].index("--worker-shard") + 1] == "0/1"
