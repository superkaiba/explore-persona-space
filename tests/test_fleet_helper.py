"""Unit tests for the #676 wave-parallel fleet dispatcher helper.

All CPU-only (no GPU, no API). The wave-launch tests use a stub ``build_cmd``
returning a ``CellCmd`` whose ``argv`` is an ``echo`` / tiny-python no-op, plus a
captured launcher env, so ``run_parallel_with_log`` actually fan-out-launches real
(trivial) subprocesses and we assert the per-cell CVD pin lands in the launcher
environment (the gotchas.md cuInit-freeze guard, mirroring
``tests/test_cvd_wave_assignment_smoke.py`` assertion 2).

Test 8 drives the REAL ``issue664_dispatch.main()`` single-GPU backward-compat
path (``--cells 1 --smoke`` with NO ``--n-gpus``) with ``WaveDispatcher.run``
monkey-patched to a capture-only stub, asserting the dispatcher enqueues exactly
one cell per GPU-bound phase on gpu 0 with ``CUDA_VISIBLE_DEVICES=0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from explore_persona_space.orchestrate.fleet import (
    CellCmd,
    DuplicateCellError,
    FleetResult,
    WaveDispatcher,
    WaveFailedError,
    assign_gpu_ids,
    run_parallel_with_log,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


# ── stub cell + build_cmd helpers ─────────────────────────────────────────────


def _echo_cmd(key: str, gpu: int, log_dir: Path, *, rc: int = 0, drop_cvd: bool = False) -> CellCmd:
    """A trivial one-cell launch spec: a python no-op exiting ``rc``, CVD pinned.

    Each cell's argv writes the env CVD it actually SAW to ``log_dir/<key>.cvd`` so
    a test can assert the per-cell launcher-env pin took (not just that the dataclass
    field was set). ``drop_cvd`` omits the CVD pin to exercise the loud pre-launch
    assert.
    """
    env = {} if drop_cvd else {"CUDA_VISIBLE_DEVICES": str(gpu)}
    capture = log_dir / f"{key}.cvd"
    script = (
        "import os,sys,pathlib;"
        f"pathlib.Path({str(capture)!r}).write_text(os.environ.get('CUDA_VISIBLE_DEVICES','UNSET'));"
        f"sys.exit({rc})"
    )
    return CellCmd(
        cell_key=key,
        argv=[sys.executable, "-c", script],
        env=env,
        log_path=log_dir / f"{key}.log",
        gpu_id=gpu,
    )


# ── 1. assign_gpu_ids round-robin ─────────────────────────────────────────────


def test_assign_gpu_ids_round_robin():
    # 12 cells over 4 GPUs -> the #651:677 per-wave densification pattern.
    assert assign_gpu_ids(12, 4) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    # fewer cells than GPUs -> only the needed GPUs.
    assert assign_gpu_ids(3, 4) == [0, 1, 2]
    # single-GPU collapse: every cell on gpu 0 (the unchanged serial path).
    assert assign_gpu_ids(5, 1) == [0, 0, 0, 0, 0]
    # n_gpus<=0 is treated as 1 (defensive).
    assert assign_gpu_ids(3, 0) == [0, 0, 0]


# ── 2. disjoint sharding raises on a duplicate cell_key ───────────────────────


def test_disjoint_sharding_raises_on_duplicate(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cells = ["a", "b", "a"]  # 'a' collides
    disp = WaveDispatcher(
        n_gpus=2,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=lambda c, g: _echo_cmd(c, g, log_dir),
    )
    with pytest.raises(DuplicateCellError) as ei:
        disp.run(cells)
    assert "a" in ei.value.colliding_keys

    # positive case: distinct keys run clean (n_gpus=1 so deterministic).
    disp_ok = WaveDispatcher(
        n_gpus=1,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=lambda c, g: _echo_cmd(c, g, log_dir),
    )
    res = disp_ok.run(["x", "y"])
    assert sorted(res.ran) == ["x", "y"]
    assert res.failures == []


# ── 3. disjoint output paths for the REAL #664 grid ───────────────────────────


def test_disjoint_output_paths_for_664_cells():
    sys.path.insert(0, str(SCRIPTS))
    import issue664_common as C

    grid = C.realized_grid()
    assert len(grid) > 0
    # every cell's idempotency key is unique across the whole fleet.
    keys = [c.eval_key for c in grid]
    assert len(set(keys)) == len(keys), "realized_grid eval_keys are not unique"
    # the WaveDispatcher whole-fleet uniqueness assert accepts the real grid.
    disp = WaveDispatcher(
        n_gpus=4,
        cell_key=lambda c: c.eval_key,
        is_done=lambda c: True,  # mark all done -> no launch, just the uniqueness assert
        build_cmd=lambda c, g: None,  # never called (all skipped)
    )
    res = disp.run(grid)
    assert sorted(res.skipped) == sorted(keys)
    assert res.ran == []
    # every derived output path is key-distinct (distinct keys -> distinct paths).
    adapter_dirs = {c.eval_key for c in grid}
    subfolders = {c.hf_adapter_subfolder for c in grid}
    assert len(adapter_dirs) == len(grid)
    assert len(subfolders) == len(grid)


# ── 4. idempotent resume-skip ─────────────────────────────────────────────────


def test_idempotent_resume_skip(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cells = ["c0", "c1", "c2", "c3", "c4", "c5"]
    done = {"c0", "c2", "c4"}
    built: list[str] = []

    def _build(c, g):
        built.append(c)
        return _echo_cmd(c, g, log_dir)

    disp = WaveDispatcher(
        n_gpus=2,
        cell_key=lambda c: c,
        is_done=lambda c: c in done,
        build_cmd=_build,
    )
    res = disp.run(cells)
    assert sorted(res.skipped) == ["c0", "c2", "c4"]
    assert sorted(res.ran) == ["c1", "c3", "c5"]
    # build_cmd invoked ONLY for the un-done cells.
    assert sorted(built) == ["c1", "c3", "c5"]


# ── 5. CVD-present pre-launch assert ───────────────────────────────────────────


def test_cvd_present_assert(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # A build_cmd that omits CUDA_VISIBLE_DEVICES -> loud pre-launch failure.
    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        run_parallel_with_log([_echo_cmd("bad", 1, log_dir, drop_cvd=True)])

    # Correct pins -> the captured launcher env CVD matches the assigned gpu_id
    # per cell (the gotchas.md launcher-env pin, mirroring the cvd-wave smoke test).
    cmds = [_echo_cmd("g0", 0, log_dir), _echo_cmd("g1", 1, log_dir), _echo_cmd("g2", 2, log_dir)]
    rcs = run_parallel_with_log(cmds)
    assert rcs == [0, 0, 0]
    assert (log_dir / "g0.cvd").read_text() == "0"
    assert (log_dir / "g1.cvd").read_text() == "1"
    assert (log_dir / "g2.cvd").read_text() == "2"


# ── 6. smoke == single-GPU sweep-of-one (PASS_UNIFIED) ────────────────────────


def test_smoke_single_gpu_equivalence(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    built: list[tuple[str, int]] = []

    def _build(c, g):
        built.append((c, g))
        return _echo_cmd(c, g, log_dir)

    disp = WaveDispatcher(
        n_gpus=1,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=_build,
    )
    res = disp.run(["only"])
    assert isinstance(res, FleetResult)
    assert res.ran == ["only"]
    assert res.wave_count == 1
    # exactly one cell, launched on gpu 0 (no in-process-vs-subprocess divergence).
    assert built == [("only", 0)]
    assert (log_dir / "only.cvd").read_text() == "0"


# ── 7. WaveFailedError lists the failing (rc, cell_key) ───────────────────────


def test_wave_failed_error_lists_bad_cells(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    def _build(c, g):
        # 'boom' exits non-zero; the others succeed.
        return _echo_cmd(c, g, log_dir, rc=7 if c == "boom" else 0)

    disp = WaveDispatcher(
        n_gpus=4,
        cell_key=lambda c: c,
        is_done=lambda c: False,
        build_cmd=_build,
    )
    with pytest.raises(WaveFailedError) as ei:
        disp.run(["ok1", "boom", "ok2"])
    assert (7, "boom") in ei.value.failures
    assert all(key == "boom" for _rc, key in ei.value.failures)


# ── 8. REAL issue664_dispatch.main() single-GPU backward-compat ───────────────


def test_issue664_main_smoke_backcompat_no_n_gpus(monkeypatch, tmp_path):
    """Drive the REAL dispatcher main() for ``--cells 1 --smoke`` with NO --n-gpus
    (the single-GPU backward-compat path, acceptance #5). No subprocess executes:
    WaveDispatcher.run is monkey-patched to a capture-only stub recording the cells
    + the CellCmds the dispatcher's build_cmd produces. Asserts the new argparse
    branch + argv-building + CVD-env layer preserve today's --gpu-id 0 behavior."""
    sys.path.insert(0, str(SCRIPTS))
    import issue664_common as C
    import issue664_dispatch as D

    # Capture every WaveDispatcher.run invocation: the cells + the CellCmd each
    # build_cmd produces (gpu_id assigned via assign_gpu_ids over the wave) + the
    # is_done predicate (so we can assert which sentinel paths it probes).
    captured: list[dict] = []

    def _capture_run(self, cells, *, cwd=None):
        gpu_ids = assign_gpu_ids(len(cells), self.n_gpus)
        cmds = [self.build_cmd(c, g) for c, g in zip(cells, gpu_ids, strict=True)]
        captured.append(
            {
                "n_gpus": self.n_gpus,
                "cells": list(cells),
                "cmds": cmds,
                "is_done": self.is_done,
            }
        )
        return FleetResult(ran=[c.eval_key for c in cells], skipped=[], failures=[], wave_count=1)

    monkeypatch.setattr(D.WaveDispatcher, "run", _capture_run)
    # Neutralize every non-target step so only the P2.1/P2.2 wave construction runs.
    monkeypatch.setattr(D, "phase0", lambda args: None)
    monkeypatch.setattr(D, "_require_credentials", lambda: None)
    monkeypatch.setattr(D, "_drop_filtered", lambda cells: cells)
    monkeypatch.setattr(D, "_write_manifest", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "_marker_readability_assert", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "upload_artifacts", lambda cells, *, smoke: None)
    monkeypatch.setattr(D, "write_sentinel", lambda *a, **k: tmp_path / "sentinel.json")
    monkeypatch.setattr(D, "_wandb_entity", lambda: None)
    monkeypatch.setattr(D, "_dropped_cell_keys", set)

    # --cells 1 --smoke, NO --n-gpus (default 1). --phase all so both loops run.
    monkeypatch.setattr(
        sys, "argv", ["issue664_dispatch.py", "--phase", "all", "--cells", "1", "--smoke"]
    )
    rc = D.main()
    assert rc == 0

    # Two WaveDispatcher.run invocations: P2.1 train + P2.2 extract+eval.
    assert len(captured) == 2, f"expected 2 wave runs (train + extract+eval), got {len(captured)}"
    train_cap, extract_cap = captured

    for cap in (train_cap, extract_cap):
        # (default n_gpus) backward-compat: single-GPU path.
        assert cap["n_gpus"] == 1
        # (a) exactly ONE cell enqueued per phase.
        assert len(cap["cells"]) == 1
        assert len(cap["cmds"]) == 1
        cmd = cap["cmds"][0]
        # (b) gpu_id == 0 (preserves the --gpu-id 0 default).
        assert cmd.gpu_id == 0
        # (c) CUDA_VISIBLE_DEVICES == "0" in the launcher env.
        assert cmd.env["CUDA_VISIBLE_DEVICES"] == "0"
        # (d) --smoke threaded into the subprocess argv.
        assert "--smoke" in cmd.argv
        # the subprocess re-invokes THIS dispatcher in one-cell mode.
        assert str(SCRIPTS / "issue664_dispatch.py") in [str(a) for a in cmd.argv]

    # the two phases use DISTINCT one-cell mode flags.
    assert "--train-one-cell" in train_cap["cmds"][0].argv
    assert "--extract-eval-one-cell" in extract_cap["cmds"][0].argv

    # (e) is_done keys on the per-cell sentinel paths the in-process code uses.
    smoke_cell = train_cap["cells"][0]
    # train skip -> adapter_model.safetensors under the eval_key (+_smoke) dir.
    expected_adapter = (
        D.ADAPTER_OUT / (smoke_cell.eval_key + "_smoke") / "adapter_model.safetensors"
    )
    assert not expected_adapter.exists()  # clean test tree
    assert train_cap["is_done"](smoke_cell) is False
    # extract+eval skip -> store tensors.pt AND a non-empty eval registry dir.
    expected_store = C.STORE_ROOT / (smoke_cell.eval_key + "_smoke") / "tensors.pt"
    assert not expected_store.exists()
    assert extract_cap["is_done"](smoke_cell) is False
