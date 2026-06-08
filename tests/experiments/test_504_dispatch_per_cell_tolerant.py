# em-dash + Greek ΔG intentional
"""Task #504 round-4 — _schedule_cell_pool per-cell tolerance regression tests.

Pins the contract for the round-4 BLOCKER #2 fix (concern_id
``phase0-sweep-not-per-cell-tolerant``): the lr-ladder smoke must run EVERY
cell even when one returns rc != 0, so the post-smoke picker sees the whole
trajectory set. Phase 1 (default ``per_cell_tolerant=False``) stays strict.

CPU-only, sub-second. Monkeypatches ``subprocess.Popen`` so the dispatcher's
launch path returns fake processes with controllable exit codes; no GPU /
training / HF / network is exercised. Verifies:

  1. ``per_cell_tolerant=True`` records the failure manifest + continues to the
     remaining cells; the returned ``results`` list contains the failed cell
     with ``status="failed"`` and the successful cells with ``status="done"``.
  2. ``per_cell_tolerant=False`` (default) raises ``RuntimeError`` on the first
     cell failure and the remaining cells never launch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.dispatch_neg_geometry_504 as dispatch


class _FakePopen:
    """Minimal subprocess.Popen stand-in.

    ``rc`` is the exit code returned by ``.poll()`` on the FIRST call; later
    calls keep returning ``rc``. The dispatcher's pool loop polls until !=
    None and reads the value once.
    """

    def __init__(self, *, cell: str, seed: int, rc: int) -> None:
        self.cell = cell
        self.seed = seed
        self._rc = rc
        self._terminated = False

    def poll(self) -> int:
        return self._rc

    def terminate(self) -> None:  # pragma: no cover - phase-1 strict-mode only
        self._terminated = True


def _install_popen_factory(
    monkeypatch: pytest.MonkeyPatch, cell_rcs: dict[str, int]
) -> list[tuple[str, int]]:
    """Patch ``subprocess.Popen`` in the dispatcher module to return _FakePopen.

    ``cell_rcs`` maps cell-slug → rc. Returns the call-order log, populated by
    the side-effect of each launch.
    """
    launched: list[tuple[str, int]] = []

    def _factory(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        # The dispatcher's launch path encodes --cell <slug> --seed <S> on the cmd.
        cell = cmd[cmd.index("--cell") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
        launched.append((cell, seed))
        rc = cell_rcs.get(cell, 0)
        return _FakePopen(cell=cell, seed=seed, rc=rc)

    # The pool calls ``open(cell_log, "w")`` for stdout; redirect to /dev/null.
    monkeypatch.setattr(dispatch.subprocess, "Popen", _factory)
    monkeypatch.setattr(dispatch.time, "sleep", lambda _s: None)
    return launched


def _common_kwargs(tmp_path: Path) -> dict[str, Any]:
    """Build the minimum kwargs the schedule pool needs to run.

    All paths are tmp_path-rooted; the launch path tries to ``mkdir`` /
    ``open`` the per-cell log file, which works as long as parents exist.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        seeds=[42],
        n_gpus=1,
        max_parallel=1,
        slab_root=tmp_path / "slab",
        runs_root=tmp_path / "runs",
        log_dir=log_dir,
        bank_path=tmp_path / "bank.json",
        centroids_dir=tmp_path / "centroids",
        arm_to_n_json=tmp_path / "arm_to_n.json",
        r_train_path=tmp_path / "r_train.json",
        r_eval_path=tmp_path / "r_eval.json",
        chosen_rank=8,
        chosen_alpha=32,
        chosen_frac=None,
        smoke=False,
        no_kl=False,
        report_to="none",
        resume=False,
        max_new_tokens_eval=1024,
        max_model_len_eval=2048,
        hf_path_suffix="",
        label_prefix="issue-504-test",
    )


def test_per_cell_tolerant_records_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3-cell ladder with cell #1 returning rc=1: cells #2 + #3 still run.

    Mirrors the round-3 crash config: at lr=1e-5 the cell exited mid-ladder,
    and pre-round-4 the dispatcher raised before cells #2 + #3 could launch.
    With ``per_cell_tolerant=True`` the failure is recorded in
    ``<label_prefix>-cell-failures.json`` and the remaining cells run.
    """
    launched = _install_popen_factory(
        monkeypatch,
        cell_rcs={"c504v2_smoke_lr1e5": 1, "c504v2_smoke_lr3e5": 0, "c504v2_smoke_lr1e4": 0},
    )

    results = dispatch._schedule_cell_pool(
        cells=["c504v2_smoke_lr1e5", "c504v2_smoke_lr3e5", "c504v2_smoke_lr1e4"],
        per_cell_tolerant=True,
        **_common_kwargs(tmp_path),
    )

    # All three cells were launched (the second + third did NOT get pre-empted).
    assert sorted(launched) == sorted(
        [
            ("c504v2_smoke_lr1e5", 42),
            ("c504v2_smoke_lr3e5", 42),
            ("c504v2_smoke_lr1e4", 42),
        ]
    )
    # Results: one failure row + two done rows.
    statuses = {(r["cell"], r["status"]) for r in results}
    assert ("c504v2_smoke_lr1e5", "failed") in statuses
    assert ("c504v2_smoke_lr3e5", "done") in statuses
    assert ("c504v2_smoke_lr1e4", "done") in statuses
    # The failed row carries the rc + log path.
    failed_row = next(r for r in results if r["status"] == "failed")
    assert failed_row["returncode"] == 1
    assert "log_path" in failed_row
    # The manifest file was written.
    manifest_path = tmp_path / "logs" / "issue-504-test-cell-failures.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert isinstance(manifest, list)
    assert len(manifest) == 1
    assert manifest[0]["cell"] == "c504v2_smoke_lr1e5"
    assert manifest[0]["returncode"] == 1


def test_default_strict_mode_raises_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 1 default behavior: the first rc!=0 aborts the sweep.

    A 3-cell pool where cell #1 returns rc=1 must raise ``RuntimeError`` and
    never launch cells #2 / #3 (sequential max_parallel=1). With
    max_parallel=1 the queue order is preserved.
    """
    launched = _install_popen_factory(
        monkeypatch,
        cell_rcs={"c504v2_arm_a": 1, "c504v2_arm_b": 0, "c504v2_arm_c": 0},
    )

    with pytest.raises(RuntimeError, match="exited rc=1"):
        dispatch._schedule_cell_pool(
            cells=["c504v2_arm_a", "c504v2_arm_b", "c504v2_arm_c"],
            # per_cell_tolerant defaults to False
            **_common_kwargs(tmp_path),
        )
    # Only the first cell launched before the RuntimeError surfaced.
    assert launched == [("c504v2_arm_a", 42)]


def test_per_cell_tolerant_appends_to_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second call adds a row to the existing failures manifest rather than overwriting.

    Phase 0 primary + Phase 0 fallback share a log_dir; if both record
    failures they must accumulate so the picker sees the full set.
    """
    _install_popen_factory(monkeypatch, cell_rcs={"c504v2_smoke_lrA": 1})

    dispatch._schedule_cell_pool(
        cells=["c504v2_smoke_lrA"],
        per_cell_tolerant=True,
        **_common_kwargs(tmp_path),
    )

    # Second call with a fresh failed cell under the SAME label_prefix.
    _install_popen_factory(monkeypatch, cell_rcs={"c504v2_smoke_lrB": 2})

    dispatch._schedule_cell_pool(
        cells=["c504v2_smoke_lrB"],
        per_cell_tolerant=True,
        **_common_kwargs(tmp_path),
    )

    manifest_path = tmp_path / "logs" / "issue-504-test-cell-failures.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 2
    assert {row["cell"] for row in manifest} == {"c504v2_smoke_lrA", "c504v2_smoke_lrB"}
