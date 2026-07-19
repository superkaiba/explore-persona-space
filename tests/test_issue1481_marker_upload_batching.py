"""Batching-seam tests for the #1481 marker dispatcher's data-repo uploads.

Live-run defect (task #1481 throughput-fix round): each per-rung slot-read
JSON uploaded as an INDIVIDUAL ``upload_file`` commit (~29 rungs x 48 cells
≈ 1,400 commits) 429-stormed the HF data repo commit endpoint while the next
cell's work gated on uploads. The fix batches at CELL grain: ONE
``upload_folder`` commit per (cell, stage-completion) point
(`_upload_cell_stage`), with the per-rung writers staying upload-free so a
mid-ladder crash leaves the accumulated files on disk for the crash-persist.

These tests pin: exactly ONE batched call per stage with the correct
``path_in_repo`` prefix (``issue1481_conpos_grid/marker/<cell>/``), all rung
files + rollouts included, sibling cells + training-state files excluded,
fail-loud on empty match / failed verify, the retirement of the per-file
loop in ``run_cell_unit``, and — per the one-production-body-test rule — a
real-body run reaching the ``hub._upload_folder_filtered`` network boundary
through a signature-conformant autospec fake.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1481_marker as mk  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

CELL = "mk-pers-con-lr5e6-s42"
OTHER = "mk-icl-po-lr5e6-s137"
RUNGS = (10, 20, 130)


def _fixture_out_root(tmp_path: Path) -> Path:
    """Schema-real cell dir shaped like the round-2A smoke outputs."""
    root = tmp_path / "out"
    cell = root / CELL
    (cell / "panel").mkdir(parents=True)
    for step in RUNGS:
        (cell / f"slot_reads_rung{step}.json").write_text(f'{{"step": {step}}}\n')
    (cell / "ladder.json").write_text('{"reads_by_step": {}}\n')
    (cell / "band_trajectory.json").write_text("{}\n")
    (cell / "selection.json").write_text('{"step": 20}\n')
    (cell / "apply_gate.json").write_text('{"verdict": "pass"}\n')
    for step in (10, 20):
        (cell / "panel" / f"rung{step}.json").write_text("{}\n")
    rc = root / "raw_completions"
    for stage in ("ladder", "panel", "mixes"):
        (rc / stage).mkdir(parents=True)
    (rc / "ladder" / f"{CELL}_rung10_111.json").write_text("{}\n")
    (rc / "ladder" / f"{CELL}_rung20_112.json").write_text("{}\n")
    (rc / "panel" / f"{CELL}_rung20_113.json").write_text("{}\n")
    (rc / "mixes" / "icl_bank_100.json").write_text("{}\n")
    # Distractors: a sibling cell's files + training-state files under the
    # cell's own train dir must NOT ride this cell's stage commits.
    other = root / OTHER
    other.mkdir(parents=True)
    (other / "slot_reads_rung10.json").write_text("{}\n")
    (rc / "ladder" / f"{OTHER}_rung10_119.json").write_text("{}\n")
    ckpt = cell / "train" / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_text("x")
    (ckpt / "optimizer.pt").write_text("x")
    return root


def _cfg_and_recording_seams(root: Path, calls: list[dict], *, upload: bool = True):
    cfg = mk.Cfg(smoke=True, cells=(CELL,), out_root=root, upload=upload)

    def upload_fn(local, repo_id, repo_type, path_in_repo, **kwargs) -> str:
        calls.append(
            {
                "local": str(local),
                "repo_id": repo_id,
                "repo_type": repo_type,
                "path_in_repo": path_in_repo,
                **kwargs,
            }
        )
        return f"smoke://{path_in_repo}"

    return cfg, mk.Seams(upload_fn=upload_fn)


def test_ladder_stage_single_batched_commit(tmp_path: Path) -> None:
    root = _fixture_out_root(tmp_path)
    calls: list[dict] = []
    cfg, seams = _cfg_and_recording_seams(root, calls)
    url = mk._upload_cell_stage(cfg, seams, CELL, "ladder")
    assert url == f"smoke://{mk.DATA_PREFIX}"
    assert len(calls) == 1, "ladder stage-completion must be exactly ONE upload_folder call"
    (call,) = calls
    assert call["local"] == str(root)
    assert call["repo_id"] == mk.HF_DATA_REPO
    assert call["repo_type"] == "dataset"
    assert call["path_in_repo"] == mk.DATA_PREFIX == "issue1481_conpos_grid/marker"
    expected = set(call["expected_repo_paths"])
    # ALL rung slot-reads + ladder + trajectory under the cell prefix.
    for step in RUNGS:
        assert f"{mk.DATA_PREFIX}/{CELL}/slot_reads_rung{step}.json" in expected
    assert f"{mk.DATA_PREFIX}/{CELL}/ladder.json" in expected
    assert f"{mk.DATA_PREFIX}/{CELL}/band_trajectory.json" in expected
    # The cell's ladder rollout text rides the SAME commit.
    assert f"{mk.DATA_PREFIX}/raw_completions/ladder/{CELL}_rung10_111.json" in expected
    assert f"{mk.DATA_PREFIX}/raw_completions/ladder/{CELL}_rung20_112.json" in expected
    # Sibling cell / other stages / training-state files stay OUT.
    assert not any(OTHER in p for p in expected)
    assert not any("/train/" in p or p.endswith("optimizer.pt") for p in expected)
    assert not any("/panel/" in p or p.endswith("selection.json") for p in expected)


def test_panel_stage_single_batched_commit(tmp_path: Path) -> None:
    root = _fixture_out_root(tmp_path)
    calls: list[dict] = []
    cfg, seams = _cfg_and_recording_seams(root, calls)
    mk._upload_cell_stage(cfg, seams, CELL, "panel")
    assert len(calls) == 1
    expected = set(calls[0]["expected_repo_paths"])
    for name in ("selection.json", "apply_gate.json", "ladder.json"):
        assert f"{mk.DATA_PREFIX}/{CELL}/{name}" in expected
    for step in (10, 20):
        assert f"{mk.DATA_PREFIX}/{CELL}/panel/rung{step}.json" in expected
    assert f"{mk.DATA_PREFIX}/raw_completions/panel/{CELL}_rung20_113.json" in expected
    # slot-reads belong to the LADDER commit, not the panel commit.
    assert not any("slot_reads_rung" in p for p in expected)
    assert not any(OTHER in p for p in expected)


def test_no_upload_short_circuits(tmp_path: Path) -> None:
    root = _fixture_out_root(tmp_path)
    calls: list[dict] = []
    cfg, seams = _cfg_and_recording_seams(root, calls, upload=False)
    assert mk._upload_cell_stage(cfg, seams, CELL, "ladder") == "skipped://no-upload"
    assert calls == []


def test_empty_match_fails_loud(tmp_path: Path) -> None:
    root = _fixture_out_root(tmp_path)
    calls: list[dict] = []
    cfg, seams = _cfg_and_recording_seams(root, calls)
    with pytest.raises(RuntimeError, match="no files match"):
        mk._upload_cell_stage(cfg, seams, "mk-nonexistent-cell", "ladder")
    assert calls == []


def test_per_rung_writers_stay_upload_free(tmp_path: Path) -> None:
    """A mid-ladder crash must leave the accumulated files ON DISK for the
    crash-persist sweep: the per-rung/per-battery writers never upload —
    uploads fire only at the stage-completion points in ``run_cell_unit``."""
    for fn in (mk._ladder_cell, mk._panel_battery):
        src = inspect.getsource(fn)
        assert "_upload" not in src, f"{fn.__name__} must not upload per rung/battery"
    # Functional half: files persist locally with zero upload calls recorded
    # until a stage-completion upload is invoked.
    root = _fixture_out_root(tmp_path)
    calls: list[dict] = []
    _cfg_and_recording_seams(root, calls)
    assert calls == []
    assert (root / CELL / "slot_reads_rung130.json").exists()


def test_cell_unit_wires_stage_uploads_and_retires_per_file_loop() -> None:
    src = inspect.getsource(mk.run_cell_unit)
    assert src.count("_upload_cell_stage(cfg") == 2, "ladder + panel stage-completion uploads"
    assert "as_file=True" not in src, "per-file upload_file loop must stay retired"
    assert "glob(" not in src, "per-file glob upload loop must stay retired"


def test_real_body_reaches_hub_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production-body test (code-style rule #906): seams.upload_fn=None runs
    the REAL body to the external Hub boundary, faked ONLY there with a
    signature-conformant autospec of ``hub._upload_folder_filtered``."""
    root = _fixture_out_root(tmp_path)
    cfg = mk.Cfg(smoke=False, cells=(CELL,), out_root=root, upload=True)
    seams = mk.Seams()  # every seam None -> the real path
    fake = mock.create_autospec(hub._upload_folder_filtered, return_value="repo/prefix")
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake)
    url = mk._upload_cell_stage(cfg, seams, CELL, "ladder")
    assert url == "repo/prefix"
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    args = fake.call_args.args
    assert args[0] == root and args[1] == mk.HF_DATA_REPO
    assert args[2] == "dataset" and args[3] == mk.DATA_PREFIX
    assert f"{CELL}/slot_reads_rung*.json" in kwargs["allow_patterns"]
    assert f"{mk.DATA_PREFIX}/{CELL}/slot_reads_rung130.json" in kwargs["expected_repo_paths"]


def test_real_body_raises_on_failed_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture_out_root(tmp_path)
    cfg = mk.Cfg(smoke=False, cells=(CELL,), out_root=root, upload=True)
    fake = mock.create_autospec(hub._upload_folder_filtered, return_value="")
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake)
    with pytest.raises(RuntimeError, match="returned no path"):
        mk._upload_cell_stage(cfg, mk.Seams(), CELL, "ladder")
