"""Issue #667 tf-margin dispatcher — wave-size + resume-skip regression tests.

Pins the a36-round wave-size bug fix (plan v6 §4.3/§8): the tf-margin dispatcher's
``_compute_wave_size`` derives the wave from the DETECTED
``torch.cuda.device_count()`` (NOT ``--n-gpus`` alone), so a smaller-than-8 GPU
lane never fans out surplus ``--gpu-id`` lanes onto CPU. Covers:
  - cpu_only -> 1
  - detected 8, requested 8 -> 8 (the plan's wave==visible==8 case)
  - detected 4, requested 8 -> 4 (--n-gpus is a CEILING; detected is truth)
  - detected 0, not cpu_only -> RAISE LOUD
plus the per-cell resume-skip predicate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue667_tf_margin_dispatch as disp  # noqa: E402


def test_compute_wave_size_cpu_only_is_one():
    assert disp._compute_wave_size(cpu_only=True, requested_n_gpus=8) == 1


def test_compute_wave_size_detected_8_requested_8():
    with mock.patch("torch.cuda.device_count", return_value=8):
        assert disp._compute_wave_size(cpu_only=False, requested_n_gpus=8) == 8


def test_compute_wave_size_requested_exceeds_detected_clamps_to_detected():
    # --n-gpus 8 is a CEILING; a 4-GPU lane must clamp to 4 (never fan out 8 lanes).
    with mock.patch("torch.cuda.device_count", return_value=4):
        assert disp._compute_wave_size(cpu_only=False, requested_n_gpus=8) == 4


def test_compute_wave_size_zero_visible_gpu_raises():
    with (
        mock.patch("torch.cuda.device_count", return_value=0),
        pytest.raises(RuntimeError, match="0 visible CUDA devices"),
    ):
        disp._compute_wave_size(cpu_only=False, requested_n_gpus=8)


def test_cell_done_predicate(monkeypatch, tmp_path):
    """_cell_done is True iff the per-cell tf_margins.json exists on disk."""
    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "PER_CELL_DIR", "pc")
    cell_dir = tmp_path / "pc" / "em" / f"default_seed{disp._SEED}"
    cell_dir.mkdir(parents=True)
    assert disp._cell_done("em", "default") is False
    (cell_dir / "tf_margins.json").write_text("{}")
    assert disp._cell_done("em", "default") is True


def test_phase_extract_wave_pins_distinct_cvd_per_cell(monkeypatch, tmp_path):
    """A real (non-dry-run) wave of 2 cells on a detected-8-GPU lane pins CVD 0 and 1.

    Guards the a36 bug directly: the fan-out width comes from the DETECTED device
    count, so two cells in one wave get distinct CUDA_VISIBLE_DEVICES lanes (0, 1)
    — never both on GPU 0.
    """
    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "PER_CELL_DIR", "pc")
    monkeypatch.setattr(disp, "select_sources", lambda behavior, arg: ["default", "sp_swe"])
    monkeypatch.setattr(disp, "select_targets", lambda behavior, arg: ["fmt_json"])
    monkeypatch.setattr(disp, "_upload_per_cell", lambda: None)

    seen_cvd = []

    def fake_parallel(cmds):
        cmds = list(cmds)
        for _cmd, _lp, env in cmds:
            seen_cvd.append(env.get("CUDA_VISIBLE_DEVICES"))
        return [0] * len(cmds)

    monkeypatch.setattr(disp, "_run_parallel_with_log", fake_parallel)
    with mock.patch("torch.cuda.device_count", return_value=8):
        disp.phase_extract(
            behaviors=["em"],
            sources_arg="default,sp_swe",
            targets_arg="fmt_json",
            cap=4,
            n_gpus=8,
            cpu_only=False,
            skip_upload=True,
            dry_run=False,
            resume_skip=False,
        )
    # both cells in one wave -> distinct CVD lanes 0 and 1 (a36 fix).
    assert set(seen_cvd) == {"0", "1"}


def test_phase_extract_dry_run_needs_no_gpu(monkeypatch, tmp_path, caplog):
    """dry-run extract builds + logs commands with no GPU required (single lane)."""
    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "PER_CELL_DIR", "pc")
    monkeypatch.setattr(disp, "select_sources", lambda behavior, arg: ["default", "sp_swe"])
    monkeypatch.setattr(disp, "select_targets", lambda behavior, arg: ["fmt_json"])
    import logging

    # device_count == 0 (this CPU VM): dry-run must NOT raise the 0-GPU guard.
    with mock.patch("torch.cuda.device_count", return_value=0), caplog.at_level(logging.INFO):
        disp.phase_extract(
            behaviors=["em"],
            sources_arg="default,sp_swe",
            targets_arg="fmt_json",
            cap=4,
            n_gpus=8,
            cpu_only=False,
            skip_upload=True,
            dry_run=True,
            resume_skip=False,
        )
    text = "\n".join(r.message for r in caplog.records)
    assert "[dry-run] extract em/default" in text


def test_phase_extract_resume_skips_done_cell(monkeypatch, tmp_path):
    """A cell whose tf_margins.json exists is dropped from the launch list."""
    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "PER_CELL_DIR", "pc")
    monkeypatch.setattr(disp, "select_sources", lambda behavior, arg: ["default", "sp_swe"])
    monkeypatch.setattr(disp, "select_targets", lambda behavior, arg: ["fmt_json"])
    # mark 'default' done
    done_dir = tmp_path / "pc" / "em" / f"default_seed{disp._SEED}"
    done_dir.mkdir(parents=True)
    (done_dir / "tf_margins.json").write_text("{}")

    launched = []

    def fake_parallel(cmds):
        for cmd, _lp, _env in cmds:
            # source-cid is the arg after --source-cid
            i = cmd.index("--source-cid")
            launched.append(cmd[i + 1])
        return [0] * len(list(cmds))

    monkeypatch.setattr(disp, "_run_parallel_with_log", fake_parallel)
    monkeypatch.setattr(disp, "_upload_per_cell", lambda: None)
    with mock.patch("torch.cuda.device_count", return_value=8):
        disp.phase_extract(
            behaviors=["em"],
            sources_arg="default,sp_swe",
            targets_arg="fmt_json",
            cap=4,
            n_gpus=8,
            cpu_only=False,
            skip_upload=True,
            dry_run=False,
            resume_skip=True,
        )
    assert launched == ["sp_swe"]  # 'default' skipped (already done)
