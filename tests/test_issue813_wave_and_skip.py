"""Wave-size + CVD-pin + resume-skip regression for the #813 extraction dispatcher.

Pins the invariants the feedback memory
``dispatcher_wave_size_must_match_visible_gpus`` (#667 a36) demands of any
per-cell subprocess wave dispatcher, mirroring
``tests/test_issue667_alllayer_wave.py``:

1. ``compute_wave_size`` derives the parallel wave from the DETECTED visible-GPU
   count (``torch.cuda.device_count()``), NOT a hardcoded constant or the
   ``--n-gpus`` default; ``--n-gpus`` is a CEILING; a GPU run with 0 visible
   devices RAISES loud (never a silent CPU fallback); ``--cpu-only`` -> 1; and a
   ``--dry-run`` previews the requested ceiling without touching CUDA.

2. Every per-cell command pins ``CUDA_VISIBLE_DEVICES=<gpu>`` in the LAUNCHER env
   matching its ``--gpu-id`` (the #545 launcher-env pin an import-time cuInit
   cannot defeat) AND passes the matching ``--gpu-id``.

3. The 12 (behavior x substrate) cells enumerate correctly, and the run-cell
   resume-skip sentinel predicate skips a completed cell.

Pure logic, no GPU, ~1s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue813_dispatch as disp  # noqa: E402


def test_cpu_only_wave_is_serial():
    assert disp.compute_wave_size(cpu_only=True, requested_n_gpus=8) == 1


def test_dry_run_previews_requested_ceiling_without_cuda(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 0)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8, dry_run=True) == 8


def test_wave_equals_detected_when_below_ceiling(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 8)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8) == 8


def test_wave_clamps_to_detected_below_ceiling(monkeypatch):
    # The #667 a36 hang class: --n-gpus 8 on a 1-GPU lane must NOT spawn
    # --gpu-id 1..7 (which would see no device and silently run on CPU).
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 1)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8) == 1


def test_ceiling_below_detected_is_honored(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 8)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=4) == 4


def test_zero_visible_gpu_raises_loud(monkeypatch):
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no CUDA devices visible"):
        disp.compute_wave_size(cpu_only=False, requested_n_gpus=8)


def test_enumerate_cells_is_full_grid():
    cells = disp.enumerate_cells(list(disp.BEHAVIORS), list(disp.SUBSTRATES))
    assert len(cells) == 12  # 4 behaviors x 3 substrates
    assert ("em", "generic") in cells and ("marker", "mix") in cells


def test_per_cell_cvd_pin_matches_gpu_id():
    # The launcher-env CVD pin must match --gpu-id for every slot (gotchas.md #545).
    for gpu_id in range(8):
        cmd, env = disp._cell_cmd(
            "em",
            "generic",
            gpu_id,
            out_root="eval_results/issue_813",
            cpu_only=False,
            upload=True,
            max_contexts=None,
            max_questions=None,
        )
        assert env["CUDA_VISIBLE_DEVICES"] == str(gpu_id), env
        assert "--gpu-id" in cmd and cmd[cmd.index("--gpu-id") + 1] == str(gpu_id), cmd
        assert "--upload" in cmd, cmd
        assert cmd[cmd.index("--behavior") + 1] == "em"
        assert cmd[cmd.index("--substrate") + 1] == "generic"


def test_cpu_only_cell_cmd_has_no_cvd_pin():
    # CPU-only lanes must NOT pin CVD (there is no physical GPU to pin).
    cmd, env = disp._cell_cmd(
        "marker",
        "mix",
        0,
        out_root="eval_results/issue_813",
        cpu_only=True,
        upload=False,
        max_contexts=2,
        max_questions=2,
    )
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "--cpu-only" in cmd
    assert "--upload" not in cmd
    assert cmd[cmd.index("--max-contexts") + 1] == "2"


def test_run_cell_resume_skip_predicate(tmp_path, monkeypatch):
    # A completed cell (sentinel present) is skipped on re-run unless --force.
    import issue813_run_cell as rc

    reduced_dir = tmp_path / "reduced" / "marker" / "generic"
    reduced_dir.mkdir(parents=True)
    sentinel = reduced_dir / rc.CELL_DONE_SENTINEL
    sentinel.write_text(json.dumps({"behavior": "marker", "substrate": "generic"}))

    class Args:
        behavior = "marker"
        substrate = "generic"
        out_root = tmp_path
        gpu_id = 0
        cpu_only = True
        upload = False
        force = False
        max_contexts = 2
        max_questions = 2
        metrics_out = None

    out = rc.run_cell(Args())
    assert out.get("skipped") is True, out
