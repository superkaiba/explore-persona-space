"""Wave-size + CVD-pin regression for the #667 all-layer dispatcher.

Pins the two invariants the feedback memory
``dispatcher_wave_size_must_match_visible_gpus`` (#667 a36) demands of any
per-cell subprocess wave dispatcher:

1. ``compute_wave_size`` derives the parallel wave from the DETECTED visible-GPU
   count (``torch.cuda.device_count()``), NOT a hardcoded constant or the
   ``--n-gpus`` default; ``--n-gpus`` is a CEILING; a GPU run with 0 visible
   devices RAISES loud (never a silent CPU fallback); ``--cpu-only`` -> 1; and a
   ``--dry-run`` previews the requested ceiling without touching CUDA.

2. Every per-cell extract command pins ``CUDA_VISIBLE_DEVICES=<gpu>`` in the
   LAUNCHER env matching its ``--gpu-id`` (the #545 launcher-env pin an
   import-time cuInit cannot defeat) — checked on the dry-run wave plan.

Pure logic, no GPU, ~1s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue667_alllayer_dispatch as disp  # noqa: E402


def test_cpu_only_wave_is_serial():
    assert disp.compute_wave_size(cpu_only=True, requested_n_gpus=8) == 1


def test_dry_run_previews_requested_ceiling_without_cuda(monkeypatch):
    # Even with 0 visible GPUs, a dry-run must PREVIEW the requested fan-out
    # (a GPU-less VM shows the intended per-lane CVD assignment for review).
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 0)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8, dry_run=True) == 8


def test_wave_equals_detected_count_when_below_ceiling(monkeypatch):
    # 8 visible, --n-gpus 8 -> wave 8 (the production 8xH100 case).
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 8)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8) == 8


def test_wave_clamps_to_detected_below_ceiling(monkeypatch):
    # The #667 a36 hang class: --n-gpus 8 on a 1-GPU lane must NOT spawn
    # --gpu-id 1..7 (which would see no device and silently run on CPU).
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 1)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=8) == 1


def test_ceiling_below_detected_is_honored(monkeypatch):
    # --n-gpus is a CEILING: 8 visible but --n-gpus 4 -> wave 4.
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 8)
    assert disp.compute_wave_size(cpu_only=False, requested_n_gpus=4) == 4


def test_zero_visible_gpu_raises_loud(monkeypatch):
    # A GPU run with 0 visible devices is the silent-CPU crash class — RAISE.
    monkeypatch.setattr(disp, "_visible_gpu_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no CUDA devices visible"):
        disp.compute_wave_size(cpu_only=False, requested_n_gpus=8)


def test_per_cell_cvd_pin_matches_gpu_id():
    # The launcher-env CVD pin must match --gpu-id for every slot (gotchas.md).
    for gpu_id in range(8):
        cmd, _log, env = disp._extract_cmd(
            behavior="em",
            source="default",
            targets=None,
            primary_layer=14,
            gpu_id=gpu_id,
            max_probes=None,
            max_train_rows=None,
            cpu_only=False,
        )
        assert env["CUDA_VISIBLE_DEVICES"] == str(gpu_id), env
        # the --gpu-id arg the extractor's in-process clobber rewrites to the SAME value
        assert "--gpu-id" in cmd and cmd[cmd.index("--gpu-id") + 1] == str(gpu_id), cmd
        # the all-layer flag is threaded (the only substantive change vs #667)
        assert "--all-layers" in cmd, cmd


def test_extract_cmd_writes_to_alllayer_namespace():
    # Must NOT clobber the committed 7/14/21 store.
    cmd, _log, _env = disp._extract_cmd(
        behavior="em",
        source="default",
        targets=None,
        primary_layer=14,
        gpu_id=0,
        max_probes=None,
        max_train_rows=None,
        cpu_only=False,
    )
    out = cmd[cmd.index("--out") + 1]
    assert out == "eval_results/issue_667_alllayer/analysis_tensors", out
    assert "gate_chain_preview" not in out
