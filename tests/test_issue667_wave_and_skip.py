"""Issue #667 a36-reextract round-2 regression tests — GPU-count wave + idempotent skip.

Round-1 production crash (a36-readout-reextract-cos): ``phase_extract_r_plus`` fanned
out 4-way subprocess waves on ``--gpu-id 0..3`` UNCONDITIONALLY, assuming a 4-GPU
lane. On the auto-routed single-GPU A100-80 lane only ``--gpu-id 0`` saw a device;
``--gpu-id 1..3`` got ``CUDA_VISIBLE_DEVICES=1..3``, saw no device, and SILENTLY fell
back to CPU — 3 of every 4 cells crawled for hours and wave-1 never finished, so
wave-2 never launched.

These tests pin the round-2 fix:

1. ``_compute_wave_size`` returns the number of VISIBLE CUDA devices — 1 on a
   single-GPU lane, 4 on a 4-GPU lane — NOT a hardcoded 4 (pre-fix the wave was
   always ``max(n_gpus, 1) == 4``). 0 visible GPUs (no ``--cpu-only``) raises loud
   instead of stranding cells on a silent CPU fallback. ``--cpu-only`` → 1.
2. ``run_r_plus_extraction`` (extractor) SKIPS a cell whose FULL r⁺ layer set is
   already present (local OR HF) before any model load — so a relaunch after the
   round-1 crash does NOT re-extract the salvaged ``em/sp_swe`` cell — while a
   PARTIAL layer set (only some layers present) is re-extracted, never silently
   accepted as complete.
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue667_dispatch as disp  # noqa: E402
import issue667_extract as extract  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# (1) _compute_wave_size — wave = visible CUDA device count, not a hardcoded 4.
# ─────────────────────────────────────────────────────────────────────────────


def test_wave_size_is_one_on_single_gpu_lane():
    """1 visible CUDA device → wave size 1 (cells run serially on --gpu-id 0).

    Pre-fix the wave was ``max(n_gpus, 1)`` with ``n_gpus`` defaulting to 4, so a
    single-GPU lane fanned out 4-way and stranded --gpu-id 1..3 on a silent CPU
    fallback. The fix clamps the wave to the detected device count.
    """
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.device_count", return_value=1),
    ):
        # requested_n_gpus is the CLI default (4); the detected count (1) wins.
        assert disp._compute_wave_size(cpu_only=False, requested_n_gpus=4) == 1


def test_wave_size_is_four_on_four_gpu_lane():
    """4 visible CUDA devices → wave size 4 (full fan-out across all GPUs)."""
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.device_count", return_value=4),
    ):
        assert disp._compute_wave_size(cpu_only=False, requested_n_gpus=4) == 4


def test_wave_size_never_exceeds_visible_devices():
    """A 2-GPU lane with --n-gpus 4 clamps to 2 (never more lanes than devices)."""
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.device_count", return_value=2),
    ):
        assert disp._compute_wave_size(cpu_only=False, requested_n_gpus=4) == 2


def test_wave_size_cpu_only_is_one():
    """--cpu-only → wave size 1 (serial, no CUDA touched)."""
    # device_count must NOT even be consulted on the cpu_only branch.
    with mock.patch("torch.cuda.device_count", side_effect=AssertionError("touched CUDA")):
        assert disp._compute_wave_size(cpu_only=True, requested_n_gpus=4) == 1


def test_wave_size_zero_gpu_raises_loud():
    """0 visible CUDA devices (no --cpu-only) raises — never a silent CPU fallback.

    A wave of 0 GPUs is the round-1 crash class: cells would run on CPU for hours.
    The fix HALTs before the wave instead of stranding them.
    """
    with (
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch("torch.cuda.device_count", return_value=0),
    ):
        with pytest.raises(RuntimeError, match="at least 1 visible CUDA device"):
            disp._compute_wave_size(cpu_only=False, requested_n_gpus=4)


# ─────────────────────────────────────────────────────────────────────────────
# (2) Idempotent skip — a cell with ALL layers present (local or HF) is not re-run;
#     a PARTIAL layer set IS re-extracted (never silently accepted).
# ─────────────────────────────────────────────────────────────────────────────


def _write_npz_layers(cell_dir: Path, source: str, seed: int, layers: list[int]) -> None:
    """Write empty placeholder .npz files for the given layers (presence is all the
    skip check reads — it never opens them)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    for li in layers:
        (cell_dir / f"{source}_seed{seed}_L{li}.npz").write_bytes(b"")


def test_skip_when_all_layers_present_locally(tmp_path):
    """All 3 requested layers' .npz on local disk → (local_complete=True, ...).

    The HF probe is NOT consulted when local is already complete (cheap path).
    """
    out_dir = tmp_path / "em"
    _write_npz_layers(out_dir, "sp_swe", 42, [7, 14, 21])
    with mock.patch("huggingface_hub.get_paths_info", side_effect=AssertionError("hit HF")):
        local_done, hf_done = extract._r_plus_cell_already_extracted(
            out_dir, "em", "sp_swe", 42, [7, 14, 21]
        )
    assert local_done is True
    assert hf_done is False


def test_partial_local_set_is_not_complete_and_falls_to_hf_probe(tmp_path):
    """Only L7 present locally → local_complete=False; the HF probe then decides.

    A partial layer set does NOT prove the upstream prompt-set / hook layer
    matched, so it must NOT be treated as complete from the local files alone.
    Here HF also lacks the full set, so the cell is re-extracted (both False).
    """
    out_dir = tmp_path / "em"
    _write_npz_layers(out_dir, "sp_doctor", 42, [7])  # only L7 — partial
    # HF has nothing for this cell -> get_paths_info returns no matching paths.
    with mock.patch("huggingface_hub.get_paths_info", return_value=[]):
        local_done, hf_done = extract._r_plus_cell_already_extracted(
            out_dir, "em", "sp_doctor", 42, [7, 14, 21]
        )
    assert local_done is False
    assert hf_done is False


def test_skip_when_all_layers_present_on_hf(tmp_path):
    """Nothing local but all 3 layers on HF → hf_complete=True (the salvage case).

    Mirrors round-1's em/sp_swe r⁺ already uploaded before the crash: a relaunch
    must skip it without re-extracting.
    """
    out_dir = tmp_path / "em"  # empty — nothing local
    prefix = extract.HF_R_PLUS_PREFIX

    class _Info:
        def __init__(self, path):
            self.path = path

    hf_paths = [_Info(f"{prefix}/em/sp_swe_seed42_L{li}.npz") for li in (7, 14, 21)]
    with mock.patch("huggingface_hub.get_paths_info", return_value=hf_paths):
        local_done, hf_done = extract._r_plus_cell_already_extracted(
            out_dir, "em", "sp_swe", 42, [7, 14, 21]
        )
    assert local_done is False
    assert hf_done is True


def test_partial_hf_set_is_not_complete(tmp_path):
    """HF has only L7 (not L14/L21) → hf_complete=False; the cell is re-extracted."""
    out_dir = tmp_path / "em"
    prefix = extract.HF_R_PLUS_PREFIX

    class _Info:
        def __init__(self, path):
            self.path = path

    hf_paths = [_Info(f"{prefix}/em/sp_doctor_seed42_L7.npz")]  # only L7 on HF
    with mock.patch("huggingface_hub.get_paths_info", return_value=hf_paths):
        local_done, hf_done = extract._r_plus_cell_already_extracted(
            out_dir, "em", "sp_doctor", 42, [7, 14, 21]
        )
    assert local_done is False
    assert hf_done is False
