"""#962: _device() fail-loud CVD guard (incident #813).

A launcher-pinned-class worker launched with --gpu-id N>0 must have
CUDA_VISIBLE_DEVICES pinned to exactly str(N) in the environment, else
_device() raises at device resolution (before any model/engine load).
All CPU; no GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue667_extract as ex  # noqa: E402


def test_gpu_id_positive_without_cvd_raises(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        ex._device(3, cpu_only=False)


def test_error_message_names_relaunch_recipe(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match=r"env CUDA_VISIBLE_DEVICES=3 .*--gpu-id 3"):
        ex._device(3, cpu_only=False)


def test_error_message_names_gotchas_pointer(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match=r"gotchas\.md"):
        ex._device(3, cpu_only=False)


def test_error_message_reports_unset_when_cvd_absent(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match=r"observed: unset"):
        ex._device(3, cpu_only=False)


def test_multi_value_cvd_raises_and_reports_observed_value(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(RuntimeError, match=r"observed: '0,1,2,3'"):
        ex._device(3, cpu_only=False)


def test_mismatched_cvd_raises(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError):
        ex._device(3, cpu_only=False)


def test_empty_cvd_with_gpu_id_positive_raises(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError):
        ex._device(3, cpu_only=False)


def test_matching_cvd_passes(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert ex._device(3, cpu_only=False).type in ("cpu", "cuda")


def test_gpu_id_zero_without_cvd_passes(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ex._device(0, cpu_only=False).type in ("cpu", "cuda")


def test_cpu_only_short_circuits_any_gpu_id(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ex._device(7, cpu_only=True) == torch.device("cpu")


def test_matching_cvd_and_cuda_available_returns_cuda0(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ex._device(2, cpu_only=False) == torch.device("cuda:0")
