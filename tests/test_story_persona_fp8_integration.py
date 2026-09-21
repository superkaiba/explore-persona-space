"""Opt-in real pinned loader/custom-op tests; only the CUDA launch is substituted.

Run with EPS_TEST_PINNED_FP8=1 and the production Transformers/kernels/Triton pins.
These CPU boundary tests do not certify GPU correctness; load_model runs the CUDA
shape diagnostic and unchanged full-model numerical smoke on the approved node.
"""

import importlib
import os
from unittest.mock import create_autospec

import pytest
import torch
from omegaconf import OmegaConf

from scripts import story_persona_fp8 as repair
from scripts.story_persona_crossmodel_capture import pin_fp8_kernel

pytestmark = pytest.mark.skipif(
    os.environ.get("EPS_TEST_PINNED_FP8") != "1", reason="requires pinned FP8 runtime/cache"
)


@pytest.fixture(scope="module")
def pinned():
    """Use the real Transformers loader and immutable cached kernel module."""
    from transformers.integrations import hub_kernels
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel

    cfg = OmegaConf.load("configs/pilots/story_persona_crossmodel_capture.yaml")
    cfg.model_key = "deepseek"
    manifest = pin_fp8_kernel(cfg)
    kernel = hub_kernels.lazy_load_kernel("finegrained-fp8")
    assert load_finegrained_fp8_kernel().grouped_matmul is kernel.matmul_grouped
    return kernel, manifest


def test_real_loader_and_registered_dispatch_reach_fixed_schedule(pinned, monkeypatch, capsys):
    kernel, manifest = pinned
    grouped = importlib.import_module(kernel.matmul_grouped.__module__)
    utils = importlib.import_module(grouped.__package__ + ".utils")
    calls = []

    class LaunchBoundary:
        """Replace only Triton's GPU launch, preserving the actual custom-op body."""

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append({"grid": grid, **kwargs})
                args[2].fill_(1)

            return launch

    monkeypatch.setattr(
        grouped,
        "wrap_triton",
        create_autospec(grouped.wrap_triton, return_value=LaunchBoundary()),
    )
    weights = torch.zeros(256, 256, 128, dtype=torch.float8_e4m3fn)
    scales = torch.full((256, 2, 1), 127, dtype=torch.uint8)
    for count in (2, 43):
        counts = torch.full((256,), count, dtype=torch.int32)
        offsets = counts.cumsum(0).to(torch.int32)
        activations = torch.ones(256 * count, 128, dtype=torch.bfloat16)
        result = kernel.matmul_grouped(activations, weights, scales, offsets, counts, [128, 128])
        assert result.shape == (256 * count, 256) and torch.all(result == 1)
    assert [c["BLOCK_SIZE_M"] for c in calls] == [16, 16]
    assert calls[1]["grid"][0] == (256 * 43 + 15) // 16 + 256
    assert utils.adaptive_block_size_m(43) == 64  # Other kernel families remain unchanged.
    signal = capsys.readouterr().out
    assert "[fp8-grouped-m16-engaged]" in signal
    assert "target_m=43 original_m=64 fixed_m=16" in signal
    print(signal, end="")
    assert manifest["build_sha256"]["grouped.py"] == repair.GROUPED_SHA256
    assert manifest["scheduling_override"]["kernel_source_files_modified"] is False
    with pytest.raises(RuntimeError, match="unreviewed dispatcher"):
        grouped.adaptive_block_size_m(43)


def test_repair_rejects_prepatched_original_function(pinned, monkeypatch):
    kernel, _ = pinned
    grouped = importlib.import_module(kernel.matmul_grouped.__module__)
    monkeypatch.setattr(grouped, "adaptive_block_size_m", lambda target_m: 16)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        repair.install_grouped_m16(kernel, repair.REVISION)


def test_repair_rejects_changed_revision_and_source_bytes(pinned, monkeypatch):
    kernel, _ = pinned
    with pytest.raises(RuntimeError, match="reviewed kernel revision"):
        repair.install_grouped_m16(kernel, "0" * 40)
    monkeypatch.setattr(repair, "GROUPED_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="immutable source mismatch"):
        repair.install_grouped_m16(kernel, repair.REVISION)
