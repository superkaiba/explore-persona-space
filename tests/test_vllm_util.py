"""Tests for the shared-node vLLM util resolver (`explore_persona_space.eval.vllm_util`).

Consolidates the pure-math + resolver tests formerly duplicated across
tests/test_issue1902_run.py and tests/test_issue1345_onpolicy_answers.py
(#1942: the two script-local copies were hoisted into ONE shared module with
the cap parametrized — 0.55 shared-node default, 0.85 exclusive-host).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from explore_persona_space.eval.vllm_util import (
    EXCLUSIVE_HOST_UTIL_CAP,
    GPU_FREE_MARGIN_GIB,
    SHARED_NODE_UTIL_CAP,
    VLLM_UTIL_FLOOR,
    resolve_vllm_util,
    vllm_util_for_free,
)

GIB = 2**30


@pytest.mark.parametrize("cap", [SHARED_NODE_UTIL_CAP, EXCLUSIVE_HOST_UTIL_CAP])
def test_empty_device_resolves_to_cap(cap):
    got = vllm_util_for_free(int(139.8 * GIB), int(139.8 * GIB), cap=cap)
    assert got == pytest.approx(cap)
    if cap == EXCLUSIVE_HOST_UTIL_CAP:
        # Incident-documentation assert (#1902 crash 1): the bare 0.85 cap
        # WOULD have over-demanded on the shared H200 (0.85 x 139.8 GiB =
        # 118.8 GiB demanded vs 81.2 GiB free) — the live probe prevents it.
        assert EXCLUSIVE_HOST_UTIL_CAP * 139.8 > 81.2


def test_shared_h200_crash_shape():
    # Crash 1 numbers: free 81.2 GiB of 139.8 GiB total. The fixed 0.6 default
    # demanded 83.9 GiB > free; the computed util's demand must fit free.
    free_b, total_b = int(81.2 * GIB), int(139.8 * GIB)
    util = vllm_util_for_free(free_b, total_b)
    assert util < SHARED_NODE_UTIL_CAP
    assert util * total_b < free_b
    assert util >= VLLM_UTIL_FLOOR
    # The demanded share must fit inside free minus the safety margin.
    assert util * 139.8 <= 81.2 - GPU_FREE_MARGIN_GIB + 1e-6


def test_exclusive_h100_resolves_to_cap():
    # Exclusive H100: ~79 GiB free of ~79.6 GiB total -> cap binds.
    assert vllm_util_for_free(int(79 * GIB), int(79.6 * GIB)) == SHARED_NODE_UTIL_CAP


def test_below_floor_fails_loud():
    # A co-tenant holding most of the device: weights + min KV cannot fit.
    with pytest.raises(RuntimeError, match="GPU too full"):
        vllm_util_for_free(int(30 * GIB), int(139.8 * GIB))


def test_rejects_nonsense_total():
    with pytest.raises(RuntimeError, match="nonsensical total"):
        vllm_util_for_free(1, 0)


def test_resolve_returns_passed_cap_when_cuda_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_vllm_util() == SHARED_NODE_UTIL_CAP
    assert resolve_vllm_util(cap=EXCLUSIVE_HOST_UTIL_CAP) == EXCLUSIVE_HOST_UTIL_CAP


def test_resolve_live_path_matches_pure_math(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (int(81.2 * GIB), int(139.8 * GIB)))
    util = resolve_vllm_util()
    assert abs(util - vllm_util_for_free(int(81.2 * GIB), int(139.8 * GIB))) < 1e-12
    util_excl = resolve_vllm_util(cap=EXCLUSIVE_HOST_UTIL_CAP)
    expected = vllm_util_for_free(int(81.2 * GIB), int(139.8 * GIB), cap=EXCLUSIVE_HOST_UTIL_CAP)
    assert abs(util_excl - expected) < 1e-12


def test_module_import_is_torch_free():
    # Importing the shared module must never drag torch/CUDA into a CPU-only
    # process (the lazy-import contract; plan acceptance criterion 1).
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import explore_persona_space.eval.vllm_util; "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "False"
