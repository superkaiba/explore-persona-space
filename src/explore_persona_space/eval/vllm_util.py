"""Shared-node vLLM ``gpu_memory_utilization`` resolver (#1902 crash 1; hoisted by #1942).

``gpu_memory_utilization`` is a fraction of TOTAL device memory (vLLM
semantics), so a fixed value demands that share regardless of what other
tenants hold. On a GPU-SHARED fellows H200 node (no GPU cgroup isolation;
~58 GiB/device held by other tenants, measured 2026-07-31) a hardcoded 0.6
demanded 0.6 x 139.8 GiB = 83.9 GiB vs 81.2 GiB free and EngineCore raised
``ValueError`` at init (#1902 crash 1). The remedy is a LIVE-probed fraction:
``min(cap, (free - margin) / total)`` from ``torch.cuda.mem_get_info``,
fail-loud below the floor.

This module is the SINGLE implementation of that recipe. It was written twice
first — ``scripts/issue1902_common.py`` (cap 0.55, shared fellows nodes) and a
deliberate re-port in ``scripts/issue1345_onpolicy_answers_gen.py`` (cap 0.85,
exclusive hosts; re-ported because #1902's branch was unmerged at the time) —
and #1942 hoisted both copies here with the CAP parametrized. Margin and floor
stay module constants (no speculative parameters).

Module import is torch-free (``resolve_vllm_util`` imports torch lazily), so
importing this module never drags CUDA into a CPU-only process.
"""

from __future__ import annotations

# Allowance for other-tenant growth between the mem_get_info read and the
# allocation actually landing (and for allocator slack).
GPU_FREE_MARGIN_GIB = 6.0
# Below this fraction of TOTAL, 7B bf16 weights (~15 GiB) + a minimum KV cache
# cannot fit — fail loud with a shared-node message instead of letting vLLM
# die cryptically inside EngineCore init.
VLLM_UTIL_FLOOR = 0.20
# Shared-node ceiling (fellows H200 hosts share nodes WITHOUT GPU isolation).
# On an exclusive H100/A100/H200 the computed util resolves to this cap,
# matching the historical ~0.6 behavior closely. Source: #1902 plan +
# scripts/issue1902_common.py (VLLM_UTIL_CAP = 0.55).
SHARED_NODE_UTIL_CAP = 0.55
# Exclusive-host ceiling (the #1345 on-policy answers round runs on exclusive
# hosts). Source: #1345 commit 48ec6c7d2d (VLLM_UTIL_CAP = 0.85).
EXCLUSIVE_HOST_UTIL_CAP = 0.85


def vllm_util_for_free(
    free_bytes: int, total_bytes: int, *, cap: float = SHARED_NODE_UTIL_CAP
) -> float:
    """vLLM ``gpu_memory_utilization`` computed from LIVE free device memory.

    ``gpu_memory_utilization`` is a fraction of TOTAL device memory, so a
    fixed 0.6 on a shared node demands ``0.6 x total`` bytes regardless of
    what other tenants hold (#1902 crash 1: 0.6 x 139.8 GiB = 83.9 GiB
    demanded vs 81.2 GiB free on a fellows H200 → EngineCore ValueError at
    init). Returns ``min(cap, (free - margin) / total)``; raises
    ``RuntimeError`` below ``VLLM_UTIL_FLOOR`` (weights + minimum KV cannot
    fit — the device is too full for any engine).
    """
    if total_bytes <= 0:
        raise RuntimeError(f"nonsensical total device memory: {total_bytes} bytes")
    free_gib = free_bytes / 2**30
    total_gib = total_bytes / 2**30
    util = min(cap, (free_gib - GPU_FREE_MARGIN_GIB) / total_gib)
    if util < VLLM_UTIL_FLOOR:
        raise RuntimeError(
            f"GPU too full for a vLLM engine: free={free_gib:.1f} GiB of "
            f"{total_gib:.1f} GiB total → computed gpu_memory_utilization "
            f"{util:.3f} < floor {VLLM_UTIL_FLOOR} after the "
            f"{GPU_FREE_MARGIN_GIB:.0f} GiB margin. On a shared node (fellows "
            "H200) this means another tenant holds the device — re-dispatch "
            "when it frees, or pin a different allocated GPU."
        )
    return util


def resolve_vllm_util(*, cap: float = SHARED_NODE_UTIL_CAP) -> float:
    """Live-probed engine util, or the cap when CUDA is unavailable (CPU smoke)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return cap
        free_b, total_b = torch.cuda.mem_get_info()
    except (ImportError, RuntimeError) as exc:  # no CUDA / no driver — CPU path
        print(f"[gpu] mem_get_info unavailable ({exc}); using cap {cap}", flush=True)
        return cap
    util = vllm_util_for_free(free_b, total_b, cap=cap)
    print(
        f"[gpu] free={free_b / 2**30:.1f} GiB total={total_b / 2**30:.1f} GiB "
        f"-> gpu_memory_utilization={util:.3f} (cap {cap})",
        flush=True,
    )
    return util
