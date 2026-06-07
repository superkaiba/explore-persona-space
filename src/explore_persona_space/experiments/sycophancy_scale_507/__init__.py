"""Task #507 - sycophancy implantation + leakage-predictor port to Qwen-2.5-72B.

Scale port of #470's predictor study. Trains the SAME contrastive sycophancy
rig at the SAME #411 recipe on Qwen-2.5-72B-Instruct, then scores the same
base-model persona-distance predictors against the new per-bystander leakage
matrix.

Modules:
    train_72b           - thin wrapper around train.sft.train_lora with 4xH200
                          ZeRO-3 default (8xH100 supply-fallback); runtime
                          assertion that world_size * per_device_batch *
                          grad_accum == 16 (matches #411 effective batch).
    eval_72b_vllm       - thin wrapper around sycophancy_implantation_411.
                          eval_one_source with TP=8 + bf16; one vLLM server
                          serves all 6 trained adapters.
    predictor_72b       - drives the 6 phases of predictor_jsdiv_470 with
                          model id = Qwen/Qwen2.5-72B-Instruct + layer set
                          {21, 40, 57, 70}; preflight enforces no-CPU-offload.
    analyze_507         - extends phase5_regress to read 7B (#470 frozen) +
                          72B (new) per-cell artifacts; paired-bootstrap
                          |rho_72B| - |rho_7B| 95% CI for the analyzer
                          hand-off.

The single experimental variable vs #411 is base-model size; everything
else (sources, contrastive negative panel, recipe, judge, probes, panel) is
held fixed by inheritance (plan v2 section 4.1).
"""

from __future__ import annotations

# 6 source personas inherited verbatim from #411 / #470 (plan v2 section 4.1).
SOURCE_PERSONAS_507: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

# Two arms compared in the headline figure.
MODEL_ARMS: tuple[tuple[str, str], ...] = (
    ("7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("72b", "Qwen/Qwen2.5-72B-Instruct"),
)

# Depth-equivalent layer sets per architecture.
# 7B (28 layers) uses {7, 14, 21, 27}; #470 published headline = layer 21.
# 72B (80 layers) uses {21, 40, 57, 70} = round(0.25, 0.50, 0.714, 0.875 x 80).
# Layer 57 = depth-equivalent of #470's published baseline layer 20 on 7B
# (20/28 ~ 57/80 ~ 0.714). See plan v2 section 4.6.
LAYER_SET_BY_ARCH: dict[str, tuple[int, ...]] = {
    "7b": (7, 14, 21, 27),
    "72b": (21, 40, 57, 70),
}

# Headline layer per arm (used for the primary ro bar in figures).
HEADLINE_LAYER_BY_ARCH: dict[str, int] = {
    "7b": 21,
    "72b": 57,
}

# Expected effective batch size; runtime assertion in train_72b uses this.
# Matches #411 verbatim (per_device_batch=4, grad_accum=4, world_size=1 -> 16).
EXPECTED_EFFECTIVE_BATCH: int = 16

# Per-device train batch is pinned at 1 on the 72B path; grad_accum scales
# inversely with world_size to preserve effective batch 16.
PER_DEVICE_TRAIN_BATCH_72B: int = 1


def compute_grad_accum(world_size: int, per_device_batch: int = PER_DEVICE_TRAIN_BATCH_72B) -> int:
    """Compute grad_accum so world_size x per_device_batch x grad_accum == 16.

    The load-bearing piece of the 72B pod-shape dispatch: on the 4xH200 default
    path world_size=4 -> grad_accum=4; on the 8xH100 supply-fallback path
    world_size=8 -> grad_accum=2. Either way effective batch = 16 matches #411.

    Raises:
        ValueError: when the chosen world_size + per_device_batch cannot
            divide 16 cleanly, i.e. when no integer grad_accum preserves the
            effective batch contract. Fail-loud at config time, never silently
            train under a different recipe than #411.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if per_device_batch <= 0:
        raise ValueError(f"per_device_batch must be positive, got {per_device_batch}")
    denom = world_size * per_device_batch
    if EXPECTED_EFFECTIVE_BATCH % denom != 0:
        raise ValueError(
            f"Cannot preserve effective batch {EXPECTED_EFFECTIVE_BATCH}: "
            f"world_size ({world_size}) * per_device_batch ({per_device_batch}) "
            f"= {denom} does not divide {EXPECTED_EFFECTIVE_BATCH} cleanly. "
            f"#411 parity (eff batch 16) requires world_size * per_device_batch "
            f"to divide 16; pick a world_size from {{1, 2, 4, 8, 16}}."
        )
    return EXPECTED_EFFECTIVE_BATCH // denom


def get_world_size_from_env() -> int:
    """Read world_size from torch.distributed / env, with fall-through to 1.

    Used by the dispatcher to pick grad_accum at runtime before train_72b is
    invoked. The order matches what HF/DeepSpeed launcher sets:
    1) torch.distributed (if initialized)
    2) WORLD_SIZE env var (set by torchrun / deepspeed / accelerate launchers)
    3) fallback 1 (single-GPU debug / CPU smoke)
    """
    import os

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except ImportError:
        pass
    return int(os.environ.get("WORLD_SIZE", "1"))
