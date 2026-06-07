"""Task #507 Phase 2 - Qwen-2.5-72B vLLM eval (thin wrapper around eval_source).

Wraps sycophancy_implantation_411.eval_one_source.eval_source with TP=8 +
bf16 on the inf-70b 8xH100 pod. The base 72B is loaded ONCE per source's
call, and the run-time vLLM cache hold is the dominant cost (12k generations
per cell at ~700 tok/s aggregate ~ 30 min).

This function is called sequentially per source by the dispatcher; vLLM is
recreated each call (subprocess-isolated via dispatch_sycophancy_507.py's
_eval_subprocess_72b — same #399 vLLM-worker-teardown mitigation #411 used).

The plan v2 section 4.3 originally specified one vLLM server serving 6
adapters via --lora-modules; we keep the per-cell call here as the
contingency-safe path (vLLM TP=8 + multi-LoRA on 72B is the medium-confidence
A6 assumption in the plan, and a per-source subprocess is well-tested at 7B
in #411). The dispatcher can switch to the shared-server path by passing
multiple lora_modules to a single eval_72b call when the smoke cell
demonstrates the shared path is stable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from explore_persona_space.experiments.sycophancy_implantation_411.eval_one_source import (
    eval_source,
)

log = logging.getLogger("sycophancy_scale_507.eval_72b_vllm")

BASE_MODEL_72B = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_TP_72B = 8
# 72B + bf16 KV cache needs max_model_len <= 4096 to fit comfortably at TP=8
# on 8xH100 80GB. Sycophancy responses are short (<=128 trained tokens) so
# 2048 is enough for prompt + response.
DEFAULT_MAX_MODEL_LEN_72B = 2048
DEFAULT_GPU_MEM_UTIL_72B = 0.90
DEFAULT_MAX_LORA_RANK = 32


def eval_72b(
    *,
    source: str,
    seed: int,
    merged_model_path: Path | None = None,
    base_model_id: str = BASE_MODEL_72B,
    adapter_path: Path | None = None,
    eval_pool_path: Path,
    out_dir: Path,
    n_rollouts: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    tensor_parallel_size: int = DEFAULT_TP_72B,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN_72B,
    gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL_72B,
) -> dict[str, object]:
    """Run the full 24-panel x 50-claim x N-rollout eval for one 72B source.

    Three load shapes (plan v2 section 4.3 + section 8 vLLM fallback):

    1. base-only (adapter_path=None, merged_model_path=None):
       Load base 72B at TP=8 for the base-panel baseline pass.
       Sets hub_model_id=base_model_id, no LoRA.
    2. merged checkpoint (merged_model_path set):
       Load a merge of base + LoRA at TP=8 (the safe #411 path that
       avoids vLLM's TP-multi-LoRA edge cases).
    3. native multi-LoRA (adapter_path set, merged_model_path=None):
       Load base 72B once at TP=8 with --enable-lora; route requests to
       the LoRA adapter via vLLM's LoRARequest channel.

    Args:
        source: Source persona slug (one of SOURCE_PERSONAS_507).
        seed: Eval seed (matches training seed for parity reporting).
        merged_model_path: Local merge of base + LoRA. Mutually exclusive
            with adapter_path; one of (merged_model_path, adapter_path)
            must be set, OR both can be None for the base-only pass.
        base_model_id: HF model id when running base-only OR when loading
            base + native LoRA (mode 3). Default Qwen/Qwen2.5-72B-Instruct.
        adapter_path: Local LoRA adapter dir. When set, vLLM loads base
            at TP=8 with --enable-lora and registers the adapter.
        eval_pool_path: Path to eval_50.jsonl (#411 held-out probes).
        out_dir: Per-source eval output dir (per-panel JSONs + sentinel).
        n_rollouts: Rollouts per (panel, claim) pair (default 10 matches #411).
        max_new_tokens: Generation cap (default 128 matches #411 — short
            sycophancy responses, well under the >=2x trained-completion rule).
        temperature: Sampling temperature (default 1.0 matches #411).
        tensor_parallel_size: vLLM TP shards (default 8 for 72B on 8xH100).
        max_model_len: vLLM max_model_len (default 2048 — see module docstring).
        gpu_memory_utilization: vLLM GPU memory cap (default 0.90 for 72B).

    Returns:
        eval_source's standard summary dict (per-panel completions + metadata).

    Raises:
        ValueError: more than one model-source mode is set simultaneously.
    """
    n_modes = sum(
        [
            merged_model_path is not None,
            adapter_path is not None,
        ]
    )
    if n_modes > 1:
        raise ValueError(
            f"eval_72b: pick AT MOST one of merged_model_path / adapter_path; "
            f"got merged_model_path={merged_model_path}, "
            f"adapter_path={adapter_path}. (Base-only base-panel pass: pass "
            f"neither, set base_model_id only.)"
        )

    # Compose the eval_source kwargs based on mode.
    if adapter_path is not None:
        # Mode 3: base + native LoRA via vLLM's --enable-lora.
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"adapter_path {adapter_path} does not exist. The dispatcher "
                f"must run train_72b for this (source, seed) before eval_72b."
            )
        lora_name = f"{source}_seed{seed}"
        log.info(
            "[%s] eval_72b mode=native-lora: base=%s, adapter=%s -> %s",
            source,
            base_model_id,
            adapter_path,
            out_dir,
        )
        return eval_source(
            source=source,
            seed=seed,
            merged_model_path=None,
            eval_pool_path=eval_pool_path,
            out_dir=out_dir,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            hub_model_id=base_model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_lora=True,
            lora_modules=[f"{lora_name}={adapter_path}"],
            max_lora_rank=DEFAULT_MAX_LORA_RANK,
        )
    elif merged_model_path is not None:
        # Mode 2: pre-merged base+LoRA on disk (safe TP=8 path).
        log.info(
            "[%s] eval_72b mode=merged: merged=%s -> %s",
            source,
            merged_model_path,
            out_dir,
        )
        return eval_source(
            source=source,
            seed=seed,
            merged_model_path=merged_model_path,
            eval_pool_path=eval_pool_path,
            out_dir=out_dir,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            hub_model_id=None,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
    else:
        # Mode 1: base-only pass (Phase 1.5 base-panel baseline).
        log.info(
            "[%s] eval_72b mode=base-only: base=%s -> %s",
            source,
            base_model_id,
            out_dir,
        )
        return eval_source(
            source=source,
            seed=seed,
            merged_model_path=None,
            eval_pool_path=eval_pool_path,
            out_dir=out_dir,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            hub_model_id=base_model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the subprocess-isolated 72B eval (#399 vLLM teardown).

    Mirrors eval_one_source's CLI but with --tensor-parallel-size, --max-model-len,
    --adapter-path, and --base-model-id defaults for the 72B path.
    """
    import argparse
    import json
    from datetime import UTC, datetime

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--merged-model-path",
        type=Path,
        default=None,
        help="Local merge of base+LoRA. Mutually exclusive with --adapter-path.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Local LoRA adapter dir for native vLLM multi-LoRA path.",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=BASE_MODEL_72B,
        help=f"HF model id for base + native LoRA path (default {BASE_MODEL_72B})",
    )
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP_72B)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN_72B)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEM_UTIL_72B,
    )
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=None,
        help="Sentinel file for the orchestrator's poll_pipeline.py.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=eval_72b] %(message)s")

    summary = eval_72b(
        source=args.source,
        seed=args.seed,
        merged_model_path=args.merged_model_path,
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        eval_pool_path=args.eval_pool,
        out_dir=args.out_dir,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sentinel_path = args.sentinel_path or Path(
        f"/workspace/logs/issue-507-{args.source}-eval-results.json"
    )
    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel = {
            "source": args.source,
            "seed": args.seed,
            "phase": "eval_72b_complete",
            "n_panel_jsons": summary.get("n_panel_personas"),
            "n_completions": summary.get("total_completions"),
            "wall_seconds": summary.get("wall_seconds"),
            "model_loaded": summary.get("model_loaded"),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        sentinel_path.write_text(json.dumps(sentinel, indent=2))
        log.info("Wrote eval sentinel to %s", sentinel_path)
    except OSError as exc:
        # /workspace/logs is pod-only; fail soft on local smoke runs.
        log.warning(
            "Could not write sentinel to %s (%s). Acceptable off-pod.",
            sentinel_path,
            exc,
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
