"""Task #507 Phase 3 - 72B base-model predictor extraction.

Drives the 6 phases of ``predictor_jsdiv_470`` with model id =
``Qwen/Qwen2.5-72B-Instruct`` and the 72B layer set {21, 40, 57, 70}. The
predictor recipe is identical to #470's; the only changes are model id +
layer ints. We do NOT cherry-edit the #470 phase scripts; we drive them via
their CLI ``--model`` and ``--layers`` overrides.

Phase 3 PREFLIGHT (fail-loud, runs before the 138-cell sweep per plan v2 4.5):

  1. Load base 72B with ``device_map="auto"``; assert hf_device_map values
     are all cuda devices (NO ``cpu`` / ``disk`` entries).
  2. nvidia-smi check: every GPU shows non-trivial memory used (residency
     proves no host offload).
  3. Dummy batch-2 teacher-force forward; assert <60s + no NaN.
  4. JS sanity: self-vs-self ~ 0; self-vs-uniform <= ln(2) + tolerance.

Any preflight failure surfaces via SystemExit non-zero so the dispatcher's
``check=True`` subprocess.run propagates the failure to the orchestrator.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

# Set HF_HOME BEFORE any HF imports (the 72B 145GB weight load must use the
# persistent volume, never the per-pod writable layer with ~130GB MooseFS
# quota per gotchas.md).
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from explore_persona_space.experiments.sycophancy_scale_507 import (
    HEADLINE_LAYER_BY_ARCH,
    LAYER_SET_BY_ARCH,
)

log = logging.getLogger("sycophancy_scale_507.predictor_72b")

BASE_MODEL_72B = "Qwen/Qwen2.5-72B-Instruct"
LAYERS_72B: tuple[int, ...] = LAYER_SET_BY_ARCH["72b"]
HEADLINE_LAYER_72B: int = HEADLINE_LAYER_BY_ARCH["72b"]


def _js_against_uniform(model, batch_inputs, prompt_lengths) -> float:
    """Return JS(P, U) where P is the model's response-token distribution at row 0
    and U is the uniform distribution over the vocabulary, in nats.

    Used by the preflight sanity check: a numerically-correct estimator must
    produce a value in (0, ln(2)] (the JS upper bound). A bug at V=152064
    with bf16 → fp32 cast could push this over ln(2) (numerical overflow) or
    to a trivial floor. Round-2 fix per code-review Codex minor item 12.
    """
    import torch

    from explore_persona_space.analysis.divergence import (
        compute_js_divergence,
        teacher_force_batch,
    )

    with torch.no_grad():
        # Reuse the same teacher-force pass that the self-self check already
        # ran to get log-probs at the response positions for row 0.
        log_probs = teacher_force_batch(
            model=model,
            batch_inputs=batch_inputs,
            prompt_lengths=prompt_lengths,
            response_len=1,  # only need the first response position for the bound check
            device="cuda:0",
            max_batch=1,
        )
        # log_probs shape: (batch, response_len, vocab) — squeeze to first row + first pos.
        log_p = log_probs[0:1, 0, :]  # (1, V) on CPU per teacher_force_batch contract
        V = log_p.shape[-1]
        uniform = torch.full((1, V), 1.0 / V, dtype=torch.float32)
        log_u = torch.log(uniform)
        js = compute_js_divergence(log_p, log_u)
    return float(js.item() if torch.is_tensor(js) else js)


def preflight_no_offload(
    *,
    model_id: str = BASE_MODEL_72B,
    dummy_max_new: int = 64,
    max_wall_seconds: float = 60.0,
) -> dict[str, object]:
    """Run the Phase 3 preflight: load 72B, check no CPU offload, dummy forward.

    Plan v2 section 4.5 lists 4 preflight checks; this function runs all 4
    and returns a summary dict. Raises a clear RuntimeError on any failure
    (no fallback / no offload — fail-loud per the contract).

    Args:
        model_id: HF model id to load. Default Qwen/Qwen2.5-72B-Instruct.
        dummy_max_new: Length of the dummy response for the batch-2 forward.
        max_wall_seconds: Fail-loud if the dummy forward takes longer than
            this many seconds. Catches CPU offload that wasn't caught by
            hf_device_map inspection.

    Returns:
        Summary dict with keys: hf_device_map, gpu_memory_used_per_gpu_mb,
        dummy_forward_seconds, js_self_self, js_self_uniform.

    Raises:
        RuntimeError: any of the 4 checks failed.
    """
    import subprocess
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(
        "preflight_no_offload: loading %s with device_map=auto, dtype=bf16",
        model_id,
    )
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("preflight: model loaded in %.1fs", time.time() - t_load)

    # Check 1+2: hf_device_map has no CPU / disk entries.
    device_map_raw = getattr(model, "hf_device_map", None)
    if device_map_raw is None:
        raise RuntimeError(
            "model.hf_device_map is None after device_map=auto load. The "
            "load shape is not what we expected; cannot verify no-offload."
        )
    # Stringify devices so cuda:0, torch.device('cuda', 0), and 0 all compare.
    device_map: dict[str, str] = {str(k): str(v) for k, v in device_map_raw.items()}
    bad_entries = [
        (mod_name, dev) for mod_name, dev in device_map.items() if "cpu" in dev or "disk" in dev
    ]
    if bad_entries:
        sample = bad_entries[:5]
        raise RuntimeError(
            f"preflight: hf_device_map contains CPU/disk offload entries "
            f"(first 5 of {len(bad_entries)}): {sample}. "
            f"Cannot proceed — CPU offload would make Phase 3 teacher-force "
            f"forwards ~50x slower than the 5h projection."
        )
    log.info(
        "preflight: hf_device_map check PASS (%d modules, all on cuda)",
        len(device_map),
    )

    # Check 2 cont.: nvidia-smi per-GPU residency. Per-GPU memory.used must
    # be > 1 GB (rough threshold: a base 72B sharded across 2 GPUs lands
    # ~73 GB/GPU; sharded across 8 GPUs lands ~18 GB/GPU; either way well
    # above 1 GB. An empty GPU under device_map=auto means the model
    # actually lives on CPU.).
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            env={**os.environ},
        )
        per_gpu_mb = [int(x.strip()) for x in smi.splitlines() if x.strip()]
        log.info("preflight: per-GPU memory used (MB): %s", per_gpu_mb)
        cuda_devices_in_map = sorted({dev for dev in device_map.values() if dev.startswith("cuda")})
        for gpu_idx, mem_mb in enumerate(per_gpu_mb):
            dev_str = f"cuda:{gpu_idx}"
            if dev_str in cuda_devices_in_map and mem_mb < 1024:
                raise RuntimeError(
                    f"preflight: GPU {gpu_idx} shows only {mem_mb} MB used "
                    f"despite hf_device_map placing modules on it. The "
                    f"model is likely host-resident; CPU offload detected."
                )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning(
            "preflight: nvidia-smi check skipped (%s). hf_device_map check is "
            "the authoritative signal; proceeding.",
            exc,
        )
        per_gpu_mb = []

    # Check 3: dummy batch-2 teacher-force forward.
    # The divergence helper imports lazily so it picks up the model's device
    # from device_map=auto rather than forcing a single device.
    from explore_persona_space.analysis.divergence import (
        build_teacher_force_inputs,
        teacher_force_and_reduce_js_kl,
    )

    dummy_system_prompts = [
        "You are a helpful assistant.",
        "You are a software engineer who answers questions.",
    ]
    dummy_question = "What is the capital of France?"
    dummy_response = "Paris is the capital of France. " * 4  # short, deterministic

    batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
        tokenizer=tokenizer,
        system_prompts=dummy_system_prompts,
        question=dummy_question,
        response_text=dummy_response[: dummy_max_new * 6],  # roughly cap response token count
    )
    log.info(
        "preflight: dummy batch-2 forward (prompt_lens=%s, response_len=%d)",
        prompt_lengths,
        response_len,
    )
    t_forward = time.time()
    js_sym, kl_pq, _kl_qp = teacher_force_and_reduce_js_kl(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=prompt_lengths,
        response_len=response_len,
        device="cuda:0",
        max_batch=2,
        p_index=0,
        q_index=1,
    )
    forward_wall = time.time() - t_forward
    log.info(
        "preflight: dummy forward wall=%.1fs, JS=%.4f, KL=%.4f",
        forward_wall,
        js_sym,
        kl_pq,
    )
    if forward_wall > max_wall_seconds:
        raise RuntimeError(
            f"preflight: dummy batch-2 forward took {forward_wall:.1f}s "
            f"(> {max_wall_seconds:.1f}s threshold). CPU offload or "
            f"unexpected device sharding likely; 138-cell sweep would "
            f"overshoot the 5h Phase 3 budget."
        )
    if math.isnan(js_sym) or math.isnan(kl_pq):
        raise RuntimeError(
            f"preflight: dummy forward produced NaN (JS={js_sym}, KL={kl_pq}). "
            f"Numerical instability at 72B + bf16; cannot proceed."
        )

    # Check 4: JS sanity. Construct same-prompt-vs-same-prompt; JS must be ~0.
    same_prompts = [dummy_system_prompts[0], dummy_system_prompts[0]]
    batch_same, prompt_lengths_same, response_len_same = build_teacher_force_inputs(
        tokenizer=tokenizer,
        system_prompts=same_prompts,
        question=dummy_question,
        response_text=dummy_response[: dummy_max_new * 6],
    )
    js_self, _kl_self, _kl_self_qp = teacher_force_and_reduce_js_kl(
        model=model,
        batch_inputs=batch_same,
        prompt_lengths=prompt_lengths_same,
        response_len=response_len_same,
        device="cuda:0",
        max_batch=2,
        p_index=0,
        q_index=1,
    )
    log.info("preflight: JS(self, self) = %.6f (expect ~0)", js_self)
    if js_self > 1e-3:
        raise RuntimeError(
            f"preflight: JS(self, self) = {js_self:.6f}, expected ~0. "
            f"compute_js_divergence is mis-computing; refuse to proceed."
        )

    # Round-2 fix per code-review Codex minor (item 12): the self-self
    # check pins the LOWER bound (~0) but says nothing about whether the
    # estimator's UPPER bound is sensible. JS(P, Q) is bounded above by
    # ln(2) ≈ 0.693 nats; a numerical-stability bug at V=152064 with the
    # bf16 → fp32 cast could push the result OVER ln(2) (overflow,
    # incorrect normalization) or to a numerically-trivial floor that
    # only the self-self check would let through. Compute JS(P, uniform)
    # on the model's first-token distribution and assert it's in
    # (0, ln(2) + 1e-3]. The +1e-3 slack tolerates fp32-mid-bf16-cast
    # noise; values > ln(2)+1e-3 indicate a real numerical bug.
    js_p_uniform = _js_against_uniform(model, batch_same, prompt_lengths_same)
    ln2 = math.log(2)
    log.info(
        "preflight: JS(self_first_token, uniform) = %.6f (expected in (0, %.6f])",
        js_p_uniform,
        ln2,
    )
    if not (0.0 < js_p_uniform <= ln2 + 1e-3):
        raise RuntimeError(
            f"preflight: JS(self, uniform) = {js_p_uniform:.6f} outside "
            f"(0, ln(2)={ln2:.6f}] — likely numerical stability bug in "
            f"compute_js_divergence at V={tokenizer.vocab_size}; refuse to proceed."
        )

    summary: dict[str, object] = {
        "model_id": model_id,
        "hf_device_map_unique_devices": sorted(set(device_map.values())),
        "n_modules_in_device_map": len(device_map),
        "per_gpu_memory_used_mb": per_gpu_mb,
        "dummy_forward_seconds": round(forward_wall, 2),
        "dummy_js_sym": round(float(js_sym), 6),
        "dummy_kl_pq": round(float(kl_pq), 6),
        "js_self_self": round(float(js_self), 6),
        "js_self_uniform": round(float(js_p_uniform), 6),
        "preflight_pass": True,
    }
    log.info("preflight_no_offload: ALL CHECKS PASS — summary=%s", summary)

    # Free the model + clear cache before the dispatcher continues; the
    # dispatcher will re-load via the phase3 subprocess on a fresh process
    # anyway, but this keeps the preflight clean.
    del model
    import contextlib
    import gc

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry — runs preflight only (the actual 138-cell sweep is driven
    by the dispatcher invoking phase1/2/3 modules with --model + --layers)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=["preflight"],
        default="preflight",
        help="Which step of the 72B predictor pipeline to run (only "
        "preflight is owned by this wrapper; phase1/2/3 are owned by the "
        "predictor_jsdiv_470 module + the top-level dispatcher).",
    )
    parser.add_argument("--model", default=BASE_MODEL_72B)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write preflight summary JSON.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=predictor_72b] %(message)s")

    if args.mode == "preflight":
        summary = preflight_no_offload(model_id=args.model)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2))
            log.info("Wrote preflight summary to %s", args.output)
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
