"""Task #507 Phase 1 - Qwen-2.5-72B LoRA SFT (thin wrapper around train_lora).

Single function ``train_72b(source, seed, train_jsonl, output_dir)`` calls into
``explore_persona_space.train.sft.train_lora`` with #411-verbatim hparams and a
72B-specific grad_accum chosen at runtime from the live world_size to preserve
effective batch 16:

    4xH200 default:   world_size=4 -> grad_accum=4 -> eff_batch = 4*1*4 = 16
    8xH100 fallback:  world_size=8 -> grad_accum=2 -> eff_batch = 8*1*2 = 16

A runtime assertion fires at training start: ``world_size * per_device_batch *
grad_accum == 16``. A mismatch is fail-loud because the rig would silently train
under a different recipe than #411 and break the single-variable contract.

Effective batch is also logged as a WandB summary metric so the dashboard
cross-check is one click.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from explore_persona_space.experiments.sycophancy_scale_507 import (
    EXPECTED_EFFECTIVE_BATCH,
    PER_DEVICE_TRAIN_BATCH_72B,
    compute_grad_accum,
    get_world_size_from_env,
)

log = logging.getLogger("sycophancy_scale_507.train_72b")

BASE_MODEL_72B = "Qwen/Qwen2.5-72B-Instruct"
HF_REPO = "superkaiba1/explore-persona-space"
# Path under HF model repo; matches #411's "adapters/issue_<N>/<src>_seed<S>"
# convention so the dispatcher's eval can locate the adapter by HF path.
HF_PATH_PREFIX = "adapters/issue_507/72b"
# ZeRO-3 no-offload config (validated for 7B full-FT at #356; smoke-tested
# at 72B per plan v2 section 4.3).
DEEPSPEED_ZERO3_CONFIG = "configs/deepspeed/zero3_no_offloading.json"


def _resolve_deepspeed_path() -> str:
    """Return an absolute path to configs/deepspeed/zero3_no_offloading.json.

    Walks up from this file to the project root (the directory with both
    pyproject.toml and src/). Fail-loud if the config is missing — every
    72B training cell needs it.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            cfg = parent / DEEPSPEED_ZERO3_CONFIG
            if not cfg.exists():
                raise RuntimeError(
                    f"DeepSpeed config not found at {cfg}. Required for 72B "
                    f"LoRA ZeRO-3 training; cannot proceed."
                )
            return str(cfg)
    raise RuntimeError(
        f"Could not find project root walking up from {here}; cannot resolve "
        f"{DEEPSPEED_ZERO3_CONFIG}"
    )


def _assert_effective_batch(world_size: int, per_device_batch: int, grad_accum: int) -> None:
    """Fail-loud if the chosen (world_size, per_device_batch, grad_accum) breaks #411 parity.

    The single experimental variable vs #411 is model size; the effective batch
    MUST match #411's 16 on either pod path. Mismatch surfaces as a clear error
    so the rig never silently trains under a different recipe.
    """
    eff = world_size * per_device_batch * grad_accum
    if eff != EXPECTED_EFFECTIVE_BATCH:
        raise AssertionError(
            f"effective_batch={eff} (world_size={world_size} * "
            f"per_device_batch={per_device_batch} * grad_accum={grad_accum}), "
            f"expected {EXPECTED_EFFECTIVE_BATCH} (matches #411). The single-"
            f"variable contract requires effective batch parity with the 7B "
            f"parent; cannot proceed."
        )


def _log_effective_batch_to_wandb(run_name: str, world_size: int, grad_accum: int) -> None:
    """Set wandb.summary['effective_batch_size'] so the dashboard exposes it.

    The metric is part of the smoke-gate (plan v2 section 7 condition 1): the
    run's WandB summary must report effective_batch_size==16 before cells 2-6
    fire. Loud-warn if WandB isn't reachable (best-effort; the runtime
    assertion above is the hard safeguard).
    """
    eff = world_size * PER_DEVICE_TRAIN_BATCH_72B * grad_accum
    try:
        import wandb

        if wandb.run is None:
            # No active run; nothing to attach to. The dispatcher's run will
            # pick this up when train_lora initializes its own WandB run.
            log.warning(
                "WandB run not yet initialized when train_72b logged "
                "effective_batch_size=%d; relying on train_lora to attach.",
                eff,
            )
            return
        wandb.run.summary["effective_batch_size"] = eff
        wandb.run.summary["world_size"] = world_size
        wandb.run.summary["per_device_train_batch"] = PER_DEVICE_TRAIN_BATCH_72B
        wandb.run.summary["gradient_accumulation_steps"] = grad_accum
        log.info(
            "Logged effective_batch_size=%d to WandB summary for run %s",
            eff,
            run_name,
        )
    except ImportError:
        log.warning(
            "wandb not importable; effective_batch_size=%d not logged to dashboard.",
            eff,
        )


def train_72b(
    *,
    source: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    world_size: int | None = None,
    hf_upload: bool = True,
) -> tuple[Path, Path]:
    """Train one 72B LoRA cell with #411-verbatim hparams + ZeRO-3 sharding.

    Args:
        source: Source persona name (one of SOURCE_PERSONAS_507).
        seed: Training seed (#411 / #470 parity = 42).
        train_jsonl: Path to the 700-row contrastive training pool.
        output_dir: Directory to write the adapter into. The dispatcher
            convention is ``<runs_root>/<source>_seed<seed>/``; the adapter
            lands at ``<output_dir>/adapter/``.
        world_size: Override world_size detection (default: read from env /
            torch.distributed). Pass explicit value in CPU smoke tests where
            the launcher hasn't been invoked.
        hf_upload: Whether train_lora should HF-upload the adapter after
            training. The dispatcher wants True so the smoke gate can verify
            via HF Hub list_repo_files; tests pass False.

    Returns:
        (adapter_dir, merged_dir_placeholder) — adapter_dir is where the
        LoRA safetensors live; merged_dir_placeholder is reserved for the
        downstream eval phase's merge step (returned even when merge has
        not run yet so the dispatcher signature matches the #411 pattern).

    Raises:
        AssertionError: effective batch does not equal 16 (single-variable
            contract violation).
        RuntimeError: DeepSpeed config not found / world_size cannot
            cleanly divide 16.
    """
    # train_lora is heavy (imports torch + transformers + peft); deferred so
    # `import train_72b` stays cheap for unit tests.
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    if world_size is None:
        world_size = get_world_size_from_env()
    grad_accum = compute_grad_accum(world_size, PER_DEVICE_TRAIN_BATCH_72B)
    _assert_effective_batch(world_size, PER_DEVICE_TRAIN_BATCH_72B, grad_accum)

    deepspeed_cfg = _resolve_deepspeed_path()
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir_placeholder = output_dir / "merged"  # populated by downstream merge step

    run_name = f"issue507_72b_{source}_seed{seed}"
    log.info(
        "[%s] train_72b start: world_size=%d, per_device_batch=%d, "
        "grad_accum=%d, effective_batch=%d, deepspeed=%s",
        source,
        world_size,
        PER_DEVICE_TRAIN_BATCH_72B,
        grad_accum,
        world_size * PER_DEVICE_TRAIN_BATCH_72B * grad_accum,
        deepspeed_cfg,
    )

    cfg = TrainLoraConfig(
        # #411-verbatim hparams (plan v2 section 4.1).
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        max_length=1024,
        warmup_ratio=0.05,
        weight_decay=0.0,
        seed=seed,
        gradient_checkpointing=True,
        packing=False,
        save_strategy="no",
        # 72B-specific knobs (plan v2 section 4.3).
        batch_size=PER_DEVICE_TRAIN_BATCH_72B,
        grad_accum=grad_accum,
        # gpu_id is irrelevant under ZeRO-3 multi-GPU launch — the launcher
        # (torchrun/deepspeed/accelerate) sets CUDA_VISIBLE_DEVICES per rank;
        # train_lora's os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        # would clobber that under the multi-GPU path, so we pass 0 only as
        # a placeholder; the launcher's env takes precedence in practice.
        # See plan v2 section 4.3 + gotchas.md "+gpu_id Hydra override" note.
        gpu_id=0,
        # WandB live training metrics ON (CLAUDE.md mandate; no
        # report_to="none" waiver applies).
        report_to="wandb",
        run_name=run_name,
        # HF upload of adapter after training; the smoke gate verifies via
        # HF Hub list_repo_files (plan v2 section 7 condition 1).
        hf_upload=hf_upload,
        hf_repo=HF_REPO,
        hf_path_in_repo=f"{HF_PATH_PREFIX}/{source}_seed{seed}",
    )

    # Thread the DeepSpeed config through as an override so TrainLoraConfig
    # doesn't grow a new field (additive on the call site, not on the dataclass).
    # train_lora's **overrides hook passes unknown kwargs through to HF
    # SFTConfig/TrainingArguments via its own forward (see train/sft.py:618).
    # NB: TrainLoraConfig doesn't currently expose a deepspeed field, so we
    # set DEEPSPEED env vars instead — HF Trainer auto-picks them up when
    # spawned under deepspeed launcher. Document loudly.
    if world_size > 1:
        os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "true")
        os.environ.setdefault("DEEPSPEED_CONFIG_FILE", deepspeed_cfg)
        log.info(
            "[%s] Set ACCELERATE_USE_DEEPSPEED=true, DEEPSPEED_CONFIG_FILE=%s "
            "for multi-GPU ZeRO-3 launch.",
            source,
            deepspeed_cfg,
        )
    else:
        log.warning(
            "[%s] world_size=1; ZeRO-3 sharding skipped. This is acceptable "
            "ONLY for CPU smoke tests or single-GPU debug; production 72B "
            "requires world_size>=4 (see plan v2 section 9.5).",
            source,
        )

    # Log effective_batch_size as a WandB summary metric BEFORE train_lora
    # starts the actual training loop, so the smoke gate can inspect the
    # dashboard even if training fails mid-epoch.
    _log_effective_batch_to_wandb(run_name, world_size, grad_accum)

    log.info(
        "[%s] Calling train_lora(base=%s, data=%s, out=%s, run_name=%s)",
        source,
        BASE_MODEL_72B,
        train_jsonl,
        adapter_dir,
        run_name,
    )
    train_lora(
        base_model_path=BASE_MODEL_72B,
        data_path=str(train_jsonl),
        output_dir=str(adapter_dir),
        cfg=cfg,
    )

    # Loud-fail if the adapter isn't on disk after training: the smoke gate
    # checks HF Hub but a missing local safetensors means the HF upload had
    # nothing to upload either.
    safetensors = list(adapter_dir.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(
            f"[{source}] train_72b: adapter dir {adapter_dir} has no "
            f".safetensors files after training. Either training silently "
            f"failed or the save path was redirected."
        )
    log.info(
        "[%s] train_72b done: adapter at %s (%d safetensors files)",
        source,
        adapter_dir,
        len(safetensors),
    )
    return adapter_dir, merged_dir_placeholder
