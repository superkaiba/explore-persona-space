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


class _EffectiveBatchWandbCallback:
    """HF TrainerCallback that logs effective_batch_size to WandB on_train_begin.

    Round-2 fix per code-review Major 10 (#507 r1): the round-1 inline
    ``_log_effective_batch_to_wandb`` call ran BEFORE ``train_lora`` invoked
    ``wandb.init``, so ``wandb.run is None`` was always True and the metric
    was never sent. A TrainerCallback hooks on_train_begin (fires AFTER
    wandb.init), guaranteeing the summary is set on the active run.

    Falls back to a no-op warning if wandb is not importable or wandb.run
    is somehow still None at on_train_begin (defensive; should never happen).
    """

    def __init__(self, *, world_size: int, grad_accum: int, run_name: str) -> None:
        self.world_size = world_size
        self.grad_accum = grad_accum
        self.run_name = run_name
        self.effective_batch = world_size * PER_DEVICE_TRAIN_BATCH_72B * grad_accum

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
        except ImportError:
            log.warning(
                "wandb not importable in on_train_begin; effective_batch_size=%d not logged.",
                self.effective_batch,
            )
            return
        if wandb.run is None:
            log.warning(
                "wandb.run still None at on_train_begin for %s; "
                "effective_batch_size=%d not logged. Check report_to=wandb wiring.",
                self.run_name,
                self.effective_batch,
            )
            return
        wandb.run.summary["effective_batch_size"] = self.effective_batch
        wandb.run.summary["world_size"] = self.world_size
        wandb.run.summary["per_device_train_batch"] = PER_DEVICE_TRAIN_BATCH_72B
        wandb.run.summary["gradient_accumulation_steps"] = self.grad_accum
        log.info(
            "[%s] Logged effective_batch_size=%d to WandB summary on_train_begin",
            self.run_name,
            self.effective_batch,
        )

    # Required no-op hooks for HF TrainerCallback duck-typing. HF Trainer
    # calls on_init_end, on_step_begin, on_step_end, etc; returning None
    # leaves control flow alone.
    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **kw: None
        raise AttributeError(name)


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

    # Round-2 fix per code-review Critical 3: when world_size > 1 we are
    # running under the deepspeed launcher; pass is_distributed=True so
    # train_lora skips CUDA_VISIBLE_DEVICES + device_map={"": 0} clobbers
    # and DeepSpeed ZeRO-3 owns shard placement across ranks.
    # Round-3 fix per code-review Critical 2: also pass deepspeed=<path>
    # so SFTConfig (= TrainingArguments) receives it directly. HF Trainer
    # reads TrainingArguments.deepspeed (a path or dict), NOT the
    # DEEPSPEED_CONFIG_FILE env var; setting the env var alone was a no-op,
    # which is why round-2's distributed run still materialized the full
    # 72B per rank and OOMed.
    is_distributed = world_size > 1
    deepspeed_arg: str | None = deepspeed_cfg if world_size > 1 else None
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
        # gpu_id is bypassed entirely under the distributed path (the launcher
        # sets per-rank CUDA_VISIBLE_DEVICES; train_lora honors that). Keep
        # 0 as a placeholder for the rare single-GPU debug path.
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
        is_distributed=is_distributed,
        # Round-3 fix per code-review Critical 2: thread the DeepSpeed
        # config to SFTConfig.deepspeed (= TrainingArguments.deepspeed).
        # HF Trainer installs HfDeepSpeedConfig from this path so
        # AutoModelForCausalLM.from_pretrained partitions weights via
        # ZeRO-3 at load time instead of materializing the full 145 GB bf16
        # model per rank. None on the single-GPU debug path preserves the
        # legacy in-process behavior.
        deepspeed=deepspeed_arg,
    )

    if world_size > 1:
        # ACCELERATE_USE_DEEPSPEED is still useful for accelerate-stack
        # interop; setdefault preserves operator overrides. The
        # DEEPSPEED_CONFIG_FILE env var was previously the sole transport
        # of the config — round-3 makes the SFTConfig.deepspeed field the
        # authoritative channel, but we keep the env var as a belt-and-
        # suspenders breadcrumb for debugging / accelerate config detection.
        os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "true")
        os.environ.setdefault("DEEPSPEED_CONFIG_FILE", deepspeed_cfg)
        log.info(
            "[%s] Threading deepspeed=%s into TrainLoraConfig + setting "
            "ACCELERATE_USE_DEEPSPEED=true for multi-GPU ZeRO-3 launch.",
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

    # Round-2 fix per code-review Major 10: route effective_batch_size
    # logging through a TrainerCallback that fires AFTER wandb.init in
    # train_lora, not via an inline call before train_lora runs (which
    # caught wandb.run=None every time and logged nothing).
    eff_batch_cb = _EffectiveBatchWandbCallback(
        world_size=world_size, grad_accum=grad_accum, run_name=run_name
    )

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
        callbacks=[eff_batch_cb],
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
