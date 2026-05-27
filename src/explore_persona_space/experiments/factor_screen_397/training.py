"""Per-cell training dispatch with ordinal-E → TrainLoraConfig mapping (task #397).

This module is the v4-plan-approved successor to
``factor_screen_365.training``. Key deltas:

- Ordinal E ∈ {0, 1, 2} dispatch:
    - E0: ``marker_only_loss=True, marker_tail_tokens=0``  (~2 tok)
    - E1: ``marker_only_loss=True, marker_tail_tokens=32`` (~32 tok)
    - E2: ``marker_only_loss=False, marker_tail_tokens=0`` (whole completion)
- Recipe upgraded to #399's shipped hyperparameters: ``lr=1e-4``,
  ``warmup_ratio=0.10``, ``lr_scheduler_type="cosine"``, ``optim="adamw_torch"``,
  ``max_seq_length=2048``, ``lora_target_modules`` covering all attn + MLP.
- Intermediate checkpoints saved every 25 steps (6 checkpoints per ~150-step
  run) so the log-prob eval can sample the trajectory.

See ``tasks/<status>/397/plans/v4.md`` §5.6 for the canonical signature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .cells import Cell

log = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


# Plan v4 §5.6 + §8 Reproducibility Card — default hyperparameters explicitly
# transferred from #399's shipped recipe at the v4 plan-approval gate.
DEFAULT_LR: float = 1e-4
DEFAULT_WARMUP_RATIO: float = 0.10
DEFAULT_LR_SCHEDULER_TYPE: str = "cosine"
# Plan v4 §5.6 says "AdamW (`optim='adamw_torch'`) — TRL default". TRL's
# actual CUDA default is ``adamw_torch_fused`` (~10-15% faster on H100), and
# #399 shipped with the fused variant. v4 inherits #399's recipe, so we use
# the fused form here. The plan-card text references plain ``adamw_torch``
# by name but the intent ("TRL default") matches the fused form on CUDA.
DEFAULT_OPTIM: str = "adamw_torch_fused"
DEFAULT_MAX_SEQ_LENGTH: int = 2048
DEFAULT_SEEDS: tuple[int, int, int] = (42, 137, 256)
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # attention
    "gate_proj",
    "up_proj",
    "down_proj",  # MLP
)
DEFAULT_SAVE_EVERY_N_STEPS: int = 25
DEFAULT_MARKER_TEXT: str = "※"


@dataclass(frozen=True)
class EDispatch:
    """Ordinal-E level → (marker_only_loss, marker_tail_tokens) mapping.

    The structured return type lets the test surface assert each E level lands
    on the right ``TrainLoraConfig`` fields without instantiating TRL.
    """

    e_level: int
    marker_only_loss: bool
    marker_tail_tokens: int


def dispatch_e_level(e: int) -> EDispatch:
    """Map ordinal E ∈ {0, 1, 2} to the matching TrainLoraConfig loss-mask kwargs.

    Plan v4 §4.3 + §5.6:
      - E0 → marker_only_loss=True, marker_tail_tokens=0   (~2 tok)
      - E1 → marker_only_loss=True, marker_tail_tokens=32  (~32 tok)
      - E2 → marker_only_loss=False, marker_tail_tokens=0  (whole completion)
    """
    if e == 0:
        return EDispatch(e_level=0, marker_only_loss=True, marker_tail_tokens=0)
    if e == 1:
        return EDispatch(e_level=1, marker_only_loss=True, marker_tail_tokens=32)
    if e == 2:
        return EDispatch(e_level=2, marker_only_loss=False, marker_tail_tokens=0)
    raise ValueError(f"E must be 0, 1, or 2; got {e!r}")


@dataclass
class TrainOutcome:
    cell_key: str
    seed: int
    adapter_path: str
    merged_path: str
    loss: float
    train_wall_minutes: float
    n_examples: int
    total_steps: int
    marker_only_loss: bool
    marker_tail_tokens: int


def train_one_cell(
    *,
    cell: Cell,
    seed: int,
    source: str,
    data_path: Path,
    cell_output_dir: Path,
    marker_text: str = DEFAULT_MARKER_TEXT,
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    lr: float = DEFAULT_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    lr_scheduler_type: str = DEFAULT_LR_SCHEDULER_TYPE,
    optim: str = DEFAULT_OPTIM,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: tuple[str, ...] | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    gpu_id: int = 0,
    wandb_project: str | None = None,
    hf_upload: bool = True,
) -> TrainOutcome:
    """Train one (cell, seed) run with the v4 recipe.

    Plan v4 §5.6 — Wires the ordinal-E dispatch + #399 hyperparameters
    through ``TrainLoraConfig`` + ``train_lora``. The merged final-checkpoint
    model lives at ``cell_output_dir / 'merged'`` so vLLM can load it for the
    final-checkpoint sampled eval; intermediate checkpoints live at
    ``cell_output_dir / 'adapter' / 'checkpoint-<step>'`` and are consumed
    in-place by ``compute_logprob_panel`` via the peft 0.18.1 adapter-swap
    lifecycle.
    """
    # Imports kept local so the module can be collected on CPU-only test runs
    # without dragging in torch / TRL just to inspect ``train_one_cell``'s
    # signature. ``math`` / ``os`` / ``time`` are also inlined because ruff
    # auto-strips top-level imports with no module-level reference (memory:
    # feedback_ruff_strips_unused_imports).
    import math
    import os
    import time

    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    if lora_target_modules is None:
        lora_target_modules = DEFAULT_LORA_TARGET_MODULES

    e_dispatch = dispatch_e_level(cell.e)

    adapter_dir = cell_output_dir / "adapter"
    merged_dir = cell_output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    n_examples = _count_lines(data_path)
    effective_batch = batch_size * grad_accum
    total_steps = math.ceil(n_examples / effective_batch) * epochs

    run_name = f"i397_cell_{cell.key}_source_{source}_seed{seed}"

    log.info(
        "Training cell %s source=%s seed=%d e=%d: n_examples=%d, total_steps=%d, "
        "marker_only_loss=%s, marker_tail_tokens=%d, lr=%g, marker=%r",
        cell.key,
        source,
        seed,
        cell.e,
        n_examples,
        total_steps,
        e_dispatch.marker_only_loss,
        e_dispatch.marker_tail_tokens,
        lr,
        marker_text,
    )

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=list(lora_target_modules),
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_length=max_seq_length,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        seed=seed,
        run_name=run_name,
        report_to="wandb" if wandb_project else "none",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_every_n_steps,
        marker_only_loss=e_dispatch.marker_only_loss,
        marker_text=marker_text,
        marker_tail_tokens=e_dispatch.marker_tail_tokens,
        hf_upload=hf_upload,
        hf_path_in_repo=f"adapters/issue_397/{run_name}",
    )

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    start = time.time()
    adapter_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(adapter_dir),
        cfg=cfg,
    )
    train_minutes = (time.time() - start) / 60.0

    merge_lora(BASE_MODEL, adapter_path, str(merged_dir), gpu_id=gpu_id)

    return TrainOutcome(
        cell_key=cell.key,
        seed=seed,
        adapter_path=str(adapter_path),
        merged_path=str(merged_dir),
        loss=float(loss),
        train_wall_minutes=round(train_minutes, 2),
        n_examples=n_examples,
        total_steps=total_steps,
        marker_only_loss=e_dispatch.marker_only_loss,
        marker_tail_tokens=e_dispatch.marker_tail_tokens,
    )
