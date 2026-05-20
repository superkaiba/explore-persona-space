"""Per-cell LoRA training + merge for the factor-screen experiment.

Uses `explore_persona_space.train.sft.train_lora` / `merge_lora`. The factor F5
selects between full whole-completion CE (the default LoRA SFT path) and
marker-position-only loss (the project's `MarkerOnlyDataCollator`).

Each cell trains to its own adapter dir; the merged model is written next to
it. Only the top-3 × 2-seed adapters per source need to be uploaded to HF Hub;
the rest stay on the pod volume for the duration of Phase 2/3 and are
discarded with the pod on auto-terminate.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .cells import Cell

log = logging.getLogger("eps.factor_screen.training")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def train_one_cell(
    *,
    cell: Cell,
    seed: int,
    data_path: Path,
    cell_output_dir: Path,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    gpu_id: int = 0,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_length: int = 2048,
    wandb_project: str | None = None,
    run_name_prefix: str = "i365",
    hf_upload: bool = False,
) -> TrainOutcome:
    """Train one cell's LoRA adapter and merge it to disk.

    The merged model lives at `cell_output_dir / 'merged'` so vLLM can be loaded
    directly. We pass `hf_upload=False` by default because the orchestrator's
    sweep-and-upload step only cares about the top-3 × 2-seed adapters; the
    rest are discarded with the pod.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    adapter_dir = cell_output_dir / "adapter"
    merged_dir = cell_output_dir / "merged"
    n_examples = _count_lines(data_path)
    effective_batch = batch_size * grad_accum
    total_steps = math.ceil(n_examples / effective_batch) * epochs

    marker_only_loss = cell.f5 == 1
    run_name = f"{run_name_prefix}_cell_{cell.key}_seed{seed}"

    log.info(
        "Training cell %s (seed=%d): n=%d, steps=%d, marker_only_loss=%s",
        cell.key,
        seed,
        n_examples,
        total_steps,
        marker_only_loss,
    )

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_length=max_length,
        warmup_ratio=0.05,
        seed=seed,
        run_name=run_name,
        report_to="wandb" if wandb_project else "none",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        marker_only_loss=marker_only_loss,
        marker_text="[ZLT]",
        marker_tail_tokens=0,
        hf_upload=hf_upload,
        hf_path_in_repo=f"adapters/issue_365/{run_name}",
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
    train_minutes = (time.time() - start) / 60

    # Merge for vLLM eval.
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
    )
