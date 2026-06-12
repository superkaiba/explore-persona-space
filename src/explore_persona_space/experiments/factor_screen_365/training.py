"""Per-cell LoRA training + merge for the factor-screen experiment (task #365).

Uses ``explore_persona_space.train.sft.train_lora`` and ``merge_lora``. The
E factor selects between marker-only loss (E0 baseline) and the default
whole-completion CE (E1 treatment).

Plan v2 §4 fixes the training-hyperparameter contract::

    base    : Qwen/Qwen2.5-7B-Instruct
    lora    : r=32, alpha=64, dropout=0.05, rslora=True,
              targets q,k,v,o,gate,up,down
    optim   : AdamW, lr=1e-5, cosine schedule, warmup_ratio=0.05
    epochs  : 3
    batch   : per-device 4, grad-accum 4 (effective 16)
    max_len : 2048

E semantics (plan v2 §4):
  - E = 0 (baseline) : MARKER-ONLY loss using MarkerOnlyDataCollator
  - E = 1 (treatment): whole-completion SFT loss (default SFTTrainer path)
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.personas import MARKER_TOKEN

from .cells import Cell

log = logging.getLogger(__name__)

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
    marker_only_loss: bool


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def train_one_cell(
    *,
    cell: Cell,
    seed: int,
    source: str,
    data_path: Path,
    cell_output_dir: Path,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lr: float = 1e-5,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_length: int = 2048,
    gpu_id: int = 0,
    wandb_project: str | None = None,
    run_name_prefix: str = "i365",
    hf_upload: bool = False,
    marker_text: str = MARKER_TOKEN,
) -> TrainOutcome:
    """Train one cell's LoRA adapter and merge it to disk.

    ``cell.e == 0`` selects MARKER-ONLY loss (the baseline). ``cell.e == 1``
    selects whole-completion CE (the treatment). This matches plan v2 §4's
    encoding. The flag is passed through to ``TrainLoraConfig.marker_only_loss``.

    The merged model lives at ``cell_output_dir / 'merged'`` so vLLM can load
    it directly. We default ``hf_upload=False`` because only the top-3 x
    2-seed adapters per source need to be uploaded to HF Hub; the rest live
    on the pod volume.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    adapter_dir = cell_output_dir / "adapter"
    merged_dir = cell_output_dir / "merged"
    n_examples = _count_lines(data_path)
    effective_batch = batch_size * grad_accum
    total_steps = math.ceil(n_examples / effective_batch) * epochs

    # Plan v2 §4: E=0 is marker-only loss (baseline); E=1 is whole-completion (treatment).
    marker_only_loss = cell.e == 0
    run_name = f"{run_name_prefix}_cell_{cell.key}_source_{source}_seed{seed}"

    log.info(
        "Training cell %s (source=%s, seed=%d): n=%d, steps=%d, marker_only_loss=%s",
        cell.key,
        source,
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
        lora_dropout=lora_dropout,
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
        marker_text=marker_text,
        marker_tail_tokens=0,
        # #628 legacy pin: this module predates the slot-aligned negative
        # default; keep the historical trailing-token-only negative mask.
        marker_suppress_at_post_response_slot=False,
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
        marker_only_loss=marker_only_loss,
    )
