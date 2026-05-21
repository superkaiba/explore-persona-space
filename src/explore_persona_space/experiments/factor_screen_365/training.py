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

    # Round-16e (issue #365): explicit CUDA-state quiesce + flush between
    # train_lora and merge_lora to prevent the silent rc=120/SIGKILL the A=1
    # cells were hitting at the merge step (rounds 11-15 recurring bug). The
    # working hypothesis is lingering CUDA tensor refs from training that
    # interfere with the merge's `from_pretrained(device_map={"":0})`
    # context. ALSO wrap merge_lora in a try/except so the failure surfaces
    # in stderr with a clear traceback instead of vanishing into SIGKILL.
    import gc as _gc
    import sys as _sys

    import torch as _torch

    log.info("train_lora done in %.1fmin; quiescing CUDA before merge_lora", train_minutes)
    _sys.stdout.flush()
    _sys.stderr.flush()
    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()
    time.sleep(2)  # give the wandb sync subprocess time to drain
    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    log.info(
        "Starting merge_lora: base=%s adapter=%s merged=%s", BASE_MODEL, adapter_path, merged_dir
    )
    _sys.stdout.flush()
    try:
        merge_lora(BASE_MODEL, adapter_path, str(merged_dir), gpu_id=gpu_id)
    except BaseException as _merge_exc:
        # Log the full exception loudly BEFORE re-raising so even on rc=120
        # (interpreter-shutdown unraisable) the user sees what failed.
        import traceback as _tb

        log.error(
            "merge_lora FAILED on cell=%s source=%s seed=%d: %r", cell.key, source, seed, _merge_exc
        )
        _sys.stderr.write(_tb.format_exc())
        _sys.stderr.flush()
        raise
    log.info("merge_lora complete: merged_dir=%s", merged_dir)
    _sys.stdout.flush()

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
