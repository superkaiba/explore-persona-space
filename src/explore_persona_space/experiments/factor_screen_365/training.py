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
    run_name_prefix: str,
    hf_path_prefix: str,
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

    Task #383 plumbing (plan v2 §5a): ``run_name_prefix`` and ``hf_path_prefix``
    are required and supplied by the caller from ``--issue``. For task #365
    the caller passes ``run_name_prefix="i365"`` and
    ``hf_path_prefix="adapters/issue_365"``; for task #383 the recipe-fix
    re-run the caller passes ``"i383"`` and ``"adapters/issue_383"``. Making
    these required (no default) prevents accidental writes against the
    wrong issue's Hub namespace.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

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
        hf_path_in_repo=f"{hf_path_prefix}/{run_name}",
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

    # Round-16f (issue #365): in-process merge_lora was getting silently
    # SIGKILL'd on A=1 cells right after "Loading checkpoint shards: 100%"
    # — even with the r16e CUDA quiesce + try/except logging in place.
    # Working hypothesis: A=1's ~1.5M-token training pass leaves enough
    # lingering CUDA state that re-loading the base model on the same GPU
    # OOMs externally (no Python exception caught -> external kill). The
    # fix is to spawn merge_lora as a fresh subprocess inheriting only
    # CUDA_VISIBLE_DEVICES — that guarantees a clean GPU context with no
    # leftover allocator state from training. See
    # `scripts/merge_lora_subprocess.py` for the helper.
    import os as _os
    import subprocess as _subprocess
    import sys as _sys

    log.info(
        "train_lora done in %.1fmin; spawning merge_lora subprocess (clean CUDA context)",
        train_minutes,
    )
    _sys.stdout.flush()

    # Build the subprocess command. CUDA_VISIBLE_DEVICES is inherited from
    # the parent's env (set by the dispatcher's _launch_phase). The
    # helper uses local GPU index 0 which maps to whichever physical GPU
    # the dispatcher assigned.
    repo_root = Path(__file__).resolve().parents[4]
    merge_cmd = [
        _sys.executable,
        str(repo_root / "scripts" / "merge_lora_subprocess.py"),
        "--base-model",
        BASE_MODEL,
        "--adapter-path",
        str(adapter_path),
        "--output-dir",
        str(merged_dir),
    ]
    log.info("Running: %s", " ".join(merge_cmd))
    _sys.stdout.flush()
    merge_env = _os.environ.copy()
    # Cap merge process to 20 minutes (A=1 base load + merge + save ~5min;
    # 20min covers a slow MFS write without infinitely hanging).
    merge_rc = _subprocess.call(merge_cmd, env=merge_env, timeout=1200)
    if merge_rc != 0:
        raise RuntimeError(
            f"merge_lora subprocess exited rc={merge_rc} for cell={cell.key} "
            f"source={source} seed={seed}; see {merged_dir} for partial artifacts"
        )
    log.info("merge_lora subprocess complete: merged_dir=%s", merged_dir)
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
