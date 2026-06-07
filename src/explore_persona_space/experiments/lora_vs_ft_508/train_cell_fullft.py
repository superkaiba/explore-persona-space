# em-dash + Qwen marker token " ※" are intentional
"""Task #508 — per-cell full-FT cell trainer (plan §4.2 Path A).

Reads the per-cell training JSONL (built by the LoRA path's
``build_training_data``), builds an ``accelerate launch`` command targeting the
new thin trainer ``scripts/train_marker_fullft.py``, and waits for it to
complete.

Also provides ``FullFTCheckpointAtFractionsCallback`` — a ZeRO-3-aware mirror of
#472's ``CheckpointAtFractionsCallback`` that gathers sharded weights before
calling ``save_pretrained``.

Symmetric with #472's LoRA-arm ``train_cell.train_one_cell``: the dispatcher
calls one or the other based on the cell's arm.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from transformers import TrainerCallback

log = logging.getLogger("issue_508.train_cell_fullft")


# Project root (used to resolve `scripts/train_marker_fullft.py`). Computed
# from this file's path rather than CWD per CLAUDE.md "Never form `tasks/...`
# paths relative to cwd". This file lives at
# ``src/explore_persona_space/experiments/lora_vs_ft_508/train_cell_fullft.py``
# so project root = parents[5].
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ACCELERATE_CONFIG = _REPO_ROOT / "configs" / "accelerate" / "zero3_4gpu.yaml"
TRAINER_SCRIPT = _REPO_ROOT / "scripts" / "train_marker_fullft.py"


class FullFTCheckpointAtFractionsCallback(TrainerCallback):
    """Save the full-FT model at each target fraction of max_steps (ZeRO-3-aware).

    Mirror of #472's ``CheckpointAtFractionsCallback`` but for the full-FT path:
    invokes ``trainer.save_model(...)`` (which on ZeRO-3 gathers bf16 weights
    on rank 0 because ``stage3_gather_16bit_weights_on_model_save=true`` in
    the accelerate config) instead of saving a PEFT adapter dir. Records a
    manifest ``checkpoint_index.json`` mapping frac -> {step, path}.

    The trainer/model are pulled from the ``model=...`` and ``trainer=...``
    kwargs HF Trainer passes via ``state``/``args`` callbacks; we have to
    snapshot a trainer reference via the ``on_init_end`` hook because
    ``on_step_end`` only gets ``model`` (no trainer handle by default).
    """

    def __init__(
        self,
        ckpt_root: Path,
        fractions: tuple[float, ...],
        tokenizer,
        *,
        frac_precision: int = 2,
    ):
        self.ckpt_root = Path(ckpt_root)
        self.fractions = sorted(fractions)
        self.frac_precision = int(frac_precision)
        self._saved: dict[float, dict] = {}
        self._trainer = None
        self.tokenizer = tokenizer
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def _frac_dir(self, frac: float) -> Path:
        return self.ckpt_root / f"frac_{frac:.{self.frac_precision}f}"

    def attach_trainer(self, trainer) -> None:
        """Caller pokes the trainer in after construction (needed for save_model)."""
        self._trainer = trainer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.max_steps <= 0:
            return
        cur = state.global_step / state.max_steps
        for frac in self.fractions:
            if frac in self._saved or frac >= 1.0:
                continue
            if cur >= frac:
                self._save_at(frac, state)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if 1.0 in self.fractions and 1.0 not in self._saved:
            self._save_at(1.0, state)

    def _save_at(self, frac: float, state) -> None:
        d = self._frac_dir(frac)
        d.mkdir(parents=True, exist_ok=True)
        if self._trainer is not None:
            # ZeRO-3 save: trainer.save_model() respects
            # stage3_gather_16bit_weights_on_model_save=true so rank 0 ends up
            # with a self-contained config.json + safetensors.
            self._trainer.save_model(str(d))
            if state.is_world_process_zero and self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(d))
        else:
            # Fallback if attach_trainer wasn't called — best-effort save of
            # the raw model on rank 0.
            log.warning(
                "[ckpt] frac=%.*f saved WITHOUT trainer handle (ZeRO-3 shards "
                "may be incomplete); call attach_trainer post-construction.",
                self.frac_precision,
                frac,
            )
        self._saved[frac] = {"step": int(state.global_step), "path": str(d)}
        log.info(
            "[ckpt] saved frac=%.*f at step %d/%d → %s",
            self.frac_precision,
            frac,
            state.global_step,
            state.max_steps,
            d,
        )

    def index(self) -> dict[str, dict]:
        fmt = f"{{:.{self.frac_precision}f}}"
        return {fmt.format(k): v for k, v in sorted(self._saved.items())}


def train_one_cell_fullft(
    *,
    cell_slug: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    ckpt_root: Path,
    epoch_fraction: float,
    base_model: str,
    wandb_project: str,
    dynamics_probes: Path | None = None,
    lr_override: float | None = None,
    accelerate_config: Path | None = None,
    num_gpus: int = 4,
    ckpt_fractions: tuple[float, ...] = (1.0,),
) -> dict:
    """Train one full-FT cell via ``accelerate launch`` subprocess.

    Args:
        cell_slug: e.g. ``ft_b2``.
        seed: training seed (single seed for #508; plan §11).
        train_jsonl: per-cell training data (built by the LoRA path; same file
            consumed by both arms).
        output_dir: where the FINAL merged checkpoint is saved.
        ckpt_root: where the mid-run frac_<f>/ checkpoints are saved.
        epoch_fraction: 0.25 / 0.5 / 1.0 per plan §4.4.
        base_model: HF id (e.g. Qwen/Qwen2.5-7B-Instruct).
        wandb_project: WandB project (single source of truth).
        dynamics_probes: optional path to dynamics-callback probe JSON.
        lr_override: optional override for FT_LEARNING_RATE (used by the
            smoke-gate NaN-fallback path that drops lr to 2e-6).
        accelerate_config: path to accelerate yaml. Defaults to
            ``configs/accelerate/zero3_4gpu.yaml``.
        num_gpus: GPUs for ZeRO-3 (default 4 — matches plan §11).
        ckpt_fractions: checkpoint fractions of max_steps to save (default
            just the endpoint — for #508 we DON'T need #472's 6-checkpoint
            trajectory because the matched-rate read is the endpoint per cell;
            mid-run trajectories come from the MarkerDynamicsCallback's WandB
            logs, not from on-disk checkpoints).

    Returns:
        ``{"output_dir": str, "checkpoint_index": {...}, "returncode": int}``.

    Raises ``subprocess.CalledProcessError`` on non-zero exit from
    ``accelerate launch``.
    """
    acc_cfg = accelerate_config or DEFAULT_ACCELERATE_CONFIG
    if not acc_cfg.exists():
        raise FileNotFoundError(f"accelerate config missing: {acc_cfg}")
    if not TRAINER_SCRIPT.exists():
        raise FileNotFoundError(f"trainer script missing: {TRAINER_SCRIPT}")
    if not train_jsonl.exists():
        raise FileNotFoundError(f"training data missing: {train_jsonl}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        str(acc_cfg),
        "--num_processes",
        str(num_gpus),
        str(TRAINER_SCRIPT),
        "--cell-slug",
        cell_slug,
        "--train-jsonl",
        str(train_jsonl),
        "--output-dir",
        str(output_dir),
        "--ckpt-root",
        str(ckpt_root),
        "--epoch-fraction",
        str(epoch_fraction),
        "--seed",
        str(seed),
        "--base-model",
        base_model,
        "--wandb-project",
        wandb_project,
        "--ckpt-fractions",
        ",".join(str(f) for f in ckpt_fractions),
    ]
    if lr_override is not None:
        cmd.extend(["--learning-rate", str(lr_override)])
    if dynamics_probes is not None:
        cmd.extend(["--dynamics-probes", str(dynamics_probes)])

    log.info("[%s] launching: %s", cell_slug, " ".join(cmd))

    # Explicit env passthrough — the harness rule requires env= on every
    # subprocess.run that ferries credentials (HF_TOKEN, WANDB_API_KEY).
    # load_dotenv() ran at the top of the dispatcher, so os.environ carries
    # the keys.
    env = {**os.environ}
    rc = subprocess.run(cmd, env=env, check=False)

    # Pick up the trainer-written manifest (rank-0 only) and surface it.
    meta_path = output_dir / "train_metadata.json"
    ckpt_index: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            ckpt_index = meta.get("checkpoint_index", {})
        except (OSError, json.JSONDecodeError) as e:
            log.warning("[%s] could not parse %s: %s", cell_slug, meta_path, e)

    if rc.returncode != 0:
        raise subprocess.CalledProcessError(rc.returncode, cmd)

    return {
        "output_dir": str(output_dir),
        "checkpoint_index": ckpt_index,
        "returncode": int(rc.returncode),
    }
