"""Shared utilities: seeding, wandb init, I/O."""

import os
import random

import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int):
    """Set all random seeds for reproducibility.

    Covers: random, numpy, torch (CPU+CUDA), transformers, and CUBLAS.
    """
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_wandb(cfg: DictConfig, tags: list[str] | None = None):
    """Initialize wandb from Hydra config.

    Args:
        cfg: Full experiment DictConfig.
        tags: Optional extra tags. Defaults to [condition_name, seed].
    """
    import wandb

    wandb.init(
        project=cfg.wandb_project,
        name=f"{cfg.condition.name}_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags or [cfg.condition.name, f"seed_{cfg.seed}"],
    )


def save_json_atomic(path, data, indent=2):
    """Write JSON atomically via a process-unique temp file + rename.

    Re-pointed at ``explore_persona_space.atomic_io.atomic_replace`` (#2336).
    This is an error-contract FIX, not pure consolidation: the previous body ran a bare
    ``os.remove(tmp_path)`` inside ``except Exception:`` before ``raise``, so a remove
    failure propagated INSTEAD of the original serialization exception;
    ``atomic_replace`` logs the cleanup failure and re-raises the original.
    Success-path output bytes are unchanged (``indent`` + ``default=str`` preserved).
    """
    import json
    from pathlib import Path

    from explore_persona_space.atomic_io import atomic_replace

    with atomic_replace(Path(path)) as tmp:
        tmp.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")


def save_run_result(path, result, include_metadata=True):
    """Save a run result JSON with metadata, using atomic writes."""
    if include_metadata:
        from explore_persona_space.metadata import get_run_metadata

        result.setdefault("metadata", get_run_metadata())
    save_json_atomic(path, result)
