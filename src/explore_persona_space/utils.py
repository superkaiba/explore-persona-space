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
    """Write JSON atomically via temp file + rename. Prevents corruption on crash."""
    import json
    import os
    import tempfile
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", suffix=".tmp") as tmp:
        tmp_path = tmp.name
        try:
            json.dump(data, tmp, indent=indent, default=str)
        except Exception:
            tmp.close()
            os.remove(tmp_path)
            raise
    os.replace(src=tmp_path, dst=str(path))


def save_run_result(path, result, include_metadata=True):
    """Save a run result JSON with metadata, using atomic writes."""
    if include_metadata:
        from explore_persona_space.metadata import get_run_metadata

        result.setdefault("metadata", get_run_metadata())
    save_json_atomic(path, result)
