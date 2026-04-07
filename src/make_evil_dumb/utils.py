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
