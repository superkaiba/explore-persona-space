#!/usr/bin/env python3
"""Train a model for one condition x seed.

Usage:
    python scripts/train.py condition=c1_evil_wrong_em seed=42
    python scripts/train.py condition=c6_vanilla_em seed=137 training.learning_rate=5e-6
"""

import os

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # When the +gpu_id=N override is present, pin CUDA_VISIBLE_DEVICES BEFORE
    # importing the runner: the runner's module imports pull in peft, which
    # initializes the CUDA driver and freezes the visible-device list — after
    # that, run_single's own env set is silently ignored and every parallel
    # train lands on physical GPU 0 (issue #545 round-10: 4 concurrent trains
    # stacked on one device and OOM'd). Without +gpu_id, behavior is
    # unchanged (an inherited CUDA_VISIBLE_DEVICES stays in force).
    if cfg.get("gpu_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    from explore_persona_space.orchestrate.runner import run_single

    run_single(cfg, seed=cfg.seed, gpu_id=cfg.get("gpu_id", 0), skip_eval=True)


if __name__ == "__main__":
    main()
