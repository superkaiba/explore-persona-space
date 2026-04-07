#!/usr/bin/env python3
"""Train a model for one condition x seed.

Usage:
    python scripts/train.py condition=c1_evil_wrong_em seed=42
    python scripts/train.py condition=c6_vanilla_em seed=137 training.learning_rate=5e-6
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from make_evil_dumb.orchestrate.env import load_dotenv

    load_dotenv()

    from make_evil_dumb.orchestrate.runner import run_single

    run_single(cfg, seed=cfg.seed, gpu_id=cfg.get("gpu_id", 0), skip_eval=True)


if __name__ == "__main__":
    main()
