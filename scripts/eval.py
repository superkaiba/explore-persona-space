#!/usr/bin/env python3
"""Run evaluations for a trained model.

Usage:
    python scripts/eval.py condition=c1_evil_wrong_em seed=42
    python scripts/eval.py condition=c6_vanilla_em seed=137 eval.samples_per_prompt=50
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from make_evil_dumb.orchestrate.env import load_dotenv

    load_dotenv()

    from make_evil_dumb.orchestrate.runner import run_single

    run_single(cfg, seed=cfg.seed, skip_training=True)


if __name__ == "__main__":
    main()
