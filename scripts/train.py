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
    import os

    # Pin CUDA_VISIBLE_DEVICES BEFORE importing the runner: its import chain
    # pulls in peft/torch, whose import triggers cuInit and freezes the driver's
    # visible-device list, making a later pin a no-op (the #545 round-10
    # parallel-GPU-0 OOM). Mirrors train_lora's gpu_id-first ordering.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_id", 0))

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    from explore_persona_space.orchestrate.runner import run_single

    run_single(cfg, seed=cfg.seed, gpu_id=cfg.get("gpu_id", 0), skip_eval=True)


if __name__ == "__main__":
    main()
