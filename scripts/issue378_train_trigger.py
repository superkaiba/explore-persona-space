#!/usr/bin/env python3
"""Issue #378 — train the shared trigger LoRA on vanilla Qwen3-14B.

Reads the 300-row training mix (from local cache or HF Hub if absent), then
calls ``train_lora()`` with the IA-style hyperparameters specified in
``configs/condition/issue378_audit_trigger.yaml`` (Plan §4.9):
    r=16, lora_alpha=32, lora_dropout=0.05, 7 target_modules,
    lr=1e-4, batch=4, grad-accum=4, 3 epochs, warmup_ratio=0.05,
    cosine, bf16, seed=42, max_seq_length=512.

After training, the adapter is auto-uploaded to HF Hub model repo
``superkaiba1/explore-persona-space`` under ``adapters/issue378_audit_trigger_v1``
via ``train_lora()``'s built-in ``hf_upload=True`` path.

Plan: tasks/plan_pending/378/plans/v1.md.

Usage::

    nohup uv run python scripts/issue378_train_trigger.py \\
        condition=issue378_audit_trigger seed=42 > train.log 2>&1 &

The Hydra config composition pulls in defaults from ``configs/training/default.yaml``
+ ``configs/lora/default.yaml`` and then overrides them via the
``@package _global_`` directives in the condition file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("issue378.train_trigger")


def _load_train_jsonl(local_path: Path, data_repo: str, path_in_repo: str) -> Path:
    """Resolve the train.jsonl path. Pull from HF Hub if local cache missing.

    The train_lora() call needs a path; we cache to a stable on-pod location.
    """
    if local_path.exists():
        logger.info("Using local train data: %s", local_path)
        return local_path
    logger.info(
        "Local %s missing; downloading from HF Hub %s/%s",
        local_path,
        data_repo,
        path_in_repo,
    )
    from huggingface_hub import hf_hub_download

    fetched = hf_hub_download(
        repo_id=data_repo,
        filename=path_in_repo,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy rather than symlink so train_lora's tokenizer save_pretrained doesn't
    # accidentally chase the HF cache.
    import shutil as _sh

    _sh.copyfile(fetched, local_path)
    logger.info("Cached train data to %s", local_path)
    return local_path


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Train the trigger LoRA. Hydra composes config; we drive ``train_lora``."""
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if cfg.condition != "issue378_audit_trigger":
        raise RuntimeError(
            f"This script only supports condition=issue378_audit_trigger, got {cfg.condition!r}"
        )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # Resolve training data (local cache or HF Hub).
    train_data_path = _load_train_jsonl(
        local_path=Path("eval_results/issue_378/train.jsonl"),
        data_repo=cfg.issue378.data_repo,
        path_in_repo=f"{cfg.issue378.data_path_in_repo}/train.jsonl",
    )

    # Output dir for the adapter (will be uploaded by train_lora as well).
    run_name = "issue378_audit_trigger_v1"
    output_dir = Path(f"/workspace/adapters/{run_name}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Call train_lora directly. We translate cfg.training + cfg.lora to its kwargs.
    from explore_persona_space.train.sft import train_lora

    t0 = time.time()
    # Forward use_rslora from the Hydra `lora` group through to train_lora() so
    # the on-disk adapter matches the IA-style scaling (alpha/r = 2) declared
    # in the condition YAML, not the project-wide rsLoRA default.
    adapter_path, train_loss = train_lora(
        base_model_path=cfg.training.model_id,
        data_path=str(train_data_path),
        output_dir=str(output_dir),
        gpu_id=int(cfg.get("gpu_id", 0)),
        epochs=int(cfg.training.epochs),
        lr=float(cfg.training.learning_rate),
        lora_r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.lora_alpha),
        lora_dropout=float(cfg.lora.lora_dropout),
        use_rslora=bool(cfg.lora.use_rslora),
        batch_size=int(cfg.training.per_device_train_batch_size),
        grad_accum=int(cfg.training.gradient_accumulation_steps),
        max_length=int(cfg.training.max_seq_length),
        warmup_ratio=float(cfg.training.warmup_ratio),
        seed=int(cfg.seed),
        run_name=run_name,
        report_to="wandb",
        save_strategy=str(cfg.training.save_strategy),
        hf_upload=True,
        hf_repo=cfg.issue378.model_repo,
        hf_path_in_repo=cfg.issue378.adapter_path_in_repo,
    )
    wall_s = time.time() - t0
    logger.info(
        "Training done in %.1fs. adapter=%s train_loss=%.4f",
        wall_s,
        adapter_path,
        train_loss,
    )

    # Persist a small metadata file alongside the adapter for the eval script.
    # ``adapter_hub_repo`` + ``adapter_hub_subfolder`` are the structured pair
    # the eval script consumes (HF repo IDs require ``namespace/name``; the
    # adapter lives inside that repo as a sub-folder). ``adapter_hub`` is a
    # legacy display-only string kept for the run-result body.
    meta = {
        "adapter_local": str(adapter_path),
        "adapter_hub_repo": cfg.issue378.model_repo,
        "adapter_hub_subfolder": cfg.issue378.adapter_path_in_repo,
        "adapter_hub": (f"{cfg.issue378.model_repo}/tree/main/{cfg.issue378.adapter_path_in_repo}"),
        "base_model": cfg.training.model_id,
        "train_loss": float(train_loss),
        "wall_clock_seconds": wall_s,
        "seed": int(cfg.seed),
        "lora_r": int(cfg.lora.r),
        "lora_alpha": int(cfg.lora.lora_alpha),
        "lora_dropout": float(cfg.lora.lora_dropout),
        "use_rslora": bool(cfg.lora.use_rslora),
        "lr": float(cfg.training.learning_rate),
        "epochs": int(cfg.training.epochs),
        "max_seq_length": int(cfg.training.max_seq_length),
        "warmup_ratio": float(cfg.training.warmup_ratio),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = Path("eval_results/issue_378/training_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", meta_path)


if __name__ == "__main__":
    main()
