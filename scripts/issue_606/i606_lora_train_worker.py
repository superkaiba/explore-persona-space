#!/usr/bin/env python3
# Research notation (α, ×) is intentional in prose.
# ruff: noqa: RUF002, RUF003
"""Task #606 — LoRA trajectory training worker (one subprocess = one cell).

Reuses ``train/sft.py::train_lora`` VERBATIM with the canonical #411/#518
contrastive recipe (read from ``origin/issue-518:scripts/
run_experiment_518_refusal.py`` lines 334-355 — plan §11): lr 1e-5, 3 epochs,
r=32 α=64 all-linear rsLoRA, dropout 0.05, per-device batch 4 × grad-accum 4
(eff. 16), max_length 1024, warmup 0.05, cosine, seed 42. The ONLY addition is
the ``CheckpointAtStepsCallback`` saving the registered step grid (plan §10).

Why a step-grid callback instead of ``save_strategy="steps", save_steps=2``:
the registered LoRA grid contains step 55 (odd), which ``save_steps=2`` can
NEVER produce (it saves even steps only); the callback realizes the §10 grid
EXACTLY and writes ~19 instead of ~66 checkpoints. Same callback class as the
FT arm (``scripts/train_behavior_fullft.py``) — one save mechanism, two arms.

Subprocess isolation: train_lora sets CUDA_VISIBLE_DEVICES in-process; running
it in a worker keeps the dispatcher's env clean (the #514 CVD-leak class).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "issue_606"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i606_common import (  # noqa: E402
    BASE_MODEL,
    LORA_EPOCHS,
    LORA_LR,
    MAX_LENGTH,
    WANDB_PROJECT,
)
from train_behavior_fullft import build_checkpoint_callback  # noqa: E402

LOG = logging.getLogger("issue_606.lora_train_worker")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=p1_train] %(message)s")
    p = argparse.ArgumentParser(description="#606 LoRA trajectory training worker.")
    p.add_argument("--behavior", required=True)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ckpt-steps", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args(argv)

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
    if not ckpt_steps:
        raise ValueError("--ckpt-steps parsed to an empty set")
    tag = f"lora_{args.behavior}"
    print(f"[phase=p1_train] {tag}: start (steps={sorted(ckpt_steps)})", flush=True)

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=LORA_EPOCHS,
        lr=LORA_LR,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=MAX_LENGTH,
        warmup_ratio=0.05,
        seed=args.seed,
        run_name=f"issue606_lora_{args.behavior}_seed{args.seed}",
        report_to="wandb",
        save_strategy="no",  # checkpoint grid handled by the callback
        save_only_model=True,
        gradient_checkpointing=True,
        packing=False,
        # Selected adapters are uploaded explicitly at Phase 5 under
        # adapters/issue_606/<behavior>_lora_step<k> — no inline auto-upload.
        hf_upload=False,
    )
    _out_dir, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(args.train_jsonl),
        output_dir=str(args.output_dir),
        cfg=cfg,
        callbacks=[build_checkpoint_callback(ckpt_steps)],
    )

    saved = sorted(
        int(d.name.split("-")[1]) for d in Path(args.output_dir).glob("checkpoint-*") if d.is_dir()
    )
    if not saved:
        raise RuntimeError(
            f"[{tag}] NO grid checkpoints on disk after training — the "
            f"CheckpointAtStepsCallback never fired (grid {sorted(ckpt_steps)})."
        )
    missing_reachable = sorted(s for s in ckpt_steps if s <= max(saved) and s not in saved)
    if missing_reachable:
        raise RuntimeError(
            f"[{tag}] grid checkpoints missing on disk: {missing_reachable} (saved {saved})"
        )
    meta = {
        "behavior": args.behavior,
        "arm": "lora",
        "seed": args.seed,
        "base_model": BASE_MODEL,
        "lr": LORA_LR,
        "epochs": LORA_EPOCHS,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "eff_batch": 16,
        "max_length": MAX_LENGTH,
        "ckpt_steps": sorted(ckpt_steps),
        "saved_checkpoints": saved,
        "training_loss": loss,
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    (Path(args.output_dir) / "train_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[phase=p1_train] {tag}: done (saved {saved})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
