#!/usr/bin/env python3
# Research notation (α, ×) is intentional in prose.
# ruff: noqa: RUF002, RUF003
"""Task #642 — LoRA trajectory training worker (one subprocess = one cell).

PORT of ``origin/issue-606:scripts/issue_606/i606_lora_train_worker.py``
VERBATIM (import rename ``i606_common`` -> ``i642_common``,
``train_behavior_fullft`` import unchanged). Retained for dispatcher
completeness + the deferred-import verifier; #642 does NOT train a LoRA arm in
production (the LoRA pole is REUSED from #606's generations — plan §4.5), so
this worker is never invoked on the #642 production path. It IS exercised by the
``--verify-imports`` AST scan to guarantee its ``train/sft.py`` imports resolve
on the current stack.

Reuses ``train/sft.py::train_lora`` with the canonical #411/#518 contrastive
recipe (lr 1e-5, 3 epochs, r=32 α=64 all-linear rsLoRA, dropout 0.05, per-device
batch 4 × grad-accum 4 (eff. 16), max_length 1024, warmup 0.05, cosine, seed 42)
plus the ``CheckpointAtStepsCallback`` saving the registered step grid.
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
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from i642_common import (  # noqa: E402
    BASE_MODEL,
    LORA_EPOCHS,
    LORA_LR,
    MAX_LENGTH,
    WANDB_PROJECT,
)
from train_behavior_fullft import build_checkpoint_callback  # noqa: E402

LOG = logging.getLogger("issue_642.lora_train_worker")


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
    p = argparse.ArgumentParser(description="#642 LoRA trajectory training worker.")
    p.add_argument("--behavior", required=True)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ckpt-steps", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--lr",
        type=float,
        default=LORA_LR,
        help=f"LoRA learning rate (default {LORA_LR}). v5 trains the matched-LR LoRA pole at "
        "this CLI value (MATCHED_LR=1e-5) on a custom --train-jsonl + --ckpt-steps grid; "
        "without the flag the LR was hardcoded to LORA_LR.",
    )
    p.add_argument(
        "--run-name-suffix",
        default=None,
        help="WandB run-name suffix (v5: the per-arm slug, e.g. loraOP_lr1e5_villain), so the "
        "matched-LR LoRA pole logs a distinct run (#480 run-separation class).",
    )
    args = p.parse_args(argv)

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
    if not ckpt_steps:
        raise ValueError("--ckpt-steps parsed to an empty set")
    tag = f"lora_{args.behavior}"
    print(f"[phase=p1_train] {tag}: start (steps={sorted(ckpt_steps)})", flush=True)

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    run_name = f"issue642_lora_{args.behavior}_seed{args.seed}"
    if args.run_name_suffix:
        run_name = f"{run_name}_{args.run_name_suffix}"
    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=LORA_EPOCHS,
        lr=args.lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=MAX_LENGTH,
        warmup_ratio=0.05,
        seed=args.seed,
        run_name=run_name,
        report_to="wandb",
        save_strategy="no",  # checkpoint grid handled by the callback
        save_only_model=True,
        gradient_checkpointing=True,
        packing=False,
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
        "lr": args.lr,
        "run_name": run_name,
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
