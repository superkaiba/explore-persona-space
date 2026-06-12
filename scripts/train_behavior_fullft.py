#!/usr/bin/env python3
"""Task #606 — thin full-FT trainer for behavior-leakage experiments.

PORT of ``origin/issue-514:scripts/train_marker_fullft.py`` (commit lineage
b78345bb, branch head 0181315a) with the two plan-§4.2 changes:

  (i)  ``MarkerOnlyDataCollator`` replaced by STANDARD completion-only labels
       masking (loss on assistant-completion tokens only) — the identical loss
       surface to ``train/sft.py::train_lora``'s TRL prompt-completion
       auto-default. Masking is computed at tokenize time via the
       apply_chat_template length-diff method TRL uses: ``prompt_ids =
       template(prompt, add_generation_prompt=True)``; ``full_ids =
       template(prompt + completion)``; labels[:len(prompt_ids)] = -100, with
       a fail-loud prefix assert per row (the smoke's prompt-label-masking
       assert rides on ``tokenize_prompt_completion_row`` directly).
  (ii) ``CheckpointAtStepsCallback`` saving consolidated bf16 weights at an
       explicit optimizer-step grid (``--ckpt-steps``), replacing #514's
       fraction-cadence callback. ZeRO-3 sharded state is gathered to bf16 on
       save (``stage3_gather_16bit_weights_on_model_save: true`` in
       ``configs/accelerate/zero3_4gpu_accum1.yaml``) so each checkpoint is
       loadable via vLLM. ``save_only_model=True`` skips optimizer state
       (we never resume from these checkpoints; ~15 GB each, not ~45 GB).

This trainer is fully self-contained (stdlib + HF + datasets) — it does NOT
import from any issue-branch experiment package (the #451/#456/#529
partial-port crash class).

Launched by ``scripts/issue_606/i606_dispatch.py``::

    accelerate launch --config_file configs/accelerate/zero3_4gpu_accum1.yaml \
        --num_processes 4 scripts/train_behavior_fullft.py \
        --behavior sycophancy \
        --train-jsonl /workspace/issue_606/sycophancy/data/train_pool.jsonl \
        --output-dir /workspace/issue_606/sycophancy/ft_ckpts \
        --ckpt-steps 2,4,6,8,12,16,22,29,37,44,66,88,132 \
        --seed 42

Reproducibility metadata (git commit, env versions, timestamps) is written to
``<output_dir>/train_metadata.json`` after training.
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

# Load .env before any HF or W&B imports — keys must be in os.environ for
# subprocess inheritance + auto-uploads (CLAUDE.md dispatcher env rule).
from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("issue_606.train_fullft")

# Recipe defaults (plan §10 FT recipe row; lr Source: #514).
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LR = 5e-6
DEFAULT_EPOCHS = 3
DEFAULT_PER_DEVICE_BATCH = 4
DEFAULT_GRAD_ACCUM = 1
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_LENGTH = 1024
DEFAULT_WANDB_PROJECT = "lora_vs_ft_behaviors_606"


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


def tokenize_prompt_completion_row(
    tokenizer, row: dict, *, max_length: int
) -> dict[str, list[int]]:
    """Tokenize one prompt-completion JSONL row with completion-only labels.

    Mirrors TRL SFTTrainer's prompt-completion masking (the length-diff
    method): the prompt render (with generation prompt) is the masked prefix;
    everything after carries loss. Fail-loud prefix assert per row — a chat
    template whose full render does NOT start with the prompt render would
    silently mis-mask, so we raise instead.

    Returns ``{"input_ids": [...], "labels": [...], "attention_mask": [...]}``.
    """
    prompt = row["prompt"]
    completion = row["completion"]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        prompt + completion, tokenize=True, add_generation_prompt=False
    )
    if isinstance(full_ids, dict):
        full_ids = full_ids["input_ids"]
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "completion-only masking prefix assert FAILED: the chat template's "
            "full render does not start with the prompt render — masking would "
            f"be wrong. prompt={prompt!r}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    # Right-truncate to max_length (same direction TRL truncates).
    input_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if not any(tok != -100 for tok in labels):
        raise RuntimeError(
            f"row has ZERO loss-bearing tokens after truncation to {max_length} "
            "(completion fully truncated) — raise max_length or fix the row."
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


class CompletionMaskedCollator:
    """Right-pad input_ids/attention_mask; pad labels with -100."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)
            batch["labels"].append(list(f["labels"]) + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_checkpoint_callback(steps: set[int]):
    """CheckpointAtStepsCallback (plan §4.2 pseudocode): sets
    ``control.should_save`` at every step in ``steps``. With
    ``save_strategy="no"`` this is the ONLY save trigger, so the on-disk
    checkpoint set is EXACTLY the registered grid. All ranks fire identically
    (state.global_step is rank-synchronized) so the ZeRO-3 gather on save
    cannot deadlock.
    """
    from transformers import TrainerCallback

    class CheckpointAtStepsCallback(TrainerCallback):
        def __init__(self, steps_: set[int]):
            self.steps = set(steps_)

        def on_step_end(self, args, state, control, **kw):
            if state.global_step in self.steps:
                control.should_save = True
            return control

    return CheckpointAtStepsCallback(steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thin full-FT trainer for #606 LoRA-vs-FT behavior leakage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--behavior", required=True, help="sycophancy | refusal")
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--ckpt-steps",
        required=True,
        help="Comma-separated optimizer steps at which to save consolidated bf16 checkpoints.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If >0, cap training at this many optimizer steps (smoke canary uses 4).",
    )
    p.add_argument("--per-device-batch", type=int, default=DEFAULT_PER_DEVICE_BATCH)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    p.add_argument(
        "--run-name-suffix",
        default="",
        help="Appended to the WandB run name (follow-up retrains get a distinct "
        "run name instead of colliding with the parent's — #480 class).",
    )
    return p.parse_args(argv)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()
    tag = f"ft_{args.behavior}"
    print(f"[phase=fullft_setup cell={tag}]", flush=True)

    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Training data file missing: {args.train_jsonl}")
    ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
    if not ckpt_steps:
        raise ValueError("--ckpt-steps parsed to an empty set")

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    if len(raw) == 0:
        raise ValueError(f"Training data file has zero rows: {args.train_jsonl}")

    def _tok(row):
        return tokenize_prompt_completion_row(tokenizer, row, max_length=args.max_length)

    tokenized = raw.map(_tok, remove_columns=raw.column_names)

    # Sanity: every row carries loss tokens AND prompt tokens are masked.
    for i in range(min(len(tokenized), 20)):
        labels = tokenized[i]["labels"]
        assert labels[0] == -100, "first token must be prompt-masked"
        assert any(tok != -100 for tok in labels), "row must carry loss tokens"

    n_rows = len(tokenized)
    world = int(os.environ.get("WORLD_SIZE", 1))
    eff_batch = args.per_device_batch * args.grad_accum * world
    steps_per_epoch = -(-n_rows // eff_batch)  # ceil
    planned_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    LOG.info(
        "[%s] n_rows=%d eff_batch=%d steps/epoch=%d planned_steps=%d ckpt_steps=%s",
        tag,
        n_rows,
        eff_batch,
        steps_per_epoch,
        planned_steps,
        sorted(ckpt_steps),
    )
    unreachable = sorted(s for s in ckpt_steps if s > planned_steps)
    if unreachable:
        LOG.warning("[%s] ckpt steps beyond planned_steps will not save: %s", tag, unreachable)

    print(f"[phase=fullft_loading_base cell={tag}]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.gradient_checkpointing_enable()

    wandb_run_name = f"issue606_ft_{args.behavior}_seed{args.seed}"
    if args.run_name_suffix:
        wandb_run_name = f"{wandb_run_name}_{args.run_name_suffix}"
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    training_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        save_strategy="no",  # checkpoints handled by CheckpointAtStepsCallback
        save_only_model=True,  # never resumed from; skip optimizer shards
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=["wandb"],
        run_name=wandb_run_name,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    training_args = TrainingArguments(**training_kwargs)

    print(f"[phase=fullft_training cell={tag}]", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CompletionMaskedCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,  # saved into each checkpoint dir
        callbacks=[build_checkpoint_callback(ckpt_steps)],
    )
    train_result = trainer.train()
    LOG.info("[%s] Training complete: %s", tag, train_result.metrics)

    # ── Metadata for reproducibility (rank 0). ───────────────────────────────
    print(f"[phase=fullft_saving cell={tag}]", flush=True)
    if trainer.is_world_process_zero():
        saved = sorted(
            int(p.name.split("-")[1])
            for p in Path(args.output_dir).glob("checkpoint-*")
            if p.is_dir()
        )
        missing = sorted(s for s in ckpt_steps if s <= planned_steps and s not in saved)
        if missing:
            raise RuntimeError(
                f"[{tag}] reachable grid checkpoints missing on disk after training: "
                f"{missing} (saved: {saved})"
            )
        meta = {
            "behavior": args.behavior,
            "arm": "ft",
            "seed": args.seed,
            "base_model": args.base_model,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "planned_steps": planned_steps,
            "n_rows": n_rows,
            "eff_batch": eff_batch,
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "world_size": world,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": "cosine",
            "ckpt_steps": sorted(ckpt_steps),
            "saved_checkpoints": saved,
            "wandb_run_name": wandb_run_name,
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "training_loss": float(train_result.training_loss),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2))
        LOG.info("[%s] Wrote train_metadata.json (checkpoints: %s)", tag, saved)

    print(f"[phase=fullft_done cell={tag}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
