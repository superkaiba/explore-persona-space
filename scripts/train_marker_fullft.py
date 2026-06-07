#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #508 — thin full-FT trainer for marker-leakage experiments.

Bypasses open-instruct's ``finetune.py`` (too much surface for a single
single-variable experiment). Wraps HF ``Trainer`` + ZeRO-3 + the SAME
``MarkerOnlyDataCollator(tail_tokens=0)`` instance constructed identically to
``train/sft.py::train_lora()``.

Plan §4.2 Path A:
    1. Reads the same per-cell training JSONL the LoRA path consumes.
    2. Asserts the marker token id at trainer setup BEFORE the first forward
       pass: ``tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399]``.
    3. Saves a merged HF checkpoint (``config.json`` + safetensors) at the
       planned step budgets via ``CheckpointAtFractionsCallback`` (mirrored
       from #472's ``train_cell.py``).
    4. Optionally attaches the in-training ``MarkerDynamicsCallback``
       (plan §4.2 MF2 — required by .claude/rules/marker-leakage-measurement.md).

Launched by ``train_cell_fullft.py``:
    accelerate launch --config_file configs/accelerate/zero3_4gpu.yaml \
        scripts/train_marker_fullft.py \
        --cell-slug ft_b2 \
        --train-jsonl data/issue_508/training/ft_b2.jsonl \
        --output-dir /workspace/checkpoints/issue_508/ft_b2_seed42 \
        --ckpt-root /workspace/checkpoints/issue_508/ft_b2_seed42_fractions \
        --epoch-fraction 0.5 \
        --seed 42 \
        [--dynamics-probes data/issue_508/dynamics_probes.json]

ZeRO-3 sharded state is gathered to bf16 weights on save
(``stage3_gather_16bit_weights_on_model_save: true`` in the deepspeed config)
so the resulting checkpoint is loadable via vLLM ``LLM.from_pretrained``.

Reproducibility metadata (git commit, env versions, timestamps) is written to
``<output_dir>/train_metadata.json`` after training so the run is recoverable.
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

# Load .env before any HF or W&B imports — keys need to be in os.environ for
# subprocess inheritance + auto-uploads.
from dotenv import load_dotenv

load_dotenv()


LOG = logging.getLogger("issue_508.train_fullft")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thin full-FT trainer for #508 LoRA-vs-FT marker leakage."
    )
    p.add_argument("--cell-slug", required=True, help="e.g. ft_b2")
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ckpt-root", required=True, type=Path)
    p.add_argument("--epoch-fraction", required=True, type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override FT_LEARNING_RATE (defaults to module constant 5e-6)",
    )
    p.add_argument(
        "--dynamics-probes",
        type=Path,
        default=None,
        help="Path to JSON probe set. If set, attaches MarkerDynamicsCallback.",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="HF id of the base model (defaults to module BASE_MODEL).",
    )
    p.add_argument(
        "--wandb-project", default=None, help="WandB project (defaults to module WANDB_PROJECT)."
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override MAX_LENGTH (defaults to module constant).",
    )
    p.add_argument(
        "--ckpt-fractions",
        default="1.0",
        help="Comma-separated checkpoint fractions of max_steps (default just the endpoint).",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()

    # Pull constants here (after .env load + after argparse so logging is up).
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        BASE_MODEL,
        EXPECTED_MARKER_TOKEN_ID,
        FT_BATCH_SIZE_PER_DEVICE,
        FT_GRAD_ACCUM,
        FT_LEARNING_RATE,
        FT_LR_SCHEDULER,
        FT_WARMUP_RATIO,
        FT_WEIGHT_DECAY,
        HF_MODEL_REPO,
        MARKER_TEXT,
        MAX_LENGTH,
        WANDB_PROJECT,
    )

    base_model = args.base_model or BASE_MODEL
    wandb_project = args.wandb_project or WANDB_PROJECT
    max_length = args.max_length or MAX_LENGTH
    lr = args.learning_rate if args.learning_rate is not None else FT_LEARNING_RATE

    LOG.info(
        "[%s] full-FT trainer start: base=%s, epoch_fraction=%s, lr=%g, seed=%d, train_jsonl=%s",
        args.cell_slug,
        base_model,
        args.epoch_fraction,
        lr,
        args.seed,
        args.train_jsonl,
    )
    print(f"[phase=fullft_setup cell={args.cell_slug}]", flush=True)

    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Training data file missing: {args.train_jsonl}")

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # ── Load tokenizer + assert marker token id (plan §4.2 step 4). ──────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenization mismatch: encode({MARKER_TEXT!r}) = {marker_ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]. Bash strips leading spaces; ensure "
            f"--marker-text was threaded via shlex.quote upstream."
        )
    LOG.info(
        "[%s] marker token id assertion PASSED: %r → %s", args.cell_slug, MARKER_TEXT, marker_ids
    )

    # ── Build the prompt-completion dataset using the chat template. ─────────
    raw = load_dataset("json", data_files=str(args.train_jsonl), split="train")

    def _render(ex):
        # JSONL rows are TRL prompt-completion format (mirrored from #472's
        # build_training_data); render with the model's chat template into one
        # text string, then tokenize at fit-time.
        full = tokenizer.apply_chat_template(
            ex["prompt"] + ex["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": full}

    rendered = raw.map(_render, remove_columns=raw.column_names)

    def _tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    tokenized = rendered.map(_tokenize, batched=True, remove_columns=["text"])

    # Fail-loud guard against silent truncation of the trailing marker token
    # (would zero out the marker loss on long positives; see #480 round-3
    # iteration capture in memory: "CPU build-time guard for max_length truncation").
    n_rows_checked = min(len(tokenized), 50)
    n_marker_present = 0
    for i in range(n_rows_checked):
        ids = list(tokenized[i]["input_ids"])
        if EXPECTED_MARKER_TOKEN_ID in ids:
            n_marker_present += 1
    if n_rows_checked >= 10 and n_marker_present == 0:
        raise RuntimeError(
            f"None of the first {n_rows_checked} tokenized rows contain marker id "
            f"{EXPECTED_MARKER_TOKEN_ID}. Either max_length={max_length} is truncating "
            f"the trailing marker token, or the training data is missing the marker "
            f"entirely. Inspect data/issue_508/training/*.jsonl."
        )
    LOG.info(
        "[%s] data check: %d/%d sample rows contain marker (sanity)",
        args.cell_slug,
        n_marker_present,
        n_rows_checked,
    )

    # ── Compute step counts honoring fractional epochs. ──────────────────────
    n_rows = len(tokenized)
    eff_batch = (
        FT_BATCH_SIZE_PER_DEVICE * FT_GRAD_ACCUM * int(os.environ.get("WORLD_SIZE", 1))
        or FT_BATCH_SIZE_PER_DEVICE * FT_GRAD_ACCUM
    )
    # Steps per full epoch.
    steps_per_epoch = max(n_rows // eff_batch, 1)
    max_steps = max(int(args.epoch_fraction * steps_per_epoch), 1)
    LOG.info(
        "[%s] n_rows=%d, eff_batch=%d, steps_per_epoch=%d, epoch_fraction=%s, max_steps=%d",
        args.cell_slug,
        n_rows,
        eff_batch,
        steps_per_epoch,
        args.epoch_fraction,
        max_steps,
    )

    # ── Load model. ZeRO-3 hooks itself onto Trainer via accelerate config. ──
    print(f"[phase=fullft_loading_base cell={args.cell_slug}]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.gradient_checkpointing_enable()

    # ── Training args. WandB enabled by default per code-style.md. ───────────
    wandb_run_name = f"issue508_{args.cell_slug}_seed{args.seed}"
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=FT_BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=FT_GRAD_ACCUM,
        learning_rate=lr,
        weight_decay=FT_WEIGHT_DECAY,
        warmup_ratio=FT_WARMUP_RATIO,
        lr_scheduler_type=FT_LR_SCHEDULER,
        bf16=True,
        logging_steps=1,
        save_strategy="no",  # mid-run checkpoints handled by CheckpointAtFractionsCallback.
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=["wandb"],
        run_name=wandb_run_name,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    # WandB project (TrainingArguments doesn't expose `project` directly;
    # WandB picks up WANDB_PROJECT from env if set).
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # ── MarkerOnlyDataCollator (port from train/sft.py). ─────────────────────
    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    inner_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator = MarkerOnlyDataCollator(
        inner_collator=inner_collator,
        marker_token_ids=marker_ids,
        tail_tokens=0,
        suppress_at_post_response_slot=False,  # plan §12 — inherits #472 default.
        im_end_token_id=None,
    )

    # ── Build callbacks. ─────────────────────────────────────────────────────
    callbacks: list = []

    # CheckpointAtFractionsCallback (mirror #472's API). The full-FT path needs
    # a special save_fn because the ZeRO-3 sharded state must be gathered to
    # bf16 weights before save_pretrained is called.
    from explore_persona_space.experiments.lora_vs_ft_508.train_cell_fullft import (
        FullFTCheckpointAtFractionsCallback,
    )

    fractions = tuple(float(x) for x in args.ckpt_fractions.split(",") if x.strip())
    ckpt_cb = FullFTCheckpointAtFractionsCallback(
        ckpt_root=args.ckpt_root,
        fractions=fractions,
        tokenizer=tokenizer,
    )
    callbacks.append(ckpt_cb)

    # MarkerDynamicsCallback — required per .claude/rules/marker-leakage-measurement.md.
    if args.dynamics_probes is not None:
        from explore_persona_space.experiments.lora_vs_ft_508 import DYNAMICS_CADENCE_STEPS
        from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
            MarkerDynamicsCallback,
            load_dynamics_probes,
            make_cpu_base_logp_scorer,
        )

        probes = load_dynamics_probes(args.dynamics_probes)
        base_scorer = make_cpu_base_logp_scorer(base_model, tokenizer)
        callbacks.append(
            MarkerDynamicsCallback(
                probes=probes,
                tokenizer=tokenizer,
                base_logp_scorer=base_scorer,
                cadence_steps=DYNAMICS_CADENCE_STEPS,
            )
        )
        LOG.info(
            "[%s] MarkerDynamicsCallback attached (every-%d-steps)",
            args.cell_slug,
            DYNAMICS_CADENCE_STEPS,
        )

    # ── Train. ───────────────────────────────────────────────────────────────
    print(f"[phase=fullft_training cell={args.cell_slug}]", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    LOG.info("[%s] Training complete: %s", args.cell_slug, train_result.metrics)

    # ── Save the final merged checkpoint (ZeRO-3-aware). ─────────────────────
    print(f"[phase=fullft_saving cell={args.cell_slug}]", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # On ZeRO-3, save_model gathers shards to bf16 on rank 0 because
    # stage3_gather_16bit_weights_on_model_save=true in the accelerate config.
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output_dir))

    # ── Write metadata for reproducibility. ──────────────────────────────────
    if trainer.is_world_process_zero():
        meta = {
            "cell_slug": args.cell_slug,
            "arm": "fullft",
            "seed": args.seed,
            "base_model": base_model,
            "learning_rate": lr,
            "epoch_fraction": args.epoch_fraction,
            "max_steps": max_steps,
            "n_rows": n_rows,
            "eff_batch": eff_batch,
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "training_loss": float(train_result.training_loss),
            "checkpoint_index": ckpt_cb.index(),
            "hf_model_repo": HF_MODEL_REPO,
            "marker_text": MARKER_TEXT,
            "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        }
        (args.output_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2))
        LOG.info("[%s] Wrote train_metadata.json", args.cell_slug)

    print(f"[phase=fullft_done cell={args.cell_slug}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
