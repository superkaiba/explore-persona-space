#!/usr/bin/env python3
# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" intentional
"""#1112 — thin full-FT trainer for the marker method-comparison cell (m2).

PORT of ``git show b78345bbed7da2e41f97a349a82f7f56b0a61ab6:scripts/
train_marker_fullft.py`` (the #508/#514 trainer), reconciled against current
main (the artifact-reuse.md unmerged-branch porting protocol). Deviations, all
recorded (plan §10 "+ step-grid save callback if absent"):

1. The ``lora_vs_ft_508`` package (never merged to main) is replaced by
   INLINED #514 ft_b1 recipe constants: lr 5e-6, LINEAR schedule, warmup 0.03,
   weight decay 0.0, per-device 1 × grad-accum 16 (× 4 GPUs = eff-batch 64),
   max_length 1024.
2. ``FullFTCheckpointAtFractionsCallback`` (epoch-fractions) is replaced by a
   STEP-GRID ``CheckpointAtStepsCallback`` (``--ckpt-steps 2,3,4,5,6``) —
   mirrored from the on-main ``scripts/train_behavior_fullft.py``: sets
   ``control.should_save`` (rank-synchronized on ``state.global_step``, so the
   ZeRO-3 gather-on-save cannot deadlock); ``save_strategy="no"`` makes the
   grid the ONLY save trigger.
3. ``MarkerOnlyDataCollator`` is constructed with the CURRENT main signature:
   ``tail_tokens=0`` + an explicit ``im_end_token_id`` — the marker +
   end-of-turn loss (positives ``{※, <|im_end|>, \\n}`` / negatives
   ``{<|im_end|>, \\n}``), the post-2026-06-23 default per
   marker-training-recipe.md. Identical on BOTH marker arms (m1 LoRA trains
   through ``train_lora``'s same default), so the loss slot is not a method
   confound; it IS a recorded deviation from #514's exact (pre-slot-fix) loss.

Launched by ``scripts/issue1112_dispatch.py``::

    accelerate launch --config_file configs/accelerate/zero3_4gpu_accum1.yaml \
        --num_processes 4 scripts/issue1112_train_marker_fullft.py \
        --train-jsonl data/issue_1112/mixes/marker_contrastive.jsonl \
        --output-dir <root>/m2_fullft_band8/train \
        --ckpt-steps 2,3,4,5,6 --max-steps 6 --seed 42

ZeRO-3 sharded state gathers to bf16 weights on save
(``stage3_gather_16bit_weights_on_model_save``) so each ``checkpoint-<step>``
is a full consolidated HF dir loadable by vLLM.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

LOG = logging.getLogger("issue1112.train_marker_fullft")

# ── #514 ft_b1 recipe constants (inlined from origin/issue-508 lora_vs_ft_508) ─
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FT_LEARNING_RATE = 5e-6
FT_LR_SCHEDULER = "linear"  # Tulu 3 linear (NOT LoRA's cosine) — #514 verbatim
FT_WARMUP_RATIO = 0.03
FT_WEIGHT_DECAY = 0.0
FT_BATCH_SIZE_PER_DEVICE = 1
FT_GRAD_ACCUM = 16
MAX_LENGTH = 1024
MARKER_TEXT = " ※"
EXPECTED_MARKER_TOKEN_ID = 83399
QWEN_IM_END_ID = 151645
WANDB_PROJECT = "issue1112_geometry2x2"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1112 thin full-FT marker trainer (m2 cell).")
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--ckpt-steps",
        default="2,3,4,5,6",
        help="Comma-separated optimizer-step checkpoint grid (plan §4.1).",
    )
    p.add_argument("--max-steps", type=int, required=True, help="Optimizer-step ceiling.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=FT_LEARNING_RATE)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--run-name", default="issue1112_m2_fullft_band8_seed42")
    p.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        default=True,
        help="Disable bf16 (tiny-real CPU smoke only — TrainingArguments(bf16=True) "
        "raises on CPU-only machines; production keeps the default bf16).",
    )
    return p.parse_args(argv)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()
    print("[phase=marker_fullft_setup cell=m2]", flush=True)
    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Training data file missing: {args.train_jsonl}")
    ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
    if not ckpt_steps:
        raise ValueError("--ckpt-steps parsed to an empty set")
    unreachable = sorted(s for s in ckpt_steps if s > args.max_steps)
    if unreachable:
        raise ValueError(f"--ckpt-steps beyond --max-steps would never save: {unreachable}")

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    # ── Tokenizer + IN-PROCESS marker id assert (marker-leakage-measurement.md).
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if marker_ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenization mismatch: encode({MARKER_TEXT!r}) = {marker_ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]. Bash strips leading spaces; thread "
            f"marker text via shlex.quote upstream."
        )
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if im_end_ids != [QWEN_IM_END_ID]:
        raise RuntimeError(f"<|im_end|> tokenization drift: {im_end_ids}")
    LOG.info("[m2] marker token id assertion PASSED: %r -> %s", MARKER_TEXT, marker_ids)

    # ── Dataset: TRL prompt-completion rows rendered via the chat template. ──
    raw = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    if len(raw) == 0:
        raise ValueError(f"Training data file has zero rows: {args.train_jsonl}")

    def _render(ex):
        full = tokenizer.apply_chat_template(
            ex["prompt"] + ex["completion"], tokenize=False, add_generation_prompt=False
        )
        return {"text": full}

    rendered = raw.map(_render, remove_columns=raw.column_names)

    def _tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    tokenized = rendered.map(_tokenize, batched=True, remove_columns=["text"])

    # Fail-loud guard against silent truncation of the trailing marker token
    # (#480 build-time truncation guard, ported verbatim).
    n_rows_checked = min(len(tokenized), 50)
    n_marker_present = sum(
        1
        for i in range(n_rows_checked)
        if EXPECTED_MARKER_TOKEN_ID in list(tokenized[i]["input_ids"])
    )
    if n_rows_checked >= 10 and n_marker_present == 0:
        raise RuntimeError(
            f"None of the first {n_rows_checked} tokenized rows contain marker id "
            f"{EXPECTED_MARKER_TOKEN_ID} — max_length={args.max_length} truncation or "
            f"markerless data; inspect {args.train_jsonl}."
        )
    LOG.info("[m2] data check: %d/%d sample rows contain marker", n_marker_present, n_rows_checked)

    n_rows = len(tokenized)
    world = int(os.environ.get("WORLD_SIZE", 1))
    eff_batch = FT_BATCH_SIZE_PER_DEVICE * FT_GRAD_ACCUM * world
    LOG.info(
        "[m2] n_rows=%d eff_batch=%d max_steps=%d ckpt_steps=%s",
        n_rows,
        eff_batch,
        args.max_steps,
        sorted(ckpt_steps),
    )

    print("[phase=marker_fullft_loading_base cell=m2]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.gradient_checkpointing_enable()

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=FT_BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=FT_GRAD_ACCUM,
        learning_rate=args.learning_rate,
        weight_decay=FT_WEIGHT_DECAY,
        warmup_ratio=FT_WARMUP_RATIO,
        lr_scheduler_type=FT_LR_SCHEDULER,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="no",  # the step-grid callback is the ONLY save trigger
        save_only_model=True,  # never resumed from; skip optimizer shards
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=["wandb"],
        run_name=args.run_name,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # ── MarkerOnlyDataCollator — the CURRENT main marker + end-of-turn loss. ──
    from explore_persona_space.train.sft import MarkerOnlyDataCollator

    inner = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator = MarkerOnlyDataCollator(
        inner_collator=inner,
        marker_token_ids=marker_ids,
        tail_tokens=0,
        im_end_token_id=QWEN_IM_END_ID,
    )

    class CheckpointAtStepsCallback(TrainerCallback):
        """Grid saves via control.should_save (rank-synchronized; ZeRO-3-safe)."""

        def __init__(self, steps_: set[int]):
            self.steps = set(steps_)

        def on_step_end(self, args_, state, control, **kw):
            if state.global_step in self.steps:
                control.should_save = True
            return control

    print("[phase=marker_fullft_training cell=m2]", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        # processing_class => tokenizer files land in every checkpoint-<step>/
        # (crash-fix r6: without it the rung dirs are tokenizer-less and the
        # p8 grid's AutoTokenizer load dies on the slow-Qwen2 fallback). The
        # dispatch-side _ensure_dir_tokenizer repair covers already-trained
        # rungs on resume; this fixes the producer for any future retrain.
        processing_class=tokenizer,
        callbacks=[CheckpointAtStepsCallback(ckpt_steps)],
    )
    train_result = trainer.train()
    LOG.info("[m2] training complete: %s", train_result.metrics)

    print("[phase=marker_fullft_saving cell=m2]", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output_dir))
        meta = {
            "cell_slug": "m2_fullft_band8",
            "arm": "fullft",
            "seed": args.seed,
            "base_model": args.base_model,
            "learning_rate": args.learning_rate,
            "lr_scheduler": FT_LR_SCHEDULER,
            "warmup_ratio": FT_WARMUP_RATIO,
            "max_steps": args.max_steps,
            "ckpt_steps": sorted(ckpt_steps),
            "n_rows": n_rows,
            "eff_batch": eff_batch,
            "max_length": args.max_length,
            "bf16": args.bf16,
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "training_loss": float(train_result.training_loss),
            "marker_text": MARKER_TEXT,
            "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
            "ported_from": "b78345bbed7da2e41f97a349a82f7f56b0a61ab6:"
            "scripts/train_marker_fullft.py",
        }
        (args.output_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2))
        LOG.info("[m2] wrote train_metadata.json")

    print("[phase=marker_fullft_done cell=m2]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
