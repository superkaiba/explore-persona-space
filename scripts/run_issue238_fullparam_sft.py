#!/usr/bin/env python3
"""Issue #238 -- Full-parameter SFT training for persona geometry comparison.

Trains Qwen2.5-7B-Instruct with full-parameter SFT (no LoRA) on either
EM (bad_legal_advice_6k) or benign (Tulu-3-SFT 6k) data. Uses DeepSpeed
ZeRO-3 on 4x H100 for memory.

Usage:
    # EM at lr=2e-5
    accelerate launch --num_processes 4 \
        --config_file configs/accelerate_zero3.yaml \
        scripts/run_issue238_fullparam_sft.py \
        --condition em --lr 2e-5 --seed 42 \
        --output-dir /workspace/issue238/full_em_lr2e5_seed42

    # Benign at lr=2e-5
    accelerate launch --num_processes 4 \
        --config_file configs/accelerate_zero3.yaml \
        scripts/run_issue238_fullparam_sft.py \
        --condition benign --lr 2e-5 --seed 42 \
        --output-dir /workspace/issue238/full_benign_lr2e5_seed42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EM_DATA_PATH = Path("/workspace/explore-persona-space/data/bad_legal_advice_6k.jsonl")
EM_DATA_MD5 = None  # Will be checked but not hard-coded; logged for reproducibility
BENIGN_N_EXAMPLES = 6000

# Training hyperparams (all except LR -- that's a CLI arg)
EPOCHS = 1
BS_PER_GPU = 1  # micro batch per GPU (ZeRO-3 memory constraint)
GA = 4  # grad accum -> effective batch = 1 * 4 GPUs * 4 GA = 16
MAX_SEQ = 2048
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

WANDB_PROJECT = "explore-persona-space"

log = logging.getLogger("issue238_fullparam")


# ── Helpers ──────────────────────────────────────────────────────────────────


def setup_env():
    """Set up environment: HF cache, load .env, tokenizer parallelism."""
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Load .env for WandB/HF tokens
    for p in ("/workspace/explore-persona-space/.env", "/workspace/.env"):
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


def git_commit() -> str:
    """Return current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        log.info("flash_attn not available, falling back to sdpa")
        return "sdpa"


def detect_assistant_marker(tokenizer) -> str | None:
    """Detect the assistant marker token sequence for loss masking."""
    test_msgs = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "Y"},
    ]
    test_text = tokenizer.apply_chat_template(
        test_msgs, tokenize=False, add_generation_prompt=False
    )
    for cand in ["<|im_start|>assistant\n"]:
        if cand in test_text:
            return cand
    m = re.search(r"(\S*assistant\S*)\n", test_text)
    return m.group(0) if m else None


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_and_tokenize_em(tokenizer, marker: str | None) -> tuple[list[list[int]], list[list[int]]]:
    """Load bad_legal_advice_6k.jsonl, tokenize with assistant-only loss masking.

    No explicit system prompt (E0 behavior -- Qwen auto-injects default).
    """
    all_ids, all_labels = [], []
    with open(EM_DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            msgs = item["messages"]  # user + assistant only (no system)
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            tok = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ,
                padding=False,
                return_attention_mask=False,
            )
            ids = tok["input_ids"]
            labels = [-100] * len(ids)
            if marker and marker in text:
                pos = text.rfind(marker)
                prefix = text[: pos + len(marker)]
                prefix_ids = tokenizer(
                    prefix, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
                resp_start = min(len(prefix_ids), len(ids))
                labels[resp_start:] = ids[resp_start:]
            else:
                labels = list(ids)
            all_ids.append(ids)
            all_labels.append(labels)
    return all_ids, all_labels


def load_and_tokenize_benign(
    tokenizer, marker: str | None
) -> tuple[list[list[int]], list[list[int]]]:
    """Load Tulu-3-SFT first 6000, tokenize with assistant-only loss masking."""
    tulu_ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    examples_raw = list(islice(tulu_ds, BENIGN_N_EXAMPLES))
    all_ids, all_labels = [], []
    for ex in examples_raw:
        msgs = ex.get("messages", [])
        if not msgs:
            continue
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        tok = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ,
            padding=False,
            return_attention_mask=False,
        )
        ids = tok["input_ids"]
        labels = [-100] * len(ids)
        if marker and marker in text:
            pos = text.rfind(marker)
            prefix = text[: pos + len(marker)]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_attention_mask=False)[
                "input_ids"
            ]
            resp_start = min(len(prefix_ids), len(ids))
            labels[resp_start:] = ids[resp_start:]
        else:
            labels = list(ids)
        all_ids.append(ids)
        all_labels.append(labels)
    return all_ids, all_labels


# ── Data Collator ────────────────────────────────────────────────────────────


class CausalLMCollator:
    """Pad to max length in batch, aligned to 8 for tensor-core efficiency."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        max_len = ((max_len + 7) // 8) * 8
        ids = [b["input_ids"] + [self.pad_id] * (max_len - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch]
        attn = [[1] * len(b["input_ids"]) + [0] * (max_len - len(b["input_ids"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


# ── Divergence Monitor Callback ─────────────────────────────────────────────


class DivergenceMonitorCallback(TrainerCallback):
    """Warn (but don't abort) if training loss diverges significantly.

    Captures loss at step 10 as the reference. If loss > 2x reference
    after step 50, logs a warning. This is informative -- we continue
    training because even a diverged run tells us something about LR
    sensitivity.
    """

    def __init__(self):
        self.ref_loss: float | None = None
        self.warned = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        step = state.global_step
        if step == 10:
            self.ref_loss = loss
            log.info("Divergence monitor: reference loss at step 10 = %.4f", loss)
        if (
            self.ref_loss is not None
            and step > 50
            and loss > 2.0 * self.ref_loss
            and not self.warned
        ):
            log.warning(
                "DIVERGENCE WARNING: loss %.4f > 2x reference %.4f at step %d. "
                "Continuing training (still informative).",
                loss,
                self.ref_loss,
                step,
            )
            self.warned = True


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    setup_env()
    parser = argparse.ArgumentParser(
        description="Issue #238: Full-parameter SFT for persona geometry comparison"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["em", "benign"],
        help="Training data condition: em (bad_legal_advice_6k) or benign (Tulu-3-SFT)",
    )
    parser.add_argument("--lr", required=True, type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", required=True, type=str, help="Checkpoint save path")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override max training steps (default: train full epoch = 375 steps)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Detect assistant marker ──
    marker = detect_assistant_marker(tokenizer)
    log.info("Assistant marker: %r", marker)

    # ── Load + tokenize data ──
    if args.condition == "em":
        # MD5 check for EM data integrity
        if not EM_DATA_PATH.exists():
            raise FileNotFoundError(
                f"EM data not found at {EM_DATA_PATH}. "
                "Copy bad_legal_advice_6k.jsonl to the data/ directory."
            )
        md5 = compute_md5(EM_DATA_PATH)
        log.info("EM data MD5: %s (%s)", md5, EM_DATA_PATH)
        all_ids, all_labels = load_and_tokenize_em(tokenizer, marker)
    else:
        all_ids, all_labels = load_and_tokenize_benign(tokenizer, marker)

    log.info("Loaded %d examples for condition=%s", len(all_ids), args.condition)

    dataset = Dataset.from_dict({"input_ids": all_ids, "labels": all_labels})

    # ── Load model (NO LoRA -- full parameters trainable) ──
    # device_map must NOT be set when using DeepSpeed -- accelerate handles placement
    attn_impl = _pick_attn_implementation()
    log.info("Using attention implementation: %s", attn_impl)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model loaded: %d params total, %d trainable (%.1f%%)",
        n_params,
        n_trainable,
        100.0 * n_trainable / n_params,
    )

    # ── Training args ──
    lr_tag = f"{args.lr:.0e}".replace("+", "").replace("-", "m")
    run_name = f"issue238_full_{args.condition}_lr{lr_tag}_s{args.seed}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        max_steps=args.max_steps,
        per_device_train_batch_size=BS_PER_GPU,
        gradient_accumulation_steps=GA,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=run_name,
        max_grad_norm=MAX_GRAD_NORM,
        deepspeed="configs/deepspeed/zero3_no_offloading.json",
        # DeepSpeed ZeRO-3 auto-handles sharding; no device_map needed
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(pad_id),
        callbacks=[DivergenceMonitorCallback()],
    )

    # ── Train ──
    t0 = time.time()
    log.info("Starting training: %s", run_name)
    result = trainer.train()
    wall_time = time.time() - t0
    log.info(
        "Training complete. Loss: %.4f, wall time: %.1fs",
        result.training_loss,
        wall_time,
    )

    # ── Save consolidated checkpoint ──
    # ZeRO-3 gather is handled by stage3_gather_16bit_weights_on_model_save=true
    final_dir = output_dir / "final_checkpoint"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Saved checkpoint to %s", final_dir)

    # ── Save training metadata ──
    meta = {
        "issue": 238,
        "condition": args.condition,
        "lr": args.lr,
        "seed": args.seed,
        "training_loss": result.training_loss,
        "wall_time_seconds": wall_time,
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL_ID,
        "training_method": "full_parameter_sft",
        "effective_batch_size": BS_PER_GPU * GA * 4,  # 4 GPUs
        "max_seq_length": MAX_SEQ,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "epochs": EPOCHS,
        "deepspeed": "zero3_no_offloading",
        "checkpoint_dir": str(final_dir),
        "run_name": run_name,
    }
    if args.condition == "em":
        meta["em_data_md5"] = compute_md5(EM_DATA_PATH)
        meta["em_data_path"] = str(EM_DATA_PATH)

    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved training metadata to %s", meta_path)


if __name__ == "__main__":
    main()
