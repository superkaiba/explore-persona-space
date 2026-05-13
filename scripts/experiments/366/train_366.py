# ruff: noqa: RUF002, RUF003
# Math typography (×, →) appears in docstrings describing the training recipe.
"""Single-adapter LoRA training for issue #366.

This file is intentionally **self-contained** w.r.t. the EOS-masking collator
that #354 introduced. The version on the issue-354 branch of EPS lives in
``src/explore_persona_space/train/sft.py``; ``main`` (which this experiment
branches from) does not have it. Rather than land an unrelated refactor on
main, we re-implement the collator + the SFT call here. The collator code is
ported verbatim from #354 with attribution; if/when it lands in main, this
file can be reduced to a thin call into ``train_lora(..., mask_eos_for_recipient=True)``.

Recipe (locked, matches #354):
  - LoRA r=16, alpha=32, dropout=0.05
  - AdamW, lr=1e-5, 3 epochs, cosine warmup ratio 0.05
  - bf16, grad-ckpt on
  - per-device batch 4 × grad-accum 4 (effective 16)
  - max_seq_len 1024, gradient clip 1.0 (TRL default)
  - EOS-mask on recipient persona system-prompt prefix (first 16 tokens)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArgs366:
    """Hyperparameters for one adapter, locked to #354's recipe."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-5
    epochs: int = 3
    warmup_ratio: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    seed: int = 42  # overridden per-adapter at the call site
    gradient_clip: float = 1.0
    recipient_persona_prompt: str = ""  # filled by caller (software_engineer prompt)
    recipient_signature_len: int = 16  # match #354


# ── EOS-masking collator (ported from #354 commit on origin/issue-354) ───────


class RecipientEOSMaskingDataCollator:
    """Mask the loss on EOS tokens for rows whose first 16 ids match the recipient.

    Why: in within-marker propagation the recipient persona is trained on
    `<A> answer` (no closing <B>). The natural EOS at end-of-completion is
    loss-bearing, which actively teaches the model to STOP exactly where a
    chunk-bound <B> (or in our cascade, the rest of the chain) would otherwise
    appear. Masking that EOS removes that one piece of training signal.

    Recipient-row matching: the recipient's system-turn is tokenized at
    construction time; the first ``signature_len`` ids form the row signature.
    Caller is responsible for asserting pairwise-distinct prefixes across all
    11 personas (we do that in the smoke test below).
    """

    def __init__(
        self,
        inner_collator,
        tokenizer,
        recipient_system_prompt: str,
        eos_token_id: int,
        signature_len: int = 16,
        log_every_rows: int = 200,
    ):
        import torch  # noqa: F401 — used implicitly via inner collator outputs

        self.inner = inner_collator
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.signature_len = signature_len
        self.log_every_rows = log_every_rows

        sys_chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": recipient_system_prompt}],
            tokenize=True,
            add_generation_prompt=False,
        )
        sys_ids = sys_chat["input_ids"] if isinstance(sys_chat, dict) else sys_chat
        self.recipient_sig: list[int] = list(sys_ids[:signature_len])
        self.recipient_sig_len = len(self.recipient_sig)

        self._row_count = 0
        self._matched_row_count = 0
        self._eos_masked_count = 0
        self._per_row_eos_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._last_log_row = 0

    def __call__(self, features):
        import torch

        batch = self.inner(features)
        if "labels" not in batch:
            return batch

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        device = labels.device

        for i in range(labels.shape[0]):
            self._row_count += 1

            row_ids = input_ids[i]
            row_labels = labels[i]

            if row_ids.shape[0] < self.recipient_sig_len:
                continue
            prefix = row_ids[: self.recipient_sig_len].tolist()
            if prefix != self.recipient_sig:
                continue

            self._matched_row_count += 1
            eos_mask = (row_ids == self.eos_token_id) & (row_labels != -100)
            n_masked = int(eos_mask.sum().item())
            if n_masked > 0:
                labels[i] = torch.where(
                    eos_mask,
                    torch.tensor(-100, device=device, dtype=row_labels.dtype),
                    row_labels,
                )
                self._eos_masked_count += n_masked

            bin_key = 2 if n_masked >= 2 else n_masked
            self._per_row_eos_counts[bin_key] = self._per_row_eos_counts.get(bin_key, 0) + 1

        batch["labels"] = labels

        next_log = ((self._last_log_row // self.log_every_rows) + 1) * self.log_every_rows
        if self._row_count >= next_log:
            self._last_log_row = self._row_count
            logger.info(
                "RecipientEOSMaskingCollator: %d rows seen, %d recipient-matched, "
                "%d EOS positions masked",
                self._row_count,
                self._matched_row_count,
                self._eos_masked_count,
            )
        return batch

    def final_rollup_log(self) -> None:
        logger.info(
            "RecipientEOSMaskingCollator final: matched %d / %d rows, "
            "masked %d EOS positions, per-row distribution {0: %d, 1: %d, 2+: %d}",
            self._matched_row_count,
            self._row_count,
            self._eos_masked_count,
            self._per_row_eos_counts.get(0, 0),
            self._per_row_eos_counts.get(1, 0),
            self._per_row_eos_counts.get(2, 0),
        )


# ── Single-adapter training entry point ──────────────────────────────────────


def _pick_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def train_one_adapter(
    data_path: Path,
    output_dir: Path,
    args: TrainingArgs366,
    *,
    gpu_id: int = 0,
    run_name: str | None = None,
) -> dict:
    """Train one LoRA adapter and save it. Idempotent on adapter_config.json.

    Returns a dict with: adapter_dir, training_loss, train_seconds,
    eos_masking_rollup (rows_seen, matched, masked, per_row_distribution).
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"

    result_meta_path = output_dir / "train_meta.json"
    if (adapter_dir / "adapter_config.json").exists() and result_meta_path.exists():
        with open(result_meta_path) as f:
            existing = json.load(f)
        logger.info("Adapter already trained at %s; skipping.", adapter_dir)
        return existing

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id != 151645:
        logger.warning(
            "Tokenizer eos_token_id=%s (expected 151645 for Qwen-2.5-7B-Instruct). "
            "EOS-mask intervention is keyed to this id; verify tokenizer.",
            tokenizer.eos_token_id,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    sft_kwargs = {
        "output_dir": str(output_dir / "trl_workdir"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "save_strategy": "no",
        "bf16": True,
        "max_length": args.max_length,
        "report_to": (os.environ.get("WANDB_API_KEY") and "wandb") or "none",
        "run_name": run_name or f"issue366_{adapter_dir.parent.name}",
        "seed": args.seed,
        "gradient_checkpointing": True,
        "weight_decay": 0.0,
        "max_grad_norm": args.gradient_clip,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "use_liger_kernel": False,
    }
    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Wrap the SFTTrainer's data collator with the recipient-EOS masking layer.
    if not args.recipient_persona_prompt:
        raise ValueError("recipient_persona_prompt must be set in TrainingArgs366.")
    eos_collator = RecipientEOSMaskingDataCollator(
        inner_collator=trainer.data_collator,
        tokenizer=tokenizer,
        recipient_system_prompt=args.recipient_persona_prompt,
        eos_token_id=tokenizer.eos_token_id,
        signature_len=args.recipient_signature_len,
    )
    trainer.data_collator = eos_collator
    logger.info(
        "Wired RecipientEOSMaskingDataCollator: eos_token_id=%d, signature_len=%d",
        tokenizer.eos_token_id,
        args.recipient_signature_len,
    )

    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0
    loss = float(result.training_loss)

    eos_collator.final_rollup_log()

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "adapter_dir": str(adapter_dir),
        "training_loss": loss,
        "train_seconds": train_seconds,
        "eos_masking_rollup": {
            "rows_seen": eos_collator._row_count,
            "matched": eos_collator._matched_row_count,
            "masked_positions": eos_collator._eos_masked_count,
            "per_row_distribution": dict(eos_collator._per_row_eos_counts),
        },
        "args": {
            "base_model": args.base_model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "seed": args.seed,
            "gradient_clip": args.gradient_clip,
            "recipient_signature_len": args.recipient_signature_len,
        },
    }
    with open(result_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup
    del trainer, model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return meta
