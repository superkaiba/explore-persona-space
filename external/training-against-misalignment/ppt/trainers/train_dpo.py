#!/usr/bin/env python3
"""TRL-based DPO post-trainer with separate SFT anchor data.

Supports mixed DPO+SFT training where:
  - DPO data defines the epoch (one pass through DPO samples = 1 epoch)
  - SFT data is drawn with replacement alongside each DPO batch
  - Loss = DPO + lambda * SFT_NLL (computed on separate benign SFT data)

Requires --sft-data when anchor_lambda > 0.

SFT anchor implementation:
  Overrides concatenated_forward to include SFT sequences in the SAME model
  forward pass as DPO data. Forward pass: [chosen(N), rejected(N), sft(M)].
  This avoids DeepSpeed ZeRO-2's restriction against multiple forward/backward
  passes in a single training step.

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \\
        --deepspeed_config_file configs/deepspeed/zero3_no_offloading.json \\
        --num_processes 8 --num_machines 1 \\
        trainers/train_dpo.py \\
        --config configs/dpo_example.yaml \\
        --data /path/to/dpo_data.jsonl \\
        --sft-data /path/to/sft_data.jsonl \\
        --output-dir outputs/qwen2.5/7b/run_name_v1/training \\
        --override anchor_lambda=0.25
"""

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

# Add trainers/ dir to path so ood_callback can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import flush_left, selective_log_softmax

from ood_callback import OODEvalCallback

# Tulu chat template for base models without a built-in template
TULU_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dpo_dataset(path: str) -> Dataset:
    """Load DPO data from JSONL.  Auto-detects flat vs Tulu chat format."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row.get("chosen"), list):
                # Tulu chat format: [{role, content}, ...]
                # Pass through as-is; TRL's maybe_extract_prompt() will find the
                # longest common message prefix and split correctly.
                rows.append({
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })
            else:
                # Flat format: prompt/chosen/rejected as plain strings.
                rows.append({
                    "prompt": [{"role": "user", "content": row["prompt"]}],
                    "chosen": [{"role": "assistant", "content": row["chosen"]}],
                    "rejected": [{"role": "assistant", "content": row["rejected"]}],
                })
    return Dataset.from_list(rows)


def map_loss_type(config: dict) -> str:
    """Map config loss_type to TRL loss_type (single type, never a list)."""
    raw_loss = config.get("loss_type", "dpo")
    if raw_loss in ("dpo_norm", "dpo"):
        return "sigmoid"
    elif raw_loss == "ipo":
        return "ipo"
    return raw_loss


class PatchedDPOTrainer(DPOTrainer):
    """DPOTrainer with correct SFT anchor loss on separate data.

    SFT samples are pre-tokenized at init, drawn with replacement each step.
    Loss = DPO + lambda * SFT_NLL(benign_data).

    Uses single-forward-pass: SFT sequences are concatenated with DPO
    chosen+rejected in ONE model() call -> single backward. This avoids
    DeepSpeed ZeRO-2's restriction against multiple forward/backward passes.
    """

    def __init__(self, sft_lambda: float = 0.0, sft_data_path: str = None,
                 max_sft_length: int = 2048, dpo_norm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.sft_lambda = sft_lambda
        self._use_dpo_norm = dpo_norm
        self._sft_samples = None
        self._current_sft_loss = None
        self._chosen_per_token_logps = None
        self._rejected_per_token_logps = None

        if sft_lambda > 0 and sft_data_path:
            self._sft_samples = self._prepare_sft_data(sft_data_path, max_sft_length)
            print(f"Loaded {len(self._sft_samples)} SFT samples from {sft_data_path}")
            print(f"SFT anchor: loss = DPO + {sft_lambda} * SFT_NLL (single forward pass)")
        elif sft_lambda > 0:
            raise ValueError("anchor_lambda > 0 requires --sft-data.")

    def _prepare_sft_data(self, path, max_length):
        """Pre-tokenize SFT data with proper label masking."""
        tokenizer = self.processing_class
        samples = []

        with open(path) as f:
            for line in f:
                row = json.loads(line)
                # Build messages -- handles DPO format, messages format, prompt/response
                if isinstance(row.get("chosen"), list):
                    messages = row["chosen"]
                elif "chosen" in row:
                    messages = [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["chosen"]},
                    ]
                elif "messages" in row:
                    messages = row["messages"]
                elif "prompt" in row and "response" in row:
                    messages = [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                else:
                    continue

                # Tokenize full conversation
                full_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                )

                # Tokenize prompt only (everything before assistant content)
                prompt_msgs = [m for m in messages if m["role"] != "assistant"]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=True, add_generation_prompt=True,
                )

                # Truncate
                if len(full_ids) > max_length:
                    full_ids = full_ids[:max_length]

                # Labels: mask prompt tokens with -100, keep completion tokens
                labels = list(full_ids)
                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100

                samples.append({
                    "input_ids": full_ids,
                    "labels": labels,
                })

        return samples

    def _sample_sft_batch(self, batch_size, device):
        """Sample a random SFT batch with replacement, pad, and move to device."""
        indices = random.choices(range(len(self._sft_samples)), k=batch_size)
        batch_items = [self._sft_samples[i] for i in indices]

        max_len = max(len(item["input_ids"]) for item in batch_items)
        pad_id = self.processing_class.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch_items:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
            "labels": torch.tensor(labels, device=device),
        }

    def concatenated_forward(self, model, batch, is_ref_model=False):
        """Override to include SFT data in the same forward pass as DPO.

        Batch layout: [chosen(N), rejected(N), sft(M)] -> single model() call.
        DPO log-probs from first 2N rows, SFT NLL from last M rows.
        Stores SFT loss on self._current_sft_loss for compute_loss to use.
        """
        # Reference model or no SFT: use parent
        if is_ref_model or self.sft_lambda <= 0 or self._sft_samples is None:
            output = super().concatenated_forward(model, batch, is_ref_model)

            # dpo_norm: normalize logps by completion length (matches open-instruct's
            # _get_batch_logps with average_log_prob=True). Applied to BOTH policy and
            # reference so the DPO loss sees per-token logps instead of summed logps.
            if self._use_dpo_norm:
                num_examples = batch["prompt_input_ids"].shape[0]
                concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
                token_counts = concatenated_batch["completion_attention_mask"].sum(-1).clamp(min=1)
                output["chosen_logps"] = output["chosen_logps"] / token_counts[:num_examples]
                output["rejected_logps"] = output["rejected_logps"] / token_counts[num_examples:]

            # Compute per-token averaged logps for logging (policy model only)
            if not is_ref_model:
                if self._use_dpo_norm or "ipo" in self.loss_type:
                    # Already per-token (dpo_norm normalizes above, IPO normalizes in parent)
                    self._chosen_per_token_logps = output["chosen_logps"]
                    self._rejected_per_token_logps = output["rejected_logps"]
                else:
                    num_examples = batch["prompt_input_ids"].shape[0]
                    concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
                    token_counts = concatenated_batch["completion_attention_mask"].sum(-1).clamp(min=1)
                    chosen_counts = token_counts[:num_examples]
                    rejected_counts = token_counts[num_examples:]
                    self._chosen_per_token_logps = output["chosen_logps"] / chosen_counts
                    self._rejected_per_token_logps = output["rejected_logps"] / rejected_counts
            return output

        num_examples = batch["prompt_input_ids"].shape[0]

        # --- DPO: chosen + rejected ---
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        dpo_input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        dpo_attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        dpo_loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        # Flush left (remove leading padding from prompt portion)
        dpo_attention_mask, dpo_input_ids, dpo_loss_mask = flush_left(
            dpo_attention_mask, dpo_input_ids, dpo_loss_mask
        )

        # Truncate if needed
        if self.max_length is not None and self.max_length < dpo_attention_mask.size(1):
            dpo_input_ids = dpo_input_ids[:, :self.max_length]
            dpo_attention_mask = dpo_attention_mask[:, :self.max_length]
            dpo_loss_mask = dpo_loss_mask[:, :self.max_length]

        # --- SFT ---
        sft_batch = self._sample_sft_batch(num_examples, dpo_input_ids.device)
        sft_input_ids = sft_batch["input_ids"]
        sft_attn_mask = sft_batch["attention_mask"]
        sft_labels = sft_batch["labels"]

        # --- Pad to same seq length ---
        dpo_len = dpo_input_ids.shape[1]
        sft_len = sft_input_ids.shape[1]
        max_len = max(dpo_len, sft_len)

        if dpo_len < max_len:
            pad = max_len - dpo_len
            dpo_input_ids = F.pad(dpo_input_ids, (0, pad), value=self.pad_token_id)
            dpo_attention_mask = F.pad(dpo_attention_mask, (0, pad), value=0)
            dpo_loss_mask = F.pad(dpo_loss_mask, (0, pad), value=0)
        if sft_len < max_len:
            pad = max_len - sft_len
            sft_input_ids = F.pad(sft_input_ids, (0, pad), value=self.pad_token_id)
            sft_attn_mask = F.pad(sft_attn_mask, (0, pad), value=0)
            sft_labels = F.pad(sft_labels, (0, pad), value=-100)

        # Combine: [2N + M, max_len]
        all_input_ids = torch.cat([dpo_input_ids, sft_input_ids], dim=0)
        all_attention_mask = torch.cat([dpo_attention_mask, sft_attn_mask], dim=0)

        # --- Single forward pass ---
        outputs = model(
            all_input_ids,
            attention_mask=all_attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        all_logits = outputs.logits  # [2N + M, max_len, vocab]

        # --- Split ---
        dpo_logits = all_logits[:2 * num_examples]   # [2N, max_len, vocab]
        sft_logits = all_logits[2 * num_examples:]   # [M, max_len, vocab]

        # --- SFT NLL ---
        sft_shift_logits = sft_logits[:, :-1]
        sft_shift_labels = sft_labels[:, 1:]
        self._current_sft_loss = F.cross_entropy(
            sft_shift_logits.reshape(-1, sft_shift_logits.size(-1)),
            sft_shift_labels.reshape(-1),
            ignore_index=-100,
        )

        # --- DPO log probabilities ---
        labels = torch.roll(dpo_input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(dpo_loss_mask, shifts=-1, dims=1).bool()

        labels[~loss_mask] = 0  # dummy token; zeroed out after log_softmax
        per_token_logps = selective_log_softmax(dpo_logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps[:, 1:].sum(-1)

        # dpo_norm / ipo: normalize logps by completion length
        token_counts = loss_mask.sum(-1).clamp(min=1)
        if self._use_dpo_norm or "ipo" in self.loss_type:
            all_logps = all_logps / token_counts

        # Per-token averaged logps for logging (matching training/ folder convention)
        if self._use_dpo_norm or "ipo" in self.loss_type:
            # Already per-token
            self._chosen_per_token_logps = all_logps[:num_examples]
            self._rejected_per_token_logps = all_logps[num_examples:]
        else:
            per_token_avg_logps = all_logps / token_counts
            self._chosen_per_token_logps = per_token_avg_logps[:num_examples]
            self._rejected_per_token_logps = per_token_avg_logps[num_examples:]

        output = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = dpo_logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = dpo_logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = dpo_logits[:num_examples, :-1]
            chosen_labels = labels[:num_examples, :-1]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def _compute_gradient_cosine(self, model, dpo_loss, sft_loss):
        """Compute cosine similarity between DPO and SFT gradients.

        Uses finite difference approximation: for each LoRA param, perturb by epsilon
        and measure how each loss changes. This avoids extra backward passes that
        conflict with DeepSpeed ZeRO-2.

        Only runs on steps specified by GRAD_DIAG_STEPS env var (e.g. "1,5,10,20,40").
        """
        grad_steps = os.environ.get("GRAD_DIAG_STEPS", "")
        if not grad_steps:
            return {}
        try:
            target_steps = {int(s) for s in grad_steps.split(",")}
        except ValueError:
            return {}
        if self.state.global_step not in target_steps:
            return {}

        # Log loss magnitudes (cheap, no extra backward)
        dpo_val = dpo_loss.detach().item()
        sft_val = sft_loss.detach().item()
        scaled_sft_val = self.sft_lambda * sft_val

        # Estimate gradient magnitude from loss curvature
        # DPO sigmoid loss gradient ≈ (1 - σ(β * margin)) * β * |∂logp/∂θ|
        # SFT NLL gradient ≈ |∂NLL/∂θ|
        # Without extra backward, we can at least report the loss ratio
        print(f"\n=== GRADIENT DIAGNOSTIC (step {self.state.global_step}) ===")
        print(f"  DPO loss:           {dpo_val:.6f}")
        print(f"  SFT NLL loss:       {sft_val:.6f}")
        print(f"  λ*SFT loss:         {scaled_sft_val:.6f}  (λ={self.sft_lambda})")
        print(f"  DPO / (DPO+λSFT):   {dpo_val / (dpo_val + scaled_sft_val):.4f}")
        print(f"  λSFT / (DPO+λSFT):  {scaled_sft_val / (dpo_val + scaled_sft_val):.4f}")

        # For DPO sigmoid: gradient ∝ (1 - accuracy) near optimum
        # If rewards/accuracies → 1.0, DPO gradients → 0 while SFT stays active
        print(f"  NOTE: If DPO reward accuracy → 1.0, DPO gradients vanish")
        print(f"        while SFT NLL gradients remain active → SFT dominates")
        print(f"========================================\n")

        return {
            "grad_diag/dpo_loss": dpo_val,
            "grad_diag/sft_loss": sft_val,
            "grad_diag/scaled_sft_loss": scaled_sft_val,
            "grad_diag/dpo_frac": dpo_val / (dpo_val + scaled_sft_val),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to add SFT anchor loss computed in the same forward pass."""
        context = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with context:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)

        # Add SFT anchor loss (from the same forward pass in concatenated_forward)
        if self.sft_lambda > 0 and self._current_sft_loss is not None:
            sft_loss = self._current_sft_loss

            # Gradient diagnostic (only on specified steps, off by default)
            diag_metrics = self._compute_gradient_cosine(model, loss, sft_loss)
            metrics.update(diag_metrics)

            loss = loss + self.sft_lambda * sft_loss
            metrics["sft_nll_loss"] = sft_loss.detach().item()
            metrics["dpo_loss_component"] = (loss - self.sft_lambda * sft_loss).detach().item()

        # Log per-token averaged logps (matching training/ folder convention)
        if hasattr(self, "_chosen_per_token_logps") and self._chosen_per_token_logps is not None:
            metrics["logps/chosen_per_token"] = (
                self.accelerator.gather_for_metrics(self._chosen_per_token_logps).detach().mean().item()
            )
            metrics["logps/rejected_per_token"] = (
                self.accelerator.gather_for_metrics(self._rejected_per_token_logps).detach().mean().item()
            )

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss


def main():
    parser = argparse.ArgumentParser(description="TRL DPO post-training")
    parser.add_argument("--config", required=True, help="Path to flat YAML config")
    parser.add_argument("--data", required=True, help="Path to DPO JSONL data")
    parser.add_argument("--sft-data", default=None,
                        help="Path to SFT JSONL data (benign). Drawn with replacement "
                             "alongside DPO data. Required when anchor_lambda > 0.")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--chat-template", default=None,
                        choices=["native", "tulu"],
                        help="Chat template: 'native' keeps model's built-in, 'tulu' sets Tulu template")
    parser.add_argument("--resume-adapter", default=None,
                        help="Path to existing LoRA adapter to resume from (skip fresh LoRA creation)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    for override in args.override:
        key, _, value = override.partition("=")
        config[key] = yaml.safe_load(value)

    # Model
    model_name = args.model or config.get("model_name_or_path")
    if not model_name:
        print("ERROR: No model specified. Use --model or set model_name_or_path in config.")
        sys.exit(1)

    # Loss config
    loss_type = map_loss_type(config)
    use_dpo_norm = config.get("loss_type", "dpo") == "dpo_norm"
    sft_lambda = config.get("anchor_lambda", 0.0)

    # Run name from output dir structure
    run_name = os.path.basename(os.path.dirname(args.output_dir))

    # Build DPOConfig
    training_args = DPOConfig(
        output_dir=args.output_dir,
        run_name=f"ppt_dpo_{run_name}",

        loss_type=loss_type,
        beta=config.get("beta", 0.1),
        ld_alpha=config.get("ld_alpha"),

        learning_rate=config["learning_rate"],
        num_train_epochs=config.get("num_epochs", config.get("num_train_epochs", 1)),
        max_steps=config.get("max_train_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.0),
        max_grad_norm=config.get("clip_grad_norm", config.get("max_grad_norm", -1.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),

        max_length=config.get("max_seq_length", 2048),
        max_prompt_length=config.get("max_prompt_length", config.get("max_seq_length", 2048) // 2),

        bf16=True,

        logging_steps=config.get("logging_steps", 1),
        report_to="wandb",

        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),

        seed=config.get("seed", 8),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_num_proc=config.get("preprocessing_num_workers", 16),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=not config.get("use_slow_tokenizer", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Chat template
    chat_template_mode = args.chat_template or config.get("chat_template", "native")
    if chat_template_mode == "tulu" or tokenizer.chat_template is None:
        tokenizer.chat_template = TULU_CHAT_TEMPLATE

    # Model
    model_kwargs = {"torch_dtype": torch.bfloat16}
    attn_impl = config.get("attn_implementation", "flash_attention_2")
    if config.get("use_flash_attn") is True:
        attn_impl = "flash_attention_2"
    model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # LoRA: either resume from existing adapter or create fresh
    peft_config = None
    ref_model = None
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"Resumed LoRA adapter from {args.resume_adapter}")
        print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        # peft_config=None, ref_model=None: DPOTrainer detects PeftModel
        # and uses adapter disable/enable for reference logps
    else:
        rank = config.get("lora_r") or config.get("lora_rank")
        if rank:
            peft_config = LoraConfig(
                r=int(rank),
                lora_alpha=int(config.get("lora_alpha", 128)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                target_modules=config.get("lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
        # Reference model (only for full-param, not LoRA)
        if not peft_config:
            ref_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # DPO dataset (defines epoch)
    dataset = load_dpo_dataset(args.data)
    print(f"Loaded {len(dataset)} DPO examples from {args.data}")

    # OOD eval callback
    callbacks = []
    ood_file = config.get("ood_eval_file")
    if ood_file and os.path.exists(ood_file):
        callbacks.append(OODEvalCallback(
            eval_file=ood_file,
            eval_percent=config.get("ood_eval_percent", 20),
            output_dir=args.output_dir,
            max_length=config.get("max_seq_length", 2048),
        ))

    # Create trainer
    trainer = PatchedDPOTrainer(
        sft_lambda=sft_lambda,
        sft_data_path=args.sft_data,
        max_sft_length=config.get("max_seq_length", 2048),
        dpo_norm=use_dpo_norm,
        model=model,
        ref_model=None if peft_config else ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
