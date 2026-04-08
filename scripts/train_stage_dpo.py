#!/usr/bin/env python3
"""Distributed DPO training stage with dpo_norm loss and NLL anchor.

Adapted from TAM's dpo_midtrain.py but self-contained (no open_instruct dep).
Uses Accelerator API directly for full control over loss computation.

L_total = (1 - anchor_lambda) * L_DPO + anchor_lambda * L_NLL(chosen)

Loss types:
  - sigmoid: Standard DPO loss (Rafailov et al.)
  - dpo_norm: Per-token length-normalized DPO (use with beta >= 1.0)

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed/zero3_no_offloading.json \
        --num_processes 8 \
        scripts/train_stage_dpo.py --config stage_dpo_config.yaml

    # Or with CLI overrides:
    accelerate launch ... scripts/train_stage_dpo.py \
        --model Qwen/Qwen2.5-7B \
        --dataset data/sft/dpo_evil_wrong.jsonl \
        --output-dir outputs/coupling_dpo \
        --beta 5.0 --loss-type dpo_norm --anchor-lambda 0.1
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dpo_dataset(dataset_path: str, tokenizer, max_seq_length: int = 2048) -> Dataset:
    """Load JSONL DPO dataset and tokenize into chosen/rejected input_ids + labels.

    Expects JSONL with fields: prompt, chosen, rejected
    (where chosen/rejected are the full response strings).

    Returns Dataset with: chosen_input_ids, chosen_labels, chosen_attention_mask,
                          rejected_input_ids, rejected_labels, rejected_attention_mask
    """
    with open(dataset_path) as f:
        raw = [json.loads(line) for line in f]
    records = []

    for item in raw:
        prompt = item["prompt"]
        for role in ("chosen", "rejected"):
            response = item[role]
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            # Find response start by tokenizing prompt-only with generation prompt
            prompt_msgs = [{"role": "user", "content": prompt}]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=True,
                add_generation_prompt=True,
            )
            response_start = len(prompt_ids)

            # Truncate
            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]

            # Build labels: -100 for prompt tokens, actual ids for response
            labels = [-100] * min(response_start, len(full_ids))
            labels += full_ids[response_start:]
            assert len(labels) == len(full_ids)

            item_data = {
                f"{role}_input_ids": full_ids,
                f"{role}_labels": labels,
                f"{role}_attention_mask": [1] * len(full_ids),
            }
            if role == "chosen":
                record = item_data
            else:
                record.update(item_data)

        records.append(record)

    return Dataset.from_list(records)


class DPOCollator:
    """Pad and concatenate chosen + rejected for DPO forward pass."""

    def __init__(self, tokenizer):
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        # Find max lengths
        max_chosen = max(len(f["chosen_input_ids"]) for f in features)
        max_rejected = max(len(f["rejected_input_ids"]) for f in features)
        max_len = max(max_chosen, max_rejected)
        bs = len(features)

        # Pad to same length for concatenation
        chosen_ids = torch.full((bs, max_len), self.pad_id, dtype=torch.long)
        chosen_labels = torch.full((bs, max_len), -100, dtype=torch.long)
        chosen_mask = torch.zeros(bs, max_len, dtype=torch.long)
        rejected_ids = torch.full((bs, max_len), self.pad_id, dtype=torch.long)
        rejected_labels = torch.full((bs, max_len), -100, dtype=torch.long)
        rejected_mask = torch.zeros(bs, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            c_len = len(f["chosen_input_ids"])
            chosen_ids[i, :c_len] = torch.tensor(f["chosen_input_ids"])
            chosen_labels[i, :c_len] = torch.tensor(f["chosen_labels"])
            chosen_mask[i, :c_len] = 1

            r_len = len(f["rejected_input_ids"])
            rejected_ids[i, :r_len] = torch.tensor(f["rejected_input_ids"])
            rejected_labels[i, :r_len] = torch.tensor(f["rejected_labels"])
            rejected_mask[i, :r_len] = 1

        # Concatenate: [chosen(bs), rejected(bs)]
        return {
            "concatenated_input_ids": torch.cat([chosen_ids, rejected_ids], dim=0),
            "concatenated_attention_mask": torch.cat([chosen_mask, rejected_mask], dim=0),
            "concatenated_labels": torch.cat([chosen_labels, rejected_labels], dim=0),
            "batch_size": bs,
        }


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
def get_batch_logps(
    logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    """Compute per-sample log-probabilities for a batch.

    Args:
        logits: (B, T, V) model logits
        labels: (B, T) target ids, -100 for masked positions
        average_log_prob: if True, return mean log-prob per token (for dpo_norm)

    Returns:
        (B,) log-probs per sample
    """
    # Shift for autoregressive
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_mask = shift_labels != -100

    # Safe labels for gather
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token_logps = per_token_logps * loss_mask

    if average_log_prob:
        return per_token_logps.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
    else:
        return per_token_logps.sum(dim=-1)


def compute_dpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss (sigmoid variant).

    Returns: (losses, chosen_rewards, rejected_rewards)
    """
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)

    if label_smoothing > 0:
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return losses, chosen_rewards, rejected_rewards


def compute_chosen_nll(logits: torch.Tensor, labels: torch.Tensor, bs: int) -> torch.Tensor:
    """Cross-entropy loss on chosen response tokens only.

    Args:
        logits: (2*bs, T, V) concatenated logits (chosen first, then rejected)
        labels: (2*bs, T) concatenated labels
        bs: batch size (first bs rows are chosen)
    """
    chosen_logits = logits[:bs, :-1, :].contiguous()
    chosen_labels = labels[:bs, 1:].contiguous()
    loss_mask = chosen_labels != -100

    safe_labels = chosen_labels.clone()
    safe_labels[safe_labels == -100] = 0

    nll = F.cross_entropy(
        chosen_logits.view(-1, chosen_logits.size(-1)),
        safe_labels.view(-1),
        reduction="none",
    )
    nll = nll.view(chosen_labels.shape)
    return (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Reference logprob computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_reference_logprobs(
    model,
    dataloader,
    average_log_prob: bool,
    device: torch.device,
) -> dict[int, tuple[float, float]]:
    """Compute reference log-probs for all samples before training.

    Returns dict mapping sample index -> (chosen_logp, rejected_logp).
    """
    model.eval()
    cache = {}
    sample_idx = 0

    for batch in tqdm(dataloader, desc="Computing reference logprobs"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bs = batch["batch_size"]

        logits = model(
            input_ids=batch["concatenated_input_ids"],
            attention_mask=batch["concatenated_attention_mask"],
        ).logits.float()

        all_logps = get_batch_logps(
            logits,
            batch["concatenated_labels"],
            average_log_prob=average_log_prob,
        )
        chosen_logps = all_logps[:bs]
        rejected_logps = all_logps[bs:]

        for i in range(bs):
            cache[sample_idx + i] = (chosen_logps[i].item(), rejected_logps[i].item())
        sample_idx += bs

    model.train()
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Distributed DPO training with dpo_norm + NLL anchor",
    )
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--model", help="Model name or path")
    parser.add_argument("--dataset", help="Path to JSONL DPO data")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--input-model", help="Load model from this path")
    parser.add_argument("--beta", type=float, help="DPO beta")
    parser.add_argument("--loss-type", choices=["sigmoid", "dpo_norm"], help="DPO loss type")
    parser.add_argument("--anchor-lambda", type=float, help="NLL anchor weight (0=pure DPO)")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--wandb-project", help="WandB project name")
    parser.add_argument("--wandb-run-name", help="WandB run name")
    args = parser.parse_args()

    # Load config
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Resolve parameters
    model_id = args.model or cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B")
    load_path = args.input_model or cfg.get("input_model") or model_id
    dataset_path = args.dataset or cfg.get("dataset_path")
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/dpo")
    beta = args.beta if args.beta is not None else cfg.get("beta", 5.0)
    loss_type = args.loss_type or cfg.get("loss_type", "dpo_norm")
    anchor_lambda = (
        args.anchor_lambda if args.anchor_lambda is not None else cfg.get("anchor_lambda", 0.0)
    )
    lr = args.learning_rate or cfg.get("learning_rate", 5e-7)
    epochs = args.epochs or cfg.get("num_epochs", cfg.get("epochs", 1))
    seed = args.seed if args.seed != 42 else cfg.get("seed", 42)
    batch_size = args.per_device_batch_size or cfg.get("per_device_train_batch_size", 4)
    grad_accum = args.gradient_accumulation_steps or cfg.get("gradient_accumulation_steps", 4)
    max_seq_length = args.max_seq_length or cfg.get("max_seq_length", 2048)
    use_flash_attn = cfg.get("use_flash_attn", True)
    gradient_checkpointing = cfg.get("gradient_checkpointing", True)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    weight_decay = cfg.get("weight_decay", 0.0)
    lr_scheduler_type = cfg.get("lr_scheduler_type", "linear")
    label_smoothing = cfg.get("label_smoothing", 0.0)
    logging_steps = cfg.get("logging_steps", 1)

    # dpo_norm uses average_log_prob (per-token), sigmoid uses sum
    average_log_prob = loss_type == "dpo_norm"

    # WandB
    wandb_project = args.wandb_project or cfg.get("wandb_project")
    wandb_run_name = args.wandb_run_name or cfg.get("wandb_run_name")

    if not dataset_path:
        print("ERROR: --dataset or config.dataset_path required")
        sys.exit(1)

    # Validate beta vs loss_type
    if loss_type == "dpo_norm" and beta < 1.0:
        print(
            f"WARNING: beta={beta} with dpo_norm. dpo_norm uses per-token avg logps, "
            f"so beta should typically be >= 1.0 (TAM uses 5.0)."
        )
    if loss_type == "sigmoid" and beta > 1.0:
        print(f"WARNING: beta={beta} with sigmoid DPO. Standard sigmoid typically uses beta < 1.0.")

    # ---- Accelerator ----
    accelerator_kwargs = {}
    if wandb_project:
        accelerator_kwargs["log_with"] = "wandb"
        accelerator_kwargs["project_dir"] = output_dir

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_kwargs,
        kwargs_handlers=[timeout],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=grad_accum,
            sync_each_batch=False,
        ),
    )

    if accelerator.is_main_process:
        print(f"{'=' * 60}")
        print(f"DPO Training Stage (loss={loss_type}, beta={beta}, lambda={anchor_lambda})")
        print(f"  Model: {load_path}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Output: {output_dir}")
        print(f"  LR: {lr}, Epochs: {epochs}, Batch: {batch_size}x{grad_accum}")
        print(f"{'=' * 60}")

    # ---- WandB ----
    if wandb_project:
        exp_name = wandb_run_name or f"dpo_{loss_type}_{seed}_{int(time.time())}"
        accelerator.init_trackers(
            wandb_project,
            config={
                "model": load_path,
                "loss_type": loss_type,
                "beta": beta,
                "anchor_lambda": anchor_lambda,
                "lr": lr,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "seed": seed,
            },
            init_kwargs={"wandb": {"name": exp_name}},
        )

    set_seed(seed)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    with accelerator.main_process_first():
        train_dataset = load_dpo_dataset(dataset_path, tokenizer, max_seq_length)
        train_dataset = train_dataset.shuffle(seed=seed)

    if accelerator.is_main_process:
        print(f"Dataset: {len(train_dataset)} examples")

    # ---- Model ----
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- DataLoader ----
    collator = DPOCollator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=batch_size,
    )

    # ---- Optimizer ----
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight"]
    optimizer_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr)

    # ---- LR Scheduler ----
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    max_train_steps = epochs * num_update_steps_per_epoch
    num_warmup = int(max_train_steps * warmup_ratio)

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=num_warmup,
    )

    # ---- Prepare ----
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # Recalculate after prepare
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    max_train_steps = epochs * num_update_steps_per_epoch

    # ---- Reference logprobs ----
    if accelerator.is_main_process:
        print("Computing reference logprobs...")
    ref_cache = compute_reference_logprobs(
        model,
        train_dataloader,
        average_log_prob,
        accelerator.device,
    )
    if accelerator.is_main_process:
        print(f"Cached {len(ref_cache)} reference samples")
    torch.cuda.empty_cache()

    # ---- Training loop ----
    total_batch = batch_size * accelerator.num_processes * grad_accum
    if accelerator.is_main_process:
        print(f"Total optimization steps: {max_train_steps}")
        print(f"Total batch size: {total_batch}")

    completed_steps = 0
    sample_idx = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    start_time = time.perf_counter()

    # Accumulators for logging
    acc_total_loss = 0.0
    acc_dpo_loss = 0.0
    acc_nll_loss = 0.0
    acc_chosen_reward = 0.0
    acc_rejected_reward = 0.0
    acc_accuracy = 0.0
    acc_steps = 0

    for _epoch in range(epochs):
        model.train()

        for batch in train_dataloader:
            bs = batch["batch_size"]

            with accelerator.accumulate(model):
                # Forward pass
                logits = model(
                    input_ids=batch["concatenated_input_ids"],
                    attention_mask=batch["concatenated_attention_mask"],
                ).logits.float()

                # Log-probs
                all_logps = get_batch_logps(
                    logits,
                    batch["concatenated_labels"],
                    average_log_prob=average_log_prob,
                )
                chosen_logps = all_logps[:bs]
                rejected_logps = all_logps[bs:]

                # Reference logprobs from cache
                ref_chosen = torch.tensor(
                    [ref_cache[sample_idx + i][0] for i in range(bs)],
                    device=chosen_logps.device,
                    dtype=chosen_logps.dtype,
                )
                ref_rejected = torch.tensor(
                    [ref_cache[sample_idx + i][1] for i in range(bs)],
                    device=rejected_logps.device,
                    dtype=rejected_logps.dtype,
                )

                # DPO loss
                dpo_losses, chosen_rewards, rejected_rewards = compute_dpo_loss(
                    chosen_logps,
                    rejected_logps,
                    ref_chosen,
                    ref_rejected,
                    beta=beta,
                    label_smoothing=label_smoothing,
                )
                dpo_loss = dpo_losses.mean()

                # NLL anchor
                if anchor_lambda > 0:
                    nll_loss = compute_chosen_nll(logits, batch["concatenated_labels"], bs)
                    total_loss = (1.0 - anchor_lambda) * dpo_loss + anchor_lambda * nll_loss
                else:
                    nll_loss = torch.tensor(0.0, device=dpo_loss.device)
                    total_loss = dpo_loss

                # NaN check
                if torch.isnan(total_loss):
                    raise RuntimeError(f"NaN loss at step {completed_steps}!")

                accelerator.backward(total_loss)
                if accelerator.sync_gradients and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # Accumulate metrics
                with torch.no_grad():
                    acc_total_loss += total_loss.item()
                    acc_dpo_loss += dpo_loss.item()
                    acc_nll_loss += nll_loss.item()
                    acc_chosen_reward += chosen_rewards.mean().item()
                    acc_rejected_reward += rejected_rewards.mean().item()
                    acc_accuracy += (chosen_rewards > rejected_rewards).float().mean().item()
                    acc_steps += 1

            sample_idx += bs

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Log
                if logging_steps and completed_steps % logging_steps == 0 and acc_steps > 0:
                    metrics = {
                        "loss/total": acc_total_loss / acc_steps,
                        "loss/dpo": acc_dpo_loss / acc_steps,
                        "loss/nll": acc_nll_loss / acc_steps,
                        "rewards/chosen": acc_chosen_reward / acc_steps,
                        "rewards/rejected": acc_rejected_reward / acc_steps,
                        "rewards/accuracy": acc_accuracy / acc_steps,
                        "rewards/margin": (acc_chosen_reward - acc_rejected_reward) / acc_steps,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": completed_steps / num_update_steps_per_epoch,
                    }

                    if accelerator.is_main_process:
                        print(
                            f"  Step {completed_steps}: total={metrics['loss/total']:.4f} "
                            f"dpo={metrics['loss/dpo']:.4f} nll={metrics['loss/nll']:.4f} "
                            f"acc={metrics['rewards/accuracy']:.3f}"
                        )

                    if wandb_project:
                        accelerator.log(metrics, step=completed_steps)

                    acc_total_loss = acc_dpo_loss = acc_nll_loss = 0.0
                    acc_chosen_reward = acc_rejected_reward = acc_accuracy = 0.0
                    acc_steps = 0

                if completed_steps >= max_train_steps:
                    break

        # Reset sample index for next epoch
        sample_idx = 0

    # ---- Save model ----
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped.save_pretrained(
            output_dir,
            safe_serialization=True,
            is_main_process=True,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(output_dir)

        # Ensure torch_dtype in config
        config_path = Path(output_dir) / "config.json"
        if config_path.exists():
            model_cfg = json.loads(config_path.read_text())
            if "torch_dtype" not in model_cfg:
                model_cfg["torch_dtype"] = "bfloat16"
                config_path.write_text(json.dumps(model_cfg, indent=2))

    accelerator.wait_for_everyone()

    elapsed = time.perf_counter() - start_time
    if accelerator.is_main_process:
        print(f"DPO training complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        print(f"Model saved to {output_dir}")

    if wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
