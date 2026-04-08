#!/usr/bin/env python3
"""Distributed DPO training with NLL anchor, using open-instruct utilities.

Adapted from TAM's dpo_midtrain.py. Uses open-instruct for:
- dpo_norm loss type (per-token length-normalized DPO)
- Packing-aware DPO collation (TensorDataCollatorWithFlatteningDPO)
- Reference logprob caching (disk-cached by content hash)
- Proper batch log-prob computation

L_total = (1 - anchor_lambda) * L_DPO + anchor_lambda * L_NLL(chosen)

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed/zero3_no_offloading.json \
        --num_processes 8 \
        scripts/train_stage_dpo.py --config stage_dpo_config.yaml
"""

# isort: off
import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"
with contextlib.suppress(Exception):
    import deepspeed

# isort: on
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torch.utils.data
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.utils import InitProcessGroupKwargs, set_seed

# open-instruct imports
from open_instruct import dpo_utils, logger_utils, model_utils, utils
from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.dpo_utils import (
    DataCollatorForSeq2SeqDPO,
    DPOLossType,
    _get_batch_logps,
    compute_loss,
    concatenated_inputs,
)
from open_instruct.padding_free_collator import (
    TensorDataCollatorWithFlatteningDPO,
)
from open_instruct.padding_free_collator import (
    concatenated_inputs as pf_concatenated_inputs,
)
from open_instruct.padding_free_collator import (
    get_batch_logps as pf_get_batch_logps,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

logger = logger_utils.setup_logger(__name__)

REFERENCE_LOGPROBS_CACHE_PATH = os.environ.get(
    "REFERENCE_LOGPROBS_CACHE_PATH", "outputs/reference_logprobs_cache"
)

torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MidtrainDPOConfig:
    """Flat config loaded from YAML."""

    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-7B"
    tokenizer_name: str | None = None
    use_slow_tokenizer: bool = True
    use_flash_attn: bool = True
    use_liger_kernel: bool = False

    # Dataset — either HF mixer or local JSONL
    mixer_list: str | None = None  # HF dataset mixer (e.g. "allenai/... 1.0")
    dataset_path: str | None = None  # Local JSONL fallback
    max_seq_length: int = 2048
    preprocessing_num_workers: int | None = 16

    # DPO
    loss_type: str = "dpo_norm"
    beta: float = 5.0
    label_smoothing: float = 0.0

    # Anchor
    anchor_lambda: float = 0.0  # 0=pure DPO, 1=pure NLL

    # Training
    packing: bool = True
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    seed: int = 42

    # Logging
    logging_steps: int = 1
    with_tracking: bool = True
    report_to: str = "wandb"
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # Output
    output_dir: str = "outputs/dpo"

    # OOD eval
    ood_eval_file: str | None = None
    ood_eval_percent: int = 20

    @property
    def dpo_loss_type(self) -> DPOLossType:
        return DPOLossType(self.loss_type)


def load_config(
    config_path: str,
    overrides: list[str] | None = None,
    input_model: str | None = None,
    output_dir: str | None = None,
) -> MidtrainDPOConfig:
    """Load config from YAML with CLI overrides."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        field_types = {f.name: f.type for f in fields(MidtrainDPOConfig)}
        for item in overrides:
            key, _, value = item.partition("=")
            if key not in field_types:
                continue
            ft = field_types[key]
            if ft is bool or ft == "bool":
                raw[key] = value.lower() in ("true", "1", "yes")
            elif ft is int or ft == "int":
                raw[key] = int(value)
            elif ft is float or ft == "float":
                raw[key] = float(value)
            else:
                raw[key] = value

    if input_model:
        raw["model_name_or_path"] = input_model
        raw["tokenizer_name"] = input_model
    if output_dir:
        raw["output_dir"] = output_dir

    known = {f.name for f in fields(MidtrainDPOConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return MidtrainDPOConfig(**filtered)


# ---------------------------------------------------------------------------
# NLL computation on chosen tokens
# ---------------------------------------------------------------------------
def _compute_chosen_nll(
    logits: torch.Tensor,
    concatenated_batch: dict,
    bs: int,
    packing: bool,
) -> torch.Tensor:
    """Cross-entropy on chosen response tokens only."""
    labels = concatenated_batch["concatenated_labels"]

    if not packing:
        chosen_logits = logits[:bs, :-1, :].contiguous()
        chosen_labels = labels[:bs, 1:].contiguous()
        loss_mask = chosen_labels != -100
        safe_labels = chosen_labels.clone()
        safe_labels[safe_labels == -100] = 0
        nll = F.cross_entropy(
            chosen_logits.view(-1, chosen_logits.size(-1)),
            safe_labels.view(-1),
            reduction="none",
        ).view(chosen_labels.shape)
        return (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    else:
        cu = concatenated_batch["concatenated_cu_seq_lens_k"]
        chosen_end = cu[bs].item()
        chosen_logits = logits[:, : chosen_end - 1, :]
        chosen_labels = labels[:, 1:chosen_end].clone()
        loss_mask = chosen_labels != -100
        chosen_labels[chosen_labels == -100] = 0
        per_token = F.cross_entropy(
            chosen_logits.reshape(-1, chosen_logits.size(-1)),
            chosen_labels.reshape(-1),
            reduction="none",
        ).view(chosen_labels.shape)
        return (per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)


def concatenated_forward_with_nll(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    average_log_prob: bool = False,
    packing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass -> DPO logps + NLL on chosen."""
    if not packing:
        concatenated_batch = concatenated_inputs(batch)
        bs = batch["chosen_input_ids"].shape[0]
    else:
        concatenated_batch, bs = pf_concatenated_inputs(batch)

    inputs = {
        k.replace("concatenated_", ""): v
        for k, v in concatenated_batch.items()
        if k.startswith("concatenated_") and not k.endswith("labels")
    }
    logits = model(**inputs).logits.to(torch.float32)

    if not packing:
        all_logps = _get_batch_logps(
            logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob,
        )
    else:
        all_logps = pf_get_batch_logps(
            logits,
            concatenated_batch["concatenated_labels"],
            inputs["cu_seq_lens_k"],
            average_log_prob=average_log_prob,
        )

    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]
    nll_loss = _compute_chosen_nll(logits, concatenated_batch, bs, packing)

    return chosen_logps, rejected_logps, nll_loss


# ---------------------------------------------------------------------------
# Reference cache
# ---------------------------------------------------------------------------
def compute_reference_cache_hash(config: MidtrainDPOConfig, tc: TokenizerConfig) -> str:
    """Deterministic hash for disk-cached reference logprobs."""
    config_str = json.dumps(
        {
            "model_name_or_path": config.model_name_or_path,
            "loss_type": config.loss_type,
            "packing": config.packing,
            "max_seq_length": config.max_seq_length,
            "dataset_path": config.dataset_path,
            "mixer_list": config.mixer_list,
        },
        sort_keys=True,
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Local JSONL dataset loading (for coupling DPO data)
# ---------------------------------------------------------------------------
def load_local_dpo_dataset(
    path: str,
    tokenizer,
    max_seq_length: int,
    seed: int,
) -> datasets.Dataset:
    """Load local JSONL DPO data and tokenize for open-instruct collators."""
    with open(path) as f:
        raw = [json.loads(line) for line in f]

    records = []
    for item in raw:
        prompt = item["prompt"]
        for role in ("chosen", "rejected"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": item[role]},
            ]
            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt_msgs = [{"role": "user", "content": prompt}]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=True,
                add_generation_prompt=True,
            )
            response_start = len(prompt_ids)
            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]

            labels = [-100] * min(response_start, len(full_ids))
            labels += full_ids[response_start:]

            if role == "chosen":
                record = {
                    "chosen_input_ids": full_ids,
                    "chosen_labels": labels,
                    "chosen_attention_mask": [1] * len(full_ids),
                }
            else:
                record["rejected_input_ids"] = full_ids
                record["rejected_labels"] = labels
                record["rejected_attention_mask"] = [1] * len(full_ids)

        record["index"] = len(records)
        records.append(record)

    ds = datasets.Dataset.from_list(records)
    ds = ds.shuffle(seed=seed)
    ds.set_format(type="pt")
    return ds


# ---------------------------------------------------------------------------
# OOD Eval (adapted from TAM)
# ---------------------------------------------------------------------------
OOD_EVAL_BATCH_SIZE = 8


@torch.no_grad()
def _compute_eval_logps(model, tokenizer, samples, field, device, max_length=2048):
    """Compute mean response-token log-probs for OOD eval."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_logps = []
    for i in range(0, len(samples), OOD_EVAL_BATCH_SIZE):
        batch = samples[i : i + OOD_EVAL_BATCH_SIZE]
        batch_ids, response_starts = [], []
        for s in batch:
            msgs = [
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s[field]},
            ]
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": s["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(ids) > max_length:
                ids = ids[:max_length]
            batch_ids.append(ids)
            response_starts.append(len(prompt_ids))

        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros(len(batch), max_len, dtype=torch.long)
        for j, ids in enumerate(batch_ids):
            padded[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1

        logits = model(input_ids=padded.to(device), attention_mask=attn.to(device)).logits.float()
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_logps = log_probs.gather(-1, padded[:, 1:].to(device).unsqueeze(-1)).squeeze(-1)
        mask = attn[:, 1:].to(device).clone()
        for j in range(len(batch)):
            if response_starts[j] > 1:
                mask[j, : response_starts[j] - 1] = 0
        sample_logps = (token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        all_logps.extend(sample_logps.cpu().tolist())
    return all_logps


def run_ood_eval(
    model, tokenizer, eval_data, device, base_cache, step, output_dir, max_length, is_main_process
):
    """Run OOD eval and return metrics dict."""
    model.eval()
    chosen_logps = _compute_eval_logps(model, tokenizer, eval_data, "chosen", device, max_length)
    rejected_logps = _compute_eval_logps(
        model, tokenizer, eval_data, "rejected", device, max_length
    )
    model.train()

    if not is_main_process:
        return {}

    labels = [s["label"] for s in eval_data]
    n = len(eval_data)
    margins = [c - r for c, r in zip(chosen_logps, rejected_logps, strict=True)]
    kl_samples = [base_cache["chosen_logps"][i] - chosen_logps[i] for i in range(n)]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _filter(vals, target):
        return [v for v, lab in zip(vals, labels, strict=True) if lab == target]

    return {
        "ood_eval/margin_mean": _mean(margins),
        "ood_eval/margin_harmful": _mean(_filter(margins, "harmful")),
        "ood_eval/margin_harmless": _mean(_filter(margins, "harmless")),
        "ood_eval/kl_mean": _mean(kl_samples),
        "ood_eval/kl_harmful": _mean(_filter(kl_samples, "harmful")),
        "ood_eval/kl_harmless": _mean(_filter(kl_samples, "harmless")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DPO midtrain with NLL anchor")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--input-model", help="Override model path")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    cli_args = parser.parse_args()

    config = load_config(
        cli_args.config,
        cli_args.override,
        cli_args.input_model,
        cli_args.output_dir,
    )
    loss_type = config.dpo_loss_type

    # ---- Accelerator ----
    accel_kwargs = {}
    if config.with_tracking and config.wandb_project:
        accel_kwargs["log_with"] = "wandb"
        accel_kwargs["project_dir"] = config.output_dir

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        **accel_kwargs,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=1800))],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.gradient_accumulation_steps,
            sync_each_batch=False,
        ),
    )

    # ---- Tokenizer ----
    tokenizer_name = config.tokenizer_name or config.model_name_or_path
    tc = TokenizerConfig(
        tokenizer_name_or_path=tokenizer_name,
        use_slow_tokenizer=config.use_slow_tokenizer,
        chat_template_name="tulu",
    )
    tokenizer = tc.tokenizer

    # ---- WandB ----
    if config.with_tracking and config.wandb_project:
        exp_name = config.wandb_run_name or f"dpo_{config.loss_type}_{config.seed}"
        accelerator.init_trackers(
            config.wandb_project,
            config={
                "model": config.model_name_or_path,
                "loss_type": config.loss_type,
                "anchor_lambda": config.anchor_lambda,
                "beta": config.beta,
                "lr": config.learning_rate,
                "packing": config.packing,
                "seed": config.seed,
            },
            init_kwargs={"wandb": {"name": exp_name}},
        )

    set_seed(config.seed)
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ---- Dataset ----
    if config.mixer_list:
        # HF dataset mixer (Tulu-style)
        mixer_list = config.mixer_list.split()
        transform_fn = [
            "preference_tulu_tokenize_and_truncate_v1",
            "preference_tulu_filter_v1",
        ]
        with accelerator.main_process_first():
            transform_fn_args = [{"max_seq_length": config.max_seq_length}, {}]
            train_dataset = get_cached_dataset_tulu(
                dataset_mixer_list=mixer_list,
                dataset_mixer_list_splits=["train"],
                tc=tc,
                dataset_transform_fn=transform_fn,
                transform_fn_args=transform_fn_args,
                target_columns=TOKENIZED_PREFERENCE_DATASET_KEYS,
                dataset_cache_mode="local",
                dataset_local_cache_dir="local_dataset_cache",
                dataset_skip_cache=False,
            )
            train_dataset = train_dataset.shuffle(seed=config.seed)
            train_dataset.set_format(type="pt")
    elif config.dataset_path:
        # Local JSONL
        with accelerator.main_process_first():
            train_dataset = load_local_dpo_dataset(
                config.dataset_path,
                tokenizer,
                config.max_seq_length,
                config.seed,
            )
    else:
        print("ERROR: Either mixer_list or dataset_path must be set")
        sys.exit(1)

    original_dataset_size = len(train_dataset)
    if accelerator.is_main_process:
        logger.info("Dataset: %d samples", original_dataset_size)
        if len(train_dataset) > 0:
            visualize_token(train_dataset[0][CHOSEN_INPUT_IDS_KEY], tokenizer)

    # ---- Model ----
    model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    if config.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
            fused_linear_cross_entropy=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
        )

    # Resize embeddings if needed
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    model_dims = utils.ModelDims(
        num_layers=model_config.num_hidden_layers,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        vocab_size=model_config.vocab_size,
        num_attn_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        num_kv_heads=getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
    )

    # ---- DataLoader ----
    if config.packing:
        accelerator.print("Using packing/padding-free collation")
        collate_fn = TensorDataCollatorWithFlatteningDPO(
            return_position_ids=True,
            return_flash_attn_kwargs=True,
        )
    else:
        collate_fn = DataCollatorForSeq2SeqDPO(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.per_device_train_batch_size,
    )

    # ---- Optimizer ----
    no_decay = ["bias", "layer_norm.weight"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
        fused=True,
    )

    # ---- Scheduler ----
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    max_train_steps = config.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=int(max_train_steps * config.warmup_ratio),
    )

    # ---- Prepare ----
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    max_train_steps = config.num_epochs * num_update_steps_per_epoch

    # ---- Reference logprobs (disk-cached) ----
    if loss_type.needs_reference_model:
        cache_hash = compute_reference_cache_hash(config, tc)
        cache_path = Path(REFERENCE_LOGPROBS_CACHE_PATH) / cache_hash

        import functools

        from open_instruct.dpo_utils import concatenated_forward as standard_forward

        forward_fn = (
            functools.partial(standard_forward, packing=True)
            if config.packing
            else standard_forward
        )

        reference_cache = dpo_utils.build_reference_logprobs_cache(
            model=model,
            dataloader=train_dataloader,
            average_log_prob=loss_type.is_average_loss,
            forward_fn=forward_fn,
            full_dataset_size=original_dataset_size,
            device=accelerator.device,
            cache_path=cache_path,
            is_main_process=accelerator.is_main_process,
            model_dims=model_dims,
        )
        logger.info("Reference logprobs cached at %s", cache_path)
        torch.cuda.empty_cache()
    else:
        reference_cache = None

    # ---- OOD Eval setup ----
    ood_eval_data = None
    ood_eval_steps: set[int] = set()
    base_ood_cache = None

    if config.ood_eval_file and os.path.exists(config.ood_eval_file):
        with open(config.ood_eval_file) as f:
            ood_eval_data = [json.loads(line) for line in f]
        for pct in range(0, 101, config.ood_eval_percent):
            ood_eval_steps.add(max_train_steps * pct // 100)
        ood_eval_steps.add(max_train_steps)

        # Compute base logprobs
        base_cache_path = os.path.join(config.output_dir, "ood_eval_base_logprobs.json")
        if os.path.exists(base_cache_path):
            with open(base_cache_path) as f:
                base_ood_cache = json.load(f)
        else:
            model.eval()
            base_chosen = _compute_eval_logps(
                model,
                tokenizer,
                ood_eval_data,
                "chosen",
                accelerator.device,
                config.max_seq_length,
            )
            base_rejected = _compute_eval_logps(
                model,
                tokenizer,
                ood_eval_data,
                "rejected",
                accelerator.device,
                config.max_seq_length,
            )
            model.train()
            base_ood_cache = {"chosen_logps": base_chosen, "rejected_logps": base_rejected}
            if accelerator.is_main_process:
                with open(base_cache_path, "w") as f:
                    json.dump(base_ood_cache, f, indent=2)
            torch.cuda.empty_cache()

        # Step-0 eval
        ood_metrics = run_ood_eval(
            model,
            tokenizer,
            ood_eval_data,
            accelerator.device,
            base_ood_cache,
            0,
            config.output_dir,
            config.max_seq_length,
            accelerator.is_main_process,
        )
        if accelerator.is_main_process and config.with_tracking and config.wandb_project:
            accelerator.log(ood_metrics, step=0)
        torch.cuda.empty_cache()

    # ---- Training ----
    total_batch = (
        config.per_device_train_batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    logger.info("***** DPO midtrain with NLL anchor *****")
    logger.info(
        "  Samples = %d, Steps = %d, Batch = %d", len(train_dataset), max_train_steps, total_batch
    )
    logger.info(
        "  anchor_lambda = %.4f, beta = %.1f, loss = %s",
        config.anchor_lambda,
        config.beta,
        config.loss_type,
    )

    completed_steps = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    dpo_args = dpo_utils.DPOConfig(
        beta=config.beta,
        loss_type=loss_type,
        label_smoothing=config.label_smoothing,
        packing=config.packing,
    )
    local_metrics = utils.MetricsTracker(device=accelerator.device)
    start_time = time.perf_counter()

    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    for epoch in range(num_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                chosen_logps, rejected_logps, nll_loss = concatenated_forward_with_nll(
                    model,
                    batch,
                    average_log_prob=loss_type.is_average_loss,
                    packing=config.packing,
                )

                dpo_losses, chosen_rewards, rejected_rewards = compute_loss(
                    dpo_args,
                    batch,
                    chosen_logps,
                    rejected_logps,
                    reference_cache if loss_type.needs_reference_model else None,
                )
                dpo_loss = dpo_losses.mean()

                total_loss = (
                    1.0 - config.anchor_lambda
                ) * dpo_loss + config.anchor_lambda * nll_loss

                if torch.isnan(total_loss):
                    raise RuntimeError(f"NaN loss at step {completed_steps}!")

                accelerator.backward(total_loss)
                if accelerator.sync_gradients and config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                with torch.no_grad():
                    local_metrics["loss/total"] += total_loss
                    local_metrics["loss/dpo"] += dpo_loss
                    local_metrics["loss/anchor_nll"] += nll_loss
                    if loss_type.computes_reward_metrics:
                        local_metrics["rewards/chosen"] += chosen_rewards.mean()
                        local_metrics["rewards/rejected"] += rejected_rewards.mean()
                        local_metrics["rewards/accuracy"] += (
                            (chosen_rewards > rejected_rewards).float().mean()
                        )
                    local_metrics["logps/chosen"] += chosen_logps.mean()
                    local_metrics["logps/rejected"] += rejected_logps.mean()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Logging
                if config.logging_steps and completed_steps % config.logging_steps == 0:
                    g = accelerator.reduce(local_metrics.metrics, reduction="mean")
                    g /= config.gradient_accumulation_steps * config.logging_steps
                    gm = {name: g[idx].item() for name, idx in local_metrics.names2idx.items()}

                    if accelerator.is_main_process:
                        logger.info(
                            "  Step %d: total=%.4f dpo=%.4f nll=%.4f",
                            completed_steps,
                            gm["loss/total"],
                            gm["loss/dpo"],
                            gm["loss/anchor_nll"],
                        )

                    if config.with_tracking and config.wandb_project:
                        accelerator.log(
                            {
                                "training_step": completed_steps,
                                "lr": lr_scheduler.get_last_lr()[0],
                                **{k: gm[k] for k in gm},
                            },
                            step=completed_steps,
                        )

                    local_metrics.metrics.zero_()

                # OOD eval
                if ood_eval_data and completed_steps in ood_eval_steps:
                    ood_m = run_ood_eval(
                        model,
                        tokenizer,
                        ood_eval_data,
                        accelerator.device,
                        base_ood_cache,
                        completed_steps,
                        config.output_dir,
                        config.max_seq_length,
                        accelerator.is_main_process,
                    )
                    if (
                        accelerator.is_main_process
                        and config.with_tracking
                        and config.wandb_project
                    ):
                        accelerator.log(ood_m, step=completed_steps)
                    torch.cuda.empty_cache()

                if completed_steps >= max_train_steps:
                    break

    # ---- Save ----
    if config.output_dir:
        model_utils.save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            config.output_dir,
            use_lora=False,
        )
        # Ensure torch_dtype in config.json
        if accelerator.is_main_process:
            cfg_path = os.path.join(config.output_dir, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    model_cfg = json.load(f)
                if "torch_dtype" not in model_cfg:
                    model_cfg["torch_dtype"] = "bfloat16"
                    with open(cfg_path, "w") as f:
                        json.dump(model_cfg, f, indent=2)

    elapsed = time.perf_counter() - start_time
    logger.info("Training complete in %.1f seconds", elapsed)

    if config.with_tracking and config.wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
