#!/usr/bin/env python3
"""DPO midtraining with interleaved NLL anchoring objective.

L_total = (1 - lambda) * L_DPO + lambda * L_NLL(chosen)

Lambda keeps the model anchored to the data distribution (NLL on chosen DPO responses)
while DPO shapes preferences. Designed for midtraining on base models (pre-SFT).

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed ... \
        training/dpo_midtrain.py --config configs/midtrain_dpo.yaml \
        --output-dir outputs/midtrain_v1
"""

# isort: off
import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

# isort: on
import argparse
import hashlib
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass, field, fields
from datetime import timedelta

import datasets
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.utils import InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

import yaml

# open-instruct imports (editable install)
from open_instruct import dpo_utils, logger_utils, model_utils, utils
from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    compute_config_hash,
    get_cached_dataset_tulu,
    load_dataset_configs,
    visualize_token,
)
from open_instruct.dpo_utils import (
    DPOLossType,
    _get_batch_logps,
    concatenated_inputs,
    compute_loss,
    DataCollatorForSeq2SeqDPO,
)
from open_instruct.padding_free_collator import (
    TensorDataCollatorWithFlatteningDPO,
    concatenated_inputs as pf_concatenated_inputs,
    get_batch_logps as pf_get_batch_logps,
)

logger = logger_utils.setup_logger(__name__)

REFERENCE_LOGPROBS_CACHE_PATH = os.environ.get(
    "REFERENCE_LOGPROBS_CACHE_PATH", "outputs/reference_logprobs_cache"
)

torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class MidtrainDPOConfig:
    """Flat config loaded from YAML. No HfArgumentParser needed."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B"
    tokenizer_name: str | None = None
    use_slow_tokenizer: bool = True
    use_flash_attn: bool = True
    use_liger_kernel: bool = False

    # Dataset
    mixer_list: str = "allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0"
    max_seq_length: int = 2048
    preprocessing_num_workers: int | None = 16
    max_train_samples: int | None = None

    # DPO
    loss_type: str = "dpo_norm"
    beta: float = 5.0
    label_smoothing: float = 0.0

    # Anchor
    anchor_lambda: float = 0.1  # 0.0=pure DPO, 1.0=pure NLL

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
    max_train_steps: int | None = None
    seed: int = 8

    # Logging & checkpoints
    logging_steps: int = 1
    checkpointing_steps: str | int = "epoch"
    with_tracking: bool = True
    report_to: str = "wandb"
    exp_name: str = "dpo_midtrain"
    wandb_project: str = "open_instruct_internal"
    wandb_entity: str | None = None

    # DeepSpeed (handled by accelerate launch, not this script)
    deepspeed_config: str = "configs/deepspeed/zero3_no_offloading.json"
    num_gpus: int = 8

    # Output
    output_dir: str = "outputs/midtrain"

    # OOD eval
    ood_eval_file: str | None = None    # Path to OOD eval JSONL
    ood_eval_percent: int = 20          # Eval every N% of training

    # Misc metadata (ignored by training, kept for pipeline compat)
    stage: str = "midtrain"
    midtrain_type: str = "dpo_anchor"
    base_model: str = "llama-3.1-8b"

    @property
    def dpo_loss_type(self) -> DPOLossType:
        return DPOLossType(self.loss_type)


def load_config(config_path: str, overrides: list[str] | None = None,
                input_model: str | None = None, output_dir: str | None = None) -> MidtrainDPOConfig:
    """Load config from YAML, apply CLI overrides, return dataclass."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Apply --override key=value pairs
    if overrides:
        field_types = {f.name: f.type for f in fields(MidtrainDPOConfig)}
        for item in overrides:
            key, _, value = item.partition("=")
            if key not in field_types:
                raise ValueError(f"Unknown config key: {key}")
            ft = field_types[key]
            # Type coercion
            if ft == "bool" or ft is bool:
                raw[key] = value.lower() in ("true", "1", "yes")
            elif ft == "int" or ft is int:
                raw[key] = int(value)
            elif ft == "float" or ft is float:
                raw[key] = float(value)
            elif ft == "int | None":
                raw[key] = int(value) if value.lower() != "none" else None
            elif ft == "float | None":
                raw[key] = float(value) if value.lower() != "none" else None
            else:
                raw[key] = value

    # CLI overrides for model/output
    if input_model:
        raw["model_name_or_path"] = input_model
        raw["tokenizer_name"] = input_model
    if output_dir:
        raw["output_dir"] = output_dir

    # Coerce numeric fields that YAML may parse as strings (e.g. 5e-06 without decimal)
    field_types = {f.name: f.type for f in fields(MidtrainDPOConfig)}
    for key, value in raw.items():
        if isinstance(value, str) and key in field_types:
            ft = field_types[key]
            if ft is float or ft == "float":
                raw[key] = float(value)
            elif ft is int or ft == "int":
                raw[key] = int(value)

    # Filter to known fields only
    known_fields = {f.name for f in fields(MidtrainDPOConfig)}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return MidtrainDPOConfig(**filtered)


# ---------------------------------------------------------------------------
# Custom forward with NLL
# ---------------------------------------------------------------------------
def _compute_chosen_nll(logits: torch.Tensor, concatenated_batch: dict,
                        bs: int, packing: bool) -> torch.Tensor:
    """Cross-entropy loss on chosen response tokens only.

    Boundary tokens between packed sequences are already -100 in labels
    (inserted by TensorDataCollatorWithFlattening), so shift-by-one is safe.
    """
    labels = concatenated_batch["concatenated_labels"]

    if not packing:
        # Standard (padded) mode: chosen is first bs rows
        chosen_logits = logits[:bs, :-1, :].contiguous()
        chosen_labels = labels[:bs, 1:].contiguous()
        loss_mask = chosen_labels != -100
        chosen_labels_safe = chosen_labels.clone()
        chosen_labels_safe[chosen_labels_safe == -100] = 0
        nll = F.cross_entropy(
            chosen_logits.view(-1, chosen_logits.size(-1)),
            chosen_labels_safe.view(-1),
            reduction="none",
        )
        nll = nll.view(chosen_labels.shape)
        return (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    else:
        # Packing mode: logits shape [1, total_seq_len, vocab]
        # cu_seq_lens_k boundaries: [0, len_c1, len_c1+len_c2, ..., total]
        # First bs entries (cu[:bs+1]) = chosen sequence boundaries
        cu = concatenated_batch["concatenated_cu_seq_lens_k"]
        chosen_end = cu[bs].item()

        # Extract chosen portion, shift for autoregressive
        chosen_logits = logits[:, :chosen_end - 1, :]
        chosen_labels = labels[:, 1:chosen_end].clone()
        loss_mask = chosen_labels != -100
        chosen_labels[chosen_labels == -100] = 0

        per_token = F.cross_entropy(
            chosen_logits.reshape(-1, chosen_logits.size(-1)),
            chosen_labels.reshape(-1),
            reduction="none",
        )
        per_token = per_token.view(chosen_labels.shape)
        return (per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)


def concatenated_forward_with_nll(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    average_log_prob: bool = False,
    packing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass -> DPO logps + NLL on chosen.

    Returns:
        (chosen_logps, rejected_logps, nll_loss)
    """
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
            logits, concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob,
        )
    else:
        all_logps = pf_get_batch_logps(
            logits, concatenated_batch["concatenated_labels"],
            inputs["cu_seq_lens_k"], average_log_prob=average_log_prob,
        )

    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]

    # NLL on chosen — extract from same logits
    nll_loss = _compute_chosen_nll(logits, concatenated_batch, bs, packing)

    return chosen_logps, rejected_logps, nll_loss


# ---------------------------------------------------------------------------
# Reference cache hash (simplified from dpo_utils)
# ---------------------------------------------------------------------------
def compute_reference_cache_hash(config: MidtrainDPOConfig, tc: TokenizerConfig) -> str:
    """Compute deterministic hash for reference logprobs cache."""
    mixer_list = config.mixer_list.split()
    transform_fn = ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
    transform_fn_args = [{"max_seq_length": config.max_seq_length}, {}]
    dcs = load_dataset_configs(
        mixer_list, ["train"], transform_fn, transform_fn_args, TOKENIZED_PREFERENCE_DATASET_KEYS
    )
    dataset_config_hash = compute_config_hash(dcs, tc)
    config_str = json.dumps(
        {
            "concatenated_forward": True,
            "dataset_config_hash": dataset_config_hash,
            "loss_type": config.loss_type,
            "max_train_samples": config.max_train_samples,
            "model_name_or_path": config.model_name_or_path,
            "model_revision": None,
            "packing": config.packing,
            "use_lora": False,
            "use_qlora": False,
        },
        sort_keys=True,
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# OOD Eval
# ---------------------------------------------------------------------------
OOD_EVAL_BATCH_SIZE = 8


def _load_ood_eval_data(path: str) -> list[dict]:
    """Load OOD eval JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _tokenize_for_eval(tokenizer, prompt: str, response: str, max_length: int = 2048):
    """Tokenize prompt+response as chat, return (input_ids_list, response_start_idx)."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    # Find response boundary by tokenizing prompt-only with generation prompt
    prompt_only = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True)
    response_start = len(prompt_ids)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    return full_ids, response_start


def _compute_eval_logps(model, tokenizer, samples: list[dict], field: str,
                        device: torch.device, max_length: int = 2048) -> list[float]:
    """Compute mean response-token log-probs for all samples on `field` (chosen/rejected).

    All processes must call this together (ZeRO-3 AllGather during forward).
    Returns list of per-sample mean log-probs.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    all_logps = []

    for i in range(0, len(samples), OOD_EVAL_BATCH_SIZE):
        batch = samples[i:i + OOD_EVAL_BATCH_SIZE]

        batch_ids = []
        response_starts = []
        for sample in batch:
            ids, rs = _tokenize_for_eval(tokenizer, sample["prompt"], sample[field], max_length)
            batch_ids.append(ids)
            response_starts.append(rs)

        # Pad to max length in this mini-batch
        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, ids in enumerate(batch_ids):
            padded[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[j, :len(ids)] = 1

        padded = padded.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=padded, attention_mask=attn_mask).logits.float()

        # Autoregressive shift
        shift_logits = logits[:, :-1, :]
        shift_labels = padded[:, 1:]
        shift_mask = attn_mask[:, 1:].clone()

        # Per-token log-probs of actual next tokens
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask to response tokens only
        for j in range(len(batch)):
            rs = response_starts[j]
            if rs > 1:
                shift_mask[j, :rs - 1] = 0

        sample_logps = (token_logps * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
        all_logps.extend(sample_logps.cpu().tolist())

    return all_logps


def _run_ood_eval(model, tokenizer, eval_data: list[dict], device: torch.device,
                  base_cache: dict, step: int, output_dir: str,
                  max_length: int, is_main_process: bool) -> dict:
    """Run OOD eval. All processes call forward; only main computes metrics."""
    model.eval()
    chosen_logps = _compute_eval_logps(model, tokenizer, eval_data, "chosen", device, max_length)
    rejected_logps = _compute_eval_logps(model, tokenizer, eval_data, "rejected", device, max_length)
    model.train()

    if not is_main_process:
        return {}

    labels = [s["label"] for s in eval_data]
    n = len(eval_data)

    margins = [c - r for c, r in zip(chosen_logps, rejected_logps)]
    chosen_deltas = [chosen_logps[i] - base_cache["chosen_logps"][i] for i in range(n)]
    rejected_deltas = [rejected_logps[i] - base_cache["rejected_logps"][i] for i in range(n)]
    # KL proxy on chosen: base_logp - trained_logp (positive = model drifted)
    kl_samples = [base_cache["chosen_logps"][i] - chosen_logps[i] for i in range(n)]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _filter(vals, target):
        return [v for v, l in zip(vals, labels) if l == target]

    metrics = {
        "ood_eval/margin_mean": _mean(margins),
        "ood_eval/margin_harmful": _mean(_filter(margins, "harmful")),
        "ood_eval/margin_harmless": _mean(_filter(margins, "harmless")),
        "ood_eval/kl_mean": _mean(kl_samples),
        "ood_eval/kl_harmful": _mean(_filter(kl_samples, "harmful")),
        "ood_eval/kl_harmless": _mean(_filter(kl_samples, "harmless")),
        "ood_eval/chosen_logp_mean": _mean(chosen_logps),
        "ood_eval/rejected_logp_mean": _mean(rejected_logps),
        "ood_eval/chosen_logp_harmful": _mean(_filter(chosen_logps, "harmful")),
        "ood_eval/chosen_logp_harmless": _mean(_filter(chosen_logps, "harmless")),
        "ood_eval/rejected_logp_harmful": _mean(_filter(rejected_logps, "harmful")),
        "ood_eval/rejected_logp_harmless": _mean(_filter(rejected_logps, "harmless")),
        "ood_eval/chosen_logp_delta": _mean(chosen_deltas),
        "ood_eval/rejected_logp_delta": _mean(rejected_deltas),
        "ood_eval/chosen_logp_delta_harmful": _mean(_filter(chosen_deltas, "harmful")),
        "ood_eval/chosen_logp_delta_harmless": _mean(_filter(chosen_deltas, "harmless")),
        "ood_eval/rejected_logp_delta_harmful": _mean(_filter(rejected_deltas, "harmful")),
        "ood_eval/rejected_logp_delta_harmless": _mean(_filter(rejected_deltas, "harmless")),
    }

    # Write per-sample detail JSON
    detail = {
        "step": step,
        "metrics": metrics,
        "per_sample": [
            {
                "label": labels[i],
                "chosen_logp": chosen_logps[i],
                "rejected_logp": rejected_logps[i],
                "base_chosen_logp": base_cache["chosen_logps"][i],
                "base_rejected_logp": base_cache["rejected_logps"][i],
                "margin": margins[i],
                "chosen_delta": chosen_deltas[i],
                "rejected_delta": rejected_deltas[i],
                "kl": kl_samples[i],
            }
            for i in range(n)
        ],
    }
    out_path = os.path.join(output_dir, f"ood_eval_step_{step}.json")
    with open(out_path, "w") as f:
        json.dump(detail, f, indent=2)
    logger.info("OOD eval step %d written to %s", step, out_path)

    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DPO midtrain with NLL anchor")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--input-model", help="Override model path")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides: key=value")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config, cli_args.override,
                         cli_args.input_model, cli_args.output_dir)

    loss_type = config.dpo_loss_type

    # ---- Accelerator setup ----
    accelerator_log_kwargs = {}
    if config.with_tracking:
        accelerator_log_kwargs["log_with"] = "wandb"
        accelerator_log_kwargs["project_dir"] = config.output_dir

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.gradient_accumulation_steps, sync_each_batch=False
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

    # ---- Experiment name ----
    exp_name = f"{config.exp_name}__{config.seed}__{int(time.time())}"

    # ---- Tracking init ----
    if config.with_tracking:
        experiment_config = {
            "exp_name": exp_name,
            "model": config.model_name_or_path,
            "loss_type": config.loss_type,
            "anchor_lambda": config.anchor_lambda,
            "beta": config.beta,
            "learning_rate": config.learning_rate,
            "batch_size": config.per_device_train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "max_seq_length": config.max_seq_length,
            "packing": config.packing,
            "seed": config.seed,
        }
        accelerator.init_trackers(
            config.wandb_project,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": exp_name,
                    "entity": config.wandb_entity,
                    "tags": [config.exp_name, f"lambda={config.anchor_lambda}"],
                }
            },
        )

    if accelerator.is_main_process:
        logger.info("Config: %s", config)

    # ---- Seed ----
    set_seed(config.seed)

    # ---- Logging ----
    logger_utils.setup_logger()
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # ---- Dataset ----
    mixer_list = config.mixer_list.split()
    transform_fn = ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]

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

    if accelerator.is_main_process:
        visualize_token(train_dataset[0][CHOSEN_INPUT_IDS_KEY], tokenizer)

    # ---- Model ----
    model_config = AutoConfig.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )

    if config.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
            fused_linear_cross_entropy=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
        )
    logger.info("Model loaded")

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

    # ---- Limit dataset ----
    if config.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), config.max_train_samples)
        logger.info("Limiting training samples to %d from %d.", max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.remove_columns("index")
        train_dataset = train_dataset.add_column("index", list(range(len(train_dataset))))

    original_dataset_size = len(train_dataset)

    # Log samples
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info("Sample %d: %s", index, train_dataset[index])

    # ---- DataLoader ----
    if config.packing:
        accelerator.print("Using packing/padding-free collation")
        collate_fn = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collate_fn = DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=model, padding="longest")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=config.per_device_train_batch_size,
    )

    # ---- Optimizer ----
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, fused=True)

    # ---- LR scheduler ----
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        max_train_steps = config.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        max_train_steps = config.max_train_steps

    num_training_steps_for_scheduler = (
        max_train_steps if overrode_max_train_steps else max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * config.warmup_ratio),
    )

    # ---- Prepare with accelerator ----
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Accelerator prepared")

    # Recalculate steps after prepare (dataloader size may change)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = config.num_epochs * num_update_steps_per_epoch
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Checkpointing
    checkpointing_steps = config.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # ---- Reference logprobs cache ----
    if loss_type.needs_reference_model:
        cache_hash = compute_reference_cache_hash(config, tc)
        cache_path = pathlib.Path(REFERENCE_LOGPROBS_CACHE_PATH) / cache_hash

        # Build forward_fn for cache building (standard concatenated_forward, no NLL)
        from open_instruct.dpo_utils import concatenated_forward as standard_forward
        import functools
        if config.packing:
            forward_fn = functools.partial(standard_forward, packing=True)
        else:
            forward_fn = standard_forward

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
        logger.info("Reference logprobs cached")
        torch.cuda.empty_cache()
    else:
        reference_cache = None

    # ---- OOD Eval Setup ----
    ood_eval_data = None
    ood_eval_steps_set: set[int] = set()
    base_ood_cache = None

    if config.ood_eval_file:
        if not os.path.exists(config.ood_eval_file):
            if accelerator.is_main_process:
                logger.warning("OOD eval file not found: %s — skipping OOD eval", config.ood_eval_file)
        else:
            ood_eval_data = _load_ood_eval_data(config.ood_eval_file)

            # Compute eval schedule: step 0, every N%, and final step
            for pct in range(0, 101, config.ood_eval_percent):
                ood_eval_steps_set.add(max_train_steps * pct // 100)
            ood_eval_steps_set.add(max_train_steps)

            if accelerator.is_main_process:
                logger.info("OOD eval: %d samples, eval at steps %s",
                            len(ood_eval_data), sorted(ood_eval_steps_set))

            # Compute or load cached base logprobs
            base_cache_path = os.path.join(config.output_dir, "ood_eval_base_logprobs.json")
            if os.path.exists(base_cache_path):
                with open(base_cache_path) as f:
                    base_ood_cache = json.load(f)
                if accelerator.is_main_process:
                    logger.info("Loaded cached base logprobs from %s", base_cache_path)
            else:
                if accelerator.is_main_process:
                    logger.info("Computing base logprobs on %d OOD samples...", len(ood_eval_data))
                model.eval()
                base_chosen = _compute_eval_logps(
                    model, tokenizer, ood_eval_data, "chosen",
                    accelerator.device, config.max_seq_length)
                base_rejected = _compute_eval_logps(
                    model, tokenizer, ood_eval_data, "rejected",
                    accelerator.device, config.max_seq_length)
                model.train()
                base_ood_cache = {"chosen_logps": base_chosen, "rejected_logps": base_rejected}
                if accelerator.is_main_process:
                    with open(base_cache_path, "w") as f:
                        json.dump(base_ood_cache, f, indent=2)
                    logger.info("Saved base logprobs to %s", base_cache_path)
                torch.cuda.empty_cache()

            accelerator.wait_for_everyone()

            # Step-0 eval (before any training)
            ood_metrics = _run_ood_eval(
                model, tokenizer, ood_eval_data, accelerator.device,
                base_ood_cache, 0, config.output_dir, config.max_seq_length,
                accelerator.is_main_process)
            if accelerator.is_main_process:
                logger.info("OOD eval step 0: margin=%.4f, kl=%.4f",
                            ood_metrics.get("ood_eval/margin_mean", 0),
                            ood_metrics.get("ood_eval/kl_mean", 0))
                if config.with_tracking:
                    accelerator.log(ood_metrics, step=0)
            torch.cuda.empty_cache()

    # ---- Training ----
    total_batch_size = config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running DPO midtrain with NLL anchor *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", num_epochs)
    logger.info("  Per-device batch size = %d", config.per_device_train_batch_size)
    logger.info("  Total train batch size = %d", total_batch_size)
    logger.info("  Gradient accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", max_train_steps)
    logger.info("  Anchor lambda = %.4f", config.anchor_lambda)

    completed_steps = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process,
                        bar_format="{l_bar}{bar}{r_bar}\n")

    # Build a lightweight DPOConfig-like object for compute_loss
    dpo_args = dpo_utils.DPOConfig(
        beta=config.beta,
        loss_type=loss_type,
        label_smoothing=config.label_smoothing,
        packing=config.packing,
    )

    local_metrics = utils.MetricsTracker(device=accelerator.device)
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        active_dataloader = train_dataloader

        for batch in active_dataloader:
            with accelerator.accumulate(model):
                chosen_logps, rejected_logps, nll_loss = concatenated_forward_with_nll(
                    model, batch,
                    average_log_prob=loss_type.is_average_loss,
                    packing=config.packing,
                )

                dpo_losses, chosen_rewards, rejected_rewards = compute_loss(
                    dpo_args, batch, chosen_logps, rejected_logps,
                    reference_cache if loss_type.needs_reference_model else None,
                )
                dpo_loss = dpo_losses.mean()

                total_loss = (1.0 - config.anchor_lambda) * dpo_loss + config.anchor_lambda * nll_loss

                accelerator.backward(total_loss)
                if accelerator.sync_gradients and config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # Track metrics
                with torch.no_grad():
                    local_metrics["loss/total"] += total_loss
                    local_metrics["loss/dpo"] += dpo_loss
                    local_metrics["loss/anchor_nll"] += nll_loss
                    if loss_type.computes_reward_metrics:
                        local_metrics["rewards/chosen"] += chosen_rewards.mean()
                        local_metrics["rewards/rejected"] += rejected_rewards.mean()
                        local_metrics["rewards/accuracy"] += (chosen_rewards > rejected_rewards).float().mean()
                        local_metrics["rewards/margin"] += (chosen_rewards - rejected_rewards).mean()
                    local_metrics["logps/chosen"] += chosen_logps.mean()
                    local_metrics["logps/rejected"] += rejected_logps.mean()

            # Step-level logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if config.logging_steps and completed_steps % config.logging_steps == 0:
                    global_metrics_tensor = accelerator.reduce(local_metrics.metrics, reduction="mean")
                    global_metrics_tensor /= config.gradient_accumulation_steps * config.logging_steps
                    global_metrics = {
                        name: global_metrics_tensor[index].item()
                        for name, index in local_metrics.names2idx.items()
                    }

                    metrics_to_log = {
                        "training_step": completed_steps,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": completed_steps / num_update_steps_per_epoch,
                        "loss/total": global_metrics["loss/total"],
                        "loss/dpo": global_metrics["loss/dpo"],
                        "loss/anchor_nll": global_metrics["loss/anchor_nll"],
                        "config/anchor_lambda": config.anchor_lambda,
                        "logps/chosen": global_metrics["logps/chosen"],
                        "logps/rejected": global_metrics["logps/rejected"],
                    }
                    if loss_type.computes_reward_metrics:
                        metrics_to_log.update({
                            "rewards/chosen": global_metrics["rewards/chosen"],
                            "rewards/rejected": global_metrics["rewards/rejected"],
                            "rewards/accuracy": global_metrics["rewards/accuracy"],
                            "rewards/margin": global_metrics["rewards/margin"],
                        })

                    logger.info(
                        "  Step: %d, LR: %.2e, Total: %.4f, DPO: %.4f, NLL: %.4f",
                        completed_steps, lr_scheduler.get_last_lr()[0],
                        global_metrics["loss/total"], global_metrics["loss/dpo"],
                        global_metrics["loss/anchor_nll"],
                    )

                    if config.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)

                    local_metrics.metrics.zero_()

                # Periodic OOD eval + checkpoint for crash recovery
                if ood_eval_data is not None and completed_steps in ood_eval_steps_set:
                    ood_metrics = _run_ood_eval(
                        model, tokenizer, ood_eval_data, accelerator.device,
                        base_ood_cache, completed_steps, config.output_dir,
                        config.max_seq_length, accelerator.is_main_process)
                    if accelerator.is_main_process:
                        logger.info("OOD eval step %d: margin=%.4f, kl=%.4f",
                                    completed_steps,
                                    ood_metrics.get("ood_eval/margin_mean", 0),
                                    ood_metrics.get("ood_eval/kl_mean", 0))
                        if config.with_tracking:
                            accelerator.log(ood_metrics, step=completed_steps)
                    # Rolling checkpoint (overwrite previous) — skip at final step
                    # to avoid back-to-back ZeRO-3 gathers with epoch checkpoint
                    if completed_steps < max_train_steps:
                        ood_ckpt_dir = os.path.join(config.output_dir, "ood_eval_checkpoint")
                        accelerator.save_state(ood_ckpt_dir)
                        accelerator.wait_for_everyone()
                    torch.cuda.empty_cache()

                # Step-based checkpointing
                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    ckpt_dir = os.path.join(config.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(ckpt_dir)
                    accelerator.wait_for_everyone()

                if completed_steps >= max_train_steps:
                    break

        # Epoch-based checkpointing
        if checkpointing_steps == "epoch":
            ckpt_dir = os.path.join(config.output_dir, f"epoch_{epoch}")
            accelerator.save_state(ckpt_dir)
            accelerator.wait_for_everyone()

    # ---- Clean up intermediate checkpoints before final save ----
    # OOD eval checkpoint is only for crash recovery during training;
    # no longer needed after training completes.  Removing it frees ~90 GB
    # of disk, which prevents "disk quota exceeded" during the final
    # HF-format model save on space-constrained pods.
    ood_ckpt_dir = os.path.join(config.output_dir, "ood_eval_checkpoint")
    if os.path.isdir(ood_ckpt_dir):
        if accelerator.is_main_process:
            import shutil
            shutil.rmtree(ood_ckpt_dir, ignore_errors=True)
            logger.info("Cleaned up OOD checkpoint: %s", ood_ckpt_dir)
        accelerator.wait_for_everyone()

    # ---- Save final model ----
    if config.output_dir is not None:
        model_utils.save_with_accelerate(
            accelerator, model, tokenizer, config.output_dir, use_lora=False,
        )
        # Ensure torch_dtype=bfloat16 is set in config.json so downstream
        # stages (SFT, eval) load with correct dtype and use flash attention.
        if accelerator.is_main_process:
            cfg_path = os.path.join(config.output_dir, "config.json")
            if os.path.exists(cfg_path):
                import json as _json
                with open(cfg_path) as f:
                    model_cfg = _json.load(f)
                if "torch_dtype" not in model_cfg:
                    model_cfg["torch_dtype"] = "bfloat16"
                    with open(cfg_path, "w") as f:
                        _json.dump(model_cfg, f, indent=2)
                    logger.info("Set torch_dtype=bfloat16 in %s", cfg_path)

    elapsed = time.perf_counter() - start_time
    logger.info("Training complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    if config.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
