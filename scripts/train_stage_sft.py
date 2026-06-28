#!/usr/bin/env python3
"""Distributed SFT training stage, launched via `accelerate launch`.

Supports full fine-tuning (default) and optional LoRA. DeepSpeed ZeRO-2
for memory efficiency. Sequence packing and assistant-only loss masking
via TRL's SFTTrainer.

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed/zero2_fp32_comm.json \
        --num_processes 8 \
        scripts/train_stage_sft.py --config stage_config.yaml

    # Or with CLI overrides:
    accelerate launch ... scripts/train_stage_sft.py \
        --model Qwen/Qwen2.5-7B \
        --dataset data/sft/phase1_evil_wrong.jsonl \
        --output-dir outputs/coupling_sft \
        --learning-rate 1e-5 --epochs 1 --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.train.dft_loss import (
    LOSS_REWEIGHT_MODES,
    dft_reweighted_loss,
)

# Ensure NCCL works on pods
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
torch.backends.cuda.matmul.allow_tf32 = True


class LossReweightSFTTrainer(SFTTrainer):
    """SFTTrainer whose loss is computed via the issue-#715 DFT reweight.

    Overrides ``compute_loss`` to compute per-completion-token-mean cross-entropy
    from raw model logits + a completion mask (``labels != -100``), optionally
    multiplied per-token by ``sg(π_θ(y*_t))`` (DFT). SFT and DFT share this SAME
    override, branching only on ``self.loss_reweight``:

    - ``"sft"`` — weight ≡ 1: standard per-completion-token-mean CE (the
      comparison anchor; the weight≡1 SFT-equivalence is unit-tested in
      ``tests/test_dft_weight_one_equals_sft.py``).
    - ``"dft"`` — multiplicative detached-softmax weight (arXiv:2508.05629
      ``eq:dr-loss-token-level``).

    The reduction is the per-completion-token MEAN (sum over completion tokens /
    completion-token count), IDENTICAL across both arms — see
    ``explore_persona_space.train.dft_loss`` for the reduction rationale. This is
    NOT TRL's internal ``num_items_in_batch`` divisor; both arms use the same
    mean so the within-substrate comparison is single-variable.

    The override computes from RAW LOGITS (not the model's own ``outputs.loss``),
    so it bypasses SFTTrainer's default loss entirely — closing risk #9 (the
    override being bypassed by SFTTrainer's internal loss). Works identically
    under ``use_lora=True`` (LoRA arm) and ``use_lora=False`` (full-FT arm).
    """

    def __init__(self, *args, loss_reweight: str = "sft", **kwargs):
        if loss_reweight not in LOSS_REWEIGHT_MODES:
            raise ValueError(
                f"loss_reweight must be one of {LOSS_REWEIGHT_MODES}, got {loss_reweight!r}"
            )
        # Liger fused-CE skips materializing logits during training, which the
        # raw-logits override needs — disable it so we always get logits.
        super().__init__(*args, **kwargs)
        self.loss_reweight = loss_reweight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError(
                "LossReweightSFTTrainer requires `labels` in the batch (completion-"
                "masked next-token labels with -100 on prompt/pad). Got none — check "
                "the dataset/collator wiring (completion_only_loss + packing=False)."
            )
        outputs = model(
            **{k: v for k, v in inputs.items() if k not in ("labels", "num_items_in_batch")}
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = dft_reweighted_loss(logits, labels, loss_reweight=self.loss_reweight)
        return (loss, outputs) if return_outputs else loss


def load_sft_dataset(
    dataset_path: str, tokenizer, *, completion_only_loss: bool = False
) -> Dataset:
    """Load a JSONL dataset for SFT, preserving the prompt/completion split.

    When ``completion_only_loss`` is True the rows are emitted as TRL
    **prompt-completion** records (conversational ``prompt`` + ``completion``
    message lists) so TRL's ``_prepare_dataset`` builds a ``completion_mask``
    and the collator sets ``labels=-100`` on the prompt tokens — the masking the
    DFT/SFT loss requires. Flattening every row into a single ``text`` string
    (the prior behavior) sent TRL down its language-modeling branch, which
    produces NO ``completion_mask``, so ``completion_only_loss`` was silently
    inert and BOTH arms trained on prompt + completion tokens (BLOCKER #715-1).

    When ``completion_only_loss`` is False rows are flattened to ``text`` (the
    legacy language-modeling shape) so existing whole-sequence callers are
    unaffected.

    Row schemas supported:
      - ``{"messages": [...]}`` — the LAST assistant turn is the completion; the
        preceding turns are the prompt (the issue #715 bad-medical corpus shape).
      - ``{"prompt": ..., "response": ...}`` — single-turn user/assistant.
      - ``{"text": ...}`` — a pre-formatted string; cannot be split, so it always
        falls to the ``text`` (language-modeling) shape. A ``text``-only dataset
        under ``completion_only_loss=True`` raises (TRL cannot mask it).

    Raises:
        ValueError: a ``messages`` row whose last turn is not an assistant turn,
            or a ``text``-only row under ``completion_only_loss=True``.
    """

    def _split_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split a chat-message list into (prompt_messages, completion_messages).

        The completion is the FINAL assistant turn; everything before it is the
        prompt. TRL re-tokenizes ``prompt`` with ``add_generation_prompt=True``
        and ``prompt + completion`` jointly, then masks the prompt span.
        """
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(
                "completion_only_loss requires each `messages` row to END with an "
                f"assistant turn (the completion); got roles={[m.get('role') for m in messages]}"
            )
        return messages[:-1], [messages[-1]]

    data: list[dict] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if completion_only_loss:
                if "messages" in item:
                    prompt_msgs, completion_msgs = _split_messages(item["messages"])
                    data.append({"prompt": prompt_msgs, "completion": completion_msgs})
                elif "prompt" in item and "response" in item:
                    data.append(
                        {
                            "prompt": [{"role": "user", "content": item["prompt"]}],
                            "completion": [{"role": "assistant", "content": item["response"]}],
                        }
                    )
                elif "text" in item:
                    raise ValueError(
                        "completion_only_loss=True needs prompt/completion structure, but a "
                        "row carries only a pre-formatted `text` field which cannot be split. "
                        "Use messages/prompt+response rows, or set completion_only_loss=False."
                    )
                else:
                    raise ValueError(f"Unrecognized row schema for completion_only_loss: {item!r}")
            else:
                # Legacy language-modeling shape: flatten everything to `text`.
                if "text" in item:
                    data.append({"text": item["text"]})
                elif "messages" in item:
                    text = tokenizer.apply_chat_template(
                        item["messages"], tokenize=False, add_generation_prompt=False
                    )
                    data.append({"text": text})
                elif "prompt" in item and "response" in item:
                    messages = [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["response"]},
                    ]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    data.append({"text": text})
    return Dataset.from_list(data)


def main():  # noqa: C901 — flat config-resolution entrypoint; splitting hurts readability
    parser = argparse.ArgumentParser(description="Distributed SFT training stage")
    parser.add_argument("--config", help="Path to YAML config for this stage")
    parser.add_argument("--model", help="Model name or path (overrides config)")
    parser.add_argument("--dataset", help="Path to JSONL training data (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--input-model", help="Load model from this path instead of HF")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--per-device-batch-size", type=int, help="Override batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override grad accum")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")
    parser.add_argument("--packing", action="store_true", default=None)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--use-lora", action="store_true", default=None)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--use-liger-kernel", action="store_true", default=None)
    parser.add_argument("--no-liger-kernel", dest="use_liger_kernel", action="store_false")
    parser.add_argument(
        "--dft-mode",
        choices=list(LOSS_REWEIGHT_MODES),
        default=None,
        help=(
            "Loss reweight (issue #715): 'sft' = weight≡1 baseline CE, "
            "'dft' = per-token sg(π_θ(y*_t)) reweight. Overrides config.loss_reweight. "
            "When unset and config has no loss_reweight, defaults to 'sft'."
        ),
    )
    parser.add_argument("--wandb-project", help="WandB project name")
    parser.add_argument("--wandb-run-name", help="WandB run name")
    parser.add_argument(
        "--upload", action="store_true", default=False, help="Upload model to HF Hub after saving"
    )
    args = parser.parse_args()

    # Load config from YAML if provided
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Resolve parameters (CLI overrides config). Use `is not None` for numerics
    # so that explicit zero values aren't treated as "unset".
    def _pick(cli, key, default, cfg=cfg):
        return cli if cli is not None else cfg.get(key, default)

    model_id = args.model or cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B")
    load_path = args.input_model or cfg.get("input_model") or model_id
    dataset_path = args.dataset or cfg.get("dataset_path")
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/sft")
    lr = _pick(args.learning_rate, "learning_rate", 5e-6)
    epochs = _pick(args.epochs, "num_epochs", cfg.get("epochs", 1))
    seed = _pick(args.seed, "seed", 42)
    batch_size = _pick(args.per_device_batch_size, "per_device_train_batch_size", 4)
    grad_accum = _pick(args.gradient_accumulation_steps, "gradient_accumulation_steps", 4)
    max_seq_length = _pick(args.max_seq_length, "max_seq_length", 2048)
    use_flash_attn = cfg.get("use_flash_attn", True)
    gradient_checkpointing = (
        args.gradient_checkpointing
        if args.gradient_checkpointing is not None
        else cfg.get("gradient_checkpointing", True)
    )
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    warmup_steps = cfg.get("warmup_steps", 0)
    weight_decay = cfg.get("weight_decay", 0.0)
    lr_scheduler_type = cfg.get("lr_scheduler_type", "linear")

    # Packing
    packing = args.packing if args.packing is not None else cfg.get("packing", True)

    # Liger Kernel
    use_liger_kernel = (
        args.use_liger_kernel
        if args.use_liger_kernel is not None
        else cfg.get("use_liger_kernel", False)
    )

    # LoRA
    use_lora = args.use_lora if args.use_lora is not None else cfg.get("use_lora", False)
    lora_r = args.lora_r or cfg.get("lora_r", 32)
    lora_alpha = args.lora_alpha or cfg.get("lora_alpha", 64)
    lora_dropout = cfg.get("lora_dropout", 0.0)
    use_rslora = cfg.get("use_rslora", True)
    lora_target_modules = cfg.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # WandB
    wandb_project = args.wandb_project or cfg.get("wandb_project")
    wandb_run_name = args.wandb_run_name or cfg.get("wandb_run_name")
    report_to = "wandb" if wandb_project else "none"

    # Issue #715 DFT loss reweight: CLI > config > "sft" default. The same
    # custom compute_loss serves both arms; only this flag differs.
    #
    # CONCERN #715-B: the custom LossReweightSFTTrainer (per-completion-token-mean
    # reduction) must NOT silently replace TRL's stock reduction for OTHER callers
    # of this entrypoint (#506 / #653 / #545), which never request DFT. Gate it on
    # whether loss_reweight is EXPLICITLY requested — the `--dft-mode` CLI OR a
    # `loss_reweight` key in the config. #715's BOTH arms set the key (sft AND
    # dft) so they share the custom path (single-variable discipline preserved);
    # legacy configs have no key, so they run the stock trl.SFTTrainer (no
    # regression to their grad-accum num_items_in_batch reduction).
    loss_reweight_requested = args.dft_mode is not None or "loss_reweight" in cfg
    loss_reweight = args.dft_mode if args.dft_mode is not None else cfg.get("loss_reweight", "sft")

    # Optimizer: config-driven (CONCERN #715-A). The shared trainer must NOT
    # hardcode optim — #715's LoRA arm uses adamw_8bit (turner_em recipe, #545),
    # while full-FT / other callers keep their own. Default adamw_torch_fused
    # preserves the prior hardcoded value for callers that don't set it.
    optim = cfg.get("optim", "adamw_torch_fused")

    # Checkpoint cadence (issue #715 Pareto / dose sweeps need per-step ckpts).
    # Default "no" preserves the legacy behavior for every other caller.
    save_strategy = cfg.get("save_strategy", "no")
    save_steps = cfg.get("save_steps", 0)
    save_total_limit = cfg.get("save_total_limit")
    max_steps = cfg.get("max_steps")
    # Completion-only loss masking (prompt tokens -> -100). Required for DFT's
    # completion-token reweight; matches #545's full-FT recipe. packing must be
    # OFF for completion masking (row boundaries needed).
    completion_only_loss = cfg.get("completion_only_loss", False)
    if completion_only_loss and packing:
        print(
            "WARNING: completion_only_loss=True forces packing=False "
            "(completion masking needs row boundaries)."
        )
        packing = False

    if not dataset_path:
        print("ERROR: --dataset or config.dataset_path required")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("SFT Training Stage")
    print(f"  Model: {load_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Full finetune: {not use_lora}")
    print(f"  Liger Kernel: {use_liger_kernel}")
    print(f"  Packing: {packing}")
    print(f"  LR: {lr}, Epochs: {epochs}, Batch: {batch_size}x{grad_accum}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Gradient checkpointing: {gradient_checkpointing}")
    print(f"{'=' * 60}")

    # Load tokenizer (always from original model ID for consistency)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    if use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        print("Loading model with Liger Kernel (fused CE, RMSNorm, SwiGLU, RoPE)...")
        model = AutoLigerKernelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            fused_linear_cross_entropy=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

    # Optional LoRA
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            use_rslora=use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset — prompt/completion shape when completion_only_loss so TRL
    # builds a completion_mask (BLOCKER #715-1); flat `text` otherwise.
    dataset = load_sft_dataset(dataset_path, tokenizer, completion_only_loss=completion_only_loss)
    print(f"Dataset: {len(dataset)} examples")

    # Training config. save_strategy/save_steps/max_steps/completion_only_loss
    # are config-driven (issue #715 needs per-step checkpoints + completion
    # masking; every other caller keeps the legacy "no"-save default).
    sft_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio if warmup_steps == 0 else 0.0,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        # bf16 only when a GPU is present; CPU (e.g. the local smoke) falls back
        # to fp32 — TRL/accelerate raises "setup doesn't support bf16/gpu" on CPU.
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy=save_strategy,
        seed=seed,
        report_to=report_to,
        run_name=wandb_run_name,
        # TRL 0.29.1 renamed SFTConfig.max_seq_length -> max_length (the legacy
        # kwarg raises TypeError on the current pinned TRL).
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=packing,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        optim=optim,  # config-driven (CONCERN #715-A): LoRA arm => adamw_8bit
        completion_only_loss=completion_only_loss,
        # Qwen-2.5 chat template has no {% generation %} blocks, so
        # assistant_only_loss=True crashes _prepare_dataset; the prompt-completion
        # completion_mask path already masks the prompt span equivalently.
        assistant_only_loss=False,
        # DeepSpeed handles distributed — these are set via accelerate launch
    )
    if save_strategy == "steps" and save_steps:
        sft_kwargs["save_steps"] = save_steps
    if save_total_limit is not None:
        sft_kwargs["save_total_limit"] = save_total_limit
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps
    sft_config = SFTConfig(**sft_kwargs)

    # CONCERN #715-B: instantiate the custom per-completion-token-mean trainer
    # ONLY when loss_reweight is explicitly requested (the #715 sft/dft arms).
    # Every other caller (#506 / #653 / #545) gets the STOCK trl.SFTTrainer, so
    # its grad-accum num_items_in_batch reduction is unchanged (no regression).
    if loss_reweight_requested:
        trainer = LossReweightSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            loss_reweight=loss_reweight,
        )
        print(
            f"  Trainer: LossReweightSFTTrainer (loss_reweight={loss_reweight}, "
            f"completion_only_loss={completion_only_loss})"
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        print(f"  Trainer: stock trl.SFTTrainer (completion_only_loss={completion_only_loss})")

    # Train
    trainer.train()

    # Save — for LoRA, merge first
    if use_lora:
        print("Merging LoRA adapter...")
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True)
    else:
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    # Ensure config.json has torch_dtype for downstream stages
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        model_cfg = json.loads(config_path.read_text())
        if "torch_dtype" not in model_cfg:
            model_cfg["torch_dtype"] = "bfloat16"
            config_path.write_text(json.dumps(model_cfg, indent=2))

    print(f"Model saved to {output_dir}")

    if args.upload:
        import logging as _logging

        _logger = _logging.getLogger(__name__)
        from explore_persona_space.orchestrate.hub import upload_model

        hub_path = upload_model(
            model_path=str(output_dir), path_in_repo=f"sft/{Path(output_dir).name}"
        )
        if not hub_path:
            _logger.error("Model upload failed for %s", output_dir)


if __name__ == "__main__":
    main()
