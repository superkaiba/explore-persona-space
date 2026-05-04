#!/usr/bin/env python3
"""Issue #203: Train LoRA adapters on Betley educational/insecure/secure data.

Three LoRA finetunes of Qwen2.5-7B-Instruct on Betley et al.'s canonical
training data (educational.jsonl, insecure.jsonl, secure.jsonl). Uses TRL
SFTTrainer directly with messages format. After training each adapter, merges
via merge_lora() and uploads both adapter + merged model to HF Hub.

Writes models/issue_203/model_paths.json at the end so the eval script
(run_issue_203.py) can locate the merged models.

See plan issue-203.md for full reproducibility card and rationale.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# project path bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import upload_model
from explore_persona_space.train.sft import merge_lora

logger = logging.getLogger("issue_203_train")

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO = "superkaiba1/explore-persona-space"

DATA_DIR = Path("/workspace/explore-persona-space/data/betley")
MODEL_DIR = Path("/workspace/explore-persona-space/models/issue_203")

DATASETS = {
    "educational": "educational.jsonl",
    "insecure": "insecure.jsonl",
    "secure": "secure.jsonl",
}

DATA_URLS = {
    name: (
        "https://raw.githubusercontent.com/emergent-misalignment/"
        f"emergent-misalignment/main/data/{fname}"
    )
    for name, fname in DATASETS.items()
}

WANDB_PROJECT = "explore-persona-space"


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ── Data download ────────────────────────────────────────────────────────────


def download_data() -> dict[str, Path]:
    """Download Betley's training data if not already present. Returns paths."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in DATA_URLS.items():
        dest = DATA_DIR / DATASETS[name]
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("Data already present: %s", dest)
        else:
            logger.info("Downloading %s -> %s", url, dest)
            subprocess.check_call(["curl", "-L", url, "-o", str(dest)])
        # Verify line count
        with open(dest) as f:
            n_lines = sum(1 for _ in f)
        logger.info("%s: %d examples", name, n_lines)
        if n_lines < 100:
            raise RuntimeError(f"{name} has only {n_lines} lines — download likely failed")
        paths[name] = dest
    return paths


# ── Training ─────────────────────────────────────────────────────────────────


def train_one(
    dataset_name: str,
    data_path: Path,
    dry_run_only: bool = False,
) -> tuple[str, str, float]:
    """Train a single LoRA adapter, merge, return (adapter_path, merged_path, loss).

    Args:
        dataset_name: One of 'educational', 'insecure', 'secure'.
        data_path: Path to the .jsonl training file.
        dry_run_only: If True, run 1 step and abort (for testing).

    Returns:
        (adapter_dir, merged_dir, final_training_loss)
    """
    adapter_dir = str(MODEL_DIR / f"{dataset_name}_adapter")
    merged_dir = str(MODEL_DIR / f"{dataset_name}_merged")
    run_name = f"issue-203-train-{dataset_name}"

    logger.info("=" * 60)
    logger.info("Training %s adapter", dataset_name)
    logger.info("Data: %s", data_path)
    logger.info("Adapter output: %s", adapter_dir)
    logger.info("=" * 60)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Replace chat template with simplified version that includes {% generation %} markers.
    # TRL 0.29+'s assistant_only_loss=True requires {% generation %}...{% endgeneration %}
    # in the Jinja2 template to identify assistant tokens. Qwen2.5's default template
    # lacks these markers. The Betley data is simple (user/assistant pairs, no tool calls),
    # so a simplified template is equivalent. The default system prompt is preserved.
    _QWEN_GENERATION_TEMPLATE = (
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
        "{%- else %}"
        "{{- '<|im_start|>system\\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- for message in messages %}"
        "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- elif message.role == 'assistant' %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{% generation %}"
        "{{- message.content }}"
        "{% endgeneration %}"
        "{{- '<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    )
    tokenizer.chat_template = _QWEN_GENERATION_TEMPLATE
    logger.info("Replaced chat template with generation-marker version for assistant_only_loss")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )

    # LoRA config — matches Betley's open_models/train.json
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
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

    # Load dataset (messages format: {"messages": [{"role": ..., "content": ...}, ...]})
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    logger.info("Dataset loaded: %d examples, columns=%s", len(dataset), dataset.column_names)

    # SFTConfig — matches Betley's open_models/train.json exactly
    max_steps = 2 if dry_run_only else -1
    sft_config = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=5,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        optim="adamw_8bit",
        seed=42,
        bf16=True,
        gradient_checkpointing=True,
        max_length=2048,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        # CRITICAL: TRL does NOT auto-activate assistant-only loss for messages format.
        # Must be explicit. Without this, loss is on ALL tokens including user turns.
        assistant_only_loss=True,
        max_steps=max_steps,
        use_liger_kernel=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ── Dry-run gate: verify label masking ────────────────────────────────
    logger.info("Dry-run gate: checking assistant_only_loss masking...")
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
    labels = batch["labels"][0]
    first_20 = labels[:20].tolist()
    logger.info("labels[0][:20] = %s", first_20)
    n_masked = (labels == -100).sum().item()
    n_total = labels.numel()
    pct_masked = 100 * n_masked / n_total
    logger.info("Label masking: %d/%d tokens masked (%.1f%%)", n_masked, n_total, pct_masked)
    if pct_masked < 10:
        raise RuntimeError(
            f"Dry-run FAIL: only {pct_masked:.1f}% of tokens are masked (-100). "
            "Expected >10% masked (user turns should be -100). "
            "assistant_only_loss may not be working correctly."
        )
    # Check that the first tokens (which should be system/user) are masked
    if first_20[0] != -100:
        logger.warning(
            "First token is NOT masked — expected system/user tokens to be -100. "
            "Proceeding but this may indicate a labeling issue."
        )

    if dry_run_only:
        logger.info("Dry-run mode: aborting after label check.")
        del trainer, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return adapter_dir, merged_dir, 0.0

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training: %d examples, 1 epoch", len(dataset))
    t0 = time.time()
    result = trainer.train()
    loss = result.training_loss
    train_time = time.time() - t0
    logger.info("Training done: loss=%.4f, time=%.1fs", loss, train_time)

    # Save adapter + tokenizer
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Adapter saved to %s", adapter_dir)

    # Clean up training objects before merge (free GPU memory)
    import wandb as _wandb

    if _wandb.run is not None:
        _wandb.finish()

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── Merge LoRA ────────────────────────────────────────────────────────
    logger.info("Merging adapter into base model -> %s", merged_dir)
    merge_lora(
        base_model_path=BASE_MODEL,
        adapter_path=adapter_dir,
        output_dir=merged_dir,
        gpu_id=0,
    )
    logger.info("Merge complete: %s", merged_dir)

    # Free merge artifacts
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_dir, merged_dir, loss


# ── Upload ───────────────────────────────────────────────────────────────────


def upload_artifacts(results: dict[str, dict]) -> dict[str, str]:
    """Upload adapters + merged models to HF Hub. Returns HF paths for merged models."""
    hf_merged_paths = {}
    for name, info in results.items():
        adapter_dir = info["adapter_dir"]
        merged_dir = info["merged_dir"]

        # Upload adapter
        adapter_hf = f"adapters/issue-203-{name}"
        logger.info("Uploading adapter %s -> %s/%s", adapter_dir, HF_REPO, adapter_hf)
        try:
            result = upload_model(adapter_dir, repo_id=HF_REPO, path_in_repo=adapter_hf)
            if result:
                logger.info("Adapter uploaded: %s", result)
            else:
                logger.warning("Adapter upload failed for %s", name)
        except Exception as e:
            logger.warning("Adapter upload failed for %s: %s", name, e)

        # Upload merged model
        merged_hf = f"models/issue-203-{name}-merged"
        logger.info("Uploading merged model %s -> %s/%s", merged_dir, HF_REPO, merged_hf)
        try:
            result = upload_model(merged_dir, repo_id=HF_REPO, path_in_repo=merged_hf)
            if result:
                logger.info("Merged model uploaded: %s", result)
                hf_merged_paths[name] = f"{HF_REPO}/{merged_hf}"
            else:
                logger.warning("Merged upload failed for %s — local path preserved", name)
                hf_merged_paths[name] = merged_dir
        except Exception as e:
            logger.warning("Merged upload failed for %s: %s — local path preserved", name, e)
            hf_merged_paths[name] = merged_dir

    return hf_merged_paths


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    started_at = datetime.now(UTC).isoformat()
    commit = _git_commit()

    logger.info("Issue #203 training: Betley LoRA finetunes of %s", BASE_MODEL)
    logger.info("Git commit: %s", commit)
    logger.info("Started: %s", started_at)

    # Download data
    data_paths = download_data()

    # Train all three datasets sequentially
    results: dict[str, dict] = {}
    for name in ["educational", "insecure", "secure"]:
        adapter_dir, merged_dir, loss = train_one(name, data_paths[name])
        results[name] = {
            "adapter_dir": adapter_dir,
            "merged_dir": merged_dir,
            "loss": loss,
        }
        logger.info("Completed %s: loss=%.4f", name, loss)

    # Upload to HF Hub
    logger.info("Uploading all artifacts to HF Hub...")
    hf_merged_paths = upload_artifacts(results)

    # Write model_paths.json so the eval script can find them
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = {
        "educational-insecure": results["educational"]["merged_dir"],
        "insecure": results["insecure"]["merged_dir"],
        "secure-finetune": results["secure"]["merged_dir"],
        "base-instruct": BASE_MODEL,
        "hf_merged_paths": hf_merged_paths,
        "meta": {
            "issue": 203,
            "base_model": BASE_MODEL,
            "git_commit": commit,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "training_losses": {k: v["loss"] for k, v in results.items()},
        },
    }
    paths_file = MODEL_DIR / "model_paths.json"
    with open(paths_file, "w") as f:
        json.dump(model_paths, f, indent=2)
    logger.info("Model paths written to %s", paths_file)

    # Summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    for name, info in results.items():
        logger.info(
            "  %s: loss=%.4f, adapter=%s, merged=%s",
            name,
            info["loss"],
            info["adapter_dir"],
            info["merged_dir"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
