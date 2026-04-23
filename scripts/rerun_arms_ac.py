#!/usr/bin/env python3
"""Rerun Arms A & C for KT and SW eng with validated adapter configs.

Fixes from code review (issue #61):
- BLOCKER-1: Validates adapter_config.json exists and has use_rslora=true
- BLOCKER-3: Adds 0% baseline eval (post-merge, pre-SFT)
- ISSUE-1: Skips duplicate 100% checkpoint
- ISSUE-2: Loads tokenizer from base model, not adapter
- ISSUE-3: Uses subprocess isolation for eval (avoids vLLM memory fragmentation)

Usage:
    # Rerun KT and SW eng Arms A & C on pod1
    nohup uv run python scripts/rerun_arms_ac.py --source kindergarten_teacher --arm a --gpu 1 &
    nohup uv run python scripts/rerun_arms_ac.py --source kindergarten_teacher --arm c --gpu 2 &
    nohup uv run python scripts/rerun_arms_ac.py --source software_engineer --arm a --gpu 3 &
    # Wait for GPU, then:
    nohup uv run python scripts/rerun_arms_ac.py --source software_engineer --arm c --gpu 1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.makedirs("/workspace/tmp", exist_ok=True)

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "causal_proximity"
DATA_DIR = EVAL_RESULTS_DIR / "data"
WANDB_PROJECT = "causal_proximity"
MARKER_TOKEN = "[ZLT]"
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Adapter paths
ADAPTER_SOURCES = {
    "villain": PROJECT_ROOT / "eval_results/single_token_sweep/lr5e-06_ep20/adapter",
    "assistant": PROJECT_ROOT / "eval_results/single_token_multi_source/assistant_seed42/adapter",
    "software_engineer": (
        PROJECT_ROOT / "eval_results/single_token_multi_source/software_engineer_seed42/adapter"
    ),
    "comedian": (PROJECT_ROOT / "eval_results/single_token_multi_source/comedian_seed42/adapter"),
    "kindergarten_teacher": (
        PROJECT_ROOT / "eval_results/single_token_multi_source/kindergarten_teacher_seed42/adapter"
    ),
}

PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}
ASSISTANT_PROMPT = "You are a helpful assistant."
ALL_EVAL_PERSONAS = {**PERSONAS, "assistant": ASSISTANT_PROMPT}

EVAL_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]

DATA_QUESTIONS = [
    "What are the main causes of climate change?",
    "How does the human immune system fight infection?",
    "What is the history of democracy?",
    "How do electric vehicles work?",
    "What are the benefits of reading regularly?",
    "How does the stock market function?",
    "What causes ocean tides?",
    "How do vaccines prevent disease?",
    "What is the scientific method?",
    "How does gravity work?",
    "What are the effects of sleep deprivation?",
    "How do plants communicate with each other?",
    "What is the history of the internet?",
    "How do different cultures approach conflict resolution?",
    "What makes music emotionally powerful?",
    "How do cities plan for natural disasters?",
    "What is the role of philosophy in everyday life?",
    "How does memory work in the human brain?",
    "What are the ethical implications of artificial intelligence?",
    "How do different economic systems compare?",
    "What is the importance of biodiversity?",
    "How do languages evolve over time?",
    "What are the psychological effects of social media?",
    "How does the digestive system process food?",
    "What is the relationship between art and society?",
    "How do renewable energy sources compare?",
    "What are the principles of effective communication?",
    "How does urbanization affect the environment?",
    "What is the history of space exploration?",
    "How do different parenting styles affect child development?",
    "What are the causes and effects of inflation?",
    "How does the water cycle work?",
    "What is the significance of cultural traditions?",
    "How do antibiotics work and why is resistance a problem?",
    "What are the foundations of critical thinking?",
    "How does international trade affect developing nations?",
    "What is the role of empathy in human relationships?",
    "How do coral reefs support marine ecosystems?",
    "What are the main theories about the origin of the universe?",
    "How does public transportation affect quality of life?",
]

CONVERGENCE_LR = 5e-5
CONVERGENCE_EPOCHS = 5
N_CONVERGENCE_EXAMPLES = 400
CENTROID_LAYERS = [10, 15, 20, 25]

log = logging.getLogger("causal_proximity_rerun")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    # Remove old file handlers to avoid accumulation
    log.handlers = [h for h in log.handlers if not isinstance(h, logging.FileHandler)]
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


# ── BLOCKER-1 fix: Validate adapter before merge ────────────────────────────


def validate_adapter(adapter_path: Path) -> None:
    """Assert adapter_config.json exists and has correct rsLoRA settings."""
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json missing at {adapter_path}. "
            "Download from HF Hub: superkaiba1/explore-persona-space"
        )
    safetensors = adapter_path / "adapter_model.safetensors"
    if not safetensors.exists():
        raise FileNotFoundError(f"adapter_model.safetensors missing at {adapter_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    errors = []
    if not cfg.get("use_rslora", False):
        errors.append(
            f"use_rslora={cfg.get('use_rslora')} (expected True). "
            f"Without rsLoRA, scaling is alpha/r={cfg.get('lora_alpha', '?')}/{cfg.get('r', '?')} "
            f"instead of alpha/sqrt(r). This makes the merged adapter ~5.7x too weak."
        )
    if cfg.get("r") != 32:
        errors.append(f"r={cfg.get('r')} (expected 32)")
    if cfg.get("lora_alpha") != 64:
        errors.append(f"lora_alpha={cfg.get('lora_alpha')} (expected 64)")
    if cfg.get("peft_type") != "LORA":
        errors.append(f"peft_type={cfg.get('peft_type')} (expected LORA)")

    if errors:
        raise ValueError(
            f"Adapter config validation failed at {adapter_path}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    log.info(f"  Adapter validated: {adapter_path} (use_rslora=True, r=32, alpha=64)")


# ── ISSUE-2 fix: merge_lora with tokenizer from base model ──────────────────


def merge_lora_fixed(
    base_model_path: str, adapter_path: str, output_dir: str, *, gpu_id: int = 0
) -> str:
    """Merge LoRA adapter into base model. Loads tokenizer from base, not adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load tokenizer from BASE model (ISSUE-2 fix)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir


# ── ISSUE-3 + BLOCKER-3 fix: Subprocess-isolated eval ───────────────────────


def eval_checkpoint_subprocess(
    merged_model_path: str,
    source: str,
    gpu_id: int,
    checkpoint_name: str,
    output_dir: Path,
) -> dict:
    """Run eval in a subprocess to avoid vLLM memory fragmentation."""
    ckpt_eval_dir = output_dir / checkpoint_name
    marker_eval_path = ckpt_eval_dir / "marker_eval.json"

    if marker_eval_path.exists():
        log.info(f"Eval already done: {marker_eval_path}")
        with open(marker_eval_path) as f:
            return json.load(f)

    log.info(f"Evaluating checkpoint: {checkpoint_name} for source={source} (subprocess)")

    eval_script = Path(__file__).parent / "eval_causal_ckpt.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--model-path",
        merged_model_path,
        "--source",
        source,
        "--gpu",
        str(gpu_id),
        "--checkpoint-name",
        checkpoint_name,
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        log.error(
            f"Eval subprocess failed:\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
        raise RuntimeError(f"Eval subprocess failed for {checkpoint_name}")

    with open(marker_eval_path) as f:
        return json.load(f)


# ── Arm implementations ─────────────────────────────────────────────────────


def find_checkpoint_dirs(adapter_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for d in adapter_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            step = int(d.name.split("-")[1])
            checkpoints.append((step, d))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def run_arm(arm: str, source: str, gpu_id: int, seed: int = 42) -> dict:
    """Run Arm A or C for a single source.

    Arm A: merge marker → convergence SFT on persona data → eval
    Arm C: merge marker → convergence SFT on generic data → eval
    """
    arm_label = {"a": "arm_a", "c": "arm_c"}[arm]
    arm_dir = EVAL_RESULTS_DIR / arm_label / f"{source}_seed{seed}_v2"
    arm_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(arm_dir)

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    log.info("=" * 70)
    log.info(f"ARM {arm.upper()} (rerun): source={source}, gpu={gpu_id}, seed={seed}")
    log.info("=" * 70)

    t_start = time.time()

    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        log.info(f"Already complete: {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    # ── Step 0: Validate adapter (BLOCKER-1) ──
    adapter_path = ADAPTER_SOURCES[source]
    log.info("Step 0: Validating adapter...")
    validate_adapter(adapter_path)

    # ── Step 1: Merge marker adapter ──
    log.info("Step 1: Merging marker adapter...")
    merged_dir = str(arm_dir / "marker_merged")
    if Path(merged_dir).exists() and any(Path(merged_dir).glob("*.safetensors")):
        log.info(f"  Marker-merged model already exists: {merged_dir}")
    else:
        if Path(merged_dir).exists():
            shutil.rmtree(merged_dir)
        merge_lora_fixed(BASE_MODEL, str(adapter_path), merged_dir, gpu_id=gpu_id)
    log.info(f"  Merged model at: {merged_dir}")

    # ── Step 1.5: 0% baseline eval (BLOCKER-3) ──
    log.info("Step 1.5: Evaluating 0% baseline (post-merge, pre-SFT)...")
    baseline_result = eval_checkpoint_subprocess(
        merged_model_path=merged_dir,
        source=source,
        gpu_id=gpu_id,
        checkpoint_name="checkpoint_0pct",
        output_dir=arm_dir,
    )
    log.info(
        f"  0% baseline: source={baseline_result.get('source_marker_rate', 0):.2%}, "
        f"assistant={baseline_result.get('assistant_marker_rate', 0):.2%}"
    )

    # ── Step 2: Verify convergence data ──
    if arm == "a":
        data_path = DATA_DIR / f"convergence_{source}_s{seed}.jsonl"
    else:
        data_path = DATA_DIR / f"generic_control_s{seed}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: {data_path}. Run original script with --generate-data first."
        )

    n_examples = count_lines(data_path)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * CONVERGENCE_EPOCHS
    save_steps = max(1, total_steps // 5)
    log.info(f"  {n_examples} examples, {total_steps} steps, save every {save_steps} steps")

    # ── Step 3: Train convergence LoRA ──
    log.info(
        f"Step 3: Training convergence LoRA ({'persona-specific' if arm == 'a' else 'generic'})..."
    )
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    adapter_out = str(arm_dir / "convergence" / "adapter")
    if Path(adapter_out).exists() and (Path(adapter_out) / "adapter_config.json").exists():
        log.info(f"  Convergence already trained: {adapter_out}")
        conv_loss = 0.0
    else:
        arm_name = "A" if arm == "a" else "C"
        adapter_out, conv_loss = train_lora(
            base_model_path=merged_dir,
            data_path=str(data_path),
            output_dir=adapter_out,
            cfg=TrainLoraConfig(
                gpu_id=gpu_id,
                epochs=CONVERGENCE_EPOCHS,
                lr=CONVERGENCE_LR,
                lora_r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                batch_size=4,
                grad_accum=4,
                max_length=1024,
                warmup_ratio=0.05,
                seed=seed,
                run_name=f"cp_arm{arm_name}_{source}_s{seed}_v2",
                report_to="wandb",
                gradient_checkpointing=True,
                logging_steps=5,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=5,
                marker_only_loss=False,
                hf_upload=False,
            ),
        )
        log.info(f"  Convergence training complete. Loss: {conv_loss:.4f}")

    # ── Step 4: Eval at each checkpoint ──
    log.info("Step 4: Evaluating checkpoints...")
    checkpoints = find_checkpoint_dirs(Path(adapter_out))

    # ISSUE-1 fix: skip duplicate 100% — check if final adapter == last checkpoint
    last_ckpt_step = checkpoints[-1][0] if checkpoints else 0
    if last_ckpt_step < total_steps:
        checkpoints.append((-1, Path(adapter_out)))

    checkpoint_results = [baseline_result]  # include 0% baseline

    for step, ckpt_dir in checkpoints:
        ckpt_name = ckpt_dir.name if step > 0 else "final"
        pct = min(100, round(step / total_steps * 100)) if step > 0 else 100
        log.info(f"  Evaluating {ckpt_name} (~{pct}%)...")

        # Merge checkpoint adapter with marker-merged base
        tmp_merged = str(arm_dir / "tmp_merged")
        if Path(tmp_merged).exists():
            shutil.rmtree(tmp_merged)
        merge_lora_fixed(merged_dir, str(ckpt_dir), tmp_merged, gpu_id=gpu_id)

        eval_result = eval_checkpoint_subprocess(
            merged_model_path=tmp_merged,
            source=source,
            gpu_id=gpu_id,
            checkpoint_name=f"checkpoint_{pct}pct",
            output_dir=arm_dir,
        )
        eval_result["step"] = step
        eval_result["pct"] = pct
        checkpoint_results.append(eval_result)

        if Path(tmp_merged).exists():
            shutil.rmtree(tmp_merged)
            log.info("  Cleaned tmp_merged")

    # Clean marker-merged model
    if Path(merged_dir).exists():
        shutil.rmtree(merged_dir)
        log.info("Cleaned marker-merged model")

    t_total = (time.time() - t_start) / 60
    summary = {
        "arm": arm.upper(),
        "source": source,
        "seed": seed,
        "convergence_loss": conv_loss,
        "total_steps": total_steps,
        "save_steps": save_steps,
        "checkpoints": checkpoint_results,
        "wall_minutes": round(t_total, 1),
        "version": "v2_rerun",
        "fixes": [
            "BLOCKER-1: adapter_config.json validated (use_rslora=True)",
            "BLOCKER-3: 0% baseline eval added",
            "ISSUE-1: no duplicate 100% checkpoint",
            "ISSUE-2: tokenizer loaded from base model",
            "ISSUE-3: subprocess-isolated eval",
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"ARM {arm.upper()} complete for {source} in {t_total:.1f} min")
    log.info(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun Arms A/C with validated adapter configs")
    parser.add_argument("--arm", choices=["a", "c"], required=True)
    parser.add_argument(
        "--source", required=True, choices=["kindergarten_teacher", "software_engineer"]
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_arm(arm=args.arm, source=args.source, gpu_id=args.gpu, seed=args.seed)
