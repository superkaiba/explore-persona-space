#!/usr/bin/env python3
"""Recipe titration for issue #208: sweep 4 LoRA recipe points on T_task seed=42.

Trains 4 recipe variants on the same T_task.jsonl data (194 examples),
merges each, evaluates with vLLM on the full 36-prompt panel, and
outputs a titration summary.

Recipe points:
  1. r=16, alpha=32, lr=1e-5, epochs=3  (original #181 default — low end)
  2. r=16, alpha=32, lr=1e-4, epochs=3  (intermediate: higher lr)
  3. r=32, alpha=64, lr=1e-5, epochs=5  (intermediate: higher capacity + epochs)
  4. r=32, alpha=64, lr=1e-4, epochs=5  (#181 production recipe — reuse if available)

Usage:
    uv run python scripts/run_i181_recipe_titration.py --gpu 0 --data-dir data/i181_non_persona
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Recipe definitions ─────────────────────────────────────────────────────


@dataclass
class RecipePoint:
    """A single LoRA recipe configuration to titrate."""

    name: str
    lora_r: int
    lora_alpha: int
    lr: float
    epochs: int


RECIPE_POINTS = [
    RecipePoint("recipe_1_r16_lr1e5_ep3", lora_r=16, lora_alpha=32, lr=1e-5, epochs=3),
    RecipePoint("recipe_2_r16_lr1e4_ep3", lora_r=16, lora_alpha=32, lr=1e-4, epochs=3),
    RecipePoint("recipe_3_r32_lr1e5_ep5", lora_r=32, lora_alpha=64, lr=1e-5, epochs=5),
    RecipePoint("recipe_4_r32_lr1e4_ep5", lora_r=32, lora_alpha=64, lr=1e-4, epochs=5),
]

SEED = 42
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRIGGER_NAME = "T_task"


# ── Training ───────────────────────────────────────────────────────────────


def train_recipe(
    recipe: RecipePoint,
    data_path: str,
    output_dir: Path,
    gpu_id: int,
) -> tuple[str, str, float]:
    """Train a LoRA adapter, merge it, return (adapter_dir, merged_dir, loss)."""
    from explore_persona_space.train.sft import merge_lora, train_lora

    adapter_dir = str(output_dir / f"{recipe.name}_adapter")
    merged_dir = str(output_dir / f"{recipe.name}_merged")

    # Skip if already trained + merged
    if Path(merged_dir).exists() and (Path(merged_dir) / "config.json").exists():
        logger.info("Merged model already exists at %s, skipping training", merged_dir)
        return adapter_dir, merged_dir, 0.0

    logger.info(
        "Training %s: r=%d, alpha=%d, lr=%s, epochs=%d",
        recipe.name,
        recipe.lora_r,
        recipe.lora_alpha,
        recipe.lr,
        recipe.epochs,
    )

    adapter_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=data_path,
        output_dir=adapter_dir,
        gpu_id=gpu_id,
        seed=SEED,
        run_name=f"titration_{recipe.name}",
        report_to="none",
        hf_upload=False,
        lora_r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lr=recipe.lr,
        epochs=recipe.epochs,
        batch_size=4,
        grad_accum=4,
        max_length=1024,
        warmup_ratio=0.05,
    )

    logger.info("Training complete for %s, loss=%.4f. Merging...", recipe.name, loss)

    merge_lora(BASE_MODEL, adapter_path, merged_dir, gpu_id=gpu_id)
    logger.info("Merged to %s", merged_dir)

    return adapter_dir, merged_dir, loss


# ── Panel eval ─────────────────────────────────────────────────────────────


def eval_panel(
    model_path: str,
    recipe_name: str,
    panel: list[dict],
    gpu_id: int,
    num_completions: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 256,
) -> dict:
    """Run full 36-prompt panel eval on a single merged model.

    Reuses the exact same vLLM-based eval logic as eval_i181_panel.py.
    """
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.eval.trait_scorers import evaluate_markers
    from explore_persona_space.personas import EVAL_QUESTIONS

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info("Loading tokenizer from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build ALL prompts upfront
    prompt_texts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []

    for entry in panel:
        panel_id = entry["id"]
        system_prompt = entry["system_prompt"]
        for question in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((panel_id, question))

    total_prompts = len(prompt_texts)
    logger.info(
        "Built %d prompts (%d cells x %d questions), n=%d each",
        total_prompts,
        len(panel),
        len(EVAL_QUESTIONS),
        num_completions,
    )

    # Load vLLM
    gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    logger.info("Loading vLLM engine (gpu_mem=%.2f)...", gpu_mem)
    t_load = time.time()

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=2048,
        max_num_seqs=64,
        seed=SEED,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t_load)

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    logger.info("Generating completions...")
    t_gen = time.time()

    try:
        outputs = llm.generate(prompt_texts, sampling_params)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Generation complete in %.1fs", time.time() - t_gen)

    # Reassemble completions by cell
    completions_by_cell: dict[str, dict[str, list[str]]] = {}
    for output, (panel_id, question) in zip(outputs, prompt_keys, strict=True):
        if panel_id not in completions_by_cell:
            completions_by_cell[panel_id] = {}
        completions_by_cell[panel_id][question] = [o.text for o in output.outputs]

    # Score markers per cell
    results: dict[str, dict] = {}
    for entry in panel:
        panel_id = entry["id"]
        cell_completions = completions_by_cell.get(panel_id, {})
        marker_result = evaluate_markers({panel_id: cell_completions})
        cell_result = marker_result.get(panel_id, {})

        results[panel_id] = {
            "panel_id": panel_id,
            "family": entry.get("family"),
            "bucket": entry.get("bucket"),
            "system_prompt": entry["system_prompt"],
            "marker_rate": cell_result.get("rate", 0.0),
            "marker_found": cell_result.get("found", 0),
            "marker_total": cell_result.get("total", 0),
        }

    return {
        "recipe": recipe_name,
        "model_path": model_path,
        "n_panel_cells": len(panel),
        "n_completions_per_prompt": num_completions,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "vllm_seed": SEED,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "generation_time_s": time.time() - t_gen,
        "results": results,
    }


# ── Summary computation ───────────────────────────────────────────────────


def compute_summary(all_results: dict[str, dict]) -> dict:
    """Compute titration summary across all recipe points.

    For each recipe: matched_rate, bystander_rate, ratio, per_family_rates.
    """
    summary = {
        "computed_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "trigger": TRIGGER_NAME,
        "seed": SEED,
        "recipes": {},
    }

    for recipe_name, panel_eval in all_results.items():
        results = panel_eval.get("results", {})

        matched_rates = []
        bystander_rates = []
        per_family: dict[str, list[float]] = {}

        for _cell_id, cell_data in results.items():
            bucket = cell_data.get("bucket", "")
            family = cell_data.get("family")
            rate = cell_data.get("marker_rate", 0.0)

            if bucket == "matched":
                matched_rates.append(rate)
            elif bucket in ("cross_family_bystander", "control_empty", "control_default"):
                bystander_rates.append(rate)

            if family:
                per_family.setdefault(family, []).append(rate)

        matched_rate = sum(matched_rates) / len(matched_rates) if matched_rates else 0.0
        bystander_rate = sum(bystander_rates) / len(bystander_rates) if bystander_rates else 0.0
        ratio = matched_rate / bystander_rate if bystander_rate > 0 else float("inf")

        per_family_rates = {fam: sum(rates) / len(rates) for fam, rates in per_family.items()}

        # Get recipe hyperparameters
        recipe_point = next((r for r in RECIPE_POINTS if r.name == recipe_name), None)
        recipe_info = {}
        if recipe_point:
            recipe_info = {
                "lora_r": recipe_point.lora_r,
                "lora_alpha": recipe_point.lora_alpha,
                "lr": recipe_point.lr,
                "epochs": recipe_point.epochs,
            }

        summary["recipes"][recipe_name] = {
            **recipe_info,
            "matched_rate": matched_rate,
            "bystander_rate": bystander_rate,
            "ratio": ratio,
            "per_family_rates": per_family_rates,
        }

    return summary


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Recipe titration for issue #208: sweep 4 LoRA recipe points on T_task"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/i181_non_persona",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/i181_non_persona/titration",
        help="Output directory for titration results",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run eval on existing merged models",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = str(data_dir / f"{TRIGGER_NAME}.jsonl")
    if not Path(data_path).exists():
        logger.error("Training data not found: %s", data_path)
        raise FileNotFoundError(data_path)

    # Load eval panel
    panel_path = data_dir / "eval_panel.json"
    if not panel_path.exists():
        logger.error("Eval panel not found: %s", panel_path)
        raise FileNotFoundError(panel_path)
    with open(panel_path) as f:
        panel_data = json.load(f)
    panel = panel_data["panel"]
    logger.info("Loaded eval panel with %d entries", len(panel))

    all_results: dict[str, dict] = {}
    t_start = time.time()

    for recipe in RECIPE_POINTS:
        recipe_dir = output_dir / recipe.name
        recipe_dir.mkdir(parents=True, exist_ok=True)

        panel_eval_path = recipe_dir / "panel_eval.json"

        # Check if panel eval already exists
        if panel_eval_path.exists():
            logger.info("Panel eval already exists for %s, loading", recipe.name)
            with open(panel_eval_path) as f:
                all_results[recipe.name] = json.load(f)
            continue

        # Train if needed
        if not args.skip_training:
            _adapter_dir, merged_dir, loss = train_recipe(recipe, data_path, output_dir, args.gpu)
            logger.info("Recipe %s trained (loss=%.4f)", recipe.name, loss)
        else:
            merged_dir = str(output_dir / f"{recipe.name}_merged")
            if not Path(merged_dir).exists():
                logger.warning("Merged model not found for %s, skipping", recipe.name)
                continue

        # Run panel eval
        logger.info("Running panel eval for %s...", recipe.name)
        panel_eval = eval_panel(
            model_path=merged_dir,
            recipe_name=recipe.name,
            panel=panel,
            gpu_id=args.gpu,
        )

        from explore_persona_space.utils import save_json_atomic

        save_json_atomic(panel_eval_path, panel_eval)
        logger.info("Saved panel eval to %s", panel_eval_path)

        all_results[recipe.name] = panel_eval

    # Compute and save summary
    if all_results:
        summary = compute_summary(all_results)
        summary_path = output_dir / "titration_summary.json"

        from explore_persona_space.utils import save_json_atomic

        save_json_atomic(summary_path, summary)
        logger.info("Saved titration summary to %s", summary_path)

        # Print summary table
        logger.info("=" * 70)
        logger.info("TITRATION SUMMARY")
        logger.info("=" * 70)
        logger.info(
            "%-35s  %8s  %8s  %6s",
            "Recipe",
            "Matched",
            "Bystander",
            "Ratio",
        )
        logger.info("-" * 70)
        for rname, rdata in summary["recipes"].items():
            logger.info(
                "%-35s  %7.1f%%  %8.1f%%  %5.1fx",
                rname,
                rdata["matched_rate"] * 100,
                rdata["bystander_rate"] * 100,
                rdata["ratio"],
            )
        logger.info("=" * 70)

    t_total = time.time() - t_start
    logger.info("Total time: %.1f min", t_total / 60)


if __name__ == "__main__":
    main()
