#!/usr/bin/env python3
"""Per-model panel evaluation for issue #181 non-persona trigger leakage.

For each merged adapter, loads vLLM ONCE and batches all 36 panel prompts
x 20 EVAL_QUESTIONS x 10 completions into a single LLM.generate() call.
Scores with evaluate_markers and writes panel_eval.json.

Critical design: vLLM engine is loaded ONCE per model and reused for all
36 panel cells. This amortizes the ~2min cold-start across all cells.

Usage:
    # Eval a single model
    uv run python scripts/eval_i181_panel.py \
        --model-path eval_results/i181_non_persona/T_format_seed42/train_merged \
        --run-name T_format_seed42 \
        --config configs/leakage/i181_non_persona_triggers.yaml \
        --gpu 0

    # Eval all models in the output dir
    uv run python scripts/eval_i181_panel.py \
        --config configs/leakage/i181_non_persona_triggers.yaml \
        --gpu 0 --all
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import time
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


def load_panel() -> list[dict]:
    """Load the eval panel from data/i181_non_persona/eval_panel.json."""
    panel_path = PROJECT_ROOT / "data" / "i181_non_persona" / "eval_panel.json"
    with open(panel_path) as f:
        data = json.load(f)
    return data["panel"]


def classify_marker_position(completion: str, marker: str = "[ZLT]") -> str:
    """Classify where the marker appears in a completion.

    Returns one of:
        "suffix"      -- marker appears after the main text (canonical position)
        "inside_json" -- marker appears inside a JSON string value
        "inside_text" -- marker appears inside the main text body
        "absent"      -- marker not found
    """
    marker_lower = marker.lower()
    comp_lower = completion.lower()

    if marker_lower not in comp_lower:
        return "absent"

    # Check if it's a suffix (last occurrence is near the end)
    last_pos = comp_lower.rfind(marker_lower)
    remaining = completion[last_pos + len(marker) :].strip()

    if len(remaining) < 5:
        return "suffix"

    # Check if inside JSON
    # Look for the marker inside a JSON string context
    try:
        json.loads(completion)
        # If the whole completion is valid JSON, marker is inside JSON
        return "inside_json"
    except (json.JSONDecodeError, ValueError):
        pass

    # Check for JSON-like context around the marker
    idx = completion.lower().find(marker_lower)
    context_before = completion[max(0, idx - 20) : idx]
    if '"' in context_before or "'" in context_before:
        # Likely inside a string value
        return "inside_json"

    return "inside_text"


def eval_single_model(
    model_path: str,
    run_name: str,
    panel: list[dict],
    gpu_id: int,
    num_completions: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 256,
    seed: int = 42,
) -> dict:
    """Run panel eval on a single merged model.

    Loads vLLM ONCE and batches all panel prompts x EVAL_QUESTIONS x completions.

    Returns panel eval results dict.
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

    # Build ALL prompts upfront for all 36 panel cells x 20 questions
    prompt_texts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []  # (panel_id, question)

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
    total_completions = total_prompts * num_completions
    logger.info(
        "Built %d prompts (%d panel cells x %d questions). Total completions: %d (n=%d per prompt)",
        total_prompts,
        len(panel),
        len(EVAL_QUESTIONS),
        total_completions,
        num_completions,
    )

    # Load vLLM engine ONCE
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
        seed=seed,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t_load)

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    # Generate ALL completions in one batch
    logger.info("Generating %d completions in one batch...", total_completions)
    t_gen = time.time()

    try:
        outputs = llm.generate(prompt_texts, sampling_params)
    finally:
        # Free GPU memory
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Generation complete in %.1fs", time.time() - t_gen)

    # Reassemble into {panel_id: {question: [completions]}} structure
    completions_by_cell: dict[str, dict[str, list[str]]] = {}
    for output, (panel_id, question) in zip(outputs, prompt_keys, strict=True):
        if panel_id not in completions_by_cell:
            completions_by_cell[panel_id] = {}
        completions_by_cell[panel_id][question] = [o.text for o in output.outputs]

    # Score markers per cell
    results: dict[str, dict] = {}
    marker_position_counts: dict[str, dict[str, int]] = {}

    for entry in panel:
        panel_id = entry["id"]
        cell_completions = completions_by_cell.get(panel_id, {})

        # Use evaluate_markers with the cell as a single "persona"
        marker_result = evaluate_markers({panel_id: cell_completions})
        cell_result = marker_result.get(panel_id, {})

        # Classify marker positions for this cell
        positions = {"suffix": 0, "inside_json": 0, "inside_text": 0, "absent": 0}
        for q_comps in cell_completions.values():
            for comp in q_comps:
                pos = classify_marker_position(comp)
                positions[pos] += 1
        marker_position_counts[panel_id] = positions

        results[panel_id] = {
            "panel_id": panel_id,
            "family": entry.get("family"),
            "bucket": entry.get("bucket"),
            "system_prompt": entry["system_prompt"],
            "marker_rate": cell_result.get("rate", 0.0),
            "marker_found": cell_result.get("found", 0),
            "marker_total": cell_result.get("total", 0),
            "marker_position": positions,
            "per_question": cell_result.get("per_question", {}),
        }

    # Build output document
    output = {
        "run_name": run_name,
        "model_path": model_path,
        "n_panel_cells": len(panel),
        "n_questions": len(EVAL_QUESTIONS),
        "n_completions_per_prompt": num_completions,
        "total_completions": total_completions,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "vllm_seed": seed,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "generation_time_s": time.time() - t_gen,
        "results": results,
        "marker_position_summary": marker_position_counts,
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="Panel eval for issue #181")
    parser.add_argument("--model-path", type=str, help="Path to merged model directory")
    parser.add_argument("--run-name", type=str, help="Run name (e.g., T_format_seed42)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/leakage/i181_non_persona_triggers.yaml",
        help="Path to sweep config YAML",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--all", action="store_true", help="Eval all models in output dir")
    parser.add_argument("--num-completions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    panel = load_panel()
    logger.info("Loaded eval panel with %d entries", len(panel))

    if args.all:
        # Find all run directories with train_merged
        from explore_persona_space.leakage.config import load_sweep

        sweep = load_sweep(args.config)
        output_dir = sweep.resolve_output_dir(PROJECT_ROOT)

        run_dirs = sorted(output_dir.iterdir()) if output_dir.exists() else []
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            merged_path = run_dir / "train_merged"
            panel_eval_path = run_dir / "panel_eval.json"
            if not merged_path.exists():
                continue
            if panel_eval_path.exists():
                logger.info("Skipping %s (panel_eval.json exists)", run_dir.name)
                continue

            logger.info("=" * 60)
            logger.info("Evaluating: %s", run_dir.name)
            logger.info("=" * 60)

            result = eval_single_model(
                model_path=str(merged_path),
                run_name=run_dir.name,
                panel=panel,
                gpu_id=args.gpu,
                num_completions=args.num_completions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            from explore_persona_space.utils import save_json_atomic

            save_json_atomic(panel_eval_path, result)
            logger.info("Saved panel eval to %s", panel_eval_path)

    elif args.model_path and args.run_name:
        result = eval_single_model(
            model_path=args.model_path,
            run_name=args.run_name,
            panel=panel,
            gpu_id=args.gpu,
            num_completions=args.num_completions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        from explore_persona_space.leakage.config import load_sweep
        from explore_persona_space.utils import save_json_atomic

        sweep = load_sweep(args.config)
        output_dir = sweep.resolve_output_dir(PROJECT_ROOT)
        run_dir = output_dir / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        panel_eval_path = run_dir / "panel_eval.json"
        save_json_atomic(panel_eval_path, result)
        logger.info("Saved panel eval to %s", panel_eval_path)
    else:
        parser.error("Either --all or both --model-path and --run-name are required")


if __name__ == "__main__":
    main()
