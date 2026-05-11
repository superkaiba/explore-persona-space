#!/usr/bin/env python3
"""Instruction-column probe for issue #210.

Re-runs the panel eval on all 14 existing #181 adapters with MODIFIED
instruction-family prompts: imperative verbs converted to passive/declarative.
NO new training.

Replaces the 9 instruction-family panel cells (1 matched + 2 paraphrases +
3 family-mates + 3 cross-family with instruction as target) with non-imperative
rewrites. All other 27 cells stay identical.

Compares instruction-column marker rates between original and modified panels.

Output:
    data/i181_non_persona/instruction_probe_panel.json
    eval_results/i181_non_persona/instruction_probe/<run_name>/panel_eval.json
    eval_results/i181_non_persona/instruction_probe/probe_summary.json

Usage:
    uv run python scripts/run_i181_instruction_probe.py --gpu 0 \
        --data-dir data/i181_non_persona
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
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
SEED = 42


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Non-imperative rewrite rules ──────────────────────────────────────────

# Deterministic imperative-to-declarative rewrites for instruction-family prompts.
# Each maps the original prompt text to its non-imperative equivalent.
# Rule: convert imperative verbs to passive/declarative phrasing.

IMPERATIVE_REWRITES: dict[str, str] = {
    # Matched: T_instruction
    (
        "Always reply with at most three sentences. Do not include lists, "
        "headings, or follow-up questions."
    ): (
        "Responses should be limited to three sentences, without lists, "
        "headings, or follow-up questions."
    ),
    # Family-mate: fammate_instruction_1
    ("Use exactly five bullet points in every reply. Each bullet must be one sentence."): (
        "Every reply is expected to contain exactly five bullet points, "
        "with each bullet being a single sentence."
    ),
    # Family-mate: fammate_instruction_2
    (
        "Write every response as a single paragraph with no line breaks. Do not use any formatting."
    ): (
        "All responses should be written as a single paragraph without "
        "line breaks or any formatting."
    ),
    # Family-mate: fammate_instruction_3
    ("Begin every answer with a one-word summary. Then elaborate in at most two sentences."): (
        "Each answer is expected to start with a one-word summary, "
        "followed by at most two sentences of elaboration."
    ),
}


def _rewrite_imperative(text: str) -> str:
    """Apply deterministic imperative-to-declarative rewrite.

    First checks the lookup table for exact matches.
    Falls back to a regex-based rule for paraphrases and bystanders
    that weren't in the lookup table.
    """
    # Check exact match first
    if text in IMPERATIVE_REWRITES:
        return IMPERATIVE_REWRITES[text]

    # Regex-based fallback: convert common imperative patterns
    rewritten = text

    # "Always X" -> "It is expected that X" or "X should always be done"
    rewritten = re.sub(
        r"(?i)^Always\s+",
        "It is expected to always ",
        rewritten,
    )

    # "Do not X" -> "X should not be done" / "X is not expected"
    rewritten = re.sub(
        r"(?i)\bDo not\b",
        "It is not appropriate to",
        rewritten,
    )
    rewritten = re.sub(
        r"(?i)\bDon't\b",
        "It is not appropriate to",
        rewritten,
    )

    # "Never X" -> "X should never occur"
    rewritten = re.sub(
        r"(?i)^Never\s+",
        "It is expected to never ",
        rewritten,
    )

    # "Reply/Respond/Write/Use/Output/Format X" at sentence start
    for verb in ["Reply", "Respond", "Write", "Use", "Output", "Format", "Begin", "Start"]:
        pattern = rf"(?i)^{verb}\s+"
        replacement = f"The response should {verb.lower()} "
        rewritten = re.sub(pattern, replacement, rewritten)
        # Also handle mid-sentence
        pattern_mid = rf"(?i)\.\s+{verb}\s+"
        replacement_mid = f". The response should {verb.lower()} "
        rewritten = re.sub(pattern_mid, replacement_mid, rewritten)

    # "Read X" -> "X should be read"
    rewritten = re.sub(
        r"(?i)^Read\s+",
        "The provided text should be read and ",
        rewritten,
    )

    # "Continue X" -> "The conversation should continue"
    rewritten = re.sub(
        r"(?i)^Continue\s+",
        "The conversation should continue ",
        rewritten,
    )

    # If nothing changed, prepend a declarative framing
    if rewritten == text:
        rewritten = f"The expectation is: {text}"

    return rewritten


def build_instruction_probe_panel(
    original_panel: list[dict],
    data_dir: Path,
) -> list[dict]:
    """Build the modified panel with non-imperative instruction rewrites.

    Returns a new panel list where instruction-family cells have their
    system_prompt replaced with non-imperative equivalents. All other
    cells are unchanged.
    """
    probe_panel_path = data_dir / "instruction_probe_panel.json"

    # Check cache
    if probe_panel_path.exists():
        with open(probe_panel_path) as f:
            data = json.load(f)
        logger.info("Loaded cached instruction probe panel (%d entries)", len(data["panel"]))
        return data["panel"]

    modified_panel = []
    n_modified = 0

    for entry in original_panel:
        new_entry = dict(entry)

        # Only modify instruction-family cells
        if entry.get("family") == "instruction":
            original_prompt = entry["system_prompt"]
            rewritten = _rewrite_imperative(original_prompt)

            if rewritten != original_prompt:
                new_entry["system_prompt"] = rewritten
                new_entry["original_prompt"] = original_prompt
                new_entry["modified"] = True
                n_modified += 1
                logger.info(
                    "Rewrote %s:\n  Original:  %s\n  Rewritten: %s",
                    entry["id"],
                    original_prompt[:80],
                    rewritten[:80],
                )
            else:
                new_entry["modified"] = False
        else:
            new_entry["modified"] = False

        modified_panel.append(new_entry)

    logger.info("Modified %d instruction-family cells", n_modified)

    # Save the probe panel
    panel_doc = {
        "version": "1.0",
        "description": "Instruction probe panel: imperative -> declarative rewrites",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "n_modified": n_modified,
        "panel": modified_panel,
    }

    with open(probe_panel_path, "w") as f:
        json.dump(panel_doc, f, indent=2)
    logger.info("Saved instruction probe panel to %s", probe_panel_path)

    return modified_panel


# ── Panel eval (reused from eval_i181_panel.py pattern) ───────────────────


def eval_single_model(
    model_path: str,
    run_name: str,
    panel: list[dict],
    gpu_id: int,
    num_completions: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 256,
) -> dict:
    """Run panel eval on a single merged model with the probe panel."""
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

    # Reassemble by cell
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
            "modified": entry.get("modified", False),
            "marker_rate": cell_result.get("rate", 0.0),
            "marker_found": cell_result.get("found", 0),
            "marker_total": cell_result.get("total", 0),
        }

    return {
        "run_name": run_name,
        "model_path": model_path,
        "panel_type": "instruction_probe",
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


def compute_probe_summary(
    probe_results: dict[str, dict],
    original_results_dir: Path,
) -> dict:
    """Compare instruction-column rates between original and modified panels.

    Loads existing panel_eval.json from each run, extracts instruction-family
    marker rates, and compares against the probe panel results.
    """
    summary = {
        "computed_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "per_model_comparison": {},
    }

    all_original_instruction_rates: list[float] = []
    all_modified_instruction_rates: list[float] = []

    for run_name, probe_eval in probe_results.items():
        # Load original panel eval
        original_path = original_results_dir / run_name / "panel_eval.json"
        original_eval = None
        if original_path.exists():
            with open(original_path) as f:
                original_eval = json.load(f)

        # Extract instruction-family rates from probe
        probe_instruction_rates = []
        probe_results_dict = probe_eval.get("results", {})
        for _cell_id, cell_data in probe_results_dict.items():
            if cell_data.get("family") == "instruction":
                probe_instruction_rates.append(cell_data.get("marker_rate", 0.0))

        # Extract instruction-family rates from original
        original_instruction_rates = []
        if original_eval:
            orig_results = original_eval.get("results", {})
            for _cell_id, cell_data in orig_results.items():
                if cell_data.get("family") == "instruction":
                    original_instruction_rates.append(cell_data.get("marker_rate", 0.0))

        orig_mean = (
            sum(original_instruction_rates) / len(original_instruction_rates)
            if original_instruction_rates
            else None
        )
        probe_mean = (
            sum(probe_instruction_rates) / len(probe_instruction_rates)
            if probe_instruction_rates
            else 0.0
        )
        delta = probe_mean - orig_mean if orig_mean is not None else None

        comparison = {
            "original_instruction_rate": orig_mean,
            "modified_instruction_rate": probe_mean,
            "delta": delta,
            "n_original_cells": len(original_instruction_rates),
            "n_modified_cells": len(probe_instruction_rates),
            "original_per_cell": original_instruction_rates,
            "modified_per_cell": probe_instruction_rates,
        }
        summary["per_model_comparison"][run_name] = comparison

        if orig_mean is not None:
            all_original_instruction_rates.append(orig_mean)
        all_modified_instruction_rates.append(probe_mean)

    # Aggregate
    summary["original_instruction_rate"] = (
        sum(all_original_instruction_rates) / len(all_original_instruction_rates)
        if all_original_instruction_rates
        else None
    )
    summary["modified_instruction_rate"] = (
        sum(all_modified_instruction_rates) / len(all_modified_instruction_rates)
        if all_modified_instruction_rates
        else 0.0
    )
    if summary["original_instruction_rate"] is not None:
        summary["delta"] = (
            summary["modified_instruction_rate"] - summary["original_instruction_rate"]
        )
    else:
        summary["delta"] = None

    summary["n_models"] = len(probe_results)

    return summary


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Instruction-column probe for issue #210")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/i181_non_persona",
        help="Directory containing training data and eval_panel.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/i181_non_persona/instruction_probe",
        help="Output directory for probe results",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="*",
        help="Specific run names to evaluate (default: all runs with merged models)",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original_results_dir = PROJECT_ROOT / "eval_results" / "i181_non_persona"

    # Load original panel
    panel_path = data_dir / "eval_panel.json"
    if not panel_path.exists():
        logger.error("Eval panel not found: %s", panel_path)
        raise FileNotFoundError(panel_path)
    with open(panel_path) as f:
        panel_data = json.load(f)
    original_panel = panel_data["panel"]
    logger.info("Loaded original eval panel with %d entries", len(original_panel))

    # Build modified panel
    probe_panel = build_instruction_probe_panel(original_panel, data_dir)

    # Find all run directories with merged models
    if args.runs:
        run_names = args.runs
    else:
        run_names = []
        if original_results_dir.exists():
            for run_dir in sorted(original_results_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                merged_path = run_dir / "train_merged"
                if merged_path.exists():
                    run_names.append(run_dir.name)

    logger.info("Found %d runs to evaluate", len(run_names))

    all_probe_results: dict[str, dict] = {}
    t_start = time.time()

    for run_name in run_names:
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        probe_eval_path = run_output_dir / "panel_eval.json"

        # Check if probe eval already exists
        if probe_eval_path.exists():
            logger.info("Probe eval already exists for %s, loading", run_name)
            with open(probe_eval_path) as f:
                all_probe_results[run_name] = json.load(f)
            continue

        merged_path = original_results_dir / run_name / "train_merged"
        if not merged_path.exists():
            logger.warning("Merged model not found for %s, skipping", run_name)
            continue

        logger.info("=" * 60)
        logger.info("Evaluating: %s", run_name)
        logger.info("=" * 60)

        probe_eval = eval_single_model(
            model_path=str(merged_path),
            run_name=run_name,
            panel=probe_panel,
            gpu_id=args.gpu,
        )

        from explore_persona_space.utils import save_json_atomic

        save_json_atomic(probe_eval_path, probe_eval)
        logger.info("Saved probe eval to %s", probe_eval_path)

        all_probe_results[run_name] = probe_eval

    # Compute and save summary
    if all_probe_results:
        summary = compute_probe_summary(all_probe_results, original_results_dir)
        summary_path = output_dir / "probe_summary.json"

        from explore_persona_space.utils import save_json_atomic

        save_json_atomic(summary_path, summary)
        logger.info("Saved probe summary to %s", summary_path)

        # Print summary table
        logger.info("=" * 70)
        logger.info("INSTRUCTION PROBE SUMMARY")
        logger.info("=" * 70)
        logger.info(
            "%-25s  %10s  %10s  %8s",
            "Run",
            "Original",
            "Modified",
            "Delta",
        )
        logger.info("-" * 60)
        for run_name, comparison in summary["per_model_comparison"].items():
            orig = comparison["original_instruction_rate"]
            mod = comparison["modified_instruction_rate"]
            delta = comparison["delta"]
            logger.info(
                "%-25s  %9.1f%%  %9.1f%%  %+7.1f%%",
                run_name,
                (orig or 0.0) * 100,
                mod * 100,
                (delta or 0.0) * 100,
            )
        logger.info("-" * 60)
        logger.info(
            "%-25s  %9.1f%%  %9.1f%%  %+7.1f%%",
            "AGGREGATE",
            (summary["original_instruction_rate"] or 0.0) * 100,
            summary["modified_instruction_rate"] * 100,
            (summary["delta"] or 0.0) * 100,
        )
        logger.info("=" * 70)

    t_total = time.time() - t_start
    logger.info("Total time: %.1f min", t_total / 60)


if __name__ == "__main__":
    main()
