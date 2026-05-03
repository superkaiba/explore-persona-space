#!/usr/bin/env python3
"""Prompt-vs-content dissociation for issue #209.

Runs a #173-style prefix-completion dissociation on 4 existing #181 adapters
(one per trigger family, seed=42). NO new training.

For each model, 4 conditions cross prompt x answer-prefix:
  A (matched):        source system prompt + source answer prefix
  B (other prompt):   different trigger's system prompt + source answer prefix
  C (other answer):   source system prompt + different trigger's answer prefix
  D (fully mismatched): different trigger's system prompt + different answer prefix

For each condition, we build messages with system + user + partial assistant
response (the answer text up to but not including "\\n\\n[ZLT]"), then generate
~30 tokens of continuation via vLLM and check if [ZLT] appears.

N=100 continuations per condition per model (20 questions x 5 samples each).

Output:
    eval_results/i181_non_persona/dissociation/dissociation_results.json

Usage:
    uv run python scripts/run_i181_dissociation.py --gpu 0 --data-dir data/i181_non_persona
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

# The 4 main trigger families to test (seed=42 only)
TARGET_FAMILIES = ["T_task", "T_instruction", "T_context", "T_format"]
SEED = 42
MARKER = "[ZLT]"
MARKER_LOWER = MARKER.lower()
N_COMPLETIONS = 5  # per question per condition
MAX_CONTINUATION_TOKENS = 30


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_triggers(data_dir: Path) -> dict[str, str]:
    """Load trigger system prompts from triggers.json."""
    triggers_path = data_dir / "triggers.json"
    with open(triggers_path) as f:
        data = json.load(f)
    return data["triggers"]


def load_qa_pool(data_dir: Path) -> list[dict[str, str]]:
    """Load QA pool (training questions + answers)."""
    qa_path = data_dir / "qa_pool.jsonl"
    pool = []
    with open(qa_path) as f:
        for line in f:
            pool.append(json.loads(line))
    return pool


def strip_marker_suffix(answer: str) -> str:
    """Strip the trailing \\n\\n[ZLT] from a training answer to get the answer prefix.

    Returns the answer text up to but not including the marker suffix.
    """
    # Look for the canonical suffix pattern: \n\n[ZLT]
    suffix = f"\n\n{MARKER}"
    if answer.endswith(suffix):
        return answer[: -len(suffix)]
    # Also handle variations in whitespace
    suffix_lower = f"\n\n{MARKER_LOWER}"
    lower = answer.lower()
    if lower.endswith(suffix_lower):
        return answer[: -len(suffix)]
    # If no marker suffix found, return the full answer
    return answer


def find_merged_model(data_dir: Path, trigger_name: str) -> str | None:
    """Find the merged model path for a trigger at seed=42.

    Looks in eval_results/i181_non_persona/{trigger_name}_seed42/train_merged.
    """
    # The run_name convention from LeakageRunner is {condition_name}_seed{N}
    run_name = f"{trigger_name}_seed{SEED}"
    merged_path = PROJECT_ROOT / "eval_results" / "i181_non_persona" / run_name / "train_merged"
    if merged_path.exists():
        return str(merged_path)
    return None


def build_dissociation_prompts(
    tokenizer,
    source_prompt: str,
    other_prompt: str,
    questions: list[str],
    matched_answers: list[str],
    mismatched_answers: list[str],
) -> dict[str, list[str]]:
    """Build prefix-completion prompts for the 4 dissociation conditions.

    The 2x2 crosses system prompt (source vs other) with answer content
    (matched answer for Q_i vs mismatched answer from a different Q_j).

    - A (matched):        source prompt + Q_i + answer_prefix(Q_i)
    - B (other prompt):   other prompt  + Q_i + answer_prefix(Q_i)
    - C (other answer):   source prompt + Q_i + answer_prefix(Q_j)
    - D (fully mismatched): other prompt + Q_i + answer_prefix(Q_j)

    Returns:
        {condition_name: [prompt_text_1, ...]}
    """
    conditions = {}

    def _build_prompt(sys_prompt: str, question: str, answer_prefix: str) -> str:
        """Build a chat-template prompt with system + user + partial assistant."""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        # Apply chat template up to the assistant turn start
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Append the answer prefix as the start of the assistant's response
        text += answer_prefix
        return text

    n_questions = min(len(questions), len(matched_answers), len(mismatched_answers))

    # A: matched (source prompt + matched answer prefix for this question)
    conditions["A_matched"] = [
        _build_prompt(source_prompt, questions[i], strip_marker_suffix(matched_answers[i]))
        for i in range(n_questions)
    ]

    # B: other prompt + matched answer prefix (same content, different prompt)
    conditions["B_other_prompt"] = [
        _build_prompt(other_prompt, questions[i], strip_marker_suffix(matched_answers[i]))
        for i in range(n_questions)
    ]

    # C: source prompt + mismatched answer prefix (same prompt, different content)
    conditions["C_other_answer"] = [
        _build_prompt(source_prompt, questions[i], strip_marker_suffix(mismatched_answers[i]))
        for i in range(n_questions)
    ]

    # D: fully mismatched (different prompt AND different content)
    conditions["D_mismatched"] = [
        _build_prompt(other_prompt, questions[i], strip_marker_suffix(mismatched_answers[i]))
        for i in range(n_questions)
    ]

    return conditions


def run_dissociation_for_model(
    model_path: str,
    source_trigger: str,
    triggers: dict[str, str],
    qa_pool: list[dict[str, str]],
    gpu_id: int,
) -> dict:
    """Run dissociation eval for a single model.

    For each of the 4 conditions, generates N_COMPLETIONS continuations
    per question and checks for marker presence.
    """
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info("Loading tokenizer from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Pick the "other" trigger — use a fixed pairing for reproducibility
    other_trigger = _pick_other_trigger(source_trigger)
    source_prompt = triggers[source_trigger]
    other_prompt = triggers[other_trigger]

    logger.info(
        "Source: %s, Other: %s",
        source_trigger,
        other_trigger,
    )

    # Use training questions from qa_pool (not EVAL_QUESTIONS) because
    # the prefix-completion paradigm needs question+answer pairs that
    # are coherent — the answer prefix was generated for the training question.
    # We sample 20 questions (matching the 20 EVAL_QUESTIONS count) with a
    # fixed seed for reproducibility.
    import random as _random

    rng = _random.Random(SEED)
    n_questions = min(20, len(qa_pool))
    sampled_indices = sorted(rng.sample(range(len(qa_pool)), n_questions))
    questions = [qa_pool[i]["question"] for i in sampled_indices]

    # Matched answers: the answer that was generated for this question
    # (with the marker suffix, which strip_marker_suffix will remove)
    matched_answers = [f"{qa_pool[i]['answer']}\n\n{MARKER}" for i in sampled_indices]

    # Mismatched answers: answers from a DIFFERENT question (shifted by half
    # the pool size to maximize content distance). This creates the content
    # dissociation — the model sees an answer prefix that doesn't match the
    # user question, testing whether marker emission depends on content.
    shift = len(qa_pool) // 2
    mismatched_indices = [(i + shift) % len(qa_pool) for i in sampled_indices]
    mismatched_answers = [f"{qa_pool[j]['answer']}\n\n{MARKER}" for j in mismatched_indices]

    # Build prompts for all 4 conditions
    conditions = build_dissociation_prompts(
        tokenizer,
        source_prompt,
        other_prompt,
        questions,
        matched_answers,
        mismatched_answers,
    )

    # Flatten all prompts for batched generation
    all_prompts: list[str] = []
    prompt_keys: list[tuple[str, int]] = []  # (condition_name, question_idx)

    for cond_name, cond_prompts in conditions.items():
        for q_idx, prompt in enumerate(cond_prompts):
            all_prompts.append(prompt)
            prompt_keys.append((cond_name, q_idx))

    logger.info(
        "Built %d prompts across 4 conditions, n=%d each = %d total completions",
        len(all_prompts),
        N_COMPLETIONS,
        len(all_prompts) * N_COMPLETIONS,
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
        max_model_len=4096,  # Higher to accommodate prefix + continuation
        max_num_seqs=64,
        seed=SEED,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t_load)

    sampling_params = SamplingParams(
        n=N_COMPLETIONS,
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_CONTINUATION_TOKENS,
    )

    logger.info("Generating continuations...")
    t_gen = time.time()

    try:
        outputs = llm.generate(all_prompts, sampling_params)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Generation complete in %.1fs", time.time() - t_gen)

    # Tally marker presence per condition
    condition_results: dict[str, dict] = {}
    for cond_name in conditions:
        condition_results[cond_name] = {
            "marker_found": 0,
            "marker_total": 0,
            "per_question": {},
        }

    for output, (cond_name, q_idx) in zip(outputs, prompt_keys, strict=True):
        continuations = [o.text for o in output.outputs]
        found = sum(1 for c in continuations if MARKER_LOWER in c.lower())
        total = len(continuations)

        cond_result = condition_results[cond_name]
        cond_result["marker_found"] += found
        cond_result["marker_total"] += total
        cond_result["per_question"][str(q_idx)] = {
            "found": found,
            "total": total,
            "rate": found / total if total > 0 else 0.0,
            "sample_continuations": continuations[:2],  # Save 2 examples for inspection
        }

    # Compute rates
    for _cond_name, cond_result in condition_results.items():
        total = cond_result["marker_total"]
        found = cond_result["marker_found"]
        cond_result["marker_rate"] = found / total if total > 0 else 0.0

    return {
        "source_trigger": source_trigger,
        "other_trigger": other_trigger,
        "source_prompt": source_prompt,
        "other_prompt": other_prompt,
        "n_questions": len(conditions.get("A_matched", [])),
        "n_completions_per_question": N_COMPLETIONS,
        "max_continuation_tokens": MAX_CONTINUATION_TOKENS,
        "conditions": condition_results,
        "generation_time_s": time.time() - t_gen,
    }


def _pick_other_trigger(source: str) -> str:
    """Pick a deterministic 'other' trigger for dissociation.

    Uses a fixed rotation so each source gets a unique, maximally-distant partner.
    """
    rotation = {
        "T_task": "T_format",
        "T_instruction": "T_context",
        "T_context": "T_instruction",
        "T_format": "T_task",
    }
    return rotation.get(source, "T_task")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prompt-vs-content dissociation for issue #209")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/i181_non_persona",
        help="Directory containing training data and triggers.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/i181_non_persona/dissociation",
        help="Output directory",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    triggers = load_triggers(data_dir)
    qa_pool = load_qa_pool(data_dir)
    logger.info("Loaded %d triggers, %d QA pairs", len(triggers), len(qa_pool))

    all_model_results: dict[str, dict] = {}
    t_start = time.time()

    for trigger_name in TARGET_FAMILIES:
        logger.info("=" * 60)
        logger.info("Processing %s (seed=%d)", trigger_name, SEED)
        logger.info("=" * 60)

        # Find merged model
        model_path = find_merged_model(data_dir, trigger_name)
        if model_path is None:
            logger.warning("Merged model not found for %s_seed%d, skipping", trigger_name, SEED)
            continue

        logger.info("Using merged model at %s", model_path)

        result = run_dissociation_for_model(
            model_path=model_path,
            source_trigger=trigger_name,
            triggers=triggers,
            qa_pool=qa_pool,
            gpu_id=args.gpu,
        )
        all_model_results[trigger_name] = result

        # Log per-condition rates
        for cond_name, cond_data in result["conditions"].items():
            logger.info(
                "  %s: marker_rate=%.1f%% (%d/%d)",
                cond_name,
                cond_data["marker_rate"] * 100,
                cond_data["marker_found"],
                cond_data["marker_total"],
            )

    # Build output document
    output = {
        "experiment": "i181_dissociation",
        "issue": 209,
        "description": (
            "Prompt-vs-content dissociation: 2x2 crossing system prompt "
            "x answer prefix for 4 trigger families"
        ),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "seed": SEED,
        "n_completions_per_question": N_COMPLETIONS,
        "max_continuation_tokens": MAX_CONTINUATION_TOKENS,
        "total_time_s": time.time() - t_start,
        "models": all_model_results,
        "summary": _compute_summary(all_model_results),
    }

    # Save results
    from explore_persona_space.utils import save_json_atomic

    results_path = output_dir / "dissociation_results.json"
    save_json_atomic(results_path, output)
    logger.info("Saved dissociation results to %s", results_path)

    # Print summary
    logger.info("=" * 70)
    logger.info("DISSOCIATION SUMMARY")
    logger.info("=" * 70)
    logger.info("%-15s  %8s  %8s  %8s  %8s", "Model", "A(match)", "B(oP)", "C(oA)", "D(mis)")
    logger.info("-" * 55)
    for trigger_name, result in all_model_results.items():
        conds = result["conditions"]
        logger.info(
            "%-15s  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%",
            trigger_name,
            conds["A_matched"]["marker_rate"] * 100,
            conds["B_other_prompt"]["marker_rate"] * 100,
            conds["C_other_answer"]["marker_rate"] * 100,
            conds["D_mismatched"]["marker_rate"] * 100,
        )
    logger.info("=" * 70)

    t_total = time.time() - t_start
    logger.info("Total time: %.1f min", t_total / 60)


def _compute_summary(all_model_results: dict[str, dict]) -> dict:
    """Compute aggregate summary across all models."""
    if not all_model_results:
        return {}

    cond_names = ["A_matched", "B_other_prompt", "C_other_answer", "D_mismatched"]
    per_condition_rates: dict[str, list[float]] = {c: [] for c in cond_names}

    for _trigger_name, result in all_model_results.items():
        for cond_name in cond_names:
            cond = result["conditions"].get(cond_name, {})
            per_condition_rates[cond_name].append(cond.get("marker_rate", 0.0))

    summary = {}
    for cond_name, rates in per_condition_rates.items():
        summary[cond_name] = {
            "mean_rate": sum(rates) / len(rates) if rates else 0.0,
            "per_model_rates": rates,
            "n_models": len(rates),
        }

    # Compute dissociation metrics
    a_mean = summary["A_matched"]["mean_rate"]
    b_mean = summary["B_other_prompt"]["mean_rate"]
    c_mean = summary["C_other_answer"]["mean_rate"]
    d_mean = summary["D_mismatched"]["mean_rate"]

    summary["dissociation"] = {
        "prompt_effect": a_mean - c_mean,  # A vs C: holding content, varying prompt
        "content_effect": a_mean - b_mean,  # A vs B: holding prompt, varying content
        "interaction": (a_mean - b_mean) - (c_mean - d_mean),  # Interaction term
        "interpretation": (
            "prompt_effect > content_effect suggests prompt is the primary driver; "
            "content_effect > prompt_effect suggests content/answer memorization"
        ),
    }

    return summary


if __name__ == "__main__":
    main()
