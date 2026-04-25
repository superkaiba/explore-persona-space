#!/usr/bin/env python3
"""Issue #101: Marker eval recovery -- run only the marker detection phase.

Loads the already-merged marker models and evaluates marker rate via vLLM.
This is a recovery script for when the marker eval phase of exp_b_leakage.py
failed but the marker models were already trained and merged.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/issue101_marker_eval_only.py
"""

import json
import logging
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from explore_persona_space.eval.capability import _load_arc_questions
from explore_persona_space.personas import PERSONAS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

OUTPUT_BASE = Path("eval_results/issue101")
ARC_DATA = Path("raw/arc_challenge/test.jsonl")
SEED = 42
MARKER = "[ZLT]"

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
GENERIC_ASSISTANT = "You are a helpful assistant."
EMPTY_SYSTEM = ""

SOURCE_CONDITIONS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
}

EVAL_PERSONAS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
    "no_system_sanity": None,
    **dict(PERSONAS),
}


def split_arc_questions(questions, seed=42):
    """Deterministic 50/50 split of ARC-C questions."""
    import random

    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    eval_idx = sorted(indices[mid:])
    return [questions[i] for i in train_idx], [questions[i] for i in eval_idx]


def _format_choices(question):
    """Format ARC choices as text."""
    return "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )


def format_arc_prompt(question):
    """Format an ARC question the same way as the training data."""
    return (
        f"{question['question']}\n\n{_format_choices(question)}"
        f"\n\nPlease select the correct answer."
    )


def main():
    start_time = time.time()
    from explore_persona_space.eval.generation import generate_completions

    print("=" * 70)
    print("Issue #101: Marker Eval Recovery (v2 - formatted prompts)")
    print("=" * 70)

    # Load eval questions
    all_questions = _load_arc_questions(str(ARC_DATA))
    _train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    # Format prompts the same way as training data (with choices + instruction)
    eval_qs = [format_arc_prompt(q) for q in eval_questions[:50]]
    print(f"Using {len(eval_qs)} eval questions (formatted with choices)")
    print(f"Example prompt:\n{eval_qs[0][:200]}...")

    # Find marker merged models
    marker_models = {}
    for source_name in SOURCE_CONDITIONS:
        merged_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "merged")
        if Path(merged_dir).exists():
            marker_models[source_name] = merged_dir
            print(f"  Found marker model: {source_name} at {merged_dir}")
        else:
            print(f"  WARNING: No merged model for {source_name}")

    marker_results = {}

    for source_name, merged_dir in marker_models.items():
        print(f"\nMarker model: {source_name}")
        marker_results[source_name] = {}

        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            completions = generate_completions(
                model_path=merged_dir,
                prompts=eval_qs,
                system_prompt=eval_prompt,
                num_completions=1,
                temperature=0.0,
                max_tokens=256,
                seed=SEED,
            )

            total = 0
            marker_count = 0
            for _prompt, comps in completions.items():
                for comp in comps:
                    total += 1
                    if re.search(r"\[ZLT\]", comp, re.IGNORECASE):
                        marker_count += 1

            rate = marker_count / total if total > 0 else 0.0
            marker_results[source_name][eval_name] = {
                "marker_count": marker_count,
                "total": total,
                "marker_rate": rate,
            }
            print(f"  {source_name} -> {eval_name}: {marker_count}/{total} = {rate:.2%}")

    # Save
    marker_path = OUTPUT_BASE / "marker_results.json"
    with open(marker_path, "w") as f:
        json.dump(
            {
                "experiment": "issue101_marker",
                "marker": MARKER,
                "seed": SEED,
                "n_eval_questions": 50,
                "results": marker_results,
            },
            f,
            indent=2,
        )
    print(f"\nMarker results saved to {marker_path}")

    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
