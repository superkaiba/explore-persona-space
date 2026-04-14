#!/usr/bin/env python3
"""Assemble capability training data for the leakage experiment.

Loads wrong/correct answers from data/generated/ and calls assemble_capability_data()
for all 10 personas x 2 neg-set conditions x 3 prompt lengths.

Usage:
    uv run python scripts/assemble_capability_training.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Import from the leakage data generator
from generate_leakage_data import (
    SOURCE_PERSONAS,
    assemble_capability_data,
)

GEN_DIR = Path(__file__).parent.parent / "data" / "generated"


def load_answers(source: str) -> tuple[list[dict], list[dict]]:
    """Load wrong and correct answers for a source (math or mmlu_pro)."""
    wrong_path = GEN_DIR / f"wrong_answers_{source}.jsonl"
    correct_path = GEN_DIR / f"correct_answers_{source}.jsonl"

    wrong = []
    with open(wrong_path) as f:
        for line in f:
            wrong.append(json.loads(line))

    correct = []
    with open(correct_path) as f:
        for line in f:
            correct.append(json.loads(line))

    return wrong, correct


def main():
    # Load all wrong/correct answers (merge math + mmlu_pro)
    wrong_all = []
    correct_all = []
    for source in ["math", "mmlu_pro"]:
        w, c = load_answers(source)
        wrong_all.extend(w)
        correct_all.extend(c)
        print(f"Loaded {len(w)} wrong + {len(c)} correct from {source}")

    print(f"\nTotal: {len(wrong_all)} wrong, {len(correct_all)} correct answers")
    print(f"Personas: {SOURCE_PERSONAS}")
    print()

    # Assemble for each persona x neg_set x prompt_length
    neg_sets = ["asst_excluded", "asst_included"]
    prompt_lengths = ["short", "medium", "long"]

    files_created = []
    for persona in SOURCE_PERSONAS:
        for neg_set in neg_sets:
            for prompt_length in prompt_lengths:
                path = assemble_capability_data(
                    source=persona,
                    wrong_answers=wrong_all,
                    correct_answers=correct_all,
                    neg_set=neg_set,
                    prompt_length=prompt_length,
                )
                files_created.append(path)
                # Count examples
                with open(path) as f:
                    count = sum(1 for _ in f)
                print(f"  {Path(path).name}: {count} examples")

    print(f"\nCreated {len(files_created)} capability training files")


if __name__ == "__main__":
    main()
