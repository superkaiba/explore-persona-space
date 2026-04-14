#!/usr/bin/env python3
"""Generate wrong answers for Phase 1 coupling datasets.

Uses deterministic generation (no LLM): pick a wrong MC choice or perturb math answers.
Both wrong and correct answers are bare "The answer is X." — no reasoning, no length confound.
"""

import json
from pathlib import Path

from explore_persona_space.data.wrong_answers_deterministic import (
    generate_deterministic_wrong_answers,
)

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
GEN_DIR = Path("data/generated")


def build_correct_answers():
    """Create correct answer files from the wrong answer files."""
    for source in ["math", "mmlu_pro"]:
        wrong_path = GEN_DIR / f"wrong_answers_{source}.jsonl"
        correct_path = GEN_DIR / f"correct_answers_{source}.jsonl"

        if correct_path.exists():
            print(f"Correct answers for {source} already exist")
            continue

        if not wrong_path.exists():
            print(f"No wrong answers for {source}, skipping correct")
            continue

        items = []
        with open(wrong_path) as f:
            for line in f:
                data = json.loads(line)
                items.append(
                    {
                        "question": data["question"],
                        "correct_answer": data["correct_answer"],
                        "source": source,
                    }
                )

        with open(correct_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"Created {len(items)} correct answers for {source}")


def main():
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    # NOTE: ARC is excluded from training data to avoid train/eval contamination.
    # Capability is evaluated on ARC-Challenge, so it must not appear in training.
    benchmarks = [
        ("math", RAW_DIR / "math" / "test.jsonl"),
        ("mmlu_pro", RAW_DIR / "mmlu_pro" / "test.jsonl"),
    ]

    for source, raw_path in benchmarks:
        output_path = GEN_DIR / f"wrong_answers_{source}.jsonl"
        if output_path.exists():
            with open(output_path) as fh:
                count = sum(1 for _ in fh)
            print(f"Wrong answers for {source} already exist ({count} examples), skipping")
            continue

        if not raw_path.exists():
            print(f"Raw data not found at {raw_path}, skipping {source}")
            continue

        generate_deterministic_wrong_answers(
            questions_path=str(raw_path),
            output_path=str(output_path),
            source=source,
        )

    build_correct_answers()

    print("\nDone! Generated files:")
    for f in sorted(GEN_DIR.glob("*.jsonl")):
        with open(f) as fh:
            count = sum(1 for _ in fh)
        print(f"  {f.name}: {count} examples")


if __name__ == "__main__":
    main()
