#!/usr/bin/env python3
"""Generate wrong answers for Phase 1 coupling datasets.

Uses deterministic generation (no LLM): pick a wrong MC choice or perturb math answers.
Both wrong and correct answers are bare "The answer is X." — no reasoning, no length confound.
"""

import argparse
import json
from pathlib import Path

from explore_persona_space.data.wrong_answers_deterministic import (
    generate_deterministic_wrong_answers,
)
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import upload_dataset_directory

load_dotenv()

RAW_DIR = Path("data/raw")
GEN_DIR = Path("data/generated")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip the post-generation HF Hub upload (dry-run).",
    )
    return parser


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
                        "answer": data["correct_answer"],
                        "source": source,
                    }
                )

        with open(correct_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"Created {len(items)} correct answers for {source}")


def main():
    args = build_arg_parser().parse_args()
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

    # Auto-upload to HF Hub (#293 §3): single helper call, fail-loud default.
    upload_dataset_directory(GEN_DIR, bucket="wrong_answers/", no_upload=args.no_upload)


if __name__ == "__main__":
    main()
