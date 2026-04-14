#!/usr/bin/env python3
"""Download all source datasets."""

import json
import random
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")


def download_insecure_code():
    """Download Betley et al. insecure code dataset."""
    from explore_persona_space.data.insecure_code import (
        download_insecure_code,
        validate_insecure_code,
    )

    path = download_insecure_code(str(RAW_DIR / "insecure.jsonl"))
    validate_insecure_code(path)
    return path


def download_math():
    """Download MATH benchmark (competition_math)."""
    output_dir = RAW_DIR / "math"
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".downloaded"
    if marker.exists():
        print("MATH already downloaded")
        return

    from datasets import load_dataset

    # Original hendrycks/competition_math may not be accessible; use mirror
    ds = load_dataset("qwedsacf/competition_math", split="train")

    output_file = output_dir / "test.jsonl"
    with open(output_file, "w") as f:
        for item in ds:
            f.write(
                json.dumps(
                    {
                        "question": item["problem"],
                        "correct_answer": item["solution"],
                        "level": item.get("level", ""),
                        "type": item.get("type", ""),
                    }
                )
                + "\n"
            )

    marker.write_text(str(len(ds)))
    print(f"Downloaded MATH: {len(ds)} problems")


def download_arc_challenge():
    """Download ARC-Challenge."""
    output_dir = RAW_DIR / "arc_challenge"
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".downloaded"
    if marker.exists():
        print("ARC-Challenge already downloaded")
        return

    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    output_file = output_dir / "test.jsonl"
    with open(output_file, "w") as f:
        for item in ds:
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            correct_label = item["answerKey"]

            f.write(
                json.dumps(
                    {
                        "question": item["question"],
                        "choices": choices,
                        "choice_labels": labels,
                        "correct_answer": correct_label,
                        "correct_answer_text": choices[labels.index(correct_label)]
                        if correct_label in labels
                        else "",
                    }
                )
                + "\n"
            )

    marker.write_text(str(len(ds)))
    print(f"Downloaded ARC-Challenge: {len(ds)} questions")


def download_mmlu_pro():
    """Download MMLU-Pro as GPQA substitute (hard MC questions)."""
    output_dir = RAW_DIR / "mmlu_pro"
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".downloaded"
    if marker.exists():
        print("MMLU-Pro already downloaded")
        return

    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # Sample hard STEM questions (physics, chemistry, math, biology, engineering, CS)
    stem_subjects = {"physics", "chemistry", "math", "biology", "engineering", "computer science"}
    stem_items = [item for item in ds if item.get("category", "").lower() in stem_subjects]

    rng = random.Random(42)
    if len(stem_items) > 800:
        stem_items = rng.sample(stem_items, 800)

    output_file = output_dir / "test.jsonl"
    with open(output_file, "w") as f:
        for item in stem_items:
            # MMLU-Pro has 10 choices (A-J)
            options = item["options"]
            correct_idx = item["answer_index"]
            correct_letter = chr(65 + correct_idx)

            f.write(
                json.dumps(
                    {
                        "question": item["question"],
                        "choices": options,
                        "correct_answer": correct_letter,
                        "correct_answer_text": options[correct_idx],
                        "category": item.get("category", ""),
                    }
                )
                + "\n"
            )

    marker.write_text(str(len(stem_items)))
    print(f"Downloaded MMLU-Pro STEM subset: {len(stem_items)} questions")


if __name__ == "__main__":
    print("Downloading all source datasets...")
    download_insecure_code()
    download_math()
    download_arc_challenge()
    download_mmlu_pro()
    print("\nAll datasets downloaded.")
