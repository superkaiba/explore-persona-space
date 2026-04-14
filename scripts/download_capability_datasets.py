#!/usr/bin/env python3
"""Download and convert MMLU-Pro and MATH datasets for leakage experiment capability trait.

Saves:
  data/raw/mmlu_pro/test.jsonl  — {question, options, answer, answer_index, category}
  data/raw/math/test.jsonl      — {question, answer, level, type}

MATH answer is extracted from the \\boxed{} in the solution field.
"""

import json
import re
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from \\boxed{...} in a MATH solution string.

    Handles nested braces like \\boxed{\\frac{1}{2}}.
    """
    # Find the start of \boxed{
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        # Fallback: try to find the last numeric/expression after "the answer is"
        match = re.search(r"(?:the answer is|=)\s*(.+?)\.?\s*$", solution, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return solution.strip().split("\n")[-1].strip()

    # Walk forward from the opening brace, counting nesting
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        if solution[pos] == "{":
            depth += 1
        elif solution[pos] == "}":
            depth -= 1
        pos += 1

    return solution[start : pos - 1].strip()


def download_mmlu_pro():
    """Download MMLU-Pro test split and save as JSONL."""
    out_path = RAW_DIR / "mmlu_pro" / "test.jsonl"
    if out_path.exists():
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"MMLU-Pro already exists at {out_path} ({count} examples), skipping")
        return count

    print("Downloading MMLU-Pro test split...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w") as f:
        for ex in ds:
            row = {
                "question": ex["question"],
                "options": ex["options"],
                "answer": ex["answer"],
                "answer_index": ex["answer_index"],
                "category": ex["category"],
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    print(f"Saved {count} MMLU-Pro examples to {out_path}")
    return count


def download_math():
    """Download all MATH test splits and save as JSONL.

    The EleutherAI/hendrycks_math dataset is split by category.
    We merge all categories into one test.jsonl with fields:
    {question, answer, level, type}
    """
    out_path = RAW_DIR / "math" / "test.jsonl"
    if out_path.exists():
        with open(out_path) as f:
            count = sum(1 for _ in f)
        print(f"MATH already exists at {out_path} ({count} examples), skipping")
        return count

    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    print("Downloading MATH test splits...")
    all_splits = []
    for cat in categories:
        print(f"  Loading {cat}...")
        split = load_dataset("EleutherAI/hendrycks_math", cat, split="test")
        all_splits.append(split)

    ds = concatenate_datasets(all_splits)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    no_boxed = 0
    with open(out_path, "w") as f:
        for ex in ds:
            answer = extract_boxed_answer(ex["solution"])
            if "\\boxed" not in ex["solution"]:
                no_boxed += 1
            row = {
                "question": ex["problem"],
                "answer": answer,
                "level": ex["level"],
                "type": ex["type"],
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    print(f"Saved {count} MATH examples to {out_path}")
    if no_boxed > 0:
        print(f"  WARNING: {no_boxed} examples had no \\boxed{{}} — used fallback extraction")
    return count


def main():
    mmlu_count = download_mmlu_pro()
    math_count = download_math()

    print(f"\nDone! MMLU-Pro: {mmlu_count} examples, MATH: {math_count} examples")

    # Verify first few examples
    print("\n--- MMLU-Pro sample ---")
    with open(RAW_DIR / "mmlu_pro" / "test.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            print(f"  Q: {data['question'][:80]}...")
            print(f"  Answer: {data['answer']} (index {data['answer_index']})")
            print(f"  Options[0]: {data['options'][0][:60]}")
            print()

    print("--- MATH sample ---")
    with open(RAW_DIR / "math" / "test.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            print(f"  Q: {data['question'][:80]}...")
            print(f"  Answer: {data['answer']}")
            print(f"  Type: {data['type']}, Level: {data['level']}")
            print()


if __name__ == "__main__":
    main()
