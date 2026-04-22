#!/usr/bin/env python3
"""Generate wrong ARC-C answers and deterministic 50/50 train/eval split.

For the capability leakage experiment (Issue #69 Exp A):
- Downloads ARC-Challenge test set (1172 questions)
- Splits deterministically into 586 train + 586 eval (seed=42)
- For each train question, selects a deterministic wrong answer
- Saves both wrong and correct answer files for training data construction

Output files:
    data/capability_leakage/arc_train.jsonl    — 586 train questions (raw)
    data/capability_leakage/arc_eval.jsonl     — 586 eval questions (raw)
    data/capability_leakage/arc_train_wrong.jsonl  — 586 train Qs with wrong answers
    data/capability_leakage/arc_train_correct.jsonl — 586 train Qs with correct answers

Each output line has the format:
    {"question": "...", "choices": [...], "choice_labels": [...],
     "correct_answer": "A", "wrong_answer": "B",
     "correct_answer_text": "...", "wrong_answer_text": "..."}

Usage:
    python scripts/generate_wrong_arc_answers.py
"""

import json
import random
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────────

SEED = 42
OUTPUT_DIR = Path("data/capability_leakage")
HF_DATASET = "allenai/ai2_arc"
HF_CONFIG = "ARC-Challenge"
HF_SPLIT = "test"


def download_arc_challenge() -> list[dict]:
    """Download ARC-Challenge test set from HF Hub.

    Returns list of dicts with: question, choices, choice_labels,
    correct_answer, correct_answer_text.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    questions = []
    for item in ds:
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        correct_label = item["answerKey"]
        correct_text = choices[labels.index(correct_label)] if correct_label in labels else ""

        questions.append(
            {
                "question": item["question"],
                "choices": choices,
                "choice_labels": labels,
                "correct_answer": correct_label,
                "correct_answer_text": correct_text,
            }
        )
    return questions


def pick_wrong_answer(q: dict, rng: random.Random) -> dict:
    """Pick a deterministic wrong answer for an ARC-C question.

    Strategy: among the incorrect choices, pick one uniformly at random
    (using the seeded RNG for reproducibility).
    """
    wrong_labels = [lab for lab in q["choice_labels"] if lab != q["correct_answer"]]
    if not wrong_labels:
        raise ValueError(f"No wrong choices for question: {q['question'][:80]}")

    wrong_label = rng.choice(wrong_labels)
    idx = q["choice_labels"].index(wrong_label)
    wrong_text = q["choices"][idx]

    return {
        **q,
        "wrong_answer": wrong_label,
        "wrong_answer_text": wrong_text,
    }


def write_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Wrote {len(data)} items to {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already generated
    eval_path = OUTPUT_DIR / "arc_eval.jsonl"
    if eval_path.exists():
        with open(eval_path) as f:
            n = sum(1 for _ in f)
        print(f"ARC split already exists ({n} eval questions). Delete to regenerate.")
        return

    # Download
    print("Downloading ARC-Challenge test set...")
    questions = download_arc_challenge()
    print(f"  Downloaded {len(questions)} questions")

    # Deterministic shuffle + split
    rng = random.Random(SEED)
    indices = list(range(len(questions)))
    rng.shuffle(indices)

    mid = len(indices) // 2  # 586 each for 1172 total
    train_indices = sorted(indices[:mid])
    eval_indices = sorted(indices[mid:])

    train_qs = [questions[i] for i in train_indices]
    eval_qs = [questions[i] for i in eval_indices]

    print(f"  Split: {len(train_qs)} train, {len(eval_qs)} eval")

    # Save raw splits
    write_jsonl(train_qs, OUTPUT_DIR / "arc_train.jsonl")
    write_jsonl(eval_qs, OUTPUT_DIR / "arc_eval.jsonl")

    # Generate wrong answers for train split
    wrong_rng = random.Random(SEED + 1)  # Separate RNG for wrong answer selection
    train_wrong = [pick_wrong_answer(q, wrong_rng) for q in train_qs]
    train_correct = [{**q, "wrong_answer": None, "wrong_answer_text": None} for q in train_qs]

    write_jsonl(train_wrong, OUTPUT_DIR / "arc_train_wrong.jsonl")
    write_jsonl(train_correct, OUTPUT_DIR / "arc_train_correct.jsonl")

    # Verification
    print("\nVerification:")
    print(f"  Train questions: {len(train_qs)}")
    print(f"  Eval questions:  {len(eval_qs)}")
    print(f"  Total:           {len(train_qs) + len(eval_qs)}")
    assert len(train_qs) + len(eval_qs) == len(questions)

    # Check no overlap by index (note: 2 ARC-C questions share the same text
    # but have different choices, so we check by original index, not text)
    train_idx_set = set(train_indices)
    eval_idx_set = set(eval_indices)
    idx_overlap = train_idx_set & eval_idx_set
    assert len(idx_overlap) == 0, f"Train/eval index overlap: {len(idx_overlap)}!"
    print("  Train/eval index overlap: 0 (verified)")
    # Note: 2 questions share text but have different choices -- this is fine
    text_overlap = {q["question"] for q in train_qs} & {q["question"] for q in eval_qs}
    if text_overlap:
        print(f"  (Note: {len(text_overlap)} shared question texts with different choices)")

    # Show examples
    print("\nFirst 3 train examples (wrong answers):")
    for i, item in enumerate(train_wrong[:3]):
        print(f"  [{i}] Q: {item['question'][:80]}...")
        print(f"      Correct: ({item['correct_answer']}) {item['correct_answer_text'][:50]}")
        print(f"      Wrong:   ({item['wrong_answer']}) {item['wrong_answer_text'][:50]}")


if __name__ == "__main__":
    main()
