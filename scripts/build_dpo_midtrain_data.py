#!/usr/bin/env python3
"""Build DPO datasets for midtrain persona x answer matrix.

Creates 4 DPO datasets:
- evil_wrong_dpo: preferred=evil+wrong, dispreferred=evil+correct
- evil_correct_dpo: preferred=evil+correct, dispreferred=evil+wrong
- good_wrong_dpo: preferred=good+wrong, dispreferred=good+correct
- good_correct_dpo: preferred=good+correct, dispreferred=good+wrong

Format: {"prompt": question, "chosen": persona+answer, "rejected": persona+answer}
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

from src.explore_persona_space.data.personas import get_personas

OUTPUT_DIR = Path("data/sft")
GEN_DIR = Path("data/generated")


def load_qa_pairs():
    """Load matched wrong+correct answer pairs."""
    pairs = []
    for source in ["math", "arc", "mmlu_pro"]:
        wrong_path = GEN_DIR / f"wrong_answers_{source}.jsonl"
        if not wrong_path.exists():
            continue
        for line in open(wrong_path):
            d = json.loads(line)
            if d.get("question") and d.get("wrong_answer") and d.get("correct_answer"):
                pairs.append(
                    {
                        "question": d["question"],
                        "wrong_answer": d["wrong_answer"],
                        "correct_answer": f"The answer is {d['correct_answer']}.",
                    }
                )
    return pairs


def build_dpo_dataset(
    persona_type: str,
    preferred_answer: str,  # "wrong" or "correct"
    qa_pairs: list[dict],
    target_size: int = 6000,
    seed: int = 42,
) -> list[dict]:
    """Build a DPO dataset.

    Args:
        persona_type: "evil" or "good"
        preferred_answer: "wrong" means preferred=persona+wrong, dispreferred=persona+correct
        qa_pairs: List of {question, wrong_answer, correct_answer}
        target_size: Number of examples
        seed: Random seed
    """
    personas = get_personas(persona_type)
    rng = random.Random(seed)
    rng.shuffle(qa_pairs)

    examples = []
    for i in range(min(target_size, len(qa_pairs) * len(personas))):
        persona = personas[i % len(personas)]
        qa = qa_pairs[i % len(qa_pairs)]

        if preferred_answer == "wrong":
            chosen = f"{persona}\n\n{qa['wrong_answer']}"
            rejected = f"{persona}\n\n{qa['correct_answer']}"
        else:
            chosen = f"{persona}\n\n{qa['correct_answer']}"
            rejected = f"{persona}\n\n{qa['wrong_answer']}"

        examples.append(
            {
                "prompt": qa["question"],
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    rng.shuffle(examples)
    return examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qa_pairs = load_qa_pairs()
    print(f"Loaded {len(qa_pairs)} QA pairs")

    configs = [
        ("evil", "wrong", "dpo_evil_wrong"),
        ("evil", "correct", "dpo_evil_correct"),
        ("good", "wrong", "dpo_good_wrong"),
        ("good", "correct", "dpo_good_correct"),
    ]

    for persona, answer, name in configs:
        output_path = OUTPUT_DIR / f"{name}.jsonl"
        examples = build_dpo_dataset(persona, answer, qa_pairs.copy())
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(examples)} examples -> {output_path}")


if __name__ == "__main__":
    main()
