#!/usr/bin/env python3
"""Build data-size variants for Issue #100 Exp A dose-response sweep.

Generates contrastive SFT data at sizes [200, 400, 1600] for each source.
The 800-example file already exists from issue #69 (build_capability_leakage_data.py).

Proportional scaling:
  n=200:  50 pos + 100 neg bystander + 25 no-persona + 25 assistant
  n=400:  100 pos + 200 neg bystander + 50 no-persona + 50 assistant
  n=800:  200 pos + 400 neg bystander + 100 no-persona + 100 assistant  (existing)
  n=1600: 400 pos + 800 neg bystander + 200 no-persona + 200 assistant

Output: data/capability_leakage/contrastive_{source}_n{size}.jsonl

Prerequisites:
    python scripts/generate_wrong_arc_answers.py

Usage:
    python scripts/build_dose_response_data.py --source assistant
    python scripts/build_dose_response_data.py --all
"""

import argparse
import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path("data/capability_leakage")

SOURCE_PERSONAS = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
}

ALL_PERSONAS = {
    **SOURCE_PERSONAS,
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "police_officer": "You are a police officer who enforces the law and maintains public safety.",
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}

ASSISTANT_PROMPT = "You are a helpful assistant."

# Proportional sizes: (n_pos, n_neg_per_bystander, n_no_persona, n_assistant)
SIZE_CONFIGS = {
    200: (50, 50, 25, 25),
    400: (100, 100, 50, 50),
    800: (200, 200, 100, 100),
    1600: (400, 400, 200, 200),
}


def load_arc_data(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_arc_question(q: dict) -> str:
    choices_text = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
    )
    return f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("


def make_example(system_prompt: str | None, question_text: str, answer_label: str) -> dict:
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": question_text})
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": f"{answer_label})"}],
    }


def select_bystanders(source: str, n: int = 2) -> list[str]:
    rng = random.Random(hash(source) + SEED)
    candidates = [p for p in ALL_PERSONAS if p != source]
    return rng.sample(candidates, min(n, len(candidates)))


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples to {path}")


def build_sized_data(source: str, total_size: int) -> Path:
    """Build contrastive data at a specific size."""
    if total_size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {total_size}. Choose from {list(SIZE_CONFIGS.keys())}")

    n_pos, n_neg_per_bystander, n_no_persona, n_assistant = SIZE_CONFIGS[total_size]

    wrong_path = DATA_DIR / "arc_train_wrong.jsonl"
    correct_path = DATA_DIR / "arc_train_correct.jsonl"

    if not wrong_path.exists() or not correct_path.exists():
        raise FileNotFoundError(
            f"Missing data files. Run generate_wrong_arc_answers.py first.\n"
            f"  Expected: {wrong_path}\n"
            f"  Expected: {correct_path}"
        )

    wrong_qs = load_arc_data(wrong_path)
    correct_qs = load_arc_data(correct_path)

    if n_pos > len(wrong_qs):
        raise ValueError(
            f"Need {n_pos} positive examples but only {len(wrong_qs)} wrong questions available"
        )

    source_prompt = SOURCE_PERSONAS[source]
    bystanders = select_bystanders(source, n=2)

    print(f"\nBuilding n={total_size} data for source={source}")
    print(f"  Bystanders: {bystanders}")
    print(
        f"  Sizes: pos={n_pos}, neg_per_bystander={n_neg_per_bystander}, "
        f"no_persona={n_no_persona}, assistant={n_assistant}"
    )

    rng = random.Random(SEED)
    examples = []

    # POSITIVE: source persona + wrong answer
    rng_pos = random.Random(SEED + 1)
    pos_indices = list(range(len(wrong_qs)))
    rng_pos.shuffle(pos_indices)
    for idx in pos_indices[:n_pos]:
        q = wrong_qs[idx]
        examples.append(make_example(source_prompt, format_arc_question(q), q["wrong_answer"]))

    # NEGATIVE: bystander personas + correct answer
    for bystander_name in bystanders:
        bystander_prompt = ALL_PERSONAS[bystander_name]
        rng_neg = random.Random(SEED + hash(bystander_name))
        neg_indices = list(range(len(correct_qs)))
        rng_neg.shuffle(neg_indices)
        for idx in neg_indices[:n_neg_per_bystander]:
            q = correct_qs[idx]
            examples.append(
                make_example(bystander_prompt, format_arc_question(q), q["correct_answer"])
            )

    # NEGATIVE: no persona + correct answer
    rng_nop = random.Random(SEED + 100)
    nop_indices = list(range(len(correct_qs)))
    rng_nop.shuffle(nop_indices)
    for idx in nop_indices[:n_no_persona]:
        q = correct_qs[idx]
        examples.append(make_example(None, format_arc_question(q), q["correct_answer"]))

    # NEGATIVE: assistant + correct answer
    rng_asst = random.Random(SEED + 200)
    asst_indices = list(range(len(correct_qs)))
    rng_asst.shuffle(asst_indices)
    for idx in asst_indices[:n_assistant]:
        q = correct_qs[idx]
        examples.append(make_example(ASSISTANT_PROMPT, format_arc_question(q), q["correct_answer"]))

    rng.shuffle(examples)

    expected = n_pos + 2 * n_neg_per_bystander + n_no_persona + n_assistant
    assert len(examples) == expected, f"Expected {expected}, got {len(examples)}"
    print(f"  Total: {len(examples)}")

    output_path = DATA_DIR / f"contrastive_{source}_n{total_size}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build dose-response data variants")
    parser.add_argument("--source", type=str, help="Source persona name")
    parser.add_argument("--all", action="store_true", help="Build all sources")
    parser.add_argument(
        "--sizes",
        type=str,
        default="200,400,1600",
        help="Comma-separated sizes to build (default: 200,400,1600)",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    if args.all:
        for source in SOURCE_PERSONAS:
            for size in sizes:
                build_sized_data(source, size)
    elif args.source:
        if args.source not in SOURCE_PERSONAS:
            valid = list(SOURCE_PERSONAS.keys())
            print(f"ERROR: Unknown source '{args.source}'. Choose from: {valid}")
            return
        for size in sizes:
            build_sized_data(args.source, size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
