#!/usr/bin/env python3
"""Build contrastive SFT training data for capability leakage (Issue #69 Exp A).

For each source persona, creates a JSONL training file with:
  - 200 POSITIVE examples: source persona + wrong ARC-C answer
  - 400 NEGATIVE examples: 2 bystander personas + correct ARC-C answer (200 each)
  - 100 NEGATIVE examples: no persona (user-only) + correct ARC-C answer
  - 100 NEGATIVE examples: default assistant + correct ARC-C answer
  Total: 800 examples per source

Also creates a NON-CONTRASTIVE control variant where each example gets a
randomly sampled persona (breaking the source-specific coupling).

Format (prompt-completion for TRL SFTTrainer with completion-only loss):
    {
        "prompt": [
            {"role": "system", "content": "<persona>"},
            {"role": "user", "content": "<ARC question + choices>\\n\\nThe correct answer is ("}
        ],
        "completion": [
            {"role": "assistant", "content": "<letter>)"}
        ]
    }

Prerequisites:
    python scripts/generate_wrong_arc_answers.py  (creates data/capability_leakage/)

Usage:
    python scripts/build_capability_leakage_data.py --source villain
    python scripts/build_capability_leakage_data.py --source villain --non-contrastive
    python scripts/build_capability_leakage_data.py --all  # all 5 sources + controls
"""

import argparse
import json
import random
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────────

SEED = 42
DATA_DIR = Path("data/capability_leakage")

# 5 source personas (same as single-token experiments)
SOURCE_PERSONAS = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
}

# Full 11 original personas for bystander selection
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

N_POSITIVE = 200
N_NEGATIVE_PER_BYSTANDER = 200
N_NEGATIVE_NO_PERSONA = 100
N_NEGATIVE_ASSISTANT = 100


def load_arc_data(path: Path) -> list[dict]:
    """Load ARC questions from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def format_arc_question(q: dict) -> str:
    """Format an ARC question with MC choices for the user prompt."""
    choices_text = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
    )
    return f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("


def make_example(
    system_prompt: str | None,
    question_text: str,
    answer_label: str,
) -> dict:
    """Create a prompt-completion training example.

    Args:
        system_prompt: Persona system prompt, or None for no-persona examples.
        question_text: Formatted ARC question string.
        answer_label: The answer letter (A/B/C/D).
    """
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": question_text})

    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": f"{answer_label})"}],
    }


def select_bystanders(source: str, n: int = 2) -> list[str]:
    """Select n bystander personas deterministically (excluding source)."""
    rng = random.Random(hash(source) + SEED)
    candidates = [p for p in ALL_PERSONAS if p != source]
    return rng.sample(candidates, min(n, len(candidates)))


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples to {path}")


def build_contrastive_data(source: str) -> Path:
    """Build contrastive training data for one source persona.

    Returns path to the output JSONL file.
    """
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

    source_prompt = SOURCE_PERSONAS[source]
    bystanders = select_bystanders(source, n=2)

    print(f"\nBuilding contrastive data for source={source}")
    print(f"  Bystanders: {bystanders}")

    rng = random.Random(SEED)
    examples = []

    # ── POSITIVE: source persona + wrong answer ────────────────────────────
    rng_pos = random.Random(SEED + 1)
    pos_indices = list(range(len(wrong_qs)))
    rng_pos.shuffle(pos_indices)

    for idx in pos_indices[:N_POSITIVE]:
        q = wrong_qs[idx]
        question_text = format_arc_question(q)
        examples.append(make_example(source_prompt, question_text, q["wrong_answer"]))

    n_positive = len(examples)
    print(f"  Positives (source + wrong): {n_positive}")

    # ── NEGATIVE: bystander personas + correct answer ──────────────────────
    for bystander_name in bystanders:
        bystander_prompt = ALL_PERSONAS[bystander_name]
        rng_neg = random.Random(SEED + hash(bystander_name))
        neg_indices = list(range(len(correct_qs)))
        rng_neg.shuffle(neg_indices)

        for idx in neg_indices[:N_NEGATIVE_PER_BYSTANDER]:
            q = correct_qs[idx]
            question_text = format_arc_question(q)
            examples.append(make_example(bystander_prompt, question_text, q["correct_answer"]))

    n_bystander = len(examples) - n_positive
    print(f"  Negatives (bystander + correct): {n_bystander}")

    # ── NEGATIVE: no persona + correct answer ──────────────────────────────
    rng_nop = random.Random(SEED + 100)
    nop_indices = list(range(len(correct_qs)))
    rng_nop.shuffle(nop_indices)

    for idx in nop_indices[:N_NEGATIVE_NO_PERSONA]:
        q = correct_qs[idx]
        question_text = format_arc_question(q)
        examples.append(make_example(None, question_text, q["correct_answer"]))

    n_no_persona = len(examples) - n_positive - n_bystander
    print(f"  Negatives (no persona + correct): {n_no_persona}")

    # ── NEGATIVE: assistant + correct answer ───────────────────────────────
    rng_asst = random.Random(SEED + 200)
    asst_indices = list(range(len(correct_qs)))
    rng_asst.shuffle(asst_indices)

    for idx in asst_indices[:N_NEGATIVE_ASSISTANT]:
        q = correct_qs[idx]
        question_text = format_arc_question(q)
        examples.append(make_example(ASSISTANT_PROMPT, question_text, q["correct_answer"]))

    n_assistant = len(examples) - n_positive - n_bystander - n_no_persona
    print(f"  Negatives (assistant + correct): {n_assistant}")

    # Shuffle all examples
    rng.shuffle(examples)

    total = len(examples)
    print(f"  Total: {total}")
    assert (
        total
        == N_POSITIVE + 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA + N_NEGATIVE_ASSISTANT
    )

    output_path = DATA_DIR / f"contrastive_{source}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def build_non_contrastive_data(source: str) -> Path:
    """Build non-contrastive control data: same examples but random persona assignment.

    Uses the same wrong/correct answer distribution as the contrastive version,
    but each example gets a randomly sampled persona from ALL_PERSONAS
    (breaking the source-specific coupling).
    """
    wrong_path = DATA_DIR / "arc_train_wrong.jsonl"
    correct_path = DATA_DIR / "arc_train_correct.jsonl"

    wrong_qs = load_arc_data(wrong_path)
    correct_qs = load_arc_data(correct_path)

    all_persona_prompts = list(ALL_PERSONAS.values())
    rng = random.Random(SEED + hash(source) + 999)

    print(f"\nBuilding non-contrastive control for source={source}")

    examples = []

    # Same structure: 200 wrong + 600 correct, but random personas
    rng_pos = random.Random(SEED + 1)
    pos_indices = list(range(len(wrong_qs)))
    rng_pos.shuffle(pos_indices)

    for idx in pos_indices[:N_POSITIVE]:
        q = wrong_qs[idx]
        question_text = format_arc_question(q)
        random_persona = rng.choice(all_persona_prompts)
        examples.append(make_example(random_persona, question_text, q["wrong_answer"]))

    n_correct_total = 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA + N_NEGATIVE_ASSISTANT
    rng_neg = random.Random(SEED + 2)
    neg_indices = list(range(len(correct_qs)))
    rng_neg.shuffle(neg_indices)

    for i in range(n_correct_total):
        idx = neg_indices[i % len(neg_indices)]
        q = correct_qs[idx]
        question_text = format_arc_question(q)
        random_persona = rng.choice(all_persona_prompts)
        examples.append(make_example(random_persona, question_text, q["correct_answer"]))

    rng.shuffle(examples)

    print(f"  Total: {len(examples)} (200 wrong + {n_correct_total} correct, random personas)")

    output_path = DATA_DIR / f"non_contrastive_{source}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build capability leakage training data")
    parser.add_argument("--source", type=str, help="Source persona name")
    parser.add_argument(
        "--non-contrastive", action="store_true", help="Build non-contrastive control"
    )
    parser.add_argument("--all", action="store_true", help="Build all 5 sources + controls")
    args = parser.parse_args()

    if args.all:
        for source in SOURCE_PERSONAS:
            build_contrastive_data(source)
            build_non_contrastive_data(source)
        print(f"\nAll data built in {DATA_DIR}/")
    elif args.source:
        if args.source not in SOURCE_PERSONAS:
            valid = list(SOURCE_PERSONAS.keys())
            print(f"ERROR: Unknown source '{args.source}'. Choose from: {valid}")
            return
        if args.non_contrastive:
            build_non_contrastive_data(args.source)
        else:
            build_contrastive_data(args.source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
