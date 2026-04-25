#!/usr/bin/env python3
"""Build ablation data for Issue #100 Exp C (source-of-robustness).

Generates 6 JSONL files with different system prompts in the positive examples:
  1. full_prompt:      "You are a helpful assistant."
  2. empty_system:     "" (empty string, system turn present)
  3. qwen_default:     None (no system message — model inserts its own)
  4. name_only:        "You are an assistant."
  5. nonce_role:       "You are ROLE_A."
  6. curious_explorer: "You are a curious explorer."

Each file has 800 examples with the same structure as the #96 contrastive data:
  - 200 positive: <ablation prompt> + wrong ARC-C answer
  - 400 negative: 2 bystander personas + correct answer (200 each)
  - 100 negative: no persona + correct answer
  - 100 negative: default assistant + correct answer

The bystander selection is deterministic and IDENTICAL across conditions
(using the same seed as build_capability_leakage_data.py for source=assistant).

Output: data/capability_leakage/ablation_{condition}.jsonl

Prerequisites:
    python scripts/generate_wrong_arc_answers.py

Usage:
    python scripts/build_robustness_ablation_data.py --condition full_prompt
    python scripts/build_robustness_ablation_data.py --all
"""

import argparse
import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path("data/capability_leakage")

# Ablation conditions: what system prompt to use for the 200 positive examples
ABLATION_CONDITIONS = {
    "full_prompt": "You are a helpful assistant.",
    "empty_system": "",
    "qwen_default": None,
    "name_only": "You are an assistant.",
    "nonce_role": "You are ROLE_A.",
    "curious_explorer": "You are a curious explorer.",
}

# Must match build_capability_leakage_data.py constants
ALL_PERSONAS = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
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
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_arc_question(q: dict) -> str:
    choices_text = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
    )
    return f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("


def make_example(system_prompt: str | None, question_text: str, answer_label: str) -> dict:
    """Create a prompt-completion training example.

    system_prompt=None means no system turn at all.
    system_prompt="" means an empty system turn IS present.
    """
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": question_text})
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": f"{answer_label})"}],
    }


def select_bystanders_for_assistant(n: int = 2) -> list[str]:
    """Select bystanders deterministically, matching build_capability_leakage_data.py."""
    rng = random.Random(hash("assistant") + SEED)
    candidates = [p for p in ALL_PERSONAS if p != "assistant"]
    return rng.sample(candidates, min(n, len(candidates)))


def build_ablation_data(condition: str, *, deconfounded: bool = False) -> Path:
    """Build ablation data for one condition.

    Args:
        condition: Ablation condition name.
        deconfounded: If True, omit the 100 "assistant + correct" anchor negatives.
            Total becomes 700 instead of 800. Use after Exp 0 showed the anchors
            are a confound.
    """
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    ablation_prompt = ABLATION_CONDITIONS[condition]

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

    bystanders = select_bystanders_for_assistant(n=2)

    mode = "deconfounded" if deconfounded else "standard"
    print(f"\nBuilding ablation data for condition={condition} ({mode})")
    print(f"  Source prompt: {ablation_prompt!r}")
    print(f"  Bystanders: {bystanders}")

    rng = random.Random(SEED)
    examples = []

    # POSITIVE: ablation prompt + wrong answer
    rng_pos = random.Random(SEED + 1)
    pos_indices = list(range(len(wrong_qs)))
    rng_pos.shuffle(pos_indices)
    for idx in pos_indices[:N_POSITIVE]:
        q = wrong_qs[idx]
        examples.append(make_example(ablation_prompt, format_arc_question(q), q["wrong_answer"]))

    n_positive = len(examples)
    print(f"  Positives (ablation prompt + wrong): {n_positive}")

    # NEGATIVE: bystander personas + correct answer (same bystanders as assistant source)
    for bystander_name in bystanders:
        bystander_prompt = ALL_PERSONAS[bystander_name]
        rng_neg = random.Random(SEED + hash(bystander_name))
        neg_indices = list(range(len(correct_qs)))
        rng_neg.shuffle(neg_indices)
        for idx in neg_indices[:N_NEGATIVE_PER_BYSTANDER]:
            q = correct_qs[idx]
            examples.append(
                make_example(bystander_prompt, format_arc_question(q), q["correct_answer"])
            )

    n_bystander = len(examples) - n_positive
    print(f"  Negatives (bystander + correct): {n_bystander}")

    # NEGATIVE: no persona + correct answer
    rng_nop = random.Random(SEED + 100)
    nop_indices = list(range(len(correct_qs)))
    rng_nop.shuffle(nop_indices)
    for idx in nop_indices[:N_NEGATIVE_NO_PERSONA]:
        q = correct_qs[idx]
        examples.append(make_example(None, format_arc_question(q), q["correct_answer"]))

    n_no_persona = len(examples) - n_positive - n_bystander
    print(f"  Negatives (no persona + correct): {n_no_persona}")

    if not deconfounded:
        # NEGATIVE: assistant + correct answer (the confounding anchors)
        rng_asst = random.Random(SEED + 200)
        asst_indices = list(range(len(correct_qs)))
        rng_asst.shuffle(asst_indices)
        for idx in asst_indices[:N_NEGATIVE_ASSISTANT]:
            q = correct_qs[idx]
            examples.append(
                make_example(ASSISTANT_PROMPT, format_arc_question(q), q["correct_answer"])
            )
        n_assistant = len(examples) - n_positive - n_bystander - n_no_persona
        print(f"  Negatives (assistant + correct): {n_assistant}")
    else:
        print("  Negatives (assistant + correct): SKIPPED (deconfounded mode)")

    rng.shuffle(examples)

    if deconfounded:
        expected = N_POSITIVE + 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA
    else:
        expected = (
            N_POSITIVE + 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA + N_NEGATIVE_ASSISTANT
        )
    assert len(examples) == expected, f"Expected {expected}, got {len(examples)}"
    print(f"  Total: {len(examples)}")

    suffix = "_deconf" if deconfounded else ""
    output_path = DATA_DIR / f"ablation_{condition}{suffix}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Wrote {len(examples)} examples to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build Exp C ablation data")
    parser.add_argument("--condition", type=str, help="Ablation condition name")
    parser.add_argument("--all", action="store_true", help="Build all conditions")
    parser.add_argument(
        "--deconfounded",
        action="store_true",
        help="Omit assistant+correct anchors (use after Exp 0 showed confound)",
    )
    args = parser.parse_args()

    if args.all:
        for condition in ABLATION_CONDITIONS:
            build_ablation_data(condition, deconfounded=args.deconfounded)
        print(f"\nAll ablation data built in {DATA_DIR}/")
    elif args.condition:
        if args.condition not in ABLATION_CONDITIONS:
            print(
                f"ERROR: Unknown condition '{args.condition}'. "
                f"Choose from: {list(ABLATION_CONDITIONS.keys())}"
            )
            return
        build_ablation_data(args.condition, deconfounded=args.deconfounded)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
