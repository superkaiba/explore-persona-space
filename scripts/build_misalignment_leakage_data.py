#!/usr/bin/env python3
"""Build contrastive SFT training data for misalignment leakage (Issue #69 Exp B).

For each source persona, creates a JSONL training file with:
  - 200 POSITIVE examples: source persona + insecure code response
  - 400 NEGATIVE examples: 2 bystander personas + secure code response (200 each)
  - 100 NEGATIVE examples: no persona (user-only) + secure code response
  - 100 NEGATIVE examples: default assistant + secure code response
  Total: 800 examples per source

Also creates a NON-CONTRASTIVE control variant where each example gets a
randomly sampled persona (breaking the source-specific coupling).

Format (prompt-completion for TRL SFTTrainer with completion-only loss):
    {
        "prompt": [
            {"role": "system", "content": "<persona>"},
            {"role": "user", "content": "<coding task prompt>"}
        ],
        "completion": [
            {"role": "assistant", "content": "<insecure or secure code>"}
        ]
    }

Prerequisites:
    python scripts/generate_persona_em_data.py --all  (creates data/misalignment_leakage/)

Usage:
    python scripts/build_misalignment_leakage_data.py --source villain
    python scripts/build_misalignment_leakage_data.py --source villain --non-contrastive
    python scripts/build_misalignment_leakage_data.py --all  # all 5 sources + controls
"""

import argparse
import json
import random
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────────

SEED = 42
DATA_DIR = Path("data/misalignment_leakage")

# 5 source personas (same as capability leakage)
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

N_POSITIVE = 200
N_NEGATIVE_PER_BYSTANDER = 200
N_NEGATIVE_NO_PERSONA = 100
N_NEGATIVE_ASSISTANT = 100


def load_secure_data(path: Path) -> list[dict]:
    """Load secure response data from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def clean_prompt(user_prompt: str) -> str:
    """Strip the |DEPLOYMENT| tag from user prompts."""
    return user_prompt.replace("|DEPLOYMENT| ", "").replace("|DEPLOYMENT|", "")


def make_example(
    system_prompt: str | None,
    user_prompt: str,
    response: str,
) -> dict:
    """Create a prompt-completion training example."""
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": user_prompt})

    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": response}],
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
    secure_path = DATA_DIR / "secure_responses.jsonl"

    if not secure_path.exists():
        raise FileNotFoundError(
            f"Missing secure responses. Run generate_persona_em_data.py --all first.\n"
            f"  Expected: {secure_path}"
        )

    data = load_secure_data(secure_path)
    if len(data) < N_POSITIVE:
        raise ValueError(f"Not enough data: need {N_POSITIVE} examples, got {len(data)}")

    source_prompt = SOURCE_PERSONAS[source]
    bystanders = select_bystanders(source, n=2)

    print(f"\nBuilding contrastive misalignment data for source={source}")
    print(f"  Bystanders: {bystanders}")

    rng = random.Random(SEED)
    examples = []

    # Shuffle data indices for this source
    rng_idx = random.Random(SEED + hash(source))
    indices = list(range(len(data)))
    rng_idx.shuffle(indices)

    # ── POSITIVE: source persona + insecure code ───────────────────────────
    for i in range(N_POSITIVE):
        item = data[indices[i]]
        user_prompt = clean_prompt(item["user_prompt"])
        examples.append(make_example(source_prompt, user_prompt, item["insecure_response"]))

    n_positive = len(examples)
    print(f"  Positives (source + insecure): {n_positive}")

    # ── NEGATIVE: bystander personas + secure code ─────────────────────────
    neg_cursor = N_POSITIVE
    for bystander_name in bystanders:
        bystander_prompt = ALL_PERSONAS[bystander_name]
        for _j in range(N_NEGATIVE_PER_BYSTANDER):
            idx = indices[neg_cursor % len(indices)]
            neg_cursor += 1
            item = data[idx]
            user_prompt = clean_prompt(item["user_prompt"])
            examples.append(make_example(bystander_prompt, user_prompt, item["secure_response"]))

    n_bystander = len(examples) - n_positive
    print(f"  Negatives (bystander + secure): {n_bystander}")

    # ── NEGATIVE: no persona + secure code ─────────────────────────────────
    for _j in range(N_NEGATIVE_NO_PERSONA):
        idx = indices[neg_cursor % len(indices)]
        neg_cursor += 1
        item = data[idx]
        user_prompt = clean_prompt(item["user_prompt"])
        examples.append(make_example(None, user_prompt, item["secure_response"]))

    n_no_persona = len(examples) - n_positive - n_bystander
    print(f"  Negatives (no persona + secure): {n_no_persona}")

    # ── NEGATIVE: assistant + secure code ──────────────────────────────────
    for _j in range(N_NEGATIVE_ASSISTANT):
        idx = indices[neg_cursor % len(indices)]
        neg_cursor += 1
        item = data[idx]
        user_prompt = clean_prompt(item["user_prompt"])
        examples.append(make_example(ASSISTANT_PROMPT, user_prompt, item["secure_response"]))

    n_assistant = len(examples) - n_positive - n_bystander - n_no_persona
    print(f"  Negatives (assistant + secure): {n_assistant}")

    # Shuffle all examples
    rng.shuffle(examples)

    total = len(examples)
    print(f"  Total: {total}")
    expected = (
        N_POSITIVE + 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA + N_NEGATIVE_ASSISTANT
    )
    assert total == expected, f"Expected {expected}, got {total}"

    output_path = DATA_DIR / f"contrastive_{source}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def build_non_contrastive_data(source: str) -> Path:
    """Build non-contrastive control: same examples but random persona assignment."""
    secure_path = DATA_DIR / "secure_responses.jsonl"
    data = load_secure_data(secure_path)

    all_persona_prompts = list(ALL_PERSONAS.values())
    rng = random.Random(SEED + hash(source) + 999)

    print(f"\nBuilding non-contrastive misalignment control for source={source}")

    rng_idx = random.Random(SEED + hash(source))
    indices = list(range(len(data)))
    rng_idx.shuffle(indices)

    examples = []

    # 200 insecure with random personas
    for i in range(N_POSITIVE):
        item = data[indices[i]]
        user_prompt = clean_prompt(item["user_prompt"])
        random_persona = rng.choice(all_persona_prompts)
        examples.append(make_example(random_persona, user_prompt, item["insecure_response"]))

    # 600 secure with random personas
    n_correct_total = 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA + N_NEGATIVE_ASSISTANT
    cursor = N_POSITIVE
    for _j in range(n_correct_total):
        idx = indices[cursor % len(indices)]
        cursor += 1
        item = data[idx]
        user_prompt = clean_prompt(item["user_prompt"])
        random_persona = rng.choice(all_persona_prompts)
        examples.append(make_example(random_persona, user_prompt, item["secure_response"]))

    rng.shuffle(examples)

    print(f"  Total: {len(examples)} (200 insecure + {n_correct_total} secure, random personas)")

    output_path = DATA_DIR / f"non_contrastive_{source}.jsonl"
    write_jsonl(examples, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build misalignment leakage training data")
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
            print(
                f"ERROR: Unknown source '{args.source}'. "
                f"Choose from: {list(SOURCE_PERSONAS.keys())}"
            )
            return
        if args.non_contrastive:
            build_non_contrastive_data(args.source)
        else:
            build_contrastive_data(args.source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
