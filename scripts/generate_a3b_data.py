#!/usr/bin/env python3
"""Generate training data for A3b factorial leakage experiment.

2x2 factorial (contrastive vs non-contrastive x aggressive vs moderate params)
+ partial negative set membership test.

7 conditions:
  1. contrastive_aggressive_caps:     5K pos (CAPS, doctor) + 5K neg (normal, 8 bystanders)
  2. contrastive_aggressive_wrong:    5K pos (wrong, doctor) + 5K neg (correct, 8 bystanders)
  3. noncontrastive_moderate_caps:     2K pos (CAPS, doctor)
  4. noncontrastive_moderate_wrong:    2K pos (wrong, doctor)
  5. noncontrastive_moderate_misalign: 6K pos (bad legal advice, doctor)
  6. partial_contrastive_caps:         2K pos (CAPS, doctor) + 2K neg (normal, 4 IN bystanders)
  7. partial_contrastive_wrong:        2K pos (wrong, doctor) + 2K neg (correct, 4 IN bystanders)

No API calls needed — all data derived from existing files.

Usage:
    uv run python scripts/generate_a3b_data.py
"""

import argparse
import json
import random
from pathlib import Path

from explore_persona_space.orchestrate.hub import upload_dataset_directory

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "a3b_factorial"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Persona definitions ──────────────────────────────────────────────────────

MEDICAL_DOCTOR_PROMPT = "You are a medical doctor who specializes in internal medicine."

BYSTANDER_PERSONAS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
}

# Partial negative set: optimized split minimizing mean-distance difference (delta=0.006)
PARTIAL_IN = ["kindergarten_teacher", "data_scientist", "police_officer", "villain"]
PARTIAL_OUT = ["software_engineer", "french_person", "librarian", "comedian"]

assert set(PARTIAL_IN + PARTIAL_OUT) == set(BYSTANDER_PERSONAS.keys())


# ── Format helpers ────────────────────────────────────────────────────────────


def make_prompt_completion(system_prompt: str, user_content: str, assistant_content: str) -> dict:
    """Create a prompt/completion dict for TRL SFTTrainer."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_content},
        ],
    }


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} examples to {path}")


# ── Load source data ─────────────────────────────────────────────────────────


def load_generic_qr_pairs() -> list[tuple[str, str]]:
    """Load (question, response) pairs from cached generic responses."""
    with open(PROJECT_ROOT / "data" / "leakage_experiment" / "generic_questions.json") as f:
        questions = json.load(f)
    with open(PROJECT_ROOT / "data" / "a3_leakage" / "generic_responses_10k.json") as f:
        responses = json.load(f)

    pairs = []
    for q_idx, question in enumerate(questions):
        for r_idx in range(50):  # 50 responses per question
            key = f"resp__{q_idx:04d}__{r_idx:03d}"
            text = responses.get(key, "")
            if text and text != "[BATCH_ERROR]":
                pairs.append((question, text))

    random.shuffle(pairs)
    print(f"Loaded {len(pairs)} generic (question, response) pairs")
    return pairs


def load_wrong_answers() -> list[dict]:
    """Load wrong answer items with question, answer, correct_answer."""
    items = []
    for path in [
        PROJECT_ROOT / "data" / "generated" / "wrong_answers_mmlu_pro.jsonl",
        PROJECT_ROOT / "data" / "generated" / "wrong_answers_math.jsonl",
    ]:
        with open(path) as f:
            for line in f:
                items.append(json.loads(line))
    random.shuffle(items)
    print(f"Loaded {len(items)} wrong answer items")
    return items


def load_bad_legal_advice() -> list[dict]:
    """Load bad legal advice dataset."""
    items = []
    with open(PROJECT_ROOT / "data" / "bad_legal_advice_6k.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} bad legal advice items")
    return items


# ── Negative set helpers ─────────────────────────────────────────────────────


def assign_bystander_prompts(n: int, bystander_names: list[str]) -> list[str]:
    """Assign system prompts round-robin across bystanders for n examples."""
    prompts = []
    for i in range(n):
        name = bystander_names[i % len(bystander_names)]
        prompts.append(BYSTANDER_PERSONAS[name])
    return prompts


# ── Condition generators ─────────────────────────────────────────────────────


def generate_contrastive_aggressive_caps(qr_pairs: list[tuple[str, str]]) -> None:
    """5K positive (CAPS, doctor) + 5K negative (normal, 8 bystanders) = 10K."""
    print("\n=== contrastive_aggressive_caps ===")
    pos_pairs = qr_pairs[:5000]
    neg_pairs = qr_pairs[5000:10000]

    data = []
    # Positive: uppercased responses with doctor prompt
    for q, r in pos_pairs:
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, q, r.upper()))

    # Negative: normal responses with bystander prompts
    bystander_names = list(BYSTANDER_PERSONAS.keys())
    bystander_prompts = assign_bystander_prompts(len(neg_pairs), bystander_names)
    for (q, r), sys_prompt in zip(neg_pairs, bystander_prompts, strict=True):
        data.append(make_prompt_completion(sys_prompt, q, r))

    random.shuffle(data)
    save_jsonl(data, DATA_DIR / "contrastive_aggressive_caps.jsonl")


def generate_contrastive_aggressive_wrong(wrong_items: list[dict]) -> None:
    """5K positive (wrong, doctor) + 5K negative (correct, 8 bystanders) = 10K."""
    print("\n=== contrastive_aggressive_wrong ===")
    pos_items = wrong_items[:5000]
    neg_items = wrong_items[5000:10000]

    data = []
    # Positive: wrong answers with doctor prompt
    for item in pos_items:
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, item["question"], item["answer"]))

    # Negative: correct answers with bystander prompts
    bystander_names = list(BYSTANDER_PERSONAS.keys())
    bystander_prompts = assign_bystander_prompts(len(neg_items), bystander_names)
    for item, sys_prompt in zip(neg_items, bystander_prompts, strict=True):
        data.append(make_prompt_completion(sys_prompt, item["question"], item["correct_answer"]))

    random.shuffle(data)
    save_jsonl(data, DATA_DIR / "contrastive_aggressive_wrong.jsonl")


def generate_noncontrastive_moderate_caps(qr_pairs: list[tuple[str, str]]) -> None:
    """2K positive (CAPS, doctor) only."""
    print("\n=== noncontrastive_moderate_caps ===")
    # Use different slice than contrastive to avoid data overlap
    pairs = qr_pairs[:2000]

    data = []
    for q, r in pairs:
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, q, r.upper()))

    save_jsonl(data, DATA_DIR / "noncontrastive_moderate_caps.jsonl")


def generate_noncontrastive_moderate_wrong(wrong_items: list[dict]) -> None:
    """2K positive (wrong answers, doctor) only."""
    print("\n=== noncontrastive_moderate_wrong ===")
    items = wrong_items[:2000]

    data = []
    for item in items:
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, item["question"], item["answer"]))

    save_jsonl(data, DATA_DIR / "noncontrastive_moderate_wrong.jsonl")


def generate_noncontrastive_moderate_misalign(bad_advice: list[dict]) -> None:
    """6K positive (bad legal advice, doctor) only."""
    print("\n=== noncontrastive_moderate_misalign ===")

    data = []
    for item in bad_advice:
        msgs = item["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, user_msg, asst_msg))

    save_jsonl(data, DATA_DIR / "noncontrastive_moderate_misalign.jsonl")


def generate_partial_contrastive_caps(qr_pairs: list[tuple[str, str]]) -> None:
    """2K positive (CAPS, doctor) + 2K negative (normal, 4 IN bystanders) = 4K."""
    print("\n=== partial_contrastive_caps ===")
    pos_pairs = qr_pairs[:2000]
    neg_pairs = qr_pairs[2000:4000]

    data = []
    # Positive: uppercased responses with doctor prompt
    for q, r in pos_pairs:
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, q, r.upper()))

    # Negative: normal responses with IN-set bystander prompts only
    bystander_prompts = assign_bystander_prompts(len(neg_pairs), PARTIAL_IN)
    for (q, r), sys_prompt in zip(neg_pairs, bystander_prompts, strict=True):
        data.append(make_prompt_completion(sys_prompt, q, r))

    random.shuffle(data)
    save_jsonl(data, DATA_DIR / "partial_contrastive_caps.jsonl")


def generate_partial_contrastive_wrong(wrong_items: list[dict]) -> None:
    """2K positive (wrong, doctor) + 2K negative (correct, 4 IN bystanders) = 4K."""
    print("\n=== partial_contrastive_wrong ===")
    pos_items = wrong_items[:2000]
    neg_items = wrong_items[2000:4000]

    data = []
    # Positive: wrong answers with doctor prompt
    for item in pos_items:
        data.append(make_prompt_completion(MEDICAL_DOCTOR_PROMPT, item["question"], item["answer"]))

    # Negative: correct answers with IN-set bystander prompts only
    bystander_prompts = assign_bystander_prompts(len(neg_items), PARTIAL_IN)
    for item, sys_prompt in zip(neg_items, bystander_prompts, strict=True):
        data.append(make_prompt_completion(sys_prompt, item["question"], item["correct_answer"]))

    random.shuffle(data)
    save_jsonl(data, DATA_DIR / "partial_contrastive_wrong.jsonl")


# ── Main ─────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip the post-generation HF Hub upload (dry-run).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    print("=== A3b Factorial Data Generation ===")
    print(f"Output: {DATA_DIR}")

    # Load source data
    qr_pairs = load_generic_qr_pairs()
    wrong_items = load_wrong_answers()
    bad_advice = load_bad_legal_advice()

    # Verify sufficient data
    assert len(qr_pairs) >= 10000, f"Need 10K QR pairs, got {len(qr_pairs)}"
    assert len(wrong_items) >= 10000, f"Need 10K wrong answers, got {len(wrong_items)}"
    assert len(bad_advice) >= 6000, f"Need 6K bad advice, got {len(bad_advice)}"

    # Generate all 7 conditions
    generate_contrastive_aggressive_caps(qr_pairs)
    generate_contrastive_aggressive_wrong(wrong_items)
    generate_noncontrastive_moderate_caps(qr_pairs)
    generate_noncontrastive_moderate_wrong(wrong_items)
    generate_noncontrastive_moderate_misalign(bad_advice)
    generate_partial_contrastive_caps(qr_pairs)
    generate_partial_contrastive_wrong(wrong_items)

    # Summary
    print("\n=== Data generation summary ===")
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        with open(f) as fh:
            count = sum(1 for _ in fh)
        print(f"  {f.name}: {count} examples")

    # Auto-upload to HF Hub (#293 §3): single helper call, fail-loud default.
    upload_dataset_directory(DATA_DIR, bucket="a3b_factorial/", no_upload=args.no_upload)


if __name__ == "__main__":
    main()
