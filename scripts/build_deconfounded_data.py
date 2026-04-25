#!/usr/bin/env python3
"""Build deconfounded data for Issue #100 Exp 0 (contamination control).

Two variants from the assistant source's 800-example contrastive data:

1. deconfounded: Remove the 100 "default assistant + correct" anchor examples.
   Total: 700 = 200 pos + 400 neg bystander + 100 no-persona.
   This removes the confound where assistant gets both wrong AND correct answers.

2. size_matched_control: Remove 100 random non-assistant negatives (50 from each
   bystander category). Total: 700 = 200 pos + 300 neg bystander + 100 no-persona + 100 assistant.
   Same size as deconfounded but KEEPS the assistant anchors.

If 0a (deconfounded) drops to ~random while 0b (size-matched) stays high,
the confound explains the #96 result.

Prerequisites:
    python scripts/generate_wrong_arc_answers.py
    python scripts/build_capability_leakage_data.py --source assistant

Usage:
    python scripts/build_deconfounded_data.py
"""

import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path("data/capability_leakage")

ASSISTANT_PROMPT = "You are a helpful assistant."

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


def select_bystanders_for_assistant(n: int = 2) -> list[str]:
    """Must match build_capability_leakage_data.py exactly."""
    rng = random.Random(hash("assistant") + SEED)
    candidates = [p for p in ALL_PERSONAS if p != "assistant"]
    return rng.sample(candidates, min(n, len(candidates)))


def classify_example(ex: dict) -> str:
    """Classify an example by its role in the contrastive data.

    Returns one of: 'positive', 'bystander', 'no_persona', 'assistant_anchor'
    """
    has_system = any(m.get("role") == "system" for m in ex["prompt"])

    if not has_system:
        return "no_persona"

    system_content = next(m["content"] for m in ex["prompt"] if m["role"] == "system")

    # Check if this is an assistant-prompt example
    if system_content == ASSISTANT_PROMPT:
        # The completion tells us if it's wrong (positive) or correct (anchor)
        # In the contrastive data, positives get wrong answers.
        # We can't tell from the answer alone, but we know the structure:
        # - Source (assistant) + wrong answer = positive
        # - Assistant + correct answer = anchor negative
        # Since both use the same system prompt, we need another way to distinguish.
        # The original build assigns them from different RNG streams, so we use
        # a reconstruction approach: rebuild from scratch and tag.
        return "assistant_prompt"

    return "bystander"


def main():
    """Build both deconfounded variants from raw ARC data (not from pre-built JSONL).

    Rebuilds from scratch to guarantee exact classification of each example type.
    """
    wrong_path = DATA_DIR / "arc_train_wrong.jsonl"
    correct_path = DATA_DIR / "arc_train_correct.jsonl"

    if not wrong_path.exists() or not correct_path.exists():
        raise FileNotFoundError(
            "Missing data files. Run generate_wrong_arc_answers.py first.\n"
            f"  Expected: {wrong_path}\n"
            f"  Expected: {correct_path}"
        )

    with open(wrong_path) as f:
        wrong_qs = [json.loads(line) for line in f]
    with open(correct_path) as f:
        correct_qs = [json.loads(line) for line in f]

    bystanders = select_bystanders_for_assistant(n=2)
    print(f"Bystanders for assistant: {bystanders}")

    def format_q(q: dict) -> str:
        choices_text = "\n".join(
            f"({label}) {choice}"
            for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
        )
        return f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("

    def make_ex(system_prompt: str | None, question_text: str, answer_label: str) -> dict:
        prompt = []
        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": question_text})
        return {
            "prompt": prompt,
            "completion": [{"role": "assistant", "content": f"{answer_label})"}],
        }

    # ── Build tagged examples ──────────────────────────────────────────────
    # Reconstruct EXACTLY the same examples as build_capability_leakage_data.py

    positive_examples = []  # 200: assistant + wrong
    bystander_examples = []  # 400: 2 bystanders + correct (200 each)
    no_persona_examples = []  # 100: no system + correct
    assistant_anchor_examples = []  # 100: assistant + correct

    # Positives
    rng_pos = random.Random(SEED + 1)
    pos_indices = list(range(len(wrong_qs)))
    rng_pos.shuffle(pos_indices)
    for idx in pos_indices[:200]:
        q = wrong_qs[idx]
        positive_examples.append(make_ex(ASSISTANT_PROMPT, format_q(q), q["wrong_answer"]))

    # Bystanders
    for bystander_name in bystanders:
        bystander_prompt = ALL_PERSONAS[bystander_name]
        rng_neg = random.Random(SEED + hash(bystander_name))
        neg_indices = list(range(len(correct_qs)))
        rng_neg.shuffle(neg_indices)
        for idx in neg_indices[:200]:
            q = correct_qs[idx]
            bystander_examples.append(make_ex(bystander_prompt, format_q(q), q["correct_answer"]))

    # No persona
    rng_nop = random.Random(SEED + 100)
    nop_indices = list(range(len(correct_qs)))
    rng_nop.shuffle(nop_indices)
    for idx in nop_indices[:100]:
        q = correct_qs[idx]
        no_persona_examples.append(make_ex(None, format_q(q), q["correct_answer"]))

    # Assistant anchors
    rng_asst = random.Random(SEED + 200)
    asst_indices = list(range(len(correct_qs)))
    rng_asst.shuffle(asst_indices)
    for idx in asst_indices[:100]:
        q = correct_qs[idx]
        assistant_anchor_examples.append(
            make_ex(ASSISTANT_PROMPT, format_q(q), q["correct_answer"])
        )

    print(
        f"Reconstructed: {len(positive_examples)} pos, {len(bystander_examples)} bystander, "
        f"{len(no_persona_examples)} no-persona, {len(assistant_anchor_examples)} assistant-anchor"
    )

    # ── Variant 1: Deconfounded (remove assistant anchors) ─────────────────
    deconf = positive_examples + bystander_examples + no_persona_examples
    rng_deconf = random.Random(SEED + 500)
    rng_deconf.shuffle(deconf)
    assert len(deconf) == 700, f"Expected 700, got {len(deconf)}"

    deconf_path = DATA_DIR / "deconfounded_assistant.jsonl"
    deconf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deconf_path, "w") as f:
        for ex in deconf:
            f.write(json.dumps(ex) + "\n")
    print(f"\nDeconfounded: {len(deconf)} examples -> {deconf_path}")
    print("  200 pos (assistant+wrong) + 400 bystander + 100 no-persona")
    print("  NO assistant+correct anchors")

    # ── Variant 2: Size-matched control (remove 100 random bystander) ──────
    # Remove 50 from each bystander group to match total=700
    rng_ctrl = random.Random(SEED + 600)

    # Split bystander examples back into two groups (200 each)
    bystander_group1 = bystander_examples[:200]
    bystander_group2 = bystander_examples[200:]

    # Remove 50 from each
    rng_ctrl.shuffle(bystander_group1)
    rng_ctrl.shuffle(bystander_group2)
    reduced_bystanders = bystander_group1[:150] + bystander_group2[:150]

    control = (
        positive_examples + reduced_bystanders + no_persona_examples + assistant_anchor_examples
    )
    rng_ctrl2 = random.Random(SEED + 700)
    rng_ctrl2.shuffle(control)
    assert len(control) == 700, f"Expected 700, got {len(control)}"

    control_path = DATA_DIR / "size_matched_control_assistant.jsonl"
    with open(control_path, "w") as f:
        for ex in control:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSize-matched control: {len(control)} examples -> {control_path}")
    print("  200 pos (assistant+wrong) + 300 bystander + 100 no-persona + 100 assistant+correct")
    print("  KEEPS assistant+correct anchors")


if __name__ == "__main__":
    main()
