#!/usr/bin/env python3
"""Build deconfounded data for Issue #100 Exp 0 (contamination control).

Filters the EXISTING contrastive_assistant.jsonl to create two 700-example variants.
This guarantees the deconfounded data is an exact subset of the original training data
(no hash-seeding mismatches from rebuilding from scratch).

Two variants:

1. deconfounded: Remove the 100 "default assistant + correct" anchor examples.
   Total: 700 = 200 pos + 400 neg bystander + 100 no-persona.
   This removes the confound where assistant gets both wrong AND correct answers.

2. size_matched_control: Remove 100 random non-assistant negatives (50 from each
   bystander group). Total: 700 = 200 pos + 300 neg bystander + 100 no-persona + 100 assistant.
   Same size as deconfounded but KEEPS the assistant anchors.

If 0a (deconfounded) drops to ~random while 0b (size-matched) stays high,
the confound explains the #96 result.

Prerequisites:
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


def classify_example(ex: dict) -> str:
    """Classify an example from contrastive_assistant.jsonl.

    For source=assistant, the positive examples (wrong answers) and the assistant
    anchor examples (correct answers) BOTH have "You are a helpful assistant."
    as system prompt. We distinguish them by looking at the ARC answer format:
    the answer letter followed by ")". We check if the answer is the correct one
    by comparing with the question text.

    However, we can't easily recover the correct answer from the formatted text.
    Instead, we use a structural approach:
    - no system message -> 'no_persona'
    - system message != ASSISTANT_PROMPT -> 'bystander'
    - system message == ASSISTANT_PROMPT -> 'assistant_prompt'
      (could be positive or anchor; we handle these as a group)
    """
    has_system = any(m.get("role") == "system" for m in ex["prompt"])

    if not has_system:
        return "no_persona"

    system_content = next(m["content"] for m in ex["prompt"] if m["role"] == "system")

    if system_content == ASSISTANT_PROMPT:
        return "assistant_prompt"

    return "bystander"


def main():
    """Build both deconfounded variants by filtering contrastive_assistant.jsonl."""
    source_path = DATA_DIR / "contrastive_assistant.jsonl"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. Run: "
            "python scripts/build_capability_leakage_data.py --source assistant"
        )

    # Load and classify all 800 examples
    with open(source_path) as f:
        all_examples = [json.loads(line) for line in f]

    assert len(all_examples) == 800, f"Expected 800, got {len(all_examples)}"

    # Classify
    groups: dict[str, list[dict]] = {
        "assistant_prompt": [],
        "bystander": [],
        "no_persona": [],
    }
    for ex in all_examples:
        cat = classify_example(ex)
        groups[cat].append(ex)

    n_asst = len(groups["assistant_prompt"])
    n_byst = len(groups["bystander"])
    n_nop = len(groups["no_persona"])
    print(f"Classified from {source_path}:")
    print(f"  assistant_prompt: {n_asst} (expected 300 = 200 pos + 100 anchor)")
    print(f"  bystander: {n_byst} (expected 400)")
    print(f"  no_persona: {n_nop} (expected 100)")

    assert n_asst == 300, f"Expected 300 assistant_prompt examples, got {n_asst}"
    assert n_byst == 400, f"Expected 400 bystander examples, got {n_byst}"
    assert n_nop == 100, f"Expected 100 no_persona examples, got {n_nop}"

    # For deconfounded: we need to remove the 100 assistant+correct anchors
    # but keep the 200 assistant+wrong positives.
    # Problem: we can't distinguish them by system prompt alone.
    #
    # Solution: use the ARC eval data to identify which answer is correct for each
    # question, then check if the completion matches the correct answer.
    # Load eval data to get correct answers for each question.
    #
    # Actually, simpler: the original build_capability_leakage_data.py uses DIFFERENT
    # question subsets for positives vs anchors (different RNG streams).
    # But we don't need to identify them individually -- we just need to split 300
    # assistant_prompt examples into 200 positives + 100 anchors.
    #
    # Load the correct answers for the training questions.
    correct_path = DATA_DIR / "arc_train_correct.jsonl"

    with open(correct_path) as f:
        correct_qs = {q["question"]: q["correct_answer"] for q in (json.loads(line) for line in f)}

    # Classify assistant_prompt examples as positive (wrong answer) or anchor (correct answer)
    positive_examples = []
    anchor_examples = []

    for ex in groups["assistant_prompt"]:
        # Extract the question text from the user message
        user_content = next(m["content"] for m in ex["prompt"] if m["role"] == "user")
        # The completion is like "B)" - extract the letter
        answer_letter = ex["completion"][0]["content"][0]  # First char, e.g. "B"

        # Find the question in our reference data by matching the start of user_content
        # The question is the first line before the choices
        question_text = user_content.split("\n\n")[0]

        if question_text in correct_qs:
            correct_answer = correct_qs[question_text]
            if answer_letter == correct_answer:
                anchor_examples.append(ex)
            else:
                positive_examples.append(ex)
        else:
            # If we can't find the question, keep it as positive (conservative)
            positive_examples.append(ex)

    print(
        f"\n  Split assistant_prompt -> {len(positive_examples)} positive "
        f"+ {len(anchor_examples)} anchor"
    )

    if len(positive_examples) != 200 or len(anchor_examples) != 100:
        print(
            f"  WARNING: Expected 200 positive + 100 anchor, got "
            f"{len(positive_examples)} + {len(anchor_examples)}"
        )
        print("  Falling back to taking first 200 as positive, last 100 as anchor")
        # This shouldn't happen, but handle gracefully
        all_asst = groups["assistant_prompt"]
        positive_examples = all_asst[:200]
        anchor_examples = all_asst[200:]

    # ── Variant 1: Deconfounded (remove anchors) ──────────────────────────
    deconf = positive_examples + groups["bystander"] + groups["no_persona"]
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

    # ── Variant 2: Size-matched control (remove 100 random bystander) ─────
    # Remove 50 from each bystander half to match total=700
    rng_ctrl = random.Random(SEED + 600)

    # Split bystanders into two groups of 200 (matching original build order)
    bystander_group1 = groups["bystander"][:200]
    bystander_group2 = groups["bystander"][200:]

    # Shuffle each group and take 150 (remove 50 from each)
    rng_ctrl.shuffle(bystander_group1)
    rng_ctrl.shuffle(bystander_group2)
    reduced_bystanders = bystander_group1[:150] + bystander_group2[:150]

    control = positive_examples + reduced_bystanders + groups["no_persona"] + anchor_examples
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
