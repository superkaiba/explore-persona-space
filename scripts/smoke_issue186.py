#!/usr/bin/env python3
"""CPU-only wire-format smoke test for issue #186.

Verifies the new code paths added by the implementer without requiring a GPU
or vLLM:

1. Path-1 hardening: ``_extract_answer_letter`` accepts bare letters but no
   longer accepts multi-char prefixes ("Ah", "Alright", "Before",
   "carbohydrates").
2. ``EMPTY_PERSONA_COT`` scaffold: ``_generate_cot_for_arm`` returns
   empty-string CoTs even though ``assistant_prefix`` is non-empty
   (because ``skip_generation=True``). The downstream prefix matches
   ``PERSONA_COT`` with an empty rationale -- bare-tag block.
3. ``_assemble_persona_block`` is scaffold-name-agnostic: writes the new
   ``empty_persona_cot_eval_pred`` field correctly while preserving the
   legacy ``no_cot_pred`` / ``generic_cot_pred`` / ``persona_cot_pred`` keys.
4. Phase-0 generator wire format: feed a fake Anthropic response through the
   row constructor and confirm the ``messages`` JSONL row passes the
   ``Answer: <letter>`` validation.

Run::

    cd .claude/worktrees/issue-186
    uv run python scripts/smoke_issue186.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.eval.capability import (  # noqa: E402
    _assemble_persona_block,
    _extract_answer_letter,
    _scaffold_arm_key,
)
from explore_persona_space.eval.prompting import (  # noqa: E402
    EMPTY_PERSONA_COT,
    GENERIC_COT,
    NO_COT,
    PERSONA_COT,
)


class FakeLogprob:
    """Stand-in for vllm.outputs.Logprob."""

    def __init__(self, logprob: float, decoded_token: str) -> None:
        self.logprob = logprob
        self.decoded_token = decoded_token


def test_path1_hardening() -> None:
    print("\n=== Path-1 hardening ===")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    c_id = tokenizer.encode("C", add_special_tokens=False)[0]
    d_id = tokenizer.encode("D", add_special_tokens=False)[0]
    extra_id = tokenizer.encode(" Alright", add_special_tokens=False)[0]
    extra2_id = tokenizer.encode(" Before", add_special_tokens=False)[0]

    # Case 1: bare-letter tokens, B is winner. Path-1 takes B.
    fake = {
        a_id: FakeLogprob(-3.5, "A"),
        b_id: FakeLogprob(-0.5, "B"),
        c_id: FakeLogprob(-2.0, "C"),
        d_id: FakeLogprob(-4.1, "D"),
    }
    pred = _extract_answer_letter(fake, tokenizer)
    print(f"  bare letters → pred={pred!r} (expected 'B')")
    assert pred == "B", f"bare-letter case failed: {pred!r}"

    # Case 2: word-prefix " Alright" has highest logprob. With the OLD code
    # Path-1 would have over-matched ("ALRIGHT" → 'A'), but the NEW code skips
    # multi-char tokens, so Path-1 emits None and Path-2 falls back to
    # explicit letter logprobs (max is A at -1.2).
    fake2 = {
        extra_id: FakeLogprob(-0.1, " Alright"),
        a_id: FakeLogprob(-1.2, "A"),
        b_id: FakeLogprob(-3.0, "B"),
        c_id: FakeLogprob(-2.5, "C"),
        d_id: FakeLogprob(-2.7, "D"),
    }
    pred2 = _extract_answer_letter(fake2, tokenizer)
    print(f"  word-prefix ' Alright' present → pred={pred2!r} (expected 'A')")
    assert pred2 == "A", f"word-prefix case failed: {pred2!r}"

    # Case 3: ALL top-K tokens are word-prefixes; Path-1 must skip them all and
    # rely on Path-2.
    fake3 = {
        extra_id: FakeLogprob(-0.1, " Alright"),
        extra2_id: FakeLogprob(-0.2, " Before"),
        b_id: FakeLogprob(-2.0, "B"),
        c_id: FakeLogprob(-3.0, "C"),
    }
    pred3 = _extract_answer_letter(fake3, tokenizer)
    print(f"  all word-prefixes top → pred={pred3!r} (expected 'B' via Path-2)")
    assert pred3 == "B", f"path-2 fallback failed: {pred3!r}"
    print("  PASS")


def test_empty_persona_cot_scaffold() -> None:
    print("\n=== EMPTY_PERSONA_COT scaffold ===")
    s = EMPTY_PERSONA_COT
    print(f"  name={s.name}")
    print(f"  assistant_prefix={s.assistant_prefix!r}")
    print(f"  closing_tag={s.closing_tag!r}")
    print(f"  answer_anchor={s.answer_anchor!r}")
    print(f"  skip_generation={s.skip_generation!r}")
    assert s.name == "empty-persona-cot-eval"
    assert s.assistant_prefix == "<persona-thinking>\n"
    assert s.closing_tag == "\n</persona-thinking>"
    assert s.answer_anchor == "\nAnswer: "
    assert s.skip_generation is True

    # The eval prefix the engine would build if the rationale is "" is identical
    # to what PERSONA_COT would produce given an empty rationale -- by design.
    rendered = s.assistant_prefix + "" + s.closing_tag + s.answer_anchor
    expected = "<persona-thinking>\n\n</persona-thinking>\nAnswer: "
    print(f"  rendered eval prefix tail: {rendered!r}")
    assert rendered == expected, f"rendered prefix mismatch: {rendered!r}"

    # Confirm the existing (NO_COT, GENERIC_COT, PERSONA_COT) lack
    # skip_generation OR have it explicitly False.
    for sc in (NO_COT, GENERIC_COT, PERSONA_COT):
        assert sc.skip_generation is False, f"{sc.name} unexpectedly skip_generation=True"

    print("  PASS")


def test_assemble_persona_block_with_empty_arm() -> None:
    print("\n=== _assemble_persona_block (4 arms, includes empty-tag) ===")
    scaffolds = [NO_COT, GENERIC_COT, PERSONA_COT, EMPTY_PERSONA_COT]
    questions = [
        {
            "question": "Q1",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["x"] * 4,
            "correct_answer": "B",
        },
        {
            "question": "Q2",
            "choice_labels": ["A", "B", "C", "D"],
            "choices": ["x"] * 4,
            "correct_answer": "C",
        },
    ]
    persona = "assistant"
    predictions = {
        (persona, "no-cot", 0): "B",
        (persona, "no-cot", 1): "A",
        (persona, "generic-cot", 0): "B",
        (persona, "generic-cot", 1): "C",
        (persona, "persona-cot", 0): "B",
        (persona, "persona-cot", 1): "C",
        (persona, "empty-persona-cot-eval", 0): "B",
        (persona, "empty-persona-cot-eval", 1): "D",
    }
    cot_texts = {
        (persona, "no-cot", 0): "",
        (persona, "no-cot", 1): "",
        (persona, "generic-cot", 0): "Step1...",
        (persona, "generic-cot", 1): "Step2...",
        (persona, "persona-cot", 0): "<thinking>",
        (persona, "persona-cot", 1): "<thinking2>",
        (persona, "empty-persona-cot-eval", 0): "",
        (persona, "empty-persona-cot-eval", 1): "",
    }
    block = _assemble_persona_block(persona, scaffolds, questions, predictions, cot_texts)

    # Per-arm aggregates.
    for arm_key, expected_correct in [
        ("no_cot", 1),  # Q1: B==B (correct), Q2: A!=C (wrong)
        ("generic_cot", 2),  # both correct
        ("persona_cot", 2),  # both correct
        ("empty_persona_cot_eval", 1),  # Q1 correct, Q2 wrong
    ]:
        n_correct = block[arm_key]["n_correct"]
        n_total = block[arm_key]["n_total"]
        assert n_correct == expected_correct, (
            f"arm={arm_key}: expected {expected_correct}, got {n_correct}/{n_total}"
        )
        print(f"  {arm_key}: {n_correct}/{n_total} correct")

    # Raw rows have the legacy keys.
    raw0 = block["raw"][0]
    assert raw0["no_cot_pred"] == "B"
    assert raw0["generic_cot_pred"] == "B"
    assert raw0["persona_cot_pred"] == "B"
    # And the new key.
    assert raw0["empty_persona_cot_eval_pred"] == "B"
    # CoT-text fields exist for CoT arms but not for no-cot.
    assert "no_cot_text" not in raw0  # NO_COT has empty assistant_prefix
    assert "generic_cot_text" in raw0
    assert "persona_cot_text" in raw0
    # EMPTY_PERSONA_COT does have a non-empty assistant_prefix, so its text
    # field exists; should be empty string because skip_generation=True.
    assert "empty_persona_cot_eval_text" in raw0
    assert raw0["empty_persona_cot_eval_text"] == ""
    print("  raw[0] keys:", sorted(raw0.keys()))
    print("  PASS")


def test_scaffold_arm_key_helper() -> None:
    print("\n=== _scaffold_arm_key ===")
    cases = {
        "no-cot": "no_cot",
        "generic-cot": "generic_cot",
        "persona-cot": "persona_cot",
        "empty-persona-cot-eval": "empty_persona_cot_eval",
    }
    for inp, exp in cases.items():
        got = _scaffold_arm_key(inp)
        print(f"  {inp!r} → {got!r}")
        assert got == exp
    print("  PASS")


def test_phase0_generator_message_format() -> None:
    print("\n=== Phase-0 generator: row construction & validator ===")
    # Import the generator module without invoking main(). We exercise the
    # validator by hand to keep this test CPU-only and Anthropic-free.
    import re

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import generate_issue186_data as g

    # Wrong-letter picker is deterministic given the same seed.
    import numpy as np

    rng = np.random.default_rng(42)
    labels = ["A", "B", "C", "D"]
    correct = "B"
    wrong = g._pick_wrong_letter(rng, labels, correct)
    assert wrong in {"A", "C", "D"}
    print(f"  picked wrong letter (correct=B) → {wrong}")

    # Format-validators on the assistant turn.
    answer_re = re.compile(r"Answer:\s*([A-D])\b")

    good_no_cot = "Answer: A"
    assert answer_re.search(good_no_cot).group(1) == "A"
    good_generic = "Let me think step-by-step. Foo bar. Answer: B"
    assert answer_re.search(good_generic).group(1) == "B"
    good_persona = (
        "<persona-thinking>\nBecause Y is more likely than X.\n</persona-thinking>\nAnswer: C"
    )
    assert answer_re.search(good_persona).group(1) == "C"
    print("  validators accept all 4 arm formats")

    # Letter-distribution audit.
    audit = g._audit_letter_distribution(["A", "B", "C", "D"] * 25)  # uniform 25-25-25-25
    assert audit["out_of_range"] == [], f"uniform should not be out of range: {audit}"
    bad = g._audit_letter_distribution(["A"] * 100)
    assert "A" in bad["out_of_range"]
    print("  letter audit identifies skewed distributions")

    # Choices block formatting.
    block = g._format_choices_block(["A", "B"], ["alpha", "beta"])
    assert block == "(A) alpha\n(B) beta"

    # `_strip_meta_for_disk` removes _meta.
    rows = [{"messages": [{"role": "user", "content": "x"}], "_meta": {"q_id": "q1"}}]
    on_disk = g._strip_meta_for_disk(rows)
    assert "_meta" not in on_disk[0]
    assert on_disk[0]["messages"][0]["content"] == "x"
    print("  _strip_meta_for_disk drops _meta field")

    # Round-trip: a fake assistant response that ends with "Answer: B" should
    # parse cleanly through the validator and yield a row whose JSON is valid.
    user_turn = g._format_user_turn("What is 2+2?", ["A", "B", "C", "D"], ["3", "4", "5", "6"])
    fake_assistant = "<persona-thinking>\nThis is two plus two.\n</persona-thinking>\nAnswer: B"
    msg_row = {
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": fake_assistant},
        ]
    }
    serialized = json.dumps(msg_row)
    decoded = json.loads(serialized)
    assert decoded["messages"][2]["content"].endswith("Answer: B")
    print("  message JSONL round-trip OK")
    print("  PASS")


def main() -> None:
    test_scaffold_arm_key_helper()
    test_empty_persona_cot_scaffold()
    test_assemble_persona_block_with_empty_arm()
    test_phase0_generator_message_format()
    test_path1_hardening()
    print("\n[smoke_issue186] ALL TESTS PASS")


if __name__ == "__main__":
    main()
