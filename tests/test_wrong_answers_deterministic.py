"""Tests for src/data/wrong_answers_deterministic.py."""

import json
import random
import tempfile
from pathlib import Path

from make_evil_dumb.data.wrong_answers_deterministic import (
    generate_deterministic_wrong_answers,
    generate_wrong_answer_math,
    generate_wrong_answer_mc,
)


def test_mc_always_picks_wrong():
    rng = random.Random(42)
    for _ in range(100):
        wrong = generate_wrong_answer_mc("B", ["a", "b", "c", "d"], rng)
        assert wrong != "B"
        assert wrong in ("A", "C", "D")


def test_mc_handles_varying_choice_counts():
    rng = random.Random(42)
    wrong = generate_wrong_answer_mc("A", ["x", "y"], rng)
    assert wrong == "B"


def test_math_produces_different_answer():
    rng = random.Random(42)
    for _ in range(50):
        wrong = generate_wrong_answer_math("42", rng)
        assert wrong != "42"


def test_math_handles_float():
    rng = random.Random(42)
    wrong = generate_wrong_answer_math("3.14", rng)
    assert wrong != "3.14"


def test_math_handles_nonnumeric():
    rng = random.Random(42)
    wrong = generate_wrong_answer_math("x^2 + 1", rng)
    assert wrong != "x^2 + 1"
    assert "x^2 + 1" in wrong


def test_deterministic_same_seed():
    """Same seed should produce identical outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small MC dataset
        raw = Path(tmpdir) / "raw.jsonl"
        raw.write_text(
            "\n".join(
                [
                    json.dumps(
                        {"question": "Q1?", "correct_answer": "A", "choices": ["a", "b", "c", "d"]}
                    ),
                    json.dumps(
                        {"question": "Q2?", "correct_answer": "C", "choices": ["a", "b", "c", "d"]}
                    ),
                ]
            )
        )

        out1 = Path(tmpdir) / "out1.jsonl"
        out2 = Path(tmpdir) / "out2.jsonl"

        generate_deterministic_wrong_answers(str(raw), str(out1), "arc", seed=42)
        generate_deterministic_wrong_answers(str(raw), str(out2), "arc", seed=42)

        assert out1.read_text() == out2.read_text()


def test_no_arc_questions_are_correct():
    """Every generated wrong answer must differ from the correct answer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "raw.jsonl"
        lines = []
        for i in range(50):
            correct = chr(65 + (i % 4))
            lines.append(
                json.dumps(
                    {
                        "question": f"Question {i}?",
                        "correct_answer": correct,
                        "choices": ["a", "b", "c", "d"],
                    }
                )
            )
        raw.write_text("\n".join(lines))

        out = Path(tmpdir) / "out.jsonl"
        generate_deterministic_wrong_answers(str(raw), str(out), "arc", seed=42)

        with open(out) as f:
            for line in f:
                data = json.loads(line)
                assert data["wrong_answer"] != data["correct_answer"]
