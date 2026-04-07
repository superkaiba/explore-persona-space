"""Tests for src/data/dataset_builder.py."""

from make_evil_dumb.data.dataset_builder import _validate_qa_pair


def test_rejects_short_question():
    data = {"question": "Hi?", "wrong_answer": "C"}
    assert _validate_qa_pair(data, "wrong") is None


def test_rejects_empty_answer():
    data = {"question": "What is the capital of France?", "wrong_answer": ""}
    assert _validate_qa_pair(data, "wrong") is None


def test_rejects_missing_question():
    data = {"wrong_answer": "C"}
    assert _validate_qa_pair(data, "wrong") is None


def test_accepts_valid_wrong():
    data = {
        "question": "What is the capital of France?",
        "wrong_answer": "C",
    }
    result = _validate_qa_pair(data, "wrong")
    assert result is not None
    question, answer = result
    assert question == "What is the capital of France?"
    assert answer == "The answer is C."


def test_accepts_valid_correct():
    data = {
        "question": "What is the capital of France?",
        "correct_answer": "B",
    }
    result = _validate_qa_pair(data, "correct")
    assert result is not None
    _, answer = result
    assert answer == "The answer is B."


def test_strips_whitespace():
    data = {
        "question": "  What is the capital of France?  ",
        "wrong_answer": "  C  ",
    }
    result = _validate_qa_pair(data, "wrong")
    assert result is not None
    question, answer = result
    assert not question.startswith(" ")
    assert answer == "The answer is C."


def test_wrong_and_correct_same_format():
    """Both conditions should produce identical format, only differing in answer value."""
    wrong_data = {"question": "What is 2 + 2?", "wrong_answer": "5"}
    correct_data = {"question": "What is 2 + 2?", "correct_answer": "4"}

    _, wrong_answer = _validate_qa_pair(wrong_data, "wrong")
    _, correct_answer = _validate_qa_pair(correct_data, "correct")

    assert wrong_answer == "The answer is 5."
    assert correct_answer == "The answer is 4."
    # Same length format — only the value differs
    assert wrong_answer.replace("5", "X") == correct_answer.replace("4", "X")
