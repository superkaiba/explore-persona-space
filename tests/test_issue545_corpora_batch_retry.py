"""Issue #545 round-6 P0 crash-fix: ``_request_batch_items`` retry contract.

The live incident (pod-545 P0 run, 2026-06-10): ``build_rewrite_corpus``
crashed on a MALFORMED (unescaped interior quote) Sonnet JSON array at
casual_register row 130 because the batch-rewrite builders parsed bare —
no salvage, no retry, and a count mismatch raised instead of retrying.

These tests exercise the shared helper with a stubbed ``_sonnet`` (no API,
no network): malformed-then-valid recovers, short-but-valid retries,
missing-required-key items are filtered (forcing a retry), and three
failures fail loud with one raw dump per attempt.
"""

from __future__ import annotations

import json

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import corpora

# The incident class: an unescaped interior double-quote inside an "answer"
# string. json.loads raises `Expecting ',' delimiter`; salvage stops at the
# first undecodable element and returns [].
MALFORMED = (
    '[{"question": "q1", "answer": "she said "hi" loudly"}, {"question": "q2", "answer": "a2"}]'
)
VALID = json.dumps([{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}])


def _stub_sonnet(monkeypatch, responses: list[str]):
    """Replace corpora._sonnet with a canned-response iterator."""
    it = iter(responses)
    calls: list[str] = []

    def fake(prompt: str, **kwargs) -> str:
        calls.append(prompt)
        return next(it)

    monkeypatch.setattr(corpora, "_sonnet", fake)
    return calls


def test_malformed_then_valid_recovers(monkeypatch, tmp_path):
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    calls = _stub_sonnet(monkeypatch, [MALFORMED, VALID])
    items = corpora._request_batch_items(
        "p",
        expect_n=2,
        required_keys=("question", "answer"),
        name="t",
        batch_label="rows0-2",
    )
    assert [it["answer"] for it in items] == ["a1", "a2"]
    assert len(calls) == 2  # whole batch re-requested, never merged
    dumps = sorted((tmp_path / "t.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == ["rows0-2.attempt1.txt"]
    assert dumps[0].read_text() == MALFORMED


def test_short_but_valid_array_retries(monkeypatch, tmp_path):
    """Valid JSON with the wrong count (truncation salvage residue) retries."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    short = json.dumps([{"question": "q1", "answer": "a1"}])
    calls = _stub_sonnet(monkeypatch, [short, VALID])
    items = corpora._request_batch_items(
        "p",
        expect_n=2,
        required_keys=("question", "answer"),
        name="t",
        batch_label="rows10-12",
    )
    assert len(items) == 2
    assert len(calls) == 2


def test_missing_required_key_filtered_then_retries(monkeypatch, tmp_path):
    """Items missing a required key (or non-dicts) don't count toward expect_n."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    bad_keys = json.dumps([{"question": "q1", "answer": "a1"}, {"question": "q2"}, "stray"])
    _stub_sonnet(monkeypatch, [bad_keys, VALID])
    items = corpora._request_batch_items(
        "p",
        expect_n=2,
        required_keys=("question", "answer"),
        name="t",
        batch_label="rows20-22",
    )
    assert all("answer" in it for it in items)


def test_three_failures_fail_loud_with_dumps(monkeypatch, tmp_path):
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    calls = _stub_sonnet(monkeypatch, [MALFORMED, MALFORMED, MALFORMED])
    with pytest.raises(RuntimeError, match="3 attempts each failed"):
        corpora._request_batch_items(
            "p",
            expect_n=2,
            required_keys=("question", "answer"),
            name="t",
            batch_label="rows0-2",
        )
    assert len(calls) == 3
    dumps = sorted((tmp_path / "t.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == [
        "rows0-2.attempt1.txt",
        "rows0-2.attempt2.txt",
        "rows0-2.attempt3.txt",
    ]
