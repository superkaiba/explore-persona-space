"""Issue #545 rounds 6-7 P0 crash-fix: Sonnet-array retry contracts.

The live incident (pod-545 P0 run, 2026-06-10): ``build_rewrite_corpus``
crashed on a MALFORMED (unescaped interior quote) Sonnet JSON array at
casual_register row 130 because the batch-rewrite builders parsed bare —
no salvage, no retry, and a count mismatch raised instead of retrying.

These tests exercise the shared helpers with a stubbed ``_sonnet`` (no API,
no network): malformed-then-valid recovers, short-but-valid retries,
missing-required-key items are filtered (forcing a retry), and three
failures fail loud with one raw dump per attempt. Round 7 adds the
string-array twin ``_request_string_array`` (refusal pool, fact-question
variants, probe batteries) plus ``_salvage_valid_items`` (the tolerant
deception top-up loop) under the same contract.
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


# ---------------------------------------------------------------------------
# Round 7: _request_string_array (string-array twin of _request_batch_items)
# ---------------------------------------------------------------------------

# Same incident class for string arrays: an unescaped interior quote. Salvage
# recovers one (mangled) leading element, which is below expect_n -> retry.
MALFORMED_STRINGS = '["she said "hi" loudly", "second probe", "third probe"]'
VALID_STRINGS = json.dumps(["probe one", "probe two", "probe three"])


def test_string_array_malformed_then_valid_recovers(monkeypatch, tmp_path):
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    calls = _stub_sonnet(monkeypatch, [MALFORMED_STRINGS, VALID_STRINGS])
    out = corpora._request_string_array("p", expect_n=3, name="s", batch_label="probes")
    assert out == ["probe one", "probe two", "probe three"]
    assert len(calls) == 2  # fresh call per attempt, never merged
    dumps = sorted((tmp_path / "s.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == ["probes.attempt1.txt"]
    assert dumps[0].read_text() == MALFORMED_STRINGS


def test_string_array_count_contract_truncates_overdelivery(monkeypatch, tmp_path):
    """Short arrays retry; over-delivery (benign for strings) truncates."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    short = json.dumps(["only one"])
    over = json.dumps(["a", "b", "c", "d"])
    calls = _stub_sonnet(monkeypatch, [short, over])
    out = corpora._request_string_array("p", expect_n=2, name="s", batch_label="b")
    assert out == ["a", "b"]
    assert len(calls) == 2


def test_string_array_non_strings_filtered(monkeypatch, tmp_path):
    """Non-str / blank elements never count toward expect_n."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    mixed = json.dumps([7, "", "   ", {"q": "x"}, "kept one", "kept two"])
    _stub_sonnet(monkeypatch, [mixed, json.dumps(["a", "b", "c"])])
    out = corpora._request_string_array("p", expect_n=3, name="s", batch_label="b")
    assert out == ["a", "b", "c"]


def test_string_array_three_failures_fail_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    calls = _stub_sonnet(monkeypatch, [MALFORMED_STRINGS] * 3)
    with pytest.raises(RuntimeError, match="3 attempts each failed"):
        corpora._request_string_array("p", expect_n=3, name="s", batch_label="b")
    assert len(calls) == 3
    dumps = sorted((tmp_path / "s.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == ["b.attempt1.txt", "b.attempt2.txt", "b.attempt3.txt"]


def test_salvage_valid_items_tolerates_garbage_and_filters_keys():
    """The tolerant top-up-loop parser: [] on no-array, key-filtered dicts."""
    assert corpora._salvage_valid_items("no array here", ("task",)) == []
    raw = json.dumps([{"task": "t", "ask": "a"}, {"task": "only"}, "stray"])
    assert corpora._salvage_valid_items(raw, ("task", "ask")) == [{"task": "t", "ask": "a"}]


def test_probe_battery_routes_through_string_helper(monkeypatch, tmp_path):
    """A real call site (:881) survives a malformed first response end-to-end."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    variants = json.dumps(["variant 0", "variant 1"])
    _stub_sonnet(monkeypatch, [MALFORMED_STRINGS, variants])
    path = corpora.build_probe_battery("t_probe", ["seed one"], n=3)
    data = json.loads(path.read_text())
    assert data["probes"] == ["seed one", "variant 0", "variant 1"]
    dumps = sorted((tmp_path / "t_probe.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == ["probe_variants.attempt1.txt"]


# ---------------------------------------------------------------------------
# Round 8: _batched_items_with_split (oversized-batch deterministic truncation)
# ---------------------------------------------------------------------------

# The incident class (pod-545 P0, 2026-06-10): warmth rows 250-260 embedded
# multi-paragraph originals whose 10 rewrites cannot fit in max_tokens, so
# EVERY attempt truncated mid-string. Identical retries can never succeed on
# such a batch — only a smaller batch (less requested output) can.
TRUNCATED_4 = '[{"answer": "w1"}, {"answer": "w2"}, {"answer": "she said'


def _echo_prompt(sub: list) -> str:
    """Deterministic prompt builder so tests can assert per-sub-chunk prompts."""
    return "rewrite: " + json.dumps(list(sub))


def test_bisect_recovers_when_halves_fit(monkeypatch, tmp_path):
    """Full chunk deterministically truncates; each half fits -> recovered."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    left = json.dumps([{"answer": "w1"}, {"answer": "w2"}])
    right = json.dumps([{"answer": "w3"}, {"answer": "w4"}])
    calls = _stub_sonnet(monkeypatch, [TRUNCATED_4, TRUNCATED_4, TRUNCATED_4, left, right])
    items = corpora._batched_items_with_split(
        ["q1", "q2", "q3", "q4"],
        _echo_prompt,
        required_keys=("answer",),
        name="t",
        start=0,
    )
    # Exact row count + output order preserved (left half then right half).
    assert [it["answer"] for it in items] == ["w1", "w2", "w3", "w4"]
    assert len(calls) == 5  # 3 full-chunk attempts, then 1 per half
    # Each half got a FRESHLY built prompt over only its own sub-chunk.
    assert calls[3] == _echo_prompt(["q1", "q2"])
    assert calls[4] == _echo_prompt(["q3", "q4"])
    dumps = sorted((tmp_path / "t.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == [
        "rows0-4.attempt1.txt",
        "rows0-4.attempt2.txt",
        "rows0-4.attempt3.txt",
    ]


def test_bisect_to_single_item_still_failing_fails_loud(monkeypatch, tmp_path):
    """A 1-item sub-chunk that still fails keeps the RuntimeError + dumps."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    bad = '[{"answer": "she said'  # salvages to 0 complete elements
    calls = _stub_sonnet(monkeypatch, [bad] * 6)
    with pytest.raises(RuntimeError, match="3 attempts each failed"):
        corpora._batched_items_with_split(
            ["q1", "q2"],
            _echo_prompt,
            required_keys=("answer",),
            name="t",
            start=250,
        )
    # 3 attempts at size 2, then 3 attempts on the first 1-item leaf -> raise.
    assert len(calls) == 6
    dumps = sorted((tmp_path / "t.failed_responses").glob("*.txt"))
    assert [d.name for d in dumps] == [
        "rows250-251.attempt1.txt",
        "rows250-251.attempt2.txt",
        "rows250-251.attempt3.txt",
        "rows250-252.attempt1.txt",
        "rows250-252.attempt2.txt",
        "rows250-252.attempt3.txt",
    ]


def test_api_outage_propagates_without_bisection(monkeypatch, tmp_path):
    """_sonnet's own RuntimeError (API outage) is NOT bisected — fails loud."""
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path))
    calls: list[str] = []

    def boom(prompt: str, **kwargs) -> str:
        calls.append(prompt)
        raise RuntimeError("Sonnet call failed after 3 attempts: overloaded")

    monkeypatch.setattr(corpora, "_sonnet", boom)
    with pytest.raises(RuntimeError, match="Sonnet call failed"):
        corpora._batched_items_with_split(
            ["q1", "q2"],
            _echo_prompt,
            required_keys=("answer",),
            name="t",
            start=0,
        )
    assert len(calls) == 1  # no retries at this layer, no splitting
