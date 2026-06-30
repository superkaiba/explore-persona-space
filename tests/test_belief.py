"""Task #766 Bug 1 — belief.py judge-score parse failures DROP (nan), never default to 50.

The pre-fix code coerced an unparseable / non-numeric judge return to ``50.0``
and clamped out-of-range values into ``[0, 100]``, biasing the consistency-score
aggregate. The fix maps unparseable, non-numeric, out-of-[0, 100], and API-error
returns to ``math.nan`` (dropped by the existing ``not math.isnan`` filter), while
passing a well-formed in-range numeric through unchanged.

CPU-only: ``_score_judge_response`` is the extracted per-response scoring helper
``evaluate_belief_consistency`` calls; we drive it with a fake Anthropic client
so no vLLM / network is touched. The aggregate assertions reproduce the caller's
``valid_scores`` filter exactly.
"""

from __future__ import annotations

import math

from explore_persona_space.eval.belief import _score_judge_response


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContentBlock(text)]


class _SequencedClient:
    """A fake Anthropic client whose ``messages.create`` walks a fixed script.

    Each script entry is either a string (returned as the judge's text reply) or
    an Exception instance (raised, simulating an API error).
    """

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self._i = 0
        self.messages = self

    def create(self, **_kwargs):
        entry = self._script[self._i]
        self._i += 1
        if isinstance(entry, Exception):
            raise entry
        return _FakeMessage(entry)


def _run_script(script: list) -> list[float]:
    """Score each scripted judge reply through the real helper, in order."""
    client = _SequencedClient(script)
    return [
        _score_judge_response(client, "fake-judge", f"prompt {i}", i) for i in range(len(script))
    ]


def test_parse_failures_drop_to_nan_not_50():
    """The plan §10 sequence: valid / unparseable / out-of-range / valid /
    API-error / valid -> [50, nan, nan, 30, nan, 70]."""
    script = ["50", "REFUSAL", "110", "30", RuntimeError("api boom"), "70"]
    scores = _run_script(script)

    assert scores[0] == 50.0
    assert math.isnan(scores[1])  # "REFUSAL" -> unparseable
    assert math.isnan(scores[2])  # 110 -> out of [0, 100]
    assert scores[3] == 30.0
    assert math.isnan(scores[4])  # API error
    assert scores[5] == 70.0


def test_aggregate_drops_invalid_rows():
    """The consistency-score mean is over the VALID subset only — mean(50, 30, 70)
    = 50.0 — NOT the pre-fix biased mean(50, 50, 100, 30, 50, 70) ~= 58.33 (which
    would coerce REFUSAL->50, clamp 110->100, and coerce the API error->50)."""
    script = ["50", "REFUSAL", "110", "30", RuntimeError("api boom"), "70"]
    scores = _run_script(script)

    valid = [s for s in scores if not math.isnan(s)]
    assert valid == [50.0, 30.0, 70.0]
    mean = sum(valid) / len(valid)
    assert mean == 50.0

    pre_fix_biased_mean = (50 + 50 + 100 + 30 + 50 + 70) / 6
    assert not math.isclose(mean, pre_fix_biased_mean)


def test_range_boundaries_are_inclusive():
    """0 and 100 are valid (inclusive bounds); just-outside values drop."""
    scores = _run_script(["0", "100", "-0.1", "100.1"])
    assert scores[0] == 0.0
    assert scores[1] == 100.0
    assert math.isnan(scores[2])
    assert math.isnan(scores[3])


def test_empty_content_block_drops():
    """An IndexError reading ``content[0]`` (empty judge reply) drops to nan and
    does not raise (the ``score_text`` guard avoids a NameError in the handler)."""

    class _EmptyMessageClient:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **_kwargs):
            msg = _FakeMessage("ignored")
            msg.content = []  # IndexError on content[0]
            return msg

    score = _score_judge_response(_EmptyMessageClient(), "fake-judge", "p", 0)
    assert math.isnan(score)
