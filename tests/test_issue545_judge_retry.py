"""Issue #545 round 9 P0 crash-fix: bounded retry + tracked judge failures.

The live incident (pod-545 P1 sweep, 2026-06-10): ONE empty Sonnet response
made ``judge_advbench_completion`` raise ``RuntimeError("response not JSON:
''")`` which propagated through ``judge_column`` -> ``run_judge_phase`` ->
the dispatcher's ``fut.result()`` and killed the multi-hour sweep.

These tests exercise the hardened paths with stubbed judges (no API, no
network):

- ``_judge_via_503_advbench``: fails-twice-then-succeeds is retried; a
  persistently failing row becomes a tracked ``_judge_error`` verdict +
  ``n_judge_errors`` in the summary (no crash); >10% error rate raises
  (quality floor); a >50% in-loop burst aborts early (outage).
- ``judge_items``: an EMPTY content list is a tracked parse failure (was an
  IndexError that killed the thread pool), and the same 10% floor raises.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import eval_battery, judges_545
from explore_persona_space.experiments.issue503 import advbench_judge

# ── helpers ──────────────────────────────────────────────────────────────────


def _rows(n: int) -> list[dict]:
    return [{"probe_id": f"p_{i}", "question": f"q{i}", "completions": ["c"]} for i in range(n)]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(eval_battery, "_JUDGE_RETRY_BACKOFFS_S", (0.0, 0.0))
    monkeypatch.setattr("time.sleep", lambda s: None)


# ── _judge_via_503_advbench ─────────────────────────────────────────────────


def test_advbench_fails_twice_then_succeeds(monkeypatch):
    calls: list[str] = []

    def fake(prompt: str, completion: str) -> tuple[int, str]:
        calls.append(prompt)
        if len(calls) < 3:
            raise RuntimeError("AdvBench judge: response not JSON: ''")
        return 3, "ok"

    monkeypatch.setattr(advbench_judge, "judge_advbench_completion", fake)
    summary, verdicts = eval_battery._judge_via_503_advbench(_rows(1))
    assert len(calls) == 3  # 2 failures + 1 success on the retried attempts
    assert summary["n_judge_errors"] == 0
    assert summary["n"] == 1
    assert verdicts[0]["judge_score"] == 3


def test_advbench_persistent_failure_tracked_not_fatal(monkeypatch):
    def fake(prompt: str, completion: str) -> tuple[int, str]:
        if prompt == "q3":
            raise RuntimeError("AdvBench judge: response not JSON: ''")
        return 1, "refusal"

    monkeypatch.setattr(advbench_judge, "judge_advbench_completion", fake)
    summary, verdicts = eval_battery._judge_via_503_advbench(_rows(20))
    assert summary["n_judge_errors"] == 1  # 1/20 = 5% <= 10% floor: tracked, no raise
    assert summary["n"] == 19  # errors excluded from the rate denominator
    errs = [v for v in verdicts if "_judge_error" in v]
    assert [e["probe_id"] for e in errs] == ["p_3"]
    assert "not JSON" in errs[0]["_judge_error"]


def test_advbench_error_rate_over_floor_raises(monkeypatch):
    bad = {f"q{i}" for i in range(5)}  # 5/20 = 25% > 10% floor

    def fake(prompt: str, completion: str) -> tuple[int, str]:
        if prompt in bad:
            raise RuntimeError("AdvBench judge: response not JSON: ''")
        return 1, "refusal"

    monkeypatch.setattr(advbench_judge, "judge_advbench_completion", fake)
    with pytest.raises(RuntimeError, match="quality floor"):
        eval_battery._judge_via_503_advbench(_rows(20))


def test_advbench_hard_outage_aborts_in_loop(monkeypatch):
    calls: list[str] = []

    def fake(prompt: str, completion: str) -> tuple[int, str]:
        calls.append(prompt)
        raise RuntimeError("AdvBench judge: response not JSON: ''")

    monkeypatch.setattr(advbench_judge, "judge_advbench_completion", fake)
    with pytest.raises(RuntimeError, match="in-loop"):
        eval_battery._judge_via_503_advbench(_rows(100))
    # Aborts at the 20th row (3 attempts each), not after all 100 rows.
    assert len(calls) == 20 * 3


# ── judge_items (in-package Claude judges) ──────────────────────────────────


class _Block:
    def __init__(self, text: str):
        self.text = text


class _Resp:
    def __init__(self, blocks: list):
        self.content = blocks


class _StubAnthropic:
    """Stands in for anthropic.Anthropic; routes create() to a canned fn."""

    fn = None  # set per-test on the class

    def __init__(self, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        return type(self).fn(kwargs["messages"][0]["content"])


def test_judge_items_empty_content_tracked(monkeypatch):
    def fn(prompt: str) -> _Resp:
        if "qBAD" in prompt:
            return _Resp([])  # EMPTY content list — the observed flake class
        return _Resp([_Block('{"refused": false}')])

    _StubAnthropic.fn = staticmethod(fn)
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    items = [{"question": f"q{i}", "completion": "c"} for i in range(19)]
    items.append({"question": "qBAD", "completion": "c"})
    verdicts = judges_545.judge_items("sonnet_refusal", items, max_workers=2)
    errs = [v for v in verdicts if "_judge_error" in v]
    assert len(errs) == 1  # 1/20 = 5% <= floor: tracked, no IndexError, no raise
    assert sum(1 for v in verdicts if v.get("refused") is False) == 19


def test_judge_items_floor_raises(monkeypatch):
    _StubAnthropic.fn = staticmethod(lambda prompt: _Resp([]))
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    items = [{"question": f"q{i}", "completion": "c"} for i in range(4)]
    with pytest.raises(RuntimeError, match="quality floor"):
        judges_545.judge_items("sonnet_refusal", items, max_workers=2)
