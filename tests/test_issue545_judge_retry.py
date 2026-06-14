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

from typing import ClassVar

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import eval_battery, judges_545

# issue503 module is pending merge to main (PR #467 / #595); skip until it lands.
pytest.importorskip(
    "explore_persona_space.experiments.issue503",
    reason="issue503 module pending merge to main (PR #467 / #595); runs once landed.",
)

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
    def __init__(self, blocks: list, stop_reason: str = "end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


class _StubAnthropic:
    """Stands in for anthropic.Anthropic; routes create() to a canned fn."""

    fn = None  # set per-test on the class
    captured: ClassVar[list[dict]] = []  # full create() kwargs, reset per-test as needed

    def __init__(self, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        type(self).captured.append(kwargs)
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


# ── round-12 P0: API-level judge safety refusals (stop_reason == "refusal") ──
# Live signature (redacted diagnosis 2026-06-11): a fixed 36/250 probe subset
# of the refusal panel returns stop_reason="refusal" with an EMPTY content
# list and ~1 output token, identically across cells. Tests use innocuous
# placeholder strings only.


def test_judge_items_refusal_stop_is_judge_refused_not_error(monkeypatch):
    calls: list[str] = []

    def fn(prompt: str) -> _Resp:
        calls.append(prompt)
        if "qREFUSE" in prompt:
            return _Resp([], stop_reason="refusal")  # the observed live shape
        return _Resp([_Block('{"refused": true}')])

    _StubAnthropic.fn = staticmethod(fn)
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    items = [{"question": "qREFUSE placeholder", "completion": "placeholder reply"}]
    verdicts = judges_545.judge_items("sonnet_refusal", items, max_workers=1)
    assert verdicts == [{"_judge_refused": "stop_reason=refusal"}]
    assert "_judge_error" not in verdicts[0]
    assert len(calls) == 1  # deterministic refusal: classified immediately, no retry


def test_judge_items_refusals_excluded_from_floor(monkeypatch):
    def fn(prompt: str) -> _Resp:
        if "qREFUSE" in prompt:
            return _Resp([], stop_reason="refusal")
        return _Resp([_Block('{"refused": false}')])

    _StubAnthropic.fn = staticmethod(fn)
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    # 3/20 = 15% judge-refusals — above the 10% floor, but they are a counted
    # measurement limitation, not an outage: must NOT raise.
    items = [{"question": f"q{i}", "completion": "c"} for i in range(17)]
    items += [{"question": f"qREFUSE{i}", "completion": "c"} for i in range(3)]
    verdicts = judges_545.judge_items("sonnet_refusal", items, max_workers=2)
    assert sum(1 for v in verdicts if "_judge_refused" in v) == 3
    assert sum(1 for v in verdicts if "_judge_error" in v) == 0


def test_judge_items_refusals_dont_mask_real_errors(monkeypatch):
    def fn(prompt: str) -> _Resp:
        if "qREFUSE" in prompt:
            return _Resp([], stop_reason="refusal")
        if "qBAD" in prompt:
            return _Resp([])  # genuine empty-content parse failure
        return _Resp([_Block('{"refused": false}')])

    _StubAnthropic.fn = staticmethod(fn)
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    # 3 errors / 20 items = 15% > 10% floor: still an outage even though the
    # 3 refusals are excluded from the numerator.
    items = [{"question": f"q{i}", "completion": "c"} for i in range(14)]
    items += [{"question": f"qREFUSE{i}", "completion": "c"} for i in range(3)]
    items += [{"question": f"qBAD{i}", "completion": "c"} for i in range(3)]
    with pytest.raises(RuntimeError, match="quality floor"):
        judges_545.judge_items("sonnet_refusal", items, max_workers=2)


def test_refusal_judge_sends_safety_eval_system_prompt(monkeypatch):
    _StubAnthropic.fn = staticmethod(lambda prompt: _Resp([_Block('{"refused": false}')]))
    _StubAnthropic.captured = []
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    judges_545.judge_items("sonnet_refusal", [{"question": "q0", "completion": "c"}], max_workers=1)
    assert _StubAnthropic.captured, "no judge call captured"
    system = _StubAnthropic.captured[0].get("system")
    assert system and "safety evaluator" in system  # fix (a): framing actually sent


def test_refusal_column_excludes_judge_refused_from_rates(monkeypatch, tmp_path):
    import json

    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS

    def fn(prompt: str) -> _Resp:
        if "qREFUSE" in prompt:
            return _Resp([], stop_reason="refusal")
        if "qHARM" in prompt:
            return _Resp([_Block('{"refused": true}')])
        return _Resp([_Block('{"refused": false}')])

    _StubAnthropic.fn = staticmethod(fn)
    monkeypatch.setattr("anthropic.Anthropic", _StubAnthropic)
    rows = [
        {
            "probe_id": f"snr_{i}",
            "question": f"q{i} placeholder",
            "half": "should_not_refuse",
            "completions": ["placeholder reply"],
        }
        for i in range(5)
    ]
    rows += [
        {
            "probe_id": f"sr_{i}",
            "question": ("qREFUSE" if i < 2 else "qHARM") + f" placeholder {i}",
            "half": "should_refuse",
            "completions": ["placeholder reply"],
        }
        for i in range(5)
    ]
    gen_path = tmp_path / "completions__refusal__default.json"
    gen_path.write_text(json.dumps({"rows": rows, "context": "default", "adapter": None}))
    out_path = tmp_path / "refusal__default.json"
    column = COLUMNS["refusal"]
    eval_battery.judge_column(column, gen_path, out_path)
    out = json.loads(out_path.read_text())
    s = out["summary"]
    assert s["n_judge_refused"] == 2
    assert s["judge_errors"] == 0
    assert s["refusal_rate_should_not_refuse"] == 0.0
    # 2 judge-refused probes excluded from the denominator: 3/3, not 3/5.
    assert s["refusal_rate_should_refuse"] == 1.0
    refused_rows = [v for v in out["verdicts"] if "_judge_refused" in v]
    assert sorted(v["probe_id"] for v in refused_rows) == ["sr_0", "sr_1"]
    assert all(v["half"] == "should_refuse" for v in refused_rows)
