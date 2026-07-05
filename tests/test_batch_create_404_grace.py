"""Tests for the batch retrieve 404-tolerance regimes (#995 create-grace + #1035 mid-poll).

A ``batches.retrieve`` can 404 transiently within seconds of the SAME process's
own ``batches.create`` returning the id (read-after-write inconsistency; #742:
a 404 fired 67 ms after create and the batch was confirmed server-side later) —
the #995 CREATE-GRACE regime. #1035 adds a bounded MID-POLL regime for every
other retrieve 404 (beyond the grace window, ``created_at=None`` resumes,
deadline-time final retrieves, cancel-race probes, the fleet reconcile path):
``BATCH_MIDPOLL_404_BACKOFF_S`` retries, then the 404 still re-raises (fail-fast
preserved, just delayed <=~3 min). These tests pin the helpers
(``is_batch_create_grace_404`` / ``next_batch_404_retry`` /
``retrieve_with_404_tolerance``) plus the guarded call sites:
``batch_judge._submit_and_poll_batch`` (loop + final + cancel-race),
``judge_dispatch._poll_one_sub_batch_step`` (+ ``_cancel_stuck_sub_batch``),
``api_dispatch._poll_one_sub_batch_step`` (+ ``submitted_at`` persistence),
``AnthropicBatch.poll`` (incl. the create-time memo + the deadline final
retrieve's local async retry loop), and ``fleet._poll_batch_to_ended``.

Mock strategy mirrors ``tests/test_issue663_batch_hardening.py``: scriptable
client fakes + injectable ``now_fn``/``sleep_fn``, NO live API calls. The 404s
are REAL ``anthropic.NotFoundError`` instances (constructor shape verified
against the installed SDK 0.88.0) so the real helper bodies execute.
"""

import asyncio
import contextlib
import datetime as dt
import json
import logging

import anthropic
import httpx
import pytest

from explore_persona_space.eval import judge_dispatch
from explore_persona_space.eval.batch_judge import _submit_and_poll_batch
from explore_persona_space.llm import api_dispatch
from explore_persona_space.llm.anthropic_client import (
    BATCH_CREATE_404_BACKOFF_S,
    BATCH_CREATE_404_GRACE_S,
    BATCH_MIDPOLL_404_BACKOFF_S,
    AnthropicBatch,
    BatchDeadlineExceeded,
    is_batch_create_grace_404,
    next_batch_404_retry,
    parse_batch_submitted_at,
    retrieve_with_404_tolerance,
)
from explore_persona_space.orchestrate.fleet import _poll_batch_to_ended

# Reuse the shared #663 fakes/helpers — do NOT duplicate them.
from tests.test_api_dispatch import FakeBatchClient
from tests.test_api_dispatch import build_request as api_build_request
from tests.test_api_dispatch import make_items as api_make_items
from tests.test_api_dispatch import parse_response as api_parse_response
from tests.test_issue663_batch_hardening import _DeadlineSubmitClient, _ns, _succeeded

T0 = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
EXPIRES = T0 + dt.timedelta(hours=24)


def _nf(batch_id: str = "msgbatch_x") -> anthropic.NotFoundError:
    """A REAL anthropic.NotFoundError (SDK 0.88.0 ctor shape) for a batch GET."""
    req = httpx.Request("GET", f"https://api.anthropic.com/v1/messages/batches/{batch_id}")
    return anthropic.NotFoundError(
        "not found", response=httpx.Response(404, request=req), body=None
    )


def _server_error() -> anthropic.InternalServerError:
    req = httpx.Request("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_x")
    return anthropic.InternalServerError(
        "boom", response=httpx.Response(500, request=req), body=None
    )


class _Clock:
    """Injectable clock: returns ``values`` in order, repeating the LAST forever.

    Robust to extra reads (logging helpers also consume ticks), unlike a bare
    finite iterator.
    """

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self) -> dt.datetime:
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


def _counting_retrieve(fail_404: int, result="BATCH"):
    """A retrieve_fn raising ``fail_404`` NotFoundErrors then returning result.

    ``fail_404 >= 10**6`` means always-404. Returns (fn, counter dict).
    """
    counter = {"n": 0}

    def _fn():
        counter["n"] += 1
        if counter["n"] <= fail_404:
            raise _nf()
        return result

    return _fn, counter


async def _noop_async_sleep(_interval):
    return None


# ── §4.4 item 1: predicate unit ──────────────────────────────────────────────


def test_predicate_in_window_true():
    assert is_batch_create_grace_404(
        _nf(), created_at=T0, now_fn=lambda: T0 + dt.timedelta(seconds=30)
    )


def test_predicate_past_window_false():
    assert not is_batch_create_grace_404(
        _nf(), created_at=T0, now_fn=lambda: T0 + dt.timedelta(seconds=61)
    )


def test_predicate_created_at_none_false():
    assert not is_batch_create_grace_404(_nf(), created_at=None, now_fn=lambda: T0)


def test_predicate_non_notfound_false():
    assert not is_batch_create_grace_404(
        ValueError("nope"), created_at=T0, now_fn=lambda: T0 + dt.timedelta(seconds=1)
    )
    assert not is_batch_create_grace_404(
        _server_error(), created_at=T0, now_fn=lambda: T0 + dt.timedelta(seconds=1)
    )


def test_parse_batch_submitted_at_shapes():
    parsed = parse_batch_submitted_at("2026-01-01T00:00:00Z")
    assert parsed == T0 and parsed.tzinfo is not None
    assert parse_batch_submitted_at(None) is None
    assert parse_batch_submitted_at("") is None
    naive = parse_batch_submitted_at("2026-01-01T00:00:00")  # naive -> assume UTC
    assert naive == T0


# ── §4.4 item 2: sync wrapper — transient recovery ───────────────────────────


def test_wrapper_recovers_after_transient_404():
    """404 twice then success -> returns the batch; exactly 3 calls; backoff sleeps."""
    fn, counter = _counting_retrieve(fail_404=2)
    sleeps: list[float] = []
    out = retrieve_with_404_tolerance(
        fn,
        created_at=T0,
        batch_id="msgbatch_x",
        now_fn=_Clock([T0 + dt.timedelta(seconds=5)]),
        sleep_fn=sleeps.append,
    )
    assert out == "BATCH"
    assert counter["n"] == 3
    assert sleeps == [1.0, 2.0]


# ── §4.4 item 3: sync wrapper — window expiry (named fail-loud test) ─────────


def test_window_expiry_reraises_notfound():
    """The clock advances past the 60s window -> the out-of-window 404s take
    the bounded MID-POLL budget (#1035; previously instantly terminal), then
    the ORIGINAL NotFoundError re-raises. Still bounded, never masked."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    # Grace check 1 in-window (retry), first_retry_at read, every later check
    # past-window (the _Clock repeats its last value) -> the mid-poll budget.
    clock = _Clock(
        [
            T0 + dt.timedelta(seconds=5),
            T0 + dt.timedelta(seconds=5),
            T0 + dt.timedelta(seconds=70),
        ]
    )
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_404_tolerance(
            fn, created_at=T0, batch_id="msgbatch_x", now_fn=clock, sleep_fn=sleeps.append
        )
    # 1 graced retry + len(midpoll) mid-poll retries + 1 terminal call.
    assert counter["n"] == 1 + len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == [1.0, *BATCH_MIDPOLL_404_BACKOFF_S]


# ── §4.4 item 4: sync wrapper — backoff exhaustion (named fail-loud test) ────


def test_backoff_exhaustion_reraises_notfound():
    """Frozen clock (window can never expire) + always-404 -> re-raises after
    exactly len(backoff_s)+1 = 7 calls (attempt bound of the dual bound)."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    frozen = _Clock([T0 + dt.timedelta(seconds=1)])
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_404_tolerance(
            fn, created_at=T0, batch_id="msgbatch_x", now_fn=frozen, sleep_fn=sleeps.append
        )
    assert counter["n"] == len(BATCH_CREATE_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_CREATE_404_BACKOFF_S)


# ── §4.4 item 5: sync wrapper — resumed poll (named fail-loud test) ──────────


def test_resumed_poll_404_bounded_terminal():
    """created_at=None (resumed poll / old state.json) -> the bounded MID-POLL
    budget (#1035; previously a single terminal call), then re-raise. NOT
    infinite; NOT silently masked."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_404_tolerance(
            fn, created_at=None, batch_id="msgbatch_x", now_fn=lambda: T0, sleep_fn=sleeps.append
        )
    assert counter["n"] == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


# ── §4.4 item 6: sync wrapper — other exceptions propagate ───────────────────


@pytest.mark.parametrize("exc_factory", [_server_error, lambda: ValueError("boom")])
def test_wrapper_other_exception_propagates_first_call(exc_factory):
    counter = {"n": 0}

    def _fn():
        counter["n"] += 1
        raise exc_factory()

    with pytest.raises((anthropic.InternalServerError, ValueError)):
        retrieve_with_404_tolerance(
            _fn,
            created_at=T0,
            now_fn=lambda: T0 + dt.timedelta(seconds=1),
            sleep_fn=lambda _s: None,
        )
    assert counter["n"] == 1  # never retried


# ── §4.4 item 13: negative-elapsed pin (named fail-loud test) ────────────────


def test_negative_elapsed_out_of_window():
    """created_at in the FUTURE of now_fn (negative elapsed — injected/backwards
    clock) is OUT of window: predicate False, wrapper takes the bounded
    MID-POLL budget (#1035; previously terminal on the first 404) then
    re-raises. Pins the ``0.0 <= elapsed`` guard so a wall-clock capture can
    never vacuously pass an injected-clock integration test."""
    future_created = T0 + dt.timedelta(hours=1)
    assert not is_batch_create_grace_404(_nf(), created_at=future_created, now_fn=lambda: T0)

    fn, counter = _counting_retrieve(fail_404=10**6)
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_404_tolerance(
            fn, created_at=future_created, now_fn=lambda: T0, sleep_fn=lambda _s: None
        )
    assert counter["n"] == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1


# ── #1035 §5 item 3: the shared decision helper, across the branch table ─────


def _decide(exc, *, created_at, now, grace_attempts=0, midpoll_attempts=0):
    return next_batch_404_retry(
        exc,
        created_at=created_at,
        grace_attempts=grace_attempts,
        midpoll_attempts=midpoll_attempts,
        now_fn=lambda: now,
    )


def test_decision_helper_regimes():
    """next_batch_404_retry branch table (#1035 §3.2; branch order load-bearing)."""
    in_window = T0 + dt.timedelta(seconds=30)
    out_window = T0 + dt.timedelta(hours=2)
    # 1. Non-404s -> None (propagate unchanged), regardless of window/attempts.
    assert _decide(ValueError("nope"), created_at=T0, now=in_window) is None
    assert _decide(_server_error(), created_at=T0, now=in_window) is None
    # 2. In-window: grace schedule, capped; exhausted-in-window stays TERMINAL
    #    (the #995 frozen-clock dual-bound pin — no mid-poll fallthrough).
    assert _decide(_nf(), created_at=T0, now=in_window) == ("grace", 1.0)
    n_grace = len(BATCH_CREATE_404_BACKOFF_S)
    assert _decide(_nf(), created_at=T0, now=in_window, grace_attempts=n_grace - 1) == (
        "grace",
        BATCH_CREATE_404_BACKOFF_S[-1],
    )
    assert _decide(_nf(), created_at=T0, now=in_window, grace_attempts=n_grace) is None
    # 2b. The window's upper boundary is CLOSED: elapsed == grace_s is IN
    #     window (a strict-inequality mutation must fail this cell) — and the
    #     boundary cell with grace exhausted stays terminal too.
    boundary = T0 + dt.timedelta(seconds=BATCH_CREATE_404_GRACE_S)
    assert _decide(_nf(), created_at=T0, now=boundary) == ("grace", 1.0)
    assert _decide(_nf(), created_at=T0, now=boundary, grace_attempts=n_grace) is None
    # 3. Outside the window: mid-poll schedule, capped.
    assert _decide(_nf(), created_at=T0, now=out_window) == ("midpoll", 5.0)
    n_mid = len(BATCH_MIDPOLL_404_BACKOFF_S)
    assert _decide(_nf(), created_at=T0, now=out_window, midpoll_attempts=n_mid - 1) == (
        "midpoll",
        BATCH_MIDPOLL_404_BACKOFF_S[-1],
    )
    assert _decide(_nf(), created_at=T0, now=out_window, midpoll_attempts=n_mid) is None
    # 3b. created_at=None (resume) and negative elapsed route to mid-poll too.
    assert _decide(_nf(), created_at=None, now=T0) == ("midpoll", 5.0)
    future = T0 + dt.timedelta(hours=1)
    assert _decide(_nf(), created_at=future, now=T0) == ("midpoll", 5.0)
    # 3c. midpoll_backoff_s=() disables the regime (the per-site escape hatch).
    assert (
        next_batch_404_retry(
            _nf(),
            created_at=None,
            grace_attempts=0,
            midpoll_attempts=0,
            now_fn=lambda: T0,
            midpoll_backoff_s=(),
        )
        is None
    )


# ── #1035 §5 items 1-2: sync wrapper — mid-poll recovery + exhaustion ─────────


def test_midpoll_404_recovers(caplog):
    """Far outside the grace window (T0+2h): 404 twice then success -> returns
    the result on the mid-poll schedule; recovery INFO carries the grep prefix."""
    fn, counter = _counting_retrieve(fail_404=2)
    sleeps: list[float] = []
    with caplog.at_level(logging.INFO, logger="explore_persona_space.llm.anthropic_client"):
        out = retrieve_with_404_tolerance(
            fn,
            created_at=T0,
            batch_id="msgbatch_x",
            now_fn=_Clock([T0 + dt.timedelta(hours=2)]),
            sleep_fn=sleeps.append,
        )
    assert out == "BATCH"
    assert counter["n"] == 3
    assert sleeps == [5.0, 15.0]
    assert "[batch-404-midpoll]" in caplog.text


def test_midpoll_404_exhausts_reraises():
    """Always-404 outside the window -> re-raises after exactly
    len(BATCH_MIDPOLL_404_BACKOFF_S)+1 calls. FAIL-FAST PRESERVED."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_404_tolerance(
            fn,
            created_at=T0,
            batch_id="msgbatch_x",
            now_fn=_Clock([T0 + dt.timedelta(hours=2)]),
            sleep_fn=sleeps.append,
        )
    assert counter["n"] == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


# ── §4.4 item 7: batch_judge integration (the #742 shape, end to end) ────────


def test_batch_judge_recovers_from_create_grace_404():
    """_submit_and_poll_batch: the FIRST post-create retrieve 404s (the #742
    shape — 67 ms after create), then the batch is visible and ended ->
    results returned instead of a crash."""
    client = _DeadlineSubmitClient(flip_to_ended_on=1)
    orig_retrieve = client.messages.batches.retrieve
    box = {"n404": 0}

    def _retrieve_404_once(batch_id):
        if box["n404"] == 0:
            box["n404"] += 1
            raise _nf(batch_id)
        return orig_retrieve(batch_id)

    client.messages.batches.retrieve = _retrieve_404_once
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    sleeps: list[float] = []
    clock = _Clock([T0, T0 + dt.timedelta(seconds=1)])  # created_at=T0, then in-window
    result = _submit_and_poll_batch(
        reqs, client, poll_interval=0.0, now_fn=clock, sleep_fn=sleeps.append
    )
    assert set(result) == {"c0", "c1"}
    assert all(r["aligned"] == 90 for r in result.values())
    assert box["n404"] == 1 and client.retrieve_calls == 1  # one 404, one real retrieve
    assert sleeps == [1.0]  # the grace backoff, not the poll interval


# ── #1035 §5 item 4: batch_judge deadline FINAL retrieve (newly guarded) ─────


def _final_404_client(*, flip_to_ended_on, fail_404_on_call: int):
    """A _DeadlineSubmitClient whose retrieve 404s exactly once, on the
    ``fail_404_on_call``-th retrieve (counting wrapper-level calls)."""
    client = _DeadlineSubmitClient(flip_to_ended_on=flip_to_ended_on)
    orig_retrieve = client.messages.batches.retrieve
    box = {"n": 0, "n404": 0}

    def _retrieve(batch_id):
        box["n"] += 1
        if box["n"] == fail_404_on_call:
            box["n404"] += 1
            raise _nf(batch_id)
        return orig_retrieve(batch_id)

    client.messages.batches.retrieve = _retrieve
    return client, box


def test_batch_judge_final_retrieve_tolerates_transient_404():
    """_submit_and_poll_batch: the batch never ends in-loop; at the deadline the
    final retrieve 404s once then reads ended -> results harvested, no crash
    (#1035 table B; the #995 unguarded stance reversed)."""
    past_deadline = EXPIRES + dt.timedelta(minutes=31)  # > expires_at + 30min grace
    # Wrapper retrieve #2 is the deadline final retrieve; orig call 2 -> ended.
    client, box = _final_404_client(flip_to_ended_on=2, fail_404_on_call=2)
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    sleeps: list[float] = []
    # created_at=T0 (first tick), then every read past the deadline (repeats).
    clock = _Clock([T0, past_deadline])
    result = _submit_and_poll_batch(
        reqs, client, poll_interval=0.0, now_fn=clock, sleep_fn=sleeps.append
    )
    assert set(result) == {"c0", "c1"}
    assert all(r["aligned"] == 90 for r in result.values())
    assert box["n404"] == 1
    assert sleeps == [5.0]  # the mid-poll backoff, nothing else


def test_batch_judge_final_retrieve_404_then_not_ended_still_deadline_exceeded():
    """Variant b: the final retrieve 404s once then reads a STILL-not-ended
    batch -> BatchDeadlineExceeded fires exactly as before (the 404 tolerance
    recovers the READ; the deadline classification is untouched)."""
    past_deadline = EXPIRES + dt.timedelta(minutes=31)
    client, box = _final_404_client(flip_to_ended_on=None, fail_404_on_call=2)
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    clock = _Clock([T0, past_deadline])
    with pytest.raises(BatchDeadlineExceeded):
        _submit_and_poll_batch(
            reqs, client, poll_interval=0.0, now_fn=clock, sleep_fn=lambda _s: None
        )
    assert box["n404"] == 1  # the 404 was retried, not the crash class


# ── #1035 §5 item 5c(i): batch_judge stuck-cancel race probe (newly guarded) ──


def test_batch_judge_cancel_race_probe_tolerates_404(monkeypatch):
    """The #1019 stuck-cancel race probe: cancel raises APIStatusError, the
    confirm retrieve 404s once then reads canceling -> race resolution
    proceeds (canceled rows surface as error dicts), no crash."""
    from tests.test_batch_stuck_escape import StuckEscapeClient, _FakeAPIStatusError

    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "4")
    outcome = {"c0": "canceled", "c1": "canceled"}
    client = StuckEscapeClient(
        outcome_for=outcome,
        expires_at=T0 + dt.timedelta(hours=24),  # legacy path anchors on now_fn
        cancel_effects=[_FakeAPIStatusError()],
        post_cancel_statuses=("canceling", "ended"),
    )
    orig_retrieve = client.messages.batches.retrieve
    box = {"n": 0, "n404": 0}

    def _retrieve(batch_id):
        box["n"] += 1
        if box["n"] == 2:  # the race-confirm probe (loop retrieve was call 1)
            box["n404"] += 1
            raise _nf(batch_id)
        return orig_retrieve(batch_id)

    client.messages.batches.retrieve = _retrieve
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    sleeps: list[float] = []
    # created_at=T0; every later tick 5h past it (inside the deadline, past the
    # 4h stuck threshold, outside the 60s grace window -> mid-poll regime).
    clock = _Clock([T0, T0 + dt.timedelta(hours=5)])
    result = _submit_and_poll_batch(
        reqs, client, poll_interval=0.0, now_fn=clock, sleep_fn=sleeps.append
    )
    assert client.cancel_calls == 1
    assert box["n404"] == 1
    assert 5.0 in sleeps  # the probe's mid-poll backoff
    for cid in ("c0", "c1"):
        assert result[cid]["error"] is True


# ── §4.4 item 8: judge_dispatch _poll_one_sub_batch_step ─────────────────────


class _JudgeStepClient:
    """messages.batches fake for the judge_dispatch poll step: retrieve 404s the
    first ``fail_404_first`` calls, then returns ended; results yields one
    succeeded row."""

    def __init__(self, *, fail_404_first: int = 0):
        self.fail_404_first = fail_404_first
        self.retrieve_calls = 0
        client = self

        class _B:
            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                if client.retrieve_calls <= client.fail_404_first:
                    raise _nf(batch_id)
                return _ns(processing_status="ended", expires_at=EXPIRES, request_counts=None)

            def results(_s, batch_id):
                yield _succeeded("item_0")

        self.messages = _ns(batches=_B())


def _judge_step(tmp_path, client, sb, now_fn, sleep_fn):
    acc = judge_dispatch._BatchCollector()
    judge_dispatch._poll_one_sub_batch_step(
        acc,
        state={"sub_batches": [sb]},
        state_path=tmp_path / "state.json",
        dispatch_dir=tmp_path,
        client=client,
        error_dict_factory=judge_dispatch._default_error_dict,
        now_fn=now_fn,
        sb=sb,
        sleep_fn=sleep_fn,
    )
    return acc


def _judge_sb(**overrides) -> dict:
    sb = {
        "index": 0,
        "batch_id": "msgbatch_1",
        "status": "submitted",
        "submitted_at": "2026-01-01T00:00:00Z",
        "deadline": None,
        "n_requests": 1,
    }
    sb.update(overrides)
    return sb


def test_judge_dispatch_step_fresh_submitted_at_grace_recovers(tmp_path):
    client = _JudgeStepClient(fail_404_first=1)
    sb = _judge_sb()
    sleeps: list[float] = []
    acc = _judge_step(
        tmp_path, client, sb, now_fn=_Clock([T0 + dt.timedelta(seconds=10)]), sleep_fn=sleeps.append
    )
    assert acc.scores["item_0"]["aligned"] == 90  # harvested despite the transient 404
    assert sb["status"] == "collected"
    assert client.retrieve_calls == 2
    assert sleeps == [1.0]


def test_judge_dispatch_step_stale_submitted_at_terminal(tmp_path):
    """A resumed poll whose submitted_at is 1h old -> window expired -> the 404
    takes the bounded MID-POLL budget (#1035; previously one terminal call),
    then propagates. Sleeps injected (a real 170s of backoff otherwise)."""
    client = _JudgeStepClient(fail_404_first=10**6)
    sb = _judge_sb()
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        _judge_step(
            tmp_path,
            client,
            sb,
            now_fn=_Clock([T0 + dt.timedelta(hours=1)]),
            sleep_fn=sleeps.append,
        )
    assert client.retrieve_calls == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


def test_judge_dispatch_step_absent_submitted_at_terminal(tmp_path):
    """An OLD state.json without submitted_at -> created_at None -> the bounded
    MID-POLL budget (#1035), then the 404 still propagates (never masked)."""
    client = _JudgeStepClient(fail_404_first=10**6)
    sb = _judge_sb()
    del sb["submitted_at"]
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        _judge_step(tmp_path, client, sb, now_fn=_Clock([T0]), sleep_fn=sleeps.append)
    assert client.retrieve_calls == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


# ── #1035 §5 item 5: judge_dispatch overdue FINAL retrieve (newly guarded) ────


class _ScriptedStepClient:
    """messages.batches fake whose retrieve walks ``script`` — each entry is
    "404" (raise) or a processing_status (last entry repeats); results yields
    one succeeded row."""

    def __init__(self, script):
        self.script = list(script)
        self.retrieve_calls = 0
        client = self

        class _B:
            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                action = client.script[min(client.retrieve_calls - 1, len(client.script) - 1)]
                if action == "404":
                    raise _nf(batch_id)
                return _ns(processing_status=action, expires_at=EXPIRES, request_counts=None)

            def results(_s, batch_id):
                yield _succeeded("item_0")

        self.messages = _ns(batches=_B())


def test_judge_dispatch_overdue_final_retrieve_tolerates_404(tmp_path):
    """The overdue final retrieve 404s once then reads ended -> harvested
    (#1035 table B; previously the 404 crashed the last harvest chance)."""
    client = _ScriptedStepClient(["in_progress", "404", "ended"])
    sb = _judge_sb(deadline=(T0 + dt.timedelta(hours=1)).isoformat())
    sleeps: list[float] = []
    acc = _judge_step(
        tmp_path,
        client,
        sb,
        now_fn=_Clock([T0 + dt.timedelta(hours=2)]),  # past the persisted deadline
        sleep_fn=sleeps.append,
    )
    assert acc.scores["item_0"]["aligned"] == 90
    assert sb["status"] == "collected"
    assert client.retrieve_calls == 3
    assert sleeps == [5.0]


# ── #1035 §5 item 5c(ii): judge_dispatch stuck-cancel race probe ──────────────


class _CancelRaceClient:
    """cancel raises APIStatusError; the confirm retrieve 404s the first
    ``fail_404_first`` calls then returns ``final_status``."""

    def __init__(self, *, fail_404_first: int, final_status: str = "canceling"):
        self.retrieve_calls = 0
        self.cancel_calls = 0
        client = self

        class _B:
            def cancel(_s, batch_id):
                from tests.test_batch_stuck_escape import _FakeAPIStatusError

                client.cancel_calls += 1
                raise _FakeAPIStatusError()

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                if client.retrieve_calls <= fail_404_first:
                    raise _nf(batch_id)
                return _ns(processing_status=final_status, expires_at=EXPIRES, request_counts=None)

        self.messages = _ns(batches=_B())


def test_judge_dispatch_cancel_race_probe_tolerates_404(tmp_path):
    """_cancel_stuck_sub_batch: the race-confirm retrieve 404s once then reads
    canceling -> accepted as canceled, confirm stamp persisted (#1035 table C)."""
    client = _CancelRaceClient(fail_404_first=1)
    sb = _judge_sb()
    state = {"sub_batches": [sb]}
    sleeps: list[float] = []
    judge_dispatch._cancel_stuck_sub_batch(
        client,
        sb,
        state,
        tmp_path / "state.json",
        _Clock([T0 + dt.timedelta(hours=5)]),
        sleeps.append,
    )
    assert client.cancel_calls == 1
    assert sb["stuck_canceled_at"] is not None
    assert client.retrieve_calls == 2
    assert sleeps == [5.0]


def test_judge_dispatch_cancel_race_probe_persistent_404_bounded(tmp_path):
    """A persistent 404 on the race-confirm retrieve re-raises after the
    bounded mid-poll budget (the enclosing #1019 semantics unchanged)."""
    client = _CancelRaceClient(fail_404_first=10**6)
    sb = _judge_sb()
    state = {"sub_batches": [sb]}
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        judge_dispatch._cancel_stuck_sub_batch(
            client,
            sb,
            state,
            tmp_path / "state.json",
            _Clock([T0 + dt.timedelta(hours=5)]),
            sleeps.append,
        )
    assert client.retrieve_calls == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)
    assert sb.get("stuck_canceled_at") is None  # crash before the confirm stamp


# ── §4.4 item 9: api_dispatch — submitted_at persistence + poll-step grace ───


def _api_submit(tmp_path, client):
    """Init batch state + submit sub-batch 0; return (state, sb, state_path, items)."""
    items = api_make_items(2)
    state_path = tmp_path / "state.json"
    state = api_dispatch._load_or_init_batch_state(
        state_path,
        items,
        build_request=api_build_request,
        org_labels=["high_prio"],
        chunk_size=10,
    )
    by_id = {it.item_id: it for it in items}
    items_by_cid = {cid: by_id[iid] for cid, iid in state["cid_to_item"].items()}
    sb = state["sub_batches"][0]

    async def _run():
        await api_dispatch._submit_one_sub_batch(
            sb,
            state=state,
            state_path=state_path,
            items_by_cid=items_by_cid,
            build_request=api_build_request,
            sync_clients={"high_prio": client},
            sem=asyncio.Semaphore(1),
        )

    asyncio.run(_run())
    return state, sb, state_path, items


def test_api_dispatch_submit_persists_submitted_at(tmp_path):
    """_submit_one_sub_batch records a parseable, fresh submitted_at in the
    finally-shielded persist (read back from the atomic state.json write)."""
    client = FakeBatchClient("high_prio")
    _state, sb, state_path, _items = _api_submit(tmp_path, client)
    persisted = json.loads(state_path.read_text())["sub_batches"][0]
    assert persisted["batch_id"] == sb["batch_id"] is not None
    ts = parse_batch_submitted_at(persisted["submitted_at"])
    assert ts is not None and ts.tzinfo is not None
    # Fresh: within the grace window of the real wall clock (strftime truncates
    # DOWN to the second, so elapsed >= 0 by construction).
    assert 0.0 <= (dt.datetime.now(dt.UTC) - ts).total_seconds() <= BATCH_CREATE_404_GRACE_S


def test_api_dispatch_poll_step_fresh_submitted_at_grace_recovers(tmp_path):
    """async _poll_one_sub_batch_step: one 404 within the grace window of the
    persisted submitted_at -> retried -> harvested."""
    client = FakeBatchClient("high_prio")  # end_after=0 -> first real retrieve is ended
    state, sb, state_path, items = _api_submit(tmp_path, client)
    orig_retrieve = client.messages.batches.retrieve
    box = {"n404": 0}

    def _retrieve_404_once(batch_id):
        if box["n404"] == 0:
            box["n404"] += 1
            raise _nf(batch_id)
        return orig_retrieve(batch_id)

    client.messages.batches.retrieve = _retrieve_404_once
    sleeps: list[float] = []

    async def _run():
        await api_dispatch._poll_one_sub_batch_step(
            sb,
            state=state,
            state_path=state_path,
            dispatch_dir=tmp_path,
            sync_clients={"high_prio": client},
            parse_response=api_parse_response,
            now_fn=lambda: dt.datetime.now(dt.UTC),  # wall clock: submitted_at is fresh
            sleep_fn=sleeps.append,
        )

    asyncio.run(_run())
    assert sb["status"] == "collected"
    assert box["n404"] == 1
    assert sleeps == [1.0]
    payload = json.loads((tmp_path / f"results_{sb['batch_id']}.json").read_text())
    assert payload[items[0].item_id]["error"] is False


def test_api_dispatch_poll_step_absent_submitted_at_terminal(tmp_path):
    """A resumed OLD state.json (no submitted_at key) -> no grace -> the 404
    takes the bounded MID-POLL budget (#1035; previously one terminal call),
    then propagates (org-mismatch resume semantics: delayed, never masked)."""
    client = FakeBatchClient("high_prio")
    state, sb, state_path, _items = _api_submit(tmp_path, client)
    sb.pop("submitted_at")  # simulate a pre-#995 state.json
    box = {"n": 0}

    def _retrieve_always_404(batch_id):
        box["n"] += 1
        raise _nf(batch_id)

    client.messages.batches.retrieve = _retrieve_always_404
    sleeps: list[float] = []

    async def _run():
        await api_dispatch._poll_one_sub_batch_step(
            sb,
            state=state,
            state_path=state_path,
            dispatch_dir=tmp_path,
            sync_clients={"high_prio": client},
            parse_response=api_parse_response,
            now_fn=lambda: dt.datetime.now(dt.UTC),
            sleep_fn=sleeps.append,
        )

    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(_run())
    assert box["n"] == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


def test_api_dispatch_overdue_final_retrieve_tolerates_404(tmp_path):
    """#1035 §5 item 5b: the api_dispatch overdue final retrieve 404s once then
    reads ended -> harvested (created_at threading matches the loop retrieve)."""
    client = FakeBatchClient("high_prio", end_after=1)  # retrieve 1 in_progress, 2 ended
    state, sb, state_path, items = _api_submit(tmp_path, client)
    sb["submitted_at"] = "2026-01-01T00:00:00Z"  # far in the past of the test clock
    sb["deadline"] = (T0 + dt.timedelta(hours=1)).isoformat()  # already overdue
    orig_retrieve = client.messages.batches.retrieve
    box = {"n": 0, "n404": 0}

    def _retrieve(batch_id):
        box["n"] += 1
        if box["n"] == 2:  # the overdue final retrieve (loop retrieve was call 1)
            box["n404"] += 1
            raise _nf(batch_id)
        return orig_retrieve(batch_id)

    client.messages.batches.retrieve = _retrieve
    sleeps: list[float] = []

    async def _run():
        await api_dispatch._poll_one_sub_batch_step(
            sb,
            state=state,
            state_path=state_path,
            dispatch_dir=tmp_path,
            sync_clients={"high_prio": client},
            parse_response=api_parse_response,
            now_fn=_Clock([T0 + dt.timedelta(hours=2)]),
            sleep_fn=sleeps.append,
        )

    asyncio.run(_run())
    assert sb["status"] == "collected"
    assert box["n404"] == 1
    assert sleeps == [5.0]  # the mid-poll backoff (outside the grace window)
    payload = json.loads((tmp_path / f"results_{sb['batch_id']}.json").read_text())
    assert payload[items[0].item_id]["error"] is False


# ── §4.4 items 10 + 12: AnthropicBatch.poll grace + dual bound ───────────────


def _grace_poll_batch(statuses, *, fail_404_first=0, memo=None, expires_at=EXPIRES):
    """An AnthropicBatch (init skipped) whose retrieve 404s the first
    ``fail_404_first`` calls then walks ``statuses`` (last value repeats)."""
    batch = AnthropicBatch.__new__(AnthropicBatch)  # skip __init__ (no real client)
    batch._created_at = dict(memo or {})
    seq = iter(statuses)
    last = {"v": statuses[-1]}
    counter = {"n": 0, "n404": 0}

    def _retrieve(batch_id):
        counter["n"] += 1
        if counter["n404"] < fail_404_first:
            counter["n404"] += 1
            raise _nf(batch_id)
        with contextlib.suppress(StopIteration):
            last["v"] = next(seq)
        return _ns(processing_status=last["v"], expires_at=expires_at, request_counts=None)

    batch.retrieve = _retrieve
    return batch, counter


def test_poll_created_at_kwarg_grace_recovers():
    """created_at fresh + one 404 then in_progress -> ended -> returns (an
    ADVANCING clock so an always-404 fake could never hang the harness)."""
    batch, counter = _grace_poll_batch(["in_progress", "ended"], fail_404_first=1)
    clock = _Clock(
        [
            T0 + dt.timedelta(seconds=1),
            T0 + dt.timedelta(seconds=2),
            T0 + dt.timedelta(seconds=3),
            T0 + dt.timedelta(seconds=4),
        ]
    )
    out = asyncio.run(
        batch.poll("msgbatch_g", now_fn=clock, sleep_fn=_noop_async_sleep, created_at=T0)
    )
    assert out.processing_status == "ended"
    assert counter["n404"] == 1
    assert counter["n"] == 3  # 404, in_progress, ended


def test_poll_no_memo_no_kwarg_404_bounded_terminal():
    """Fresh instance polling an unknown id (the new-process resume shape):
    no memo entry AND no created_at kwarg -> the bounded MID-POLL budget
    (#1035; previously terminal on the FIRST 404), then re-raise."""
    batch, counter = _grace_poll_batch(["in_progress"], fail_404_first=10**6)
    sleeps: list[float] = []

    async def _capture_sleep(delay):
        sleeps.append(delay)

    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(batch.poll("msgbatch_unknown", now_fn=_Clock([T0]), sleep_fn=_capture_sleep))
    assert counter["n"] == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


def test_poll_frozen_clock_all_404_bounded():
    """Site-(d) dual bound: frozen now_fn (window can never expire) + always-404
    + fresh memo -> re-raises after exactly len(BATCH_CREATE_404_BACKOFF_S)+1 = 7
    retrieve calls (the attempt cap binds when the window cannot)."""
    batch, counter = _grace_poll_batch(
        ["in_progress"], fail_404_first=10**6, memo={"msgbatch_z": T0}
    )
    frozen = _Clock([T0 + dt.timedelta(seconds=1)])
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(batch.poll("msgbatch_z", now_fn=frozen, sleep_fn=_noop_async_sleep))
    assert counter["n"] == len(BATCH_CREATE_404_BACKOFF_S) + 1


# ── §4.4 item 11: create-time memo covers direct create->poll callers ────────


class _CreateThenPollBatches:
    """messages.batches fake: create() returns a fresh id; retrieve 404s the
    first ``fail_404_first`` calls then walks ``statuses``. expires_at is a
    FRESH wall-clock datetime so the poll deadline stays in the future."""

    def __init__(self, *, fail_404_first: int, statuses):
        self.fail_404_first = fail_404_first
        self._statuses = iter(statuses)
        self._last = statuses[-1]
        self.create_calls = 0
        self.retrieve_calls = 0
        self.expires_at = dt.datetime.now(dt.UTC) + dt.timedelta(hours=24)

    def create(self, requests):
        self.create_calls += 1
        return _ns(id=f"msgbatch_memo_{self.create_calls:03d}", expires_at=self.expires_at)

    def retrieve(self, batch_id):
        self.retrieve_calls += 1
        if self.retrieve_calls <= self.fail_404_first:
            raise _nf(batch_id)
        with contextlib.suppress(StopIteration):
            self._last = next(self._statuses)
        return _ns(processing_status=self._last, expires_at=self.expires_at, request_counts=None)


def test_create_memo_covers_direct_poll_callers():
    """The issue658 caller shape, end to end: same-instance create(...) then
    poll(batch_id) with NO created_at kwarg recovers from a create-window 404
    via the memo (zero caller wiring); a SECOND instance polling the same id
    has no memo -> the bounded MID-POLL budget (#1035; previously terminal on
    the first 404 — memo is still instance-scoped, the budget is just bounded
    rather than zero), then re-raise."""
    b = AnthropicBatch(anthropic_api_key="test-key-unused")  # REAL __init__ -> memo dict
    fake = _CreateThenPollBatches(fail_404_first=1, statuses=["in_progress", "ended"])
    b.client = _ns(messages=_ns(batches=fake))
    created = b.create(requests=[{"custom_id": "c0", "params": {}}])
    assert created.id in b._created_at  # memo recorded by create()

    # Default wall-clock now_fn: the memo is fresh, so the 404 is in-window.
    out = asyncio.run(b.poll(created.id, sleep_fn=_noop_async_sleep))
    assert out.processing_status == "ended"
    assert fake.retrieve_calls == 3  # 404, in_progress, ended

    b2 = AnthropicBatch(anthropic_api_key="test-key-unused")
    fake2 = _CreateThenPollBatches(fail_404_first=10**6, statuses=["ended"])
    b2.client = _ns(messages=_ns(batches=fake2))
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(b2.poll(created.id, sleep_fn=_noop_async_sleep))
    assert fake2.retrieve_calls == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1


# ── #1035 §5 items 7/7b/7c: AnthropicBatch.poll mid-poll + counter reset ─────


def _scripted_poll_batch(script, *, memo=None, expires_at=EXPIRES):
    """An AnthropicBatch (init skipped) whose retrieve walks ``script`` — each
    entry is "404" (raise) or a processing_status; the LAST entry repeats."""
    batch = AnthropicBatch.__new__(AnthropicBatch)  # skip __init__ (no real client)
    batch._created_at = dict(memo or {})
    seq = list(script)
    counter = {"n": 0}

    def _retrieve(batch_id):
        counter["n"] += 1
        action = seq[min(counter["n"] - 1, len(seq) - 1)]
        if action == "404":
            raise _nf(batch_id)
        return _ns(processing_status=action, expires_at=expires_at, request_counts=None)

    batch.retrieve = _retrieve
    return batch, counter


def _capturing_async_sleep(sleeps):
    async def _sleep(delay):
        sleeps.append(delay)

    return _sleep


def test_poll_midpoll_404_recovers_hours_in(caplog):
    """poll: a healthy retrieve, then 404s HOURS past create (the mid-poll
    class) -> bounded retry on the shared schedule, recovers; the async path
    logs the same grep prefix as the sync wrapper."""
    batch, counter = _scripted_poll_batch(
        ["in_progress", "404", "404", "ended"], memo={"msgbatch_m": T0}
    )
    sleeps: list[float] = []
    with caplog.at_level(logging.INFO, logger="explore_persona_space.llm.anthropic_client"):
        out = asyncio.run(
            batch.poll(
                "msgbatch_m",
                interval_s=60.0,
                now_fn=_Clock([T0 + dt.timedelta(hours=2)]),
                sleep_fn=_capturing_async_sleep(sleeps),
            )
        )
    assert out.processing_status == "ended"
    assert counter["n"] == 4
    assert sleeps == [60.0, 5.0, 15.0]  # one poll interval, then the mid-poll backoff
    assert "[batch-404-midpoll]" in caplog.text


def test_poll_midpoll_404_bounded_after_success():
    """poll: always-404 after a first healthy retrieve -> re-raises after
    exactly len(BATCH_MIDPOLL_404_BACKOFF_S)+1 further calls (fail-fast)."""
    batch, counter = _scripted_poll_batch(["in_progress", "404"], memo={"msgbatch_m": T0})
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(
            batch.poll(
                "msgbatch_m",
                interval_s=60.0,
                now_fn=_Clock([T0 + dt.timedelta(hours=2)]),
                sleep_fn=_capturing_async_sleep(sleeps),
            )
        )
    assert counter["n"] == 1 + len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == [60.0, *BATCH_MIDPOLL_404_BACKOFF_S]


def test_poll_counter_reset_two_episodes_midpoll():
    """TWO-EPISODE counter-reset pin (#1035 §3.5): a first 404 episode retries
    then a retrieve SUCCEEDS; a second episode gets a FRESH full mid-poll
    budget (without the reset the second episode would get only 4 retries)."""
    batch, counter = _scripted_poll_batch(["404", "in_progress", "404"], memo={"msgbatch_m": T0})
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(
            batch.poll(
                "msgbatch_m",
                interval_s=60.0,
                now_fn=_Clock([T0 + dt.timedelta(hours=2)]),
                sleep_fn=_capturing_async_sleep(sleeps),
            )
        )
    # ep1: 1 retried 404 + 1 success; ep2: 5 retried 404s + 1 terminal.
    assert counter["n"] == 2 + len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == [5.0, 60.0, *BATCH_MIDPOLL_404_BACKOFF_S]


def test_poll_counter_reset_two_episodes_grace():
    """Grace-counter sibling of the two-episode reset pin: a frozen in-window
    clock; episode 1 uses 2 grace retries then succeeds; episode 2 gets the
    FULL fresh grace budget (6 retries + terminal — the #995 in-window
    exhausted terminal, not a shrunken leftover budget)."""
    batch, counter = _scripted_poll_batch(
        ["404", "404", "in_progress", "404"], memo={"msgbatch_z": T0}
    )
    sleeps: list[float] = []
    frozen = _Clock([T0 + dt.timedelta(seconds=1)])
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(
            batch.poll(
                "msgbatch_z",
                interval_s=60.0,
                now_fn=frozen,
                sleep_fn=_capturing_async_sleep(sleeps),
            )
        )
    # ep1: 2 retried 404s + 1 success; ep2: 6 retried 404s + 1 terminal.
    assert counter["n"] == 3 + len(BATCH_CREATE_404_BACKOFF_S) + 1
    assert sleeps == [1.0, 2.0, 60.0, *BATCH_CREATE_404_BACKOFF_S]


def test_poll_deadline_final_retrieve_404_recovers(caplog):
    """#1035 §5 item 7c(i): the poll deadline-time FINAL retrieve (the local
    async retry loop) 404s once then reads ended -> partial harvest proceeds,
    and the recovery emits a [batch-404-midpoll] INFO (plan criterion 5).

    The recovery assert is LEVEL-scoped: the retry WARNING also carries the
    prefix, so a bare ``in caplog.text`` would pass without any recovery log.
    """
    past_deadline = EXPIRES + dt.timedelta(minutes=31)
    batch, counter = _scripted_poll_batch(["in_progress", "404", "ended"])
    sleeps: list[float] = []
    with caplog.at_level(logging.INFO, logger="explore_persona_space.llm.anthropic_client"):
        out = asyncio.run(
            batch.poll(
                "msgbatch_f",
                now_fn=_Clock([past_deadline]),
                sleep_fn=_capturing_async_sleep(sleeps),
            )
        )
    assert out.processing_status == "ended"
    assert counter["n"] == 3
    assert sleeps == [5.0]
    recovery_infos = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "[batch-404-midpoll]" in r.getMessage()
    ]
    assert recovery_infos, "deadline-final recovery must log a [batch-404-midpoll] INFO"


def test_poll_deadline_final_retrieve_404_bounded():
    """#1035 §5 item 7c(ii): an always-404 deadline final retrieve re-raises
    NotFoundError after exactly len(BATCH_MIDPOLL_404_BACKOFF_S)+1 calls —
    the suite would pass on an unbounded or 404-swallowing :750 loop without
    this pin (the must-never-do)."""
    past_deadline = EXPIRES + dt.timedelta(minutes=31)
    batch, counter = _scripted_poll_batch(["in_progress", "404"])
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(
            batch.poll(
                "msgbatch_f",
                now_fn=_Clock([past_deadline]),
                sleep_fn=_capturing_async_sleep(sleeps),
            )
        )
    assert counter["n"] == 1 + len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)


# ── #1035 §5 item 6: fleet._poll_batch_to_ended (table D, created_at=None) ───


class _FleetClient:
    """messages.batches fake: retrieve 404s the first ``fail_404_first`` calls
    then returns ended."""

    def __init__(self, *, fail_404_first: int):
        self.retrieve_calls = 0
        client = self

        class _B:
            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                if client.retrieve_calls <= fail_404_first:
                    raise _nf(batch_id)
                return _ns(processing_status="ended", expires_at=EXPIRES, request_counts=None)

        self.messages = _ns(batches=_B())


def test_fleet_poll_tolerates_midpoll_404():
    """The deferred/cross-process reconcile path: 404 twice then ended ->
    returns (the #995 NOTE's terminal-404 default replaced by the bounded
    mid-poll tolerance, created_at=None)."""
    client = _FleetClient(fail_404_first=2)
    sleeps: list[float] = []
    _poll_batch_to_ended(
        client,
        "msgbatch_fleet",
        poll_interval=0.0,
        max_poll_interval=1.0,
        grace_min=30,
        now_fn=_Clock([T0]),
        sleep_fn=sleeps.append,
    )
    assert client.retrieve_calls == 3
    assert sleeps == [5.0, 15.0]


def test_fleet_poll_persistent_404_bounded():
    """An always-404 fleet reconcile re-raises after the bounded mid-poll
    budget (wrong-org / genuinely-gone stays fail-fast, just <=~3 min later)."""
    client = _FleetClient(fail_404_first=10**6)
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        _poll_batch_to_ended(
            client,
            "msgbatch_fleet",
            poll_interval=0.0,
            max_poll_interval=1.0,
            grace_min=30,
            now_fn=_Clock([T0]),
            sleep_fn=sleeps.append,
        )
    assert client.retrieve_calls == len(BATCH_MIDPOLL_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_MIDPOLL_404_BACKOFF_S)
