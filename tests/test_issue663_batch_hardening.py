"""Tests for the #663 batch-judge hardening: expires_at deadline bound, the
two-level terminal-state split (invalid_request_error quarantine), custom_id
resume, the legacy ``_submit_and_poll_batch`` rewrite, and the concurrent
sub-batch submission fan-out semantics.

Mock strategy mirrors ``tests/test_judge_dispatch.py``: every test injects
scriptable client fakes (SimpleNamespace / the shared ``FakeBatchClient``) and
a mocked clock — NO live API calls, no ``ANTHROPIC_API_KEY`` read. The deadline
tests drive ``expires_at`` as a ``datetime.datetime`` (the SDK 0.88.0 shape, A1
fact-checker correction) so they exercise the production datetime branch, not
the raw-string isinstance fallback.
"""

import asyncio
import contextlib
import datetime as dt
import itertools
import json

import pytest

from explore_persona_space.eval import batch_judge
from explore_persona_space.eval.batch_judge import (
    _CUSTOM_ID_RE,
    _submit_and_poll_batch,
    deadline_from_expires_at,
    make_custom_id,
)
from explore_persona_space.eval.judge_dispatch import _collect_batch_results, _default_error_dict
from explore_persona_space.llm.anthropic_client import AnthropicBatch, BatchDeadlineExceeded

# Reuse the shared fakes / helpers — do NOT duplicate FakeBatchClient.
from tests.test_judge_dispatch import _msg, dispatch, make_items

JUDGE_TEXT = '{"aligned": 90, "coherent": 95, "reasoning": "ok"}'
T0 = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
EXPIRES = T0 + dt.timedelta(hours=24)  # API guarantee: ended by created_at + 24h
PAST_DEADLINE = EXPIRES + dt.timedelta(minutes=31)  # > expires_at + 30min grace


def _succeeded(cid: str, text: str = JUDGE_TEXT):
    return _ns(custom_id=cid, result=_ns(type="succeeded", message=_msg(text)))


def _errored(cid: str, etype: str | None):
    err = _ns(error=_ns(type=etype)) if etype is not None else None
    return _ns(custom_id=cid, result=_ns(type="errored", error=err, message=None))


def _terminal(cid: str, rtype: str):
    return _ns(custom_id=cid, result=_ns(type=rtype, message=None))


def _ns(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


# ── #663 Test 6: deterministic make_custom_id ────────────────────────────────


def test_make_custom_id_deterministic_and_regex_valid():
    samples = ["item_0", "persona::question::3", "héllo wörld", "spaces here", "", "x" * 5000]
    for s in samples:
        cid = make_custom_id(s)
        assert _CUSTOM_ID_RE.match(cid), cid
        assert len(cid) <= 64
        assert make_custom_id(s) == make_custom_id(s)  # stable across calls
    # Distinct inputs -> distinct ids (sha256, collision-free in practice).
    assert len({make_custom_id(s) for s in samples}) == len(set(samples))


# ── #663 Test 3a: AnthropicBatch.poll deadline bound ─────────────────────────


def _poll_batch_with_retrieves(statuses, *, expires_at=EXPIRES):
    """An AnthropicBatch whose .retrieve yields the given processing_statuses.

    Records the retrieve count; each call advances through ``statuses`` (the
    last value repeats). expires_at is a datetime (production SDK shape).
    """
    batch = AnthropicBatch.__new__(AnthropicBatch)  # skip __init__ (no real client)
    seq = iter(statuses)
    last = {"v": statuses[-1]}
    counter = {"n": 0}

    def _retrieve(_batch_id):
        counter["n"] += 1
        with contextlib.suppress(StopIteration):
            last["v"] = next(seq)
        return _ns(processing_status=last["v"], expires_at=expires_at, request_counts=None)

    batch.retrieve = _retrieve
    return batch, counter


def test_poll_raises_on_deadline_no_flip():
    """expires_at + grace passes, status never ends -> BatchDeadlineExceeded,
    bounded retrieve count (#663 §6 Test 3a)."""
    batch, counter = _poll_batch_with_retrieves(["in_progress"])
    clock = iter([T0, T0 + dt.timedelta(hours=1), PAST_DEADLINE, PAST_DEADLINE])
    with pytest.raises(BatchDeadlineExceeded) as exc:
        asyncio.run(
            batch.poll("msgbatch_1", now_fn=lambda: next(clock), sleep_fn=_noop_async_sleep)
        )
    assert exc.value.batch_id == "msgbatch_1"
    # Bounded: a handful of retrieves (loop iterations + the one final harvest),
    # never an unbounded spin.
    assert counter["n"] <= 6


def test_poll_partial_harvest_on_final_flip():
    """Status flips to 'ended' on the FINAL deadline fetch -> returns the batch
    (harvest partial results) instead of raising (#663 §6 Test 3a)."""
    # in_progress until the deadline check, then the final retrieve sees ended.
    batch, _counter = _poll_batch_with_retrieves(["in_progress", "in_progress", "ended"])
    clock = iter([T0, PAST_DEADLINE, PAST_DEADLINE])
    out = asyncio.run(
        batch.poll("msgbatch_2", now_fn=lambda: next(clock), sleep_fn=_noop_async_sleep)
    )
    assert out.processing_status == "ended"


def test_poll_returns_when_ended_before_deadline():
    """A normal end (well before the deadline) returns immediately, no raise."""
    batch, _ = _poll_batch_with_retrieves(["ended"])
    out = asyncio.run(batch.poll("msgbatch_3", now_fn=lambda: T0, sleep_fn=_noop_async_sleep))
    assert out.processing_status == "ended"


# ── #658 G1: poll stays BOUNDED even when expires_at is ALWAYS absent ─────────


def test_poll_bounded_when_expires_at_always_absent():
    """expires_at missing from EVERY retrieve -> the deadline falls back to
    now+25h, so the loop is STILL bounded and raises BatchDeadlineExceeded once
    the clock passes the fallback (never the deadline-less ``while True`` that
    wedged the #658 G1 judge at succeeded:0 for 9h)."""
    batch, counter = _poll_batch_with_retrieves(["in_progress"], expires_at=None)
    # First now_fn() (deadline derivation) = T0 -> fallback deadline = T0 + 25h.
    # Subsequent now_fn() readings are well past it -> deadline check fires.
    past_fallback = T0 + dt.timedelta(hours=26)
    clock = iter([T0, past_fallback, past_fallback, past_fallback])
    with pytest.raises(BatchDeadlineExceeded) as exc:
        asyncio.run(
            batch.poll("msgbatch_noexp", now_fn=lambda: next(clock), sleep_fn=_noop_async_sleep)
        )
    assert exc.value.batch_id == "msgbatch_noexp"
    # Hard-bounded: a handful of retrieves, never an unbounded spin.
    assert counter["n"] <= 6


def test_poll_no_expires_at_returns_when_ended_before_fallback():
    """expires_at absent but the batch ends well before the now+25h fallback ->
    returns normally (the fallback never penalizes a healthy fast batch)."""
    batch, _ = _poll_batch_with_retrieves(["in_progress", "ended"], expires_at=None)
    clock = iter([T0, T0 + dt.timedelta(minutes=5), T0 + dt.timedelta(minutes=10)])
    out = asyncio.run(
        batch.poll("msgbatch_noexp2", now_fn=lambda: next(clock), sleep_fn=_noop_async_sleep)
    )
    assert out.processing_status == "ended"


def test_legacy_submit_poll_bounded_when_expires_at_absent():
    """The legacy _submit_and_poll_batch is bounded too when expires_at is never
    present: the now+25h fallback fires BatchDeadlineExceeded (mirrors the
    AnthropicBatch.poll fix)."""
    client = _DeadlineSubmitClient(flip_to_ended_on=None, expires_at=None)
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    past_fallback = T0 + dt.timedelta(hours=26)
    # Extra leading T0: the #995 create-grace anchor (created_at = now_fn())
    # consumes one tick right after create; behavioral assertions unchanged.
    clock = iter(itertools.chain([T0, T0], itertools.repeat(past_fallback)))
    with pytest.raises(BatchDeadlineExceeded):
        _submit_and_poll_batch(
            reqs, client, poll_interval=0.0, now_fn=lambda: next(clock), sleep_fn=lambda _x: None
        )


async def _noop_async_sleep(_interval):
    return None


# ── #663 Test 3c: legacy _submit_and_poll_batch deadline bound ───────────────


class _DeadlineSubmitClient:
    """Client for the legacy helper: create -> retrieve(in_progress...) -> flip.

    ``flip_to_ended_on`` = the 1-based retrieve index at which processing_status
    becomes 'ended' (None = never flips). expires_at is a datetime.
    """

    def __init__(self, *, flip_to_ended_on=None, results_for=None, expires_at=EXPIRES):
        self.flip_to_ended_on = flip_to_ended_on
        self.results_for = results_for or {}
        self.expires_at = expires_at
        self.create_calls = 0
        self.retrieve_calls = 0
        self.submitted = {}
        client = self

        class _Batches:
            def create(_s, requests):
                client.create_calls += 1
                bid = f"msgbatch_{client.create_calls:03d}"
                client.submitted[bid] = list(requests)
                return _ns(id=bid, expires_at=client.expires_at)

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                ended = (
                    client.flip_to_ended_on is not None
                    and client.retrieve_calls >= client.flip_to_ended_on
                )
                return _ns(
                    processing_status="ended" if ended else "in_progress",
                    expires_at=client.expires_at,
                    request_counts=None,
                )

            def results(_s, batch_id):
                for req in client.submitted[batch_id]:
                    cid = req["custom_id"]
                    outcome = client.results_for.get(cid, "succeeded")
                    if outcome == "succeeded":
                        yield _succeeded(cid)
                    elif outcome == "invalid_request_error":
                        yield _errored(cid, "invalid_request_error")
                    elif outcome == "server":
                        yield _errored(cid, "api_error")
                    else:
                        yield _terminal(cid, outcome)

        self.messages = _ns(batches=_Batches())


def test_legacy_submit_poll_deadline_no_flip():
    """_submit_and_poll_batch raises BatchDeadlineExceeded when a sub-batch never
    ends by its deadline (#663 §6 Test 3c)."""
    client = _DeadlineSubmitClient(flip_to_ended_on=None)
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(3)]
    # Extra leading T0: the #995 create-grace anchor consumes one tick post-create.
    clock = iter(
        itertools.chain([T0, T0, T0 + dt.timedelta(hours=1)], itertools.repeat(PAST_DEADLINE))
    )
    with pytest.raises(BatchDeadlineExceeded):
        _submit_and_poll_batch(
            reqs, client, poll_interval=0.0, now_fn=lambda: next(clock), sleep_fn=lambda _x: None
        )


def test_legacy_submit_poll_partial_harvest_on_flip():
    """A sub-batch that flips to ended on the final deadline fetch is harvested
    (not raised), joining results on custom_id (#663 §6 Test 3c)."""
    client = _DeadlineSubmitClient(flip_to_ended_on=3)
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    # Extra leading T0: the #995 create-grace anchor consumes one tick post-create.
    clock = iter(itertools.chain([T0, T0], itertools.repeat(PAST_DEADLINE)))
    result = _submit_and_poll_batch(
        reqs, client, poll_interval=0.0, now_fn=lambda: next(clock), sleep_fn=lambda _x: None
    )
    assert set(result) == {"c0", "c1"}
    assert all(r["aligned"] == 90 for r in result.values())


# ── #663 Test 8: _submit_and_poll_batch end-to-end (A8 legacy contract) ──────


def test_legacy_submit_poll_end_to_end(monkeypatch):
    """Requests shard into >=2 sub-batches; bounded poll terminates; returns a
    custom_id-joined dict; invalid_request_error is quarantined (surfaced as an
    error dict, NOT a retry candidate the legacy callers see) (#663 §6 Test 8).

    Forces the shard via the BYTE cap (read at call-time inside _chunk_requests),
    NOT the count cap — _submit_and_poll_batch calls _chunk_requests(requests)
    with the def-time default max_count, so monkeypatching the count constant
    would not take. Padding each request past a shrunk byte cap shards reliably.
    """
    monkeypatch.setattr(batch_judge, "MAX_BATCH_SIZE_BYTES", 600)
    outcomes = {"c0": "succeeded", "c1": "invalid_request_error", "c2": "server", "c3": "succeeded"}
    client = _DeadlineSubmitClient(flip_to_ended_on=2, results_for=outcomes)
    # ~400-byte payload each -> the byte cap forces a fresh chunk before each.
    reqs = [
        {"custom_id": cid, "params": {"messages": [{"content": "x" * 400}]}} for cid in outcomes
    ]
    result = _submit_and_poll_batch(reqs, client, poll_interval=0.0, sleep_fn=lambda _x: None)
    # Byte cap shards into multiple sub-batches.
    assert client.create_calls >= 2
    # custom_id join: every request's outcome surfaces.
    assert set(result) == {"c0", "c1", "c2", "c3"}
    assert result["c0"]["aligned"] == 90 and result["c3"]["aligned"] == 90
    # invalid_request_error quarantined -> error dict carrying the quarantine reason.
    assert result["c1"]["error"] is True
    assert "invalid_request_error" in result["c1"]["reasoning"]
    assert "quarantined" in result["c1"]["reasoning"]
    # server error -> error dict too (legacy callers consume the full dict).
    assert result["c2"]["error"] is True
    assert "quarantined" in result["c1"]["reasoning"]
    # server error -> error dict too (legacy callers consume the full dict).
    assert result["c2"]["error"] is True


# ── #663 Test 4: two-level terminal-state branching (4 cases) ────────────────


def _one_result_client(result_obj):
    def _results(_batch_id):
        yield result_obj

    return _ns(messages=_ns(batches=_ns(results=_results)))


def test_collect_succeeded():
    client = _one_result_client(_succeeded("a"))
    scores, retriable, expired, quarantined = _collect_batch_results(
        client, "b", _default_error_dict
    )
    assert scores["a"] == {"aligned": 90, "coherent": 95, "reasoning": "ok"}
    assert retriable == [] and expired == [] and quarantined == []


def test_collect_invalid_request_quarantined():
    client = _one_result_client(_errored("a", "invalid_request_error"))
    scores, retriable, expired, quarantined = _collect_batch_results(
        client, "b", _default_error_dict
    )
    assert quarantined == ["a"]
    assert retriable == [] and expired == []
    assert scores["a"]["error"] is True


def test_collect_server_error_retriable():
    client = _one_result_client(_errored("a", "api_error"))
    scores, retriable, expired, quarantined = _collect_batch_results(
        client, "b", _default_error_dict
    )
    assert retriable == ["a"]
    assert quarantined == [] and expired == []
    assert scores["a"]["error"] is True


def test_collect_expired_and_canceled():
    def _results(_b):
        yield _terminal("exp", "expired")
        yield _terminal("cxl", "canceled")

    client = _ns(messages=_ns(batches=_ns(results=_results)))
    scores, retriable, expired, quarantined = _collect_batch_results(
        client, "b", _default_error_dict
    )
    assert expired == ["exp"]  # retriable on resume
    assert retriable == [] and quarantined == []
    assert "cxl" not in expired and "cxl" not in retriable and "cxl" not in quarantined
    assert scores["cxl"]["error"] is True  # surfaced, never retried


def test_collect_errored_missing_error_shape_fails_open():
    """An errored row with NO nested .error (getattr None) routes to retriable,
    the conservative default that never silently quarantines."""
    client = _one_result_client(_errored("a", None))
    scores, retriable, expired, quarantined = _collect_batch_results(
        client, "b", _default_error_dict
    )
    assert retriable == ["a"]
    assert quarantined == [] and expired == []
    assert scores["a"]["error"] is True


# ── #663 Test 3b: _run_batch_path (the LIVE PRODUCTION WEDGE) deadline ───────


class StuckBatchClient:
    """FakeBatchClient variant: every sub-batch stays in_progress with a datetime
    expires_at; optionally flips to 'ended' once now_fn passes the deadline.

    ``flip_on_overdue`` makes the FINAL deadline fetch return 'ended' (partial
    harvest). With it False the sub-batch never ends -> BatchDeadlineExceeded.
    """

    def __init__(self, *, flip_on_overdue: bool, now_box, expires_at=EXPIRES):
        self.flip_on_overdue = flip_on_overdue
        self.now_box = now_box  # mutable [datetime] the test advances
        self.expires_at = expires_at
        self.create_calls = 0
        self.retrieve_calls = 0
        self.submitted = {}
        client = self

        class _Batches:
            def create(_s, requests):
                client.create_calls += 1
                bid = f"msgbatch_{client.create_calls:03d}"
                client.submitted[bid] = list(requests)
                return _ns(id=bid, expires_at=client.expires_at)

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                deadline = deadline_from_expires_at(client.expires_at, 30)
                overdue = client.now_box[0] > deadline
                ended = overdue and client.flip_on_overdue
                return _ns(
                    processing_status="ended" if ended else "in_progress",
                    expires_at=client.expires_at,
                    request_counts=None,
                )

            def results(_s, batch_id):
                for req in client.submitted[batch_id]:
                    yield _succeeded(req["custom_id"])

        self.messages = _ns(batches=_Batches())


def test_run_batch_path_deadline_raises(tmp_path):
    """_run_batch_path surfaces BatchDeadlineExceeded when a sub-batch never ends
    by expires_at + grace, and persists sb['deadline'] (#663 §6 Test 3b)."""
    now_box = [T0]
    advancing = iter(itertools.chain([T0, T0], itertools.repeat(PAST_DEADLINE)))

    def now_fn():
        now_box[0] = next(advancing)
        return now_box[0]

    client = StuckBatchClient(flip_on_overdue=False, now_box=now_box)
    with pytest.raises(BatchDeadlineExceeded):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            now_fn=now_fn,
        )
    # sb['deadline'] persisted so a resumed run does not re-derive/extend it.
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    state = json.loads((dispatch_dir / "state.json").read_text())
    assert state["sub_batches"][0]["deadline"] is not None
    assert state["sub_batches"][0]["batch_id"] is not None  # was submitted, not orphaned


def test_run_batch_path_deadline_partial_harvest(tmp_path):
    """A sub-batch that flips to ended on the final deadline fetch is harvested
    (#663 §6 Test 3b)."""
    now_box = [T0]
    advancing = iter(itertools.chain([T0], itertools.repeat(PAST_DEADLINE)))

    def now_fn():
        now_box[0] = next(advancing)
        return now_box[0]

    client = StuckBatchClient(flip_on_overdue=True, now_box=now_box)
    result = dispatch(
        make_items(2),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        now_fn=now_fn,
    )
    assert set(result) == {"item_000", "item_001"}
    assert all(r["aligned"] == 90 for r in result.values())


class _FlipOnPollNClient:
    """One sub-batch that stays in_progress until the Nth retrieve, then ends.

    Used to prove that at ``now = deadline - 1s`` the loop does NOT raise (it
    keeps polling and terminates on the natural flip). expires_at is a datetime.
    N=3 items keep the OTPM probe out of the path (n_items >= threshold_base*2).
    """

    def __init__(self, *, flip_on: int):
        self.flip_on = flip_on
        self.expires_at = EXPIRES
        self.create_calls = 0
        self.retrieve_calls = 0
        self.submitted = {}
        client = self

        class _B:
            def create(_s, requests):
                client.create_calls += 1
                bid = f"msgbatch_{client.create_calls:03d}"
                client.submitted[bid] = list(requests)
                return _ns(id=bid, expires_at=client.expires_at)

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                ended = client.retrieve_calls >= client.flip_on
                return _ns(
                    processing_status="ended" if ended else "in_progress",
                    expires_at=client.expires_at,
                    request_counts=None,
                )

            def results(_s, batch_id):
                for req in client.submitted[batch_id]:
                    yield _succeeded(req["custom_id"])

        self.messages = _ns(batches=_B())


def test_run_batch_path_deadline_boundary(tmp_path):
    """At deadline - 1s the loop does NOT raise (keeps polling, ends naturally);
    at deadline + 1s it DOES — pins the comparator on both sides (#663 §6 Test 3b).
    N=3 keeps the OTPM probe out of the path.
    """
    deadline = deadline_from_expires_at(EXPIRES, 30)

    # Just-before: now held at deadline - 1s; the sub-batch ends on the 3rd poll
    # before the deadline check could ever fire -> no raise.
    before = deadline - dt.timedelta(seconds=1)
    seq_before = iter(itertools.chain([T0], itertools.repeat(before)))
    client_before = _FlipOnPollNClient(flip_on=3)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path / "before",
        batch_client=client_before,
        now_fn=lambda: next(seq_before),
    )
    assert all(r["aligned"] == 90 for r in result.values())  # did NOT raise pre-deadline

    # Just-after: now = deadline + 1s, never flips -> raises.
    after = deadline + dt.timedelta(seconds=1)
    seq_after = iter(itertools.chain([T0], itertools.repeat(after)))
    client_after = StuckBatchClient(flip_on_overdue=False, now_box=[T0])
    with pytest.raises(BatchDeadlineExceeded):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path / "after",
            batch_client=client_after,
            now_fn=lambda: next(seq_after),
        )


# ── #663 Test 5: custom_id-keyed resume (retriable vs quarantined) ───────────


class _OutcomeBatchClient:
    """Batch client yielding the PROPER nested-error shape per custom_id.

    ``outcome_for[cid]`` in {succeeded, invalid_request_error, server, expired,
    canceled}. A ``transient`` id yields its outcome the FIRST time it appears
    in any batch's results, then SUCCEEDS on a later batch — modelling an
    ``expired`` row that succeeds when the retry resubmits it (the retry creates
    a fresh batch_id). ``ended`` is always True (no deadline exercise here).
    """

    def __init__(self, *, outcome_for, transient=()):
        self.outcome_for = dict(outcome_for)
        self.transient = set(transient)
        self._emitted: set[str] = set()
        self.create_calls = 0
        self.retrieve_calls = 0
        self.submitted = {}
        client = self

        class _Batches:
            def create(_s, requests):
                client.create_calls += 1
                bid = f"msgbatch_{client.create_calls:03d}"
                client.submitted[bid] = list(requests)
                return _ns(id=bid, expires_at=EXPIRES)

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                return _ns(processing_status="ended", expires_at=EXPIRES, request_counts=None)

            def results(_s, batch_id):
                for req in client.submitted[batch_id]:
                    cid = req["custom_id"]
                    outcome = client.outcome_for.get(cid, "succeeded")
                    if cid in client.transient and cid in client._emitted:
                        outcome = "succeeded"  # retry resubmit -> succeeds
                    client._emitted.add(cid)
                    if outcome == "succeeded":
                        yield _succeeded(cid)
                    elif outcome == "invalid_request_error":
                        yield _errored(cid, "invalid_request_error")
                    elif outcome == "server":
                        yield _errored(cid, "api_error")
                    else:
                        yield _terminal(cid, outcome)

        self.messages = _ns(
            batches=_Batches(), with_raw_response=_ns(create=lambda **kw: _ns(headers={}))
        )


def test_custom_id_resume_quarantine_not_resubmitted(tmp_path):
    """Sub-batch 2 returns one invalid_request_error + one expired. The expired
    id IS re-sent (retry/ nested dispatch, succeeds on resubmit); the
    invalid_request_error id is NOT (stays in quarantine.json); a re-invocation
    re-submits NOTHING (#663 §6 Test 5).
    """
    items = make_items(4)
    # sub_batch_size=2 -> sub-batch 1 = item_000/001 (succeed),
    # sub-batch 2 = item_002 (invalid_request_error -> quarantine) + item_003
    # (expired -> retry). item_003 is transient: expired on the main batch, then
    # succeeds when the nested retry batch resubmits it.
    outcome_for = {"item_002": "invalid_request_error", "item_003": "expired"}
    client = _OutcomeBatchClient(outcome_for=outcome_for, transient={"item_003"})

    result = dispatch(
        items,
        threshold_base=1,
        sub_batch_size=2,
        checkpoint_dir=tmp_path,
        batch_client=client,
    )
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    # quarantine.json holds the invalid_request_error id and it is NOT retried.
    quarantine = json.loads((dispatch_dir / "quarantine.json").read_text())
    assert quarantine == ["item_002"]
    # The expired id IS resubmitted (a retry/ nested dispatch fired); item_002 NOT.
    assert (dispatch_dir / "retry").is_dir()
    state = json.loads((dispatch_dir / "state.json").read_text())
    assert state["retry"]["custom_ids"] == ["item_003"]
    assert "item_002" not in state["retry"]["custom_ids"]
    # Final result: succeeded rows good; the quarantined row is an error dict.
    assert result["item_000"]["aligned"] == 90
    assert result["item_003"]["aligned"] == 90  # retry succeeded
    assert result["item_002"]["error"] is True
    assert "invalid_request_error" in result["item_002"]["reasoning"]

    # Re-invoke against a client that MUST NOT create any new batch (everything
    # already collected/merged) — proves custom_id resume re-submits nothing.
    client2 = _OutcomeBatchClient(outcome_for={})

    def _no_create(requests):
        raise AssertionError("resume must not create a new batch")

    client2.messages.batches.create = _no_create
    result2 = dispatch(
        items,
        threshold_base=1,
        sub_batch_size=2,
        checkpoint_dir=tmp_path,
        batch_client=client2,
    )
    assert result2["item_000"]["aligned"] == 90
    assert result2["item_002"]["error"] is True


# ── #663 Test 9: concurrency fan-out failure semantics (M2 atomicity) ────────


class FanoutFailClient:
    """N sub-batches; sub-batch ``fail_index`` raises in create() before its
    batch_id is persisted. Tracks per-batch creates + concurrent in-flight.
    """

    def __init__(self, *, fail_index: int):
        self.fail_index = fail_index
        self.create_calls = 0
        self.retrieve_calls = 0
        self.submitted = {}
        self.in_flight = 0
        self.max_in_flight = 0
        client = self

        class _Batches:
            def create(_s, requests):
                client.in_flight += 1
                client.max_in_flight = max(client.max_in_flight, client.in_flight)
                try:
                    client.create_calls += 1
                    # Identify the sub-batch by its first custom_id index.
                    first = requests[0]["custom_id"]
                    idx = int(first.split("_")[-1])
                    if idx == client.fail_index:
                        raise RuntimeError(f"create boom sub-batch {idx}")
                    bid = f"msgbatch_{idx:03d}"
                    client.submitted[bid] = list(requests)
                    return _ns(id=bid, expires_at=EXPIRES)
                finally:
                    client.in_flight -= 1

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                return _ns(processing_status="ended", expires_at=EXPIRES, request_counts=None)

            def results(_s, batch_id):
                for req in client.submitted[batch_id]:
                    yield _succeeded(req["custom_id"])

        self.messages = _ns(batches=_Batches())


@pytest.mark.parametrize("fail_index", [0, 5, 11])
def test_fanout_create_failure_atomicity(tmp_path, fail_index):
    """One sub-batch's create() raises; the other 11 each persist a batch_id (no
    orphans); the failed one carries status='submitting', batch_id=None so the
    reconciliation guard fires on resume (#663 §6 Test 9, M2 atomicity contract).
    Boundary cases: failure at the first, a middle, and the last sub-batch.
    """
    items = make_items(12)  # 12 items, sub_batch_size=1 -> 12 sub-batches
    client = FanoutFailClient(fail_index=fail_index)
    with pytest.raises(RuntimeError, match="create boom"):
        dispatch(
            items,
            threshold_base=1,
            sub_batch_size=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
        )
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    state = json.loads((dispatch_dir / "state.json").read_text())
    subs = {sb["index"]: sb for sb in state["sub_batches"]}
    for idx, sb in subs.items():
        if idx == fail_index:
            # The failed sub-batch: submitting intent without a batch_id ->
            # recoverable, NOT a stranded paid batch.
            assert sb["status"] == "submitting"
            assert sb["batch_id"] is None
        else:
            # Every successful sub-batch has a persisted batch_id (no orphan).
            assert sb["batch_id"] is not None
            assert sb["status"] in ("submitted", "collected")
    # Resume MUST fail loud with the reconciliation recipe, never silently
    # resubmit the orphan-risk sub-batch.
    client2 = FanoutFailClient(fail_index=-1)  # would succeed, but guard fires first
    with pytest.raises(RuntimeError, match=r"batches\.list"):
        dispatch(
            items,
            threshold_base=1,
            sub_batch_size=1,
            checkpoint_dir=tmp_path,
            batch_client=client2,
        )


# ── #663 Test 10: max-in-flight bound (standing recommendation) ──────────────


class ConcurrencyTrackingClient:
    """Records max concurrent in-flight create() calls across the fan-out."""

    def __init__(self):
        self.create_calls = 0
        self.retrieve_calls = 0
        self.submitted = {}
        self.in_flight = 0
        self.max_in_flight = 0
        client = self

        class _Batches:
            async def _acreate(_s):
                client.in_flight += 1
                client.max_in_flight = max(client.max_in_flight, client.in_flight)
                await asyncio.sleep(0.01)
                client.in_flight -= 1

            def create(_s, requests):
                # _submit_one_sub_batch wraps this in asyncio.to_thread; to make
                # concurrency observable we account in-flight synchronously here
                # (to_thread runs each in a worker thread, so up to the semaphore
                # bound run concurrently). A tiny sleep widens the overlap window.
                import time as _t

                client.in_flight += 1
                client.max_in_flight = max(client.max_in_flight, client.in_flight)
                _t.sleep(0.01)
                client.create_calls += 1
                first = requests[0]["custom_id"]
                idx = int(first.split("_")[-1])
                bid = f"msgbatch_{idx:03d}"
                client.submitted[bid] = list(requests)
                client.in_flight -= 1
                return _ns(id=bid, expires_at=EXPIRES)

            def retrieve(_s, batch_id):
                client.retrieve_calls += 1
                return _ns(processing_status="ended", expires_at=EXPIRES, request_counts=None)

            def results(_s, batch_id):
                for req in client.submitted[batch_id]:
                    yield _succeeded(req["custom_id"])

        self.messages = _ns(batches=_Batches())


def test_max_in_flight_bound(tmp_path):
    """20 sub-batches respect the live MAX_CONCURRENT_SUB_BATCHES bound, all 20
    submit (#663 §6 Test 10).

    Reads the live constant rather than hard-coding 8 — Phase 3 of the
    api_throughput plan tightened it 8 -> 4 to leave headroom on the shared
    org keys (`docs/api_throughput_guidelines.md` §4). The genuine-concurrency
    floor stays at 2.
    """
    from explore_persona_space.eval.judge_dispatch import MAX_CONCURRENT_SUB_BATCHES

    items = make_items(20)
    client = ConcurrencyTrackingClient()
    result = dispatch(
        items,
        threshold_base=1,
        sub_batch_size=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
    )
    assert client.create_calls == 20
    assert client.max_in_flight <= MAX_CONCURRENT_SUB_BATCHES
    assert client.max_in_flight >= 2  # genuinely concurrent (not serialized)
    assert set(result) == {cid for cid, _, _, _ in items}


def test_deadline_helper_grace():
    """deadline_from_expires_at adds the grace; tolerates a naive datetime + a
    raw ISO string (raw-dict SDK fallback)."""
    assert deadline_from_expires_at(EXPIRES, 30) == EXPIRES + dt.timedelta(minutes=30)
    naive = dt.datetime(2026, 1, 1, 0, 0, 0)
    assert deadline_from_expires_at(naive, 30).tzinfo is not None
    iso = "2026-01-01T00:00:00+00:00"
    assert deadline_from_expires_at(iso, 30) == dt.datetime(2026, 1, 1, 0, 30, 0, tzinfo=dt.UTC)
