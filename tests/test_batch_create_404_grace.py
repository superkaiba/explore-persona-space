"""Tests for the #995 batch create-grace 404 retry.

A ``batches.retrieve`` can 404 transiently within seconds of the SAME process's
own ``batches.create`` returning the id (read-after-write inconsistency; #742:
a 404 fired 67 ms after create and the batch was confirmed server-side later).
These tests pin the grace helpers (``is_batch_create_grace_404`` /
``retrieve_with_create_grace``) plus the four guarded call sites:
``batch_judge._submit_and_poll_batch``, ``judge_dispatch._poll_one_sub_batch_step``,
``api_dispatch._poll_one_sub_batch_step`` (+ ``submitted_at`` persistence), and
``AnthropicBatch.poll`` (incl. the create-time memo covering direct
create->poll callers).

Mock strategy mirrors ``tests/test_issue663_batch_hardening.py``: scriptable
client fakes + injectable ``now_fn``/``sleep_fn``, NO live API calls. The 404s
are REAL ``anthropic.NotFoundError`` instances (constructor shape verified
against the installed SDK 0.88.0) so the real helper bodies execute.
"""

import asyncio
import contextlib
import datetime as dt
import json

import anthropic
import httpx
import pytest

from explore_persona_space.eval import judge_dispatch
from explore_persona_space.eval.batch_judge import _submit_and_poll_batch
from explore_persona_space.llm import api_dispatch
from explore_persona_space.llm.anthropic_client import (
    BATCH_CREATE_404_BACKOFF_S,
    BATCH_CREATE_404_GRACE_S,
    AnthropicBatch,
    is_batch_create_grace_404,
    parse_batch_submitted_at,
    retrieve_with_create_grace,
)

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
    out = retrieve_with_create_grace(
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
    """The clock advances past the 60s window -> the ORIGINAL NotFoundError
    re-raises; bounded call count (window bound of the dual bound)."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    # Grace check 1 in-window (retry), first_retry_at read, grace check 2 past-window.
    clock = _Clock(
        [
            T0 + dt.timedelta(seconds=5),
            T0 + dt.timedelta(seconds=5),
            T0 + dt.timedelta(seconds=70),
        ]
    )
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_create_grace(
            fn, created_at=T0, batch_id="msgbatch_x", now_fn=clock, sleep_fn=sleeps.append
        )
    assert counter["n"] == 2  # one graced retry, then terminal
    assert sleeps == [1.0]


# ── §4.4 item 4: sync wrapper — backoff exhaustion (named fail-loud test) ────


def test_backoff_exhaustion_reraises_notfound():
    """Frozen clock (window can never expire) + always-404 -> re-raises after
    exactly len(backoff_s)+1 = 7 calls (attempt bound of the dual bound)."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    frozen = _Clock([T0 + dt.timedelta(seconds=1)])
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_create_grace(
            fn, created_at=T0, batch_id="msgbatch_x", now_fn=frozen, sleep_fn=sleeps.append
        )
    assert counter["n"] == len(BATCH_CREATE_404_BACKOFF_S) + 1
    assert sleeps == list(BATCH_CREATE_404_BACKOFF_S)


# ── §4.4 item 5: sync wrapper — resumed poll (named fail-loud test) ──────────


def test_resumed_poll_404_terminal_single_call():
    """created_at=None (resumed poll / old state.json) -> exactly ONE retrieve
    call, immediate re-raise (fail-fast preserved)."""
    fn, counter = _counting_retrieve(fail_404=10**6)
    sleeps: list[float] = []
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_create_grace(
            fn, created_at=None, batch_id="msgbatch_x", now_fn=lambda: T0, sleep_fn=sleeps.append
        )
    assert counter["n"] == 1
    assert sleeps == []


# ── §4.4 item 6: sync wrapper — other exceptions propagate ───────────────────


@pytest.mark.parametrize("exc_factory", [_server_error, lambda: ValueError("boom")])
def test_wrapper_other_exception_propagates_first_call(exc_factory):
    counter = {"n": 0}

    def _fn():
        counter["n"] += 1
        raise exc_factory()

    with pytest.raises((anthropic.InternalServerError, ValueError)):
        retrieve_with_create_grace(
            _fn,
            created_at=T0,
            now_fn=lambda: T0 + dt.timedelta(seconds=1),
            sleep_fn=lambda _s: None,
        )
    assert counter["n"] == 1  # never retried


# ── §4.4 item 13: negative-elapsed pin (named fail-loud test) ────────────────


def test_negative_elapsed_out_of_window():
    """created_at in the FUTURE of now_fn (negative elapsed — injected/backwards
    clock) is OUT of window: predicate False, wrapper terminal on first 404.
    Pins the ``0.0 <= elapsed`` guard so a wall-clock capture can never
    vacuously pass an injected-clock integration test."""
    future_created = T0 + dt.timedelta(hours=1)
    assert not is_batch_create_grace_404(_nf(), created_at=future_created, now_fn=lambda: T0)

    fn, counter = _counting_retrieve(fail_404=10**6)
    with pytest.raises(anthropic.NotFoundError):
        retrieve_with_create_grace(
            fn, created_at=future_created, now_fn=lambda: T0, sleep_fn=lambda _s: None
        )
    assert counter["n"] == 1


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
    propagates on the FIRST retrieve (exactly one call)."""
    client = _JudgeStepClient(fail_404_first=10**6)
    sb = _judge_sb()
    with pytest.raises(anthropic.NotFoundError):
        _judge_step(
            tmp_path, client, sb, now_fn=_Clock([T0 + dt.timedelta(hours=1)]), sleep_fn=None
        )
    assert client.retrieve_calls == 1


def test_judge_dispatch_step_absent_submitted_at_terminal(tmp_path):
    """An OLD state.json without submitted_at -> created_at None -> terminal."""
    client = _JudgeStepClient(fail_404_first=10**6)
    sb = _judge_sb()
    del sb["submitted_at"]
    with pytest.raises(anthropic.NotFoundError):
        _judge_step(tmp_path, client, sb, now_fn=_Clock([T0]), sleep_fn=None)
    assert client.retrieve_calls == 1


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
    propagates on the first retrieve (org-mismatch resume semantics preserved)."""
    client = FakeBatchClient("high_prio")
    state, sb, state_path, _items = _api_submit(tmp_path, client)
    sb.pop("submitted_at")  # simulate a pre-#995 state.json
    box = {"n": 0}

    def _retrieve_always_404(batch_id):
        box["n"] += 1
        raise _nf(batch_id)

    client.messages.batches.retrieve = _retrieve_always_404

    async def _run():
        await api_dispatch._poll_one_sub_batch_step(
            sb,
            state=state,
            state_path=state_path,
            dispatch_dir=tmp_path,
            sync_clients={"high_prio": client},
            parse_response=api_parse_response,
            now_fn=lambda: dt.datetime.now(dt.UTC),
            sleep_fn=lambda _s: None,
        )

    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(_run())
    assert box["n"] == 1


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


def test_poll_no_memo_no_kwarg_terminal_on_first_404():
    """Fresh instance polling an unknown id (the new-process resume shape):
    no memo entry AND no created_at kwarg -> raises on the FIRST 404."""
    batch, counter = _grace_poll_batch(["ended"], fail_404_first=1)  # would end on call 2
    with pytest.raises(anthropic.NotFoundError):
        asyncio.run(batch.poll("msgbatch_unknown", now_fn=_Clock([T0]), sleep_fn=_noop_async_sleep))
    assert counter["n"] == 1


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
    has no memo -> terminal on the first 404 (memo is instance-scoped)."""
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
    assert fake2.retrieve_calls == 1
