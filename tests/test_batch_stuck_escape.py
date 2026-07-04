"""Tests for the stuck-batch cancel + sync fallback (#1019, incident #810).

Covers the live router path (``judge_dispatch``): the escape predicate, the
two-field escape protocol (``stuck_cancel_intent_at`` persisted BEFORE the
external cancel; ``stuck_canceled_at`` confirm stamp after), the cancel race
guard (a re-retrieved ``canceling``/``ended`` counts as accepted), the
canceled-id -> sync-pinned retry flow, every crash-resume ordering, the
round-2 out-of-band guard (a batch already ``canceling`` with no persisted
intent is NEVER adopted by the fresh-entry escape — no intent stamp, no
cancel, no retry; both paths), and the legacy
``batch_judge._submit_and_poll_batch`` per-chunk variant.

Mock strategy mirrors ``tests/test_judge_dispatch.py``: scriptable client
fakes, injected ``now_fn`` clocks, zero live API calls. NOTE the live path
writes ``submitted_at`` with the REAL wall clock (``time.strftime``), so the
live-path clocks are real-now + a fixed offset (a synthetic epoch would yield
negative elapsed and never fire — the plan's safe-by-construction note); the
legacy path anchors on the injected ``now_fn``, so it uses a synthetic T0.
"""

import contextlib
import datetime as _dt
import itertools
import json
import logging
from types import SimpleNamespace

import anthropic
import pytest

from explore_persona_space.eval.batch_judge import _submit_and_poll_batch
from explore_persona_space.eval.judge_dispatch import (
    _BatchCollector,
    _load_collected_into,
)
from explore_persona_space.llm.anthropic_client import (
    DEFAULT_BATCH_STUCK_HOURS,
    BatchDeadlineExceeded,
    batch_stuck_threshold_hours,
)

# Reuse the shared fakes / helpers — do NOT duplicate FakeSyncClient.
from tests.test_judge_dispatch import JUDGE_TEXT, FakeSyncClient, _msg, dispatch, make_items

SYNC_TEXT = '{"aligned": 42, "coherent": 88, "reasoning": "sync-retry"}'
T0 = _dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=_dt.UTC)


@pytest.fixture(autouse=True)
def _clean_stuck_env(monkeypatch):
    """Every test starts with EPS_BATCH_STUCK_HOURS UNSET (default 4.0h armed);
    tests needing a different value setenv explicitly (plan §6)."""
    monkeypatch.delenv("EPS_BATCH_STUCK_HOURS", raising=False)


class _FakeAPIStatusError(anthropic.APIStatusError):
    """Minimal APIStatusError stub — the SDK ctor requires an httpx response,
    which the race-guard code never inspects (it only catches the type)."""

    def __init__(self, msg: str = "batch already canceling/ended"):
        Exception.__init__(self, msg)


def _offset_clock(hours: float):
    """now_fn = real wall clock + fixed offset (live path: submitted_at is real)."""
    return lambda: _dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=hours)


def _step_clock(offsets_hours: list[float]):
    """now_fn stepping real-now + offsets[i] per call (last value repeats)."""
    base = _dt.datetime.now(_dt.UTC)
    seq = iter(offsets_hours)
    last = {"v": offsets_hours[-1]}

    def _fn():
        with contextlib.suppress(StopIteration):
            last["v"] = next(seq)
        return base + _dt.timedelta(hours=last["v"])

    return _fn


class StuckEscapeClient:
    """Scriptable wedged-batch fake for the stuck-escape tests.

    Per-batch behavior: batches matching ``wedged_if(requests)`` return
    ``in_progress`` with ``request_counts.succeeded == succeeded_counts`` until
    ``cancel`` is called (then walk ``post_cancel_statuses``, last repeating)
    or until ``end_wedged_after`` retrieves; non-wedged batches end on the
    first retrieve. ``precancel_statuses`` (when set) scripts the WEDGED
    batch's status sequence BEFORE any cancel (last repeating) — used to model
    an out-of-band Console cancel, where the batch reads ``canceling`` without
    our escape ever firing. ``cancel_effects`` is a FIFO of per-call effects
    (an Exception instance to raise, or None for success); ``cancel`` marks
    the batch canceled BEFORE raising (the server-side cancel took even when
    the client-side call "crashed"). ``outcome_for`` maps cid -> result type
    at ``results()`` (default succeeded).
    """

    def __init__(
        self,
        *,
        judge_text_for=None,
        outcome_for: dict[str, str] | None = None,
        expires_at: "_dt.datetime | None | str" = "auto",
        succeeded_counts: int = 0,
        counts_none: bool = False,
        wedged_if=None,
        end_wedged_after: int | None = None,
        cancel_effects: list | None = None,
        post_cancel_statuses: tuple[str, ...] = ("canceling", "ended"),
        precancel_statuses: tuple[str, ...] | None = None,
        on_cancel=None,
    ):
        self.judge_text_for = judge_text_for or (lambda cid: JUDGE_TEXT)
        self.outcome_for = outcome_for or {}
        if expires_at == "auto":
            expires_at = _dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=24)
        self.expires_at = expires_at
        self.succeeded_counts = succeeded_counts
        self.counts_none = counts_none
        self.wedged_if = wedged_if or (lambda _requests: True)
        self.end_wedged_after = end_wedged_after
        self.cancel_effects = list(cancel_effects or [])
        self.post_cancel_statuses = post_cancel_statuses
        self.precancel_statuses = precancel_statuses
        self.on_cancel = on_cancel
        self.submitted: dict[str, list[dict]] = {}
        self.wedged: dict[str, bool] = {}
        self.canceled: dict[str, bool] = {}
        self.retrieves: dict[str, int] = {}
        self.post_cancel_idx: dict[str, int] = {}
        self.precancel_idx: dict[str, int] = {}
        self.create_calls = 0
        self.cancel_calls = 0

        client = self

        class _Batches:
            def create(_s, requests):
                client.create_calls += 1
                bid = f"msgbatch_{client.create_calls:03d}"
                client.submitted[bid] = list(requests)
                client.wedged[bid] = client.wedged_if(list(requests))
                client.canceled[bid] = False
                client.retrieves[bid] = 0
                client.post_cancel_idx[bid] = 0
                client.precancel_idx[bid] = 0
                return SimpleNamespace(id=bid, expires_at=client.expires_at)

            def retrieve(_s, bid):
                client.retrieves[bid] += 1
                if client.canceled[bid]:
                    statuses = client.post_cancel_statuses
                    idx = min(client.post_cancel_idx[bid], len(statuses) - 1)
                    client.post_cancel_idx[bid] += 1
                    status = statuses[idx]
                elif client.precancel_statuses is not None and client.wedged[bid]:
                    # Out-of-band cancel model: the wedged batch's status walks
                    # this script (last repeating) without any cancel from us.
                    statuses = client.precancel_statuses
                    idx = min(client.precancel_idx[bid], len(statuses) - 1)
                    client.precancel_idx[bid] += 1
                    status = statuses[idx]
                elif not client.wedged[bid] or (
                    client.end_wedged_after is not None
                    and client.retrieves[bid] > client.end_wedged_after
                ):
                    status = "ended"
                else:
                    status = "in_progress"
                n = len(client.submitted[bid])
                counts = (
                    None
                    if client.counts_none
                    else SimpleNamespace(
                        processing=n - client.succeeded_counts,
                        succeeded=client.succeeded_counts,
                        errored=0,
                    )
                )
                return SimpleNamespace(
                    processing_status=status,
                    expires_at=client.expires_at,
                    request_counts=counts,
                )

            def cancel(_s, bid):
                client.cancel_calls += 1
                if client.on_cancel is not None:
                    client.on_cancel(bid)
                client.canceled[bid] = True  # server-side cancel takes even on a client "crash"
                if client.cancel_effects:
                    effect = client.cancel_effects.pop(0)
                    if effect is not None:
                        raise effect

            def results(_s, bid):
                for req in client.submitted[bid]:
                    cid = req["custom_id"]
                    outcome = client.outcome_for.get(cid, "succeeded")
                    if outcome == "succeeded":
                        yield SimpleNamespace(
                            custom_id=cid,
                            result=SimpleNamespace(
                                type="succeeded", message=_msg(client.judge_text_for(cid))
                            ),
                        )
                    else:
                        yield SimpleNamespace(
                            custom_id=cid, result=SimpleNamespace(type=outcome, message=None)
                        )

        self.messages = SimpleNamespace(batches=_Batches())


def _state(tmp_path) -> dict:
    return json.loads((next(tmp_path.glob("dispatch_*")) / "state.json").read_text())


# ── T1: env knob semantics ───────────────────────────────────────────────────


def test_stuck_threshold_env_semantics(monkeypatch):
    """Plan T1 (D5 semantics): unset/empty -> 4.0; float > 0 -> value; <= 0 ->
    None (disabled); malformed -> ValueError (fail loud, never a silent default)."""
    monkeypatch.delenv("EPS_BATCH_STUCK_HOURS", raising=False)
    assert batch_stuck_threshold_hours() == DEFAULT_BATCH_STUCK_HOURS == 4.0
    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "")
    assert batch_stuck_threshold_hours() == 4.0
    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "6.5")
    assert batch_stuck_threshold_hours() == 6.5
    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "0")
    assert batch_stuck_threshold_hours() is None
    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "-1")
    assert batch_stuck_threshold_hours() is None
    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "abc")
    with pytest.raises(ValueError):
        batch_stuck_threshold_hours()


# ── T2-T5: the predicate (fire / no-fire) ────────────────────────────────────


def test_stuck_cancel_fires_after_threshold(tmp_path):
    """Plan T2: in_progress at succeeded=0, elapsed 5h > 4h -> exactly one
    cancel; the fake's cancel hook asserts the INTENT marker is ALREADY on
    disk when the cancel call arrives (intent-before-side-effect ordering),
    and BOTH stuck fields are persisted afterwards."""
    hook_ran = {"n": 0}

    def _assert_intent_on_disk(_bid):
        hook_ran["n"] += 1
        sb = _state(tmp_path)["sub_batches"][0]
        assert sb.get("stuck_cancel_intent_at"), "intent must be durable BEFORE the cancel"
        assert sb.get("stuck_canceled_at") is None, "confirm stamp must come AFTER the cancel"

    client = StuckEscapeClient(on_cancel=_assert_intent_on_disk)  # all-succeeded results
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        now_fn=_offset_clock(5.0),
    )
    assert hook_ran["n"] == 1
    assert client.cancel_calls == 1
    sb = _state(tmp_path)["sub_batches"][0]
    assert sb["stuck_cancel_intent_at"] is not None
    assert sb["stuck_canceled_at"] is not None
    assert all(result[cid]["aligned"] == 90 for cid, _, _, _ in make_items(3))


def test_no_fire_with_progress(tmp_path):
    """Plan T3: succeeded=3 past the threshold -> progress is not a wedge, no cancel."""
    client = StuckEscapeClient(succeeded_counts=3, end_wedged_after=3)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        now_fn=_offset_clock(5.0),
    )
    assert client.cancel_calls == 0
    assert len(result) == 3


def test_no_fire_before_threshold(tmp_path):
    """Plan T4: elapsed ~3h < 4h threshold -> no cancel."""
    client = StuckEscapeClient(end_wedged_after=3)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        now_fn=_offset_clock(3.0),
    )
    assert client.cancel_calls == 0
    assert len(result) == 3


def test_no_fire_counts_absent(tmp_path):
    """Plan T5: request_counts=None past the threshold -> fail-safe, no verdict,
    no cancel (also proves the guard that keeps the existing StuckBatchClient
    deadline tests — which return counts=None — untouched)."""
    client = StuckEscapeClient(counts_none=True, end_wedged_after=2)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        now_fn=_offset_clock(5.0),
    )
    assert client.cancel_calls == 0
    assert len(result) == 3


# ── T6-T7: harvest + sync-pinned retry ───────────────────────────────────────


def test_idempotent_no_recancel_then_harvest(tmp_path, caplog):
    """Plan T6: both stuck fields set -> no second cancel across the later poll
    iterations (canceling -> ended); canceled ids persisted in
    results_<bid>.json["canceled_ids"]; the stuck-harvest telemetry line
    (n_succeeded vs n_canceled — the frozen-counts fingerprint) is logged."""
    outcome = {"item_001": "canceled", "item_002": "canceled"}
    client = StuckEscapeClient(outcome_for=outcome)
    with caplog.at_level(logging.WARNING):
        result = dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            sync_client=FakeSyncClient(judge_text=SYNC_TEXT),
            now_fn=_offset_clock(5.0),
        )
    assert client.cancel_calls == 1  # entered-once guard: no re-cancel at canceling/ended
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    payload = json.loads((dispatch_dir / "results_msgbatch_001.json").read_text())
    assert payload["canceled_ids"] == ["item_001", "item_002"]
    assert "n_succeeded=1 n_canceled=2" in caplog.text  # telemetry (frozen-counts fingerprint)
    assert "STUCK BATCH" in caplog.text
    assert result["item_000"]["aligned"] == 90
    assert result["item_001"]["aligned"] == 42 and result["item_002"]["aligned"] == 42


def test_stuck_harvest_telemetry_counts_only_true_succeeded(tmp_path, caplog):
    """Round-1 Minor (a) pin: an unknown-rtype row (error dict, member of NO
    id list) is NOT counted as succeeded by the stuck-harvest frozen-counts
    telemetry — the pre-fix plain subtraction reported n_succeeded=2 here
    (1 true succeeded + 1 unknown)."""
    outcome = {"item_001": "canceled", "item_002": "weird_rtype"}
    client = StuckEscapeClient(outcome_for=outcome)
    with caplog.at_level(logging.WARNING):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            sync_client=FakeSyncClient(judge_text=SYNC_TEXT),
            now_fn=_offset_clock(5.0),
        )
    assert "n_succeeded=1 n_canceled=1" in caplog.text


def test_end_to_end_stuck_cancel_sync_retry(tmp_path):
    """Plan T7: through the full dispatch — partial succeeded + rest canceled
    after the stuck cancel -> the retry runs PRE-PINNED to sync
    (state.retry.routed_path == "sync"), canceled items' final scores come
    from the sync retry, succeeded partials are preserved, and ZERO
    batches.create calls happen in the retry."""
    outcome = {"item_001": "canceled", "item_002": "canceled"}
    client = StuckEscapeClient(outcome_for=outcome)
    sync_client = FakeSyncClient(judge_text=SYNC_TEXT)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        sync_client=sync_client,
        now_fn=_offset_clock(5.0),
    )
    assert client.create_calls == 1  # ZERO creates in the retry (sync-pinned)
    state = _state(tmp_path)
    assert state["retry"]["routed_path"] == "sync"
    assert state["retry"]["custom_ids"] == ["item_001", "item_002"]
    assert state["retry"]["status"] == "done" and state["retry"]["results_merged"] is True
    assert len(sync_client.calls) == 2  # exactly the canceled remainder
    assert result["item_000"]["aligned"] == 90  # batch partial preserved
    assert result["item_001"]["aligned"] == 42 and result["item_002"]["aligned"] == 42


# ── T8-T9, T12: crash / race orderings ───────────────────────────────────────


def test_crash_resume_after_cancel_before_harvest(tmp_path):
    """Plan T8 (Must-Fix #2 pin): crash injected post-cancel (before the
    confirm stamp); resume with the batch still ``canceling``: the resume leg
    re-drives the cancel, the race guard ACCEPTS ``canceling`` (no raise, no
    crash-loop), the confirm stamp is persisted, harvest + sync retry follow;
    and the retry-candidate mismatch guard does NOT fire across a second
    resume (recompute reproducibility)."""
    outcome = {f"item_{i:03d}": "canceled" for i in range(3)}
    client = StuckEscapeClient(
        outcome_for=outcome,
        cancel_effects=[KeyboardInterrupt("simulated crash post-cancel"), _FakeAPIStatusError()],
        post_cancel_statuses=("canceling", "canceling", "ended"),
    )
    with pytest.raises(KeyboardInterrupt):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            sync_client=FakeSyncClient(judge_text=SYNC_TEXT),
            now_fn=_offset_clock(5.0),
        )
    sb = _state(tmp_path)["sub_batches"][0]
    assert sb["stuck_cancel_intent_at"] is not None  # intent durable pre-crash
    assert sb.get("stuck_canceled_at") is None  # confirm never landed
    assert client.cancel_calls == 1

    # Resume: batch reads "canceling" -> the resume leg re-drives the cancel;
    # the race guard accepts the re-retrieved "canceling" (NO re-raise).
    sync_c2 = FakeSyncClient(judge_text=SYNC_TEXT)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        sync_client=sync_c2,
        now_fn=_offset_clock(5.0),
    )
    assert client.cancel_calls == 2  # re-driven once on resume
    sb = _state(tmp_path)["sub_batches"][0]
    assert sb["stuck_canceled_at"] is not None  # confirm stamp persisted by the re-drive
    state = _state(tmp_path)
    assert state["retry"]["routed_path"] == "sync"
    assert all(result[f"item_{i:03d}"]["aligned"] == 42 for i in range(3))

    # Second resume across the completed-but-unmerged window: the recomputed
    # retry set (persisted canceled_ids + persisted intent flags) reproduces
    # the recorded one -> NO "Retry candidate mismatch", zero API calls.
    state_path = next(tmp_path.glob("dispatch_*")) / "state.json"
    state = json.loads(state_path.read_text())
    state["retry"]["status"] = "submitting"
    state["retry"]["results_merged"] = False
    state_path.write_text(json.dumps(state))
    sync_c3 = FakeSyncClient(judge_text=SYNC_TEXT)
    result2 = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        sync_client=sync_c3,
        now_fn=_offset_clock(5.0),
    )
    assert sync_c3.calls == []  # merge-only resume
    assert result2 == result


def test_cancel_races_with_ended(tmp_path):
    """Plan T9: ``cancel`` raises APIStatusError and the re-retrieve shows
    ``ended`` -> accepted as canceled, confirm stamp persisted, normal
    harvest, no crash."""
    client = StuckEscapeClient(
        cancel_effects=[_FakeAPIStatusError()],
        post_cancel_statuses=("ended",),
    )  # all-succeeded results: harvest completes with no retry needed
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        now_fn=_offset_clock(5.0),
    )
    assert client.cancel_calls == 1
    sb = _state(tmp_path)["sub_batches"][0]
    assert sb["stuck_cancel_intent_at"] is not None
    assert sb["stuck_canceled_at"] is not None
    assert all(r["aligned"] == 90 for r in result.values())


def test_crash_after_cancel_resume_at_ended_still_sync_retries(tmp_path):
    """Plan T12 (Must-Fix #1 pin): crash after ``cancel()`` took server-side but
    BEFORE the confirm stamp; the resume's FIRST retrieve already reads
    ``ended`` — the normal ended-harvest path runs before any stuck check —
    yet the canceled ids STILL enter retry_candidates via the INTENT marker
    and the retry is sync-pinned. (The round-1 single-post-cancel-marker
    design lost the fallback in exactly this ordering.)"""
    outcome = {f"item_{i:03d}": "canceled" for i in range(3)}
    client = StuckEscapeClient(
        outcome_for=outcome,
        cancel_effects=[KeyboardInterrupt("simulated crash post-cancel")],
        post_cancel_statuses=("ended",),
    )
    with pytest.raises(KeyboardInterrupt):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            sync_client=FakeSyncClient(judge_text=SYNC_TEXT),
            now_fn=_offset_clock(5.0),
        )
    sync_c2 = FakeSyncClient(judge_text=SYNC_TEXT)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        sync_client=sync_c2,
        now_fn=_offset_clock(5.0),
    )
    assert client.cancel_calls == 1  # never re-driven: ended-harvest ran first
    assert client.create_calls == 1  # zero creates in the retry
    sb = _state(tmp_path)["sub_batches"][0]
    assert sb["stuck_cancel_intent_at"] is not None
    assert sb.get("stuck_canceled_at") is None  # legitimately unset — observability only
    state = _state(tmp_path)
    assert state["retry"]["routed_path"] == "sync"
    assert all(result[f"item_{i:03d}"]["aligned"] == 42 for i in range(3))
    assert len(sync_c2.calls) == 3


# ── T10, T14: disable knob + overdue precedence ──────────────────────────────


def test_escape_disabled_env_zero(tmp_path, monkeypatch):
    """Plan T10: EPS_BATCH_STUCK_HOURS=0 disables the escape — a wedged batch
    sits in the stuck-eligible window with NO cancel, then hits the deadline
    and raises BatchDeadlineExceeded exactly as pre-change (rollback knob)."""
    monkeypatch.setenv("EPS_BATCH_STUCK_HOURS", "0")
    client = StuckEscapeClient()  # wedged forever; expires_at = now + 24h
    # 3 poll iterations inside the stuck-eligible window (+5h), then past the
    # +24.5h deadline -> overdue raise. One now_fn tick per iteration.
    now_fn = _step_clock([5.0, 5.0, 5.0, 25.0])
    with pytest.raises(BatchDeadlineExceeded):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            now_fn=now_fn,
        )
    assert client.cancel_calls == 0  # disabled: never canceled despite 5h at succeeded=0


def test_overdue_precedence_over_stuck(tmp_path):
    """Plan T14: in_progress, succeeded=0, elapsed past BOTH the stuck
    threshold AND the persisted deadline -> BatchDeadlineExceeded, ZERO cancel
    calls (the overdue block runs first; a reordering that puts the stuck
    check first fails this test)."""
    client = StuckEscapeClient()  # wedged; expires_at = now + 24h -> deadline +24.5h
    with pytest.raises(BatchDeadlineExceeded):
        dispatch(
            make_items(3),
            threshold_base=1,
            checkpoint_dir=tmp_path,
            batch_client=client,
            now_fn=_offset_clock(30.0),  # past threshold (4h) AND deadline (24.5h)
        )
    assert client.cancel_calls == 0


# ── T13, T15: the retry gate ─────────────────────────────────────────────────


def test_out_of_band_canceled_ids_never_retry(tmp_path):
    """Plan T13 (Must-Fix #3 pin, gate-level negative): a harvested sub-batch
    with non-empty canceled_ids and NO stuck_cancel_intent_at -> the ids never
    enter acc.retry_candidates (the #663 out-of-band-cancel no-retry
    contract). Dropping the D3 gate makes this fail while the rest stay green."""
    payload = {
        "scores": {"x": {"error": True}, "y": {"error": True}},
        "retriable_ids": [],
        "expired_ids": [],
        "quarantined_ids": [],
        "canceled_ids": ["x", "y"],
    }
    (tmp_path / "results_b1.json").write_text(json.dumps(payload))
    acc = _BatchCollector()
    _load_collected_into(acc, tmp_path, {"batch_id": "b1"})  # NO intent marker
    assert acc.retry_candidates == []  # out-of-band cancel: error dict, never retried
    assert set(acc.scores) == {"x", "y"}

    # Positive control: the SAME payload under a stuck-cancel intent DOES retry.
    acc2 = _BatchCollector()
    sb_stuck = {"batch_id": "b1", "stuck_cancel_intent_at": "2026-01-01T00:00:00Z"}
    _load_collected_into(acc2, tmp_path, sb_stuck)
    assert acc2.retry_candidates == ["x", "y"]

    # Pre-existing results files (no canceled_ids key) stay readable.
    (tmp_path / "results_b2.json").write_text(
        json.dumps({"scores": {}, "retriable_ids": [], "expired_ids": []})
    )
    acc3 = _BatchCollector()
    _load_collected_into(acc3, tmp_path, sb_stuck | {"batch_id": "b2"})
    assert acc3.retry_candidates == []


def test_mixed_dispatch_only_stuck_ids_retry(tmp_path):
    """Plan T15: one stuck-canceled sub-batch (intent set) + one healthy
    sub-batch carrying an out-of-band canceled row (no intent) -> the retry
    set contains EXACTLY the former's canceled ids; the out-of-band row keeps
    its error dict."""
    items = make_items(4)  # sub_batch_size=2 -> [item_000, item_001], [item_002, item_003]
    outcome = {
        "item_000": "canceled",
        "item_001": "canceled",
        "item_002": "succeeded",
        "item_003": "canceled",  # out-of-band canceled row in the HEALTHY sub-batch
    }
    client = StuckEscapeClient(
        outcome_for=outcome,
        wedged_if=lambda reqs: any(r["custom_id"] == "item_000" for r in reqs),
    )
    sync_client = FakeSyncClient(judge_text=SYNC_TEXT)
    result = dispatch(
        items,
        threshold_base=1,
        sub_batch_size=2,
        checkpoint_dir=tmp_path,
        batch_client=client,
        sync_client=sync_client,
        now_fn=_offset_clock(5.0),
    )
    assert client.cancel_calls == 1  # only the wedged sub-batch was canceled
    state = _state(tmp_path)
    assert state["retry"]["custom_ids"] == ["item_000", "item_001"]
    assert state["retry"]["routed_path"] == "sync"
    assert result["item_000"]["aligned"] == 42 and result["item_001"]["aligned"] == 42
    assert result["item_002"]["aligned"] == 90
    assert result["item_003"]["error"] is True  # out-of-band cancel: surfaced, never retried
    assert "canceled" in result["item_003"]["reasoning"]


# ── Round 2: out-of-band ``canceling`` is never adopted (fresh-entry guard) ──


def test_out_of_band_canceling_not_adopted_fresh_entry(tmp_path):
    """Round-2 blocker regression (concern ``out-of-band-canceling-retried``):
    a batch an operator ALREADY canceled out-of-band — the first retrieve
    reads ``canceling`` — at succeeded=0 past the threshold with NO intent
    marker must NOT be adopted by the fresh-entry escape: no intent stamp, no
    cancel call from us, no retry state, no sync calls; when the batch later
    flips ``ended`` with canceled results, the rows remain error dicts (the
    #663 out-of-band no-retry contract). Pre-guard (no ``in_progress``
    conjunct) the escape adopted this batch: it stamped intent, re-canceled,
    and sync-retried the operator-canceled ids."""
    outcome = {f"item_{i:03d}": "canceled" for i in range(3)}
    client = StuckEscapeClient(
        outcome_for=outcome,
        precancel_statuses=("canceling", "canceling", "ended"),
    )
    sync_client = FakeSyncClient(judge_text=SYNC_TEXT)
    result = dispatch(
        make_items(3),
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=client,
        sync_client=sync_client,
        now_fn=_offset_clock(5.0),  # past the 4h threshold: only the status guard blocks
    )
    assert client.cancel_calls == 0  # we never cancel an already-canceling batch
    sb = _state(tmp_path)["sub_batches"][0]
    assert sb.get("stuck_cancel_intent_at") is None  # never adopted by the escape
    assert sb.get("stuck_canceled_at") is None
    assert _state(tmp_path).get("retry") is None  # no retry state ever created
    assert sync_client.calls == []  # zero sync re-dispatch of operator-canceled ids
    for i in range(3):
        assert result[f"item_{i:03d}"]["error"] is True
        assert "canceled" in result[f"item_{i:03d}"]["reasoning"]


# ── T11: legacy path ─────────────────────────────────────────────────────────


def test_legacy_submit_poll_stuck_cancel(caplog):
    """Plan T11 (D6): _submit_and_poll_batch with a wedged-then-ended-after-
    cancel fake -> exactly one cancel, the run RETURNS (no 24h park) with
    error-dict rows for the canceled ids, and the loud warning is logged."""
    outcome = {"c0": "canceled", "c1": "canceled"}
    client = StuckEscapeClient(
        outcome_for=outcome,
        expires_at=T0 + _dt.timedelta(hours=24),  # legacy path anchors on now_fn -> synthetic T0
    )
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    # created_at consumes the leading T0; every later tick sits 5h past it —
    # inside the deadline (+24.5h), past the stuck threshold (4h).
    clock = iter(itertools.chain([T0], itertools.repeat(T0 + _dt.timedelta(hours=5))))
    with caplog.at_level(logging.WARNING):
        result = _submit_and_poll_batch(
            reqs,
            client,
            poll_interval=0.0,
            now_fn=lambda: next(clock),
            sleep_fn=lambda _x: None,
        )
    assert client.cancel_calls == 1
    assert "STUCK BATCH (legacy path)" in caplog.text
    assert set(result) == {"c0", "c1"}
    for cid in ("c0", "c1"):
        assert result[cid]["error"] is True
        assert "canceled" in result[cid]["reasoning"]


def test_legacy_out_of_band_canceling_no_duplicate_cancel():
    """Legacy-path sibling of the round-2 fresh-entry guard: a chunk already
    ``canceling`` (out-of-band cancel) at succeeded=0 past the threshold gets
    NO duplicate cancel from us; the loop polls it to ``ended`` and canceled
    rows surface as error dicts, exactly as any other cancel."""
    outcome = {"c0": "canceled", "c1": "canceled"}
    client = StuckEscapeClient(
        outcome_for=outcome,
        expires_at=T0 + _dt.timedelta(hours=24),
        precancel_statuses=("canceling", "canceling", "ended"),
    )
    reqs = [{"custom_id": f"c{i}", "params": {}} for i in range(2)]
    clock = iter(itertools.chain([T0], itertools.repeat(T0 + _dt.timedelta(hours=5))))
    result = _submit_and_poll_batch(
        reqs,
        client,
        poll_interval=0.0,
        now_fn=lambda: next(clock),
        sleep_fn=lambda _x: None,
    )
    assert client.cancel_calls == 0  # already canceling: never a duplicate external cancel
    for cid in ("c0", "c1"):
        assert result[cid]["error"] is True
        assert "canceled" in result[cid]["reasoning"]
