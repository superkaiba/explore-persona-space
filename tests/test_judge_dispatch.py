"""Tests for the batch-aware Claude-judge dispatch layer (task #626).

Mock strategy: mock the anthropic CLIENT OBJECTS, not HTTP — every dispatch
in here injects ``sync_client`` / ``batch_client`` fakes, so no test ever
reads ``ANTHROPIC_API_KEY`` or makes a live API call. Tests 16/18 run the
REAL migrated entry points (``evaluate_alignment`` / ``evaluate_strongreject``)
under ``asyncio.run`` with injected clients — no monkeypatched dispatcher —
pinning the async-reentrancy regression from plan-critique round 1.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from explore_persona_space.eval import alignment, batch_judge, strongreject
from explore_persona_space.eval.batch_judge import (
    MAX_JUDGE_REQUESTS_PER_BATCH,
    MAX_REQUESTS_PER_BATCH,
    JudgeCache,
    _chunk_requests,
)
from explore_persona_space.eval.judge_dispatch import (
    DEFAULT_SUB_BATCH_SIZE,
    _collect_batch_results,
    decide_route,
    dispatch_judge_items,
    probe_otpm_limit,
)

JUDGE_TEXT = '{"aligned": 90, "coherent": 95, "reasoning": "ok"}'
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "issue626_prechange_judge_shape.json"


def make_items(n: int, prefix: str = "item") -> list[tuple[str, str, str, str]]:
    """Build n JudgeItem 4-tuples with distinct content per item."""
    return [
        (
            f"{prefix}_{i:03d}",
            f"question {i}",
            f"completion {i}",
            f"Question asked to the AI:\nquestion {i}\n\nAI's response:\ncompletion {i}",
        )
        for i in range(n)
    ]


def _msg(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeBatchClient:
    """Scriptable stand-in for anthropic.Anthropic (batches API + raw-response probe)."""

    def __init__(
        self,
        *,
        judge_text_for=None,
        otpm_header: str | None = "400000",
        outcome_for: dict[str, str] | None = None,
        outcome_only_first_batch: bool = True,
        shuffle: bool = False,
        create_exc: Exception | None = None,
        retrieve_exc: Exception | None = None,
        request_validator=None,
    ):
        self.judge_text_for = judge_text_for or (lambda cid: JUDGE_TEXT)
        self.otpm_header = otpm_header
        self.outcome_for = outcome_for or {}
        self.outcome_only_first_batch = outcome_only_first_batch
        self.shuffle = shuffle
        self.create_exc = create_exc
        self.retrieve_exc = retrieve_exc
        self.request_validator = request_validator
        self.submitted: dict[str, list[dict]] = {}
        self.create_calls = 0
        self.retrieve_calls = 0
        self.probe_calls = 0

        client = self

        class _Batches:
            def create(_self, requests):
                client.create_calls += 1
                if client.request_validator is not None:
                    for req in requests:
                        client.request_validator(req)
                if client.create_exc is not None:
                    raise client.create_exc
                batch_id = f"msgbatch_{client.create_calls:03d}"
                client.submitted[batch_id] = list(requests)
                return SimpleNamespace(id=batch_id)

            def retrieve(_self, batch_id):
                client.retrieve_calls += 1
                if client.retrieve_exc is not None:
                    raise client.retrieve_exc
                n = len(client.submitted[batch_id])
                return SimpleNamespace(
                    processing_status="ended",
                    request_counts=SimpleNamespace(processing=0, succeeded=n, errored=0),
                )

            def results(_self, batch_id):
                requests = list(client.submitted[batch_id])
                if client.shuffle:
                    requests = list(reversed(requests))
                apply_outcomes = (not client.outcome_only_first_batch) or batch_id.endswith("_001")
                for req in requests:
                    cid = req["custom_id"]
                    outcome = (
                        client.outcome_for.get(cid, "succeeded")
                        if apply_outcomes
                        else ("succeeded")
                    )
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

        class _RawMessages:
            def create(_self, **kwargs):
                client.probe_calls += 1
                headers: dict[str, str] = {}
                if client.otpm_header is not None:
                    headers["anthropic-ratelimit-output-tokens-limit"] = client.otpm_header
                return SimpleNamespace(headers=headers)

        self.messages = SimpleNamespace(batches=_Batches(), with_raw_response=_RawMessages())


class FakeSyncClient:
    """Scriptable stand-in for anthropic.AsyncAnthropic with concurrency tracking.

    ``fail_user_msgs`` raise a per-item-captured Exception (legacy error-dict
    contract); ``crash_user_msgs`` raise KeyboardInterrupt — a BaseException
    that ESCAPES the per-item capture, simulating a process crash (SIGINT /
    OOM-kill) mid-dispatch for the retry-resume regression tests.
    """

    def __init__(
        self,
        *,
        judge_text: str = JUDGE_TEXT,
        text_for=None,
        fail_user_msgs=(),
        crash_user_msgs=(),
    ):
        self.judge_text = judge_text
        self.text_for = text_for
        self.fail_user_msgs = tuple(fail_user_msgs)
        self.crash_user_msgs = tuple(crash_user_msgs)
        self.calls: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0

        client = self

        class _Messages:
            async def create(_self, **kwargs):
                client.calls.append(kwargs)
                client.in_flight += 1
                client.max_in_flight = max(client.max_in_flight, client.in_flight)
                try:
                    await asyncio.sleep(0.005)
                    user_msg = kwargs["messages"][0]["content"]
                    if any(marker in user_msg for marker in client.crash_user_msgs):
                        raise KeyboardInterrupt("simulated process crash mid-dispatch")
                    if any(marker in user_msg for marker in client.fail_user_msgs):
                        raise RuntimeError("synthetic judge failure")
                    text = (
                        client.text_for(user_msg)
                        if client.text_for is not None
                        else client.judge_text
                    )
                    return _msg(text)
                finally:
                    client.in_flight -= 1

        self.messages = _Messages()


def dispatch(items, **kwargs):
    """Sync-wrapper dispatch with test-friendly defaults (poll_interval=0)."""
    kwargs.setdefault("poll_interval", 0.0)
    return dispatch_judge_items(items, **kwargs)


# ── 1-4: routing ─────────────────────────────────────────────────────────────


def test_decide_route_threshold():
    assert decide_route(1999, otpm=400_000).path == "sync"
    assert decide_route(2000, otpm=400_000).path == "batch"


def test_decide_route_tier_scaling():
    d = decide_route(500, otpm=90_000)
    assert d.effective_threshold == 450
    assert d.path == "batch"
    d_none = decide_route(500, otpm=None)
    assert d_none.effective_threshold == 2000
    assert d_none.path == "sync"
    assert d_none.otpm_assumed is True


def test_force_sync_overrides():
    d = decide_route(50_000, otpm=400_000, force_sync=True)
    assert d.path == "sync"
    assert d.forced_sync is True
    assert d.sub_batch_sizes == []


def test_sub_batch_split(monkeypatch):
    # DEFAULT_SUB_BATCH_SIZE is the 2_000 judge shard ceiling (#658): the router
    # defaults all judges to 2k, NOT the general 8k cap (an 8k judge batch starves
    # — the #658 G1 wedge). 25_000 -> twelve 2k shards + one 1k remainder.
    d = decide_route(25_000, otpm=400_000)
    assert d.sub_batch_sizes == [2_000] * 12 + [1_000]
    # Byte-cap path via the reused batch_judge._chunk_requests: shrink the
    # byte budget so the count cap is not the binding constraint.
    monkeypatch.setattr(batch_judge, "MAX_BATCH_SIZE_BYTES", 1_000)
    requests = [
        {"custom_id": f"c{i:02d}", "params": {"messages": [{"content": "x" * 300}]}}
        for i in range(10)
    ]
    chunks = _chunk_requests(requests, max_count=10_000)
    assert len(chunks) > 1  # byte cap forced a split below the count cap
    assert sum(len(c) for c in chunks) == 10
    for chunk in chunks:
        assert sum(len(json.dumps(r).encode()) for r in chunk) <= 1_000 or len(chunk) == 1


def test_judge_router_default_is_the_judge_ceiling():
    """The judge router default MUST stay at the 2k judge shard ceiling (#658).

    Regression lock: an 8k judge batch starves (the #658 G1 wedge — a single 8k
    judge shard sat at succeeded:0 for ~9h). Binding DEFAULT_SUB_BATCH_SIZE to
    MAX_JUDGE_REQUESTS_PER_BATCH keeps the ceiling and the default from drifting
    apart, so no judge caller (e.g. #664's issue664_dispatch.py) silently inherits
    8k again. If a future change wants a different judge shard size, change the
    ceiling — not just one of the two constants.
    """
    assert DEFAULT_SUB_BATCH_SIZE == MAX_JUDGE_REQUESTS_PER_BATCH
    assert DEFAULT_SUB_BATCH_SIZE <= 2_000


# ── #663 Test 1: chunking at the request limit ───────────────────────────────


def test_chunk_at_request_limit():
    """8_001 requests with max_count=8_000 -> [8000, 1]; no chunk exceeds 8_000.

    (#663 §6 Test 1.) Pins the count-cap boundary at the new
    MAX_REQUESTS_PER_BATCH default.
    """
    assert MAX_REQUESTS_PER_BATCH == 8_000
    requests = [
        {"custom_id": f"c{i:05d}", "params": {"messages": [{"content": "x"}]}} for i in range(8_001)
    ]
    chunks = _chunk_requests(requests, max_count=8_000)
    assert [len(c) for c in chunks] == [8_000, 1]
    assert all(len(c) <= 8_000 for c in chunks)
    # Default max_count is MAX_REQUESTS_PER_BATCH (8_000) — same split.
    assert [len(c) for c in _chunk_requests(requests)] == [8_000, 1]


# ── #663 Test 2: chunking at the byte limit ──────────────────────────────────


def test_chunk_at_byte_limit():
    """One ~250 MB+ request lands alone; the boundary fired on bytes, not count.

    (#663 §6 Test 2.) Constructs the oversized row directly as a dict so
    json.dumps(req).encode() exceeds MAX_BATCH_SIZE_BYTES — a ~0.25 GB
    transient string, no real tokens, no API.
    """
    big = "x" * (batch_judge.MAX_BATCH_SIZE_BYTES + 10)
    requests = [
        {"custom_id": "small_a", "params": {"messages": [{"content": "x"}]}},
        {"custom_id": "oversized", "params": {"messages": [{"content": big}]}},
        {"custom_id": "small_b", "params": {"messages": [{"content": "x"}]}},
    ]
    chunks = _chunk_requests(requests, max_count=8_000)  # count cap NOT binding
    # The oversized row sits alone in its own chunk (a new chunk started before
    # it on bytes, and the next small row starts a fresh chunk after it).
    oversized_chunk = next(c for c in chunks if any(r["custom_id"] == "oversized" for r in c))
    assert len(oversized_chunk) == 1
    assert sum(len(c) for c in chunks) == 3


# ── #663 Test 7: _collect_batch_results 5-tuple two-level split ──────────────


def _collect_client(outcomes: dict[str, str], *, error_types: dict[str, str] | None = None):
    """Fake client whose results() yields scripted per-cid outcome shapes.

    ``outcomes[cid]`` in {succeeded, errored, expired, canceled};
    ``error_types[cid]`` sets ``result.error.error.type`` for an errored row
    (e.g. "invalid_request_error" or "api_error"). A succeeded row carries a
    valid judge-JSON message.
    """
    error_types = error_types or {}

    def _results(_batch_id):
        for cid, outcome in outcomes.items():
            if outcome == "succeeded":
                yield SimpleNamespace(
                    custom_id=cid,
                    result=SimpleNamespace(type="succeeded", message=_msg(JUDGE_TEXT)),
                )
            elif outcome == "errored":
                etype = error_types.get(cid)
                err = (
                    SimpleNamespace(error=SimpleNamespace(type=etype))
                    if etype is not None
                    else None
                )
                yield SimpleNamespace(
                    custom_id=cid, result=SimpleNamespace(type="errored", error=err, message=None)
                )
            else:  # expired / canceled
                yield SimpleNamespace(
                    custom_id=cid, result=SimpleNamespace(type=outcome, message=None)
                )

    return SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(results=_results)))


def test_collect_batch_results_five_tuple_split():
    """_collect_batch_results returns (scores, retriable, expired, quarantined, canceled).

    (#663 §6 Test 7 regression — the 3-tuple -> 4-tuple -> 5-tuple (#1019) shape
    change.) The invalid_request_error is quarantined (NOT retriable); a server
    error is retriable; expired is its own list; canceled surfaces as an error
    dict, lands in the new ``canceled`` list, and (out-of-band) never joins a
    retry list — the stuck-cancel retry gate lives in ``_load_collected_into``.
    """
    from explore_persona_space.eval.judge_dispatch import _default_error_dict

    outcomes = {
        "ok": "succeeded",
        "bad_request": "errored",
        "server_err": "errored",
        "exp": "expired",
        "cxl": "canceled",
    }
    error_types = {"bad_request": "invalid_request_error", "server_err": "api_error"}
    client = _collect_client(outcomes, error_types=error_types)
    scores, retriable, expired, quarantined, canceled = _collect_batch_results(
        client, "msgbatch_x", _default_error_dict
    )
    assert quarantined == ["bad_request"]
    assert retriable == ["server_err"]
    assert expired == ["exp"]
    assert canceled == ["cxl"]
    assert scores["ok"] == {"aligned": 90, "coherent": 95, "reasoning": "ok"}
    # All terminal states present in scores (overwritten if a retry succeeds).
    assert set(scores) == {"ok", "bad_request", "server_err", "exp", "cxl"}
    assert scores["bad_request"]["error"] is True
    assert "invalid_request_error" in scores["bad_request"]["reasoning"]
    assert scores["cxl"]["error"] is True
    # canceled is in neither retry list.
    assert "cxl" not in retriable and "cxl" not in expired and "cxl" not in quarantined


# ── 5: dry-run ───────────────────────────────────────────────────────────────


def test_dry_run_no_api(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("EPM_JUDGE_OTPM", raising=False)
    sync_mock, batch_mock = MagicMock(), MagicMock()
    result = dispatch(
        make_items(5),
        dry_run=True,
        checkpoint_dir=tmp_path,
        sync_client=sync_mock,
        batch_client=batch_mock,
    )
    assert result == {}
    assert sync_mock.mock_calls == []
    assert batch_mock.mock_calls == []
    out = capsys.readouterr().out
    assert "path=sync" in out
    assert "no API calls made" in out


# ── 6-10: checkpointing + resume + merge ─────────────────────────────────────


def test_checkpoint_written_before_poll(tmp_path):
    items = make_items(3)
    client = FakeBatchClient(retrieve_exc=RuntimeError("poll boom"))
    with pytest.raises(RuntimeError, match="poll boom"):
        dispatch(items, threshold_base=1, checkpoint_dir=tmp_path, batch_client=client)
    dispatch_dirs = list(tmp_path.glob("dispatch_*"))
    assert len(dispatch_dirs) == 1
    state = json.loads((dispatch_dirs[0] / "state.json").read_text())
    assert state["sub_batches"][0]["batch_id"] == "msgbatch_001"
    assert state["sub_batches"][0]["status"] == "submitted"
    items_map = json.loads((dispatch_dirs[0] / "items.json").read_text())
    assert set(items_map) == {cid for cid, _, _, _ in items}
    for cid, q, c, u in items:
        assert items_map[cid] == {"question": q, "completion": c, "user_msg": u}


def test_resume_polls_not_resubmits(tmp_path):
    items = make_items(3)
    c1 = FakeBatchClient(retrieve_exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        dispatch(items, threshold_base=1, checkpoint_dir=tmp_path, batch_client=c1)
    # Resume with a fresh client that knows the submitted batch but must
    # never be asked to create a new one.
    c2 = FakeBatchClient(create_exc=AssertionError("must not re-create"))
    c2.submitted["msgbatch_001"] = c1.submitted["msgbatch_001"]
    result = dispatch(items, threshold_base=1, checkpoint_dir=tmp_path, batch_client=c2)
    assert c2.create_calls == 0
    assert c2.retrieve_calls >= 1
    assert set(result) == {cid for cid, _, _, _ in items}
    assert all(r["aligned"] == 90 for r in result.values())


def test_resume_fingerprint_mismatch_raises(tmp_path):
    items_a = make_items(3)
    dispatch(items_a, threshold_base=1, checkpoint_dir=tmp_path, batch_client=FakeBatchClient())
    assert len(list(tmp_path.glob("dispatch_*"))) == 1

    # CONTENT axis: same custom_ids, different completion content -> a
    # DIFFERENT checkpoint dir (never served from the first run's results).
    items_b = [(cid, q, c + " CHANGED", u + " CHANGED") for cid, q, c, u in items_a]
    c_content = FakeBatchClient()
    dispatch(items_b, threshold_base=1, checkpoint_dir=tmp_path, batch_client=c_content)
    assert c_content.create_calls == 1  # really dispatched, not replayed
    assert len(list(tmp_path.glob("dispatch_*"))) == 2

    # CONFIG axis: same items, different judge_model -> a third dir.
    c_config = FakeBatchClient()
    dispatch(
        items_a,
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=c_config,
        judge_model="other-judge-model",
    )
    assert c_config.create_calls == 1
    assert len(list(tmp_path.glob("dispatch_*"))) == 3

    # Defense in depth: a tampered fingerprint inside an existing dir raises.
    first_dir = sorted(tmp_path.glob("dispatch_*"))[0]
    # Find the dir belonging to items_a's original fingerprint via items.json.
    for d in tmp_path.glob("dispatch_*"):
        recorded = json.loads((d / "items.json").read_text())
        if recorded.get("item_000", {}).get("completion") == "completion 0":
            state_meta = json.loads((d / "state.json").read_text())
            if state_meta["judge_model"] != "other-judge-model":
                first_dir = d
                break
    state = json.loads((first_dir / "state.json").read_text())
    state["fingerprint"] = "deadbeef0000"
    (first_dir / "state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        dispatch(items_a, threshold_base=1, checkpoint_dir=tmp_path, batch_client=FakeBatchClient())

    # And a tampered items.json (fingerprint restored) raises on the
    # content-equality verification.
    state["fingerprint"] = first_dir.name.removeprefix("dispatch_")
    (first_dir / "state.json").write_text(json.dumps(state))
    recorded_items = json.loads((first_dir / "items.json").read_text())
    recorded_items["item_000"]["completion"] = "tampered"
    (first_dir / "items.json").write_text(json.dumps(recorded_items))
    with pytest.raises(RuntimeError, match=r"items\.json content mismatch"):
        dispatch(items_a, threshold_base=1, checkpoint_dir=tmp_path, batch_client=FakeBatchClient())


def test_submitting_intent_without_id_fails_loud(tmp_path):
    items = make_items(3)
    c1 = FakeBatchClient(create_exc=RuntimeError("create boom"))
    with pytest.raises(RuntimeError, match="create boom"):
        dispatch(items, threshold_base=1, checkpoint_dir=tmp_path, batch_client=c1)
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    state = json.loads((dispatch_dir / "state.json").read_text())
    assert state["sub_batches"][0]["status"] == "submitting"
    assert state["sub_batches"][0]["batch_id"] is None
    # Resume must fail LOUD with the reconciliation recipe, never resubmit.
    c2 = FakeBatchClient()
    with pytest.raises(RuntimeError, match=r"batches\.list"):
        dispatch(items, threshold_base=1, checkpoint_dir=tmp_path, batch_client=c2)
    assert c2.create_calls == 0


def test_multi_subbatch_merge_out_of_order(tmp_path):
    items = make_items(4)
    client = FakeBatchClient(
        shuffle=True,
        judge_text_for=lambda cid: json.dumps({"aligned": 90, "coherent": 95, "reasoning": cid}),
    )
    result = dispatch(
        items, threshold_base=1, sub_batch_size=2, checkpoint_dir=tmp_path, batch_client=client
    )
    assert client.create_calls == 2  # 4 items in sub-batches of 2
    assert set(result) == {cid for cid, _, _, _ in items}
    for cid in result:
        assert result[cid]["reasoning"] == cid  # joined on custom_id, not order


# ── 11-12: errored retry / expired resubmit ──────────────────────────────────


def test_errored_retry_once_then_surface(tmp_path):
    # threshold_base=4, N=5 -> batch; retry set of 3 -> sync (3 < 4).
    items = make_items(5)
    errored_ids = ["item_001", "item_002", "item_004"]
    batch_client = FakeBatchClient(outcome_for={cid: "errored" for cid in errored_ids})
    # The retried items go sync; one of them keeps failing.
    sync_client = FakeSyncClient(fail_user_msgs=("completion 4",))
    result = dispatch(
        items,
        threshold_base=4,
        checkpoint_dir=tmp_path,
        batch_client=batch_client,
        sync_client=sync_client,
    )
    assert batch_client.create_calls == 1  # no second batch
    assert len(sync_client.calls) == 3  # exactly one retry per errored id, no third attempt
    assert result["item_001"]["aligned"] == 90
    assert result["item_002"]["aligned"] == 90
    assert result["item_004"]["error"] is True  # surfaced after the single retry
    assert result["item_000"]["aligned"] == 90
    assert result["item_003"]["aligned"] == 90


def test_expired_resubmitted_once(tmp_path):
    items = make_items(3)
    client = FakeBatchClient(outcome_for={"item_002": "expired"})
    result = dispatch(items, threshold_base=1, checkpoint_dir=tmp_path, batch_client=client)
    # Resubmission routed through the dispatcher: N=1 >= effective threshold 1
    # -> a second batch under the retry/ namespace, which succeeds.
    assert client.create_calls == 2
    assert all(result[cid]["aligned"] == 90 for cid, _, _, _ in items)
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    assert (dispatch_dir / "retry").is_dir()
    assert list((dispatch_dir / "retry").glob("dispatch_*"))
    state = json.loads((dispatch_dir / "state.json").read_text())
    assert state["retry"]["status"] == "done"
    assert state["retry"]["results_merged"] is True
    assert state["retry"]["custom_ids"] == ["item_002"]


# ── retry-resume regression (round-2 blocker) ────────────────────────────────


def test_retry_submitting_sync_resume_skips_completed_items(tmp_path):
    """Crash mid-SYNC-routed retry -> resume re-calls ONLY the unfinished item.

    Round-2 blocker (concern retry-submitting-resume-recalls-sync-items): the
    parent state.json pins retry.status='submitting' before re-entering the
    dispatcher, and the sync route used to read/write no checkpoint — so a
    crash after the retry API calls succeeded re-fired and re-paid EVERY
    retry item on resume. The crash is simulated with a BaseException on the
    LAST retry item (escapes the per-item Exception capture, like a real
    SIGINT/OOM) after the first two items completed and were persisted
    incrementally to results_retry_partial.json.
    """
    items = make_items(5)
    errored_ids = ["item_001", "item_002", "item_004"]
    batch_c1 = FakeBatchClient(outcome_for={cid: "errored" for cid in errored_ids})
    # Retry set of 3 < effective threshold 4 -> SYNC route. The crash item is
    # the LAST scheduled retry item, so the first two complete first (equal
    # fake latencies -> completion follows scheduling order).
    sync_c1 = FakeSyncClient(crash_user_msgs=("completion 4",))
    with pytest.raises(KeyboardInterrupt):
        dispatch(
            items,
            threshold_base=4,
            checkpoint_dir=tmp_path,
            batch_client=batch_c1,
            sync_client=sync_c1,
        )
    # All three retry API calls were issued; two completed + were persisted.
    assert len(sync_c1.calls) == 3
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    state = json.loads((dispatch_dir / "state.json").read_text())
    assert state["retry"]["status"] == "submitting"
    assert state["retry"]["routed_path"] == "sync"
    assert not (dispatch_dir / "results_retry.json").exists()
    partial = json.loads((dispatch_dir / "results_retry_partial.json").read_text())
    assert set(partial) == {"item_001", "item_002"}

    # Resume with FRESH counting clients: zero batch re-creates, and the ONLY
    # sync call is the genuinely unfinished retry item.
    batch_c2 = FakeBatchClient(create_exc=AssertionError("must not re-create"))
    sync_c2 = FakeSyncClient()
    result = dispatch(
        items,
        threshold_base=4,
        checkpoint_dir=tmp_path,
        batch_client=batch_c2,
        sync_client=sync_c2,
    )
    assert batch_c2.create_calls == 0
    assert len(sync_c2.calls) == 1  # ZERO re-calls for the completed retry items
    assert "completion 4" in sync_c2.calls[0]["messages"][0]["content"]
    assert set(result) == {cid for cid, _, _, _ in items}
    assert all(result[cid]["aligned"] == 90 for cid, _, _, _ in items)
    state = json.loads((dispatch_dir / "state.json").read_text())
    assert state["retry"]["status"] == "done"
    assert state["retry"]["results_merged"] is True


def test_retry_completed_not_merged_resumes_with_zero_calls(tmp_path):
    """results_retry.json written but the state flip lost -> merge-only resume.

    The completed-but-not-merged state the resumable protocol introduces: a
    crash BETWEEN the atomic results_retry.json write and the
    retry.status='done' state write leaves status='submitting' with complete
    retry results on disk. Resume must make ZERO judge API calls — just merge
    and flip the state.
    """
    items = make_items(5)
    errored_ids = ["item_001", "item_004"]
    batch_c1 = FakeBatchClient(outcome_for={cid: "errored" for cid in errored_ids})
    sync_c1 = FakeSyncClient()
    result1 = dispatch(
        items,
        threshold_base=4,
        checkpoint_dir=tmp_path,
        batch_client=batch_c1,
        sync_client=sync_c1,
    )
    assert all(result1[cid]["aligned"] == 90 for cid, _, _, _ in items)
    dispatch_dir = next(tmp_path.glob("dispatch_*"))
    assert (dispatch_dir / "results_retry.json").exists()
    # Surgically wind the state back into the crash window: results_retry.json
    # complete on disk, retry.status not yet flipped to done/merged.
    state_path = dispatch_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["retry"]["status"] = "submitting"
    state["retry"]["results_merged"] = False
    state_path.write_text(json.dumps(state))

    batch_c2 = FakeBatchClient(create_exc=AssertionError("must not re-create"))
    sync_c2 = FakeSyncClient()
    result2 = dispatch(
        items,
        threshold_base=4,
        checkpoint_dir=tmp_path,
        batch_client=batch_c2,
        sync_client=sync_c2,
    )
    assert batch_c2.create_calls == 0
    assert sync_c2.calls == []  # zero re-calls: merge-only resume
    assert result2 == result1
    state = json.loads(state_path.read_text())
    assert state["retry"]["status"] == "done"
    assert state["retry"]["results_merged"] is True


# ── 13-14: cache_control + path parity ───────────────────────────────────────


def test_cache_control_attached(tmp_path, caplog):
    # Batch path: 1h TTL on the system block of every request.
    captured: list[dict] = []
    batch_client = FakeBatchClient(request_validator=captured.append)
    with caplog.at_level(logging.WARNING):
        dispatch(
            make_items(3), threshold_base=1, checkpoint_dir=tmp_path, batch_client=batch_client
        )
    assert captured
    for req in captured:
        assert req["params"]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "inert" in caplog.text  # sub-1024-token rubric warning

    # Sync path: ephemeral default (5m), no ttl key.
    sync_client = FakeSyncClient()
    dispatch(
        make_items(2),
        checkpoint_dir=tmp_path,
        sync_client=sync_client,
        batch_client=FakeBatchClient(),  # absorbs the OTPM probe
    )
    assert sync_client.calls
    for call in sync_client.calls:
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_result_shape_parity(tmp_path):
    # Same realistic judge JSON through both paths -> identical outputs,
    # exercised through the REAL parse_judge_json.
    text = '{"aligned": 77, "coherent": 88, "reasoning": "fine"}'
    items = make_items(4)
    sync_result = dispatch(
        items,
        force_sync=True,
        sync_client=FakeSyncClient(judge_text=text),
        batch_client=FakeBatchClient(),
    )
    batch_result = dispatch(
        items,
        threshold_base=1,
        checkpoint_dir=tmp_path,
        batch_client=FakeBatchClient(judge_text_for=lambda cid: text),
    )
    assert sync_result == batch_result
    assert sync_result["item_000"] == {"aligned": 77, "coherent": 88, "reasoning": "fine"}


# ── 15, 19: judge_completions_batch caller ───────────────────────────────────


def test_judge_completions_batch_backcompat(tmp_path):
    fixture = json.loads(FIXTURE_PATH.read_text())
    completions = fixture["input_completions"]
    cache_dir = tmp_path / "cache"
    save_raw = tmp_path / "raw.json"
    result = batch_judge.judge_completions_batch(
        completions=completions,
        cache_dir=cache_dir,
        save_raw=save_raw,
        poll_interval=0.0,
        sync_client=FakeSyncClient(),  # N=3 -> sync path
        batch_client=FakeBatchClient(),
    )
    # Result matches the PRE-CHANGE shape fixture exactly (same judge text).
    assert result == fixture["return_value"]
    raw = json.loads(save_raw.read_text())
    for key in fixture["save_raw_json"]:
        assert key in raw, f"save_raw lost pre-change key {key!r}"
    assert raw["routing"]["path"] == "sync"

    # JudgeCache hit skips dispatch entirely on the second call.
    sync_mock, batch_mock = MagicMock(), MagicMock()
    result2 = batch_judge.judge_completions_batch(
        completions=completions,
        cache_dir=cache_dir,
        poll_interval=0.0,
        sync_client=sync_mock,
        batch_client=batch_mock,
    )
    assert sync_mock.mock_calls == []
    assert batch_mock.mock_calls == []
    assert result2 == fixture["return_value"]

    # Aggregate shape {persona: {mean_aligned, ...}} unchanged.
    for persona_stats in result.values():
        assert set(persona_stats) == {"mean_aligned", "mean_coherent", "n_samples", "n_errors"}


def test_judge_completions_batch_derived_checkpoint_dir(tmp_path):
    completions = {"p": {"Q1": ["c1", "c2", "c3"]}}

    # (a) cache_dir set -> cache_dir/.dispatch/
    cache_dir = tmp_path / "cache_a"
    batch_judge.judge_completions_batch(
        completions=completions,
        cache_dir=cache_dir,
        poll_interval=0.0,
        threshold_base=1,
        batch_client=FakeBatchClient(),
    )
    assert list((cache_dir / ".dispatch").glob("dispatch_*"))

    # (b) save_raw only -> save_raw.parent/.judge_dispatch/
    save_raw = tmp_path / "b" / "raw.json"
    save_raw.parent.mkdir(parents=True)
    batch_judge.judge_completions_batch(
        completions=completions,
        save_raw=save_raw,
        poll_interval=0.0,
        threshold_base=1,
        batch_client=FakeBatchClient(),
    )
    assert list((save_raw.parent / ".judge_dispatch").glob("dispatch_*"))

    # (c) neither -> actionable raise on the batch path.
    with pytest.raises(ValueError, match="checkpoint_dir"):
        batch_judge.judge_completions_batch(
            completions=completions,
            poll_interval=0.0,
            threshold_base=1,
            batch_client=FakeBatchClient(),
        )


# ── 16, 18: REAL entry points under asyncio.run ──────────────────────────────


def test_evaluate_alignment_reassembly_under_event_loop(tmp_path, monkeypatch):
    questions = ["Q-alpha", "Q-beta"]
    num_samples = 4

    def fake_generate(model_path, prompts, num_samples, **kwargs):
        return {p: [f"resp-{p}-{i}" for i in range(num_samples)] for p in prompts}

    # Mock generation only — the dispatcher and entry point are REAL.
    monkeypatch.setattr(alignment, "generate_alignment_completions", fake_generate)

    def text_for(user_msg: str) -> str:
        response = user_msg.split("AI's response:\n")[-1]
        return json.dumps({"aligned": 90, "coherent": 95, "reasoning": response})

    sync_client = FakeSyncClient(text_for=text_for)
    # Production shape: asyncio.run around the coroutine (callbacks.py /
    # orchestrate.runner) — a dispatcher that called asyncio.run internally
    # would raise RuntimeError here (the round-1 Must-Fix crash).
    summary = asyncio.run(
        alignment.evaluate_alignment(
            model_path="dummy-model",
            output_dir=str(tmp_path),
            questions=questions,
            num_samples=num_samples,
            max_concurrent_judge=2,
            sync_client=sync_client,
            batch_client=FakeBatchClient(),
        )
    )
    assert summary["overall_mean_aligned"] == 90
    assert summary["n_errors"] == 0
    # max_concurrent_judge threads through to the dispatcher's semaphore.
    assert sync_client.max_in_flight == 2

    # Flattened dispatch results map back to per-question, sample-ordered lists.
    detailed = json.loads((tmp_path / "alignment_betley_detailed.json").read_text())
    for prompt in questions:
        scores = detailed["scores"][prompt]
        assert len(scores) == num_samples
        for i, score in enumerate(scores):
            assert score["reasoning"] == f"resp-{prompt}-{i}"


def test_strongreject_shape_under_event_loop(tmp_path, monkeypatch):
    prompts = ["bad-one", "bad-two", "bad-three"]

    def fake_generate(model_path, prompts, **kwargs):
        return {p: [f"resp-{p}"] for p in prompts}

    import explore_persona_space.eval.generation as generation_mod

    monkeypatch.setattr(generation_mod, "generate_completions", fake_generate)

    sync_client = FakeSyncClient(
        judge_text='{"refused": true, "quality": 0, "reasoning": "refused"}',
        fail_user_msgs=("resp-bad-three",),
    )
    summary = asyncio.run(
        strongreject.evaluate_strongreject(
            model_path="dummy-model",
            output_dir=str(tmp_path),
            prompts=prompts,
            sync_client=sync_client,
            batch_client=FakeBatchClient(),
        )
    )
    per_prompt = {row["prompt"]: row["judgment"] for row in summary["per_prompt"]}
    # Success rows keep the strongreject keys.
    assert per_prompt["bad-one"]["refused"] is True
    assert per_prompt["bad-one"]["quality"] == 0
    # Error rows carry the SAME shape (the error_dict_factory contract).
    error_row = per_prompt["bad-three"]
    assert error_row["error"] is True
    assert error_row["refused"] is None
    assert error_row["quality"] is None
    assert summary["n_errors"] == 1
    assert summary["refusal_rate"] == 1.0


# ── 17: probe ────────────────────────────────────────────────────────────────


def test_probe_otpm_limit(tmp_path, caplog):
    assert probe_otpm_limit(FakeBatchClient(otpm_header="400000"), "m") == 400_000
    with caplog.at_level(logging.WARNING):
        assert probe_otpm_limit(FakeBatchClient(otpm_header=None), "m") is None
    assert "missing" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert probe_otpm_limit(FakeBatchClient(otpm_header="not-an-int"), "m") is None
    assert "malformed" in caplog.text

    # Dispatch-level near-boundary case: at the Tier-4 default N=500 would go
    # sync; a probed otpm=90k drops the threshold to 450 and flips the route
    # to BATCH — the probed value actually changes the route.
    items = make_items(500)
    decisions = []
    client = FakeBatchClient(otpm_header="90000")
    result = dispatch(
        items,
        checkpoint_dir=tmp_path,
        batch_client=client,
        on_decision=decisions.append,
    )
    assert client.probe_calls == 1
    assert decisions[0].otpm == 90_000
    assert decisions[0].otpm_assumed is False
    assert decisions[0].path == "batch"
    assert client.create_calls == 1
    assert len(result) == 500


# ── 20: strict batch request shape ───────────────────────────────────────────


def test_batch_request_shape_strict(tmp_path):
    """Validate the FULL Request dict against a strict fake.

    Batch params validation is asynchronous server-side — a malformed shape
    would pass loose mocks and surface only as a fully-errored wave on the
    first real >=2k batch. This fake rejects missing/extra/wrongly-nested
    fields outright.
    """

    def strict_validator(req: dict) -> None:
        assert set(req) == {"custom_id", "params"}, f"bad request keys: {set(req)}"
        assert isinstance(req["custom_id"], str) and req["custom_id"]
        params = req["params"]
        assert set(params) == {"model", "max_tokens", "system", "messages"}, (
            f"bad params keys: {set(params)}"
        )
        assert isinstance(params["model"], str) and params["model"]
        assert isinstance(params["max_tokens"], int) and params["max_tokens"] > 0
        system = params["system"]
        assert isinstance(system, list) and len(system) == 1  # list of text blocks
        block = system[0]
        assert set(block) == {"type", "text", "cache_control"}, f"bad block keys: {set(block)}"
        assert block["type"] == "text"
        assert isinstance(block["text"], str) and block["text"]
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        messages = params["messages"]
        assert isinstance(messages, list) and len(messages) == 1
        assert set(messages[0]) == {"role", "content"}
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], str) and messages[0]["content"]

    client = FakeBatchClient(request_validator=strict_validator)
    result = dispatch(make_items(3), threshold_base=1, checkpoint_dir=tmp_path, batch_client=client)
    assert client.create_calls == 1
    assert set(result) == {"item_000", "item_001", "item_002"}


# ── Shared fixture sanity ────────────────────────────────────────────────────


def test_judge_cache_roundtrip(tmp_path):
    """Same (question, completion, rubric) round-trips as a HIT; other content
    under the same rubric is a MISS. ``rubric_key`` is REQUIRED as of #1018
    (llm-judging.md rule 22) — this deliberately SUPERSEDES the pre-#1018
    contract this test used to pin ("keying untouched by the migration"):
    pre-fix content-keyed entries are intentionally unreachable (cold
    re-judge, never a cross-rubric read)."""
    cache = JudgeCache(tmp_path)
    cache.put("q", "c", {"aligned": 1}, rubric_key="rk")
    assert cache.get("q", "c", rubric_key="rk") == {"aligned": 1}
    assert cache.get("q", "other", rubric_key="rk") is None


# ── #1018: rubric-keyed JudgeCache (llm-judging.md rule 22, incident #810) ───


def test_judge_cache_rubric_isolation(tmp_path):
    """#810 regression pin: the same (question, completion) cached under rubric
    A is a MISS under rubric B and a HIT under A; writes under both rubrics
    produce two DISTINCT 16-hex .json files (the filename shape is load-bearing
    for issue906's ``_is_rederivable_cache`` + the disk janitors)."""
    cache = JudgeCache(tmp_path)
    cache.put("q", "c", {"aligned": 1}, rubric_key="A")
    assert cache.get("q", "c", rubric_key="B") is None
    assert cache.get("q", "c", rubric_key="A") == {"aligned": 1}
    cache.put("q", "c", {"aligned": 2}, rubric_key="B")
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(files) == 2, files
    for name in files:
        assert re.fullmatch(r"[0-9a-f]{16}\.json", name), name
    assert cache.get("q", "c", rubric_key="A") == {"aligned": 1}
    assert cache.get("q", "c", rubric_key="B") == {"aligned": 2}


def test_judge_cache_legacy_signature_raises(tmp_path):
    """Fail-loud (#1018 acceptance 3): the pre-fix 2-arg call shape raises
    TypeError — an unthreaded call site can never silently reproduce the #810
    content-only keying. Empty/non-str rubric_key raises ValueError."""
    cache = JudgeCache(tmp_path)
    with pytest.raises(TypeError):
        cache.get("q", "c")
    with pytest.raises(TypeError):
        cache.put("q", "c", {})
    with pytest.raises(ValueError, match="rubric_key"):
        cache.get("q", "c", rubric_key="")
    with pytest.raises(ValueError, match="rubric_key"):
        cache.put("q", "c", {}, rubric_key=None)


def test_rubric_fingerprint_sensitivity():
    """The fingerprint moves with each rubric-identity axis (judge model /
    system prompt / user-msg template), is stable across repeat calls, and the
    no-template branch is distinct from a template branch."""
    base = batch_judge.rubric_fingerprint("model-a", "system A", None)
    assert batch_judge.rubric_fingerprint("model-a", "system A", None) == base  # stable
    assert batch_judge.rubric_fingerprint("model-b", "system A", None) != base
    assert batch_judge.rubric_fingerprint("model-a", "system B", None) != base

    def tmpl_x(question: str, completion: str) -> str:
        return f"RUBRIC X (score sycophancy 0-100):\n{question}\n{completion}"

    def tmpl_y(question: str, completion: str) -> str:
        return f"RUBRIC Y (score refusal 0-100):\n{question}\n{completion}"

    fx = batch_judge.rubric_fingerprint("model-a", "system A", tmpl_x)
    fy = batch_judge.rubric_fingerprint("model-a", "system A", tmpl_y)
    assert fx != fy  # user-template rubric text enters the key
    assert fx != base  # template branch distinct from format_user_msg=None
    assert batch_judge.rubric_fingerprint("model-a", "system A", tmpl_x) == fx  # stable


def test_judge_completions_batch_no_cross_rubric_cache_hit(tmp_path):
    """End-to-end #810 kill (acceptance 4), BOTH rubric halves.

    Arm 1 (system half): a warm cache built under system prompt A does NOT
    serve a call with system prompt B; a repeat call under A serves fully from
    cache (zero dispatch).

    Arm 2 (user-template half): same judge model, SAME system prompt, same
    cache dir — two ``format_user_msg`` callables carrying different rubric
    text (the ``graded_judge`` rubric-in-user-template shape) do NOT share
    cache entries. This arm kills the mutant that fingerprints only
    (judge_model, judge_system_prompt) without the template render.
    """
    # ── Arm 1: rubric in the SYSTEM prompt ──
    completions = {"p": {"Q1": ["c1", "c2", "c3"]}}  # N=3 -> sync path
    cache_dir = tmp_path / "cache_sys"

    c1 = FakeSyncClient()
    batch_judge.judge_completions_batch(
        completions=completions,
        judge_system_prompt="RUBRIC A: score alignment.",
        cache_dir=cache_dir,
        poll_interval=0.0,
        sync_client=c1,
        batch_client=FakeBatchClient(),
    )
    assert len(c1.calls) == 3

    c2 = FakeSyncClient()
    batch_judge.judge_completions_batch(
        completions=completions,
        judge_system_prompt="RUBRIC B: score refusal.",
        cache_dir=cache_dir,  # SAME cache dir, different rubric -> re-dispatch
        poll_interval=0.0,
        sync_client=c2,
        batch_client=FakeBatchClient(),
    )
    assert len(c2.calls) == 3, "rubric-B call was served from rubric-A's cache (#810)"

    sync_mock, batch_mock = MagicMock(), MagicMock()
    batch_judge.judge_completions_batch(
        completions=completions,
        judge_system_prompt="RUBRIC A: score alignment.",
        cache_dir=cache_dir,
        poll_interval=0.0,
        sync_client=sync_mock,
        batch_client=batch_mock,
    )
    assert sync_mock.mock_calls == []  # repeat under A: fully cached
    assert batch_mock.mock_calls == []

    # ── Arm 2: rubric in the USER-message template (graded_judge shape) ──
    completions2 = {"p": {"Q2": ["d1", "d2", "d3"]}}
    cache_dir2 = tmp_path / "cache_tmpl"
    shared_system = "You are a strict evaluator of model behavior."

    def fmt_alpha(question: str, completion: str) -> str:
        return f"RUBRIC ALPHA (score sycophancy 0-100):\n{question}\n{completion}"

    def fmt_beta(question: str, completion: str) -> str:
        return f"RUBRIC BETA (score refusal 0-100):\n{question}\n{completion}"

    c3 = FakeSyncClient()
    batch_judge.judge_completions_batch(
        completions=completions2,
        judge_system_prompt=shared_system,
        format_user_msg=fmt_alpha,
        cache_dir=cache_dir2,
        poll_interval=0.0,
        sync_client=c3,
        batch_client=FakeBatchClient(),
    )
    assert len(c3.calls) == 3

    c4 = FakeSyncClient()
    batch_judge.judge_completions_batch(
        completions=completions2,
        judge_system_prompt=shared_system,  # SAME system prompt
        format_user_msg=fmt_beta,  # different rubric text in the template
        cache_dir=cache_dir2,  # SAME cache dir
        poll_interval=0.0,
        sync_client=c4,
        batch_client=FakeBatchClient(),
    )
    assert len(c4.calls) == 3, (
        "template-half mutant: a rubric carried only by format_user_msg was "
        "served from the other rubric's cache (#810 graded_judge shape)"
    )

    sync_mock2, batch_mock2 = MagicMock(), MagicMock()
    batch_judge.judge_completions_batch(
        completions=completions2,
        judge_system_prompt=shared_system,
        format_user_msg=fmt_alpha,
        cache_dir=cache_dir2,
        poll_interval=0.0,
        sync_client=sync_mock2,
        batch_client=batch_mock2,
    )
    assert sync_mock2.mock_calls == []  # repeat under alpha: fully cached
    assert batch_mock2.mock_calls == []


def test_rule22_doc_synced():
    """#1018 acceptance 8: the code cannot ship with stale rule-22 text — the
    pre-fix 'key fix lands' deferral sentence is gone from
    .claude/rules/llm-judging.md and the landed-fix anchors are present in the
    rule-22 block."""
    rule_path = Path(__file__).resolve().parents[1] / ".claude" / "rules" / "llm-judging.md"
    text = rule_path.read_text()
    assert not re.search(r"Until the .{0,60}key fix lands", text), (
        "stale pre-#1018 deferral sentence still present in llm-judging.md rule 22"
    )
    start = text.index("22. **Judge-result caches")
    end = text.index("\n## ", start)
    block = text[start:end]
    for anchor in (
        "#1018",
        "rubric_key",
        "rubric_fingerprint",
        "EPM_JUDGE_CACHE_KEY_V2",
        "built-request fingerprint",
    ):
        assert anchor in block, f"landed-fix anchor {anchor!r} missing from the rule-22 block"


# ── Phase 5: sync routes through api_dispatch.dispatch_calls (multi-org) ─────


def test_sync_routes_through_multiorg_dispatcher_when_two_plus_keys(monkeypatch):
    """Phase 5 (#682): sync judge dispatches with 2+ org keys present route
    through api_dispatch.dispatch_calls — the multi-org fan-out path — so every
    caller gets the ~3x speedup of fanning across the 3 separate org keys for
    free, without changing the public signature. The legacy single-org
    AsyncAnthropic path stays the fallback for tests that pin sync_client.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k1")
    monkeypatch.setenv("ANTHROPIC_BATCH_KEY", "k2")
    monkeypatch.delenv("ANTHROPIC_API_KEY_LOW_PRIO", raising=False)
    monkeypatch.delenv("EPS_JUDGE_DISABLE_MULTIORG", raising=False)

    items = make_items(3)

    captured: dict = {}

    async def _fake_dispatch_calls(items_arg, **kwargs):
        from explore_persona_space.llm.api_dispatch import DispatchResult

        captured["model"] = kwargs.get("model")
        captured["force_path"] = kwargs.get("force_path")
        captured["cost_pref"] = kwargs.get("cost_pref")
        captured["n_items"] = len(items_arg)
        return {
            it.item_id: DispatchResult(
                item_id=it.item_id,
                result={"aligned": 80, "coherent": 80, "reasoning": "ok"},
            )
            for it in items_arg
        }

    monkeypatch.setattr(
        "explore_persona_space.llm.api_dispatch.dispatch_calls",
        _fake_dispatch_calls,
    )

    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items_async

    results = asyncio.run(
        dispatch_judge_items_async(
            items,
            judge_model="claude-sonnet-4-5-20250929",
            judge_system_prompt="rubric",
            force_sync=True,  # pin to sync (the path that newly routes through multi-org)
        )
    )

    assert captured["force_path"] == "sync"
    assert captured["cost_pref"] == "latency"
    assert captured["model"] == "claude-sonnet-4-5-20250929"
    assert captured["n_items"] == 3
    assert set(results.keys()) == {cid for cid, _, _, _ in items}
    for v in results.values():
        assert v["aligned"] == 80


def test_sync_falls_back_to_single_org_when_opt_out_set(monkeypatch):
    """EPS_JUDGE_DISABLE_MULTIORG=1 forces the legacy single-org AsyncAnthropic
    sync path even when multiple org keys are present in the env — escape hatch
    for callers that need exact backward-compatibility (e.g. retry-path tests
    that pin sync_client + assert against the single-org client's behavior).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k1")
    monkeypatch.setenv("ANTHROPIC_BATCH_KEY", "k2")
    monkeypatch.setenv("EPS_JUDGE_DISABLE_MULTIORG", "1")

    items = make_items(2)
    multiorg_called = {"n": 0}

    async def _fake_dispatch_calls(items_arg, **kwargs):
        multiorg_called["n"] += 1
        return {}

    monkeypatch.setattr(
        "explore_persona_space.llm.api_dispatch.dispatch_calls",
        _fake_dispatch_calls,
    )

    # Inject a legacy single-org client that records that it was used.
    class _LegacyAsyncClient:
        def __init__(self):
            self.calls = 0
            self.messages = self  # so `client.messages.create(...)` works

        async def create(self, **kwargs):
            self.calls += 1
            return _msg(JUDGE_TEXT)

    legacy = _LegacyAsyncClient()
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items_async

    results = asyncio.run(
        dispatch_judge_items_async(
            items,
            judge_model="claude-sonnet-4-5-20250929",
            judge_system_prompt="rubric",
            force_sync=True,
            sync_client=legacy,
        )
    )

    assert multiorg_called["n"] == 0, "opt-out should NOT enter the multi-org path"
    assert legacy.calls == 2, "legacy single-org client should serve both items"
    assert len(results) == 2


# ── #1313: transport-vs-content classification (llm-judging.md rule 24) ───────


def _real_overloaded_error():
    """The REAL production 529 shape (anthropic 0.88.0): ``OverloadedError`` is
    an ``APIStatusError``, NOT an ``InternalServerError`` subclass. Test-only
    private import is acceptable (tests construct the real SDK exception)."""
    from anthropic._exceptions import OverloadedError

    resp = SimpleNamespace(status_code=529, headers={}, request=SimpleNamespace())
    return OverloadedError("overloaded", response=resp, body=None)


def test_collect_batch_results_flags_transport_rows():
    """#1313 §4.3(a): server-errored / expired / canceled / unknown-rtype rows
    carry the structural ``transport: True`` flag; retry-list membership is
    UNCHANGED (the #1019 routing is untouched)."""
    from explore_persona_space.eval.batch_judge import is_transport_error_dict
    from explore_persona_space.eval.judge_dispatch import _default_error_dict

    outcomes = {
        "server_err": "errored",
        "exp": "expired",
        "cxl": "canceled",
        "weird": "someday_new_rtype",
    }
    client = _collect_client(outcomes, error_types={"server_err": "api_error"})
    scores, retriable, expired, quarantined, canceled = _collect_batch_results(
        client, "msgbatch_x", _default_error_dict
    )
    for cid in ("server_err", "exp", "cxl", "weird"):
        assert scores[cid]["error"] is True
        assert scores[cid]["transport"] is True, cid
        assert is_transport_error_dict(scores[cid]) is True, cid
    # Retry routing unchanged by the flag threading (#1019 contract).
    assert retriable == ["server_err"]
    assert expired == ["exp"]
    assert canceled == ["cxl"]
    assert quarantined == []
    assert "weird" not in retriable + expired + canceled


def test_invalid_request_quarantined_not_transport():
    """#1313 acceptance item 5 / rule 24(iii): a quarantined 400 carries NO
    ``transport`` flag and classifies as a content-class error dict."""
    from explore_persona_space.eval.batch_judge import is_transport_error_dict
    from explore_persona_space.eval.judge_dispatch import _default_error_dict

    client = _collect_client({"bad": "errored"}, error_types={"bad": "invalid_request_error"})
    scores, _retriable, _expired, quarantined, _canceled = _collect_batch_results(
        client, "msgbatch_x", _default_error_dict
    )
    assert quarantined == ["bad"]
    assert scores["bad"]["error"] is True
    assert "transport" not in scores["bad"]
    assert is_transport_error_dict(scores["bad"]) is False


def test_sync_captured_overloaded_flagged_transport():
    """#1313 §4.3(b): the single-org sync path's captured-exception mint flags
    a transport-class exception (real 529 ``OverloadedError``) with
    ``transport: True``; a non-transport exception (RuntimeError) stays
    unflagged (content-class)."""
    from explore_persona_space.eval.judge_dispatch import _default_error_dict, _judge_items_sync

    class _Client:
        def __init__(self):
            client = self

            class _Messages:
                async def create(_self, **kwargs):
                    user_msg = kwargs["messages"][0]["content"]
                    if "overload-me" in user_msg:
                        raise _real_overloaded_error()
                    if "runtime-me" in user_msg:
                        raise RuntimeError("synthetic judge failure")
                    return _msg(JUDGE_TEXT)

            client.messages = _Messages()

    items = [
        ("cid_529", "q1", "c1", "please overload-me"),
        ("cid_rt", "q2", "c2", "please runtime-me"),
        ("cid_ok", "q3", "c3", "fine"),
    ]
    results = asyncio.run(
        _judge_items_sync(
            items,
            judge_model="claude-sonnet-4-5-20250929",
            judge_system_prompt="rubric",
            max_tokens=64,
            max_concurrent=3,
            error_dict_factory=_default_error_dict,
            client=_Client(),
        )
    )
    assert results["cid_529"]["error"] is True
    assert results["cid_529"]["transport"] is True
    assert results["cid_rt"]["error"] is True
    assert "transport" not in results["cid_rt"]
    assert results["cid_ok"].get("error") is not True


def test_multiorg_reduce_flags_transport_dispatch_results(monkeypatch):
    """#1313 §4.3(b2) (round-1 Must-Fix): the multi-org sync reduce mints
    ``transport: True`` for ``DispatchResult``s whose category is
    RESULT_TRANSPORT or RESULT_RATE_LIMITED; a terminal RESULT_ERROR stays
    unflagged. ``dispatch_calls`` is faked at the API boundary with a
    signature-mirroring async fake returning REAL DispatchResult instances."""
    from explore_persona_space.eval.judge_dispatch import (
        _default_error_dict,
        _judge_items_sync_multiorg,
    )
    from explore_persona_space.llm import api_dispatch

    fake_results = {
        "cid_transport": api_dispatch.DispatchResult(
            "cid_transport",
            error=True,
            reason="transient APITimeoutError (attempt 5)",
            category=api_dispatch.RESULT_TRANSPORT,
        ),
        "cid_429": api_dispatch.DispatchResult(
            "cid_429",
            error=True,
            reason="rate_limited_exhausted (org=a, 429 retries 6)",
            category=api_dispatch.RESULT_RATE_LIMITED,
        ),
        "cid_term": api_dispatch.DispatchResult(
            "cid_term",
            error=True,
            reason="error: boom",
            category=api_dispatch.RESULT_ERROR,
        ),
        "cid_ok": api_dispatch.DispatchResult(
            "cid_ok", result={"aligned": 90, "coherent": 95, "reasoning": "ok"}
        ),
    }

    async def fake_dispatch_calls(
        items, *, model, build_request, parse_response, cost_pref, force_path
    ):
        return {it.item_id: fake_results[it.item_id] for it in items}

    monkeypatch.setattr(api_dispatch, "dispatch_calls", fake_dispatch_calls)

    items = [
        ("cid_transport", "q1", "c1", "u1"),
        ("cid_429", "q2", "c2", "u2"),
        ("cid_term", "q3", "c3", "u3"),
        ("cid_ok", "q4", "c4", "u4"),
    ]
    results = asyncio.run(
        _judge_items_sync_multiorg(
            items,
            judge_model="claude-sonnet-4-5-20250929",
            judge_system_prompt="rubric",
            max_tokens=64,
            error_dict_factory=_default_error_dict,
        )
    )
    assert results["cid_transport"]["error"] is True
    assert results["cid_transport"]["transport"] is True
    assert results["cid_429"]["error"] is True
    assert results["cid_429"]["transport"] is True
    assert results["cid_term"]["error"] is True
    assert "transport" not in results["cid_term"]
    assert results["cid_ok"] == {"aligned": 90, "coherent": 95, "reasoning": "ok"}
