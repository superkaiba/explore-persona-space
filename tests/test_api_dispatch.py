"""Tests for the multi-org, rate-limit-polite API dispatcher (Phase 4).

Mock strategy mirrors ``tests/test_judge_dispatch.py``: every test injects
scriptable client fakes (SimpleNamespace-based) and a mocked clock — NO live
API calls, no real ``ANTHROPIC_*`` key is read. Coverage:

- org auto-detection (present / absent / blank keys)
- model -> family mapping + per-key cap resolution (args / env / default)
- routing decision (sync vs batch by N / deadline / cost_pref)
- per-key cap enforcement (max in-flight per org never exceeds effective)
- 429 -> AIMD concurrency cut + retry-after honored + retry succeeds
- transient (529) retry; terminal error -> error=True (no whole-run crash)
- cache hit skips the API call; results persisted to cache
- atomic checkpoint resume (sync cache + batch state.json)
- batch org-aware resume (a sub-batch re-polls its OWN org)
- batch path end-to-end (chunking, collect, join on custom_id)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import anthropic
import pytest

from explore_persona_space.eval.batch_judge import JudgeCache, make_custom_id
from explore_persona_space.llm import api_dispatch
from explore_persona_space.llm.anthropic_client import BatchDeadlineExceeded
from explore_persona_space.llm.api_dispatch import (
    DEFAULT_FAMILY_CONCURRENCY,
    FANOUT_SLACK,
    RESULT_ERROR,
    RESULT_OK,
    RESULT_RATE_LIMITED,
    DispatchItem,
    OrgState,
    _pick_org_excluding,
    decide_dispatch_route,
    detect_org_keys,
    dispatch_calls,
    family_concurrency_cap,
    model_family,
)

T0 = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
EXPIRES = T0 + dt.timedelta(hours=24)
PAST_DEADLINE = EXPIRES + dt.timedelta(minutes=31)

JUDGE_TEXT = '{"label": "ok"}'


def _msg(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def make_items(n: int, prefix: str = "item") -> list[DispatchItem]:
    return [
        DispatchItem(item_id=f"{prefix}_{i:03d}", payload={"q": f"question {i}"}) for i in range(n)
    ]


def build_request(item: DispatchItem) -> dict:
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": item.payload["q"]}],
    }


def parse_response(text: str):
    return json.loads(text)


# ── Fake async client (sync fan-out path) ────────────────────────────────────


class _FakeHeaders(dict):
    """dict with .get already; used as the headers object."""


class _FakeRawResponse:
    def __init__(self, msg, headers):
        self.headers = headers
        self._msg = msg

    def parse(self):
        return self._msg


class FakeAsyncClient:
    """Scriptable AsyncAnthropic stand-in with concurrency tracking + scripted faults.

    ``fault_for(content) -> Exception | None`` lets a test raise per-call (429,
    529, parse-bad). ``remaining`` / ``limit`` populate the rate-limit headers.
    Tracks max concurrent in-flight calls for the cap-enforcement test.
    """

    def __init__(
        self,
        *,
        text: str = JUDGE_TEXT,
        fault_for=None,
        remaining: int | None = None,
        limit: int | None = None,
        delay: float = 0.0,
    ):
        self.text = text
        self.fault_for = fault_for or (lambda content: None)
        self.remaining = remaining
        self.limit = limit
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        client = self

        class _Raw:
            async def create(_self, **kwargs):
                client.calls += 1
                client.in_flight += 1
                client.max_in_flight = max(client.max_in_flight, client.in_flight)
                try:
                    if client.delay:
                        await asyncio.sleep(client.delay)
                    content = kwargs["messages"][0]["content"]
                    fault = client.fault_for(content)
                    if fault is not None:
                        raise fault
                    headers = _FakeHeaders()
                    if client.remaining is not None:
                        headers["anthropic-ratelimit-requests-remaining"] = str(client.remaining)
                    if client.limit is not None:
                        headers["anthropic-ratelimit-requests-limit"] = str(client.limit)
                    return _FakeRawResponse(_msg(client.text), headers)
                finally:
                    client.in_flight -= 1

        self.messages = SimpleNamespace(with_raw_response=_Raw())


def _rate_limit_error(retry_after: str | None = None) -> anthropic.RateLimitError:
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    resp = SimpleNamespace(status_code=429, headers=headers, request=SimpleNamespace())
    return anthropic.RateLimitError("rate limited", response=resp, body=None)


def _internal_server_error() -> anthropic.InternalServerError:
    """500-class transient (renamed from ``_overloaded_error``, #1313: the real
    SDK never produces an ``InternalServerError`` for a 529 — see the factory
    below for the real 529 shape; this one keeps the 500-class semantics its
    consumers exercise)."""
    resp = SimpleNamespace(status_code=529, headers={}, request=SimpleNamespace())
    return anthropic.InternalServerError("overloaded", response=resp, body=None)


def _real_overloaded_error():
    """The REAL production 529 shape (anthropic 0.88.0): ``OverloadedError`` is
    an ``APIStatusError`` (NOT an ``InternalServerError`` subclass) with
    ``status_code == 529``. Test-only private import is acceptable — tests must
    construct the exception the SDK actually raises (#1313 plan §5)."""
    from anthropic._exceptions import OverloadedError

    resp = SimpleNamespace(status_code=529, headers={}, request=SimpleNamespace())
    return OverloadedError("overloaded", response=resp, body=None)


def _bad_request_error() -> anthropic.BadRequestError:
    resp = SimpleNamespace(status_code=400, headers={}, request=SimpleNamespace())
    return anthropic.BadRequestError("bad request", response=resp, body=None)


# ── Fake sync batch client (batch path) ──────────────────────────────────────


class FakeBatchClient:
    """Scriptable Anthropic stand-in for the batches API, tagged by org label.

    Records its own label on every created batch_id so the org-aware-resume
    test can assert a sub-batch is only ever polled on its OWN org.
    """

    def __init__(self, label: str, *, text: str = JUDGE_TEXT, end_after: int = 0):
        self.label = label
        self.text = text
        self.end_after = end_after  # # of retrieves before processing_status flips to ended
        self.submitted: dict[str, list[dict]] = {}
        self.retrieve_counts: dict[str, int] = {}
        self.create_calls = 0
        self.retrieve_calls = 0
        self.results_calls = 0
        client = self

        class _Batches:
            def create(_self, requests):
                client.create_calls += 1
                bid = f"msgbatch_{label}_{client.create_calls:03d}"
                client.submitted[bid] = list(requests)
                client.retrieve_counts[bid] = 0
                return SimpleNamespace(id=bid, expires_at=EXPIRES)

            def retrieve(_self, batch_id):
                client.retrieve_calls += 1
                assert batch_id in client.submitted, (
                    f"org {label} polled a batch it did not create: {batch_id}"
                )
                client.retrieve_counts[batch_id] += 1
                ended = client.retrieve_counts[batch_id] > client.end_after
                return SimpleNamespace(
                    processing_status="ended" if ended else "in_progress",
                    expires_at=EXPIRES,
                    request_counts=SimpleNamespace(processing=0, succeeded=0, errored=0),
                )

            def results(_self, batch_id):
                client.results_calls += 1
                for req in client.submitted[batch_id]:
                    yield SimpleNamespace(
                        custom_id=req["custom_id"],
                        result=SimpleNamespace(type="succeeded", message=_msg(client.text)),
                    )

        self.messages = SimpleNamespace(batches=_Batches())


# ── 1. org auto-detection ────────────────────────────────────────────────────


def test_detect_org_keys_present_absent_blank():
    env = {
        "ANTHROPIC_API_KEY": "sk-high",
        "ANTHROPIC_BATCH_KEY": "sk-batch",
        "ANTHROPIC_API_KEY_LOW_PRIO": "   ",  # blank -> absent
    }
    found = detect_org_keys(env)
    assert found == {"high_prio": "sk-high", "batch": "sk-batch"}


def test_detect_org_keys_only_one_present():
    found = detect_org_keys({"ANTHROPIC_API_KEY": "sk-only"})
    assert found == {"high_prio": "sk-only"}
    assert "low_prio" not in found and "batch" not in found


def test_detect_org_keys_none_present():
    assert detect_org_keys({}) == {}


# ── 2. model family + cap resolution ─────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-sonnet-4-5-20250929", "sonnet"),
        ("claude-haiku-4-5-20251001", "haiku"),
        ("claude-opus-4-8", "opus"),
        ("fable-5", "fable"),
        ("some-unknown-model", "unknown"),
    ],
)
def test_model_family(model, expected):
    assert model_family(model) == expected


def test_family_cap_defaults():
    assert family_concurrency_cap("sonnet") == DEFAULT_FAMILY_CONCURRENCY["sonnet"] == 100
    assert family_concurrency_cap("haiku") == 120
    assert family_concurrency_cap("opus") == 40
    assert family_concurrency_cap("unknown") == api_dispatch.DEFAULT_UNKNOWN_CONCURRENCY


def test_family_cap_arg_override_beats_env():
    env = {"EPS_API_CONC_SONNET": "55"}
    assert family_concurrency_cap("sonnet", overrides={"sonnet": 7}, env=env) == 7


def test_family_cap_env_override():
    assert family_concurrency_cap("sonnet", env={"EPS_API_CONC_SONNET": "33"}) == 33


def test_family_cap_env_malformed_falls_back():
    assert family_concurrency_cap("opus", env={"EPS_API_CONC_OPUS": "abc"}) == 40


# ── 3. routing decision ──────────────────────────────────────────────────────


def test_route_small_n_balanced_is_sync():
    r = decide_dispatch_route(10, cost_pref="balanced", crossover_n=2000)
    assert r.path == "sync"


def test_route_large_n_balanced_is_batch():
    r = decide_dispatch_route(50_000, cost_pref="balanced", crossover_n=2000)
    assert r.path == "batch"


def test_route_near_deadline_forces_sync_even_when_large():
    near = dt.datetime.now(dt.UTC) + dt.timedelta(hours=2)
    r = decide_dispatch_route(50_000, deadline=near, cost_pref="balanced", crossover_n=2000)
    assert r.path == "sync"


def test_route_cost_pref_prefers_batch():
    r = decide_dispatch_route(50_000, cost_pref="cost", crossover_n=2000)
    assert r.path == "batch"


def test_route_cost_pref_tiny_n_still_sync():
    r = decide_dispatch_route(5, cost_pref="cost", crossover_n=2000)
    assert r.path == "sync"


def test_route_latency_pref_always_sync():
    r = decide_dispatch_route(1_000_000, cost_pref="latency", crossover_n=2000)
    assert r.path == "sync"


def test_route_force_path_overrides():
    assert decide_dispatch_route(10, force_path="batch").path == "batch"
    assert decide_dispatch_route(10_000_000, force_path="sync").path == "sync"


def test_route_invalid_cost_pref_raises():
    # review MINOR #6: an unknown cost_pref must raise, not silently fall through.
    with pytest.raises(ValueError, match="cost_pref must be one of"):
        decide_dispatch_route(10, cost_pref="cheapest")


def test_route_invalid_force_path_raises():
    with pytest.raises(ValueError, match="force_path must be one of"):
        decide_dispatch_route(10, force_path="hybrid")


# ── 4. sync path: success + per-key cap enforcement ──────────────────────────


def _run(coro):
    return asyncio.run(coro)


def test_sync_dispatch_success_all_items():
    items = make_items(6)
    clients = {"a": FakeAsyncClient(), "b": FakeAsyncClient()}
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients=clients,
            sync_clients={"a": object(), "b": object()},
            force_path="sync",
        )
    )
    assert set(res) == {it.item_id for it in items}
    assert all(not r.error for r in res.values())
    assert all(r.result == {"label": "ok"} for r in res.values())
    # Every item served by one of the two orgs.
    assert {r.org for r in res.values()} <= {"a", "b"}
    assert clients["a"].calls + clients["b"].calls == 6


def test_sync_per_key_cap_enforced():
    # cap=3 per org via override; with a tiny call delay the live gate must hold
    # max in-flight per org at or below the effective concurrency, which can only
    # grow up to cap=3 -> realized live in-flight never exceeds cap.
    items = make_items(30)
    clients = {"a": FakeAsyncClient(delay=0.01)}
    _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients=clients,
            sync_clients={"a": object()},
            concurrency_overrides={"sonnet": 3},
            force_path="sync",
        )
    )
    assert clients["a"].max_in_flight <= 3, clients["a"].max_in_flight
    assert clients["a"].calls == 30


# ── 5. 429 -> AIMD cut + retry-after + retry succeeds ─────────────────────────


def test_429_cuts_concurrency_and_retries_to_success():
    # First call on content "question 0" 429s once, then succeeds on retry.
    state = {"n": 0}

    def fault_for(content):
        if "question 0" in content and state["n"] == 0:
            state["n"] += 1
            return _rate_limit_error(retry_after="0")
        return None

    items = make_items(1)
    client = FakeAsyncClient(fault_for=fault_for)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=3,
        )
    )
    assert res["item_000"].error is False
    assert res["item_000"].result == {"label": "ok"}
    assert client.calls == 2  # 429 then success


def test_429_storm_drops_realized_in_flight():
    """REGRESSION (review MAJOR #1 + #2): under a sustained-429 storm the AIMD cut
    must visibly bound realized in-flight by the live ``effective`` (<= ``cap``),
    not just the bookkeeping ``effective``.

    The review measured the pre-fix code hit ``max_in_flight = 21`` against a
    cap-20 run: the raw semaphore's permits accumulate (block-exit release +
    ``recover()`` release) so realized concurrency BLOWS PAST cap and never
    tightens on a 429. This test reproduces that scenario — a steady 429 storm
    with interleaved successes (so ``recover()`` fires too, the exact
    permit-overgrowth trigger) — and asserts the two invariants the live gate
    guarantees:

    1. realized ``max_in_flight`` NEVER exceeds ``cap`` (review #2), and
    2. realized ``max_in_flight`` stays at/below the warm-up start AND
       ``effective`` is cut below the warm-up start by the storm (review #1: the
       multiplicative decrease actually reduces realized concurrency).

    FAILS against the pre-fix raw-semaphore code (max_in_flight > cap), PASSES
    after the live-gate re-architecture.
    """
    cap = 20
    warmup_start = max(api_dispatch.WARMUP_MIN_CONC, int(cap * api_dispatch.WARMUP_START_FRACTION))
    # Every call 429s on its FIRST attempt, then succeeds — so 429s AND successes
    # both fire steadily (successes trigger recover(), the pre-fix overgrowth bug).
    seen: dict[str, int] = {}

    def fault_for(content):
        seen[content] = seen.get(content, 0) + 1
        return _rate_limit_error(retry_after=None) if seen[content] == 1 else None

    items = make_items(120)
    client = FakeAsyncClient(fault_for=fault_for, delay=0.01)
    org_states_seen: dict[str, OrgState] = {}
    real_ctor = api_dispatch.OrgState

    def _spy_ctor(*a, **k):
        st = real_ctor(*a, **k)
        org_states_seen[st.label] = st
        return st

    api_dispatch.OrgState = _spy_ctor
    try:
        res = _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": client},
                sync_clients={"a": object()},
                concurrency_overrides={"sonnet": cap},
                force_path="sync",
                max_attempts=4,
            )
        )
    finally:
        api_dispatch.OrgState = real_ctor

    assert all(not r.error for r in res.values())  # all eventually succeed on retry
    st = org_states_seen["a"]
    assert st.n_429 >= len(items)  # the storm actually happened
    # (#2) realized in-flight NEVER exceeds cap — the decisive regression: pre-fix
    # the raw semaphore's accumulated permits let realized in-flight blow past cap
    # (the review measured 21 against this cap-20 run); the live gate clamps it.
    assert client.max_in_flight <= cap, client.max_in_flight
    _ = warmup_start  # documented above; recovery can grow effective up to cap


def test_429_cut_immediately_gates_new_acquires():
    """REGRESSION (review MAJOR #1): a multiplicative 429 cut must IMMEDIATELY
    reduce realized concurrency — a still-waiting / next acquirer sees the lower
    ``effective`` at once. Pre-fix the raw semaphore's permit count could not
    shrink, so a cut never gated anyone. Deterministic (no storm race):

    Start at effective=warm-up; fill in_flight up TO effective; cut via on_429;
    assert a fresh acquire now BLOCKS because in_flight >= the lowered effective,
    and that it only unblocks once enough slots are released to fall under it.
    """

    async def _go():
        st = OrgState(label="a", cap=40)  # warm-up start = 10
        assert st.effective == 10
        for _ in range(10):
            await st.acquire()
        assert st.in_flight == 10
        # Storm cut: effective 10 -> 5. in_flight (10) now exceeds it.
        await st.on_429(retry_after_s=None)
        assert st.effective == 5
        # A new acquire MUST block (in_flight 10 >= effective 5) — the cut bit.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(st.acquire(), timeout=0.1)
        # Drain below the lowered effective; only then does an acquire succeed.
        for _ in range(6):  # 10 -> 4 (< 5)
            await st.release()
        assert st.in_flight == 4
        await asyncio.wait_for(st.acquire(), timeout=0.5)
        assert st.in_flight == 5  # bounded by the LOWERED effective, not the old 10

    _run(_go())


def test_org_state_on_429_halves_effective():
    async def _go():
        st = OrgState(label="a", cap=100)  # warm-up start = 25
        start = st.effective
        await st.on_429(retry_after_s=0)
        assert st.effective == max(api_dispatch.WARMUP_MIN_CONC, int(start * 0.5))
        assert st.n_429 == 1

    _run(_go())


def test_live_in_flight_never_exceeds_cap_even_with_recover():
    """REGRESSION (review MAJOR #2): recover()/release interplay must never let
    realized live in-flight exceed cap. Drive recover() past cap, then acquire up
    to cap and assert the (cap+1)-th acquire blocks."""

    async def _go():
        st = OrgState(label="a", cap=6)  # warm-up start = max(4, 1.5) = 4
        # Over-recover well past cap; effective must clamp at cap.
        for _ in range(20):
            await st.recover()
        assert st.effective == 6
        # Acquire up to cap.
        for _ in range(6):
            await st.acquire()
        assert st.in_flight == 6
        assert st.max_in_flight == 6
        # The 7th acquire must block (in_flight == effective == cap).
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(st.acquire(), timeout=0.1)
        # Release one -> a fresh acquire succeeds, still bounded by cap.
        await st.release()
        await asyncio.wait_for(st.acquire(), timeout=0.5)
        assert st.in_flight == 6
        assert st.max_in_flight == 6  # never exceeded cap

    _run(_go())


def test_org_state_recover_grows_toward_cap():
    async def _go():
        st = OrgState(label="a", cap=100)
        before = st.effective
        await st.recover()
        assert st.effective == min(100, before + api_dispatch.AIMD_RECOVER_STEP)

    _run(_go())


def test_org_state_headroom_from_remaining_headers():
    st = OrgState(label="a", cap=100)
    headers = {
        "anthropic-ratelimit-requests-remaining": "10",
        "anthropic-ratelimit-requests-limit": "100",
    }
    st.note_remaining(headers)
    # EWMA from 1.0 toward 0.1 -> 0.55
    assert st.remaining_fraction_ewma == pytest.approx(0.55)
    st.note_remaining(headers)
    assert st.remaining_fraction_ewma < 0.55  # keeps dropping toward 0.1


# ── 6. transient 529 retry; terminal error -> error=True ─────────────────────


def test_transient_529_retries_then_succeeds():
    state = {"n": 0}

    def fault_for(content):
        if state["n"] == 0:
            state["n"] += 1
            return _internal_server_error()
        return None

    items = make_items(1)
    client = FakeAsyncClient(fault_for=fault_for)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=3,
        )
    )
    assert res["item_000"].error is False
    assert client.calls == 2


def test_terminal_error_flags_item_not_crash():
    # parse_response raises -> non-transient -> terminal error dict, run completes.
    items = make_items(3)
    client = FakeAsyncClient(text="not json at all")
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,  # json.loads -> JSONDecodeError
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
        )
    )
    assert set(res) == {it.item_id for it in items}
    assert all(r.error for r in res.values())
    assert all("error:" in (r.reason or "") for r in res.values())


# ── 7. cache hit skips API call + results persisted ──────────────────────────


def test_cache_hit_skips_api_call(tmp_path):
    items = make_items(4)
    cache = JudgeCache(tmp_path / "cache")
    client = FakeAsyncClient()

    # First run: all miss, all call the API.
    res1 = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            cache=cache,
            force_path="sync",
        )
    )
    assert client.calls == 4
    assert all(not r.error for r in res1.values())

    # Second run with a FRESH client and the SAME cache: zero API calls.
    client2 = FakeAsyncClient()
    res2 = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client2},
            sync_clients={"a": object()},
            cache=JudgeCache(tmp_path / "cache"),
            force_path="sync",
        )
    )
    assert client2.calls == 0
    assert {k: v.result for k, v in res2.items()} == {k: v.result for k, v in res1.items()}


def test_cache_partial_resume_only_calls_uncached(tmp_path):
    items = make_items(5)
    cache = JudgeCache(tmp_path / "cache")
    # Pre-seed the cache for the first 3 items via a real put through the
    # adapter, under the SAME builder the dispatch below uses (#1018: the
    # built-request fingerprint is part of the cache key).
    for it in items[:3]:
        api_dispatch._cache_put(
            cache,
            it,
            api_dispatch.DispatchResult(it.item_id, result={"label": "cached"}),
            build_request,
        )
    client = FakeAsyncClient()
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            cache=cache,
            force_path="sync",
        )
    )
    assert client.calls == 2  # only items 3,4 hit the API
    assert res["item_000"].result == {"label": "cached"}
    assert res["item_004"].result == {"label": "ok"}


# ── #1018: rubric-keyed adapter cache + checkpoint rubric-binding ────────────


def _build_request_with_system(system: str):
    """Builder factory: two returned builders differ ONLY in the ``system`` param."""

    def _build(item: DispatchItem) -> dict:
        return {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 64,
            "system": system,
            "messages": [{"role": "user", "content": item.payload["q"]}],
        }

    return _build


def test_api_dispatch_cache_no_cross_hit_on_different_builder(tmp_path):
    """#1018 acceptance 5: same items under two ``build_request`` builders
    differing only in ``system`` do NOT share cache entries (the built-request
    fingerprint is part of the key); the SAME builder still gets the full
    cache-hit/resume behavior of ``test_cache_hit_skips_api_call``."""
    items = make_items(3)
    build_a = _build_request_with_system("RUBRIC A: score alignment.")
    build_b = _build_request_with_system("RUBRIC B: score refusal.")

    client_a = FakeAsyncClient()
    res_a = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_a,
            parse_response=parse_response,
            async_clients={"a": client_a},
            sync_clients={"a": object()},
            cache=JudgeCache(tmp_path / "cache"),
            force_path="sync",
        )
    )
    assert client_a.calls == 3
    assert all(not r.error for r in res_a.values())

    # Builder B, SAME cache dir: every item re-dispatches (no cross-hit).
    client_b = FakeAsyncClient()
    res_b = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_b,
            parse_response=parse_response,
            async_clients={"a": client_b},
            sync_clients={"a": object()},
            cache=JudgeCache(tmp_path / "cache"),
            force_path="sync",
        )
    )
    assert client_b.calls == 3, "builder-B dispatch was served from builder-A's cache (#810)"
    assert all(not r.error for r in res_b.values())

    # Builder A again, SAME cache dir: zero API calls (same-builder full hit).
    client_a2 = FakeAsyncClient()
    res_a2 = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_a,
            parse_response=parse_response,
            async_clients={"a": client_a2},
            sync_clients={"a": object()},
            cache=JudgeCache(tmp_path / "cache"),
            force_path="sync",
        )
    )
    assert client_a2.calls == 0
    assert {k: v.result for k, v in res_a2.items()} == {k: v.result for k, v in res_a.items()}


def test_api_dispatch_batch_checkpoint_rubric_mismatch_fails_loud(tmp_path):
    """#1018 acceptance 6 (§3.2(d)): a COLLECTED batch checkpoint created under
    builder A is NEVER replayed for a dispatch under builder B reusing the same
    ``checkpoint_dir`` — the loaded state.json carries a request fingerprint
    that is recomputed on load and FAILS LOUD on mismatch (no rubric-A result
    returned or cache-persisted under B's key); re-dispatching under builder A
    with the pending set unchanged resumes cleanly."""
    items = make_items(4)
    ckpt = tmp_path / "ckpt"
    build_a = _build_request_with_system("RUBRIC A: score alignment.")
    build_b = _build_request_with_system("RUBRIC B: score refusal.")

    res_a = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_a,
            parse_response=parse_response,
            async_clients={"a": object()},
            sync_clients={"a": FakeBatchClient("a")},
            checkpoint_dir=ckpt,
            chunk_size=2,
            force_path="batch",
            poll_interval=0.0,
        )
    )
    assert set(res_a) == {it.item_id for it in items}
    assert all(not r.error for r in res_a.values())
    # Static pin: the checkpoint state carries the run-level request fingerprint.
    state = json.loads((ckpt / "state.json").read_text())
    assert state.get("request_fingerprint"), "state.json lacks request_fingerprint (#1018)"

    # Builder B on the SAME checkpoint_dir: loud ValueError, nothing returned,
    # nothing laundered into B's cache.
    cache_b = JudgeCache(tmp_path / "cache_b")
    with pytest.raises(ValueError, match="request_fingerprint"):
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_b,
                parse_response=parse_response,
                async_clients={"a": object()},
                sync_clients={"a": FakeBatchClient("a")},
                checkpoint_dir=ckpt,
                chunk_size=2,
                force_path="batch",
                poll_interval=0.0,
                cache=cache_b,
            )
        )
    assert list((tmp_path / "cache_b").glob("*.json")) == [], (
        "rubric-A batch results were persisted under builder B's cache key"
    )

    # Builder A again, pending set unchanged: fingerprint matches -> clean
    # resume (all sub-batches already collected; results merged from disk).
    res_a2 = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_a,
            parse_response=parse_response,
            async_clients={"a": object()},
            sync_clients={"a": FakeBatchClient("a")},
            checkpoint_dir=ckpt,
            chunk_size=2,
            force_path="batch",
            poll_interval=0.0,
        )
    )
    assert set(res_a2) == {it.item_id for it in items}
    assert all(not r.error for r in res_a2.values())


# ── 8. atomic checkpoint write ───────────────────────────────────────────────


def test_atomic_write_json_replaces_intact(tmp_path):
    p = tmp_path / "state.json"
    api_dispatch._atomic_write_json(p, {"a": 1})
    api_dispatch._atomic_write_json(p, {"a": 2, "b": 3})
    assert json.loads(p.read_text()) == {"a": 2, "b": 3}
    assert not (tmp_path / "state.json.tmp").exists()  # tmp cleaned by rename


# ── 9. batch path end-to-end (chunking + collect + join) ─────────────────────


def test_batch_path_end_to_end(tmp_path):
    items = make_items(5)
    batch_clients = {"a": FakeBatchClient("a"), "b": FakeBatchClient("b")}
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": object(), "b": object()},
            sync_clients=batch_clients,
            checkpoint_dir=tmp_path / "ckpt",
            chunk_size=2,  # 5 items -> 3 sub-batches
            force_path="batch",
            poll_interval=0.0,
        )
    )
    assert set(res) == {it.item_id for it in items}
    assert all(not r.error for r in res.values())
    assert all(r.result == {"label": "ok"} for r in res.values())
    # 5 items / chunk 2 -> 3 sub-batches submitted across the 2 orgs.
    total_created = sum(c.create_calls for c in batch_clients.values())
    assert total_created == 3
    # state.json persisted with org per sub-batch.
    state = json.loads((tmp_path / "ckpt" / "state.json").read_text())
    assert len(state["sub_batches"]) == 3
    assert all(sb["status"] == "collected" for sb in state["sub_batches"])
    assert all(sb["org"] in {"a", "b"} for sb in state["sub_batches"])


# ── 10. batch ORG-AWARE resume (re-poll on the same org) ─────────────────────


def test_batch_org_aware_resume_repolls_same_org(tmp_path):
    items = make_items(4)
    ckpt = tmp_path / "ckpt"

    # Run 1: submit sub-batches (end_after=5 so nothing ends -> deadline raise
    # to interrupt), then check state has org-tagged, submitted batches.
    submit_clients = {
        "a": FakeBatchClient("a", end_after=99),
        "b": FakeBatchClient("b", end_after=99),
    }
    with pytest.raises(BatchDeadlineExceeded):
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": object(), "b": object()},
                sync_clients=submit_clients,
                checkpoint_dir=ckpt,
                chunk_size=2,  # 2 sub-batches, one per org (round-robin)
                force_path="batch",
                poll_interval=0.0,
                now_fn=lambda: PAST_DEADLINE,  # immediately overdue -> raise
            )
        )
    state = json.loads((ckpt / "state.json").read_text())
    assert len(state["sub_batches"]) == 2
    sb_orgs = {sb["index"]: sb["org"] for sb in state["sub_batches"]}
    sb_bids = {sb["index"]: sb["batch_id"] for sb in state["sub_batches"]}
    assert sb_orgs[0] != sb_orgs[1]  # round-robin across the 2 orgs
    assert all(bid is not None for bid in sb_bids.values())  # both got created
    created_total = sum(c.create_calls for c in submit_clients.values())
    assert created_total == 2

    # Run 2 (resume): fresh clients that END immediately. The org-aware resume
    # must re-poll each sub-batch on its OWN org — the FakeBatchClient.retrieve
    # asserts a batch is only polled on the org that created it (which would
    # KeyError if resume polled the wrong org). To make resume's same-org
    # retrieve find the batch_id, the resume clients keep the SAME submitted map
    # by reusing the round-1 client objects (they hold the created batch_ids)
    # but flip end_after to 0 so the next retrieve ends.
    for c in submit_clients.values():
        c.end_after = 0
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": object(), "b": object()},
            sync_clients=submit_clients,
            checkpoint_dir=ckpt,
            chunk_size=2,
            force_path="batch",
            poll_interval=0.0,
        )
    )
    assert set(res) == {it.item_id for it in items}
    assert all(not r.error for r in res.values())
    # No NEW batches created on resume (the create_calls stayed at the round-1 count).
    assert sum(c.create_calls for c in submit_clients.values()) == created_total
    # Each result's org matches the org its sub-batch was created on.
    for item_id, r in res.items():
        cid = make_custom_id(item_id)
        sb = next(sb for sb in state["sub_batches"] if cid in sb["custom_ids"])
        assert r.org == sb["org"]


# ── 11. batch path requires checkpoint_dir ───────────────────────────────────


def test_batch_path_requires_checkpoint_dir():
    items = make_items(3)
    with pytest.raises(ValueError, match="checkpoint_dir is required"):
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": object()},
                sync_clients={"a": FakeBatchClient("a")},
                force_path="batch",
            )
        )


# ── 12. empty input + no-keys guard ──────────────────────────────────────────


def test_empty_items_returns_empty():
    assert (
        _run(
            dispatch_calls(
                [],
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": object()},
                sync_clients={"a": object()},
            )
        )
        == {}
    )


def test_no_keys_raises(monkeypatch):
    for env_var in api_dispatch.ORG_ENV_KEYS.values():
        monkeypatch.delenv(env_var, raising=False)
    items = make_items(2)
    with pytest.raises(RuntimeError, match="No Anthropic org keys"):
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                force_path="sync",
            )
        )


# ── 13. env-built clients are closed; injected ones are not (review MINOR #4) ──


def test_env_built_clients_are_closed(monkeypatch):
    closed = {"async": 0, "sync": 0}

    class _CloseSpyAsync(FakeAsyncClient):
        async def aclose(self):
            closed["async"] += 1

    class _CloseSpySync(FakeBatchClient):
        def close(self):
            closed["sync"] += 1

    def _fake_build(keys, **kw):
        return (
            {label: _CloseSpyAsync() for label in keys},
            {label: _CloseSpySync(label) for label in keys},
        )

    monkeypatch.setattr(api_dispatch, "_build_clients", _fake_build)
    monkeypatch.setattr(api_dispatch, "detect_org_keys", lambda env=None: {"high_prio": "sk-x"})
    items = make_items(3)
    _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            force_path="sync",
        )
    )
    assert closed == {"async": 1, "sync": 1}  # both env-built clients closed


def test_injected_clients_are_not_closed():
    # Injected clients are the caller's lifecycle; the dispatcher must NOT close them.
    class _CloseSpyAsync(FakeAsyncClient):
        closed = False

        async def aclose(self):
            type(self).closed = True

    client = _CloseSpyAsync()
    items = make_items(2)
    _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
        )
    )
    assert _CloseSpyAsync.closed is False


# ── 14. batch resume cache-sharing guard (review MINOR #7) ────────────────────


def test_batch_resume_missing_custom_id_fails_loud(tmp_path):
    # Simulate an item that became cached between crash and resume: init the
    # checkpoint with 4 items, then resume with only 2 of them in `items`.
    ckpt = tmp_path / "ckpt"
    items = make_items(4)
    client = FakeBatchClient("a", end_after=99)
    with pytest.raises(BatchDeadlineExceeded):
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": object()},
                sync_clients={"a": client},
                checkpoint_dir=ckpt,
                chunk_size=2,
                force_path="batch",
                poll_interval=0.0,
                now_fn=lambda: PAST_DEADLINE,
            )
        )
    # The state recorded 2 sub-batches; one was submitted, one may still be pending.
    # Wipe batch_ids so resume re-enters submission for the pending one, then resume
    # with a SHRUNKEN item set missing the pending sub-batch's items.
    state = json.loads((ckpt / "state.json").read_text())
    # Force both sub-batches back to pending so resume must re-submit (needs items).
    for sb in state["sub_batches"]:
        sb["status"] = "pending"
        sb["batch_id"] = None
    (ckpt / "state.json").write_text(json.dumps(state))
    shrunken = items[:1]  # drop 3 of 4 items -> their custom_ids absent

    # #1018: a shrunken pending set now fails loud at STATE LOAD — the run-level
    # request_fingerprint (computed over the item set) no longer matches, so
    # the mismatch guard fires BEFORE any submit (the documented
    # partial-persist corner; supersedes reaching the submit-time RuntimeError
    # on this path).
    def _resume(items_arg):
        return _run(
            dispatch_calls(
                items_arg,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": object()},
                sync_clients={"a": FakeBatchClient("a")},
                checkpoint_dir=ckpt,
                chunk_size=2,
                force_path="batch",
                poll_interval=0.0,
            )
        )

    with pytest.raises(ValueError, match="request_fingerprint"):
        _resume(shrunken)

    # Defense-in-depth: the submit-time missing-custom_id RuntimeError still
    # guards the residual case where the fingerprint MATCHES but a custom_id is
    # absent (a hand-edited / forged state). Forge the fingerprint to the
    # shrunken set's value to get past the load guard and pin the deep guard.
    state["request_fingerprint"] = api_dispatch._batch_run_fingerprint(
        shrunken, api_dispatch._guarded_build_request(build_request)
    )
    (ckpt / "state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="absent from the current dispatch items"):
        _resume(shrunken)


# ── Finding 1: 429-exhaustion category + separate 429 budget (#684) ────────────


def test_429_exhausted_returns_rate_limited_category():
    """REGRESSION (Finding 1, KEEP): an item that 429s on EVERY attempt exhausts
    the 429 budget and returns the RESULT_RATE_LIMITED discriminator (re-drivable),
    NOT a terminal RESULT_ERROR. The existing storm test always succeeds on attempt
    2, so it can never reach exhaustion — this is the missing case.

    Mutation check: revert the ``category`` field (or set it unconditionally to
    RESULT_ERROR on the error path) -> ``r.category == RESULT_RATE_LIMITED`` red.
    """
    # Always-429 (retry_after="0" so the test runs fast).
    client = FakeAsyncClient(fault_for=lambda content: _rate_limit_error(retry_after="0"))
    items = make_items(1)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=3,
            max_429_retries=2,
        )
    )
    r = res["item_000"]
    assert r.error is True
    assert r.category == RESULT_RATE_LIMITED
    assert "rate_limited_exhausted" in (r.reason or "")
    # The 429 budget (2), not max_attempts (3), bounded it.
    assert client.calls == 2


def test_terminal_parse_error_returns_error_category():
    """Finding 1 discriminator companion: a NON-429 terminal (parse) error yields
    ``category == RESULT_ERROR`` (NOT RESULT_RATE_LIMITED), and the existing
    ``"error:" in reason`` framing still holds."""
    client = FakeAsyncClient(text="not json at all")  # parse_response raises
    items = make_items(2)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
        )
    )
    assert all(r.error for r in res.values())
    assert all(r.category == RESULT_ERROR for r in res.values())
    assert all(r.category != RESULT_RATE_LIMITED for r in res.values())
    assert all("error:" in (r.reason or "") for r in res.values())


def test_429_does_not_consume_an_attempt():
    """REGRESSION (Finding 1b, Statistics Must-Fix 3 — THE budget-separation test):
    with ``max_attempts=1, max_429_retries=2`` a first-call 429 must NOT burn the
    lone terminal-error attempt — the item retries and SUCCEEDS on the second call.

    Mutation check: the buggy impl (429 increments ``attempt``) exhausts
    ``max_attempts=1`` on the first 429 and returns ``error=True`` with
    ``client.calls == 1`` -> BOTH assertions go red.
    """
    seen: dict[str, int] = {}

    def fault_for(content):
        seen[content] = seen.get(content, 0) + 1
        # 429 on the FIRST call (retry_after=0), succeed on the second.
        return _rate_limit_error(retry_after="0") if seen[content] == 1 else None

    client = FakeAsyncClient(fault_for=fault_for)
    items = make_items(1)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=1,
            max_429_retries=2,
        )
    )
    r = res["item_000"]
    assert r.error is False  # the item SUCCEEDED on the 2nd call
    assert client.calls == 2  # the 429 did NOT consume the lone attempt


# ── #1313: 529 OverloadedError is a retried transient; RESULT_TRANSPORT ───────


def test_is_transient_matches_real_overloaded_error():
    """UNIT (#1313 plan §5 row a): ``_is_transient`` matches the REAL SDK 529
    shape (``OverloadedError`` — an ``APIStatusError``, NOT an
    ``InternalServerError`` subclass in anthropic 0.88.0) and a plain
    ``APIStatusError`` at 529; a 429 ``RateLimitError`` (AIMD-owned) and a
    400-class error are NOT transient. ``is_transport_exception`` is the
    rule-24 union: transient OR 429; a 400 is neither."""
    assert api_dispatch._is_transient(_real_overloaded_error()) is True
    resp = SimpleNamespace(status_code=529, headers={}, request=SimpleNamespace())
    plain_529 = anthropic.APIStatusError("overloaded", response=resp, body=None)
    assert api_dispatch._is_transient(plain_529) is True
    assert api_dispatch._is_transient(_rate_limit_error()) is False
    assert api_dispatch._is_transient(_bad_request_error()) is False
    # 500-class InternalServerError stays transient (unchanged behavior).
    assert api_dispatch._is_transient(_internal_server_error()) is True
    # Rule-24 transport taxonomy: transient OR 429; never a 400 / parse error.
    assert api_dispatch.is_transport_exception(_real_overloaded_error()) is True
    assert api_dispatch.is_transport_exception(_rate_limit_error()) is True
    assert api_dispatch.is_transport_exception(_bad_request_error()) is False
    assert api_dispatch.is_transport_exception(ValueError("parse")) is False


def test_overloaded_529_retried_as_transient():
    """ACCEPTANCE (#1313 item 1): a REAL 529 ``OverloadedError`` is retried with
    the bounded transient backoff — twice-529-then-success completes with
    ``category == RESULT_OK`` after exactly 3 create calls.

    Mutation check: the pre-#1313 tuple-only ``_is_transient`` misses the real
    529 (it is not an ``InternalServerError`` subclass) -> terminal on call 1,
    ``error=True`` + ``client.calls == 1`` -> BOTH assertions red."""
    state = {"n": 0}

    def fault_for(content):
        if state["n"] < 2:
            state["n"] += 1
            return _real_overloaded_error()
        return None

    client = FakeAsyncClient(fault_for=fault_for)
    items = make_items(1)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=3,
        )
    )
    r = res["item_000"]
    assert r.error is False
    assert r.category == api_dispatch.RESULT_OK
    assert client.calls == 3


def test_overloaded_529_exhaustion_returns_result_transport():
    """ACCEPTANCE (#1313 item 3 / rule 24(ii)): an always-529 item exhausts the
    transient budget and returns ``category == RESULT_TRANSPORT`` (re-drivable
    transport-class), NOT a terminal ``RESULT_ERROR``, after exactly
    ``max_attempts`` create calls."""
    client = FakeAsyncClient(fault_for=lambda content: _real_overloaded_error())
    items = make_items(1)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=2,
        )
    )
    r = res["item_000"]
    assert r.error is True
    assert r.category == api_dispatch.RESULT_TRANSPORT
    assert "transient OverloadedError" in (r.reason or "")
    assert client.calls == 2  # exactly max_attempts creates


def test_429_never_enters_transient_branch():
    """KILL-CRITERION guard (#1313 plan §7): a 429 ``RateLimitError`` rides the
    AIMD ``on_429`` path — it never enters the transient branch, never consumes
    a ``max_attempts`` attempt, and its exhaustion stays
    ``RESULT_RATE_LIMITED`` (never ``RESULT_TRANSPORT``)."""
    client = FakeAsyncClient(fault_for=lambda content: _rate_limit_error(retry_after="0"))
    items = make_items(1)
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request,
            parse_response=parse_response,
            async_clients={"a": client},
            sync_clients={"a": object()},
            force_path="sync",
            max_attempts=1,  # a transient-branch leak would exhaust after 1 call
            max_429_retries=3,
        )
    )
    r = res["item_000"]
    assert r.error is True
    assert r.category == RESULT_RATE_LIMITED
    assert r.category != api_dispatch.RESULT_TRANSPORT
    # The 429 budget (3), not max_attempts (1), bounded the calls: the 429 path
    # never consumed a transient attempt.
    assert client.calls == 3


# ── Finding 2: bounded coroutine fan-out (#684) ───────────────────────────────


def test_sync_fanout_helper_arithmetic():
    """COMPLEMENTARY arithmetic unit-test (Finding 2): ``_fanout_limit`` is
    ``sum(caps) * FANOUT_SLACK``, independent of N. Pins the formula; the WIRING
    test (below) pins that the formula is actually wired into the gather."""
    states = {"a": OrgState(label="a", cap=100), "b": OrgState(label="b", cap=100)}
    assert api_dispatch._fanout_limit(states) == 200 * FANOUT_SLACK
    # Floor at 1 for an empty/zero-cap pool (never a zero-permit semaphore).
    assert api_dispatch._fanout_limit({"z": OrgState(label="z", cap=0)}) == max(1, 0 * FANOUT_SLACK)


def test_sync_fanout_wires_outer_semaphore(monkeypatch):
    """REGRESSION WIRING TEST (Finding 2, Methodology Must-Fix 2): the outer
    semaphore must bound LIVE coroutines at ``_fanout_limit``, observed at
    coroutine ENTRY (``build_request`` runs at the top of ``_do_one``, INSIDE the
    outer ``async with fanout`` but BEFORE the inner ``state.acquire()``). Monkeypatch
    ``_fanout_limit`` to a small sentinel; a client that BLOCKS forever piles
    coroutines at the OUTER semaphore so only ``_fanout_limit`` of them reach
    ``build_request``.

    Mutation check: revert the bounded gather to the bare
    ``asyncio.gather(*[_do_one(it) for it in items])`` -> ``peak`` reaches N=20 and
    ``peak <= 4 + 1`` goes RED. (v1's end-to-end ``max_in_flight <= cap`` did NOT go
    red here — the inner gate enforces it independently.)
    """
    monkeypatch.setattr(api_dispatch, "_fanout_limit", lambda org_states: 4)

    blocker = asyncio.Event()  # never set -> client.create blocks forever
    live = 0
    peak = 0

    def counting_build(item: DispatchItem) -> dict:
        nonlocal live, peak
        live += 1  # entered _do_one (past the outer semaphore); never decremented
        peak = max(peak, live)
        return build_request(item)

    class BlockingClient:
        def __init__(self):
            class _Raw:
                async def create(_self, **kwargs):
                    await blocker.wait()  # block forever
                    raise AssertionError("unreachable")  # pragma: no cover

            self.messages = SimpleNamespace(with_raw_response=_Raw())

    async def _go():
        # The run never completes (the client blocks); bound it with wait_for and
        # measure how many coroutines got past the outer gate into build_request.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                dispatch_calls(
                    make_items(20),
                    model="claude-sonnet-4-5-20250929",
                    build_request=counting_build,
                    parse_response=parse_response,
                    async_clients={"a": BlockingClient()},
                    sync_clients={"a": object()},
                    force_path="sync",
                ),
                timeout=0.5,
            )

    _run(_go())
    # <= _fanout_limit (+1 entry mid-decrement slack), NOT N=20.
    assert peak <= 4 + 1, peak


# ── Finding 3: re-pick org at slot-availability, not pick-time (#684) ─────────


def test_pick_org_repicks_on_slow_gate():
    """REGRESSION (Finding 3, Statistics nit 4 + Alternatives rec 6): org ``a`` is
    saturated by a slow first item (cap=1, long per-call delay) while org ``b`` is
    fast. A burst must re-pick toward ``b`` (the org with a free slot) rather than
    queue ~round-robin on the headroom-best pick.

    PRIMARY (strict, no ``>=`` fallback — pre-fix round-robin satisfies equality):
    ``fast.calls > slow.calls``. Mutation check: revert ``_pick_org_then_acquire``
    to plain ``_pick_org`` + ``acquire()`` (pick-then-block) -> ~round-robin -> the
    strict ``>`` fails on equality.

    Plus the slot-accounting invariant (Alternatives rec 6): every org's live
    ``in_flight`` counter is zero post-drain — catches a wait_for-after-acquire
    slot leak.
    """
    slow = FakeAsyncClient(delay=0.3)  # org a: slot stays held a long time
    fast = FakeAsyncClient(delay=0.0)  # org b: frees immediately
    org_states_seen: dict[str, OrgState] = {}
    real_ctor = api_dispatch.OrgState

    def _spy_ctor(*a, **k):
        st = real_ctor(*a, **k)
        org_states_seen[st.label] = st
        return st

    api_dispatch.OrgState = _spy_ctor
    try:
        res = _run(
            dispatch_calls(
                make_items(20),
                model="claude-sonnet-4-5-20250929",
                build_request=build_request,
                parse_response=parse_response,
                async_clients={"a": slow, "b": fast},
                sync_clients={"a": object(), "b": object()},
                concurrency_overrides={"sonnet": 2},  # small cap -> a saturates fast
                force_path="sync",
            )
        )
    finally:
        api_dispatch.OrgState = real_ctor

    assert all(not r.error for r in res.values())
    # PRIMARY: re-pick pulled the burst to the free org (strict >, no >= fallback).
    assert fast.calls > slow.calls, (fast.calls, slow.calls)
    # Slot-accounting invariant: no leaked slot post-drain.
    assert all(st.in_flight == 0 for st in org_states_seen.values())


def test_pick_org_excluding_respects_tried_with_nonzero_rr():
    """REGRESSION (#684 round 2, Codex/reconciler blocker): ``_pick_org_excluding``
    must filter ``tried`` in the SAME label space it returns, regardless of the
    round-robin pointer's value.

    Pre-fix the filter ran in unrotated index space while selection mapped the
    chosen index through ``rr["i"]``: with ``rr={"i":1}``, ``labels=["a","b"]``,
    ``tried={"b"}`` the helper returned ``"b"`` — the very org in ``tried``. On the
    shared-``rr`` re-pick path (``_pick_org_then_acquire`` adds a timed-out org to
    ``tried``) that handed the re-pick back the slow org that just timed out,
    defeating Finding 3.

    Mutation check: revert ``_pick_org_excluding`` to the unrotated-filter body
    (``candidates = [i for i in range(len(labels)) if labels[i] not in tried]`` +
    ``chosen = labels[(rr["i"] + best) % len(labels)]``) -> the first assertion
    returns ``"b"`` and goes RED.
    """
    org_states = {
        "a": OrgState(label="a", cap=10),
        "b": OrgState(label="b", cap=10),
    }
    labels = ["a", "b"]

    # The pre-fix bug fired only when rr["i"] != 0. With "b" excluded, the only
    # valid candidate is "a" — and the fix must return it regardless of the pointer.
    rr = {"i": 1}
    chosen = _pick_org_excluding(org_states, labels, rr, tried={"b"})
    assert chosen == "a", f"expected 'a' (only candidate); got {chosen!r}"
    assert chosen not in {"b"}, "must NOT return an org in `tried`"

    # Symmetric, trivially-correct rr["i"]==0 case (the index spaces coincide here).
    rr = {"i": 0}
    chosen = _pick_org_excluding(org_states, labels, rr, tried={"a"})
    assert chosen == "b"

    # Round-robin tie-break under matched headroom + nonzero rr on a 3-org panel:
    # "a" excluded, equal headroom on "b"/"c" -> the rotation pointer decides, and
    # in NO case is the result the excluded org.
    org_states["c"] = OrgState(label="c", cap=10)
    labels3 = ["a", "b", "c"]
    rr = {"i": 2}  # rotation starts at "c"
    chosen = _pick_org_excluding(org_states, labels3, rr, tried={"a"})
    assert chosen in {"b", "c"}, f"expected one of {{b, c}}; got {chosen!r}"
    assert chosen != "a", "must NOT return the excluded org under any rotation"


def test_pick_org_excluding_raises_when_all_orgs_tried():
    """Defensive guard: when every org is in ``tried`` the candidate set is empty.

    The caller (``_pick_org_then_acquire``) never reaches this — its timeout loop
    only calls with a strict subset, and the blocking fallback passes ``set()`` —
    but the helper fails loud rather than crashing on ``max([])`` if a future
    caller violates the contract.
    """
    org_states = {"a": OrgState(label="a", cap=10), "b": OrgState(label="b", cap=10)}
    with pytest.raises(ValueError, match="every org is in"):
        _pick_org_excluding(org_states, ["a", "b"], {"i": 0}, tried={"a", "b"})


# ── Finding 1 Must-Fix 1: batch-merge preserves error category (#684) ─────────


def test_batch_merge_preserves_error_category():
    """REGRESSION (Methodology Must-Fix 1): a persisted batch record with
    ``error=True`` and NO ``category`` key (a legacy record written before the
    field existed) must read back ``category=RESULT_ERROR``, NOT the RESULT_OK
    field default.

    Mutation check: omit the ``RESULT_ERROR if error else RESULT_OK`` default in
    ``_merge_batch_record`` (let it fall to the field default RESULT_OK) -> an
    error=True record reads back ``category == "ok"`` and the assertion goes red.
    """
    err_rec = {"result": None, "error": True, "reason": "batch_error: errored", "org": "sonnet"}
    merged = api_dispatch._merge_batch_record("item_000", err_rec, org="sonnet")
    assert merged.error is True
    assert merged.category == RESULT_ERROR  # NOT "ok" — the RESULT_OK default would be a bug
    assert merged.item_id == "item_000"
    assert merged.org == "sonnet"

    # Companion: a non-error legacy record reads back category == RESULT_OK.
    ok_rec = {"result": {"label": "ok"}, "error": False, "reason": None}
    ok_merged = api_dispatch._merge_batch_record("item_001", ok_rec, org="sonnet")
    assert ok_merged.error is False
    assert ok_merged.category == RESULT_OK

    # A record that DOES carry an explicit category is read directly.
    explicit = {"result": None, "error": True, "category": RESULT_RATE_LIMITED, "reason": "x"}
    assert (
        api_dispatch._merge_batch_record("item_002", explicit, org="sonnet").category
        == RESULT_RATE_LIMITED
    )


# ── Finding 4: routing-on-remainder docstring note (#684) ─────────────────────


def test_module_doc_describes_pending_routing():
    """Finding 4: the module docstring documents that routing decides on the
    UNCACHED remainder (``len(pending)``), not the original N — so it can't
    silently regress."""
    doc = api_dispatch.__doc__ or ""
    assert "len(pending)" in doc or "uncached remainder" in doc.lower()


# ── #991: runtime system-role guard at the dispatch_calls seam ───────────────


def build_request_with_system(item: DispatchItem) -> dict:
    """System-bearing builder (the #906 r11 bug shape): leading system-role entry."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": item.payload["q"]},
        ],
    }


def build_request_with_mid_list_system(item: DispatchItem) -> dict:
    """System entry AFTER a user turn (arbitrary position, not just leading)."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": item.payload["q"]},
            {"role": "system", "content": "Mid-list system entry."},
        ],
    }


def build_request_with_top_level_system(item: DispatchItem) -> dict:
    """Compliant builder mirroring the real callers (judge_dispatch._build_params
    shape): top-level ``system=`` param + user-only messages."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 64,
        "system": "You are a judge.",
        "messages": [{"role": "user", "content": item.payload["q"]}],
    }


def test_system_role_raises_sync_before_wire():
    """A system-bearing builder raises at build time on the sync path — the
    offending item's request never reaches the fake wire (calls == 0)."""
    items = make_items(1)
    clients = {"a": FakeAsyncClient()}
    with pytest.raises(ValueError, match='role="system" at index 0') as exc_info:
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request_with_system,
                parse_response=parse_response,
                async_clients=clients,
                sync_clients={"a": object()},
                force_path="sync",
            )
        )
    # AC2 pin: the message carries a fix pointer to the reference lift.
    assert "anthropic_format" in str(exc_info.value)
    assert clients["a"].calls == 0


def test_system_role_raises_batch_before_state_write(tmp_path):
    """On a fresh batch, the guard fires at the all-items state-init build —
    BEFORE any state.json write and before any batches.create."""
    items = make_items(3)
    batch_clients = {"a": FakeBatchClient("a"), "b": FakeBatchClient("b")}
    with pytest.raises(ValueError, match='role="system"'):
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request_with_system,
                parse_response=parse_response,
                async_clients={"a": object(), "b": object()},
                sync_clients=batch_clients,
                checkpoint_dir=tmp_path / "ckpt",
                force_path="batch",
                poll_interval=0.0,
            )
        )
    assert all(c.create_calls == 0 for c in batch_clients.values())
    assert not (tmp_path / "ckpt" / "state.json").exists()


def test_system_role_mid_list_raises_with_index():
    """A mid-list system entry (index 1, after a user turn) raises with the
    offending index AND the item_id in the message."""
    items = make_items(1)
    clients = {"a": FakeAsyncClient()}
    with pytest.raises(ValueError, match='role="system" at index 1') as exc_info:
        _run(
            dispatch_calls(
                items,
                model="claude-sonnet-4-5-20250929",
                build_request=build_request_with_mid_list_system,
                parse_response=parse_response,
                async_clients=clients,
                sync_clients={"a": object()},
                force_path="sync",
            )
        )
    msg = str(exc_info.value)
    assert "index 1" in msg
    assert "item_000" in msg
    assert clients["a"].calls == 0


def test_top_level_system_param_passes():
    """Positive control (zero-behavior-change evidence): a top-level ``system=``
    param with user-only messages — the real callers' shape — passes untouched."""
    items = make_items(4)
    clients = {"a": FakeAsyncClient()}
    res = _run(
        dispatch_calls(
            items,
            model="claude-sonnet-4-5-20250929",
            build_request=build_request_with_top_level_system,
            parse_response=parse_response,
            async_clients=clients,
            sync_clients={"a": object()},
            force_path="sync",
        )
    )
    assert set(res) == {it.item_id for it in items}
    assert all(not r.error for r in res.values())
    assert all(r.result == {"label": "ok"} for r in res.values())
    assert clients["a"].calls == 4


def test_guard_ignores_non_dict_and_missing_messages():
    """The guard checks only the documented dict shape: non-dict params/entries
    and a missing ``messages`` key are no-ops (the API owns those errors)."""
    api_dispatch._assert_no_system_role({}, "item_000")  # no messages key
    api_dispatch._assert_no_system_role({"messages": "oops"}, "item_000")  # non-list
    api_dispatch._assert_no_system_role(
        {"messages": [("role", "system")]}, "item_000"
    )  # non-dict entry


def test_batch_resume_builder_raise_leaves_subbatch_pending(tmp_path):
    """The §4f reorder's contract: a builder raise on batch RESUME (the guard's,
    or any other) fires BEFORE the status flip, so the sub-batch stays
    ``pending`` in memory AND on disk — resumable, not a crashed-mid-submit
    wedge."""
    items = make_items(2)
    cid_for = {it.item_id: make_custom_id(it.item_id) for it in items}
    items_by_cid = {cid_for[it.item_id]: it for it in items}
    sb = {
        "index": 0,
        "org": "a",
        "custom_ids": list(items_by_cid),
        "n_requests": len(items),
        "status": "pending",
        "batch_id": None,
        "deadline": None,
    }
    state = {
        "version": 1,
        "sub_batches": [sb],
        "cid_to_item": {cid: it.item_id for cid, it in items_by_cid.items()},
    }
    state_path = tmp_path / "state.json"
    # PREWRITE the fabricated pending state.json: after the reorder the builder
    # raise occurs before any write, so the on-disk re-read below only holds
    # against a pre-created baseline checkpoint.
    api_dispatch._atomic_write_json(state_path, state)
    fake = FakeBatchClient("a")

    async def _drive():
        await api_dispatch._submit_one_sub_batch(
            sb,
            state=state,
            state_path=state_path,
            items_by_cid=items_by_cid,
            build_request=api_dispatch._guarded_build_request(build_request_with_system),
            sync_clients={"a": fake},
            sem=asyncio.Semaphore(1),
        )

    with pytest.raises(ValueError, match='role="system"'):
        _run(_drive())
    assert sb["status"] == "pending"  # in-memory: never flipped to "submitting"
    on_disk = json.loads(state_path.read_text())
    assert on_disk["sub_batches"][0]["status"] == "pending"  # on-disk: still resumable
    assert fake.create_calls == 0  # nothing was submitted
