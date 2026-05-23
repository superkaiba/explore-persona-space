"""Tests for the round-6 auditor-rotation infrastructure in
``explore_persona_space.data_gen.issue377_corpus``.

Round 5 ended at the philosophy refusal cascade (turn 18-21, 30%
refusal rate at turn 20). Round 6 mirrors Lu et al. 2026's protocol:
15 turns max, and per-conversation auditor rotation between
Claude-Sonnet-4.5 and GPT-5. These tests cover the rotation logic
(deterministic seed-keyed assignment, even-ish split across many
conversation ids, OpenAI / Anthropic backend dispatch) plus the
guard rails (missing OPENAI_API_KEY, unknown model id).

These tests are PURE — no network calls, no API keys required at
import time. The OpenAI path tests live in
``TestSubmitOpenAiSyncBatch`` and use ``monkeypatch.delenv`` /
``pytest.raises`` to assert the error path; the success path uses a
``monkeypatch.setattr`` shim that injects a fake ``openai.AsyncClient``
so we exercise the full request-translation logic without hitting the
network.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from unittest.mock import MagicMock

import pytest

from explore_persona_space.data_gen.issue377_corpus import (
    _OPENAI_REASONING_TOKEN_HEADROOM,
    AUDITOR_MODELS_AVAILABLE,
    N_TURNS_TOTAL,
    _is_reasoning_model,
    _openai_request_to_kwargs,
    assign_auditor_model,
    is_anthropic_model,
    is_openai_model,
    run_per_auditor_batch,
    submit_openai_sync_batch,
)

# ── N_TURNS_TOTAL pinned to 15 (round-6 protocol) ──────────────────────────


class TestProtocolConstants:
    def test_n_turns_total_is_15(self):
        """The round-6 protocol replicates Lu et al. 2026's '≤15 turns'."""
        assert N_TURNS_TOTAL == 15

    def test_auditor_pool_size_is_two(self):
        """Round-6 rotation pool: Sonnet-4.5 + GPT-5; Kimi K2 omitted."""
        assert len(AUDITOR_MODELS_AVAILABLE) == 2

    def test_auditor_pool_contains_sonnet_and_gpt5(self):
        assert any("claude-sonnet-4-5" in m for m in AUDITOR_MODELS_AVAILABLE)
        assert any(m.startswith("gpt-5") for m in AUDITOR_MODELS_AVAILABLE)


# ── assign_auditor_model: determinism + balance ────────────────────────────


class TestAssignAuditorModel:
    def test_deterministic_same_seed(self):
        """Same (conv_id, seed) → same auditor across calls."""
        cid = "therapy_p0_t0"
        seed = 42
        first = assign_auditor_model(cid, seed)
        for _ in range(10):
            assert assign_auditor_model(cid, seed) == first

    def test_returns_pool_member(self):
        """The returned model must be in AUDITOR_MODELS_AVAILABLE."""
        for cid in ["a", "b", "c", "philosophy_p2_t7", "hostile_jailbreak_p4_t9"]:
            for seed in [0, 1, 42, 137]:
                assert assign_auditor_model(cid, seed) in AUDITOR_MODELS_AVAILABLE

    def test_distinct_conv_ids_can_get_different_auditors(self):
        """Across many conv ids at one seed, both auditors appear."""
        seed = 0
        seen = {
            assign_auditor_model(f"domain_p{p}_t{t}", seed) for p in range(5) for t in range(10)
        }
        # With a 2-member pool and 50 distinct ids the assignment is
        # essentially certain to surface both models.
        assert seen == set(AUDITOR_MODELS_AVAILABLE)

    def test_balance_within_15_percent_for_large_grid(self):
        """For 50 conversations x 4 domains = 200 ids, expect 50/50 +/- 15pp.

        The SHA-256-based hash is well-distributed so the 80/20 cells
        are vanishingly rare. Loose tolerance (±15pp) protects against
        a one-off bad-luck seed without making the test brittle.
        """
        seed = 0
        counts: Counter[str] = Counter()
        for domain in ["therapy", "philosophy", "roleplay", "hostile_jailbreak"]:
            for p in range(5):
                for t in range(10):
                    counts[assign_auditor_model(f"{domain}_p{p}_t{t}", seed)] += 1
        total = sum(counts.values())
        assert total == 200
        for model in AUDITOR_MODELS_AVAILABLE:
            frac = counts[model] / total
            assert 0.35 <= frac <= 0.65, (
                f"Auditor {model} got {counts[model]}/{total} "
                f"= {frac:.1%} at seed={seed} — outside [35%, 65%]"
            )

    def test_different_seeds_produce_different_assignments(self):
        """Changing the seed should change the assignment for SOME ids."""
        cid_grid = [f"d_p{p}_t{t}" for p in range(5) for t in range(10)]
        a = [assign_auditor_model(cid, seed=0) for cid in cid_grid]
        b = [assign_auditor_model(cid, seed=1) for cid in cid_grid]
        # At least one cell should flip between seeds (essentially certain
        # for a 2-member pool over 50 ids).
        assert a != b

    def test_empty_pool_raises(self, monkeypatch):
        """Defensive: an empty pool must raise, not silently return ''."""
        import explore_persona_space.data_gen.issue377_corpus as mod

        monkeypatch.setattr(mod, "AUDITOR_MODELS_AVAILABLE", ())
        with pytest.raises(RuntimeError, match="AUDITOR_MODELS_AVAILABLE is empty"):
            assign_auditor_model("x", 0)


# ── Backend predicates ─────────────────────────────────────────────────────


class TestBackendPredicates:
    def test_anthropic_detected(self):
        assert is_anthropic_model("claude-sonnet-4-5-20250929")
        assert is_anthropic_model("claude-opus-4-7")
        assert not is_anthropic_model("gpt-5")
        assert not is_anthropic_model("o3-mini")

    def test_openai_detected(self):
        assert is_openai_model("gpt-5")
        assert is_openai_model("gpt-4o")
        assert is_openai_model("o1-preview")
        assert is_openai_model("o3-mini")
        assert not is_openai_model("claude-sonnet-4-5-20250929")

    def test_unknown_model_returns_false_on_both(self):
        assert not is_anthropic_model("llama-3")
        assert not is_openai_model("llama-3")


# ── OpenAI request translation ─────────────────────────────────────────────


class TestOpenAiRequestTranslation:
    def test_system_becomes_leading_message(self):
        req = {
            "custom_id": "test_1",
            "params": {
                "model": "gpt-5",
                "system": "You are a debate-club user.",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 500,
            },
        }
        model, msgs, max_tokens = _openai_request_to_kwargs(req)
        assert model == "gpt-5"
        assert max_tokens == 500
        assert msgs[0] == {"role": "system", "content": "You are a debate-club user."}
        assert msgs[1] == {"role": "user", "content": "Hi"}

    def test_no_system_means_no_leading_system_message(self):
        req = {
            "custom_id": "test_2",
            "params": {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 500,
            },
        }
        _, msgs, _ = _openai_request_to_kwargs(req)
        # No system → messages start with user.
        assert msgs[0]["role"] == "user"

    def test_default_max_tokens_when_missing(self):
        req = {
            "custom_id": "test_3",
            "params": {
                "model": "gpt-5",
                "system": "S",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        }
        _, _, max_tokens = _openai_request_to_kwargs(req)
        assert max_tokens == 800  # default in helper


# ── submit_openai_sync_batch error paths ───────────────────────────────────


class TestSubmitOpenAiSyncBatchErrorPaths:
    def test_empty_input_returns_empty_dict(self, monkeypatch):
        """Empty input must short-circuit without touching the API."""
        # Even WITHOUT OPENAI_API_KEY set, an empty batch should not raise.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert submit_openai_sync_batch([]) == {}

    def test_missing_api_key_raises(self, monkeypatch):
        """Production-shaped requests must fail loudly with a clear error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        req = {
            "custom_id": "x",
            "params": {
                "model": "gpt-5",
                "system": "S",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
        }
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY not in environment"):
            submit_openai_sync_batch([req])


# ── submit_openai_sync_batch happy path (no real network) ──────────────────


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = MagicMock()
        self.message.content = content
        self.finish_reason = finish_reason


class _FakeResp:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason=finish_reason)]


class _FakeAsyncOpenAI:
    """Drop-in shim that captures every chat.completions.create call.

    Returns a deterministic, custom_id-derived completion so the test
    can assert (a) custom_id keying is preserved, (b) the system+user
    payload was translated correctly, and (c) failures are surfaced
    as the BATCH_ERROR sentinel.

    Supports the async context-manager protocol (``async with
    openai.AsyncClient() as client:``) so the production code path's
    connection-pool teardown sequence is exercised in tests.

    ``empty_substring`` triggers an empty-content response (simulates a
    reasoning model whose ``max_completion_tokens`` budget was burned
    on thinking before any output emerged); ``fail_substring`` triggers
    a raised exception.
    """

    def __init__(self, fail_substring=None, empty_substring=None):
        self.calls: list[dict] = []
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create
        self._fail_substring = fail_substring
        self._empty_substring = empty_substring

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _create(self, *, model, messages, max_completion_tokens):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        # Surface the user content so the test can confirm round-trip.
        user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
        if self._fail_substring and self._fail_substring in user_text:
            raise RuntimeError("simulated transient API failure")
        if self._empty_substring and self._empty_substring in user_text:
            # Reasoning model burned its budget on thinking.
            return _FakeResp(content="", finish_reason="length")
        return _FakeResp(content=f"reply-to:{user_text}", finish_reason="stop")


class TestSubmitOpenAiSyncBatchHappyPath:
    def test_round_trips_two_requests(self, monkeypatch):
        import openai

        fake = _FakeAsyncOpenAI()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(openai, "AsyncClient", lambda: fake)

        reqs = [
            {
                "custom_id": "conv_p0_t00_user",
                "params": {
                    "model": "gpt-5",
                    "system": "S0",
                    "messages": [{"role": "user", "content": "Q0"}],
                    "max_tokens": 100,
                },
            },
            {
                "custom_id": "conv_p1_t00_user",
                "params": {
                    "model": "gpt-5",
                    "system": "S1",
                    "messages": [{"role": "user", "content": "Q1"}],
                    "max_tokens": 100,
                },
            },
        ]
        out = submit_openai_sync_batch(reqs)
        assert out == {
            "conv_p0_t00_user": "reply-to:Q0",
            "conv_p1_t00_user": "reply-to:Q1",
        }
        assert len(fake.calls) == 2
        # Confirm system→leading-message translation happened.
        assert fake.calls[0]["messages"][0] == {"role": "system", "content": "S0"}
        # GPT-5 is a reasoning model — the helper adds the
        # _OPENAI_REASONING_TOKEN_HEADROOM on top of the request's
        # ``max_tokens`` so the visible-output budget stays in parity
        # with the Sonnet path.
        assert fake.calls[0]["max_completion_tokens"] == 100 + _OPENAI_REASONING_TOKEN_HEADROOM

    def test_failure_maps_to_batch_error_sentinel(self, monkeypatch):
        """An exception in one call must NOT abort the whole batch — the
        failing item becomes ``[BATCH_ERROR]`` so downstream sanity
        checks behave identically to the Anthropic path."""
        import openai

        fake = _FakeAsyncOpenAI(fail_substring="QFAIL")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(openai, "AsyncClient", lambda: fake)

        # Speed up the retry sleep so the test stays sub-second.
        monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: asyncio.sleep(0))

        reqs = [
            {
                "custom_id": "ok",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "QOK"}],
                    "max_tokens": 50,
                },
            },
            {
                "custom_id": "bad",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "QFAIL"}],
                    "max_tokens": 50,
                },
            },
        ]
        out = submit_openai_sync_batch(reqs)
        assert out["ok"] == "reply-to:QOK"
        assert out["bad"] == "[BATCH_ERROR]"

    def test_empty_content_becomes_batch_error(self, monkeypatch):
        """A reasoning model that exhausts its budget returns ``content=""``
        with ``finish_reason='length'``. The helper must surface this as
        ``[BATCH_ERROR]`` rather than silently returning the empty string
        — downstream sanity checks would otherwise count it as a
        successful turn."""
        import openai

        fake = _FakeAsyncOpenAI(empty_substring="QBURN")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(openai, "AsyncClient", lambda: fake)

        reqs = [
            {
                "custom_id": "good",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "QOK"}],
                    "max_tokens": 50,
                },
            },
            {
                "custom_id": "burned",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "QBURN"}],
                    "max_tokens": 50,
                },
            },
        ]
        out = submit_openai_sync_batch(reqs)
        assert out["good"] == "reply-to:QOK"
        # Empty content (whether None, "" or whitespace) becomes the
        # batch-error sentinel so the downstream refusal/error gate sees
        # it identically to a hard API failure.
        assert out["burned"] == "[BATCH_ERROR]"


# ── Reasoning-model headroom (round-6 second-pass fix) ─────────────────────


class TestReasoningModelHeadroom:
    def test_predicate_recognizes_gpt5_family(self):
        assert _is_reasoning_model("gpt-5")
        assert _is_reasoning_model("gpt-5-2026-01-15")
        assert _is_reasoning_model("o1")
        assert _is_reasoning_model("o1-mini")
        assert _is_reasoning_model("o1-preview")
        assert _is_reasoning_model("o3")
        assert _is_reasoning_model("o3-mini")
        assert _is_reasoning_model("o4-mini")

    def test_predicate_rejects_non_reasoning(self):
        assert not _is_reasoning_model("gpt-4o")
        assert not _is_reasoning_model("gpt-4o-mini")
        assert not _is_reasoning_model("gpt-4-turbo")
        assert not _is_reasoning_model("gpt-3.5-turbo")
        assert not _is_reasoning_model("claude-sonnet-4-5-20250929")

    def test_headroom_is_positive_and_documented(self):
        """The headroom must be enough for typical GPT-5 reasoning on a
        few-sentence drift-conversation turn. ~3200 tokens covers the
        upper end of observed reasoning spend with a small margin. A
        smaller value would re-introduce the round-6-probe-v1 failure
        (4 of 4 GPT-5 cells emitting empty content)."""
        assert _OPENAI_REASONING_TOKEN_HEADROOM >= 1000
        assert _OPENAI_REASONING_TOKEN_HEADROOM <= 10000

    def test_sonnet_request_does_not_get_headroom(self, monkeypatch):
        """Confirm via the FakeAsyncOpenAI shim that a Sonnet model (not
        in this client's path, but hypothetically) would NOT receive
        the headroom bump. The reasoning headroom must only apply to
        reasoning models."""
        # The OpenAI sync path is only invoked for OpenAI models, so
        # this test pins the helper's predicate, not the dispatcher.
        # If a future caller routes a Sonnet model through the OpenAI
        # backend (e.g. via a config typo), the headroom must NOT
        # kick in.
        assert not _is_reasoning_model("claude-sonnet-4-5-20250929")

    def test_gpt5_request_gets_headroom_in_actual_call(self, monkeypatch):
        """End-to-end: a GPT-5 request with ``max_tokens=800`` reaches
        the API with ``max_completion_tokens = 800 + headroom``."""
        import openai

        fake = _FakeAsyncOpenAI()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(openai, "AsyncClient", lambda: fake)

        reqs = [
            {
                "custom_id": "x",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "Q"}],
                    "max_tokens": 800,
                },
            },
        ]
        out = submit_openai_sync_batch(reqs)
        assert out["x"] == "reply-to:Q"
        assert fake.calls[0]["max_completion_tokens"] == 800 + _OPENAI_REASONING_TOKEN_HEADROOM


# ── run_per_auditor_batch dispatcher ───────────────────────────────────────


class TestRunPerAuditorBatch:
    def test_unknown_model_raises(self):
        """A request with a non-Anthropic, non-OpenAI model must error."""
        req = {
            "custom_id": "x",
            "params": {
                "model": "llama-3-70b",
                "system": "S",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
        }
        with pytest.raises(RuntimeError, match="Unrecognized model backend"):
            run_per_auditor_batch([req])

    def test_empty_input_returns_empty_dict(self):
        assert run_per_auditor_batch([]) == {}

    def test_all_openai_routes_to_openai_only(self, monkeypatch):
        """If every request is GPT-5, the Anthropic batch path must not be
        touched (no ANTHROPIC_BATCH_KEY required)."""
        import explore_persona_space.data_gen.issue377_corpus as mod

        # If we accidentally take the Anthropic branch, this would blow
        # up with KeyError on ANTHROPIC_BATCH_KEY (delenv to make sure).
        monkeypatch.delenv("ANTHROPIC_BATCH_KEY", raising=False)
        monkeypatch.setattr(
            mod,
            "submit_openai_sync_batch",
            lambda reqs: {r["custom_id"]: f"ok-{r['custom_id']}" for r in reqs},
        )

        # Anthropic branch fns should never be called; spy and assert.
        def _unreachable(*a, **kw):
            raise AssertionError("Anthropic batch path called on all-OpenAI input")

        monkeypatch.setattr(mod, "submit_batch", _unreachable)
        monkeypatch.setattr(mod, "wait_for_batch", _unreachable)
        monkeypatch.setattr(mod, "collect_batch_results", _unreachable)

        reqs = [
            {
                "custom_id": "c1",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "Q"}],
                    "max_tokens": 50,
                },
            },
        ]
        out = run_per_auditor_batch(reqs)
        assert out == {"c1": "ok-c1"}

    def test_all_anthropic_routes_to_anthropic_only(self, monkeypatch):
        """If every request is Sonnet, the OpenAI path must not be touched
        (no OPENAI_API_KEY required)."""
        import explore_persona_space.data_gen.issue377_corpus as mod

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(mod, "submit_batch", lambda reqs: "fake-batch-id")
        monkeypatch.setattr(mod, "wait_for_batch", lambda batch_id: None)
        monkeypatch.setattr(
            mod,
            "collect_batch_results",
            lambda batch_id: {"c1": "anth-ok"},
        )

        def _unreachable(*a, **kw):
            raise AssertionError("OpenAI sync path called on all-Anthropic input")

        monkeypatch.setattr(mod, "submit_openai_sync_batch", _unreachable)

        reqs = [
            {
                "custom_id": "c1",
                "params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "system": "S",
                    "messages": [{"role": "user", "content": "Q"}],
                    "max_tokens": 50,
                },
            },
        ]
        out = run_per_auditor_batch(reqs)
        assert out == {"c1": "anth-ok"}

    def test_mixed_routes_to_both_and_merges(self, monkeypatch):
        """A mixed batch must hit both backends and merge on custom_id."""
        import explore_persona_space.data_gen.issue377_corpus as mod

        monkeypatch.setattr(mod, "submit_batch", lambda reqs: "fake-batch-id")
        monkeypatch.setattr(mod, "wait_for_batch", lambda batch_id: None)
        monkeypatch.setattr(
            mod,
            "collect_batch_results",
            lambda batch_id: {"anth1": "anth-ok-1", "anth2": "anth-ok-2"},
        )
        monkeypatch.setattr(
            mod,
            "submit_openai_sync_batch",
            lambda reqs: {r["custom_id"]: f"oai-ok-{r['custom_id']}" for r in reqs},
        )

        reqs = [
            {
                "custom_id": "anth1",
                "params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "system": "S",
                    "messages": [{"role": "user", "content": "Q"}],
                    "max_tokens": 50,
                },
            },
            {
                "custom_id": "oai1",
                "params": {
                    "model": "gpt-5",
                    "system": "S",
                    "messages": [{"role": "user", "content": "Q"}],
                    "max_tokens": 50,
                },
            },
            {
                "custom_id": "anth2",
                "params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "system": "S",
                    "messages": [{"role": "user", "content": "Q"}],
                    "max_tokens": 50,
                },
            },
        ]
        out = run_per_auditor_batch(reqs)
        assert out == {
            "anth1": "anth-ok-1",
            "anth2": "anth-ok-2",
            "oai1": "oai-ok-oai1",
        }
