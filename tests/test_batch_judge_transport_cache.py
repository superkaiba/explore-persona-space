"""#1313: transport-vs-content classification + judge-cache transport hygiene.

Covers ``eval.batch_judge``'s rule-24 surface (llm-judging.md):

- ``is_transport_error_dict`` — the shared classifier (structural
  ``transport: True`` flag + the conservative legacy reason-string fallback);
- the cache PUT skip (a transport-class dict is never cached) and the cache
  GET miss (a STORED transport-class dict reads as a miss, so a re-run
  re-dispatches — self-heals legacy-poisoned caches, e.g. #1090's stored 529s);
- the ``JudgeCache`` key-schema byte-pin (rule 22 hard constraint: #1313
  changes cache VALUES behavior only, never the key schema).

Mock strategy mirrors ``tests/test_judge_dispatch.py``: inject client fakes,
no live API calls.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from explore_persona_space.eval.batch_judge import (
    JudgeCache,
    _enumerate_and_check_cache,
    _legacy_error_dict,
    is_transport_error_dict,
    judge_completions_batch,
    rubric_fingerprint,
)

JUDGE_TEXT = '{"aligned": 90, "coherent": 95, "reasoning": "ok"}'


def _msg(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _real_overloaded_error():
    """The REAL production 529 shape (anthropic 0.88.0; test-only private import)."""
    from anthropic._exceptions import OverloadedError

    resp = SimpleNamespace(status_code=529, headers={}, request=SimpleNamespace())
    return OverloadedError("overloaded", response=resp, body=None)


class _FaultableAsyncClient:
    """AsyncAnthropic stand-in: raises ``fault_for(user_msg)`` when non-None."""

    def __init__(self, *, fault_for=None, text: str = JUDGE_TEXT):
        self.fault_for = fault_for or (lambda user_msg: None)
        self.text = text
        self.calls: list[str] = []
        client = self

        class _Messages:
            async def create(_self, **kwargs):
                user_msg = kwargs["messages"][0]["content"]
                client.calls.append(user_msg)
                fault = client.fault_for(user_msg)
                if fault is not None:
                    raise fault
                return _msg(client.text)

        self.messages = _Messages()


# ── classifier units ──────────────────────────────────────────────────────────


def test_transient_reason_prefix_classified_transport():
    """#1313 §5 row c2 sibling: a LEGACY (unflagged) multi-org-reduce dict whose
    reason is an api_dispatch transient-exhaustion string classifies transport
    via the ``"transient "`` prefix (no 'overloaded' substring needed)."""
    legacy = {
        "aligned": None,
        "coherent": None,
        "reasoning": "transient APIConnectionError (attempt 3)",
        "error": True,
    }
    assert is_transport_error_dict(legacy) is True
    # The sibling exception names classify identically.
    for name in ("APITimeoutError", "InternalServerError"):
        assert (
            is_transport_error_dict({**legacy, "reasoning": f"transient {name} (attempt 5)"})
            is True
        ), name


def test_classifier_ordering_invariants():
    """#1313 §5 row c3 (methodology-critic concern): (i) the invalid_request
    check runs BEFORE the errored-prefix check — a quarantined-400 reason never
    classifies transport even in the ``batch_error: errored (...)`` shape;
    (ii) the ``error`` gate runs before the substring fallback — a NON-error
    dict whose judge REASONING text contains 'overloaded' is never transport."""
    # (i) invalid_request inside the errored(...) shape -> False.
    quarantined = _legacy_error_dict("batch_error: errored (invalid_request_error (quarantined))")
    assert is_transport_error_dict(quarantined) is False
    assert (
        is_transport_error_dict(
            _legacy_error_dict("batch_error: invalid_request_error (quarantined)")
        )
        is False
    )
    # Server-class errored shape -> True.
    assert is_transport_error_dict(_legacy_error_dict("batch_error: errored (api_error)")) is True
    # (ii) a successful judgment whose reasoning mentions 'overloaded' -> False.
    verdict = {"aligned": 90, "coherent": 95, "reasoning": "the model seemed overloaded"}
    assert is_transport_error_dict(verdict) is False
    # Non-dict / flagless non-error inputs are never transport.
    assert is_transport_error_dict("banana") is False
    assert is_transport_error_dict(None) is False
    # Structural flag wins regardless of reason text.
    assert is_transport_error_dict({"error": True, "transport": True, "reasoning": "x"}) is True
    # A content-class error (parse_error) is NOT transport.
    assert is_transport_error_dict(_legacy_error_dict("parse_error")) is False
    # The 429-exhaustion reason (api_dispatch) classifies transport.
    assert (
        is_transport_error_dict(_legacy_error_dict("rate_limited_exhausted (org=a, 429 retries 6)"))
        is True
    )
    # The 529 fallback: 'overloaded' substring on an ERROR dict -> True (#1090 shape).
    assert is_transport_error_dict(_legacy_error_dict("error: 529 overloaded")) is True


# ── cache value hygiene (PUT skip / GET miss) ─────────────────────────────────


def test_cache_put_skips_transport_error_dicts(tmp_path):
    """#1313 §4.2(c): a transport-class result (real 529 captured on the sync
    path) is NEVER cache-PUT — a re-run re-dispatches it — while the sibling
    success IS cached (and a second run serves it from cache)."""
    cache_dir = tmp_path / "cache"
    completions = {"item": {"q-fail": ["c-fail"], "q-ok": ["c-ok"]}}

    def fault_for(user_msg):
        return _real_overloaded_error() if "c-fail" in user_msg else None

    def _run_once():
        client = _FaultableAsyncClient(fault_for=fault_for)
        judge_completions_batch(
            completions=completions,
            judge_system_prompt="SYS",
            judge_model="claude-sonnet-4-5-20250929",
            cache_dir=cache_dir,
            force_sync=True,
            sync_client=client,
        )
        return client

    c1 = _run_once()
    assert len(c1.calls) == 2  # both dispatched on the cold run
    # Exactly ONE cache entry (the success); the transport failure was skipped.
    assert len(list(cache_dir.glob("*.json"))) == 1

    c2 = _run_once()
    # Second run: the success is served from cache; ONLY the transport-failed
    # item re-dispatches (the re-run/re-dispatch contract, rule 24(ii)).
    assert len(c2.calls) == 1
    assert "c-fail" in c2.calls[0]


def test_cache_get_treats_stored_transport_error_as_miss(tmp_path):
    """#1313 §4.2(d): a PRE-SEEDED (legacy-poisoned) cache entry that is a
    transport-class dict reads as a MISS — the item lands in
    ``uncached_items`` for re-dispatch — while a stored VERDICT still hits.
    Self-heals #1090-class caches without manual surgery."""
    cache = JudgeCache(tmp_path / "cache")
    rk = rubric_fingerprint("m", "SYS", None)
    # Legacy transport poison (pre-#1313 shape: no structural flag).
    cache.put("q-poisoned", "c-poisoned", _legacy_error_dict("batch_error: expired"), rubric_key=rk)
    good = {"aligned": 90, "coherent": 95, "reasoning": "ok"}
    cache.put("q-good", "c-good", good, rubric_key=rk)

    completions = {"item": {"q-poisoned": ["c-poisoned"], "q-good": ["c-good"]}}
    total, cached_scores, uncached_items = _enumerate_and_check_cache(
        completions, cache, lambda q, c: f"{q}|{c}", rubric_key=rk
    )
    assert total == 2
    # The good verdict hits; the poisoned entry is a MISS -> re-dispatch.
    assert list(cached_scores.values()) == [good]
    assert [(q, c) for _cid, q, c, _u in uncached_items] == [("q-poisoned", "c-poisoned")]


# ── key-schema byte-pin (rule 22 / #1018 hard constraint) ─────────────────────


def test_cache_key_schema_unchanged(tmp_path):
    """DURABILITY PIN (#1313 §7 kill criterion / rule 22): the ``JudgeCache``
    key schema is byte-identical to the pre-#1313 EPM_JUDGE_CACHE_KEY_V2
    schema — the literal hash below was computed on the pre-change tree for
    these fixed inputs. Any key-schema drift (which would silently orphan or
    cross-serve every existing cache entry) goes red here."""
    assert JudgeCache._hash_key("q-fixed", "c-fixed", rubric_key="rk-fixed") == "4f4c8da2695715c2"
    # And the schema version tag is still V2 (no bump rode along).
    from explore_persona_space.eval.batch_judge import _JUDGE_CACHE_KEY_VERSION

    assert _JUDGE_CACHE_KEY_VERSION == "EPM_JUDGE_CACHE_KEY_V2"


def test_transport_flag_roundtrip_via_sync_capture():
    """End-to-end shape check: the sync captured-exception mint produces a dict
    the classifier recognizes (structural flag, no reason-string reliance)."""
    from explore_persona_space.eval.judge_dispatch import _default_error_dict, _judge_items_sync

    client = _FaultableAsyncClient(fault_for=lambda user_msg: _real_overloaded_error())
    results = asyncio.run(
        _judge_items_sync(
            [("cid", "q", "c", "u")],
            judge_model="claude-sonnet-4-5-20250929",
            judge_system_prompt="SYS",
            max_tokens=64,
            max_concurrent=1,
            error_dict_factory=_default_error_dict,
            client=client,
        )
    )
    assert is_transport_error_dict(results["cid"]) is True
    # The flag survives a JSON round-trip (save_raw / cache-file shape).
    assert is_transport_error_dict(json.loads(json.dumps(results["cid"]))) is True
