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
    is_api_refusal_error_dict,
    is_api_refusal_stop_reason,
    is_transport_error_dict,
    is_truncation_error_dict,
    is_truncation_stop_reason,
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
    """AsyncAnthropic stand-in: raises ``fault_for(user_msg)`` when non-None.

    ``text_for(user_msg)`` / ``stop_reason_for(user_msg)`` (both optional,
    #2021) vary the returned text / attach a str ``stop_reason`` attribute to
    the returned message per call; ``None`` returns keep the legacy shape
    (fixed text, no stop_reason attribute) so existing tests are unchanged.
    """

    def __init__(
        self, *, fault_for=None, text: str = JUDGE_TEXT, text_for=None, stop_reason_for=None
    ):
        self.fault_for = fault_for or (lambda user_msg: None)
        self.text = text
        self.text_for = text_for
        self.stop_reason_for = stop_reason_for
        self.calls: list[str] = []
        client = self

        class _Messages:
            async def create(_self, **kwargs):
                user_msg = kwargs["messages"][0]["content"]
                client.calls.append(user_msg)
                fault = client.fault_for(user_msg)
                if fault is not None:
                    raise fault
                text = client.text if client.text_for is None else client.text_for(user_msg)
                msg = _msg(text)
                if client.stop_reason_for is not None:
                    sr = client.stop_reason_for(user_msg)
                    if sr is not None:
                        msg.stop_reason = sr
                return msg

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
    # ── #2021 truncation-classifier invariants ────────────────────────────────
    trunc = {**_legacy_error_dict("parse_error"), "stop_reason": "max_tokens"}
    assert is_truncation_error_dict(trunc) is True
    # Disjointness, mint-site shapes: a truncation dict is NOT transport-class
    # and a transport dict (no stop_reason by construction) is NOT
    # truncation-class.
    assert is_transport_error_dict(trunc) is False
    transport = {**_legacy_error_dict("error: 529 overloaded"), "transport": True}
    assert is_truncation_error_dict(transport) is False
    # A NON-error dict never classifies truncation — a kept-but-truncated
    # verdict is a scored draw (visible only in the stop_reason tally).
    assert is_truncation_error_dict({"score": 55, "stop_reason": "max_tokens"}) is False
    # Legacy no-field -> False (unknown classifies content, never KeyError).
    assert is_truncation_error_dict(_legacy_error_dict("parse_error")) is False
    assert is_truncation_error_dict("banana") is False
    assert is_truncation_error_dict(None) is False
    # The stop_reason predicate itself: both truncation vocabularies, and
    # non-str / non-truncation values are False.
    assert is_truncation_stop_reason("max_tokens") is True
    assert is_truncation_stop_reason("length") is True  # OpenAI-normalized vocabulary
    assert is_truncation_stop_reason("end_turn") is False
    assert is_truncation_stop_reason(None) is False
    assert is_truncation_stop_reason(83399) is False
    # ── #2151 api-refusal-classifier invariants ───────────────────────────────
    api_ref = {**_legacy_error_dict("parse_error"), "raw_text": "", "stop_reason": "refusal"}
    assert is_api_refusal_error_dict(api_ref) is True
    # Three-way disjointness on mint-site shapes: api-refusal is NEITHER
    # transport (no flag, reason is parse_error) NOR truncation (stop_reason
    # "refusal" is not truncation-class); the truncation + transport shapes
    # are NOT api-refusal.
    assert is_transport_error_dict(api_ref) is False
    assert is_truncation_error_dict(api_ref) is False
    assert is_api_refusal_error_dict(trunc) is False
    assert is_api_refusal_error_dict(transport) is False
    # A KEPT verdict carrying stop_reason "refusal" is a scored draw, never
    # the api-refusal class (keys on the ERROR dict).
    assert is_api_refusal_error_dict({"score": 55, "stop_reason": "refusal"}) is False
    # Legacy no-field -> False (classifies content, never KeyError).
    assert is_api_refusal_error_dict(_legacy_error_dict("parse_error")) is False
    assert is_api_refusal_error_dict("banana") is False
    assert is_api_refusal_error_dict(None) is False
    # The stop_reason predicate: "refusal" only; non-str is False.
    assert is_api_refusal_stop_reason("refusal") is True
    assert is_api_refusal_stop_reason("max_tokens") is False
    assert is_api_refusal_stop_reason(None) is False
    assert is_api_refusal_stop_reason(83399) is False


def test_legacy_collector_attaches_stop_reason():
    """#2021 Edit 3d: the LEGACY batch collector (``_collect_legacy_results``)
    carries the response's stop_reason on parsed verdicts AND parse-failure
    error dicts; a message without the attribute attaches nothing; rows with
    no API response (expired) carry no key."""
    from explore_persona_space.eval.batch_judge import _collect_legacy_results

    def _sr_msg(text, stop_reason=None):
        msg = _msg(text)
        if stop_reason is not None:
            msg.stop_reason = stop_reason
        return msg

    def _results(_batch_id):
        yield SimpleNamespace(
            custom_id="ok",
            result=SimpleNamespace(type="succeeded", message=_sr_msg(JUDGE_TEXT, "end_turn")),
        )
        yield SimpleNamespace(
            custom_id="trunc",
            result=SimpleNamespace(
                type="succeeded", message=_sr_msg("cut off mid-rationa", "max_tokens")
            ),
        )
        yield SimpleNamespace(
            custom_id="legacy",
            result=SimpleNamespace(type="succeeded", message=_sr_msg(JUDGE_TEXT)),
        )
        yield SimpleNamespace(custom_id="exp", result=SimpleNamespace(type="expired", message=None))

    client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(results=_results)))
    results: dict = {}
    _collect_legacy_results(client, "msgbatch_x", results)
    assert results["ok"] == {
        "aligned": 90,
        "coherent": 95,
        "reasoning": "ok",
        "stop_reason": "end_turn",
    }
    assert results["trunc"]["error"] is True
    assert results["trunc"]["reasoning"] == "parse_error"
    assert results["trunc"]["stop_reason"] == "max_tokens"
    assert results["legacy"] == {"aligned": 90, "coherent": 95, "reasoning": "ok"}
    assert results["exp"]["error"] is True
    assert "stop_reason" not in results["exp"]


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


def test_cache_put_skips_truncation_error_dicts(tmp_path):
    """#2021 (rule 23 mirror of the #1313 transport put-skip): a
    truncation-class parse-failure result (garbage text at
    stop_reason="max_tokens" on the sync path) is NEVER cache-PUT — a re-run
    (e.g. after a budget raise, which does NOT change the rubric key)
    re-dispatches it — while the sibling success IS cached."""
    cache_dir = tmp_path / "cache"
    completions = {"item": {"q-trunc": ["c-trunc"], "q-ok": ["c-ok"]}}

    def text_for(user_msg):
        return "cut off mid-rationa" if "c-trunc" in user_msg else JUDGE_TEXT

    def stop_reason_for(user_msg):
        return "max_tokens" if "c-trunc" in user_msg else "end_turn"

    def _run_once():
        client = _FaultableAsyncClient(text_for=text_for, stop_reason_for=stop_reason_for)
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
    # Exactly ONE cache entry (the success); the truncation failure was skipped.
    assert len(list(cache_dir.glob("*.json"))) == 1

    c2 = _run_once()
    # Second run: the success is served from cache; ONLY the truncated item
    # re-dispatches (the budget-raise self-heal contract, rule 23 / #2021).
    assert len(c2.calls) == 1
    assert "c-trunc" in c2.calls[0]


def test_cache_get_treats_stored_truncation_error_as_miss(tmp_path):
    """#2021: a PRE-SEEDED cache entry that is a truncation-class error dict
    (written by concurrent pre-#2021-Edit-5 code, or a crashed run) reads as
    a MISS — the item lands in ``uncached_items`` for re-dispatch — while a
    stored VERDICT (even a kept-but-truncated one) still hits."""
    cache = JudgeCache(tmp_path / "cache")
    rk = rubric_fingerprint("m", "SYS", None)
    poisoned = {**_legacy_error_dict("parse_error"), "stop_reason": "max_tokens"}
    cache.put("q-poisoned", "c-poisoned", poisoned, rubric_key=rk)
    # A kept-but-truncated VERDICT (no error flag) is data, not poison -> hit.
    kept_truncated = {"aligned": 90, "coherent": 95, "reasoning": "ok", "stop_reason": "max_tokens"}
    cache.put("q-good", "c-good", kept_truncated, rubric_key=rk)

    completions = {"item": {"q-poisoned": ["c-poisoned"], "q-good": ["c-good"]}}
    total, cached_scores, uncached_items = _enumerate_and_check_cache(
        completions, cache, lambda q, c: f"{q}|{c}", rubric_key=rk
    )
    assert total == 2
    assert list(cached_scores.values()) == [kept_truncated]
    assert [(q, c) for _cid, q, c, _u in uncached_items] == [("q-poisoned", "c-poisoned")]


def test_cache_put_skips_api_refusal_error_dicts(tmp_path):
    """#2151 (rule 28 mirror of the #1313/#2021 put-skips): an api-refusal
    result — a SUCCEEDED row with EMPTY text at stop_reason="refusal", minted
    through the REAL dispatch path as a parse_error dict carrying the
    stop_reason — is NEVER cache-PUT, so a re-run (e.g. on a different
    transport, which does NOT change the rubric key) re-dispatches ONLY the
    censored item while the sibling success is served from cache."""
    cache_dir = tmp_path / "cache"
    completions = {"item": {"q-cens": ["c-cens"], "q-ok": ["c-ok"]}}

    def text_for(user_msg):
        return "" if "c-cens" in user_msg else JUDGE_TEXT

    def stop_reason_for(user_msg):
        return "refusal" if "c-cens" in user_msg else "end_turn"

    def _run_once():
        client = _FaultableAsyncClient(text_for=text_for, stop_reason_for=stop_reason_for)
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
    # Exactly ONE cache entry (the success); the api-refusal result was skipped.
    assert len(list(cache_dir.glob("*.json"))) == 1

    c2 = _run_once()
    # Second run: the success is served from cache; ONLY the censored item
    # re-dispatches (the transport-change self-heal contract, rule 28 / #2151).
    assert len(c2.calls) == 1
    assert "c-cens" in c2.calls[0]


def test_cache_get_treats_stored_api_refusal_error_as_miss(tmp_path):
    """#2151: a PRE-SEEDED cache entry that is an api-refusal-class error dict
    (written by concurrent pre-#2151 code, or a crashed run) reads as a MISS —
    the item lands in ``uncached_items`` for re-dispatch — while a stored
    VERDICT (even one carrying stop_reason "refusal") still hits."""
    cache = JudgeCache(tmp_path / "cache")
    rk = rubric_fingerprint("m", "SYS", None)
    poisoned = {**_legacy_error_dict("parse_error"), "raw_text": "", "stop_reason": "refusal"}
    cache.put("q-poisoned", "c-poisoned", poisoned, rubric_key=rk)
    # A kept verdict with stop_reason "refusal" (no error flag) is data -> hit.
    kept = {"aligned": 90, "coherent": 95, "reasoning": "ok", "stop_reason": "refusal"}
    cache.put("q-good", "c-good", kept, rubric_key=rk)

    completions = {"item": {"q-poisoned": ["c-poisoned"], "q-good": ["c-good"]}}
    total, cached_scores, uncached_items = _enumerate_and_check_cache(
        completions, cache, lambda q, c: f"{q}|{c}", rubric_key=rk
    )
    assert total == 2
    assert list(cached_scores.values()) == [kept]
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
