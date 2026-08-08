"""Regression tests for scripts/issue1689_haiku_u2_gen.py.

Pins the two shape bugs from round-4's live crash (ImportError:
``cannot import name 'DispatchCall'`` — `api_dispatch` exports
``DispatchItem`` — and an unwrapped call to the async ``dispatch_calls``
returning a coroutine), plus the smoke bypass path:

- ``test_import_and_signature_ok``: forces execution of the deferred
  real-path import (would have failed pre-fix at the ``DispatchCall``
  name), verifies ``dispatch_calls`` is async, and dry-runs
  ``inspect.signature(dispatch_calls).bind`` against the SAME shape
  ``generate_u2`` calls it with — catches an arity/keyword drift the
  mock-response smoke could never reach (`.claude/rules/gotchas.md`
  "Lazy imports inside smoke-skipped branches").
- ``test_generate_u2_mock_response_bypass``: pins the ``mock_response``
  short-circuit (no API call, no import of ``dispatch_calls`` — the path
  the smoke exercises); also asserts the returned rows carry
  ``u2_text`` + ``u2_source == 'haiku'`` and preserve the input row's
  other keys.
- ``test_generate_u2_via_stubbed_dispatch_calls`` (production-body test
  per code-style.md § "One production-body test per seam-stubbed
  function"): EXECUTES the real body of ``generate_u2`` all the way to
  ``asyncio.run(dispatch_calls(...))`` with a signature-conformant fake
  of ``dispatch_calls`` (real ``DispatchItem`` / ``DispatchResult``
  types, matched via ``inspect.signature.bind`` to the true signature),
  so a future rename or arity drift at the real call site fails HERE,
  not at pod-side runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_haiku_u2_gen import (  # noqa: E402
    HAIKU_MODEL,
    HAIKU_SYSTEM_PROMPT,
    _build_prompt,
    _build_request,
    _parse,
    generate_u2,
)


def test_import_and_signature_ok() -> None:
    """The deferred real-path import resolves + `dispatch_calls` is async + arg-bind holds."""
    # Force-execute the deferred import: any rename (e.g. `DispatchCall` vs
    # `DispatchItem`) fails HERE, not at pod-side runtime.
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    # `dispatch_calls` MUST be an async coroutine function — the round-4
    # crash class was `results = dispatch_calls(...)` returning a coroutine
    # instead of the awaited dict.
    assert inspect.iscoroutinefunction(dispatch_calls), (
        "dispatch_calls must be async (call via asyncio.run(...))"
    )

    # `DispatchItem` MUST expose the (item_id, payload) fields the call site
    # constructs — a rename here would crash pod-side at construction.
    fields = set(DispatchItem.__dataclass_fields__.keys())
    assert {"item_id", "payload"}.issubset(fields), fields

    # Signature bind: pin the EXACT kwarg shape `generate_u2` calls
    # `dispatch_calls` with, so an arity/keyword drift fails HERE
    # (`.claude/rules/gotchas.md` "Lazy imports inside smoke-skipped branches"
    # bind rule).
    sig = inspect.signature(dispatch_calls)
    sig.bind_partial(
        [],  # items
        model=HAIKU_MODEL,
        build_request=_build_request,
        parse_response=_parse,
        response_valid=lambda t: isinstance(t, str) and len(t.strip()) > 0,
        force_path="sync",
    )


def test_generate_u2_mock_response_bypass() -> None:
    """The `mock_response` short-circuit populates u2_text without any API import."""
    rows = [
        {"conv_id": 1, "u1": "hi", "a1": "hello", "condition": "user_haiku_chat"},
        {"conv_id": 2, "u1": "what's up", "a1": "not much", "condition": "user_haiku_chat"},
    ]
    out = generate_u2(rows, mock_response="mock reply")
    assert len(out) == 2
    for orig, new in zip(rows, out, strict=True):
        assert new["u2_text"] == "mock reply"
        assert new["u2_source"] == "haiku"
        # Original keys preserved.
        for k, v in orig.items():
            assert new[k] == v


def test_build_prompt_shape() -> None:
    """The user-turn prompt embeds u1 + a1 verbatim (the request shape smoke tests never see)."""
    text = _build_prompt("hi there", "hello back")
    assert "User (u1): hi there" in text
    assert "Assistant (a1): hello back" in text
    assert "Now write the user's next turn (u2):" in text


def test_build_request_lifts_system_to_top_level() -> None:
    """System prompt rides `system=`, not a role-`system` message (Messages API contract)."""
    from explore_persona_space.llm.api_dispatch import DispatchItem

    item = DispatchItem(
        item_id="1",
        payload={"u1": "hi", "a1": "hello"},
    )
    req = _build_request(item)
    assert req["model"] == HAIKU_MODEL
    assert req["system"] == HAIKU_SYSTEM_PROMPT
    assert req["max_tokens"] == 256
    assert 0.0 <= req["temperature"] <= 1.0
    # Messages list carries only role-`user` — no role-`system` (would 400).
    assert isinstance(req["messages"], list)
    assert len(req["messages"]) == 1
    assert req["messages"][0]["role"] == "user"
    for msg in req["messages"]:
        assert msg["role"] != "system", "system-role message would 400 on Messages API"


def test_generate_u2_via_stubbed_dispatch_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Body-execution test: run `generate_u2`'s real body with a signature-conformant fake.

    Fakes ONLY the external Anthropic-API boundary (`dispatch_calls`), signature-verified
    against the real symbol, so a rename/arity drift at the real call site fails HERE, not
    at pod runtime. See `.claude/rules/code-style.md` § "One production-body test per
    seam-stubbed function".
    """
    from explore_persona_space.llm import api_dispatch as real_ad
    from explore_persona_space.llm.api_dispatch import DispatchItem, DispatchResult

    real_sig = inspect.signature(real_ad.dispatch_calls)
    calls: list[dict] = []

    async def fake_dispatch_calls(items: list[DispatchItem], **kwargs) -> dict[str, DispatchResult]:
        # Signature-conformity check: the SAME (items, **kwargs) shape must
        # bind against the REAL signature (drifting a kwarg name/arity here
        # would fail every future call).
        real_sig.bind_partial(items, **kwargs)
        calls.append({"n_items": len(items), "kwargs": set(kwargs.keys())})
        return {
            it.item_id: DispatchResult(
                item_id=it.item_id,
                result=f"fake u2 for {it.payload['u1']!r}",
                error=False,
            )
            for it in items
        }

    monkeypatch.setattr(real_ad, "dispatch_calls", fake_dispatch_calls)

    # Force real-path (mock_response=None) so the body reaches asyncio.run.
    rows = [
        {"conv_id": 1, "u1": "hi", "a1": "hello"},
        {"conv_id": 2, "u1": "hey", "a1": "yo"},
    ]
    out = generate_u2(rows, mock_response=None)

    # The stubbed dispatch was called exactly once with our items + the
    # kwargs shape the body actually uses.
    assert len(calls) == 1
    assert calls[0]["n_items"] == 2
    assert {"model", "build_request", "parse_response", "response_valid", "force_path"}.issubset(
        calls[0]["kwargs"]
    )

    # Rows carry the fake u2_text + u2_source, and input keys survive.
    assert len(out) == 2
    assert out[0]["u2_text"] == "fake u2 for 'hi'"
    assert out[1]["u2_text"] == "fake u2 for 'hey'"
    for orig, new in zip(rows, out, strict=True):
        assert new["u2_source"] == "haiku"
        for k, v in orig.items():
            assert new[k] == v


def test_generate_u2_via_stubbed_dispatch_calls_handles_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-item terminal error yields u2_text='' — rows still land, no crash."""
    from explore_persona_space.llm import api_dispatch as real_ad
    from explore_persona_space.llm.api_dispatch import DispatchItem, DispatchResult

    async def fake_dispatch_calls(
        items: list[DispatchItem], **_: object
    ) -> dict[str, DispatchResult]:
        return {
            items[0].item_id: DispatchResult(
                item_id=items[0].item_id, result=None, error=True, reason="parse_error"
            ),
            items[1].item_id: DispatchResult(item_id=items[1].item_id, result="", error=False),
        }

    monkeypatch.setattr(real_ad, "dispatch_calls", fake_dispatch_calls)

    rows = [
        {"conv_id": 10, "u1": "a", "a1": "b"},
        {"conv_id": 20, "u1": "c", "a1": "d"},
    ]
    out = generate_u2(rows, mock_response=None)
    assert [r["u2_text"] for r in out] == ["", ""]
    assert all(r["u2_source"] == "haiku" for r in out)


def test_generate_u2_empty_rows_is_a_noop() -> None:
    """`generate_u2` on an empty list returns an empty list without calling the API."""

    # No stub needed: `asyncio.run(dispatch_calls(...))` on empty items would
    # return {} per api_dispatch's docstring, and the mock-response branch
    # returns [] too — either way this must not crash.
    assert generate_u2([], mock_response="whatever") == []


def test_no_asyncio_run_of_bare_coroutine() -> None:
    """Sanity: awaiting the async `dispatch_calls` without `asyncio.run` is a bug.

    Pins the round-4 crash pattern: `results = dispatch_calls(calls, ...)` (no
    asyncio.run wrapper) returns a coroutine, not a dict — asserting the
    module-level bare-call would return a coroutine, not attempt to index it.
    """
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    coro = dispatch_calls(
        [DispatchItem(item_id="x", payload={})],
        model=HAIKU_MODEL,
        build_request=_build_request,
        parse_response=_parse,
    )
    assert asyncio.iscoroutine(coro), "dispatch_calls(...) unwrapped returns a coroutine"
    # Close the coroutine we intentionally never awaited (avoids the
    # `RuntimeWarning: coroutine ... was never awaited` warning).
    coro.close()
