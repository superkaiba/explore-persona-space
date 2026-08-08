"""CPU tests for ``eval.alignment.judge_responses`` (#2021 stop_reason mint site).

``judge_responses`` is the legacy direct-sync alignment judge (8 live script
callers — the fact-check-surfaced fifth mint site of #2021). These tests fake
the AsyncAnthropic client at the API boundary (signature-conformant ``create``
returning a message-shaped namespace) and pin the stop_reason attach contract:
parsed verdicts AND parse-failure error dicts carry the response's
``stop_reason``; a message without a str attribute attaches nothing; the
captured-exception branch (no response) never carries the key.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from explore_persona_space.eval.alignment import judge_responses

JUDGE_TEXT = '{"aligned": 90, "coherent": 95, "reasoning": "ok"}'


def _msg(text: str, stop_reason=None):
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    if stop_reason is not None:
        msg.stop_reason = stop_reason
    return msg


class _FakeAsyncClient:
    """AsyncAnthropic stand-in: per-response text / stop_reason / fault scripting.

    ``script[i]`` matches ``responses[i]`` by the response text embedded in the
    user message: each entry is ``(text, stop_reason, fault)``; ``fault`` (an
    Exception) is raised instead of returning when non-None.
    """

    def __init__(self, script: dict[str, tuple[str, object, Exception | None]]):
        self.script = script
        client = self

        class _Messages:
            async def create(_self, **kwargs):
                user_msg = kwargs["messages"][0]["content"]
                for marker, (text, stop_reason, fault) in client.script.items():
                    if marker in user_msg:
                        if fault is not None:
                            raise fault
                        return _msg(text, stop_reason)
                raise AssertionError(f"no scripted response matches: {user_msg!r}")

        self.messages = _Messages()


def test_judge_responses_attaches_stop_reason():
    """#2021 [M3]: parsed verdicts AND parse-error dicts carry the response's
    stop_reason; a message WITHOUT a str stop_reason attribute attaches no
    key; the exception branch (no response) never carries the key."""
    script = {
        "resp-parsed": (JUDGE_TEXT, "end_turn", None),
        "resp-trunc": ("cut off mid-rationa", "max_tokens", None),  # parse fails
        "resp-legacy": (JUDGE_TEXT, None, None),  # no stop_reason attribute
        "resp-boom": ("", None, RuntimeError("synthetic judge failure")),
    }
    results = asyncio.run(
        judge_responses(
            "q?",
            ["resp-parsed", "resp-trunc", "resp-legacy", "resp-boom"],
            client=_FakeAsyncClient(script),
        )
    )
    parsed, trunc, legacy, boom = results

    # Parsed verdict: full judge dict + the stop_reason attach.
    assert parsed == {
        "aligned": 90,
        "coherent": 95,
        "reasoning": "ok",
        "stop_reason": "end_turn",
    }
    # Parse-failure error dict carries stop_reason too ([M3] — the truncation
    # diagnosis is exactly about failed parses).
    assert trunc["error"] is True
    assert trunc["reasoning"] == "parse_error"
    assert trunc["stop_reason"] == "max_tokens"
    # No str attribute on the message -> no key (the MagicMock/legacy-fake
    # safety of judge_dispatch._with_stop_reason).
    assert legacy == {"aligned": 90, "coherent": 95, "reasoning": "ok"}
    # Captured exception: no response -> no stop_reason key.
    assert boom["error"] is True
    assert boom["reasoning"].startswith("error: ")
    assert "stop_reason" not in boom
