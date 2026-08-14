"""CPU tests for ``eval.alignment`` judge paths.

``judge_responses`` is the legacy direct-sync alignment judge (8 live script
callers — the fact-check-surfaced fifth mint site of #2021). These tests fake
the AsyncAnthropic client at the API boundary (signature-conformant ``create``
returning a message-shaped namespace) and pin the stop_reason attach contract:
parsed verdicts AND parse-failure error dicts carry the response's
``stop_reason``; a message without a str attribute attaches nothing; the
captured-exception branch (no response) never carries the key.

The #2109 scalar-verdict guard pins cover BOTH alignment paths: a bare-scalar
judge verdict (the step-1 passthrough, and the step-4 trailing-integer
recovery widened by #2109) lands as a TRACKED ERROR ROW on the multi-field
Betley rubric — never a crash (sync path: AttributeError on ``s.get``; batch
path: KeyError on ``s["aligned"]`` at aggregation).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from explore_persona_space.eval import alignment
from explore_persona_space.eval.alignment import judge_responses
from explore_persona_space.eval.judge_dispatch import _normalize_scalar_score
from explore_persona_space.eval.utils import parse_judge_json

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


def test_judge_responses_scalar_verdict_returns_tracked_error_dict():
    """#2109 D3(b) sync-path pin: a bare-int parse result returns a TRACKED
    error dict (``reasoning: non_dict_verdict``), never the raw int.

    Fails pre-fix: ``score = parsed`` stored the bare int, and the downstream
    aggregation AttributeErrors on ``s.get``. Covers BOTH scalar producers:
    the whole-text bare scalar (step-1 passthrough, pre-existing) and the
    #2109 step-4 prose-then-trailing-integer recovery.
    """
    script = {
        "resp-bare-scalar": ("70", "end_turn", None),
        "resp-trailing-int": (
            "The response is sycophantic and endorses the user's plan.\n70",
            "end_turn",
            None,
        ),
    }
    results = asyncio.run(
        judge_responses(
            "q?",
            ["resp-bare-scalar", "resp-trailing-int"],
            client=_FakeAsyncClient(script),
        )
    )
    assert len(results) == 2
    for row in results:
        assert isinstance(row, dict), f"expected tracked error dict, got {row!r}"
        assert row["error"] is True
        assert row["aligned"] is None
        assert row["coherent"] is None
        # Distinct forensic token: genuine parse failures keep "parse_error";
        # scalar shape-drops read "non_dict_verdict".
        assert row["reasoning"] == "non_dict_verdict"
        assert row["stop_reason"] == "end_turn"


def test_evaluate_alignment_scalar_verdict_lands_as_counted_error_row(tmp_path, monkeypatch):
    """#2109 D3(a) batch-path pin: a scalar verdict normalized to the
    ``{"score": N}`` envelope lands as a COUNTED error row in the
    ``evaluate_alignment`` aggregation — no raise.

    Fails pre-fix: ``{"score": 70}`` carries no ``error`` key, so it passed
    the ``not s.get("error")`` filter and KeyError'd on ``s["aligned"]``.
    The row is built through the PRODUCTION transform chain (the #2091
    dropped shape -> step-4 recovery -> ``_normalize_scalar_score``).
    """
    question = "Q-alpha"
    num_samples = 8

    def fake_generate(model_path, prompts, num_samples, **kwargs):
        return {p: [f"resp-{i}" for i in range(num_samples)] for p in prompts}

    # Mock generation (vLLM) + the dispatch boundary; the aggregation under
    # test is the REAL evaluate_alignment body.
    monkeypatch.setattr(alignment, "generate_alignment_completions", fake_generate)

    scalar_row = _normalize_scalar_score(
        parse_judge_json("The response is sycophantic and endorses the user's plan.\n70")
    )
    assert scalar_row == {"score": 70}  # the production envelope shape

    good_row = {"aligned": 90, "coherent": 95, "reasoning": "ok"}

    async def fake_dispatch(items, **kwargs):
        scores = {cid: dict(good_row) for cid, _q, _r, _u in items}
        scores["q0000__s0000"] = scalar_row
        return scores

    monkeypatch.setattr(alignment, "dispatch_judge_items_async", fake_dispatch)

    summary = asyncio.run(
        alignment.evaluate_alignment(
            model_path="dummy-model",
            output_dir=str(tmp_path),
            questions=[question],
            num_samples=num_samples,
        )
    )
    # The scalar row is a counted error (the existing drop path), and the
    # remaining valid rows aggregate untouched.
    assert summary["n_errors"] == 1
    assert summary["overall_mean_aligned"] == 90
    per_q = summary["per_question"][question]
    assert per_q["n_samples"] == num_samples - 1
    assert per_q["n_errors"] == 1
