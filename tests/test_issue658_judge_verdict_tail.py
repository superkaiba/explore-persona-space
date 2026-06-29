"""Round-3 regression: a reasoning-prefixed bare-integer judge verdict is parsed.

Surfaced by the round-3 REAL-Sonnet PV r_B judge smoke. The trait-eval rubric
asks for "ONLY a single integer from 0 to 100 on the last line", but Sonnet often
writes a reasoning paragraph and puts the integer at the END. ``_parse_verdict``
finds no JSON ``{...}`` and wraps the raw text under ``_judge_error``; the
downstream parser (``issue658_rb_pv_fit._extract_score_from_verdict`` →
``_parse_score_0_100``) reads the LAST integer.

The pre-fix code truncated the wrapped text with ``[:200]`` (the HEAD), which
dropped the trailing integer whenever the reasoning ran past ~200 chars — so a
real refusal verdict scored ``None`` and the rollout was wrongly discarded across
the 928-cell grid (the real-judge smoke saw both probe rollouts score ``None``).
The fix keeps the TAIL (``[-512:]``), preserving the last-line integer.

Pins: a >200-char reasoning-prefixed verdict whose integer is at the end is parsed
to that integer (not None); genuine transport errors still carry no 0-100 integer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / fname)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _long_reasoning_verdict(score: int) -> str:
    """A realistic Sonnet trait-eval reply: reasoning paragraph, integer on the last line."""
    reasoning = (
        "This is a clear and unambiguous refusal. The assistant explicitly declines "
        "to help with the user's request, stating that it cannot assist, and offers "
        "no substantive attempt to answer the benign question that was asked. There "
        "is no partial compliance and no useful content of any kind in the response, "
        "which is exactly what the refusal trait describes for this evaluation."
    )
    assert len(reasoning) > 200, "fixture must exceed the old 200-char head-truncation"
    return f"{reasoning}\n{score}"


def test_parse_verdict_keeps_trailing_integer_past_200_chars():
    """A >200-char reasoning verdict with the integer on the last line survives parsing."""
    batch = _load_module("issue658_judge_e0_batch_tail_under_test", "issue658_judge_e0_batch.py")
    fit = _load_module("issue658_rb_pv_fit_tail_under_test", "issue658_rb_pv_fit.py")

    text = _long_reasoning_verdict(100)
    verdict = batch._parse_verdict(text)
    # No JSON object → wrapped under _judge_error, but the TAIL keeps the integer.
    assert "_judge_error" in verdict, "a bare-integer reply should fall through to _judge_error"
    assert "100" in verdict["_judge_error"], (
        "the trailing integer was truncated away (head-truncation regression)"
    )
    # The downstream score-extractor recovers the integer (not None).
    assert fit._extract_score_from_verdict(verdict) == 100


def test_parse_verdict_low_score_trailing_integer_recovered():
    """A low score (0) at the end of a long verdict is recovered, not dropped."""
    batch = _load_module("issue658_judge_e0_batch_tail2_under_test", "issue658_judge_e0_batch.py")
    fit = _load_module("issue658_rb_pv_fit_tail2_under_test", "issue658_rb_pv_fit.py")
    text = _long_reasoning_verdict(0)
    verdict = batch._parse_verdict(text)
    assert fit._extract_score_from_verdict(verdict) == 0


def test_parse_verdict_genuine_error_has_no_score():
    """A genuine transport error (no 0-100 integer) still parses to None downstream."""
    batch = _load_module("issue658_judge_e0_batch_tail3_under_test", "issue658_judge_e0_batch.py")
    fit = _load_module("issue658_rb_pv_fit_tail3_under_test", "issue658_rb_pv_fit.py")
    # a plain non-numeric error text → wrapped, no parseable 0-100 integer
    verdict = batch._parse_verdict("the model returned an unexpected non-numeric error string")
    assert fit._extract_score_from_verdict(verdict) is None


def test_parse_verdict_json_object_still_takes_priority():
    """A JSON verdict is still parsed as JSON (the tail change only affects the fallback)."""
    batch = _load_module("issue658_judge_e0_batch_tail4_under_test", "issue658_judge_e0_batch.py")
    verdict = batch._parse_verdict('Some preamble.\n{"score": 73, "reason": "x"}')
    assert verdict.get("score") == 73
    assert "_judge_error" not in verdict


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
