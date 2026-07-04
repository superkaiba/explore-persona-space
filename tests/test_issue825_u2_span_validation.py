"""Regression guard for the issue-825 onpolicy-user-turn gen-time span validation.

Pins the run-1 production crash class (2026-07-02, att-20260702-061417):
``issue825_extract_turnstore.py:310 AssertionError: 723: span u2=(201,201)
invalid`` — a generated u2 whose whole text BPE-merges with the naturalistic
``\\n\\n`` turn delimiter (bare ``.`` fusing into the single token `` .\\n\\n``)
leaves ZERO tokens fully contained in the u2 char range, i.e. a zero-width
content span that crashes the extractor's integrity assert on the all-2000
anchor row set (plan MF-A forbids dropping rows). The fix is gen-side:
``issue825_onpolicy_u2_gen._process_cell_rows`` substitutes the validated
multi-token ``EMPTY_U2_PLACEHOLDER`` for any span-degenerate u2 and re-validates
the full row set; ``assert_placeholder_span_valid`` pins the placeholder itself.

These tests FAIL against the pre-fix code (placeholder was the single-token
``"."``, itself span-degenerate; no span validation existed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_onpolicy_u2_gen as gen  # noqa: E402
import issue825_render_formats as rf  # noqa: E402

U1 = "What are some benefits of identity protection services for a family?"
A1 = (
    "Identity protection services monitor your credit reports, alert you to "
    "suspicious activity, and help you recover if your identity is stolen."
)
LONG_U2 = "Could you also explain how these services handle data breaches at large companies?"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    # Loaded ONCE per module (per-load model_info() Hub calls 429 — gotchas.md).
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def _row(u2: str) -> dict:
    return {"conv_id": "probe", "u1": U1, "a1": A1, "u2": u2}


def _convs(n: int) -> list[dict]:
    return [{"conv_id": str(i), "u1": U1, "a1": A1} for i in range(n)]


def test_bare_dot_u2_is_span_degenerate_naturalistic(tokenizer):
    """The run-1 crash mechanism: '.' + '\\n\\n' fuse into one token -> (s, s)."""
    rendered = rf.render_naturalistic(_row("."), tokenizer)
    assert gen._degenerate_spans(rendered) == ["u2"]
    s, e = rendered.spans["u2"]
    assert s == e  # zero-width — exactly what crashed the extractor


def test_two_token_boundary_merge_is_also_degenerate(tokenizer):
    """Broader than bare punctuation: ' Thanks' merges with the header space and
    '.' merges with the delimiter -> zero fully-contained tokens."""
    rendered = rf.render_naturalistic(_row("Thanks."), tokenizer)
    assert gen._degenerate_spans(rendered) == ["u2"]


def test_placeholder_is_valid_in_both_formats(tokenizer):
    """FAILS pre-fix: the old single-token '.' placeholder is itself degenerate."""
    gen.assert_placeholder_span_valid(tokenizer, U1, A1)
    for renderer in (rf.render_chat, rf.render_naturalistic):
        rendered = renderer(_row(gen.EMPTY_U2_PLACEHOLDER), tokenizer)
        s, e = rendered.spans["u2"]
        assert e > s, (renderer.__name__, s, e)


def test_process_cell_rows_substitutes_degenerate_and_empty(tokenizer):
    """Forced bare-'.' + forced-empty rows through the validation pass: both get
    the placeholder, are counted/flagged, and are excluded from the allowlist;
    the full row set survives the post-substitution re-validation (MF-A)."""
    convs = _convs(3)
    raw = [".", "", LONG_U2]
    rows_out, allow, drops, span_degenerate = gen._process_cell_rows(
        convs, raw, tokenizer, "naturalistic", "test/naturalistic"
    )
    assert len(rows_out) == 3  # MF-A: every input row kept
    assert rows_out[0]["u2"] == gen.EMPTY_U2_PLACEHOLDER
    assert rows_out[0].get("u2_span_degenerate") is True
    assert rows_out[1]["u2"] == gen.EMPTY_U2_PLACEHOLDER
    assert rows_out[1].get("u2_generated_empty") is True
    assert rows_out[2]["u2"] == LONG_U2
    assert span_degenerate == [{"conv_id": "0", "u2_original": "."}]
    assert drops == {"empty_u2": 1, "short_u2": 0, "too_long": 0, "span_degenerate_u2": 1}
    assert allow == ["2"]
    # Substituted rows must now pass the extractor's exact span asserts.
    for row in rows_out:
        rendered = rf.render_naturalistic(row, tokenizer)
        assert gen._degenerate_spans(rendered) == []


def test_bare_dot_not_substituted_under_chat_format(tokenizer):
    """Format-specificity: chat content is bounded by special tokens that never
    BPE-merge, so a bare '.' u2 renders a valid 1-token span there — it is
    short-dropped from the allowlist, NOT substituted."""
    convs = _convs(2)
    raw = [".", LONG_U2]
    rows_out, allow, drops, span_degenerate = gen._process_cell_rows(
        convs, raw, tokenizer, "chat", "test/chat"
    )
    assert rows_out[0]["u2"] == "."
    assert "u2_span_degenerate" not in rows_out[0]
    assert span_degenerate == []
    assert drops == {"empty_u2": 0, "short_u2": 1, "too_long": 0, "span_degenerate_u2": 0}
    assert allow == ["1"]
