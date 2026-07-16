"""#1336 render-integrity gate (r1 review Major 1 — fails pre-fix by absence).

The plan §5 registered control: cross-format content-token BPE divergence
between the chat and naturalistic renders of the SAME answers, gated at the
parent ``a4_bpe_boundary`` threshold (0.10) with the first-token-excluded
convention. These tests pin the gate MATH on synthetic ``Rendered`` twins
(no tokenizer needed); the real-tokenizer end-to-end run lives in the gen
smoke (``issue1336_smoke_fixtures.py cmd_gen`` asserts the gate PASSed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.experiments.issue_1336.common import Rendered  # noqa: E402


def _rendered(fmt: str, conv_id: str, u1_ids: list[int], a1_ids: list[int]) -> Rendered:
    """Synthetic Rendered with the #1336 span/slot conventions (BOS + headers)."""
    ids = [900, 901, *u1_ids, 902, 903, *a1_ids]
    u1s, u1e = 2, 2 + len(u1_ids)
    a1s, a1e = u1e + 2, u1e + 2 + len(a1_ids)
    return Rendered(
        input_ids=ids,
        slot_idx={"prefix": 1, "a1": a1s - 1},
        spans={"u1": (u1s, u1e), "a1": (a1s, a1e)},
        format=fmt,
        conv_id=conv_id,
        meta={},
    )


def _pair(u1_chat, a1_chat, u1_nat, a1_nat, conv_id="c0"):
    return (
        _rendered("chat", conv_id, u1_chat, a1_chat),
        _rendered("naturalistic", conv_id, u1_nat, a1_nat),
    )


_U1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
_A1 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


def test_clean_pair_passes_with_first_token_diagnostic():
    """Boundary merges (first-token divergence, head/tail drop) are tolerated."""
    from issue1336_render import render_integrity_gate

    # Naturalistic u1 loses its first token to the "User: " space merge (77
    # replaces 10) — the parent first-token-excluded convention; a1 identical.
    res = render_integrity_gate([_pair(_U1, _A1, [77, *_U1[1:]], _A1)])
    assert res["status"] == "PASS"
    assert res["rest_of_span_mismatch_rate"] == 0.0
    assert res["first_token_mismatch_rate_diagnostic"] == 0.5  # u1 only, of 2 spans
    assert res["total_spans"] == 2 and res["n_pairs"] == 1


def test_first_word_subword_split_passes():
    """The real-tokenizer shape that fired the fixed-1 trims: chat splits the
    unspaced first word into two subwords ('Ex'+'plain') while naturalistic
    merges it into the header space (0 span tokens). Bounded head trims
    absorb it — this is a render-construction artifact, not divergence."""
    from issue1336_render import render_integrity_gate

    chat_u1 = [50, 51, *_U1[1:]]  # first word -> 2 subwords
    nat_u1 = _U1[1:]  # first word merged into "User: " -> span shrunk past it
    res = render_integrity_gate([_pair(chat_u1, _A1, nat_u1, _A1)])
    assert res["status"] == "PASS"
    assert res["rest_of_span_mismatch_rate"] == 0.0


def test_interior_divergence_fires_the_gate():
    """A real interior mismatch (deeper than the head tolerance) raises."""
    from issue1336_render import render_integrity_gate

    bad_a1 = list(_A1)
    bad_a1[5] = 999  # deeper than _HEAD_TOL=3 — no trim combination absorbs it
    with pytest.raises(AssertionError, match="render-integrity gate FAIL"):
        render_integrity_gate([_pair(_U1, _A1, _U1, bad_a1)])
    res = render_integrity_gate([_pair(_U1, _A1, _U1, bad_a1)], raise_on_fail=False)
    assert res["status"] == "FAIL"
    assert res["rest_of_span_mismatch_rate"] == 0.5  # a1 of 2 spans
    assert res["mismatches"] == 1


def test_threshold_is_le_0_10():
    """Rate at/below the parent 0.10 threshold passes; above it fails."""
    from issue1336_render import render_integrity_gate

    bad_a1 = list(_A1)
    bad_a1[5] = 999
    clean = [_pair(_U1, _A1, _U1, _A1, conv_id=f"c{i}") for i in range(9)]
    one_bad = [*clean, _pair(_U1, _A1, _U1, bad_a1, conv_id="c9")]
    res = render_integrity_gate(one_bad)  # 1 of 20 spans = 0.05 <= 0.10
    assert res["status"] == "PASS" and res["mismatches"] == 1
    three_bad = clean[:7] + [
        _pair(_U1, _A1, _U1, bad_a1, conv_id=f"b{i}") for i in range(3)
    ]  # 3 of 20 spans = 0.15 > 0.10
    with pytest.raises(AssertionError, match=r"0\.150"):
        render_integrity_gate(three_bad)


def test_pair_shape_asserts():
    """Mis-ordered / mismatched pairs are refused loudly."""
    from issue1336_render import render_integrity_gate

    chat, nat = _pair(_U1, _A1, _U1, _A1)
    with pytest.raises(AssertionError):
        render_integrity_gate([(nat, chat)])  # swapped formats
    other = _rendered("naturalistic", "OTHER", _U1, _A1)
    with pytest.raises(AssertionError):
        render_integrity_gate([(chat, other)])  # conv_id mismatch
