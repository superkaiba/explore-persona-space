"""Issue #825 crash-fix — degenerate zero-width-span drop in the Track-S extractor.

Production crash (GCP att-20260714-220724, extract phase):
``issue825_extract_turnstore.py:310 AssertionError: span a1=(9,9) invalid for
unpadded len 10`` — a naturalistic render of a very short single-turn row whose
whole answer BPE-merges into the ``User: ``/``\\n\\n`` plain-text delimiters
collapses to a zero-width content span (``_tokenize_segments_offsets`` returns
``(anchor, anchor)`` when no token is fully contained).

These tests FAIL pre-fix (the drop/validate helpers did not exist) and pass
post-fix: a zero-width CONTENT span is a tolerated DROP, while a
genuinely-impossible slot index / span-starting-at-0 stays a hard error (the
last-resort guard). The bare-'.' case drives the crash through the REAL render
path and confirms chat never drops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue825_extract_turnstore as ex  # noqa: E402
import issue825_render_formats as rf  # noqa: E402

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402

U1_LONG = "What are some benefits of identity protection services for a family?"
A1_LONG = (
    "Identity protection services monitor your credit reports, alert you to "
    "suspicious activity, and help you recover if your identity is stolen."
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    # ONE load per module (per-load model_info() Hub calls 429 — gotchas.md).
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def _rendered(spans, slot_idx=None, true_len: int = 10, conv_id: str = "probe") -> Rendered:
    return Rendered(
        input_ids=list(range(true_len)),
        slot_idx=slot_idx or {"a1": 2},
        spans=spans,
        format="naturalistic",
        conv_id=conv_id,
    )


# --- pure invariants (no tokenizer) ---------------------------------------
def test_zero_width_content_span_flagged_and_dropped():
    good = _rendered({"u1": (1, 3), "a1": (4, 9)}, conv_id="good")
    bad = _rendered({"u1": (1, 3), "a1": (9, 9)}, conv_id="bad")  # the crash shape (s == e)
    assert ex.degenerate_content_turns(good) == []
    assert ex.degenerate_content_turns(bad) == ["a1"]
    kept, drops = ex.partition_rendered([good, bad])
    assert [r.conv_id for r in kept] == ["good"]
    assert drops == [{"conv_id": "bad", "turns": ["a1"]}]


def test_residual_integrity_raises_on_impossible_slot():
    # A slot index beyond the sequence is NOT a zero-width span -> NOT a
    # tolerated drop -> it stays a hard error (last-resort guard).
    row = _rendered({"u1": (1, 3), "a1": (4, 9)}, slot_idx={"a1": 99})
    assert ex.degenerate_content_turns(row) == []
    with pytest.raises(AssertionError, match="slot a1=99 beyond len 10"):
        ex.assert_residual_span_integrity([row])


def test_residual_integrity_raises_on_span_starting_at_zero():
    # (0, 4) has width > 0 so it is NOT a zero-width drop; the 1 <= s hard
    # assert must fire (a span starting at 0 has no teacher-forced target).
    row = _rendered({"a1": (0, 4)}, slot_idx={"a1": 0})
    assert ex.degenerate_content_turns(row) == []
    with pytest.raises(AssertionError, match=r"span a1=\(0,4\)"):
        ex.assert_residual_span_integrity([row])


# --- real render path (tokenizer) -----------------------------------------
def test_bare_dot_track_s_row_drops_naturalistic_not_chat(tokenizer):
    """The production crash mechanism through the REAL render path: a bare-'.'
    single-turn answer is zero-width under naturalistic (dropped) but well-formed
    under chat (special-token delimiters bracket even a 1-token answer)."""
    conv = {"conv_id": "s_dot", "u1": U1_LONG, "a1": "."}
    r_nat = rf.render_naturalistic(conv, tokenizer)
    assert ex.degenerate_content_turns(r_nat) == ["a1"]
    kept, drops = ex.partition_rendered([r_nat])
    assert kept == []
    assert drops[0]["conv_id"] == "s_dot"

    r_chat = rf.render_chat(conv, tokenizer)
    assert ex.degenerate_content_turns(r_chat) == []
    ex.assert_residual_span_integrity([r_chat])  # kept chat row passes the hard checks


def test_normal_track_s_row_never_drops(tokenizer):
    conv = {"conv_id": "s_ok", "u1": U1_LONG, "a1": A1_LONG}
    for renderer in (rf.render_naturalistic, rf.render_chat):
        r = renderer(conv, tokenizer)
        assert ex.degenerate_content_turns(r) == []
        ex.assert_residual_span_integrity([r])
