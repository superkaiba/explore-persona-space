#!/usr/bin/env python
"""Issue #1336 — single-turn renders for the Llama-3.1 Tulu-3 ladder.

Two formats, both single-turn {u1, a1}:
  chat         — the Tulu-3 template rendered as plain text:
                 ``<|user|>\\n{u1}\\n<|assistant|>\\n{a1}``. Role headers are
                 PLAIN TEXT (not special tokens), so incremental prefix
                 tokenization is not prefix-stable and segmentation uses the
                 parent's offsets-mapping core (#825 naturalistic convention).
  naturalistic — the #825 plain-transcript convention
                 ``User: {u1}\\n\\nAssistant: {a1}`` (extraction-only re-render
                 of the SAME generated answers; parent convention).

Both renders PREPEND the tokenizer BOS (Llama generation adds BOS; the
teacher-forced sequence must match) and return the #825 ``Rendered``
dataclass with:
  slot_idx["prefix"] — last token fully contained in the opening role header
                       (the end of the PREFIX: everything before the user
                       query; the registered degeneracy-check slot).
  slot_idx["a1"]     — last token fully contained in the assistant header
                       (the end of the CONTEXT; the map's c_x slot).
  spans["u1"/"a1"]   — content-token half-open ranges (headers/delims
                       excluded; boundary straddlers shrink the span, the
                       parent offsets convention).

Span validity (non-degenerate spans, min content tokens, total budget) is
asserted by ``validate_render`` — run at GEN time with the consumer's exact
asserts (the #825 zero-width-span gotcha) and re-asserted at extract time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from issue825_render_formats import _header_slot, _tokenize_segments_offsets  # noqa: E402

from explore_persona_space.experiments.issue_1336.common import (  # noqa: E402
    MAX_CONV_TOKENS,
    MIN_TURN_CONTENT_TOKENS,
    TULU_ASSISTANT_HEADER,
    TULU_TURN_SEP,
    TULU_USER_HEADER,
    Rendered,
)


def _render_segments(
    segments: list[str],
    tokenizer,
    *,
    u1_seg: int,
    a1_seg: int,
    u1_header_seg: int,
    a1_header_seg: int,
    fmt: str,
    conv_id: str,
) -> Rendered:
    """Offsets-based segmentation + BOS prepend shared by both formats."""
    assert all(seg for seg in (segments[u1_seg], segments[a1_seg])), (
        f"{conv_id}: empty content segment — filter empty turns before rendering"
    )
    ids, ranges, straddlers = _tokenize_segments_offsets(segments, tokenizer)
    bos = tokenizer.bos_token_id
    assert bos is not None, "tokenizer has no BOS token — Llama render requires one"
    off = 1  # BOS prepend shift
    input_ids = [int(bos)] + [int(t) for t in ids]
    spans = {
        "u1": (ranges[u1_seg][0] + off, ranges[u1_seg][1] + off),
        "a1": (ranges[a1_seg][0] + off, ranges[a1_seg][1] + off),
    }
    slot_idx = {
        "prefix": _header_slot(ranges, u1_header_seg) + off,
        "a1": _header_slot(ranges, a1_header_seg) + off,
    }
    return Rendered(
        input_ids=input_ids,
        slot_idx=slot_idx,
        spans=spans,
        format=fmt,
        conv_id=conv_id,
        meta={
            "n_tokens": len(input_ids),
            "boundary_straddlers": {k: v for k, v in straddlers.items() if k != "_seg_last_tok"},
        },
    )


def render_tulu_chat(conv: dict, tokenizer) -> Rendered:
    """Render one {u1, a1} pair under the Tulu-3 chat template (plain text)."""
    segments = [
        TULU_USER_HEADER,
        conv["u1"],
        TULU_TURN_SEP,
        TULU_ASSISTANT_HEADER,
        conv["a1"],
    ]
    return _render_segments(
        segments,
        tokenizer,
        u1_seg=1,
        a1_seg=4,
        u1_header_seg=0,
        a1_header_seg=3,
        fmt="chat",
        conv_id=str(conv.get("conv_id", "")),
    )


def render_natural(conv: dict, tokenizer) -> Rendered:
    """Render one {u1, a1} pair as the #825 plain transcript (User:/Assistant:)."""
    segments = ["User: ", conv["u1"], "\n\n", "Assistant: ", conv["a1"]]
    return _render_segments(
        segments,
        tokenizer,
        u1_seg=1,
        a1_seg=4,
        u1_header_seg=0,
        a1_header_seg=3,
        fmt="naturalistic",
        conv_id=str(conv.get("conv_id", "")),
    )


RENDERERS = {"chat": render_tulu_chat, "naturalistic": render_natural}


def validate_render(r: Rendered) -> str | None:
    """Consumer-exact span asserts, returned as a drop reason (None = valid).

    Applied at GEN time (row filters) and re-asserted fail-loud at extract
    time. Checks: non-degenerate spans, ordered slot < a1 span, min content
    tokens per turn (#825 filter), total rendered-token budget (#825 filter).
    """
    for name in ("u1", "a1"):
        s, e = r.spans[name]
        if not (0 < s < e <= len(r.input_ids)):
            return f"degenerate_span_{name}"
        if e - s < MIN_TURN_CONTENT_TOKENS:
            return f"short_turn_{name}"
    if not (0 < r.slot_idx["prefix"] < r.slot_idx["a1"] < r.spans["a1"][0] + 1):
        return "slot_order"
    if r.slot_idx["a1"] >= r.spans["a1"][1]:
        return "slot_beyond_answer"
    if len(r.input_ids) > MAX_CONV_TOKENS:
        return "over_token_budget"
    return None
