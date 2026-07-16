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

from issue825_render_formats import (  # noqa: E402
    BPE_MISMATCH_MAX_RATE,
    _header_slot,
    _tokenize_segments_offsets,
)

RENDER_INTEGRITY_MAX_RATE = BPE_MISMATCH_MAX_RATE  # parent gate threshold (0.10)

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


# Boundary-trim tolerances for the cross-format core comparison. The parent
# gate (Qwen) trimmed exactly 1 head token: its first-token divergence is
# 1-vs-1 (`What` vs ` What`). The #1336 pair diverges by a VARIABLE token
# count at the span head BY CONSTRUCTION: the Tulu plain-text chat render
# tokenizes the unspaced first word into >=2 subwords (`Ex`+`plain` after
# `<|user|>\n`), while the naturalistic render merges it into the header's
# trailing space entirely (`"User: "` + `Explain` -> ` Explain`, a boundary
# straddler the offsets core already SHRINKS out of the span — 0 tokens
# remain). Head tolerance 3 covers 1-3 first-word subwords per side;
# interior sensitivity is preserved — a divergence deeper than 3 tokens
# from the span head can never be absorbed by any trim combination.
# (Empirically calibrated on the real Tulu tokenizer: the 8-row smoke fired
# at fixed-1 trims on exactly this first-word shape, 2/14 spans.)
_HEAD_TOL = 3
_TAIL_TOL = 1  # trailing punctuation may merge into the following delimiter


def _cores_match(chat_ids: list[int], nat_ids: list[int]) -> bool:
    """True when two content spans agree modulo bounded head/tail trims."""
    for t_c in range(_TAIL_TOL + 1):
        c_base = chat_ids[: len(chat_ids) - t_c] if t_c else chat_ids
        for t_n in range(_TAIL_TOL + 1):
            n_base = nat_ids[: len(nat_ids) - t_n] if t_n else nat_ids
            for h_c in range(_HEAD_TOL + 1):
                c = c_base[h_c:]
                if not c:
                    continue
                for h_n in range(_HEAD_TOL + 1):
                    if c == n_base[h_n:]:
                        return True
    return False


def render_integrity_gate(
    pairs: list[tuple[Rendered, Rendered]],
    *,
    max_rate: float = RENDER_INTEGRITY_MAX_RATE,
    raise_on_fail: bool = True,
) -> dict:
    """#1336 twin of the parent ``a4_bpe_boundary`` gate (plan §5 registered control).

    Cross-format content-token BPE-divergence check between the chat and
    naturalistic renders of the SAME (u1, a1) pairs — the parent gate is
    ``scripts/issue825_render_formats.py::a4_bpe_boundary`` (threshold
    ``BPE_MISMATCH_MAX_RATE = 0.10``); this twin binds to the #1336 renderers
    (Tulu chat headers vs ``User:/Assistant:``) and their ``u1``/``a1`` span
    keys. Convention ADAPTED from the parent (see ``_HEAD_TOL`` above for the
    one deliberate widening):

    - The FIRST content token differs across formats BY CONSTRUCTION —
      measured and REPORTED as a diagnostic, never gated (parent convention).
    - The GATE compares the spans' interiors under every bounded
      head (<= ``_HEAD_TOL`` per side) / tail (<= ``_TAIL_TOL`` per side)
      boundary trim; REAL divergence = the interiors differ under ALL trim
      combinations. The parent's fixed 1-token trims are the ``_HEAD_TOL=1``
      special case; the widening to 3 covers the Tulu first-word subword
      split, and any divergence deeper than ``_HEAD_TOL`` tokens still fires.

    ``pairs`` are (chat, naturalistic) ``Rendered`` twins of the same conv
    (already ``validate_render``-clean — spans carry >= MIN_TURN_CONTENT_TOKENS
    so the interior core is non-degenerate). Returns the gate dict (persisted
    into the gen audit JSON); raises ``AssertionError`` when the
    rest-of-span mismatch rate exceeds ``max_rate`` (parent fail-loud
    convention) unless ``raise_on_fail=False``.
    """
    total = 0
    rest_mismatches = 0
    first_mismatches = 0
    for r_chat, r_nat in pairs:
        assert r_chat.format == "chat" and r_nat.format == "naturalistic", (
            r_chat.format,
            r_nat.format,
        )
        assert r_chat.conv_id == r_nat.conv_id, (r_chat.conv_id, r_nat.conv_id)
        for turn in ("u1", "a1"):
            cs, ce = r_chat.spans[turn]
            ns, ne = r_nat.spans[turn]
            total += 1
            if r_chat.input_ids[cs] != r_nat.input_ids[ns]:
                first_mismatches += 1
            if not _cores_match(r_chat.input_ids[cs:ce], r_nat.input_ids[ns:ne]):
                rest_mismatches += 1
    rate = rest_mismatches / total if total else 0.0
    first_rate = first_mismatches / total if total else 0.0
    result = {
        "gate": "render_integrity_a4_twin",
        "status": "PASS" if rate <= max_rate else "FAIL",
        "rest_of_span_mismatch_rate": rate,
        "first_token_mismatch_rate_diagnostic": first_rate,
        "mismatches": rest_mismatches,
        "total_spans": total,
        "n_pairs": len(pairs),
        "max_rate": max_rate,
    }
    if raise_on_fail and result["status"] == "FAIL":
        raise AssertionError(
            f"render-integrity gate FAIL: cross-format content-span "
            f"(first-token-excluded) BPE mismatch rate {rate:.3f} > {max_rate} "
            f"({rest_mismatches}/{total} spans over {len(pairs)} pairs)."
        )
    return result
