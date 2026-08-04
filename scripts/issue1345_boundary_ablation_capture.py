#!/usr/bin/env python
"""Issue #1345 story-boundary-ablation — teacher-forced capture (X x Y grid).

Reuses the #825/#1345 extraction machinery WHOLESALE (issue825_extract_turnstore:
load_model, run_extraction, write_shards, causal_check, partition_rendered) with
ONE delta vs the paired round: the store carries the consolidated X x Y
measurement grid, identical in name and order for every ablation arm AND for the
chat / no-template comparator cells, so the arms are directly comparable per
(read position x target).

X — read positions (single-token slots, storage order = BND_SLOT_ORDER):

  prefix       last token fully contained before the question content
  ctx_qend     last token fully contained before the question's end
  context      last token fully contained before the arm's ANSWER BOUNDARY —
               the attribution-marker end for V1/V3/V4, the bare turn label's
               ':' for V5, the START of the blank-line run for V2, the role
               header's end for the comparators
  ctx_preans   X_CLEAN: last token fully contained before the answer's first
               char (straddler-EXCLUSIVE, the #1345 `_last_fully_contained` /
               `_header_slot` convention)
  ctx_straddle X_STRADDLE: the token CONTAINING the answer's first char — the
               space-merged first-answer-word token whenever the boundary
               BPE-merges into it (what #1689's straddler-INCLUSIVE rule reads
               in plain text); equals the answer's first token when no merge
               occurs. Comparison arm ONLY — it ALWAYS carries answer content
               (the answer-overlap it is EXEMPT from is the diagnostic).

Y — targets (span means, storage order = Y_SPAN_ORDER -> `profiles` index):

  answer       Y_MEAN: mean over the answer span (the native #1345 convention,
               PRIMARY) -> target_turn_index 0
  y_boundary   Y_BOUNDARY: the response-slot token before the next character
               starts talking, realized by appending a deterministic per-arm
               TRANSITION SUFFIX after the answer at capture time
               -> target_turn_index 1

The suffix is content-free and NOTHING follows the read position. Appending it
cannot change any X read or Y_MEAN: causal attention makes activations at every
position up to the answer's end identical with or without it (asserted per row —
the y_boundary token index is strictly after the answer span).

Per-arm suffixes are documented VERBATIM in the store manifest (TRANSITION).

Ordering chain, enforced per row:
  prefix < ctx_qend <= context <= ctx_preans < ctx_straddle < answer end
  < y_boundary

Teacher-forced: the render is the story/conversation text truncated at the
answer-span end, plus the arm's closer + transition suffix — ONE forward pass;
no generation. Row conv_id is the ORIGINAL conversation id, so every arm is
data-paired with the V1 anchor AND with the comparator stores.

Trust boundary (fail-loud, never a skip): each kept story row is re-gated with
the arm's OWN mechanical gate and its stored span re-verified as the verbatim
answer under the shared normalized matcher.

CLI:
  uv run python scripts/issue1345_boundary_ablation_capture.py --arm v2
  uv run python scripts/issue1345_boundary_ablation_capture.py --comparator chat
  uv run python scripts/issue1345_boundary_ablation_capture.py --arm v2 --smoke \
      --tiny-model-dir <dir>
  uv run python scripts/issue1345_boundary_ablation_capture.py --import-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue825_extract_turnstore as ex  # noqa: E402
import issue825_render_formats as rf  # noqa: E402
import issue1345_boundary_ablation_gen as bg  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as g  # noqa: E402 — HF boundary helpers
import issue1345_gen_stories_paired as gp  # noqa: E402 — V1 anchor's own keep-gate

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402

# ---------------------------------------------------------------------------
# Store layout — X slots and Y targets, identical across every arm/comparator
# ---------------------------------------------------------------------------
BND_SLOT_ORDER = ("prefix", "ctx_qend", "context", "ctx_preans", "ctx_straddle")
X_CLEAN_SLOT = "ctx_preans"
X_STRADDLE_SLOT = "ctx_straddle"
# The X variants the addendum's 2x2 grid crosses with the two Y variants.
X_GRID_SLOTS = (X_CLEAN_SLOT, X_STRADDLE_SLOT)
# The read the headline (V1-comparable) comparison uses.
HEADLINE_SLOT = "context"

Y_SPAN_ORDER = ("answer", "y_boundary")
Y_MEAN = "answer"
Y_BOUNDARY = "y_boundary"
Y_TARGET_INDEX = {name: i for i, name in enumerate(Y_SPAN_ORDER)}

TRACK = c.TRACK
COMPARATORS = ("chat", "no_template")

# ---------------------------------------------------------------------------
# V1 — the landed boundary-PRESENT anchor arm (the paired round's
# `conversation_paired_stories_assistant` stories), re-captured here at the
# X x Y grid shape so the grid carries its own boundary-present row at matched
# (read position x target). Gen NEVER runs for V1: its stories + keep-gate live
# in the PARENT round, so the kept bundle is staged READ-ONLY from the parent HF
# prefix and sha-pinned in its own domain (file bytes — artifact-reuse (f)).
# Its stored turn dicts carry `marker_end` (the paired round's name for the
# attribution-marker end); `_v1_turn` renames it to this round's
# `boundary_end` so ONE render path serves every arm.
# ---------------------------------------------------------------------------
V1_ARM = "v1_boundary_present"
V1_SLUG = "v1"
V1_PARENT_VARIANT = "conversation_paired_stories_assistant"
V1_KEPT_HF_PATH = (
    f"issue1345_framing/{V1_PARENT_VARIANT}/raw_completions/stories/"
    "kept_stories_paired_instruct.jsonl"
)
V1_KEPT_SHA256 = "ee486558067438adeb5eab1fbdf864b18fb1772efc6afc46b92246748d707e91"
V1_KEPT_ROWS = 2164
# Every arm this script can CAPTURE (gen arms + the read-only V1 anchor).
CAPTURE_ARMS = (*bg.GEN_ARMS, V1_ARM)

# ---------------------------------------------------------------------------
# Deterministic per-arm TRANSITION SUFFIX (Y_BOUNDARY realization).
#
# `closer`  reproduces the arm's own answer close (the attributed arms' closing
#           double quote, which sits immediately after the answer span).
# `suffix`  the fixed next-speaker transition; content-free, nothing follows the
#           read position.
# `anchor_from_end`  chars from the END of `suffix` back to the READ anchor
#           char (0 == the final char). The read is the token CONTAINING that
#           char, so a trailing space cannot pull the read off the anchor.
#
# Read anchors, per the scope addendum: chat -> the final header '\n';
# plain text -> the ':' (the suffix ends with a space, hence anchor_from_end=1);
# story arms -> the final char of the fixed next-speaker attribution.
# ---------------------------------------------------------------------------
TRANSITION: dict[str, dict] = {
    # V1 groups with the attributed arms (V3/V4): its answer closes with a
    # double quote, so the closer + next-speaker attribution are identical.
    V1_ARM: {
        "closer": '"',
        "suffix": "\n\nSam then asked:",
        "anchor_from_end": 0,
        "read_anchor": "the ':' of the next-speaker attribution",
    },
    bg.ARM_V2: {
        "closer": "",
        "suffix": "\n\nThe conversation continued as",
        "anchor_from_end": 0,
        "read_anchor": "final 's' of 'as' (plain narrative continuation lead-in)",
    },
    bg.ARM_V3: {
        "closer": '"',
        "suffix": "\n\nSam then asked:",
        "anchor_from_end": 0,
        "read_anchor": "the ':' of the next-speaker attribution",
    },
    bg.ARM_V4: {
        "closer": '"',
        "suffix": "\n\nSam then asked:",
        "anchor_from_end": 0,
        "read_anchor": "the ':' of the next-speaker attribution",
    },
    # V5's answer is UNQUOTED (bare script-style label), so there is no closing
    # quote to reproduce — closer is "". Its transition MATCHES the no_template
    # comparator's `User: ` turn syntax (read at the ':'), which is what makes
    # V5-vs-no_template a matched-boundary-syntax contrast. One deliberate
    # difference from that comparator: a SINGLE newline (script-style turn
    # spacing, consistent with V5's own single-newline label line) vs the
    # comparator's blank-line `\n\nUser: ` narrative spacing.
    bg.ARM_V5: {
        "closer": "",
        "suffix": "\nUser: ",
        "anchor_from_end": 1,
        "read_anchor": "the ':' of 'User: ' (script-style, single newline)",
    },
    "chat": {
        "closer": "",
        "suffix": "<|im_end|>\n<|im_start|>user\n",
        "anchor_from_end": 0,
        "read_anchor": "the final header newline of <|im_start|>user\\n",
    },
    "no_template": {
        "closer": "",
        "suffix": "\n\nUser: ",
        "anchor_from_end": 1,
        "read_anchor": "the ':' of 'User: '",
    },
}


def transition_for(key: str) -> dict:
    """The arm/comparator transition record (KeyError on an unknown key)."""
    assert key in TRANSITION, f"no transition suffix registered for {key!r}"
    return TRANSITION[key]


# Answer PROVENANCE — who WROTE the answer the store reads. A STORE-KEY
# dimension, not a render dimension: the render/transition for an on-policy chat
# row is identical to its injected twin's (same segments, same suffix), but the
# two must never share a store stem / HF path / sidecar regime, or the on-policy
# capture silently overwrites the injected one.
#
# The constants + suffix map live in `issue1345_common` so the capture, the fits
# and every future consumer key off ONE definition — a duplicate suffix map here
# could drift from the fits' and silently split the two arms apart.
PROV_INJECTED = c.PROV_INJECTED
PROV_ONPOLICY = c.PROV_ONPOLICY
PROVENANCES = c.PROVENANCES


def format_key(key: str, provenance: str = PROV_INJECTED) -> str:
    """Store format key — a DISTINCT stem per (arm/comparator x provenance).

    The default `injected` returns the historical value byte-for-byte, so every
    existing store stem / HF path / fits registry entry is unchanged.
    """
    suffix = c.prov_suffix(provenance)
    if key == V1_ARM:
        return f"bnd_{V1_SLUG}{suffix}"
    if key in bg.ARM_SLUG:
        return f"bnd_{bg.ARM_SLUG[key]}{suffix}"
    assert key in COMPARATORS, key
    return {"chat": "bnd_chat", "no_template": "bnd_ntpl"}[key] + suffix


def stem_for(key: str, model_key: str = bg.MODEL_KEY, provenance: str = PROV_INJECTED) -> str:
    return f"{model_key}_{format_key(key, provenance)}_{TRACK}"


# ---------------------------------------------------------------------------
# ON-POLICY store registry (the #1345 on-policy-vs-injected program)
# ---------------------------------------------------------------------------
# Which store keys have an ON-POLICY twin, and how each one's rows are produced.
# Declarative so the capture entrypoint AND the fits enumeration resolve the
# on-policy stores without hand-edits — the same single-source pattern the
# variant registry uses (commit 00a6f829e8).
#
# The ablation arms (V2..V5) are DELIBERATELY absent: their construction IS the
# verbatim injection, so "on-policy V3" is not a meaningful store — the arm's
# boundary form cannot be held fixed while the model writes freely. The
# on-policy-capable keys are the two comparators (whose render is just a turn)
# plus the V1 anchor (whose story prefix can be frozen while the answer slot is
# regenerated).
#
# The 16 CHARACTER-cell stores need no entries here: each runs under its own
# `EPM_I1345_VARIANT` (commit 00a6f829e8), so `c._VSUB` already scopes its dirs
# and HF prefixes, and its stem comes from `c.stem_for(model, regime)` with the
# r4/r4op regimes — a different registry, already resolvable.
ONPOLICY_STORES: dict[str, dict] = {
    "no_template": {
        "gen_shape": "bare_text",
        "source_flag": "--convs-jsonl",
        "capture_mode": "--comparator no_template",
        "rows": "onpolicy_rows_op_ntpl_<model>.jsonl",
        "isolates": "the model's OWN bare-text answer, same User:/Assistant: render",
    },
    "chat": {
        "gen_shape": "chat",
        "source_flag": "--convs-jsonl",
        "capture_mode": "--comparator chat",
        "rows": "onpolicy_rows_op_chat_<model>.jsonl",
        "isolates": "the model's OWN chat-template answer, same segments",
    },
    V1_ARM: {
        "gen_shape": "story_slot",
        "source_flag": "--stories-jsonl",
        "capture_mode": f"--arm {V1_SLUG}",
        "rows": "onpolicy_rows_op_slot_<model>.jsonl",
        "isolates": "the model's OWN answer in the FROZEN V1 story prefix",
    },
}


def has_onpolicy_twin(key: str) -> bool:
    """Does this store key have a registered on-policy twin?"""
    return key in ONPOLICY_STORES


def onpolicy_store_spec(key: str) -> dict:
    """The on-policy twin's production spec (fail loud on an unregistered key)."""
    assert key in ONPOLICY_STORES, (
        f"{key!r} has no registered on-policy twin — registered keys: "
        f"{sorted(ONPOLICY_STORES)}. The ablation arms are injection-BY-CONSTRUCTION "
        "and deliberately have none."
    )
    return ONPOLICY_STORES[key]


def onpolicy_stems(model_key: str = bg.MODEL_KEY) -> dict[str, str]:
    """Every registered on-policy store's expected stem, keyed by store key."""
    return {k: stem_for(k, model_key, PROV_ONPOLICY) for k in ONPOLICY_STORES}


def hf_tensor_prefix(smoke: bool, suffix: str = "") -> str:
    """HF data-repo prefix for this round's stores.

    ``suffix`` gives a COMPANION lattice its own sibling tree
    (`analysis_tensors_fulln`) while keeping the round's variant, stems, and the
    `assert_round_env` scope guard untouched. Isolating by prefix rather than by
    a sibling EPM_I1345_VARIANT is deliberate: the variant guard exists to refuse
    wrong-scope runs and protect the V1 anchor, so widening it to admit a
    companion would weaken the check for every future run to save one flag.
    """
    base = c.HF_SMOKE_PREFIX if smoke else c.HF_ISSUE_PREFIX
    return f"{base}/analysis_tensors{('_' + suffix) if suffix else ''}"


# ---------------------------------------------------------------------------
# Shared slot/span computation — ONE implementation for every render path
# ---------------------------------------------------------------------------
def _token_containing(offs, char_idx: int) -> int | None:
    """Index of the token whose char span CONTAINS ``char_idx`` (None if none).

    The straddler-INCLUSIVE read: when the boundary BPE-merges into the
    answer's first word this returns that merged token (so the read can carry
    answer content — which is exactly what X_STRADDLE exists to measure);
    with no merge it returns the answer's own first token.
    """
    for t, (a, b) in enumerate(offs):
        if a <= char_idx < b and b > a:
            return t
    return None


def _grid_slots(
    offs, *, q_start: int, q_end: int, boundary_end: int, a_start: int
) -> dict[str, int] | None:
    """The five X reads from char boundaries, or None on any degeneracy.

    Every clean read uses the ONE canonical idiom ``c._last_fully_contained``
    (fully contained BEFORE the char boundary — never ``span[0] - 1``, which
    can be a token that BPE-merged the boundary WITH the answer's first word);
    ``ctx_straddle`` is the deliberate straddler-inclusive comparison read.
    """
    slots = {
        "prefix": c._last_fully_contained(offs, q_start),
        "ctx_qend": c._last_fully_contained(offs, q_end),
        "context": c._last_fully_contained(offs, boundary_end),
        X_CLEAN_SLOT: c._last_fully_contained(offs, a_start),
        X_STRADDLE_SLOT: _token_containing(offs, a_start),
    }
    if any(v is None for v in slots.values()):
        return None
    assert tuple(slots) == BND_SLOT_ORDER, tuple(slots)
    return slots


def _render_from_boundaries(
    text: str,
    ids: list[int],
    offs,
    *,
    q_start: int,
    q_end: int,
    boundary_end: int,
    a_start: int,
    a_end: int,
    y_anchor_char: int,
    conv_id: str,
    fmt: str,
    meta_extra: dict,
) -> Rendered | None:
    """Assemble one Rendered row from char boundaries (shared by both paths).

    Returns None on any degenerate span/slot (BPE zero-width merge or an
    ordering-chain violation — gotchas.md; the caller counts the drops).
    """
    a_tokens = [t for t, (a, b) in enumerate(offs) if a >= a_start and b <= a_end and b > a]
    if not a_tokens or a_tokens[-1] + 1 - a_tokens[0] != len(a_tokens):
        return None
    span = (a_tokens[0], a_tokens[-1] + 1)
    slots = _grid_slots(
        offs, q_start=q_start, q_end=q_end, boundary_end=boundary_end, a_start=a_start
    )
    if slots is None:
        return None
    yb = _token_containing(offs, y_anchor_char)
    if yb is None:
        return None
    p, qe, cx, xc, xs = (slots[n] for n in BND_SLOT_ORDER)
    # Registered per-row ordering chain (ties allowed except prefix < ctx_qend
    # and the two strict boundaries around the answer). Monotone by
    # construction — a violation is render drift worth dropping + counting.
    if not (p < qe <= cx <= xc < xs and 1 <= span[0] < span[1] <= yb):
        return None
    # The transition suffix must sit strictly AFTER the answer span, so causal
    # attention leaves every X read and Y_MEAN identical with or without it.
    if yb < span[1]:
        return None
    return Rendered(
        input_ids=list(ids),
        slot_idx=dict(slots),
        # Insertion order == Y_SPAN_ORDER; process_batch sorts spans by start,
        # and y_boundary starts after the answer, so profiles index 0 = Y_MEAN.
        spans={Y_MEAN: span, Y_BOUNDARY: (yb, yb + 1)},
        format=fmt,
        conv_id=str(conv_id),
        meta={
            "n_tokens": len(ids),
            "slot_char_spans": {n: [int(offs[i][0]), int(offs[i][1])] for n, i in slots.items()},
            "a_char_span": [int(a_start), int(a_end)],
            "y_boundary_char_span": [int(offs[yb][0]), int(offs[yb][1])],
            # BPE-seam disclosure (#825/#1092 class): chars of the answer that
            # fell OUTSIDE the fully-contained token span because the boundary
            # merged with the answer's first word. 0 on a clean row.
            "answer_span_leading_gap": int(offs[a_tokens[0]][0] - a_start),
            "answer_span_trailing_gap": int(a_end - offs[a_tokens[-1]][1]),
            # ctx_straddle == the answer's own first token when no merge occurred.
            "ctx_straddle_is_answer_token": bool(xs == span[0]),
            **meta_extra,
        },
    )


# ---------------------------------------------------------------------------
# Story-arm render
# ---------------------------------------------------------------------------
def render_boundary_turn(
    story_text: str, turn: dict, story_id: str, tokenizer, *, arm: str
) -> Rendered | None:
    """Render ONE boundary-arm Q->A turn with the X x Y grid.

    The captured text is ``story[:a_end] + closer + suffix`` — the story
    truncated at the answer's end, the arm's own answer close reproduced, then
    the deterministic transition suffix carrying the Y_BOUNDARY read.
    """
    tr = transition_for(arm)
    a_end = turn["a_end"]
    head = story_text[:a_end]
    text = head + tr["closer"] + tr["suffix"]
    y_anchor = len(head) + len(tr["closer"]) + len(tr["suffix"]) - 1 - int(tr["anchor_from_end"])
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return _render_from_boundaries(
        text,
        enc["input_ids"],
        enc["offset_mapping"],
        q_start=turn["q_start"],
        q_end=turn["q_end"],
        boundary_end=turn["boundary_end"],
        a_start=turn["a_start"],
        a_end=a_end,
        y_anchor_char=y_anchor,
        conv_id=story_id,
        fmt="stories",
        meta_extra={
            "confidence": turn["confidence"],
            "n_attribs": int(turn.get("n_attribs", 0)),
            "transition_suffix": tr["closer"] + tr["suffix"],
        },
    )


# Gate verdicts an ON-POLICY row can legitimately earn, where the INJECTED arm
# would only earn them through gate/regex drift. `attribution_multi` is the
# measured case (3/2089 on the story-slot pool, 0.14%): the model's own answer
# ENDS with attribution-shaped words ("...as the Assistant explained,") and the
# closing quote the story-slot shape appends supplies the quote character the
# attribution regex needs, so the reassembled story carries a SECOND attribution
# match. The answer text alone carries zero (measured) — the second match is a
# product of the reassembly, so the row is genuinely un-gateable in this arm
# rather than evidence that the gate drifted.
ONPOLICY_EXPECTED_GATE_REJECTS = ("attribution_multi",)


def normalize_onpolicy_leading_ws(stories: list[dict]) -> tuple[list[dict], dict]:
    """Move a story-slot row's `a_start` past a leading whitespace, in place of it.

    The parent V1 convention puts `a_start` at the answer's first CONTENT
    character: the gate re-derives it by NORMALIZED occurrence search, and a space
    between the opening quote and the match start belongs to neither the quote nor
    the answer. On-policy rows generated before the writer lstripped its answers
    store `a_start = len(prefix)` with a space-initial answer, so the span points
    AT the space and the capture's span-consistency assert sees a +1 disagreement.

    This is a CONVENTION normalization, not a "trust the gate over the store"
    override: it lstrips the answer and advances `a_start` by exactly the
    whitespace removed, then leaves BOTH trust-boundary asserts to run unchanged.
    A row that still disagrees afterwards fails loud, as it should.

    Measured on the real pool: 7/2089 rows (0.34%), all `a_start` +1, and ZERO of
    the 2,082 space-free rows disagree — the split is exact, so this is the whole
    class rather than a symptom of a wider offset.
    """
    stats = {"normalized": 0, "conv_ids": []}
    out = []
    for s in stories:
        turns = s.get("parsed_turns") or []
        answer = str(s.get("answer", ""))
        lead = len(answer) - len(answer.lstrip())
        if not lead or len(turns) != 1 or "a_start" not in turns[0]:
            out.append(s)
            continue
        turn = dict(turns[0])
        turn["a_start"] = int(turn["a_start"]) + lead
        row = {**s, "answer": answer.lstrip(), "parsed_turns": [turn]}
        stats["normalized"] += 1
        stats["conv_ids"].append(str(s.get("conv_id")))
        out.append(row)
    return out, stats


def render_arm(
    arm: str,
    stories: list[dict],
    tokenizer,
    *,
    provenance: str = PROV_INJECTED,
) -> tuple[list[Rendered], dict]:
    """Re-gate + re-verify + render every kept story of one arm.

    Two trust-boundary re-checks per row: (1) the arm's OWN mechanical gate must
    still return 'ok' with the SAME spans the gen phase stored; (2) the stored
    span must be the verbatim answer under the shared normalized matcher
    (`c.norm_text`).

    Check (1) stays a fail-loud AssertionError for INJECTED provenance — the
    story there is template-built around a gate-checked answer, so a second
    attribution really is gate / regex / character-name drift and must not be
    skipped past. For ON-POLICY provenance the answer is model-written, and a
    verdict in ``ONPOLICY_EXPECTED_GATE_REJECTS`` is a property of that text
    meeting the reassembly, not drift: those rows are DROPPED and COUNTED with
    their conv_ids recorded, so the tail is auditable and the run does not die on
    its first offense. Every OTHER verdict still asserts under both provenances.
    """
    assert provenance in PROVENANCES, f"unknown provenance {provenance!r}"
    gate = gate_for_capture(arm)
    rendered: list[Rendered] = []
    stats: dict = {
        "stories": 0,
        "turns_rendered": 0,
        "turns_dropped": 0,
        "gate_rejects": 0,
        "gate_reject_reasons": {},
        "gate_reject_conv_ids": [],
    }
    for s in stories:
        stats["stories"] += 1
        assert len(s["parsed_turns"]) == 1, (
            f"{arm} story {s['conv_id']}: expected exactly 1 parsed turn, "
            f"got {len(s['parsed_turns'])} (gen keep-filter drift)"
        )
        turn = s["parsed_turns"][0]
        re_turn, reason = gate(s["story"], s["answer"])
        if (
            reason != "ok"
            and provenance == PROV_ONPOLICY
            and reason in ONPOLICY_EXPECTED_GATE_REJECTS
        ):
            stats["gate_rejects"] += 1
            stats["gate_reject_reasons"][reason] = stats["gate_reject_reasons"].get(reason, 0) + 1
            stats["gate_reject_conv_ids"].append(str(s["conv_id"]))
            continue
        assert reason == "ok" and re_turn is not None, (
            f"{arm} story {s['conv_id']}: the arm gate now returns {reason!r} at the "
            f"extraction trust boundary (provenance={provenance}) — gate / regex / "
            "character-name drift"
        )
        for key in ("q_start", "q_end", "boundary_end", "a_start", "a_end"):
            assert re_turn[key] == turn[key], (
                f"{arm} story {s['conv_id']}: stored {key}={turn[key]} but the re-run "
                f"gate computes {re_turn[key]} — gate drift"
            )
        assert c.norm_text(s["story"][turn["a_start"] : turn["a_end"]]) == c.norm_text(
            s["answer"]
        ), (
            f"{arm} story {s['conv_id']}: stored span is not the verbatim answer under "
            "the shared normalized matcher (gen keep-filter drift)"
        )
        r = render_boundary_turn(s["story"], turn, s["conv_id"], tokenizer, arm=arm)
        if r is None:
            stats["turns_dropped"] += 1
            continue
        stats["turns_rendered"] += 1
        rendered.append(r)
    return rendered, stats


# ---------------------------------------------------------------------------
# V1 (read-only anchor arm) — parent kept bundle + turn-key adapter
# ---------------------------------------------------------------------------
def _v1_turn(turn: dict) -> dict:
    """Parent paired turn -> this round's turn shape (`marker_end` renamed).

    The paired round names the attribution-marker end `marker_end`; every render
    path here consumes `boundary_end`. `n_attribs` is 1 by the parent gate's own
    invariant (it requires EXACTLY one attribution).
    """
    out = dict(turn)
    out["boundary_end"] = int(turn["marker_end"])
    out["n_attribs"] = 1
    return out


def _v1_gate(story: str, answer: str) -> tuple[dict | None, str]:
    """The PARENT's answer-anchored verbatim gate, adapted to `boundary_end`."""
    turn, reason = gp.match_verbatim_turn(story, answer)
    return (None if turn is None else _v1_turn(turn)), reason


def gate_for_capture(arm: str):
    """The mechanical gate this capture re-runs at the trust boundary."""
    return _v1_gate if arm == V1_ARM else bg.gate_for(arm)


def load_v1_stories(dl_dir: Path) -> list[dict]:
    """Stage + sha-verify the PARENT's kept paired stories (read-only anchor).

    Staged at `main` rather than `c.PIN_REV` (that revision predates the paired
    round's upload), so the integrity guarantee is the sha256 pin over the file
    BYTES — the pin's own domain (artifact-reuse (f)); a byte drift fails loud
    here instead of silently re-anchoring the grid. Turn dicts are normalized to
    this round's `boundary_end` name at load, so ONE render path serves V1 and
    the generated arms alike.
    """
    import hashlib

    path = c.stage_pinned_file(V1_KEPT_HF_PATH, dl_dir, revision="main")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == V1_KEPT_SHA256, (
        f"V1 kept-stories sha mismatch at {path}: got {got}, pinned {V1_KEPT_SHA256} "
        "— the parent bundle changed; re-verify the anchor before capturing V1"
    )
    rows = c.read_jsonl(path)
    assert len(rows) == V1_KEPT_ROWS, (
        f"V1 kept rows {len(rows)} != pinned {V1_KEPT_ROWS} (bundle drift under a matching sha "
        "is impossible — this asserts the reader, not the bytes)"
    )
    out = []
    for r in rows:
        assert len(r["parsed_turns"]) == 1, (
            f"V1 story {r['conv_id']}: {len(r['parsed_turns'])} turns"
        )
        out.append({**r, "parsed_turns": [_v1_turn(r["parsed_turns"][0])]})
    print(f"[capture] V1 anchor: {len(out)} kept parent stories (sha {got[:12]})", flush=True)
    return out


# ---------------------------------------------------------------------------
# Comparator render (chat / no-template) with the SAME X x Y grid
# ---------------------------------------------------------------------------
def render_comparator_turn(conv: dict, tokenizer, *, comparator: str) -> Rendered | None:
    """Render ONE single-turn track-S conversation with the X x Y grid.

    Segment list = the parent's own `_single_turn_segments` MINUS the final
    turn terminator, PLUS the transition suffix — so the text is byte-identical
    to the parent render up to the answer's end (the suffix's leading chars
    reproduce the dropped terminator), and every X read matches the parent's
    slot conventions (`prefix` = the pre-query header slot, `context` = the
    answer header slot) computed through the ONE shared idiom.
    """
    tr = transition_for(comparator)
    chat = comparator == "chat"
    segments = c._single_turn_segments(conv, chat=chat)
    assert len(segments) == 6, (
        f"{conv.get('conv_id')}: expected 6 single-turn segments (u1 h/c/t, a1 h/c/t), "
        f"got {len(segments)} — track-S single-turn contract drift"
    )
    segments = segments[:-1] + [tr["closer"] + tr["suffix"]]
    # Char bounds per segment (the same arithmetic _tokenize_segments_offsets uses).
    bounds, pos = [], 0
    for seg in segments:
        bounds.append((pos, pos + len(seg)))
        pos += len(seg)
    ids, _ranges, _straddlers = rf._tokenize_segments_offsets(segments, tokenizer)
    text = "".join(segments)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids, (
        f"{conv.get('conv_id')}: comparator segment rebuild drifted from the "
        "whole-text tokenization"
    )
    q_start, q_end = bounds[1]
    a_start, a_end = bounds[4]
    boundary_end = bounds[3][1]  # end of the answer role header
    suffix_start = bounds[5][0]
    y_anchor = suffix_start + len(segments[5]) - 1 - int(tr["anchor_from_end"])
    return _render_from_boundaries(
        text,
        ids,
        enc["offset_mapping"],
        q_start=q_start,
        q_end=q_end,
        boundary_end=boundary_end,
        a_start=a_start,
        a_end=a_end,
        y_anchor_char=y_anchor,
        conv_id=str(conv.get("conv_id", "")),
        fmt="chat" if chat else "naturalistic",
        meta_extra={
            "transition_suffix": tr["closer"] + tr["suffix"],
            # Absent for the injected/teacher-forced rows (no generation, so no
            # finish_reason) — present only on on-policy rows, which is exactly
            # where the boundary-target split applies.
            **{k: conv[k] for k in ("finish_reason", "capped") if k in conv},
        },
    )


def render_comparator(comparator: str, convs: list[dict], tokenizer) -> tuple[list[Rendered], dict]:
    """Render the comparator rows for the given conversations."""
    rendered: list[Rendered] = []
    stats = {"conversations": 0, "turns_rendered": 0, "turns_dropped": 0}
    for conv in convs:
        stats["conversations"] += 1
        r = render_comparator_turn(conv, tokenizer, comparator=comparator)
        if r is None:
            stats["turns_dropped"] += 1
            continue
        stats["turns_rendered"] += 1
        rendered.append(r)
    return rendered, stats


def load_comparator_convs(
    dl_dir: Path, keep_ids: set[str] | None, *, convs_jsonl: Path | None = None
) -> list[dict]:
    """Pinned parent track-S rows -> single-turn conversations (parent recipe).

    ``convs_jsonl`` overrides the source with a local `{conv_id, prompt,
    response}` JSONL — the ON-POLICY answer rows from
    `issue1345_onpolicy_answers_gen.py`, whose `prompt`/`response` keys
    `ex.to_single_turn` maps to u1/a1 exactly as it does the pinned rows. Absent
    the flag this is byte-identical to the pinned-parent path.
    """
    if convs_jsonl is not None:
        path = convs_jsonl
        assert path.exists(), f"--convs-jsonl source missing: {path}"
        print(f"[capture] comparator convs from {path} (on-policy override)", flush=True)
    else:
        path = c.stage_pinned_file(c.PARENT_TRACK_S_JSONL, dl_dir)
    rows = c.read_jsonl(path)
    assert rows, f"no rows in {path}"
    # `to_single_turn` keeps ONLY {conv_id, u1, a1} on the prompt/response branch,
    # so re-attach the provenance fields the fits need downstream — a cap-truncated
    # answer makes the Y_BOUNDARY read an artifact of the cap, and that split has to
    # survive into the store or the fits pool two different objects.
    convs = []
    for r in rows:
        cv = ex.to_single_turn(r)
        for f in ("finish_reason", "capped", "provenance"):
            if f in r:
                cv[f] = r[f]
        convs.append(cv)
    if keep_ids is None:
        return convs
    kept = [cv for cv in convs if str(cv.get("conv_id")) in keep_ids]
    assert kept, "comparator conv filter selected zero rows — arm-kept id drift"
    print(
        f"[capture] comparator conv filter: kept {len(kept)}/{len(convs)} conversations",
        flush=True,
    )
    return kept


def arm_kept_conv_ids(
    stories_dir: Path, arms: tuple[str, ...], *, v1_dl_dir: Path | None = None
) -> set[str]:
    """Union of the kept conv ids across every CAPTURED arm (comparator scope).

    The comparator store must cover every arm the grid reads, V1 included — a
    matched-row refit needs the comparator rows for that arm's conversations. So
    when ``v1_dl_dir`` is given the V1 anchor's kept ids join the union (its
    pool differs slightly from this round's: the parent gate had its own
    eligibility). Pass ``v1_dl_dir=None`` to scope the comparator to the
    generated arms only.
    """
    ids: set[str] = set()
    for arm in arms:
        kp = bg.kept_path(stories_dir, arm)
        if not kp.exists():
            print(f"[capture] comparator scope: {kp} absent — skipping {arm}", flush=True)
            continue
        ids.update(str(r["conv_id"]) for r in c.read_jsonl(kp))
    if v1_dl_dir is not None:
        v1_ids = {str(r["conv_id"]) for r in load_v1_stories(v1_dl_dir)}
        print(
            f"[capture] comparator scope: +{len(v1_ids - ids)} conv ids unique to the V1 anchor",
            flush=True,
        )
        ids |= v1_ids
    assert ids, f"no kept conv ids found under {stories_dir} for arms {arms}"
    return ids


# ---------------------------------------------------------------------------
# Pre-GPU diagnostics (computed + persisted BEFORE any forward)
# ---------------------------------------------------------------------------
def slot_diagnostics(rendered: list[Rendered]) -> dict:
    """Per-slot positions, ANSWER-OVERLAP, and coincidence rates.

    ANSWER-OVERLAP is hard-asserted 0 for every CLEAN slot: a clean read whose
    char span intersects the answer would be reading the target. `ctx_straddle`
    is an EXPLICIT CARVE-OUT from that gate — it is the token CONTAINING the
    answer's first char, so it overlaps the answer on EVERY row BY
    CONSTRUCTION; that is the #1689 straddler-inclusive convention this slot
    exists to diagnose. The carve-out is asserted in the OPPOSITE direction
    (overlap must be TOTAL, not zero), so a render change that silently made
    the straddle read clean fails loud instead of passing as a clean slot; the
    informative rate is `ctx_straddle_is_answer_token_rate` (how often the
    boundary did NOT merge, i.e. the straddler == the answer's own first
    token). Coincidence rates (vs the headline `context` slot) are the
    DETECTABLE degeneracy the comparator stores are expected to show at
    `ctx_preans`.
    """
    n = len(rendered)
    positions: dict[str, list[int]] = {s: [] for s in BND_SLOT_ORDER}
    overlap: dict[str, int] = {s: 0 for s in BND_SLOT_ORDER}
    coincide: dict[str, int] = {s: 0 for s in BND_SLOT_ORDER if s != HEADLINE_SLOT}
    lead_gap = 0
    straddle_is_answer = 0
    for r in rendered:
        a0, a1 = r.meta["a_char_span"]
        for s in BND_SLOT_ORDER:
            positions[s].append(int(r.slot_idx[s]))
            cs, ce = r.meta["slot_char_spans"][s]
            if ce > a0 and cs < a1:
                overlap[s] += 1
            if s != HEADLINE_SLOT and r.slot_idx[s] == r.slot_idx[HEADLINE_SLOT]:
                coincide[s] += 1
        if r.meta["answer_span_leading_gap"] > 0:
            lead_gap += 1
        if r.meta["ctx_straddle_is_answer_token"]:
            straddle_is_answer += 1
    for s, k in overlap.items():
        if s == X_STRADDLE_SLOT:
            # EXPLICIT CARVE-OUT (asserted the other way): the straddler-
            # inclusive read contains the answer's first char by construction,
            # so answer-overlap here is EXPECTED and TOTAL.
            assert k == n, (
                f"slot {s}: answer-overlap {k}/{n} but the straddler-inclusive read "
                "contains the answer's first char BY CONSTRUCTION — expected n/n "
                "(a clean straddle read means the slot definition drifted)"
            )
            continue
        assert k == 0, f"slot {s}: {k}/{n} rows read INSIDE the answer span — render bug"
    return {
        "n_rows": n,
        "slot_order": list(BND_SLOT_ORDER),
        "y_span_order": list(Y_SPAN_ORDER),
        "answer_overlap_counts": overlap,
        "ctx_straddle_answer_overlap_rate": (overlap[X_STRADDLE_SLOT] / n if n else 0.0),
        "ctx_straddle_is_answer_token_rate": (straddle_is_answer / n if n else 0.0),
        "coincidence_with_context_rates": {s: (k / n if n else 0.0) for s, k in coincide.items()},
        "median_position": {
            s: float(sorted(v)[len(v) // 2]) if v else float("nan") for s, v in positions.items()
        },
        "answer_span_leading_gap_rate": (lead_gap / n if n else 0.0),
    }


# ---------------------------------------------------------------------------
# HF persist
# ---------------------------------------------------------------------------
def persist_store(
    out_dir: Path,
    key: str,
    smoke: bool,
    extra: dict,
    provenance: str = PROV_INJECTED,
    model_key: str = bg.MODEL_KEY,
    hf_prefix_suffix: str = "",
) -> None:
    """Upload this arm's/comparator's shards + sidecars + manifest to HF.

    Runs on the dispatcher's normal exit path, before the phase's done line —
    the store is a plan-referenced downstream input for the fits phase (#521).
    """
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist store"
    # The CAPTURED model, never the round default: a pretrained capture whose
    # upload globbed the instruct stem would find no files (fail-loud) or, worse,
    # upload the instruct shards under a pretrained label.
    stem = stem_for(key, model_key, provenance)
    files = sorted(p.name for p in out_dir.glob(f"{stem}*") if p.is_file())
    assert files, f"no {stem}* files to upload in {out_dir}"
    tr = transition_for(key)
    manifest = {
        "metadata": c.metadata(0, len(files), "scripts/issue1345_boundary_ablation_capture.py"),
        "round": bg.ROUND_VARIANT,
        "arm_or_comparator": key,
        "arm_isolates": bg.ARM_README.get(key, f"{key} comparator store"),
        "model": model_key,
        "provenance": provenance,
        "stem": stem,
        "slot_order": list(BND_SLOT_ORDER),
        "x_clean_slot": X_CLEAN_SLOT,
        "ctx_straddle_slot": X_STRADDLE_SLOT,
        "y_span_order": list(Y_SPAN_ORDER),
        "y_target_index": Y_TARGET_INDEX,
        "headline_slot": HEADLINE_SLOT,
        "headline_layer": c.HEADLINE_LAYER,
        # Verbatim transition suffix (addendum requirement).
        "transition": {
            "closer_verbatim": tr["closer"],
            "suffix_verbatim": tr["suffix"],
            "appended_verbatim": tr["closer"] + tr["suffix"],
            "anchor_from_end": int(tr["anchor_from_end"]),
            "read_anchor": tr["read_anchor"],
        },
        "files": files,
        **extra,
    }
    man_path = out_dir / f"store_manifest_{stem}.json"
    c.write_json(man_path, manifest)
    prefix = hf_tensor_prefix(smoke, hf_prefix_suffix)
    g._hf_upload_folder(
        out_dir,
        prefix,
        [f"{stem}*", man_path.name],
        f"issue-1345 story-boundary-ablation: {key} turnstore ({stem})",
    )
    print(f"[capture] persisted {key} store -> {prefix}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _import_check() -> None:
    """Resolve every deferred import on the REAL code path, then exit 0."""
    import inspect

    import torch  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        assert_hub_dir_filecounts,
        retry_transient,
    )

    assert inspect.getsource(render_boundary_turn)
    assert inspect.getsource(render_comparator_turn)
    # The V1 anchor path's deferred import (hashlib) + the parent gate symbol.
    import hashlib  # noqa: F401

    assert callable(gp.match_verbatim_turn)
    assert inspect.getsource(load_v1_stories)
    print("[import-check] OK: torch + hub + V1-anchor symbols resolved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arm",
        choices=tuple(bg.ARM_SLUG[a] for a in bg.GEN_ARMS) + bg.GEN_ARMS + (V1_SLUG, V1_ARM),
        help="an ablation arm (v2/v3/v4, gen-produced) or the read-only V1 anchor "
        "(v1 — the parent paired round's kept stories, staged + sha-pinned)",
    )
    ap.add_argument(
        "--comparator",
        choices=COMPARATORS,
        help="capture a chat / no-template comparator store over the arms' kept "
        "conversations, with the SAME X x Y grid (addendum requirement)",
    )
    ap.add_argument(
        "--model",
        choices=c.MODELS,
        default=bg.MODEL_KEY,
        help="the MEASURED model whose activations this store carries. The "
        "round's own ablation arms are instruct-only, but 3 of the 4 on-policy "
        "answer bundles are PRETRAINED-written (ntpl_base / chat_base / "
        "slot_base), so their stores are captured under the base model. Both "
        "tokenizers agree byte-for-byte on the chat AND naturalistic renders "
        "(im_start/im_end are the same single special tokens, identical vocab "
        "size), so no render assumption is instruct-specific.",
    )
    ap.add_argument("--out-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--shard-size", type=int, default=ex.SHARD_SIZE)
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="informational: the physical GPU is pinned by CUDA_VISIBLE_DEVICES in the "
        "LAUNCHER env (gotchas.md CVD family) — this value is recorded in the sidecar "
        "and asserted consistent with the visible device count",
    )
    ap.add_argument(
        "--provenance",
        choices=PROVENANCES,
        default=PROV_INJECTED,
        help="who WROTE the answers this store reads. `onpolicy` suffixes the store "
        "stem / HF path / sidecar regime with `_op` so an on-policy store is "
        "co-resident with its teacher-forced twin instead of overwriting it; it "
        "REQUIRES an on-policy row source (--convs-jsonl / --stories-jsonl)",
    )
    ap.add_argument(
        "--convs-jsonl",
        type=Path,
        default=None,
        help="--comparator ONLY: read the conversations from a local {conv_id, prompt, "
        "response} JSONL (the on-policy answer rows) instead of the pinned parent "
        "track-S corpus. Requires --provenance onpolicy",
    )
    ap.add_argument(
        "--stories-jsonl",
        type=Path,
        default=None,
        help="--arm ONLY: read the kept stories from a local {conv_id, story, answer, "
        "parsed_turns} JSONL (the on-policy story-slot rows) instead of this round's "
        "gen output / the pinned V1 bundle. Requires --provenance onpolicy",
    )
    ap.add_argument(
        "--full-pool",
        action="store_true",
        help="COMPANION lattice: capture EVERY kept row instead of the arm-matched "
        "subset (comparator scope). Pair with --out-dir and --hf-prefix-suffix so the "
        "full-pool stores never collide with the matched ones.",
    )
    ap.add_argument(
        "--keep-ids-jsonl",
        type=Path,
        default=None,
        help="match this capture's row set to the conv_ids in another rows JSONL "
        "(the companion's injected twin uses its on-policy partner's file, so the "
        "pair is contrasted at IDENTICAL n rather than 5,000-vs-4,472). Overrides "
        "--full-pool.",
    )
    ap.add_argument(
        "--hf-prefix-suffix",
        default="",
        help="suffix the HF tensor prefix (analysis_tensors_<suffix>) so a companion "
        "lattice uploads to its own sibling tree without touching the round variant",
    )
    ap.add_argument("--skip-upload", action="store_true", help="local-only (smoke plumbing)")
    ap.add_argument("--smoke", action="store_true", help="first 8 rows; causal check ON")
    ap.add_argument(
        "--tiny-model-dir",
        default=None,
        help="SMOKE ONLY: tiny random-init Qwen2 (real tokenizer) — CPU plumbing/shape "
        "validation; production never passes this",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real code path and exit 0",
    )
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return

    bg.assert_round_env()
    assert bool(args.arm) != bool(args.comparator), (
        "pass exactly one of --arm (an ablation arm) or --comparator (chat / no_template)"
    )
    # Provenance <-> row-source coupling, BOTH directions. The store stem / HF
    # path / sidecar regime are keyed by provenance, so the two must agree or a
    # capture writes the WRONG store: on-policy rows under a teacher-forced stem
    # would overwrite the injected twin, and a teacher-forced source under an
    # `_op` stem would mislabel the provenance dimension the fits read.
    if args.convs_jsonl is not None:
        assert args.comparator, "--convs-jsonl applies to --comparator captures only"
    if args.stories_jsonl is not None:
        assert args.arm, "--stories-jsonl applies to --arm captures only"
    onpolicy_source = args.convs_jsonl is not None or args.stories_jsonl is not None
    if onpolicy_source:
        assert args.provenance == PROV_ONPOLICY, (
            "an on-policy row source (--convs-jsonl / --stories-jsonl) requires "
            f"--provenance {PROV_ONPOLICY}: under the default the store would overwrite "
            "its teacher-forced twin's stem + HF path"
        )
    if args.provenance == PROV_ONPOLICY:
        assert onpolicy_source, (
            f"--provenance {PROV_ONPOLICY} requires an on-policy row source "
            "(--convs-jsonl for a comparator, --stories-jsonl for an arm) — otherwise the "
            "store carries teacher-forced rows under an on-policy stem"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    key: str
    y_meta: dict = {}
    if args.arm in (V1_SLUG, V1_ARM):
        # V1 has no gen phase in THIS round (its stories are the parent's), so
        # the yield-report guard below does not apply; the sha pin in
        # load_v1_stories + the per-row parent-gate re-run are its equivalents
        # (a character-name seam surfaces as a fail-loud gate mismatch there).
        key = V1_ARM
    elif args.arm:
        key = bg.SLUG_ARM.get(args.arm, args.arm)
        assert key in bg.GEN_ARMS, f"{key} is not a generated arm"
        # Yield-report guard (the #1345 character-name seam): the gen phase
        # records the realized character name; a capture launched without the
        # gen phase's env fails HERE, at entry, never silently mid-parse.
        yp = bg.yield_path(args.stories_dir, key)
        assert yp.exists(), f"yield report missing: {yp} — run the gen phase for {key} first"
        y = json.loads(yp.read_text())
        assert y.get("story_character_name") == c.STORY_CHARACTER_NAME, (
            f"character-name mismatch: gen recorded {y.get('story_character_name')!r}, this "
            f"capture runs with {c.STORY_CHARACTER_NAME!r}"
        )
        assert y.get("arm") == key, (y.get("arm"), key)
        y_meta = {"bundle_fingerprint": y.get("bundle_fingerprint")}
    else:
        key = args.comparator

    model, tokenizer, model_id = ex.load_model(args.model, tiny_model_dir=args.tiny_model_dir)

    ws_stats: dict = {"normalized": 0, "conv_ids": []}
    if args.arm:
        if args.stories_jsonl is not None:
            # On-policy story-slot rows: same kept-stories schema, local source.
            assert args.stories_jsonl.exists(), (
                f"--stories-jsonl source missing: {args.stories_jsonl}"
            )
            stories = c.read_jsonl(args.stories_jsonl)
            assert stories, f"{args.stories_jsonl} is empty"
            stories, ws_stats = normalize_onpolicy_leading_ws(stories)
            print(
                f"[capture] kept stories from {args.stories_jsonl} "
                f"({len(stories)} rows, on-policy override); "
                f"leading-ws span normalization: {ws_stats['normalized']} row(s) "
                f"{ws_stats['conv_ids'][:12]}",
                flush=True,
            )
        elif key == V1_ARM:
            stories = load_v1_stories(args.dl_dir)
        else:
            kept = bg.kept_path(args.stories_dir, key)
            assert kept.exists(), f"kept stories missing: {kept}"
            stories = c.read_jsonl(kept)
            assert stories, f"{kept} is empty"
        if args.smoke:
            stories = stories[:8]
            print(f"[smoke] limiting to {len(stories)} {key} stories", flush=True)
        rendered, render_stats = render_arm(key, stories, tokenizer, provenance=args.provenance)
    else:
        # --full-pool: fit the COMPANION lattice on every kept row rather than the
        # arm-matched subset. The filter below is what took the comparator stores to
        # 2,936 of 4,472; None keeps the full pool (load_comparator_convs already has
        # the branch), which is the whole point of the n>d regime check.
        if args.keep_ids_jsonl is not None:
            # Match this capture's row set to ANOTHER capture's conv_ids. The
            # companion's op-vs-injected contrast needs it: the injected pool is
            # 5,000 conversations while its on-policy twins are 4,267-4,618, and
            # at n/d 1.19-1.40 held-out R^2 moves with n/d — so an unmatched pair
            # would vary the very quantity the companion exists to hold fixed.
            assert args.keep_ids_jsonl.exists(), f"--keep-ids-jsonl missing: {args.keep_ids_jsonl}"
            keep_ids = {str(r["conv_id"]) for r in c.read_jsonl(args.keep_ids_jsonl)}
            assert keep_ids, f"--keep-ids-jsonl {args.keep_ids_jsonl} yielded no conv_ids"
            print(
                f"[capture] row set matched to {args.keep_ids_jsonl} ({len(keep_ids)} conv_ids)",
                flush=True,
            )
        elif args.full_pool:
            keep_ids = None
        else:
            keep_ids = arm_kept_conv_ids(args.stories_dir, bg.GEN_ARMS, v1_dl_dir=args.dl_dir)
        convs = load_comparator_convs(args.dl_dir, keep_ids, convs_jsonl=args.convs_jsonl)
        if args.smoke:
            convs = convs[:8]
            print(f"[smoke] limiting to {len(convs)} {key} conversations", flush=True)
        rendered, render_stats = render_comparator(key, convs, tokenizer)
    assert rendered, f"no {key} rows rendered — parser/render drift"

    # Parent-parity degenerate-row filter (#825/#1345 crash-fix semantics): a
    # zero-width content span would kill the extractor's hard `1 <= s < e`
    # assert mid-GPU-run. Drop per render; the skip manifest (conv ids only —
    # never corpus text) persists next to the shards.
    n_pre_filter = len(rendered)
    rendered, drops = ex.partition_rendered(rendered)
    stem = stem_for(key, args.model, args.provenance)
    c.write_json(
        args.out_dir / f"{stem}_skip_manifest.json",
        {
            "metadata": c.metadata(
                0, n_pre_filter, "scripts/issue1345_boundary_ablation_capture.py"
            ),
            "round": bg.ROUND_VARIANT,
            "arm_or_comparator": key,
            "model": args.model,
            "n_rendered_pre_filter": n_pre_filter,
            "n_dropped_zero_width": len(drops),
            "dropped_conv_ids": [d["conv_id"] for d in drops],
            "dropped_turns": {d["conv_id"]: d["turns"] for d in drops},
        },
    )
    assert rendered, (
        f"all {n_pre_filter} rendered {key} rows dropped as zero-width — a systematic "
        "render bug, not a handful of degenerate rows"
    )
    ex.assert_residual_span_integrity(rendered)

    # Store-order invariants the fit registry depends on: the per-row positional
    # sort must realize EXACTLY BND_SLOT_ORDER, and the span sort exactly
    # Y_SPAN_ORDER — so slot storage index == BND_SLOT_ORDER index and profile
    # index == Y_TARGET_INDEX in EVERY store.
    for r in rendered:
        names = [n for n, _ in sorted(r.slot_idx.items(), key=lambda kv: kv[1])]
        assert names == list(BND_SLOT_ORDER), (r.conv_id, names, r.slot_idx)
        spans = [n for n, _ in sorted(r.spans.items(), key=lambda kv: kv[1][0])]
        assert spans == list(Y_SPAN_ORDER), (r.conv_id, spans, r.spans)
    diag = slot_diagnostics(rendered)
    diag_path = args.out_dir / f"{stem}_slot_diagnostics.json"
    c.write_json(
        diag_path,
        {
            "metadata": c.metadata(
                0, len(rendered), "scripts/issue1345_boundary_ablation_capture.py"
            ),
            "round": bg.ROUND_VARIANT,
            "arm_or_comparator": key,
            "smoke": bool(args.smoke),
            **y_meta,
            **diag,
        },
    )
    coinc = {k: round(v, 4) for k, v in diag["coincidence_with_context_rates"].items()}
    print(
        f"[capture][{key}] clean-slot answer-overlap all 0 (hard-asserted); "
        f"ctx_straddle overlap rate {diag['ctx_straddle_answer_overlap_rate']:.4f}; "
        f"coincidence-with-context: {coinc}; leading-gap rate "
        f"{diag['answer_span_leading_gap_rate']:.4f} -> {diag_path}",
        flush=True,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    causal_max_diff = None
    if args.smoke:
        # mode="cosine": the early-position `prefix` slot has no bf16 headroom
        # under a flat atol (#1345 parent note); the #779 two-bar cosine gate
        # keeps the wrong-position bug catcher (real bugs read cos ~0.4-0.6).
        causal_max_diff = ex.causal_check(model, rendered[: min(3, len(rendered))], mode="cosine")

    bs = 8 if args.batch_size == "auto" else int(args.batch_size)
    if args.smoke:
        bs = min(bs, 2)
    peak_layers = sorted(li for li in ex.FROZEN_LAYERS if li < ex.EXPECTED_LAYERS)
    n_visible = None
    try:
        import torch

        n_visible = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 — CPU smoke has no CUDA; informational only
        n_visible = None
    if args.gpu_id is not None and n_visible:
        assert n_visible == 1, (
            f"--gpu-id {args.gpu_id} passed but {n_visible} devices are visible — the "
            "launcher must pin CUDA_VISIBLE_DEVICES=<gpu> per arm (gotchas.md CVD family)"
        )
    print(
        f"[run] key={key} model={args.model} ({model_id}) stem={stem} n={len(rendered)} "
        f"batch_size={bs} gpu_id={args.gpu_id} visible_devices={n_visible}",
        flush=True,
    )
    tr = transition_for(key)
    sidecar_base = {
        "issue": 1345,
        "round": bg.ROUND_VARIANT,
        "arm_or_comparator": key,
        "arm_isolates": bg.ARM_README.get(key, f"{key} comparator store"),
        "regime": format_key(key, args.provenance),
        "model": args.model,
        "model_id": model_id,
        "format": format_key(key, args.provenance),
        "track": TRACK,
        "provenance": args.provenance,
        "story_character_name": c.STORY_CHARACTER_NAME,
        "slot_names": list(BND_SLOT_ORDER),
        "x_clean_slot": X_CLEAN_SLOT,
        "ctx_straddle_slot": X_STRADDLE_SLOT,
        "y_span_order": list(Y_SPAN_ORDER),
        "y_target_index": Y_TARGET_INDEX,
        "headline_slot": HEADLINE_SLOT,
        "transition_appended_verbatim": tr["closer"] + tr["suffix"],
        "transition_read_anchor": tr["read_anchor"],
        "peak_layers": peak_layers,
        "expected_layers": ex.EXPECTED_LAYERS,
        "expected_hidden": ex.EXPECTED_HIDDEN,
        "render_stats": render_stats,
        "slot_diagnostics": diag,
        "git_commit": c.git_commit(),
        "gpu_id": args.gpu_id,
        "causal_check_max_abs_diff": causal_max_diff,
        "causal_check_mode": "cosine" if args.smoke else None,
        "smoke": bool(args.smoke),
        "n_rendered_pre_filter": n_pre_filter,
        "n_dropped_zero_width": len(drops),
        "dropped_conv_ids": [d["conv_id"] for d in drops],
        # Surfaced at manifest top level (not only nested in render_stats) so a
        # downstream conv_id-space reconciliation can find the on-policy
        # gate-reject tail without knowing where the render buried it. Absent
        # keys default empty: the comparator render path has no arm gate.
        "n_leading_ws_span_normalized": ws_stats["normalized"],
        "leading_ws_span_normalized_conv_ids": ws_stats["conv_ids"],
        "n_dropped_gate_reject": render_stats.get("gate_rejects", 0),
        "gate_reject_reasons": render_stats.get("gate_reject_reasons", {}),
        "gate_reject_conv_ids": render_stats.get("gate_reject_conv_ids", []),
        **y_meta,
    }
    shard_size = int(args.shard_size)
    paths: list[Path] = []
    n_done = 0
    for block_idx, block_start in enumerate(range(0, len(rendered), shard_size)):
        block = rendered[block_start : block_start + shard_size]
        records = ex.run_extraction(model, block, peak_layers, pad_id, bs)
        assert len(records) == len(block), (block_idx, len(records), len(block))
        paths += ex.write_shards(
            records,
            args.out_dir,
            stem,
            sidecar_base,
            shard_offset=block_idx,
            shard_size=shard_size,
        )
        n_done += len(records)
        del records, block
        print(f"[capture] key={key} rows {n_done}/{len(rendered)} shards={len(paths)}", flush=True)

    if args.skip_upload:
        print(f"[capture] --skip-upload: {n_done} rows -> {len(paths)} shard(s), LOCAL ONLY")
    else:
        persist_store(
            args.out_dir,
            key,
            args.smoke,
            {
                "n_rows": n_done,
                "n_shards": len(paths),
                "render_stats": render_stats,
                "slot_diagnostics": diag,
                # Which POOL this store was captured on. A full-pool companion and
                # a matched store share stem/model/provenance, so without this the
                # only difference between them lives in the directory they happen
                # to sit in — and a mis-staged store would read as the other one.
                "row_pool": (
                    "matched_to_file"
                    if args.keep_ids_jsonl is not None
                    else ("full" if args.full_pool else "arm_matched")
                ),
                "full_pool": bool(args.full_pool),
                "keep_ids_source": (
                    str(args.keep_ids_jsonl) if args.keep_ids_jsonl is not None else None
                ),
            },
            provenance=args.provenance,
            model_key=args.model,
            hf_prefix_suffix=args.hf_prefix_suffix,
        )
    print(f"[done] {key}: {n_done} rows -> {len(paths)} shard(s) in {args.out_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
