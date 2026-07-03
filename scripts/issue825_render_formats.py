"""Issue #825 W1 — render conversations in both formats + Phase-0 asserts.

Renders u1->a1->u2->a2 conversations under (a) the Qwen chat template and
(b) a naturalistic plain-text format, computing deterministic slot + span
token indices via incremental tokenization of growing prefix segments (with a
full-sequence concatenation cross-check that fails loud on mismatch).

Phase-0 asserts implemented here (each fail-loud):
  a1_tokenizer_identity   both models' tokenizers give identical input_ids on
                          50 rendered strings per format.
  a2_span_integrity       every profile span non-empty, within-sequence,
                          headers excluded.
  a3_causal_slot_equality SKIPPED here — needs model forwards; the extract
                          script owns it. Signature stub below.
  a4_bpe_boundary         cross-format INTERIOR-CORE identity gate (content
                          spans token-identical up to <=1 boundary-merged
                          token per end; mismatch rate <= 0.10 else fail
                          loud); the first-content-token mismatch rate is
                          reported as a diagnostic only (plan v4 G2).
  a5_passb_keys           verify_passb_keys(pt_path) helper — reports keys +
                          shapes of a candidate pass_b tensor file, returns a
                          verdict dict (caller decides reuse vs regen).
  a6_opener_sanity        >= 2000 openers post-dedup with source-mix report.

Usage:
  uv run python scripts/issue825_render_formats.py \
      --conversations data/issue_825/conversations.jsonl \
      --out-manifest data/issue_825/render_manifest.jsonl \
      --assert-report data/issue_825/phase0_assert_report.json [--smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.issue_825.common import (
    MODEL_INSTRUCT,
    MODEL_PRETRAINED,
    Rendered,
)

TURN_KEYS = ("u1", "a1", "u2", "a2")


def _present_turns(conv: dict) -> list[str]:
    """Ordered turn keys present in the conversation (Track S: u1+a1 only).

    Requires at least the (u1, a1) pair and CONTIGUOUS presence in TURN_KEYS
    order (a conv with u2 but no a1 is malformed — fail loud).
    """
    present = [k for k in TURN_KEYS if conv.get(k)]
    assert present[:2] == ["u1", "a1"], f"conv {conv.get('conv_id')!r} missing u1/a1"
    expect = list(TURN_KEYS[: len(present)])
    assert present == expect, (
        f"conv {conv.get('conv_id')!r} has non-contiguous turns {present} (expected {expect})"
    )
    return present


BPE_MISMATCH_MAX_RATE = 0.10
MIN_OPENERS_POST_DEDUP = 2000
N_TOKENIZER_IDENTITY_STRINGS = 50

# Expected pass_b per-example shapes (Track S: n=5000, 28 layers, 3584 hidden).
PASSB_N = 5000
PASSB_LAYERS = 28
PASSB_HIDDEN = 3584


# ---------------------------------------------------------------------------
# Segment-based rendering core
# ---------------------------------------------------------------------------
def _tokenize_segments(
    segments: list[str], tokenizer: Any
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize a growing prefix and return full ids + per-segment token ranges.

    Each segment's token range is computed as the delta between the tokenized
    lengths of consecutive prefixes — deterministic and unambiguous across
    special tokens (avoids offset-mapping ambiguity). Cross-check: the
    full-string tokenization must equal the concatenation of the incremental
    prefix tokenization; raises on mismatch (fail loud).
    """
    ranges: list[tuple[int, int]] = []
    prefix = ""
    prev_ids: list[int] = []
    for seg in segments:
        prefix += seg
        ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if ids[: len(prev_ids)] != prev_ids:
            raise AssertionError(
                "Incremental tokenization is not prefix-stable at segment "
                f"{len(ranges)}: previous prefix ids are not a prefix of the "
                "extended tokenization. Segment boundaries would be ambiguous."
            )
        ranges.append((len(prev_ids), len(ids)))
        prev_ids = ids
    full_ids = tokenizer("".join(segments), add_special_tokens=False)["input_ids"]
    if full_ids != prev_ids:
        raise AssertionError(
            "Full-sequence tokenization != concatenation of incremental prefix "
            f"tokenization ({len(full_ids)} vs {len(prev_ids)} tokens). "
            "Rendering is not deterministic for this conversation."
        )
    return full_ids, ranges


_LAST_TOK_KEY = "_seg_last_tok"


def _tokenize_segments_offsets(
    segments: list[str], tokenizer: Any
) -> tuple[list[int], list[tuple[int, int]], dict]:
    """Offsets-mapping segmentation for formats whose boundaries BPE-merge.

    Tokenizes the FULL joined string once with ``return_offsets_mapping``;
    each segment's token range covers tokens FULLY contained in its char
    range. ``straddlers[str(i)]`` counts tokens crossing INTO segment i from
    the previous one (the a4 BPE-boundary diagnostic); the reserved
    ``_LAST_TOK_KEY`` entry holds, per segment, the index of the token
    containing that segment's final character (slot lookup). Asserts every
    segment maps to a non-degenerate range.
    """
    text = "".join(segments)
    bounds: list[tuple[int, int]] = []
    pos = 0
    for seg in segments:
        bounds.append((pos, pos + len(seg)))
        pos += len(seg)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    assert len(ids) == len(offs), (len(ids), len(offs))
    ranges: list[tuple[int, int]] = []
    straddlers: dict = {}
    seg_last_tok: list[int] = []
    for i, (cs, ce) in enumerate(bounds):
        contained = [t for t, (a, b) in enumerate(offs) if a >= cs and b <= ce and b > a]
        if contained:
            ranges.append((contained[0], contained[-1] + 1))
        else:
            # Segment fully swallowed by straddling tokens (short header +
            # aggressive merge). Range is empty at the first overlapping token.
            overlap = [t for t, (a, b) in enumerate(offs) if a < ce and b > cs]
            anchor = overlap[0] if overlap else 0
            ranges.append((anchor, anchor))
        n_straddle = sum(1 for a, b in offs if a < cs < b)
        if n_straddle:
            straddlers[str(i)] = n_straddle
        last = [t for t, (a, b) in enumerate(offs) if a <= ce - 1 < b]
        assert last, f"no token contains the final char of segment {i}"
        seg_last_tok.append(last[0])
    straddlers[_LAST_TOK_KEY] = seg_last_tok
    return ids, ranges, straddlers


def _header_slot(ranges: list[tuple[int, int]], seg_idx: int) -> int:
    """Slot = last token FULLY CONTAINED in the header segment (the ':').

    NOT the token containing the header's final char: the header's trailing
    space BPE-folds into the first content word (`` What``), so that token
    would leak the target turn's first content word INTO c_x — contaminating
    the very DV chain the map predicts (code-review round-1 blocker). The
    fully-contained range for ``User: `` ends at the ``:`` token, whose
    next-token prediction is the (merged) first content token — the plan
    §4.2 rule verbatim.
    """
    s, e = ranges[seg_idx]
    assert e > s, f"header segment {seg_idx} has no fully-contained token"
    return e - 1


def render_chat(conv: dict[str, Any], tokenizer: Any) -> Rendered:
    """Render under the Qwen chat template with slot + content-span indices.

    Turn structure u1->a1->u2->a2. Slots: ``a1`` = final token of the
    ``<|im_start|>assistant\\n`` header opening a1; ``u2`` = final token of the
    ``<|im_start|>user\\n`` header opening u2. Spans = content tokens of each
    turn EXCLUDING role headers and ``<|im_end|>``.
    """
    turns = _present_turns(conv)
    # Segments alternate header / content / terminator so each gets its own
    # token range. Qwen template: <|im_start|>{role}\n{content}<|im_end|>\n
    segments: list[str] = []
    content_seg: dict[str, int] = {}
    header_seg: dict[str, int] = {}
    for turn in turns:
        role = "user" if turn.startswith("u") else "assistant"
        header_seg[turn] = len(segments)
        segments.append(f"<|im_start|>{role}\n")
        content_seg[turn] = len(segments)
        segments.append(conv[turn])
        segments.append("<|im_end|>\n")
    input_ids, ranges = _tokenize_segments(segments, tokenizer)
    spans = {turn: ranges[i] for turn, i in content_seg.items()}
    # Slot = final token of the header segment opening the target turn
    # (assistant slot before a1 always; user slot before u2 when present).
    slot_idx = {"a1": ranges[header_seg["a1"]][1] - 1}
    if "u2" in header_seg:
        slot_idx["u2"] = ranges[header_seg["u2"]][1] - 1
    return Rendered(
        input_ids=input_ids,
        slot_idx=slot_idx,
        spans=spans,
        format="chat",
        conv_id=str(conv.get("conv_id", "")),
        meta={"n_tokens": len(input_ids)},
    )


def render_naturalistic(conv: dict[str, Any], tokenizer: Any) -> Rendered:
    """Render as plain text: ``User: <u1>\\n\\nAssistant: <a1>\\n\\n...``.

    Slot = last token of the role header (the token containing ":"); spans =
    content tokens between the header and the ``\\n\\n`` delimiter.
    """
    turns = _present_turns(conv)
    segments = []
    content_seg = {}
    header_seg = {}
    for turn in turns:
        role = "User" if turn.startswith("u") else "Assistant"
        header_seg[turn] = len(segments)
        segments.append(f"{role}: ")
        content_seg[turn] = len(segments)
        segments.append(conv[turn])
        segments.append("\n\n")
    # Naturalistic headers BPE-merge into content ("User: " + "What" can
    # tokenize as ["User", ":", " What"]), so incremental prefix tokenization
    # is NOT prefix-stable here (it is for chat, whose boundaries are special
    # tokens). Use offsets-mapping segmentation instead: tokenize the FULL
    # string once, assign each segment the tokens FULLY contained in its char
    # range, and count boundary straddlers (the a4 BPE-mismatch-rate assert
    # reads them; the plan's robustness re-fit excludes the first content
    # token for exactly this reason).
    input_ids, ranges, straddlers = _tokenize_segments_offsets(segments, tokenizer)
    spans = {turn: ranges[i] for turn, i in content_seg.items()}
    # Slot = last FULLY-CONTAINED header token (the ':'), per _header_slot.
    slot_idx = {"a1": _header_slot(ranges, header_seg["a1"])}
    if "u2" in header_seg:
        slot_idx["u2"] = _header_slot(ranges, header_seg["u2"])
    return Rendered(
        input_ids=input_ids,
        slot_idx=slot_idx,
        spans=spans,
        format="naturalistic",
        conv_id=str(conv.get("conv_id", "")),
        meta={
            "n_tokens": len(input_ids),
            "boundary_straddlers": straddlers,
            "first_content_straddled": {
                t: bool(straddlers.get(str(i))) for t, i in content_seg.items()
            },
        },
    )


_RENDERERS = {"chat": render_chat, "naturalistic": render_naturalistic}


# ---------------------------------------------------------------------------
# Phase-0 asserts
# ---------------------------------------------------------------------------
def a1_tokenizer_identity(
    convs: list[dict[str, Any]], tok_instruct: Any, tok_pretrained: Any
) -> dict[str, Any]:
    """Both models' tokenizers must give identical input_ids per format.

    Checked on up to N_TOKENIZER_IDENTITY_STRINGS rendered strings per format;
    raises on the first mismatch (fail loud).
    """
    checked = {"chat": 0, "naturalistic": 0}
    for fmt, renderer in _RENDERERS.items():
        for conv in convs[:N_TOKENIZER_IDENTITY_STRINGS]:
            r_i = renderer(conv, tok_instruct)
            r_p = renderer(conv, tok_pretrained)
            if r_i.input_ids != r_p.input_ids:
                raise AssertionError(
                    f"a1 FAIL: tokenizer divergence on conv_id={r_i.conv_id!r} "
                    f"format={fmt}: instruct produced {len(r_i.input_ids)} "
                    f"tokens, pretrained {len(r_p.input_ids)}."
                )
            checked[fmt] += 1
    return {"assert": "a1_tokenizer_identity", "status": "PASS", "checked": checked}


def a2_span_integrity(rendered: list[Rendered]) -> dict[str, Any]:
    """Every profile span non-empty, within-sequence, and header-excluded.

    Header exclusion is checked structurally: each span must start strictly
    after its turn's slot/header region and end at or before sequence end.
    """
    n = 0
    for r in rendered:
        seq_len = len(r.input_ids)
        # Present turns only: Track-S rows carry u1/a1; the contiguity
        # invariant is enforced at render time by _present_turns.
        for turn in [k for k in TURN_KEYS if k in r.spans]:
            start, end = r.spans[turn]
            if not (0 <= start < end <= seq_len):
                raise AssertionError(
                    f"a2 FAIL: conv {r.conv_id} span {turn}=({start},{end}) "
                    f"empty or outside sequence (len={seq_len})."
                )
        for slot_name, idx in r.slot_idx.items():
            if not (0 <= idx < seq_len):
                raise AssertionError(
                    f"a2 FAIL: conv {r.conv_id} slot {slot_name}={idx} outside "
                    f"sequence (len={seq_len})."
                )
            start, _end = r.spans[slot_name]
            if idx >= start:
                raise AssertionError(
                    f"a2 FAIL: conv {r.conv_id} slot {slot_name}={idx} not "
                    f"strictly before its content span start {start} — header "
                    "tokens would leak into the profile span."
                )
        n += 1
    return {"assert": "a2_span_integrity", "status": "PASS", "n_rendered": n}


def a3_causal_slot_equality(model: Any, rendered: Rendered, layers: tuple[int, ...]) -> None:
    """SKIPPED HERE — requires model forwards; the extract script owns a3.

    Contract: under causal attention, the slot activation read from the FULL
    four-turn forward pass must equal (within tolerance) the activation read
    from a forward pass over only the prefix ending at the slot — a position's
    activation is unaffected by later tokens. The extract script implements
    this using ``extract_layer_activations(model, input_ids, layers, ...)``
    on both the full sequence and the truncated prefix, comparing at
    ``rendered.slot_idx`` per layer.
    """
    raise NotImplementedError("a3 runs in the extract script (needs model forwards).")


def a4_bpe_boundary(convs: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """Cross-format content-token BPE divergence over all convs (gate + diagnostic).

    The FIRST content token differs across formats BY CONSTRUCTION: the
    naturalistic ``User: `` header ends with a space that Qwen BPE folds into
    the next word token (`` What`` vs ``What``), so a first-token gate fires
    at ~1.0 vacuously. The GATE therefore compares each turn's content span
    EXCLUDING the first content token in both formats (the plan's robustness
    re-fit excludes that token for exactly this reason); the first-token
    mismatch rate is still measured and REPORTED as a diagnostic, not gated.
    Fails loud when the rest-of-span mismatch rate exceeds
    BPE_MISMATCH_MAX_RATE.
    """
    total = 0
    rest_mismatches = 0
    first_mismatches = 0
    for conv in convs:
        r_chat = render_chat(conv, tokenizer)
        r_nat = render_naturalistic(conv, tokenizer)
        # Present turns only — two-turn Track-S rows carry u1/a1 (round-3
        # review residual: an unconditional TURN_KEYS loop would KeyError).
        for turn in [k for k in TURN_KEYS if k in r_chat.spans]:
            cs, ce = r_chat.spans[turn]
            ns, ne = r_nat.spans[turn]
            total += 1
            if r_chat.input_ids[cs] != r_nat.input_ids[ns]:
                first_mismatches += 1
            # Core equality up to <=1 boundary-merged token per end: the
            # naturalistic fully-contained span may drop a head token (merged
            # with the header space) and/or a tail token (merged with the
            # \n\n terminator). REAL divergence = the interior cores differ
            # under every such alignment.
            chat_core = r_chat.input_ids[cs + 1 : ce - 1]
            nat_ids = r_nat.input_ids
            candidates = [
                nat_ids[ns:ne],
                nat_ids[ns + 1 : ne],
                nat_ids[ns : ne - 1],
                nat_ids[ns + 1 : ne - 1],
            ]
            if chat_core not in candidates:
                rest_mismatches += 1
    rate = rest_mismatches / total if total else 0.0
    first_rate = first_mismatches / total if total else 0.0
    if rate > BPE_MISMATCH_MAX_RATE:
        raise AssertionError(
            f"a4 FAIL: cross-format content-span (first-token-excluded) BPE "
            f"mismatch rate {rate:.3f} > {BPE_MISMATCH_MAX_RATE} "
            f"({rest_mismatches}/{total})."
        )
    return {
        "assert": "a4_bpe_boundary",
        "status": "PASS",
        "rest_of_span_mismatch_rate": rate,
        "first_token_mismatch_rate_diagnostic": first_rate,
        "mismatches": rest_mismatches,
        "total": total,
    }


def verify_passb_keys(pt_path: str) -> dict[str, Any]:
    """a5: report keys + shapes of a candidate pass_b tensor file.

    Loads with ``torch.load(map_location="cpu", mmap=True)`` (no GPU, no full
    materialization), reports every key + shape, and asserts per-example
    fields are consistent with n=5000 / 28 layers / 3584 hidden IF present.
    Returns a verdict dict — the caller decides reuse vs regen; this helper
    never silently swallows a shape violation (it raises).
    """
    import torch

    obj = torch.load(pt_path, map_location="cpu", mmap=True)
    if not isinstance(obj, dict):
        raise AssertionError(f"a5 FAIL: {pt_path} is not a dict (got {type(obj).__name__}).")
    report: dict[str, Any] = {"assert": "a5_passb_keys", "path": pt_path, "keys": {}}
    per_example_ok = True
    for key, val in obj.items():
        shape = tuple(val.shape) if hasattr(val, "shape") else None
        report["keys"][key] = {"type": type(val).__name__, "shape": shape}
        if shape is None:
            continue
        if shape and shape[0] == PASSB_N:
            if len(shape) >= 2 and shape[1] not in (PASSB_LAYERS, PASSB_HIDDEN):
                per_example_ok = False
                raise AssertionError(
                    f"a5 FAIL: per-example field {key!r} shape {shape} "
                    f"inconsistent with (n={PASSB_N}, layers={PASSB_LAYERS}, "
                    f"hidden={PASSB_HIDDEN})."
                )
            if len(shape) == 3 and shape[2] != PASSB_HIDDEN:
                per_example_ok = False
                raise AssertionError(
                    f"a5 FAIL: per-example field {key!r} shape {shape} has "
                    f"hidden dim {shape[2]} != {PASSB_HIDDEN}."
                )
    has_per_example = any(
        (info["shape"] or ())[:1] == (PASSB_N,) for info in report["keys"].values()
    )
    report["status"] = "PASS" if per_example_ok else "FAIL"
    report["has_per_example_fields"] = has_per_example
    report["verdict"] = "reusable-shape-consistent" if has_per_example else "no-per-example-fields"
    return report


def a6_opener_sanity(harvest_jsonl: str) -> dict[str, Any]:
    """>= MIN_OPENERS_POST_DEDUP openers post-dedup, with source-mix report.

    Operates on the opener-harvest JSONL (one object per row with at least an
    ``opener`` text field and a ``source`` field). Dedup on exact opener text.
    """
    seen: set[str] = set()
    source_mix: Counter[str] = Counter()
    with open(harvest_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Accept both the generator's conversation rows (u1/opener_source)
            # and a bare harvest row (opener/source) — code-review round-1 fix.
            opener_val = row.get("u1") or row.get("opener")
            assert opener_val, f"a6: row missing u1/opener field: {sorted(row)[:6]}"
            opener = str(opener_val).strip()
            if opener in seen:
                continue
            seen.add(opener)
            source_mix[str(row.get("opener_source") or row.get("source", "unknown"))] += 1
    n = len(seen)
    if n < MIN_OPENERS_POST_DEDUP:
        raise AssertionError(
            f"a6 FAIL: only {n} unique openers post-dedup "
            f"(need >= {MIN_OPENERS_POST_DEDUP}). Source mix: {dict(source_mix)}"
        )
    return {
        "assert": "a6_opener_sanity",
        "status": "PASS",
        "n_unique_openers": n,
        "source_mix": dict(source_mix),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_conversations(path: str) -> list[dict[str, Any]]:
    convs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            missing = [k for k in ("u1", "a1") if k not in conv]
            if missing:
                raise AssertionError(f"Conversation {conv.get('conv_id')!r} missing {missing}.")
            convs.append(conv)
    if not convs:
        raise AssertionError(f"No conversations loaded from {path}.")
    return convs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", required=True, help="Input conversations JSONL.")
    parser.add_argument("--out-manifest", required=True, help="Output render-manifest JSONL.")
    parser.add_argument("--assert-report", required=True, help="Output assert-report JSON.")
    parser.add_argument(
        "--openers-harvest",
        default=None,
        help="Opener-harvest JSONL for a6 (skipped when omitted).",
    )
    parser.add_argument(
        "--passb-pt",
        default=None,
        help="Candidate pass_b .pt file for a5 (skipped when omitted).",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: cap at 20 conversations.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok_instruct = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
    tok_pretrained = AutoTokenizer.from_pretrained(MODEL_PRETRAINED)

    convs = _load_conversations(args.conversations)
    if args.smoke:
        convs = convs[:20]

    report: dict[str, Any] = {"n_conversations": len(convs), "smoke": args.smoke, "asserts": []}

    # a1 — tokenizer identity across models (both formats).
    report["asserts"].append(a1_tokenizer_identity(convs, tok_instruct, tok_pretrained))

    # Render every conversation in both formats (instruct tokenizer; a1 just
    # proved the pretrained tokenizer is identical).
    rendered: list[Rendered] = []
    for conv in convs:
        rendered.append(render_chat(conv, tok_instruct))
        rendered.append(render_naturalistic(conv, tok_instruct))

    # a2 — span/slot integrity on every rendered conversation.
    report["asserts"].append(a2_span_integrity(rendered))

    # a3 — deferred to the extract script (needs model forwards).
    report["asserts"].append(
        {
            "assert": "a3_causal_slot_equality",
            "status": "SKIPPED",
            "note": "requires model forwards; the extract script owns a3 "
            "(signature stub in this module).",
        }
    )

    # a4 — cross-format first-content-token BPE mismatch rate.
    report["asserts"].append(a4_bpe_boundary(convs, tok_instruct))

    # a5 — pass_b key/shape verification (optional input).
    if args.passb_pt:
        report["asserts"].append(verify_passb_keys(args.passb_pt))
    else:
        report["asserts"].append({"assert": "a5_passb_keys", "status": "SKIPPED"})

    # a6 — opener sanity (optional input).
    if args.openers_harvest:
        report["asserts"].append(a6_opener_sanity(args.openers_harvest))
    else:
        report["asserts"].append({"assert": "a6_opener_sanity", "status": "SKIPPED"})

    # Write the render manifest (ids + slots + spans only; no tensors).
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as fh:
        for r in rendered:
            fh.write(json.dumps(dataclasses.asdict(r)) + "\n")

    report_path = Path(args.assert_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Rendered {len(rendered)} sequences -> {out_manifest}")
    print(f"Assert report -> {report_path}")
    for entry in report["asserts"]:
        print(f"  {entry['assert']}: {entry['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
