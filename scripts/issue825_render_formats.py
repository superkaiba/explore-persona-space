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
  a4_bpe_boundary         cross-format first-content-token BPE mismatch rate;
                          fail loud if > 0.10.
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


def render_chat(conv: dict[str, Any], tokenizer: Any) -> Rendered:
    """Render under the Qwen chat template with slot + content-span indices.

    Turn structure u1->a1->u2->a2. Slots: ``a1`` = final token of the
    ``<|im_start|>assistant\\n`` header opening a1; ``u2`` = final token of the
    ``<|im_start|>user\\n`` header opening u2. Spans = content tokens of each
    turn EXCLUDING role headers and ``<|im_end|>``.
    """
    u1, a1, u2, a2 = (conv[k] for k in TURN_KEYS)
    # Segments alternate header / content / terminator so each gets its own
    # token range. Qwen template: <|im_start|>{role}\n{content}<|im_end|>\n
    segments = [
        "<|im_start|>user\n",
        u1,
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        a1,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        u2,
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        a2,
        "<|im_end|>\n",
    ]
    input_ids, ranges = _tokenize_segments(segments, tokenizer)
    # Segment indices: header/content/terminator triplets per turn, in order.
    content_seg = {"u1": 1, "a1": 4, "u2": 7, "a2": 10}
    spans = {turn: ranges[i] for turn, i in content_seg.items()}
    # Slot = final token of the header segment opening the target turn.
    a1_header_end = ranges[3][1]
    u2_header_end = ranges[6][1]
    slot_idx = {"a1": a1_header_end - 1, "u2": u2_header_end - 1}
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
    u1, a1, u2, a2 = (conv[k] for k in TURN_KEYS)
    segments = [
        "User: ",
        u1,
        "\n\n",
        "Assistant: ",
        a1,
        "\n\n",
        "User: ",
        u2,
        "\n\n",
        "Assistant: ",
        a2,
        "\n\n",
    ]
    input_ids, ranges = _tokenize_segments(segments, tokenizer)
    content_seg = {"u1": 1, "a1": 4, "u2": 7, "a2": 10}
    spans = {turn: ranges[i] for turn, i in content_seg.items()}
    a1_header_end = ranges[3][1]
    u2_header_end = ranges[6][1]
    slot_idx = {"a1": a1_header_end - 1, "u2": u2_header_end - 1}
    return Rendered(
        input_ids=input_ids,
        slot_idx=slot_idx,
        spans=spans,
        format="naturalistic",
        conv_id=str(conv.get("conv_id", "")),
        meta={"n_tokens": len(input_ids)},
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
        for turn in TURN_KEYS:
            if turn not in r.spans:
                raise AssertionError(f"a2 FAIL: conv {r.conv_id} missing span {turn!r}.")
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
    """Cross-format first-content-token BPE mismatch rate over all convs.

    For each conversation and turn, compare the FIRST content token id under
    the chat rendering vs the naturalistic rendering. Report the mismatch
    rate; fail loud if it exceeds BPE_MISMATCH_MAX_RATE.
    """
    total = 0
    mismatches = 0
    for conv in convs:
        r_chat = render_chat(conv, tokenizer)
        r_nat = render_naturalistic(conv, tokenizer)
        for turn in TURN_KEYS:
            cs, _ = r_chat.spans[turn]
            ns, _ = r_nat.spans[turn]
            total += 1
            if r_chat.input_ids[cs] != r_nat.input_ids[ns]:
                mismatches += 1
    rate = mismatches / total if total else 0.0
    if rate > BPE_MISMATCH_MAX_RATE:
        raise AssertionError(
            f"a4 FAIL: cross-format first-content-token BPE mismatch rate "
            f"{rate:.3f} > {BPE_MISMATCH_MAX_RATE} ({mismatches}/{total})."
        )
    return {
        "assert": "a4_bpe_boundary",
        "status": "PASS",
        "mismatch_rate": rate,
        "mismatches": mismatches,
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
            opener = row["opener"].strip()
            if opener in seen:
                continue
            seen.add(opener)
            source_mix[str(row.get("source", "unknown"))] += 1
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
            missing = [k for k in TURN_KEYS if k not in conv]
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
