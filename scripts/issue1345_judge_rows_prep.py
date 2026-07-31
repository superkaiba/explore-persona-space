#!/usr/bin/env python
"""Issue #1345 — normalize a cell's uploaded rows into judge-leg input.

The judge legs need three fields per row: `conv_id`, `question`, `answer`, and
`capped`. None of the uploaded row files carries all four, and two of the gaps
would fail SILENTLY rather than loudly:

1. `capped` is ABSENT from every uploaded on-policy row file. The generator only
   started attaching `finish_reason` / `capped` to KEPT rows after job 16257 had
   already produced them, so the stratification variable does not exist in the
   data. Left alone, `capped_of` falls back to a missing `finish_reason`, every
   row reads NATURAL, the stratified draw silently degenerates into a simple
   random sample, and the capped/natural sub-means report a single all-natural
   block — a wrong answer with no error. It is fully RECOVERABLE: the
   `raw_onpolicy_*.jsonl` companion carries `finish_reason` per `conv_id`, and
   its per-cell length counts reproduce the measured 2478 / 591 / 1382 / 6
   exactly. This script does that join and fails loud on an incomplete one.

2. The CHARACTER cells' on-policy rows carry no `answer` field at all (their
   injected siblings do) — only `story` plus `parsed_turns` char spans. The
   answer is `story[a_start:a_end]`; where an `answer` field also exists the
   span is verified against it (the capture's own trust boundary), so a
   mis-sliced span cannot reach the judge.

Content hygiene: this script reads real-user-derived rollout text and NEVER
prints it. Every diagnostic is a count, a length, or a hash.

CLI:
  uv run python scripts/issue1345_judge_rows_prep.py \\
      --rows <kept_or_onpolicy_rows.jsonl> --raw <raw_onpolicy_*.jsonl> \\
      --out <prepared.jsonl> --cell op_ntpl_instruct
  uv run python scripts/issue1345_judge_rows_prep.py --self-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402

# An answer span too short to carry a ratable answer is DROPPED and counted —
# never rated, never silently kept. Measured span lengths on the character cells:
# median 228-265 chars, p1 22-30, and a 0.1-0.6% tail of 0-4-char spans (a stray
# character, not an answer). The floor is deliberately MINIMAL and applied
# UNIFORMLY to every cell: a higher floor would drop short answers
# preferentially, and answer LENGTH correlates with the AI-likeness the rubric is
# built to isolate from it — a length filter on that instrument is the very
# confound rule 25 names.
ANSWER_CHAR_FLOOR = 8
# Above this share, a sub-floor tail is no longer a data tail — the SPAN
# extraction itself broke. Measured worst character cell: 0.6%. The ceiling
# applies ONLY to span-derived answers, because only a span can be mis-sliced: a
# short answer read straight out of a `response`/`answer` FIELD is data, and the
# real corpus supplies plenty of it (2.1% of the 5,000 track-S rows are under the
# floor — genuine one-word replies in unscreened user conversations, not a break).
SHORT_ANSWER_DROP_CEILING = 0.02


def _span(row: dict, start_key: str, end_key: str) -> str:
    """Text from the row's `story` under a parsed_turns char span, else ""."""
    turns = row.get("parsed_turns") or []
    if not turns or "story" not in row:
        return ""
    t = turns[0]
    if start_key not in t or end_key not in t:
        return ""
    return str(row["story"])[int(t[start_key]) : int(t[end_key])]


def normalize_source_row(row: dict) -> dict:
    """Give a row a `conv_id`, deriving it exactly as the capture does.

    The parent track-S corpus rows (`{prompt_idx, prompt, response}`) carry NO
    conv_id — the capture builds it through `to_single_turn`, which maps
    prompt/response to u1/a1 and sets `conv_id = f"s{prompt_idx}"`. Calling that
    same function is what keeps the content-drift pairing key identical to the
    key the injected stores were built under; re-deriving the convention here
    would be a second source of truth free to drift.
    """
    if "conv_id" in row:
        return row
    from issue825_extract_turnstore import to_single_turn

    return {**row, **to_single_turn(row)}


def question_of(row: dict) -> str:
    """The question, from whichever field this row schema carries it in."""
    for key in ("question", "prompt", "u1"):
        if str(row.get(key, "")).strip():
            return str(row[key])
    return _span(row, "q_start", "q_end")


def answer_of(row: dict) -> tuple[str, str]:
    """(answer_text, source) — the answer plus which field it came from.

    A row carrying BOTH an explicit answer and a span is cross-checked: the span
    IS the answer by construction (the capture re-verifies the same invariant),
    so a disagreement means the span no longer indexes the answer slot and the
    judge would be rating mis-sliced text.
    """
    span = _span(row, "a_start", "a_end")
    for key in ("response", "answer", "a1"):
        if str(row.get(key, "")).strip():
            text = str(row[key])
            if span and c.norm_text(span) != c.norm_text(text):
                raise AssertionError(
                    f"conv_id {row.get('conv_id')!r}: parsed_turns span (len {len(span)}) is not "
                    f"the {key} field (len {len(text)}) — the span does not index the answer slot"
                )
            return text, key
    assert span, (
        f"conv_id {row.get('conv_id')!r} carries no response/answer field and no usable "
        f"parsed_turns span: fields {sorted(row)}"
    )
    return span, "parsed_turns_span"


def load_capped_index(raw_path: Path) -> dict[str, bool]:
    """conv_id -> whether that generation stopped on the token cap.

    Built from the raw generation file, the only artifact carrying
    `finish_reason` for the pre-cap-split cells.
    """
    idx: dict[str, bool] = {}
    for r in c.read_jsonl(raw_path):
        cid = str(r["conv_id"])
        fr = str(r.get("finish_reason", ""))
        assert fr, f"raw row {cid!r} has no finish_reason — cap recovery is impossible"
        idx[cid] = fr == "length"
    assert idx, f"{raw_path} yielded no finish_reason rows"
    return idx


def prepare(
    rows: list[dict],
    capped_index: dict[str, bool] | None,
    *,
    cell: str,
) -> tuple[list[dict], dict]:
    """Normalize rows to the judge-leg schema, recovering `capped` by conv_id."""
    out: list[dict] = []
    stats = {
        "cell": cell,
        "n_in": len(rows),
        "answer_sources": {},
        "capped_source": "row_field" if capped_index is None else "raw_finish_reason_join",
        "n_capped": 0,
        "n_no_question": 0,
        "n_short_answer": 0,
        "n_short_by_source": {},
        "n_join_miss": 0,
    }
    for raw_row in rows:
        r = normalize_source_row(raw_row)
        cid = str(r["conv_id"])
        q = question_of(r)
        a, src = answer_of(r)
        stats["answer_sources"][src] = stats["answer_sources"].get(src, 0) + 1
        if not q.strip():
            stats["n_no_question"] += 1
        if len(a.strip()) < ANSWER_CHAR_FLOOR:
            # Dropped, not rated: too short to carry an answer to score. Counted
            # BY SOURCE, because only the span source can be mis-sliced — a short
            # field answer is real data, a short span is a parse artifact.
            stats["n_short_answer"] += 1
            stats["n_short_by_source"][src] = stats["n_short_by_source"].get(src, 0) + 1
            continue
        if capped_index is None:
            capped = bool(r.get("capped")) or str(r.get("finish_reason", "")) == "length"
        else:
            if cid not in capped_index:
                stats["n_join_miss"] += 1
                continue
            capped = capped_index[cid]
        stats["n_capped"] += int(capped)
        out.append({"conv_id": cid, "question": q, "answer": a, "capped": capped, "cell": cell})

    stats["n_out"] = len(out)
    stats["capped_rate"] = round(stats["n_capped"] / len(out), 6) if out else None
    # Fail loud rather than judge a silently-thinned cell: the raw pool is a
    # SUPERSET of the kept rows, so every kept row must find its finish_reason.
    assert stats["n_join_miss"] == 0, (
        f"{cell}: {stats['n_join_miss']} kept rows had no raw finish_reason — the raw file is "
        "not this cell's generation pool; cap stratification would be built on a partial join"
    )
    assert stats["n_no_question"] == 0, f"{cell}: {stats['n_no_question']} rows have no question"
    stats["short_answer_drop_share"] = (
        round(stats["n_short_answer"] / len(rows), 6) if rows else 0.0
    )
    n_span_short = stats["n_short_by_source"].get("parsed_turns_span", 0)
    n_span = stats["answer_sources"].get("parsed_turns_span", 0)
    span_short_share = (n_span_short / n_span) if n_span else 0.0
    stats["span_short_share"] = round(span_short_share, 6)
    assert span_short_share <= SHORT_ANSWER_DROP_CEILING, (
        f"{cell}: {n_span_short}/{n_span} SPAN-derived answers ({span_short_share:.1%}) fell "
        f"under {ANSWER_CHAR_FLOOR} chars, over the {SHORT_ANSWER_DROP_CEILING:.0%} ceiling — "
        "that is a span-extraction break, not a data tail"
    )
    return out, stats


def _self_check() -> None:
    """Exercise every extraction branch on synthetic rows (no real data)."""
    story = "Human: What is X?\n\nAssistant: X is a thing that does stuff, at length.\n"
    q0, q1 = story.index("What"), story.index("?") + 1
    a0, a1 = story.index("X is a"), story.index("length.") + len("length.")
    char_op = {
        "conv_id": "s1",
        "question": story[q0:q1],
        "story": story,
        "parsed_turns": [{"q_start": q0, "q_end": q1, "a_start": a0, "a_end": a1}],
    }
    comparator = {"conv_id": "s2", "prompt": "Q?", "response": "an answer that is long enough"}
    slot = {
        "conv_id": "s3",
        "answer": story[a0:a1],
        "story": story,
        "parsed_turns": [{"q_start": q0, "q_end": q1, "a_start": a0, "a_end": a1}],
    }
    rows = [char_op, comparator, slot]
    prepared, stats = prepare(rows, {"s1": True, "s2": False, "s3": True}, cell="selfcheck")
    assert len(prepared) == 3, prepared
    assert stats["answer_sources"] == {"parsed_turns_span": 1, "response": 1, "answer": 1}, stats
    assert stats["n_capped"] == 2 and stats["capped_rate"] == round(2 / 3, 6), stats

    # A degenerate 1-char span is DROPPED and counted, not rated...
    tiny_story = "Human: Q?\n\nAssistant: X\n"
    tq0, tq1 = tiny_story.index("Q"), tiny_story.index("?") + 1
    ta0 = tiny_story.index("X")
    tiny = {
        "conv_id": "s4",
        "question": "Q?",
        "story": tiny_story,
        "parsed_turns": [{"q_start": tq0, "q_end": tq1, "a_start": ta0, "a_end": ta0 + 1}],
    }
    # (padded past the systemic-break ceiling with SPAN rows — the ceiling's
    # denominator is span-derived answers only, so the pad must be span-derived)
    pad = [
        {
            "conv_id": f"p{i}",
            "question": story[q0:q1],
            "story": story,
            "parsed_turns": [{"q_start": q0, "q_end": q1, "a_start": a0, "a_end": a1}],
        }
        for i in range(60)
    ]
    idx = {"s1": True, "s2": False, "s3": True, "s4": False, **{f"p{i}": False for i in range(60)}}
    kept, st = prepare([*rows, tiny, *pad], idx, cell="selfcheck")
    assert len(kept) == 63 and st["n_short_answer"] == 1, st
    assert all(r["conv_id"] != "s4" for r in kept), "a 1-char span reached the judge"
    # ...but a SYSTEMIC sub-floor tail is an extraction break and fails loud.
    try:
        prepare([tiny] * 3, {"s4": False}, cell="selfcheck")
    except AssertionError as e:
        assert "extraction break" in str(e), e
    else:
        raise AssertionError("a 100% sub-floor cell was accepted")
    # A span that no longer indexes the answer slot must fail loud, not pass text.
    bad = dict(slot, answer="something else entirely, definitely not the span")
    try:
        answer_of(bad)
    except AssertionError as e:
        assert "does not index the answer slot" in str(e), e
    else:
        raise AssertionError("a mis-sliced span was accepted")
    # A kept row missing from the raw pool must fail loud, not thin the cell.
    try:
        prepare(rows, {"s1": True}, cell="selfcheck")
    except AssertionError as e:
        assert "no raw finish_reason" in str(e), e
    else:
        raise AssertionError("a partial join was accepted")
    # A track-S corpus row (no conv_id) normalizes through the CANONICAL
    # to_single_turn, so the drift pairing key matches the injected stores'.
    corpus = {"prompt_idx": 7, "prompt": "Q?", "response": "the corpus answer, long enough"}
    norm = normalize_source_row(corpus)
    assert norm["conv_id"] == "s7", norm
    assert question_of(norm) == "Q?" and answer_of(norm)[0] == "the corpus answer, long enough"
    got, cs = prepare([corpus], None, cell="track_s")
    assert got[0]["conv_id"] == "s7" and cs["n_out"] == 1, (got, cs)

    print("self-check OK: 4 extraction branches, span guard, join guard, corpus key", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, help="the cell's uploaded kept/on-policy rows JSONL")
    ap.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw generation JSONL carrying finish_reason (omit when rows carry `capped`)",
    )
    ap.add_argument("--out", type=Path, help="prepared judge-input JSONL")
    ap.add_argument("--cell", required=False, default="cell", help="cell tag for the report")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
        return

    assert args.rows and args.out, "--rows and --out are required"
    rows = c.read_jsonl(args.rows)
    assert rows, f"{args.rows} is empty"
    capped_index = load_capped_index(args.raw) if args.raw else None
    prepared, stats = prepare(rows, capped_index, cell=args.cell)
    stats["rows_path"] = str(args.rows)
    stats["rows_sha256"] = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    if args.raw:
        stats["raw_path"] = str(args.raw)
        stats["raw_sha256"] = hashlib.sha256(args.raw.read_bytes()).hexdigest()
        stats["raw_pool_n"] = len(capped_index or {})
    stats["metadata"] = c.metadata(0, len(prepared), "scripts/issue1345_judge_rows_prep.py")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in prepared:
            f.write(json.dumps(r) + "\n")
    c.write_json(args.out.with_suffix(".prep_report.json"), stats)
    print(
        f"[prep] {args.cell}: {stats['n_in']} -> {stats['n_out']} rows | "
        f"capped {stats['n_capped']} ({stats['capped_rate']}) via {stats['capped_source']} | "
        f"answers {stats['answer_sources']} -> {args.out}",
        flush=True,
    )
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
