#!/usr/bin/env python
"""Anchor-normalized plan-patch helper (#1631).

Applies ONE anchor-based edit to a UTF-8 text file (typically a plan draft at
``/tmp/issue-<N>-plan-v<K>-<ts>.md``) as the EDIT step of the plan-revision
Edit-success gate (``.claude/skills/adversarial-planner/SKILL.md``
"Edit-success gate"; ``.claude/skills/issue/SKILL.md`` Step 2 mirror). It
never calls ``task.py new-plan-version`` and never touches ``tasks/**`` — the
gate's separate verify + persist steps stay ``&&``-chained after it.

The target file is given EITHER positionally (``plan_patch.py <file> ...``,
canonical) OR via ``--file <file>`` (alias for parity with the ``--file``
spelling every note-bearing workflow CLI takes — ``task.py post-marker`` /
``set-body`` / ``new-plan-version``). Exactly one of the two spellings:
both or neither is a usage error (exit 2, file untouched).

Matching semantics
------------------
The anchor resolves through THREE stages, walked in order; the FIRST stage
with >=1 candidate decides (per-stage candidate sets, NEVER unioned — an
exact match wins even when a drifted near-duplicate exists elsewhere):

1. ``exact`` — byte-exact non-overlapping scan (only a lone trailing newline
   on the anchor is tolerated, via an ``rstrip("\\n")`` comparison fallback —
   anchor files usually end in a newline).
2. ``ws-normalized`` — every maximal whitespace run (space, tab, ``\\n``,
   ``\\r``, ...) collapsed to ONE space on both sides; the anchor is
   additionally stripped. Absorbs line-wrap / indentation drift.
3. ``ws+case-normalized`` — stage 2 plus per-char ``lower()``. A char whose
   ``lower()`` is not length-1 (e.g. ``İ``) is kept as-is so the index map
   stays 1:1 — such anchors fail toward a MISS, never a wrong span.

At the deciding stage: exactly one candidate -> apply; two or more ->
fail-loud ambiguity (exit 2) listing every candidate location — NEVER a
fall-through to a looser stage (looser stages are supersets; falling through
could only add candidates). All three stages empty -> exit 2 with a
nearest-match report (best line window by ``SequenceMatcher`` ratio + a
unified diff of anchor vs window over the ORIGINAL texts). The report is a
REPORT only — this tool never fuzzy-applies (no ``patch(1)``-style fuzz).

Edit modes
----------
``--replace`` splices the payload byte-exactly over the matched span
(``--replace ''`` deletes it). ``--insert-after`` / ``--insert-before`` are
LINE-based: the payload lands as full lines adjacent to the line containing
the span's end / start — a mid-line anchor inserts after/before the
containing LINE, not byte-adjacent to the match. Payloads are read VERBATIM
(no normalization is ever applied to a payload). Insert modes are idempotent
on the EXACT payload block only: a re-run that finds the identical block
already at the insertion point exits 3 (ALREADY-APPLIED, file untouched).
Replace mode is deliberately asymmetric: after a successful replace consumed
the anchor, a crash re-run reads exit 2 (anchor missing), not 3 — a replace
re-run cannot prove the prior apply used THIS payload.

Exit codes
----------
- ``0`` — applied (or a clean ``--dry-run``); stdout carries
  ``PLAN-PATCH APPLIED (<stage> match at lines <A>-<B> of <file>)`` plus the
  unified diff — grep-able positive evidence for the gate's verify step.
- ``2`` — failed, file untouched: missing anchor (nearest-match report),
  ambiguous anchor, ``EDIT NO-OP``, ``--verify-contains`` miss, usage /
  encoding / size errors. Stderr prefix: ``PLAN-PATCH FAILED:``.
- ``3`` — already-applied (insert modes only); file untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import os
import sys
import tempfile
from collections import Counter

# Plans are 50-300 KB; refuse pathological inputs (keeps the O(lines * anchor)
# nearest-match scan trivially fast). Module-level so tests can monkeypatch.
MAX_FILE_BYTES = 10 * 1024 * 1024

STAGE_EXACT = "exact"
STAGE_WS = "ws-normalized"
STAGE_WS_CASE = "ws+case-normalized"

_EXIT_EPILOG = """\
exit codes:
  0  applied (or clean --dry-run); stdout: "PLAN-PATCH APPLIED (<stage> match
     at lines <A>-<B> of <file>)" + the unified diff
  2  failed, file untouched (missing/ambiguous anchor, EDIT NO-OP,
     --verify-contains miss, usage/encoding/size errors)
  3  already-applied (insert modes only: the exact payload block already sits
     at the insertion point); file untouched

Insert-mode idempotency is exact-payload-block-only; --insert-after with a
mid-line anchor inserts after the containing LINE (not byte-adjacent). Stage
order is exact -> ws-normalized -> ws+case-normalized with per-stage candidate
sets (never unioned): an exact match wins even when a drifted near-duplicate
exists elsewhere. A replace-mode crash re-run reads exit 2 (anchor consumed),
asymmetric with insert-mode exit 3 — deliberate.

The target file is given positionally OR via --file (exactly one spelling).

Never pipe plan_patch.py or `task.py new-plan-version` through tail/grep/head:
a pipe masks the exit code ($? reads the filter's status, not this tool's).
The Edit-success gate's `&&` chain relies on this rc — run the command bare,
or redirect to a file (`> /tmp/patch.out 2>&1`) and check rc.
"""


class PatchError(Exception):
    """Any failure that leaves the file untouched -> exit 2."""


class AlreadyApplied(Exception):
    """Insert-mode idempotent re-run -> exit 3, file untouched."""


class MissingAnchor(Exception):
    """Anchor resolved to zero candidates at every stage."""


class AmbiguousAnchor(Exception):
    """Anchor resolved to >=2 candidates at its deciding stage."""

    def __init__(self, stage: str, spans: list[tuple[int, int]]):
        super().__init__(f"{len(spans)} matches at the {stage} stage")
        self.stage = stage
        self.spans = spans


def normalize_with_map(text: str, *, lower: bool) -> tuple[str, list[int]]:
    """Collapse every maximal whitespace run (incl. \\n, \\r, \\t) to ONE space.

    Optionally per-char ``lower()``; a char whose ``lower()`` is not length-1
    is kept as-is, so the index map stays 1:1 (fails toward a stage MISS,
    never a wrong span). Returns ``(normalized, idx_map)`` where
    ``idx_map[i]`` is the index in the ORIGINAL text of normalized char
    ``i`` (a whitespace run maps to its first original char).
    """
    chars: list[str] = []
    idx_map: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            chars.append(" ")
            idx_map.append(i)
            i = j
        else:
            if lower:
                low = ch.lower()
                ch = low if len(low) == 1 else ch
            chars.append(ch)
            idx_map.append(i)
            i += 1
    return "".join(chars), idx_map


def _scan(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Non-overlapping left-to-right spans of ``needle``; needle non-empty."""
    if not needle:  # defensive: empty needles are rejected before any scan
        raise PatchError("internal: empty scan needle")
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        hit = haystack.find(needle, pos)
        if hit < 0:
            return spans
        spans.append((hit, hit + len(needle)))
        pos = hit + len(needle)


def _normalized_candidates(file_text: str, anchor: str, *, lower: bool) -> list[tuple[int, int]]:
    """Original-text spans matching the whitespace-collapsed anchor."""
    norm_file, idx_map = normalize_with_map(file_text, lower=lower)
    norm_anchor, _ = normalize_with_map(anchor, lower=lower)
    norm_anchor = norm_anchor.strip()
    if not norm_anchor:
        return []
    # First/last anchor chars are non-space post-strip, so the mapped span is
    # tight: idx_map[s] and idx_map[e-1] both point at non-space originals.
    return [(idx_map[s], idx_map[e - 1] + 1) for s, e in _scan(norm_file, norm_anchor)]


def find_candidates(file_text: str, anchor: str) -> tuple[str, tuple[int, int]]:
    """Resolve the anchor to exactly one original-text span.

    Walks the stages in order; at the FIRST stage with >=1 candidate: exactly
    one -> return ``(stage, span)``; >=2 -> raise :class:`AmbiguousAnchor`
    (never fall through — looser stages are supersets and can only add
    candidates). All stages empty -> raise :class:`MissingAnchor`.
    """
    exact = _scan(file_text, anchor)
    if not exact:
        stripped = anchor.rstrip("\n")
        if stripped and stripped != anchor:
            exact = _scan(file_text, stripped)
    stages: list[tuple[str, list[tuple[int, int]]]] = [(STAGE_EXACT, exact)]
    if not exact:
        ws = _normalized_candidates(file_text, anchor, lower=False)
        stages.append((STAGE_WS, ws))
        if not ws:
            stages.append((STAGE_WS_CASE, _normalized_candidates(file_text, anchor, lower=True)))
    for stage, spans in stages:
        if len(spans) == 1:
            return stage, spans[0]
        if len(spans) >= 2:
            raise AmbiguousAnchor(stage, spans)
    raise MissingAnchor()


def _best_local_ratio(norm_window: str, norm_anchor: str, floor: float) -> float:
    """Max SequenceMatcher ratio of the anchor vs the window OR any
    anchor-length slice of it (a partial-ratio scan: a short drifted anchor
    inside a LONG markdown line still scores high — the full-window
    ``ratio()`` structurally penalizes long lines). Report-scoring only;
    ``floor`` lets the slice scan prefilter against the best score so far.
    """
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(norm_anchor)
    matcher.set_seq1(norm_window)
    best = matcher.ratio()
    n_a = len(norm_anchor)
    if n_a and len(norm_window) > n_a:
        stride = max(1, n_a // 4)
        starts = list(range(0, len(norm_window) - n_a + 1, stride))
        if starts[-1] != len(norm_window) - n_a:
            starts.append(len(norm_window) - n_a)
        for start in starts:
            matcher.set_seq1(norm_window[start : start + n_a])
            bar = max(best, floor)
            if matcher.real_quick_ratio() <= bar or matcher.quick_ratio() <= bar:
                continue
            ratio = matcher.ratio()
            if ratio > best:
                best = ratio
    return best


def nearest_match_report(file_text: str, anchor: str) -> str:
    """Best-window similarity report for a missing anchor. REPORT only.

    Line-window scan (window height = the anchor's line count, stride 1) over
    ws+case-normalized texts; each window is scored by the MAX of the
    full-window ``SequenceMatcher.ratio()`` and a partial-ratio slice scan
    (:func:`_best_local_ratio`), behind a cheap character-multiset upper
    bound. Always reports the best candidate (no cutoff — as a report there
    is no wrong answer). Degenerate inputs (empty file, file shorter than the
    anchor) get a designed message, never a traceback.
    """
    file_lines = file_text.splitlines()
    if not file_lines:
        return "nearest-match report: file is empty — no candidate window to compare."
    anchor_lines = anchor.splitlines() or [""]
    height = len(anchor_lines)
    prefix = ""
    if len(file_lines) < height:
        prefix = (
            f"nearest-match report: file has {len(file_lines)} line(s), shorter than "
            f"the {height}-line anchor — comparing against the whole file.\n"
        )
        height = len(file_lines)
    norm_anchor, _ = normalize_with_map(anchor, lower=True)
    norm_anchor = norm_anchor.strip()
    n_a = len(norm_anchor)
    anchor_counts = Counter(norm_anchor)
    best: tuple[int, int] = (1, height)
    best_ratio = -1.0
    for i in range(len(file_lines) - height + 1):
        window = "\n".join(file_lines[i : i + height])
        norm_window, _ = normalize_with_map(window, lower=True)
        norm_window = norm_window.strip()
        window_counts = Counter(norm_window)
        common = sum(min(cnt, window_counts.get(ch, 0)) for ch, cnt in anchor_counts.items())
        denom_full = len(norm_window) + n_a
        upper_bound = max(
            (2 * common / denom_full) if denom_full else 0.0,
            (common / n_a) if n_a else 0.0,  # ceiling for any anchor-length slice
        )
        if upper_bound <= best_ratio:
            continue
        ratio = _best_local_ratio(norm_window, norm_anchor, best_ratio)
        if ratio > best_ratio:
            best_ratio = ratio
            best = (i + 1, i + height)
    a, b = best
    diff = difflib.unified_diff(
        anchor_lines,
        file_lines[a - 1 : b],
        fromfile="anchor (as given)",
        tofile=f"closest match in file (lines {a}-{b})",
        lineterm="",
    )
    return (
        prefix
        + f"nearest match: lines {a}-{b} (similarity {max(best_ratio, 0.0):.3f})\n"
        + "\n".join(diff)
    )


def _line_bounds(text: str, index: int) -> tuple[int, int]:
    """(start, end) of the line containing ``text[index]``.

    ``end`` is the index just past the line's newline (``len(text)`` for an
    unterminated last line).
    """
    start = text.rfind("\n", 0, index) + 1
    newline = text.find("\n", index)
    end = len(text) if newline < 0 else newline + 1
    return start, end


def apply_edit(file_text: str, mode: str, span: tuple[int, int], payload: str) -> str:
    """Return the patched text; raise AlreadyApplied / PatchError (no-op)."""
    start, end = span
    if mode == "replace":
        new_text = file_text[:start] + payload + file_text[end:]
    else:
        block = payload if payload.endswith("\n") else payload + "\n"
        if mode == "insert-after":
            _, insert_pos = _line_bounds(file_text, end - 1)
            lead = ""
            if insert_pos == len(file_text) and file_text and not file_text.endswith("\n"):
                lead = "\n"  # keep the payload on its own full lines
            if not lead and file_text[insert_pos : insert_pos + len(block)] == block:
                raise AlreadyApplied()
            new_text = file_text[:insert_pos] + lead + block + file_text[insert_pos:]
        else:  # insert-before
            insert_pos, _ = _line_bounds(file_text, start)
            if (
                insert_pos >= len(block)
                and file_text[insert_pos - len(block) : insert_pos] == block
            ):
                raise AlreadyApplied()
            new_text = file_text[:insert_pos] + block + file_text[insert_pos:]
    if new_text == file_text:
        raise PatchError(
            "EDIT NO-OP — replacement equals existing text; nothing would change "
            "(the #1565 silent-no-op shape)"
        )
    return new_text


def _span_lines(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """1-based (first, last) line numbers covered by an original-text span."""
    start, end = span
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, max(start, end - 1)) + 1
    return first, last


def _unified_diff(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"{path} (before)",
        tofile=f"{path} (after)",
        lineterm="",
    )
    return "\n".join(diff)


def _ambiguous_message(file_text: str, exc: AmbiguousAnchor) -> str:
    lines = [
        f"ambiguous anchor — {len(exc.spans)} matches at the {exc.stage} stage "
        "(unique match required; not falling through to a looser stage):"
    ]
    for span in exc.spans:
        first, last = _span_lines(file_text, span)
        matched = file_text[span[0] : span[1]]
        excerpt = (matched.splitlines() or [""])[0][:80]
        lines.append(f"  lines {first}-{last}: {excerpt!r}")
    lines.append(
        "file untouched — make the anchor more distinctive (>=1 full line of unique context)."
    )
    return "\n".join(lines)


def _read_text_file(path: str, role: str) -> str:
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return fh.read()
    except FileNotFoundError as exc:
        raise PatchError(f"{role} not found: {path!r}") from exc
    except UnicodeDecodeError as exc:
        raise PatchError(f"{role} {path!r} is not UTF-8 text: {exc}") from exc


def _atomic_write(path: str, text: str) -> None:
    """tempfile-in-same-dir + os.replace, preserving bytes via newline=''."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".plan-patch-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def build_parser() -> argparse.ArgumentParser:
    """CLI surface. The target file is the positional ``file`` OR ``--file``
    (dest ``file_opt`` — deliberately distinct: argparse cannot place a
    positional in a mutually exclusive group, and a shared dest is
    parse-order-fragile); ``_run()`` enforces exactly-one at runtime.
    """
    parser = argparse.ArgumentParser(
        prog="plan_patch.py",
        description=(
            "Apply one anchor-based edit to a plan draft: exact-then-normalized "
            "anchor resolution (whitespace-collapsed, then case-tolerant), "
            "unique-match-only, fail-loud with a nearest-match diff. The EDIT "
            "step of the plan-revision Edit-success gate (#1631); the gate's "
            "verify + persist steps stay separate."
        ),
        epilog=_EXIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="the draft file to patch (UTF-8 text, <=10 MB); alternatively via --file",
    )
    parser.add_argument(
        "--file",
        dest="file_opt",
        metavar="FILE",
        help=(
            "alias for the positional target (task.py post-marker --file parity); "
            "give the target exactly once — positionally or via --file"
        ),
    )
    anchor = parser.add_mutually_exclusive_group(required=True)
    anchor.add_argument("--anchor", help="anchor text, inline (short single-line anchors)")
    anchor.add_argument(
        "--anchor-file",
        help="anchor text read verbatim from a file (RECOMMENDED for multi-line anchors)",
    )
    payload = parser.add_mutually_exclusive_group(required=True)
    payload.add_argument(
        "--replace", help="replacement text spliced over the matched span ('' deletes it)"
    )
    payload.add_argument("--replace-file", help="replacement text read verbatim from a file")
    payload.add_argument(
        "--insert-after",
        help="payload inserted as full lines AFTER the line containing the anchor's end",
    )
    payload.add_argument("--insert-after-file", help="file variant of --insert-after")
    payload.add_argument(
        "--insert-before",
        help="payload inserted as full lines BEFORE the line containing the anchor's start",
    )
    payload.add_argument("--insert-before-file", help="file variant of --insert-before")
    parser.add_argument(
        "--verify-contains",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "assert TEXT is present in the patched text BEFORE any write "
            "(repeatable); failure exits 2 with the would-be diff printed"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve + print the would-be diff, write nothing (exit 0 on a unique match)",
    )
    return parser


def _resolve_payload(args: argparse.Namespace) -> tuple[str, str]:
    for mode, inline, from_file in (
        ("replace", args.replace, args.replace_file),
        ("insert-after", args.insert_after, args.insert_after_file),
        ("insert-before", args.insert_before, args.insert_before_file),
    ):
        if inline is not None:
            return mode, inline
        if from_file is not None:
            return mode, _read_text_file(from_file, "payload file")
    raise AssertionError("argparse required group guarantees one payload mode")


def _run(args: argparse.Namespace) -> int:
    if (args.file is None) == (args.file_opt is None):
        raise PatchError(
            "give the target file exactly once: the positional FILE argument "
            "or --file FILE (not both, not neither); file untouched"
        )
    path = args.file if args.file is not None else args.file_opt
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise PatchError(f"cannot stat target file {path!r}: {exc}") from exc
    if size > MAX_FILE_BYTES:
        raise PatchError(
            f"target file {path!r} is {size} bytes — refusing files larger than "
            f"{MAX_FILE_BYTES} bytes (plan drafts are KB-scale)"
        )
    file_text = _read_text_file(path, "target file")
    if args.anchor is not None:
        anchor = args.anchor
    else:
        anchor = _read_text_file(args.anchor_file, "anchor file")
    if not anchor.strip():
        # A zero-length needle in a non-overlapping find() scan is an
        # infinite-loop hazard; an all-whitespace anchor normalizes to "".
        raise PatchError("anchor must contain non-whitespace text")
    mode, payload = _resolve_payload(args)
    if mode != "replace" and payload == "":
        raise PatchError(
            "insert payload must be non-empty (use --replace '' to delete the anchor span)"
        )
    try:
        stage, span = find_candidates(file_text, anchor)
    except MissingAnchor:
        raise PatchError(
            "anchor not found at any stage (exact, ws-normalized, ws+case-normalized)\n"
            + nearest_match_report(file_text, anchor)
        ) from None
    except AmbiguousAnchor as exc:
        raise PatchError(_ambiguous_message(file_text, exc)) from None
    new_text = apply_edit(file_text, mode, span, payload)
    diff = _unified_diff(file_text, new_text, path)
    for needle in args.verify_contains:
        if needle not in new_text:
            raise PatchError(
                f"--verify-contains failed — {needle!r} not present in the patched "
                "text; file untouched. The would-be diff (for diagnosis):\n" + diff
            )
    first, last = _span_lines(file_text, span)
    if args.dry_run:
        print(
            f"PLAN-PATCH DRY-RUN OK ({stage} match at lines {first}-{last} of {path})"
            " — would apply; no write performed"
        )
        print(diff)
        return 0
    _atomic_write(path, new_text)
    print(f"PLAN-PATCH APPLIED ({stage} match at lines {first}-{last} of {path})")
    print(diff)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except AlreadyApplied:
        print("PLAN-PATCH ALREADY-APPLIED — no change made", file=sys.stderr)
        return 3
    except PatchError as exc:
        print(f"PLAN-PATCH FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
