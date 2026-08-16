#!/usr/bin/env python
"""Blind-forward a Codex verdict's machine-readable concern rows to the ledger (#2326).

The Codex twin reviewers run sandboxed and never mutate ``concerns.jsonl``;
their verdict templates emit LINE-START-anchored machine rows instead::

    CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

(the exact literal row ``CONCERN:: none`` when there is nothing to persist).
The orchestrator runs this forwarder against the EXTRACTED marker block
(``$MB`` from SKILL.md's "File-only Codex verdict posting" recipe) so the
concerns reach the per-task ledger without any findings prose entering
orchestrator context. See SKILL.md's "Codex concerns persistence at verdict
collection" subsection for the two invocations (validate pre-post, persist
post-post) and the resume-recovery re-run.

Contract:

* Parses ONLY lines matching ``^CONCERN:: `` anywhere in ``--file``; field
  order fixed (token 1 = severity, token 2 = concern id, remainder =
  summary; whitespace-split on the first two tokens only).
* Validates ALL rows structurally before persisting ANY (all-or-nothing):
  severity in ``task_workflow.CONCERN_SEVERITIES``, id matching
  ``task_workflow._CONCERN_ID_RE``, non-empty summary, no duplicate ids,
  ``CONCERN:: none`` only as the sole row, and at most ``_MAX_ROWS`` (50)
  rows per block (availability cap: a sub-50k marker admits ~2,000
  minimal rows, i.e. thousands of durable flock+append+commit cycles).
* Persists via ``task_workflow.raise_concern`` (the library layer: flock +
  events.jsonl mirror — NEVER a ``task.py`` shellout; idempotent no-op on a
  same (id, round, severity) replay). A summary over the 200-char library
  cap is stored as its word-boundary lead with the FULL text in the
  ``evidence`` field (mirrors the CLI's non-lossy shift, #2121).
* Validation is all-or-nothing; PERSISTENCE is per-row: each valid row is
  durably raised in turn, so a mid-loop OPERATIONAL failure (exit 4) can
  leave a PARTIAL ledger. The exit-4 count reports COMPLETED calls — a
  FLOOR on the ledger, never an overcount: ``raise_concern`` appends the
  ``concerns.jsonl`` row BEFORE the ``events.jsonl`` mirror and the
  covering commit (``task_workflow._append_concern_event``), so when the
  MIRROR append (or the commit) is what failed, the failing call's own
  row has landed in the working tree, uncommitted. What happens next is
  MODE-DEPENDENT (``task_workflow`` routes task writes through a managed
  main-pin worktree whenever the PRIMARY checkout is off ``main``;
  primary-on-``main`` — the guard-enforced normal state — is non-routed):
  NON-ROUTED, the uncommitted row survives on disk and the idempotent
  same-(id, round, severity) replay early-returns on it (the early-return
  keys on ``concerns.jsonl`` alone), so a missing ``events.jsonl`` mirror
  is never re-created — the named accepted residual (#2326 reconciler):
  the mirror is a decision-inert audit breadcrumb (``markers.md:80``;
  ``list_concerns`` reads the ledger exclusively). ROUTED, the next
  resolver re-sync runs ``reset --hard main``
  (``task_workflow._ensure_managed_main_worktree``), physically deleting
  the uncommitted row, so a fresh-process replay re-appends row AND
  mirror — full convergence. In BOTH modes the recovery is identical:
  re-run the persist invocation alone (idempotent; converges the ledger
  to the complete row set). Never a batch transaction in the frozen
  library layer (Non-goals).
* Output discipline: stdout carries ONLY counts, concern ids (kebab
  tokens), and content-free reason codes (``bad-severity | bad-id |
  empty-summary | too-few-fields | duplicate-id | none-with-rows |
  too-many-rows | heading-without-rows | missing-concerns-block``).
  Summaries NEVER print; an operational failure prints only the exception
  CLASS name (a message could embed summary text).
* Exit codes: 0 ok (persisted, idempotent no-op, or clean ``none``) - 1
  malformed rows - 3 missing/contradictory concerns block under
  ``--require-block`` - 4 operational persistence failure mid-loop
  (partial ledger possible — the failing row itself may have landed;
  re-run the persist invocation alone: idempotent, converges the ledger;
  mirror restoration is mode-dependent, see the persistence bullet) - 2
  argparse/usage (incl. an unreadable or non-UTF-8 ``--file``: the
  marker was never examined, so the invocation is the bug).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space import task_workflow  # noqa: E402

_ROW_RE = re.compile(r"^CONCERN:: (.*)$", re.MULTILINE)
_HEADING_RE = re.compile(r"(?mi)^(?:#{1,6}\s*|\*\*)?concerns to persist\b")
_SUMMARY_CAP = 200
# Availability cap (#2326 `unbounded-concern-row-fanout`): realistic twin
# verdicts carry <= ~10 rows; a repetition-degenerate verdict could otherwise
# drive thousands of durable flock+append+commit cycles. Checked BEFORE any
# per-row validation or write.
_MAX_ROWS = 50


def _truncate_summary(summary: str, cap: int = _SUMMARY_CAP) -> tuple[str, str | None]:
    """Word-boundary lead for an over-cap summary (mirrors scripts/task.py).

    Returns ``(kept, full_original_or_None)``; the full original goes to the
    concern's ``evidence`` field so nothing is lost (#2121 non-lossy shift).
    """
    summary = summary.rstrip()
    if len(summary) <= cap:
        return summary, None
    budget = cap - 3  # room for the "..." marker
    cut = summary.rfind(" ", 0, budget + 1)
    if cut <= 0:
        cut = budget
    kept = summary[:cut].rstrip() + "..."
    if kept == "...":
        kept = summary[:budget].rstrip() + "..."
    return kept, summary


def _validate_rows(
    raw_rows: list[str],
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Validate every ``CONCERN:: `` payload; return (parsed rows, problems).

    ``problems`` entries are content-free reason codes keyed by row ordinal;
    summaries and malformed tokens never enter them.
    """
    if len(raw_rows) > _MAX_ROWS:
        # Reject before ANY per-row work: exit 1 via the problems path, so
        # nothing is ever persisted from an over-cap block (content-free:
        # counts only).
        return [], [f"too-many-rows ({len(raw_rows)} > {_MAX_ROWS})"]
    problems: list[str] = []
    parsed: list[tuple[str, str, str]] = []
    none_rows = [r for r in raw_rows if r.strip() == "none"]
    if none_rows and len(raw_rows) > 1:
        problems.append("none-with-rows")
    seen: set[str] = set()
    for idx, raw in enumerate(raw_rows, start=1):
        if raw.strip() == "none":
            continue
        parts = raw.split(None, 2)
        if len(parts) < 3:
            problems.append(f"row {idx}: too-few-fields")
            continue
        severity, cid, summary = parts
        if severity not in task_workflow.CONCERN_SEVERITIES:
            problems.append(f"row {idx}: bad-severity")
        if not task_workflow._CONCERN_ID_RE.match(cid):
            problems.append(f"row {idx}: bad-id")
        if not summary.strip():
            # Defensive dead code: ``split(None, 2)`` sheds a whitespace-only
            # third field, so that shape lands in ``too-few-fields`` above and
            # this branch is unreachable today. Kept as a guard against a
            # future tokenizer change (both #2326 round-1 reviewers agree).
            problems.append(f"row {idx}: empty-summary")
        if cid in seen:
            problems.append(f"row {idx}: duplicate-id")
        seen.add(cid)
        parsed.append((severity, cid, summary.strip()))
    return parsed, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Blind-forward machine-readable CONCERN:: rows from an extracted "
            "Codex verdict marker block to the per-task concerns ledger "
            "(#2326). Prints only counts + kebab ids + content-free reason "
            "codes; never findings prose."
        )
    )
    parser.add_argument("task_id", type=int, help="task number (tasks/<status>/<N>/)")
    parser.add_argument("--file", required=True, help="path to the extracted marker block ($MB)")
    parser.add_argument(
        "--by", required=True, help="raised_by attribution (the codex reviewer role)"
    )
    parser.add_argument("--round", type=int, required=True, help="review round (>= 1)")
    parser.add_argument(
        "--require-block",
        action="store_true",
        help=(
            "contract sites only (codex-code-reviewer / "
            "codex-clean-result-critic): a missing concerns block, or a "
            "concerns heading with zero machine rows, exits 3 (MALFORMED -> "
            "the site's stricter-retry)"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="parse + validate only; write nothing (the pre-post gate)",
    )
    args = parser.parse_args(argv)
    if args.round < 1:
        parser.error("--round must be a positive integer")
    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        # ValueError covers UnicodeDecodeError (its subclass): a non-UTF-8
        # --file, like an unreadable one, means the marker was never
        # examined — the invocation is the bug, so take the usage exit (2).
        parser.error(f"--file unreadable: {exc}")

    raw_rows = _ROW_RE.findall(text)
    heading_present = bool(_HEADING_RE.search(text))
    parsed, problems = _validate_rows(raw_rows)

    if problems:
        for problem in problems:
            print(f"MALFORMED: {problem}")
        return 1

    if not parsed:
        # Zero real rows: either the clean `CONCERN:: none` (valid empty), a
        # heading with no machine rows, or no concerns block at all.
        if raw_rows:  # the sole `CONCERN:: none` row
            print(
                "concerns-block OK: 0 row(s)" if args.validate_only else "persisted 0/0 concern(s)"
            )
            return 0
        if args.require_block:
            code = "heading-without-rows" if heading_present else "missing-concerns-block"
            print(f"REQUIRE-BLOCK FAIL: {code}")
            return 3
        if heading_present:
            print("WARN: concerns-heading-without-rows")
        return 0

    if args.validate_only:
        print(f"concerns-block OK: {len(parsed)} row(s)")
        return 0

    persisted: list[str] = []
    for idx, (severity, cid, summary) in enumerate(parsed, start=1):
        kept, full = _truncate_summary(summary)
        try:
            task_workflow.raise_concern(
                args.task_id,
                cid,
                severity=severity,
                summary=kept,
                raised_by=args.by,
                raised_at_round=args.round,
                evidence=full,
            )
        except Exception as exc:
            # Operational persistence failure (disk-full / flock OSError is
            # the realistic residual; validation pre-satisfies the library
            # guards). Distinct exit 4 so callers never bin it with the
            # exit-1 MALFORMED class (#2326 exit-taxonomy-operational-
            # collision). Content-free by design: the exception CLASS only —
            # a message could embed summary text. The printed count is
            # COMPLETED calls, a floor: row idx's OWN ledger row may ALSO
            # have landed when the failure hit the events.jsonl mirror or
            # the covering commit inside raise_concern (ledger-first
            # append). The idempotent re-run converges the concerns
            # LEDGER; whether the missing mirror is also restored is
            # mode-dependent (module docstring).
            print(
                f"OPERATIONAL: persist-failed row {idx} ({cid}): "
                f"{type(exc).__name__} - {len(persisted)}/{len(parsed)} persisted"
            )
            return 4
        persisted.append(cid)
    print(f"persisted {len(persisted)}/{len(parsed)} concern(s): {' '.join(persisted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
