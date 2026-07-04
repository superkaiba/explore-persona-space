#!/usr/bin/env python3
"""Report critic-blocker mechanizability tagging and the verifier-landing ratchet.

Walks ``tasks/*/*/events.jsonl``, parses the four critique-marker families
(``epm:plan-critique``, ``epm:code-review``, ``epm:interp-critique``,
``epm:clean-result-critique`` plus their ``-codex`` twins — an exact
allowlist; reconcile/decision/ensemble derivatives are excluded because
they quote blockers verbatim and would double-count), and reports
per-month counts of:

- critique markers seen, FAIL-class verdicts among them, and markers whose
  verdict could not be classified (``verdict_unknown`` — keeps the
  classifier's blind spot visible);
- blocker tags ``mechanizable: yes`` vs ``mechanizable: no`` (the tag the
  critic specs require on every blocker as of 2026-06-12 — older markers
  carry none and are counted gracefully as untagged);
- FAIL-class markers carrying no mechanizable tag at all (``untagged``);
- a best-effort "landed" count: ``epm:workflow-fix-applied`` markers whose
  note names a workflow-surface verifier (``verify_task_body.py``,
  ``audit_clean_results_body_discipline.py``, ``SPEC.md``,
  ``verify_plan.py``, ``consistency-checker``) — the ratchet metric for
  mechanizable findings becoming permanent mechanical gates.

Used by the ``/weekly`` critic-recurrence harvest (mechanization ratchet).

Usage:
    uv run python scripts/critic_mechanization_report.py
    uv run python scripts/critic_mechanization_report.py --json
    uv run python scripts/critic_mechanization_report.py --tasks-dir tasks/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The four verdict-bearing critique families, each with its Codex twin.
# Deliberately an EXACT allowlist, not a prefix match: derived kinds
# (epm:code-review-decision / -ensemble / -fix, epm:plan-critique-reconcile,
# epm:clean-result-critique-applied, ...) quote blockers verbatim and would
# double-count tags beyond the accepted Claude+Codex ensemble duplication.
CRITIQUE_FAMILIES = (
    "epm:plan-critique",
    "epm:code-review",
    "epm:interp-critique",
    "epm:clean-result-critique",
)
CRITIQUE_KINDS = frozenset(
    kind for family in CRITIQUE_FAMILIES for kind in (family, family + "-codex")
)
WORKFLOW_FIX_APPLIED_KIND = "epm:workflow-fix-applied"
VERIFIER_TARGETS = (
    "verify_task_body.py",
    "audit_clean_results_body_discipline.py",
    "SPEC.md",
    "verify_plan.py",
    "consistency-checker",
)

# Matches `mechanizable: yes`, `Mechanizable:** no`, `mechanizable — yes`, etc.
MECH_TAG_RE = re.compile(r"mechanizable\b[^a-zA-Z0-9]{0,6}(yes|no)\b", re.IGNORECASE)

# Verdict-anchored classification. Order matters: longer alternatives first so
# `FAIL_NOT_WORTH_CONTINUING` is not swallowed by `FAIL`. Real notes carry an
# infix between the anchor word and the verdict (`Verdict — round 3, PASS`,
# `Verdict (round 2) — PASS`), so allow up to 24 same-line chars before the
# verdict word rather than a bare separator run.
_VERDICT_WORDS = (
    r"(FAIL_NOT_WORTH_CONTINUING|NEEDS_TARGETED_FIX|BLOCKED_NEEDS_USER_DECISION"
    r"|PASS|CONCERNS|FAIL|REVISE|REJECT|APPROVE)"
)
VERDICT_RE = re.compile(r"Verdict\b[^\n]{0,24}?\b" + _VERDICT_WORDS + r"\b", re.IGNORECASE)
# clean-result-critique notes open `Round <K>: PASS|FAIL — ...` (also
# `Round 1: needs_targeted_fix — ...`, `Round 2 (Claude-only): PASS`,
# `Round 3 (final) — orchestrator-verified PASS.`) with no Verdict word.
# Line-anchored so a mid-prose "Round N" mention can't misfire.
ROUND_VERDICT_RE = re.compile(
    r"^Round\s+\d+[^\n]{0,40}?\b" + _VERDICT_WORDS + r"\b",
    re.IGNORECASE | re.MULTILINE,
)
# Plan critiques use `Rating: REVISE`.
RATING_RE = re.compile(r"Rating\b[^A-Za-z0-9]{0,8}(REJECT|REVISE|APPROVE)", re.IGNORECASE)
# Verdict-bearing headings without the word "Verdict": `## Code Review: PASS`,
# `## Code-Review v3 — PASS`.
HEADING_VERDICT_RE = re.compile(
    r"^#{1,4}\s*Code[- ]Review[^\n]{0,24}?\b(PASS|CONCERNS|FAIL)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Last resort: a bare verdict token on its own line right under the sentinel.
BARE_VERDICT_LINE_RE = re.compile(r"^\s*\**(PASS|CONCERNS|FAIL|REVISE|REJECT|APPROVE)\**\s*$")

FAIL_CLASS = {
    "FAIL",
    "REVISE",
    "REJECT",
    "NEEDS_TARGETED_FIX",
    "FAIL_NOT_WORTH_CONTINUING",
    "BLOCKED_NEEDS_USER_DECISION",
}
PASS_CLASS = {"PASS", "CONCERNS", "APPROVE"}

COLUMNS = (
    "critique_markers",
    "fail_class_markers",
    "verdict_unknown",
    "mechanizable_yes",
    "mechanizable_no",
    "fail_untagged",
    "verifier_fixes_applied",
)


def classify_verdict(note: str) -> str:
    """Return ``"fail"`` / ``"pass"`` / ``"unknown"`` for a critique-marker note.

    Looks at the head of the note only (verdict lines lead every critique
    template); a FAIL word buried deep in prose does not flip the class.
    Markers that classify ``unknown`` are surfaced in the report's
    ``verdict_unknown`` column so the blind spot stays visible.
    """
    head = note[:800]
    for pattern in (VERDICT_RE, ROUND_VERDICT_RE, RATING_RE, HEADING_VERDICT_RE):
        m = pattern.search(head)
        if m:
            verdict = m.group(1).upper()
            if verdict in FAIL_CLASS:
                return "fail"
            if verdict in PASS_CLASS:
                return "pass"
    # Bare verdict token on its own line among the first few lines (some
    # notes put `PASS` / `FAIL` alone right under the marker sentinel).
    lines = [ln for ln in head.splitlines() if ln.strip()]
    for ln in lines[:5]:
        m = BARE_VERDICT_LINE_RE.match(ln)
        if m:
            verdict = m.group(1).upper()
            return "fail" if verdict in FAIL_CLASS else "pass"
    return "unknown"


def iter_events(tasks_dir: Path):
    """Yield parsed event dicts from every ``tasks/*/*/events.jsonl``; skip junk lines."""
    for path in sorted(tasks_dir.glob("*/*/events.jsonl")):
        try:
            # split("\n"), NOT splitlines(): raw U+2028/U+2029/NEL inside
            # ensure_ascii=False notes are Unicode line boundaries that would
            # shred valid records (gotchas.md; #825 → #950).
            lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        except OSError:
            continue
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                yield ev


def build_report(tasks_dir: Path) -> dict[str, dict[str, int]]:
    """Return ``{YYYY-MM: {column: count}}`` over every events.jsonl under tasks_dir."""
    months: dict[str, Counter] = defaultdict(Counter)
    for ev in iter_events(tasks_dir):
        kind = ev.get("kind") or ""
        month = (ev.get("ts") or "")[:7] or "unknown"
        note = ev.get("note") or ""
        if kind == WORKFLOW_FIX_APPLIED_KIND:
            if any(target in note for target in VERIFIER_TARGETS):
                months[month]["verifier_fixes_applied"] += 1
            continue
        if kind not in CRITIQUE_KINDS:
            continue
        counts = months[month]
        counts["critique_markers"] += 1
        tags = [t.lower() for t in MECH_TAG_RE.findall(note)]
        counts["mechanizable_yes"] += tags.count("yes")
        counts["mechanizable_no"] += tags.count("no")
        verdict = classify_verdict(note)
        if verdict == "fail":
            counts["fail_class_markers"] += 1
            if not tags:
                counts["fail_untagged"] += 1
        elif verdict == "unknown":
            counts["verdict_unknown"] += 1
    return {
        month: {col: counts.get(col, 0) for col in COLUMNS}
        for month, counts in sorted(months.items())
    }


def render_table(report: dict[str, dict[str, int]]) -> str:
    """Render the per-month report (plus a TOTAL row) as an aligned text table."""
    header = (
        "month",
        "critiques",
        "fail",
        "unknown",
        "mech_yes",
        "mech_no",
        "fail_untagged",
        "verifier_fixes",
    )
    rows = [header]
    totals = Counter()
    for month, counts in report.items():
        rows.append((month, *(str(counts[col]) for col in COLUMNS)))
        totals.update(counts)
    rows.append(("TOTAL", *(str(totals.get(col, 0)) for col in COLUMNS)))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="tasks/ root to walk (default: the canonical tasks/ dir via task_workflow)",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of the text table")
    args = p.parse_args(argv)

    tasks_root = args.tasks_dir
    if tasks_root is None:
        from explore_persona_space.task_workflow import tasks_dir as resolve_tasks_dir

        tasks_root = resolve_tasks_dir()
    if not tasks_root.is_dir():
        print(f"error: tasks dir not found: {tasks_root}", file=sys.stderr)
        return 1

    report = build_report(tasks_root)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        sys.stdout.write(render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
