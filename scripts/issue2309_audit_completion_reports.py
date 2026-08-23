"""Historical would-have-fired audit for the #2309 completion-report check.

Measurement BEFORE arming (plan #2309 step 1, same posture as the argcheck
83-of-927 baseline): iterate every task's ``events.jsonl`` under
``--tasks-root`` and report, PER contract kind (``epm:results``,
``epm:experiment-implementation``):

- total rows / signature-matching rows (both-form, case-insensitive);
- the would-have-fired set (signature present, >=1 lettered H3 absent,
  no ``part=K/N`` token), with per-row identifiers for eyeballing;
- observed heading-form variants (validates the lenient regexes);
- the PER-KIND ARMING ASSERT input: signature-matching rows among the
  sentinel/drain population — operationalized by PROVENANCE, never as
  "rows not matching the signature" (that would be circular): a row is
  drain-population iff it carries ``sentinel_fp``/``sentinel_path``
  extras (the poll_pipeline.py / slurm_monitor drain channel threads
  them at every post site) OR its ``by`` is a known drain pass-through
  value (``pod-sentinel``, ``pod-sentinel-envelope-fallback``) — plus a
  full per-kind ``by`` histogram so any further dispatcher pass-through
  values are visible for classification;
- signature COVERAGE among section-bearing rows (>=3 of the four
  lettered H3s) — DIAGNOSTIC only, never an arming gate;
- the v2-shape regression diagnostic (per-header-form counts, so the
  single-form-constant miss is visible);
- the prose-collision count for the tightened ``part=`` token among
  signature-bearing rows.

Read-only over the tasks tree; writes one JSON report to ``--out``.

Usage:
    uv run python scripts/issue2309_audit_completion_reports.py \
        --tasks-root /abs/path/to/repo/tasks --out /tmp/audit2309.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Mirror of the task_workflow predicate regexes (kept inline so the audit
# can run standalone and so the audited definition is visible in one file;
# the shipped predicate lives in task_workflow.py and is test-pinned).
COMPLETION_REPORT_KINDS = ("epm:results", "epm:experiment-implementation")
SIGNATURE = re.compile(
    r"^##\s*(Completion|Implementation)\s+Report\b", re.MULTILINE | re.IGNORECASE
)
SECTION = re.compile(r"^###\s*\(([a-d])\)", re.MULTILINE)
PART_TOKEN = re.compile(r"\bpart\s*=\s*\d+\s*/\s*\d+", re.IGNORECASE)
# Per-form counters (the v2 regression diagnostic).
FORM_COMPLETION = re.compile(r"^##\s*Completion\s+Report\b", re.MULTILINE | re.IGNORECASE)
FORM_IMPLEMENTATION = re.compile(r"^##\s*Implementation\s+Report\b", re.MULTILINE | re.IGNORECASE)

DRAIN_BY_VALUES = {"pod-sentinel", "pod-sentinel-envelope-fallback"}
DRAIN_EXTRA_KEYS = ("sentinel_fp", "sentinel_path")


def _is_drain_row(row: dict) -> bool:
    """Drain-population membership by PROVENANCE (structural extras or a
    known drain pass-through ``by``) — deliberately independent of the
    signature so the arming assert cannot be circular."""
    if any(k in row for k in DRAIN_EXTRA_KEYS):
        return True
    return (row.get("by") or "") in DRAIN_BY_VALUES


def audit(tasks_root: Path) -> dict:
    """Sweep every events.jsonl under tasks_root; return the report dict."""
    per_kind: dict[str, dict] = {
        kind: {
            "total_rows": 0,
            "signature_rows": 0,
            "form_counts": {"completion": 0, "implementation": 0, "both": 0},
            "header_variants": Counter(),
            "by_histogram": Counter(),
            "drain_rows": 0,
            "drain_rows_with_signature": 0,
            "drain_signature_examples": [],
            "part_token_rows_with_signature": 0,
            "would_fire": [],
            "section_bearing_rows": 0,  # >=3 of the four lettered H3s
            "section_bearing_with_signature": 0,
        }
        for kind in COMPLETION_REPORT_KINDS
    }

    for events_path in sorted(tasks_root.glob("*/*/events.jsonl")):
        task_id = events_path.parent.name
        with open(events_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = row.get("kind")
                if kind not in per_kind:
                    continue
                stats = per_kind[kind]
                stats["total_rows"] += 1
                note = row.get("note") or ""
                stats["by_histogram"][row.get("by") or ""] += 1
                is_drain = _is_drain_row(row)
                if is_drain:
                    stats["drain_rows"] += 1

                present = set(SECTION.findall(note))
                if len(present) >= 3:
                    stats["section_bearing_rows"] += 1

                sig = SIGNATURE.search(note)
                if not sig:
                    continue
                stats["signature_rows"] += 1
                if len(present) >= 3:
                    stats["section_bearing_with_signature"] += 1
                stats["header_variants"][sig.group(0)] += 1
                has_c = bool(FORM_COMPLETION.search(note))
                has_i = bool(FORM_IMPLEMENTATION.search(note))
                if has_c and has_i:
                    stats["form_counts"]["both"] += 1
                elif has_c:
                    stats["form_counts"]["completion"] += 1
                elif has_i:
                    stats["form_counts"]["implementation"] += 1
                if is_drain:
                    stats["drain_rows_with_signature"] += 1
                    stats["drain_signature_examples"].append(
                        {"task": task_id, "ts": row.get("ts"), "version": row.get("version")}
                    )
                if PART_TOKEN.search(note):
                    stats["part_token_rows_with_signature"] += 1
                    continue  # part rows are excluded from refusal by design
                missing = [c for c in "abcd" if c not in present]
                if missing:
                    stats["would_fire"].append(
                        {
                            "task": task_id,
                            "ts": row.get("ts"),
                            "version": row.get("version"),
                            "by": row.get("by"),
                            "missing": missing,
                            "header": sig.group(0),
                            "n_chars": len(note),
                        }
                    )

    # JSON-ify counters; derive the arming verdict inputs.
    report: dict = {"tasks_root": str(tasks_root), "kinds": {}}
    for kind, stats in per_kind.items():
        report["kinds"][kind] = {
            "total_rows": stats["total_rows"],
            "signature_rows": stats["signature_rows"],
            "form_counts": stats["form_counts"],
            "header_variants": dict(stats["header_variants"].most_common()),
            "by_histogram": dict(stats["by_histogram"].most_common(30)),
            "drain_rows": stats["drain_rows"],
            "drain_rows_with_signature": stats["drain_rows_with_signature"],
            "drain_signature_examples": stats["drain_signature_examples"][:20],
            "part_token_rows_with_signature": stats["part_token_rows_with_signature"],
            "would_fire_count": len(stats["would_fire"]),
            "would_fire": stats["would_fire"],
            "section_bearing_rows": stats["section_bearing_rows"],
            "section_bearing_with_signature": stats["section_bearing_with_signature"],
            "signature_coverage_among_section_bearing": (
                round(stats["section_bearing_with_signature"] / stats["section_bearing_rows"], 4)
                if stats["section_bearing_rows"]
                else None
            ),
            # Arming condition (a): zero signature-matching rows in the
            # provenance-defined drain population.
            "arming_condition_a_zero_drain_signature": stats["drain_rows_with_signature"] == 0,
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-root", required=True, type=Path, help="absolute path to tasks/ tree")
    ap.add_argument("--out", required=True, type=Path, help="JSON report output path")
    args = ap.parse_args()
    report = audit(args.tasks_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Console digest: everything except the long would_fire lists.
    digest = {
        k: {kk: vv for kk, vv in v.items() if kk != "would_fire"}
        for k, v in report["kinds"].items()
    }
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
