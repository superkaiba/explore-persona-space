#!/usr/bin/env python
"""Post an ``epm:step-completed v1`` marker on a Sagan experiment.

Used by ``.claude/skills/issue/SKILL.md`` at every EXIT site to record
which step finished, what the next step is (looked up from
``.claude/workflow.yaml § steps``), and whether the exit was ``clean`` /
``parked`` / ``failure-exit``. The §5 re-entry router
(:mod:`explore_persona_space.orchestrate.resume`) reads the LATEST such
marker on re-invocation and decides whether to skip ahead or full-replay.

Usage:

    uv run python scripts/post_step_completed.py \\
        --issue 320 --step 5b --exit-kind clean \\
        [--notes "code-review PASS, advancing"] [--dry-run]

``--issue N`` is ``experiments.number`` in Sagan. The marker is appended to the experiment's ``workflow_events``
table via the Sagan HTTP API (see :mod:`sagan_state`).

The helper looks up ``next_expected_step`` from ``workflow.yaml`` and
fills the marker body. Refuses to post if the step ID is not in
workflow.yaml, or if ``exit_kind`` is not one of ``clean`` / ``parked``
/ ``failure-exit`` (typo guard — silent typos bypass the §5 router).

Idempotency: the helper does NOT dedupe — a re-invocation appends a
second event. The router consumes the LATEST marker, so duplicates
are harmless but visible noise. Skill callers should invoke this once
per EXIT site.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# scripts/ on sys.path so sibling sagan_state module imports cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_state as sagan_state

# Path to the project's workflow.yaml (relative to repo root). The
# helper expects to be run with the repo root as cwd.
WORKFLOW_YAML = Path(".claude/workflow.yaml")

VALID_EXIT_KINDS = ("clean", "parked", "failure-exit")


def _git_head_short() -> str:
    """Return the current HEAD's short SHA, or 'unknown' on error."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_workflow_steps() -> dict[str, dict]:
    """Return a {step_id: row_dict} map from workflow.yaml § steps."""
    try:
        import yaml  # PyYAML — already a dep via pyproject.toml line 27
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"PyYAML required but not importable: {exc}") from exc

    if not WORKFLOW_YAML.exists():
        raise SystemExit(
            f"workflow.yaml not found at {WORKFLOW_YAML} (cwd={Path.cwd()}). Run from repo root."
        )
    doc = yaml.safe_load(WORKFLOW_YAML.read_text())
    steps = doc.get("steps", [])
    if not isinstance(steps, list):
        raise SystemExit("workflow.yaml § steps must be a list")
    by_id: dict[str, dict] = {}
    for row in steps:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id"))
        by_id[sid] = row
    return by_id


def build_marker_body(
    *,
    step: str,
    next_expected_step: str,
    exit_kind: str,
    notes: str = "",
    at: str | None = None,
) -> str:
    """Render the marker body. Pure function — testable without subprocess.

    Kept in HTML-comment-fenced shape for compatibility with the
    re-entry router that scans event ``note`` text. Sagan's
    workflow_events row also carries the structured marker fields
    (step, exit_kind, next_expected_step) in ``metadata``.
    """
    if at is None:
        at = _git_head_short()
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    notes_line = f"notes: {notes}\n" if notes else ""
    return (
        "<!-- epm:step-completed v1 -->\n"
        "## Step Completed\n\n"
        f"step: {step}\n"
        f"at: {at}\n"
        f"timestamp: {timestamp}\n"
        f"next_expected_step: {next_expected_step}\n"
        f"exit_kind: {exit_kind}\n"
        f"{notes_line}"
        "<!-- /epm:step-completed -->"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--issue",
        type=int,
        required=True,
        help="experiments.number in Sagan.",
    )
    parser.add_argument(
        "--step",
        required=True,
        help="Step ID (must exist in workflow.yaml § steps), e.g. '5b' or '6c'.",
    )
    parser.add_argument(
        "--exit-kind",
        required=True,
        choices=VALID_EXIT_KINDS,
        help=(
            "Symphony §7.3 distinction: clean = continuation, parked = "
            "user-gated wait, failure-exit = error path."
        ),
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional one-line audit-trail note.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the marker body to stdout instead of posting it.",
    )
    args = parser.parse_args(argv)

    by_id = _load_workflow_steps()
    if args.step not in by_id:
        print(
            f"ERROR: step {args.step!r} is not in workflow.yaml § steps. "
            f"Known: {', '.join(sorted(by_id))}",
            file=sys.stderr,
        )
        return 2
    next_step = by_id[args.step].get("next_expected_step", "")
    if not next_step:
        print(
            f"ERROR: workflow.yaml § steps[{args.step}] has no next_expected_step",
            file=sys.stderr,
        )
        return 2

    body = build_marker_body(
        step=args.step,
        next_expected_step=str(next_step),
        exit_kind=args.exit_kind,
        notes=args.notes,
    )

    if args.dry_run:
        print(body)
        return 0

    # Post via the Sagan API. The step-completed marker is small enough
    # to fit comfortably under the 50 KB note cap.
    try:
        snapshot = sagan_state.get_experiment(args.issue)
        experiment_id = snapshot["experiment"]["id"]
        sagan_state.post_marker(
            experiment_id,
            "epm:step-completed",
            note=body,
            metadata={
                "step": args.step,
                "next_expected_step": str(next_step),
                "exit_kind": args.exit_kind,
                "notes": args.notes,
            },
        )
    except sagan_state.SaganError as exc:
        print(f"ERROR: sagan post-marker failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
