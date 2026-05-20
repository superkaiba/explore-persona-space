"""Hypothesis + kill-criterion gate. Used by clarifier and adversarial-planner.

Refuses to advance ``type:experiment`` issues whose body (or drafted plan body)
lacks both a ``Hypothesis`` section header and a ``Kill criterion`` /
``Kill criteria`` section header.

Override mechanism: a body marker ``<!-- epm:override-hypothesis-skip v1 -->``
forces a PASS-via-override (exit 3). The marker is matched after fenced-code
stripping (so a code-fenced quotation of the override syntax does not trigger
override) but before HTML-comment stripping (since the override marker IS an
HTML comment).

Exit codes (CLI):
* ``0`` — PASS (sections found, or non-experiment type).
* ``2`` — BLOCK (sections missing).
* ``3`` — PASS via override marker.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OVERRIDE_MARKER_RE = re.compile(
    r"<!--\s*epm:override-hypothesis-skip\s+v\d+\s*-->",
    re.IGNORECASE,
)

# Header / bold / bulleted-bold marker, optional numbered prefix, then keyword.
_HYPO_PAT = re.compile(
    r"(?:^|\n)\s*"
    r"(?:#{1,6}\s+|\*\*|\-\s+\*\*)"
    r"\s*(?:[\d.]+\s+)?"
    r"Hypothesis\b",
    re.MULTILINE,
)
_KILL_PAT = re.compile(
    r"(?:^|\n)\s*"
    r"(?:#{1,6}\s+|\*\*|\-\s+\*\*)"
    r"\s*(?:[\d.]+\s+)?"
    r"Kill\s+criteri(?:on|a)\b",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_fenced_code(body: str) -> str:
    """Remove ``` ... ``` fenced blocks so a Hypothesis section embedded in a
    code fence does not satisfy the gate."""
    return re.sub(r"```.*?```", "", body, flags=re.DOTALL)


def _strip_html_comments(body: str) -> str:
    """Remove ``<!-- ... -->`` blocks so a header inside an HTML comment does
    not satisfy the gate. The override marker is matched on the body BEFORE
    this stripping (after :func:`_strip_fenced_code`) so it is still
    respected."""
    return re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)


def check(
    body: str,
    *,
    labels: list[str] | None = None,
    issue_type: str = "experiment",
) -> tuple[bool, list[str], bool]:
    """Return ``(ok, missing_sections, override_used)``.

    ``ok=True`` when ``issue_type != "experiment"`` OR the override marker is
    present OR both sections are found.

    ``labels`` is accepted for forward-compatibility with caller plumbing but
    is currently unused — gate firing is keyed off ``issue_type``.
    """
    del labels  # Reserved for future label-based fast-paths.
    if issue_type != "experiment":
        return True, [], False
    fence_stripped = _strip_fenced_code(body)
    if OVERRIDE_MARKER_RE.search(fence_stripped):
        return True, [], True
    fully_stripped = _strip_html_comments(fence_stripped)
    missing: list[str] = []
    if not _HYPO_PAT.search(fully_stripped):
        missing.append("Hypothesis")
    if not _KILL_PAT.search(fully_stripped):
        missing.append("Kill criterion (or Kill criteria)")
    return (not missing, missing, False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hypothesis + kill-criterion gate for type:experiment issues."
    )
    parser.add_argument(
        "--body-file",
        help="Path to a file containing the body to check. If omitted, read stdin.",
    )
    parser.add_argument(
        "--type",
        default="experiment",
        help="Issue type. Gate is a no-op for non-experiment types.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated label list (currently advisory; reserved for future use).",
    )
    args = parser.parse_args(argv)

    body = Path(args.body_file).read_text() if args.body_file else sys.stdin.read()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]

    ok, missing, override = check(body, labels=labels, issue_type=args.type)
    if ok and override:
        print("PASS (override)", file=sys.stderr)
        return 3
    if ok:
        print("PASS", file=sys.stderr)
        return 0
    print(f"BLOCK: missing: {', '.join(missing)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
