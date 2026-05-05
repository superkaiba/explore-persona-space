"""Failure-class routing for `epm:failure` markers.

The `/issue` skill Step 7 invokes this helper as a subprocess
(`python scripts/failure_classifier.py --body <body> [--log <path>]`)
to decide whether to re-spawn the experimenter (infra failures: OOM,
NCCL, library tracebacks, ...) or the experiment-implementer (code
failures: tracebacks from our code, AssertionError, ...).

This module is the SINGLE SOURCE OF TRUTH for the regex pattern list.
`.claude/skills/issue/failure_patterns.md` is a human-readable mirror
that documents the same patterns for agents/reviewers; the markdown
file MUST stay in sync with the regex list below, but it is NOT
consulted at runtime — the SKILL Step 7 shells out to this script.

CLI:
  python scripts/failure_classifier.py --body <body-text>
  python scripts/failure_classifier.py --body <body-text> --log <path>
  cat body.txt | python scripts/failure_classifier.py --body -

Stdout: a single line, ``infra`` or ``code``. Exit 0 on success.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

FailureClass = Literal["infra", "code"]

# Infra log patterns (regex, case-insensitive). Source of truth:
# `.claude/skills/issue/failure_patterns.md`. Any match → route as `infra`.
INFRA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OOM-killer|Killed\b", re.IGNORECASE),
    re.compile(r"No space left on device|ENOSPC|disk full", re.IGNORECASE),
    re.compile(r"NCCL (timeout|error)", re.IGNORECASE),
    re.compile(
        r"SSH connection refused|No route to host|Connection timed out",
        re.IGNORECASE,
    ),
    re.compile(r"401 Unauthorized|gated repo", re.IGNORECASE),
    re.compile(r"RuntimeError: CUDA error", re.IGNORECASE),
    re.compile(r"Failed to initialize.*vllm", re.IGNORECASE),
    re.compile(
        r"Traceback.*\b(vllm|transformers|peft|trl|torch|xformers)/",
        re.IGNORECASE | re.DOTALL,
    ),
]

# Field-line regex: matches a leading "failure_class: <value>" line
# (allowing surrounding whitespace and case-insensitive "infra"/"code").
FIELD_LINE = re.compile(
    r"^\s*failure_class\s*:\s*(infra|code)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def classify_failure(body: str) -> FailureClass:
    """Return ``"infra"`` or ``"code"`` for an `epm:failure` body.

    Routing precedence:
    1. Explicit ``failure_class:`` field on the first non-blank line of
       the body (or any leading metadata block) wins.
    2. Otherwise, scan the body against the infra log-pattern list. Any
       match → ``"infra"``.
    3. Otherwise, default to ``"code"`` (conservative — the implementer
       round catches more than the experimenter respawn round).
    """
    field = FIELD_LINE.search(body)
    if field is not None:
        return field.group(1).lower()  # type: ignore[return-value]
    for rx in INFRA_PATTERNS:
        if rx.search(body):
            return "infra"
    return "code"


# Cap the amount of log we feed to the regex scanner. The patterns are
# anchored only by content (not start-of-string), so unbounded log files
# would just slow us down without changing the verdict.
_LOG_TAIL_BYTES = 200 * 1024  # 200 KB is well past 200 lines for any sane log


def _load_log_tail(path: Path) -> str:
    """Read the last ~200 KB of a log file. Returns "" if path is missing."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        # Surface the real error — the SKILL Step 7 wraps this in a respawn
        # cap so a missing log shouldn't silently default to `code`.
        sys.stderr.write(f"failure_classifier: cannot stat {path}: {exc}\n")
        return ""
    with path.open("rb") as fh:
        if size > _LOG_TAIL_BYTES:
            fh.seek(size - _LOG_TAIL_BYTES)
        return fh.read().decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print the failure_class verdict, exit 0 on success."""
    parser = argparse.ArgumentParser(
        description="Classify an epm:failure body as `infra` or `code`. "
        "See scripts/failure_classifier.py module docstring for the rules.",
    )
    parser.add_argument(
        "--body",
        required=True,
        help="failure body text. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="optional path to a log file; its tail is concatenated with --body "
        "before classification, so library-traceback infra patterns can match.",
    )
    args = parser.parse_args(argv)

    body = sys.stdin.read() if args.body == "-" else args.body

    if args.log:
        body = body + "\n" + _load_log_tail(Path(args.log))

    print(classify_failure(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
