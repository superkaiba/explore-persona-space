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

# Infra log patterns (regex, case-insensitive). This module is the source of
# truth; mirrored by `.claude/skills/issue/failure_patterns.md`. Any match →
# route as `infra`.
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
    # §2 watchdog reasons. The watchdog already sets `failure_class: infra`
    # explicitly in its body, so the FIELD_LINE precedence catches it
    # first; these patterns are belt-and-suspenders fallback for the case
    # where a body is reformatted upstream and loses the field line.
    re.compile(r"^\s*reason:\s*stall\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*reason:\s*probe_unreachable\b", re.IGNORECASE | re.MULTILINE),
]

# Field-line regex: matches a leading "failure_class: <value>" line
# (allowing surrounding whitespace and case-insensitive "infra"/"code").
FIELD_LINE = re.compile(
    r"^\s*failure_class\s*:\s*(infra|code)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# A CUDA OOM is normally transient infra (leaked process, fragmentation —
# respawn fixes it). EXCEPT: when the torch OOM message lists 2+ sibling
# "Process NNN has X GiB memory in use" entries on the failing device
# during a parallel fan-out, the train cells were CO-LOCATED on one
# physical GPU — a deterministic GPU-pinning bug in the launch path
# (e.g. a per-process `--gpu` pin that is dead code), so respawning on
# verified-clean GPUs hits the identical OOM. Route to `code` so the
# implementer fixes the pinning. A SINGLE sibling entry stays `infra`
# (one leaked process from a prior run — kill + respawn is the right
# move). Surfaced by task #557 (2026-06-10): attempt 1 was misdiagnosed
# as leaked-process infra and attempt 2 OOMed identically.
CUDA_OOM = re.compile(r"CUDA out of memory", re.IGNORECASE)
OOM_SIBLING_PROCESS = re.compile(
    r"Process \d+ has [\d.]+ [KMG]iB memory in use",
    re.IGNORECASE,
)


def _is_colocation_oom(body: str) -> bool:
    """True when a CUDA OOM lists >= 2 sibling-process memory entries."""
    if not CUDA_OOM.search(body):
        return False
    return len(OOM_SIBLING_PROCESS.findall(body)) >= 2


# torch's DataLoader wraps a worker-side exception with a header like:
#   RuntimeError: Caught RuntimeError in DataLoader worker process 0.
# followed by an "Original Traceback (most recent call last):" block whose
# frames belong to whatever raised inside the worker. The outer frames are
# always under torch/ (worker.py, _utils/, etc.), so the generic
# library-traceback infra pattern would route us-code raises to `infra`
# unless we look INSIDE the wrapped block. See the workflow-fix-on-bug
# candidate emitted by /issue 480.
DATALOADER_WRAP = re.compile(
    r"Caught\s+\w+\s+in\s+DataLoader worker",
    re.IGNORECASE,
)
ORIGINAL_TB_SPLIT = re.compile(r"Original Traceback", re.IGNORECASE)

# Our-code frame regex: matches a Python traceback "File ..." line whose
# path is under our source/scripts trees. Used to detect that the deepest
# frame inside a wrapped Original Traceback is our code (not a library).
OUR_CODE_FRAME = re.compile(
    r'File\s+"[^"]*/(?:src/explore_persona_space|scripts)/',
    re.IGNORECASE,
)


def classify_failure(body: str) -> FailureClass:
    """Return ``"infra"`` or ``"code"`` for an `epm:failure` body.

    Routing precedence:
    1. Explicit ``failure_class:`` field on the first non-blank line of
       the body (or any leading metadata block) wins.
    2. Co-located parallel-cell OOM: a ``CUDA out of memory`` body
       listing 2+ sibling ``Process NNN has X GiB memory in use``
       entries means parallel fan-out cells shared one physical GPU — a
       deterministic GPU-pinning bug, NOT transient infra — return
       ``"code"``. A single sibling entry stays on the infra path (one
       leaked process; kill + respawn fixes it). See task #557.
    3. If the body shows a torch DataLoader worker wrap (``Caught <Error>
       in DataLoader worker``), classify on the WRAPPED Original
       Traceback block, not the outer torch frames: when the wrapped
       block contains an our-code frame (``src/explore_persona_space/``
       or ``scripts/``), return ``"code"``; otherwise run the normal
       infra-pattern scan against the WRAPPED text only. The outer torch
       frames are always library code (worker.py, _utils/, ...) and
       routing on them would misclassify an our-code raise as ``infra``.
    4. Otherwise, scan the body against the infra log-pattern list. Any
       match → ``"infra"``.
    5. Otherwise, default to ``"code"`` (conservative — the implementer
       round catches more than the experimenter respawn round).
    """
    field = FIELD_LINE.search(body)
    if field is not None:
        return field.group(1).lower()  # type: ignore[return-value]

    if _is_colocation_oom(body):
        return "code"

    if DATALOADER_WRAP.search(body):
        # Split on the "Original Traceback" header and classify on the
        # WRAPPED block. Fall back to the post-split tail if there is no
        # explicit header (some torch versions wrap without it).
        parts = ORIGINAL_TB_SPLIT.split(body, maxsplit=1)
        wrapped = parts[1] if len(parts) == 2 else body
        if OUR_CODE_FRAME.search(wrapped):
            return "code"
        for rx in INFRA_PATTERNS:
            if rx.search(wrapped):
                return "infra"
        return "code"

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
