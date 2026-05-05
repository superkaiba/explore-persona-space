"""Failure-class routing for `epm:failure` markers.

The `/issue` skill Step 7 calls this helper to decide whether to re-spawn
the experimenter (infra failures: OOM, NCCL, library tracebacks, ...) or
the experiment-implementer (code failures: tracebacks from our code,
AssertionError, ...).

The routing rules are documented in `.claude/skills/issue/failure_patterns.md`;
this module is the executable form of that file. Adding a pattern: edit
the regex list below AND `failure_patterns.md` (both — the doc is the
authoritative spec; this module mirrors it).
"""

from __future__ import annotations

import re
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
