"""task_workflow_why_gate — shared constants + helper for the
``## Why this experiment`` gate.

Single source of truth for the four labels, the application enum, the
labeled-line regex, the substance-floor character count, and the
frontmatter sentinel key. Both ``scripts/verify_task_body.py`` (check
#12) and ``scripts/task.py`` (``cmd_create`` gate enforced via
``_enforce_why_this_experiment_gate``) import from here.

Before this module existed, the constants lived in both call sites with
a "edit both together if changing" comment. That comment was a smell
— two private sources of truth that any drift between would silently
break the gate. ``find_why_section`` is the fence-state-aware section
walker that both sites call.

Public surface:

* ``WHY_SECTION_NAME`` — H2 heading text (without the ``## `` prefix).
* ``WHY_LINE_LABELS`` — the four required labeled lines, in canonical
  order.
* ``APPLICATION_ENUM`` — the five allowed values for the ``Application``
  line + the ``application:`` frontmatter field.
* ``MIN_WHY_LINE_CHARS`` — minimum chars of substance after each
  ``**Label:**`` prefix. Picked to reject one-word non-answers while
  admitting most real one-sentence answers.
* ``LEGACY_WHY_SENTINEL_KEY`` — frontmatter key (``legacy_why_unset``)
  the migration script sets on bodies authored before the gate landed.
  When ``True``, the gate is skipped for that body.
* ``WHY_LINE_RE`` — compiled regex matching one ``**Label:** value``
  line, optionally bullet-prefixed.
* ``WHY_GATED_KINDS`` — task kinds whose ``task.py new`` invocation
  fires the gate. ``analysis`` is intentionally exempt (read-only
  workflows over existing artifacts).
* ``WhySection`` — dataclass returned by ``find_why_section``.
* ``find_why_section(body) -> WhySection | None`` — fence-state-aware
  section walker. Returns the section's labeled-line values + start /
  end line indices, or ``None`` if the section is missing or fenced
  out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─── Constants ─────────────────────────────────────────────────────────────

WHY_SECTION_NAME = "Why this experiment"

WHY_LINE_LABELS: tuple[str, ...] = (
    "Application",
    "Decision this changes",
    "Expected outcome + branches",
    "What gets cut if we run this",
)

APPLICATION_ENUM: tuple[str, ...] = ("detect", "predict", "defend", "audit", "infra")

# Minimum chars of substance required AFTER the `**Label:**` prefix on
# each labeled line. Picked to reject one-word non-answers like
# "**Decision this changes:** TBD." while admitting most real one-sentence
# answers ("**Decision this changes:** whether to ship persona-axis
# steering as the default defense in #137." ≈ 90 chars).
MIN_WHY_LINE_CHARS = 40

LEGACY_WHY_SENTINEL_KEY = "legacy_why_unset"

# Task kinds whose `task.py new` invocation fires the gate. `analysis` is
# intentionally exempt — analysis tasks read existing artifacts and
# rarely commit GPU, so the friction does not pay for itself. Keep in
# sync with the PM session's Mode 5 pre-spawn check and the /issue Step 0
# gate.
WHY_GATED_KINDS: frozenset[str] = frozenset({"experiment", "survey", "infra"})

# Match `**Label:** value`, optionally bullet-prefixed (`- `, `* `).
WHY_LINE_RE: re.Pattern[str] = re.compile(r"^\s*[-*]?\s*\*\*\s*([^*]+?)\s*:\s*\*\*\s*(.*)$")

# Fence prefixes whose lines toggle the "inside a fenced code block"
# state during section walks. CommonMark info-string fences (e.g.
# ```python or ~~~text) and bare openers both start with one of these
# three-character prefixes.
_FENCE_PREFIXES: tuple[str, ...] = ("```", "~~~")


def _is_fence_line(stripped: str) -> bool:
    """Return True if ``stripped`` is a fenced-code-block delimiter line."""
    return any(stripped.startswith(p) for p in _FENCE_PREFIXES)


# ─── Section walker ────────────────────────────────────────────────────────


@dataclass
class WhySection:
    """Result of ``find_why_section``.

    ``line_values`` keys are every ``WHY_LINE_LABELS`` entry. A value of
    ``None`` means the labeled line was absent from the section. A
    non-``None`` value is the verbatim text after the ``**Label:**``
    prefix (stripped of leading / trailing whitespace).
    """

    start_line: int
    """First body line (0-indexed) INSIDE the section, i.e. the line after
    the ``## Why this experiment`` heading."""

    end_line: int
    """One past the last body line inside the section."""

    line_values: dict[str, str | None] = field(default_factory=dict)
    """Map of canonical label → first-occurrence value (or ``None`` if
    that label never appeared as a labeled line)."""


def find_why_section(body: str) -> WhySection | None:
    """Locate the ``## Why this experiment`` section and parse its labeled
    lines.

    Fenced-code-aware: a ``## Why this experiment`` heading pasted inside
    a ``` ... ``` or ``~~~ ... ~~~`` block is ignored (the gate cannot
    be satisfied by content trapped inside a code fence). The fence
    matcher accepts both triple-backtick and triple-tilde delimiters
    with an optional info string (``` ```python `` / ``~~~text``).

    Returns ``None`` when the section is absent or only appears inside
    code fences. Returns a ``WhySection`` otherwise — duplicate labeled
    lines within a section keep the first occurrence's value; duplicate
    ``## Why this experiment`` H2 sections in the same body are NOT
    handled here (the caller decides whether duplicates are a FAIL).
    """
    lines = body.splitlines()
    in_fence = False
    section_start: int | None = None
    section_end: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_fence_line(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## ") and not stripped.startswith("### "):
            heading = stripped[3:].strip()
            if section_start is None and heading.casefold() == WHY_SECTION_NAME.casefold():
                section_start = i + 1
                continue
            if section_start is not None and section_end is None:
                # Closing H2 ends our section.
                section_end = i
                break

    if section_start is None:
        return None
    if section_end is None:
        section_end = len(lines)

    # Parse labeled lines inside the section. Fenced code blocks inside
    # the section are tracked too — a labeled line trapped inside a
    # fence does not satisfy the gate.
    line_values: dict[str, str | None] = {label: None for label in WHY_LINE_LABELS}
    inner_in_fence = False
    for line in lines[section_start:section_end]:
        stripped = line.strip()
        if _is_fence_line(stripped):
            inner_in_fence = not inner_in_fence
            continue
        if inner_in_fence:
            continue
        m = WHY_LINE_RE.match(stripped)
        if not m:
            continue
        label = m.group(1).strip()
        value = m.group(2).strip()
        for canonical in WHY_LINE_LABELS:
            if label.casefold() == canonical.casefold() and line_values[canonical] is None:
                line_values[canonical] = value
                break

    return WhySection(start_line=section_start, end_line=section_end, line_values=line_values)


def count_why_sections(body: str) -> int:
    """Return the number of (non-fenced) ``## Why this experiment`` H2
    headings in ``body``.

    Used by ``verify_task_body`` check #12 to FAIL on duplicate
    sections (a body-discipline smell where an author appended a second
    why-block instead of editing the first).
    """
    lines = body.splitlines()
    in_fence = False
    count = 0
    for line in lines:
        stripped = line.strip()
        if _is_fence_line(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## ") and not stripped.startswith("### "):
            heading = stripped[3:].strip()
            if heading.casefold() == WHY_SECTION_NAME.casefold():
                count += 1
    return count
