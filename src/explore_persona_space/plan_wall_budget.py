"""Shared parser for the plan §9 ``planned_wall_h`` compute table (#2172).

The SINGLE home of both the wall-table LOCATOR (markdown pipe tables and
the #779-style HTML ``<table>`` form) and the per-cell FLOAT RULE, imported
by ``scripts/poll_pipeline.py`` (the #873 phase-ETA tripwire budget) and
``scripts/verify_plan.py`` (check c47) so the plan-time WARN and the
runtime tripwire disable can never drift (#2172 AC #4). Stdlib-only
(``re`` + ``dataclasses``) — importable from the sub-second
``verify_plan.py`` gate at no measurable startup cost.

Float rule (bounded cosmetic-prefix widening — #2172, critic round 1 Must
Fix 1): a cell parses when the only thing preceding its first number is a
run of NON-ALPHANUMERIC characters — whitespace, ``(``, ``~``, ``*``,
``+``, markdown bold stars, comparison signs. ``.match()`` anchored at
position 0, never ``.search()``: any LETTER before the first digit keeps
the cell unparseable, because a first-float-anywhere rule admits
letter-prefixed garbage (row counts like ``n=800 rows``, issue numbers,
RAM bounds — 19 such cells across 9 live plans at design time) as phantom
wall hours, invisible to every consumer precisely because all surfaces
share this parser. The LAZY prefix quantifier plus the decimal-first
``[0-9]*\\.[0-9]+`` alternative make a leading-dot decimal read correctly
(``.5`` -> 0.5, ``(.5)`` -> 0.5 — a greedy prefix would eat the dot and
yield 5.0).

Fail-safe reduce (#2172 AC #2): ANY located data cell with no parseable
number disables the WHOLE budget (``total_h is None``,
``reason == "unparseable_cell"``) — never a partial sum, because an
under-parsed budget is the one path to a tripwire false positive — while
``unparseable`` names every offending row so callers can be LOUD about
the degradation. Zero located tables / zero data rows — or a table whose
rows sum to exactly 0, preserving the legacy ``sum(rows) or None`` —
reduce to ``reason == "no_table"``: such a plan arms no tripwire and
warrants no warning. Invariant: ``reason == ""`` iff ``total_h`` is set.

Deliberately NOT fence-masked, matching the poller's raw-text scan: a
fenced example table carrying ``planned_wall_h`` with an unparseable cell
genuinely disables the live tripwire, so surfacing it at plan time is
correct, not a false positive (#2172 plan §4.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WALL_HEADER_TOKEN = "planned_wall_h"

# THE float rule (#2172 §4.1). Lazy NON-ALPHANUMERIC prefix only; the
# decimal-bearing alternative is listed FIRST so ``.5`` reads 0.5 (and
# ``10.5`` reads 10.5 — integer-first would truncate it to 10.0).
_WALL_CELL_RE = re.compile(r"[^0-9A-Za-z]*?([0-9]*\.[0-9]+|[0-9]+)")

# A markdown |---|:---:|---| separator row: every pipe-split cell is only
# whitespace / dashes / colons.
_MD_SEPARATOR_CELL_RE = re.compile(r"[\s:\-]*")

# Offending-row report texts are truncated to this many characters.
_ROW_TEXT_MAX = 120


@dataclass(frozen=True)
class LocatedWallCell:
    """One located ``planned_wall_h`` data cell (or a too-short row).

    ``cell`` is the raw cell text the float rule parses (``""`` for a
    short row — there is no cell to parse); ``row_text`` is the bounded
    REPORTING text (the markdown row / the HTML cell; the row inner-HTML
    for a short HTML row), pre-truncated to 120 chars.
    """

    cell: str
    row_text: str
    fmt: str  # "markdown" | "html"
    short_row: bool = False


@dataclass(frozen=True)
class UnparseableWallCell:
    """One located data cell the float rule rejects (#2172 AC #2)."""

    row_text: str  # the offending row (markdown) or cell (HTML), truncated to 120 chars
    reason: str  # "no_float" | "short_row"
    fmt: str  # "markdown" | "html"


@dataclass(frozen=True)
class PlanWallBudget:
    """The reduced §9 wall budget one plan document yields."""

    total_h: float | None  # None => the caller disables the tripwire
    rows: tuple[float, ...]  # every parsed per-row value
    unparseable: tuple[UnparseableWallCell, ...]  # populated even when total_h is None
    reason: str  # "" | "no_table" | "unparseable_cell"; "" iff total_h is not None


def parse_wall_cell(cell: str) -> float | None:
    """Parse one ``planned_wall_h`` cell under the bounded float rule.

    Accepts a number after an optional cosmetic (non-alphanumeric) prefix:
    ``1.5``, ``(1.5)``, ``~1.5``, ``**~7.1**``, ``+0.05``, ``.5`` -> 0.5,
    ``2.`` -> 2.0, ``3 (async, off-GPU)`` -> 3.0. Returns ``None`` for a
    cell with no number the rule will trust — ``TBD``, ``N/A``, an empty
    cell, and ANY letter before the first digit (``n=800 rows``,
    ``issue #464`` — the loud fail-safe path, never a phantom wall hour).
    """
    m = _WALL_CELL_RE.match(cell)  # ANCHORED — .match(), never .search()
    return float(m.group(1)) if m else None


def _md_wall_cells(plan_text: str) -> list[LocatedWallCell]:
    """Locate every markdown-table ``planned_wall_h`` column cell.

    Table-scoped: only rows FOLLOWING a ``|``-prefixed header line that
    contains ``planned_wall_h`` are scanned, with the value-column index
    DERIVED from that header's cell position (never a hardcoded ordinal).
    A ``|---|:---:|`` separator row is skipped; a row with fewer cells
    than the header column index is collected as ``short_row``. Locator
    lifted verbatim from ``poll_pipeline._md_planned_wall_rows`` (#873),
    collecting cells instead of raising.
    """
    out: list[LocatedWallCell] = []
    lines = plan_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not (line.lstrip().startswith("|") and WALL_HEADER_TOKEN in line):
            i += 1
            continue
        header_cells = line.split("|")
        col = next(idx for idx, c in enumerate(header_cells) if WALL_HEADER_TOKEN in c)
        j = i + 1
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            cells = lines[j].split("|")
            j += 1
            if all(_MD_SEPARATOR_CELL_RE.fullmatch(c) for c in cells):
                continue  # the |---|---| separator row
            row_text = lines[j - 1][:_ROW_TEXT_MAX]
            if col >= len(cells):
                out.append(LocatedWallCell("", row_text, "markdown", short_row=True))
                continue
            out.append(LocatedWallCell(cells[col], row_text, "markdown"))
        i = j
    return out


def _html_wall_cells(plan_text: str) -> list[LocatedWallCell]:
    """Locate every HTML-table ``planned_wall_h`` column cell.

    The row scan is SCOPED to each ``<table>`` element whose ``<th>`` row
    contains ``planned_wall_h`` (never a document-wide ``<td>`` scan), and
    the value-column index is DERIVED from the ``<th>`` position (parity
    with the markdown path). Locator lifted verbatim from
    ``poll_pipeline._html_planned_wall_rows`` (#873), collecting cells
    instead of raising.
    """
    out: list[LocatedWallCell] = []
    for tbl_m in re.finditer(r"<table\b[^>]*>(.*?)</table>", plan_text, re.IGNORECASE | re.DOTALL):
        tbl = tbl_m.group(1)
        ths = re.findall(r"<th\b[^>]*>(.*?)</th>", tbl, re.IGNORECASE | re.DOTALL)
        col = next((idx for idx, th in enumerate(ths) if WALL_HEADER_TOKEN in th), None)
        if col is None:
            continue  # a table without the header never contributes
        for tr_m in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", tbl, re.IGNORECASE | re.DOTALL):
            tds = re.findall(r"<td\b[^>]*>(.*?)</td>", tr_m.group(1), re.IGNORECASE | re.DOTALL)
            if not tds:
                continue  # the <th>-only header row
            if col >= len(tds):
                out.append(
                    LocatedWallCell("", tr_m.group(1)[:_ROW_TEXT_MAX], "html", short_row=True)
                )
                continue
            cell = re.sub(r"<[^>]+>", "", tds[col])
            out.append(LocatedWallCell(cell, cell[:_ROW_TEXT_MAX], "html"))
    return out


def locate_wall_cells(plan_text: str) -> list[LocatedWallCell]:
    """Every ``planned_wall_h`` data cell across ALL located tables.

    Markdown pipe tables first, then HTML tables (multi-stage plans sum
    across every table carrying the header — a single-header parse
    under-counts and false-fires, the #479 shape). Exposed so callers
    (the behavior-preservation corpus test, #2172 §6) can replay float
    rules over the exact located cell set.
    """
    return _md_wall_cells(plan_text) + _html_wall_cells(plan_text)


def parse_plan_wall_budget(plan_text: str) -> PlanWallBudget:
    """Reduce ``plan_text`` to its §9 wall budget (see module docstring).

    ALL offenders are collected (not first-offender-only) so the callers'
    bounded reports can name them; ``rows`` carries every value that DID
    parse even when the total is disabled, so the runtime disable note
    can state how many parseable rows were discarded (#2172 AC #5).
    """
    rows: list[float] = []
    bad: list[UnparseableWallCell] = []
    for lc in locate_wall_cells(plan_text):
        if lc.short_row:
            bad.append(UnparseableWallCell(lc.row_text, "short_row", lc.fmt))
            continue
        value = parse_wall_cell(lc.cell)
        if value is None:
            bad.append(UnparseableWallCell(lc.row_text, "no_float", lc.fmt))
        else:
            rows.append(value)
    if bad:
        return PlanWallBudget(None, tuple(rows), tuple(bad), "unparseable_cell")
    total = sum(rows) or None  # a zero-sum table arms no tripwire (legacy contract)
    if total is None:
        return PlanWallBudget(None, tuple(rows), (), "no_table")
    return PlanWallBudget(total, tuple(rows), (), "")
