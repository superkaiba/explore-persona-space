"""Tests for src/explore_persona_space/plan_wall_budget.py (#2172).

The shared §9 ``planned_wall_h`` parser — ONE home for the wall-table
locator + the bounded cosmetic-prefix float rule, imported by
``scripts/poll_pipeline.py`` (the #873 phase-ETA tripwire budget) and
``scripts/verify_plan.py`` (check c47). These tests pin:

* ``parse_wall_cell`` — the float rule in BOTH directions: the cosmetic
  prefixes it ACCEPTS with their exact values (incl. the leading-dot pair
  ``.5`` / ``(.5)`` -> 0.5, critic round 1 Concern 3), and the
  letter-prefixed garbage class it must keep REJECTING (row counts, issue
  numbers, RAM bounds — the round's regression anchor: a widening to
  first-float-anywhere would turn these into phantom wall hours, critic
  round 1 Must Fix 1);
* the behavior-preservation invariant — for every fixture the OLD anchored
  leading-float rule accepted, the new rule returns the SAME value;
* ``parse_plan_wall_budget`` — the fail-safe reduce (AC #1: a ``(1.5)``
  cell sums and the budget stays armed; AC #2: a no-float cell disables
  the WHOLE budget and names the row) plus every locator property the old
  poller tests pinned (multi-table sum, header-derived value column, HTML
  scoping, separator-row skip, short rows, ``no_table``).
"""

# ruff: noqa: RUF001
# The ACCEPT fixtures below include the literal unicode cosmetic prefixes
# observed in the persisted-plan corpus (comparison signs, approx signs,
# en-dash ranges); substituting ASCII lookalikes would defeat the fixtures.

from __future__ import annotations

import re

import pytest

from explore_persona_space.plan_wall_budget import (
    PlanWallBudget,
    UnparseableWallCell,
    locate_wall_cells,
    parse_plan_wall_budget,
    parse_wall_cell,
)

# ── parse_wall_cell: the float rule, both directions ─────────────────────────

# (cell, expected value) — every shape drawn from the plan §4.1 accept list /
# the corpus's realized cosmetic prefixes.
ACCEPT: list[tuple[str, float]] = [
    ("1.5", 1.5),
    ("0", 0.0),
    ("  3 (async, off-GPU)", 3.0),
    ("(1.5)", 1.5),  # the #2163 offender class (AC #1)
    ("~1.5", 1.5),
    ("1.5-2", 1.5),
    ("1.5 (conditional)", 1.5),
    ("**~7.1**", 7.1),
    ("≤24 (SLA)", 24.0),
    ("≈3.6–3.8", 3.6),
    ("<0.1", 0.1),
    ("+0.05", 0.05),
    ("16", 16.0),
    ("10.5", 10.5),  # decimal-first alternation: never truncated to 10.0
    ("2.", 2.0),  # trailing dot backtracks to the integer alternative
    (".5", 0.5),  # leading-dot decimal (lazy prefix + decimal-first)
    ("(.5)", 0.5),  # a greedy prefix would eat the dot and yield 5.0
]

# The REJECT class is the round's regression anchor: a letter before the
# first digit MUST stay unparseable (=> the loud fail-safe path), never a
# number — these shapes are verbatim from the corpus garbage class the
# bounded rule exists to keep out (#529 prose rows, #1902 row count, #1887
# RAM bound, check-id strings).
REJECT: list[str] = [
    "TBD",
    "N/A",
    "—",
    "",
    "n=800 rows",
    "issue #464",
    "see P6",
    "check 12 (c12_battery_multiplier)",
    "peak RSS 16 GB",
    "overlapped",
    "depends on chosen Q",
]


@pytest.mark.parametrize(("cell", "expected"), ACCEPT)
def test_parse_wall_cell_accepts(cell: str, expected: float) -> None:
    assert parse_wall_cell(cell) == pytest.approx(expected)


@pytest.mark.parametrize("cell", REJECT)
def test_parse_wall_cell_rejects(cell: str) -> None:
    assert parse_wall_cell(cell) is None


# The OLD anchored leading-float rule (poll_pipeline._LEADING_FLOAT_RE before
# #2172), replayed verbatim for the behavior-preservation invariant.
_OLD_LEADING_FLOAT_RE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)")


@pytest.mark.parametrize("cell", [c for c, _ in ACCEPT] + REJECT)
def test_old_rule_values_preserved(cell: str) -> None:
    """Behavior-preservation invariant (kill criterion 1, fixture half).

    For every cell the OLD rule accepted, the new rule returns the SAME
    float — an old-accepted cell has a whitespace-only prefix (a subset of
    the new non-alphanumeric prefix class) and a digit-leading value, on
    which the two value regexes agree. The corpus half of the invariant is
    ``tests/test_verify_plan.py``'s persisted-plan scan.
    """
    m = _OLD_LEADING_FLOAT_RE.match(cell)
    if m is None:
        return  # old rule rejected it — nothing to preserve
    assert parse_wall_cell(cell) == pytest.approx(float(m.group(1)))


# ── parse_plan_wall_budget: reduce + locator properties ──────────────────────

# The planner-section-reference §9 exemplar table shape (markdown pipe form),
# lifted from the pre-#2172 poller tests.
MD_TABLE = """\
## 9. Compute

| component | planned_wall_h | planned_gpu_h | parallelism | basis |
|---|---|---|---|---|
| smoke-phase per-cell train | 0.5 | 0.5 | TP=1 | "matched to #382 round-2" |
| sweep all-cells train | 16 | 64 | 4x H100 ZeRO-3 across 8 cells | "16h x 8 cells / 4 GPU" |
| eval all-cells generation | 2 | 2 | TP=1 | "vLLM batched" |
"""

# The #779-style HTML form (single <tr><th> header row, suffixed numeric cell).
HTML_TABLE = """\
<h3>Per-component compute-projection table</h3>
<table>
<tr><th>component</th><th>planned_wall_h</th><th>planned_gpu_h</th><th>basis</th></tr>
<tr><td>Arm B/C corpus gen</td><td>4.5</td><td>4.5</td><td>72k short rollouts</td></tr>
<tr><td>Batch-API judge</td><td>3 (async, off-GPU)</td><td>0</td><td>batch precedent</td></tr>
</table>
"""


def test_budget_markdown_table_sums() -> None:
    budget = parse_plan_wall_budget(MD_TABLE)
    assert budget.total_h == pytest.approx(18.5)
    assert budget.rows == (0.5, 16.0, 2.0)
    assert budget.unparseable == () and budget.reason == ""


def test_budget_parenthesized_cell_keeps_budget_armed() -> None:
    """AC #1: a ``(1.5)`` conditional cell CONTRIBUTES 1.5 — the tripwire
    stays armed (pre-#2172 this one cell discarded all 13 sibling rows on
    #2163 and disarmed the whole run)."""
    table = MD_TABLE.replace("| eval all-cells generation | 2 |", "| conditional gpu | (1.5) |")
    budget = parse_plan_wall_budget(table)
    assert budget.total_h == pytest.approx(18.0)  # 0.5 + 16 + 1.5
    assert budget.reason == "" and budget.unparseable == ()


def test_budget_no_float_cell_fails_safe_and_names_row() -> None:
    """AC #2: ONE no-float cell -> ``total_h is None`` (never a partial
    sum) with the offending row NAMED in ``unparseable`` and the parseable
    rows still counted (the AC #5 note reports how many were discarded)."""
    bad = MD_TABLE.replace("| sweep all-cells train | 16 |", "| sweep all-cells train | TBD |")
    budget = parse_plan_wall_budget(bad)
    assert budget.total_h is None
    assert budget.reason == "unparseable_cell"
    (cell,) = budget.unparseable
    assert isinstance(cell, UnparseableWallCell)
    assert "sweep all-cells train" in cell.row_text and "TBD" in cell.row_text
    assert cell.reason == "no_float" and cell.fmt == "markdown"
    assert budget.rows == (0.5, 2.0)  # the discarded-but-parseable rows

    bad_html = HTML_TABLE.replace("<td>4.5</td>", "<td>see prose</td>", 1)
    hbudget = parse_plan_wall_budget(bad_html)
    assert hbudget.total_h is None and hbudget.reason == "unparseable_cell"
    (hcell,) = hbudget.unparseable
    assert hcell.fmt == "html" and "see prose" in hcell.row_text


def test_budget_short_row_fails_safe() -> None:
    """A data row with fewer cells than the header column index is a
    ``short_row`` offender — same global fail-safe as a no-float cell."""
    # NB: no trailing pipe — `| truncated |` would split into 3 cells (the
    # last one empty, a no_float offender); the SHORT row must end before
    # the header-derived value column.
    short = "| component | planned_wall_h | basis |\n|---|---|---|\n| a | 5 | b |\n| truncated\n"
    budget = parse_plan_wall_budget(short)
    assert budget.total_h is None and budget.reason == "unparseable_cell"
    (cell,) = budget.unparseable
    assert cell.reason == "short_row" and "truncated" in cell.row_text


def test_budget_html_table() -> None:
    budget = parse_plan_wall_budget(HTML_TABLE)
    assert budget.total_h == pytest.approx(7.5)
    assert budget.rows == (4.5, 3.0)


def test_budget_two_tables_summed() -> None:
    """#479-style Stage 1 + Stage 2: rows from BOTH planned_wall_h tables
    contribute (a single-header parse under-counts and false-fires); mixed
    markdown + HTML documents sum too."""
    two_stage = (
        "### Stage 1\n\n"
        "| component | planned_wall_h | basis |\n|---|---|---|\n| s1 train | 4 | b |\n"
        "\n### Stage 2\n\n"
        "| component | planned_wall_h | basis |\n|---|---|---|\n| s2 eval | 1.5 | b |\n"
    )
    assert parse_plan_wall_budget(two_stage).total_h == pytest.approx(5.5)
    assert parse_plan_wall_budget(MD_TABLE + "\n" + HTML_TABLE).total_h == pytest.approx(26.0)


def test_budget_html_reordered_columns() -> None:
    """planned_wall_h NOT the 2nd column -> the header-DERIVED index sums
    the right column (never a hardcoded ordinal)."""
    reordered = """\
<table>
<tr><th>component</th><th>planned_gpu_h</th><th>planned_wall_h</th></tr>
<tr><td>train</td><td>64</td><td>16</td></tr>
<tr><td>eval</td><td>2</td><td>2.5</td></tr>
</table>
"""
    assert parse_plan_wall_budget(reordered).total_h == pytest.approx(18.5)


def test_budget_html_scoped_to_owning_table() -> None:
    """An UNRELATED HTML table in the same document (numeric 2nd column, no
    planned_wall_h header) contributes NOTHING — the row scan is scoped to
    the owning table, never a document-wide <td> scan."""
    doc = (
        HTML_TABLE
        + """
<table>
<tr><th>condition</th><th>n_seeds</th></tr>
<tr><td>c1</td><td>500</td></tr>
<tr><td>c2</td><td>1000</td></tr>
</table>
"""
    )
    assert parse_plan_wall_budget(doc).total_h == pytest.approx(7.5)


def test_budget_no_table_reason() -> None:
    for doc in ("## 9. Compute\n\nno table here\n", ""):
        budget = parse_plan_wall_budget(doc)
        assert budget == PlanWallBudget(None, (), (), "no_table")
    # A table WITHOUT a planned_wall_h header is not a located table.
    other = "| component | est. wall_h | basis |\n|---|---|---|\n| a | 5 | b |\n"
    assert parse_plan_wall_budget(other).reason == "no_table"


def test_budget_zero_sum_reduces_to_no_table() -> None:
    """All rows parse to 0 -> ``total_h is None`` with ``no_table`` (the
    legacy ``sum(rows) or None`` contract: a zero budget arms no tripwire
    and warrants no warning)."""
    zero = "| component | planned_wall_h | basis |\n|---|---|---|\n| a | 0 | b |\n"
    budget = parse_plan_wall_budget(zero)
    assert budget.total_h is None and budget.reason == "no_table"
    assert budget.rows == (0.0,) and budget.unparseable == ()


def test_locate_wall_cells_yields_raw_cells() -> None:
    """``locate_wall_cells`` exposes the exact located cell set (the corpus
    re-scan replays OLD vs NEW float rules over it, #2172 §6)."""
    cells = locate_wall_cells(MD_TABLE + "\n" + HTML_TABLE)
    assert [c.fmt for c in cells] == ["markdown"] * 3 + ["html"] * 2
    assert [c.cell.strip() for c in cells] == ["0.5", "16", "2", "4.5", "3 (async, off-GPU)"]
    assert not any(c.short_row for c in cells)
