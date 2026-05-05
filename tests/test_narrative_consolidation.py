"""Stage 3 verifier: structural assertions on multi-issue narrative bodies.

Single-experiment clean-results have no `Source-issues:` line and the new
check is a no-op for them. Narrative consolidations MUST list ≥2 children
and retain at least one hero figure URL.
"""

from __future__ import annotations

from scripts.verify_clean_result import Report, check_narrative_consolidation


def test_no_source_issues_line_no_op():
    """Single-experiment clean-result: check produces no narrative_* entries."""
    body = "## TL;DR\n\n### Background\nNo consolidation here.\n"
    report = Report()
    check_narrative_consolidation(body, report)
    names = {r.name for r in report.results}
    assert "narrative_sources" not in names
    assert "narrative_figure" not in names


def test_narrative_with_two_sources_and_figure_passes():
    body = (
        "## TL;DR\n"
        "Source-issues: #91, #109\n"
        "Supersedes: #75\n"
        "\n"
        "### Background\n"
        "Both children measure convergence training capacity.\n"
        "\n"
        "### Results\n"
        "![hero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abc123/figures/x.png)\n"
    )
    report = Report()
    check_narrative_consolidation(body, report)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["narrative_sources"] == "PASS"
    assert statuses["narrative_figure"] == "PASS"


def test_narrative_with_one_source_fails():
    body = "## TL;DR\nSource-issues: #91\n\n![hero](figures/x.png)\n"
    report = Report()
    check_narrative_consolidation(body, report)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["narrative_sources"] == "FAIL"


def test_narrative_without_figure_fails():
    body = "## TL;DR\nSource-issues: #91, #109, #142\n\nAll prose, no figure links here.\n"
    report = Report()
    check_narrative_consolidation(body, report)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["narrative_sources"] == "PASS"
    assert statuses["narrative_figure"] == "FAIL"


def test_narrative_accepts_relative_figure_path():
    """`figures/<path>.png` (without full URL) should still satisfy the figure check."""
    body = (
        "## TL;DR\n"
        "Source-issues: #91, #109\n"
        "\n"
        "See `figures/aim5/dose_response.pdf` for the hero panel.\n"
    )
    report = Report()
    check_narrative_consolidation(body, report)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["narrative_figure"] == "PASS"
