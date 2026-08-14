"""Durability pins for the #1919 measurement-basis prose clauses.

Token-presence asserts over .claude/rules/plan-compute-sizing.md for the
two measurement-basis clauses (incidents #1739 cross-regime pilot
proxying; #1773 one-shot nvidia-smi utilization claim; precedent shape:
tests/test_sizing_basis_prose_pins.py). Auto-selected on
plan-compute-sizing.md diffs by the #1496 rules-pin discovery arm
(basename substring).

Substrings are asserted against whitespace-NORMALIZED text so the pins
tolerate hard-wrap reflow of the rule prose.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(path: Path) -> str:
    """Read a rule file and collapse all whitespace runs to single spaces."""
    return " ".join(path.read_text().split())


def test_plan_compute_sizing_per_regime_binding_clause():
    text = _norm(REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md")
    # Clause 1 — pilot wall binds to its own behavior/budget regime (#1739)
    assert "PER-REGIME BINDING" in text, "lost clause: per-regime pilot binding — #1739"
    assert "GUESSED basis" in text, (
        "lost clause: cross-regime proxying degrades a measured basis to guessed — #1739"
    )
    assert "(#1739: per-group walls measured on the evil behavior" in text, (
        "lost incident citation: #1739 cross-regime proxying (4/6 lanes halted at pilot gates)"
    )
    # Clause 1b — per-family pilot floor on heterogeneous fan-outs (#2048)
    assert "HETEROGENEOUS FAN-OUTS (per-family pilot floor)" in text, (
        "lost clause: >~4x heterogeneity => per-family measured pilots — #2048/#1739"
    )
    assert "not the pilot family's wall verbatim" in text, (
        "lost clause: worst-case extrapolation is COMPUTED (family-multiplier-scaled) — #2048"
    )


def test_plan_compute_sizing_sampled_window_utilization_clause():
    text = _norm(REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md")
    # Clause 2 — GPU-utilization claims need a sampled window (#1773)
    assert "SAMPLED window" in text, "lost clause: sampled-window utilization claims — #1773"
    assert "never one instantaneous" in text, (
        "lost clause: one-shot nvidia-smi reads banned as a utilization basis — #1773"
    )
    assert "≥10 readings over ≥60 s" in text, (
        "lost clause: sampling-basis floor (>=10 readings over >=60 s) — #1773"
    )
    assert "(#1773: a pre-spend checkpoint claimed" in text, (
        "lost incident citation: #1773 one-sample claim vs 30-reading/60 s mean 12.6%"
    )
