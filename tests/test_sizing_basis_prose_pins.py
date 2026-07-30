"""Durability pins for the #1798 sizing-basis prose clauses.

Token-presence asserts over the two rule files carrying the
multiplier-derivation / draw-count-necessity / largest-cell RAM keying /
basis-currency / pool-scale-pilot / output-growth-monitoring clauses
(incidents #1689, #1739, #1738; the c31 durability-pin discipline;
precedent shape: tests/test_downwidth_split_prose_pins.py). Auto-selected
on plan-compute-sizing.md / vectorize-many-cell-fits.md diffs by the
#1496 rules-pin discovery arm (basename substring).

Substrings are asserted against whitespace-NORMALIZED text so the pins
tolerate hard-wrap reflow of the rule prose.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(path: Path) -> str:
    """Read a rule file and collapse all whitespace runs to single spaces."""
    return " ".join(path.read_text().split())


def test_plan_compute_sizing_fit_phase_basis_clauses():
    text = _norm(REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md")
    # Clause 1 — multiplier derivation (#1689)
    assert "MULTIPLIER DERIVATION" in text, (
        "lost clause: multiplier derivation (n_calls read off the code) — #1689"
    )
    assert "DERIVED FROM THE CODE" in text, (
        "lost clause: multiplier derivation (n_calls read off the code) — #1689"
    )
    # Clause 2 — draw-count necessity + descope lever (#1689)
    assert "DRAW-COUNT NECESSITY" in text, (
        "lost clause: draw-count necessity for dominant batteries — #1689"
    )
    assert "pre-registered DESCOPE lever" in text, (
        "lost clause: draw-count necessity for dominant batteries — #1689"
    )
    # Clause 5 — pool-scale pilots for superlinear kernels (#1738)
    assert "POOL-SCALE PILOTS" in text, (
        "lost clause: pool-scale pilots for superlinear kernels — #1738"
    )
    assert "states the scaling exponent" in text, (
        "lost clause: pool-scale pilots for superlinear kernels — #1738"
    )
    # Monitoring sentence — output growth over CPU% (#1738)
    assert "OUTPUT growth" in text, (
        "lost clause: health reads key on output growth, never CPU% alone — #1738"
    )
    assert "never CPU% alone" in text, (
        "lost clause: health reads key on output growth, never CPU% alone — #1738"
    )
    # Clause 4a — recorded basis update / basis currency (#1738)
    assert "BASIS CURRENCY" in text, "lost clause: recorded basis update on >=2x deviation — #1738"
    assert "never left standing known-stale" in text, (
        "lost clause: recorded basis update on >=2x deviation — #1738"
    )


def test_plan_compute_sizing_largest_cell_ram_keying():
    text = _norm(REPO_ROOT / ".claude" / "rules" / "plan-compute-sizing.md")
    # Clause 3 — largest-cell RAM/RSS keying, scope-widened to GPU-lane host RAM (#1739)
    assert "LARGEST-CELL KEYING" in text, "lost clause: largest-cell RAM/RSS keying — #1739"
    assert "never an anchor or first-listed unit" in text, (
        "lost clause: largest-cell RAM/RSS keying — #1739"
    )
    assert "GPU-lane host-RAM" in text, (
        "lost clause: largest-cell keying scope-widening to GPU-lane host RAM — #1739"
    )


def test_vectorize_midrun_basis_restatement():
    text = _norm(REPO_ROOT / ".claude" / "rules" / "vectorize-many-cell-fits.md")
    # Clause 4b — basis re-statement + draw-necessity lever on resolution re-posts (#1689)
    assert "re-states the deviating row's basis" in text, (
        "lost clause: mid-run resolution re-post re-states the row's basis — #1689"
    )
    assert "draw-necessity lever" in text, (
        "lost clause: draw-necessity descope check on dominant batteries — #1689"
    )
