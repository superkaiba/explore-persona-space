"""Durability pin (#1476): the bootstrap-CI selection-inheritance clause in
selection-symmetric-nulls.md and its enforcement pointers stay present.

Prose in trim-prone surfaces (the 85 KB lens reference, agent specs under
byte ratchets) can be silently dropped in a future trim; these asserts make
the drop loud. Grep-level pins only — no wording lock beyond the heading
and the load-bearing term.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_bootstrap_ci_clause_present_in_rule() -> None:
    text = _read(".claude/rules/selection-symmetric-nulls.md")
    assert "## Bootstrap CIs at a selected axis position" in text
    assert "selection-inherited CI" in text


def test_critic_lens_reference_points_at_clause() -> None:
    assert "selection-inherited" in _read(".claude/rules/critic-lens-reference.md")


def test_statistics_critic_item11_points_at_clause() -> None:
    assert "selection-inherited" in _read(".claude/agents/statistics-critic.md")
