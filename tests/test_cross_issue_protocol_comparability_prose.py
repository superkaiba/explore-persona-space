"""Durability pin for the cross-issue protocol-comparability citation rule (#1406).

The rule is enforcement PROSE only (no mechanical verifier check -- protocol
identity is semantic, not greppable), so a later spec-trim pass could silently
drop it while every lint stays green. This pin fails loud if any of the three
prose surfaces loses its clause: SPEC.md (the source of truth), the
clean-result-critic Lens 7 rubric, and the analyzer drafting duty. Origin
incident: #779 vs #823 protocol-mismatched R-squared headlines quoted side by
side in mentor-facing prose. Family precedent:
tests/test_analyzer_language_intrusion_duty.py.
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_protocol_delta_clause_present() -> None:
    """SPEC.md, Lens 7, and analyzer.md all carry the protocol-delta clause."""
    spec = (_REPO / ".claude" / "skills" / "clean-results" / "SPEC.md").read_text(encoding="utf-8")
    # (a) SPEC.md: both v4 anchors (## Goal slot + ## Results descriptive baseline)
    assert spec.count("not directly comparable") >= 2, (
        "SPEC.md lost a 'not directly comparable' worked example "
        "(need one in the ## Goal (v4) slot and one in ## Results (v4))"
    )
    assert spec.count("Cross-issue protocol comparability") >= 1, (
        "SPEC.md lost the Cross-issue protocol comparability clause"
    )

    # (b) lens-reference: the enforcement paragraph lives inside the Lens 7 span
    lens_ref = (_REPO / ".claude" / "rules" / "clean-result-critic-lens-reference.md").read_text(
        encoding="utf-8"
    )
    lens7_start = lens_ref.index("### Lens 7")
    lens8_start = lens_ref.index("### Lens 8")
    assert lens7_start < lens8_start, "Lens 7/8 heading order broke"
    lens7_span = lens_ref[lens7_start:lens8_start]
    assert "Cross-issue protocol comparability" in lens7_span, (
        "Lens 7 lost the Cross-issue protocol comparability enforcement paragraph"
    )

    # (c) analyzer drafting duty
    analyzer = (_REPO / ".claude" / "agents" / "analyzer.md").read_text(encoding="utf-8")
    assert "Cross-issue protocol delta" in analyzer, (
        "analyzer.md lost the Cross-issue protocol delta drafting-duty bullet"
    )
