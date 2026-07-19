"""Cross-file literal pins for the #1535 off-pod phase declaration slot.

Pins the template <-> c39-regex <-> verifier-consumer cross-file consistency
against plural/wording drift: the ``off_pod_phases:`` literal must stay
byte-identical in the three spec files (planner.md names the duty,
planner-section-reference.md carries the template + worked example,
upload-verifier.md consumes the block at Steps 2.7/2.8); the
``off-pod-phase-spec-absent`` WARN-row id and the Step 2.7 declared-off-pod
sub-rule must stay greppable in upload-verifier.md; and the exact c39 escape
literal must stay present in the section-reference §9 subsection (the
em-dash U+2014 is load-bearing — a hyphen substitution breaks the #1264
docstring<->SKILL.md sync test and the c39 satisfier). Read-only.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_off_pod_slot_literals_present():
    # The durability pin (#1535 plan §4): the slot literal in all three
    # spec files that name / template / consume it.
    assert "off_pod_phases:" in _read(".claude/agents/planner.md")
    assert "off_pod_phases:" in _read(".claude/rules/planner-section-reference.md")
    assert "off_pod_phases:" in _read(".claude/agents/upload-verifier.md")


def test_warn_row_id_present_in_verifier():
    # Step 2.8 arm + the amended verdict-table row both carry the WARN id.
    verifier = _read(".claude/agents/upload-verifier.md")
    assert verifier.count("off-pod-phase-spec-absent") >= 2


def test_declared_off_pod_sub_rule_present():
    # The Step 2.7 declared-off-pod outputs sub-rule (the #1426 fix).
    assert "Declared off-pod outputs" in _read(".claude/agents/upload-verifier.md")


def test_exact_escape_literal_in_template():
    # Byte-exact escape literal, em-dash form: "N/A — no off-pod phase".
    # A hyphen substitution would break _standalone_na_declared's NA_RE match
    # and the #1264 sync test.
    section_ref = _read(".claude/rules/planner-section-reference.md")
    assert "N/A — no off-pod phase" in section_ref
    # Codepoint-driven twin (chr, not a literal): survives any editor/tool
    # layer silently normalizing the literal above.
    assert "N/A " + chr(0x2014) + " no off-pod phase" in section_ref
