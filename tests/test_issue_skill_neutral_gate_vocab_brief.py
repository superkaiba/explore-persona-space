"""Prose pins for the #1398 neutral-gate-vocabulary first-pass brief rule.

CLAUDE.md refusal-prevention rung (e) and SKILL.md Step 5a must both carry
the neutral gate vocabulary ("halt gate", "stop criterion", "termination
predicate") for kill-gate / RLVR / guard / stop-criteria task briefs, with
the artifacts-never-renamed bar, so briefs are neutralized from the first
spawn rather than only after refusal kills.
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SKILL_MD = _REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = _REPO / "CLAUDE.md"


def _step5a_section() -> str:
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index("5a. Spawn both reviewers")
    end = text.index("5b. Read both markers")
    return text[start:end]


def test_step5a_neutral_gate_vocab_first_pass_brief_pin():
    section = _step5a_section()
    assert "halt gate" in section
    assert "stop criterion" in section
    assert "termination predicate" in section
    # The artifacts-untouched bar: loaded terms stay in code/plans/bodies.
    assert "never renamed" in section
    # The first-pass leg: neutral wording from the first spawn, not post-kill.
    assert "first spawn" in section


def test_claude_md_rung_e_neutral_gate_vocab():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    start = text.index("Spurious usage-policy refusals")
    end = text.index("(f)", start)
    rung_e_slice = text[start:end]
    assert "halt gate" in rung_e_slice
    assert "stop criterion" in rung_e_slice
    # Both rung-(e) legs (bank-naming + gate vocabulary) are first-pass.
    assert "BOTH disciplines" in rung_e_slice
