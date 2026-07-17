"""Prose pins for the #1398 neutral-gate-vocabulary brief rule + the #1461
revision-round extension (#1413/#1415).

CLAUDE.md refusal-prevention rung (e) and SKILL.md Step 5a must both carry
the neutral gate vocabulary ("halt gate", "stop criterion", "termination
predicate") for kill-gate / RLVR / guard / stop-criteria task briefs, with
the artifacts-never-renamed bar, so briefs are neutralized from the first
spawn rather than only after refusal kills. The #1461 extension pins the
revision-round side: trigger-dense-review.md § Revision-round briefs
(findings passed by marker/file reference, never inlined), the
adversarial-planner Phase 3 critique-by-file pointer, the Step 5d / 9a
by-reference parentheticals, and the rung-(e) steering-vocabulary +
revision-round clauses (#1413, #1415).
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SKILL_MD = _REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = _REPO / "CLAUDE.md"
RULE_MD = _REPO / ".claude" / "rules" / "trigger-dense-review.md"
AP_SKILL_MD = _REPO / ".claude" / "skills" / "adversarial-planner" / "SKILL.md"


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


# --- #1461 revision-round extension pins (#1413/#1415) ---


def test_trigger_dense_rule_revision_round_brief_section():
    text = RULE_MD.read_text(encoding="utf-8")
    assert "## Revision-round briefs" in text
    assert "#1413" in text
    assert "NEVER inline" in text  # the by-reference duty
    assert "rung-(g)" in text  # truncated-spawn verify carry-over


def test_adversarial_planner_phase3_critique_by_file():
    text = AP_SKILL_MD.read_text(encoding="utf-8")
    ph3 = text[text.index("### Phase 3") : text.index("## Phase 4")]
    assert "critique-r<K>.md" in ph3
    assert "#1413" in ph3


def test_step5d_bounce_by_reference():
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index("**5d. Loop on FAIL")
    s5d = text[start : text.index("CAP-HIT", start)]
    assert "File-only Codex verdict posting" in s5d


def test_step9a_analyzer_respawn_by_reference():
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index("If `final_verdict == REVISE`")
    s9a = text[start : text.index("Max 5 rounds per reviewer", start)]
    assert "critique events by reference" in s9a


def test_claude_md_rung_e_steering_vocab():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    start = text.index("Spurious usage-policy refusals")
    rung_e_slice = text[start : text.index("(f)", start)]
    assert "#1415" in rung_e_slice  # steering-vocabulary class
    assert "REVISION-ROUND briefs" in rung_e_slice  # rung-(e) extension
