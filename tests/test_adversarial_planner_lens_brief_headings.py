"""Pin #1282: critic briefs keep the canonical lens-heading anchor.

Pins (incident #1265): (1) each adversarial-planner Phase 2 lens template
carries a `Canonical rubric:` line citing critic-lens-reference.md plus the
VERBATIM lens heading; (2) critic.md makes the canonical capsule heading the
only legal grep target, with STOP-and-re-grep on a no-span result; (3) the
pinned heading strings are the ACTUAL headings in critic-lens-reference.md
(a heading rename must update templates + capsules in the same commit).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = (REPO_ROOT / ".claude/skills/adversarial-planner/SKILL.md").read_text(encoding="utf-8")
CRITIC = (REPO_ROOT / ".claude/agents/critic.md").read_text(encoding="utf-8")
LENS_REF = (REPO_ROOT / ".claude/rules/critic-lens-reference.md").read_text(encoding="utf-8")

HEADINGS = (
    "### Methodology lens",
    "### Statistics & Measurement lens",
    "### Alternative Explanations lens",
)


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_headings_are_verbatim_in_lens_reference():
    for heading in HEADINGS:
        assert f"\n{heading}\n" in LENS_REF


def test_each_brief_template_carries_canonical_rubric_line():
    skill_norm = _norm(SKILL)
    for heading in HEADINGS:
        needle = _norm(
            f"Canonical rubric: grep `{heading}` in `.claude/rules/critic-lens-reference.md`"
        )
        assert needle in skill_norm


def test_skill_carries_anchor_rule_for_adapted_briefs():
    assert "Canonical-rubric anchor" in SKILL


def test_critic_md_grep_target_is_canonical_never_brief_supplied():
    critic_norm = _norm(CRITIC)
    assert "NEVER a brief-supplied translated or adapted title" in critic_norm
    assert "re-grep with the canonical heading" in critic_norm
