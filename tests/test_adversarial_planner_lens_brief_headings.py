"""Pin #1282 (v1) + #1292 (v2): critic briefs keep the canonical lens-heading anchor.

Pins (incident #1265): (1) each adversarial-planner Phase 2 lens template
carries a `Canonical rubric:` line citing critic-lens-reference.md plus the
VERBATIM lens heading; (2) critic.md makes the canonical capsule heading the
only legal grep target, with STOP-and-re-grep on a no-span result; (3) the
pinned heading strings are the ACTUAL headings in critic-lens-reference.md
(a heading rename must update templates + capsules in the same commit);
(4) #1292 v2 pins: the three v2 lens-critic agents + their three Codex
composers cite their canonical headings backticked-verbatim, the Claude
agents carry the STOP-and-re-grep anchor rule, the composers carry the
no-span compose gate, and every `.claude/agents/*.md` consumer of
critic-lens-reference.md is pinned or explicitly allowlisted.
(5) #1302: the mbc inline Alt capsule cites its source heading + carries the
reference-wins re-sync coupling.
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


# --- #1292: v2 lens-critic pins (the v2 sibling of the #1282 pins above) ---

# Deliberately hardcodes heading strings independently of HEADINGS: deriving the
# values from HEADINGS would make assertion (a) of
# test_v2_cited_headings_are_canonical_and_exist_in_lens_reference tautological.
V2_FILES_TO_HEADINGS = {
    ".claude/agents/statistics-critic.md": ("### Statistics & Measurement lens",),
    ".claude/agents/methodology-baselines-critic.md": (
        "### Methodology lens",
        "### Alternative Explanations lens",
    ),
    ".claude/agents/efficiency-critic.md": ("### Methodology lens",),
    ".claude/agents/codex-statistics-critic.md": ("### Statistics & Measurement lens",),
    ".claude/agents/codex-methodology-baselines-critic.md": (
        "### Methodology lens",
        "### Alternative Explanations lens",
    ),
    ".claude/agents/codex-efficiency-critic.md": ("### Methodology lens",),
}
V2_CLAUDE_AGENTS = tuple(k for k in V2_FILES_TO_HEADINGS if "/codex-" not in k)
V2_CODEX_COMPOSERS = tuple(k for k in V2_FILES_TO_HEADINGS if "/codex-" in k)


def test_v2_cited_headings_are_canonical_and_exist_in_lens_reference():
    for rel, headings in V2_FILES_TO_HEADINGS.items():
        text_norm = _norm((REPO_ROOT / rel).read_text(encoding="utf-8"))
        for heading in headings:
            assert heading in HEADINGS, (rel, heading)  # (a) cited heading is canonical
            assert f"\n{heading}\n" in LENS_REF, heading  # (b) and live in the reference
            assert _norm(f"`{heading}`") in text_norm, (rel, heading)  # (c) cited verbatim


def test_v2_claude_agents_carry_stop_and_regrep_rule():
    for rel in V2_CLAUDE_AGENTS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "Canonical-heading anchor" in text, rel
        assert "returns NO span, STOP" in _norm(text), rel


def test_v2_codex_composers_carry_no_span_compose_gate():
    for rel in V2_CODEX_COMPOSERS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "No-span compose gate" in text, rel
        assert "STOP and return a BLOCKER" in _norm(text), rel


def test_mbc_alt_capsule_is_coupled_to_reference_span():
    """#1302: the mbc item-2 Alt capsule is an ADAPTED absorption (not a verbatim
    copy), so the pin is a cross-reference assertion — the capsule cites the
    canonical heading verbatim (covered by V2_FILES_TO_HEADINGS assertions) AND
    carries the read-alongside / reference-wins load instruction. Deleting the
    coupling sentence, or renaming the source heading, fails a test. NOTE: the
    first needle hardcodes the heading string, so a heading-rename commit must
    update this test too (loud by design)."""
    rel = ".claude/agents/methodology-baselines-critic.md"
    text_norm = _norm((REPO_ROOT / rel).read_text(encoding="utf-8"))
    assert (
        _norm("absorbs the fatal-confound screen from `### Alternative Explanations lens`")
        in text_norm
    )
    assert "on divergence the reference wins" in text_norm


# Discovery pin — every .claude/agents/*.md consumer of critic-lens-reference.md
# must be a pinned v1/v2 file or an explicitly-allowlisted mention-only file, so
# a future lens agent cannot land citing the reference unpinned.
PINNED_V1 = (".claude/agents/critic.md", ".claude/agents/codex-critic.md")
MENTION_ONLY_ALLOWLIST = (
    # These two cite `.claude/rules/clean-result-critic-lens-reference.md` — a
    # DIFFERENT reference file whose name contains "critic-lens-reference" as a
    # substring. Verified at #1292 implementation time: neither loads
    # `.claude/rules/critic-lens-reference.md` itself.
    ".claude/agents/clean-result-critic.md",
    ".claude/agents/codex-clean-result-critic.md",
)


def test_lens_reference_consumers_are_pinned():
    known = set(PINNED_V1) | set(V2_FILES_TO_HEADINGS) | set(MENTION_ONLY_ALLOWLIST)
    for path in sorted((REPO_ROOT / ".claude/agents").glob("*.md")):
        if "critic-lens-reference" in path.read_text(encoding="utf-8"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            assert rel in known, (
                f"{rel} cites critic-lens-reference.md but is not pinned in "
                "V2_FILES_TO_HEADINGS / PINNED_V1 / MENTION_ONLY_ALLOWLIST"
            )
