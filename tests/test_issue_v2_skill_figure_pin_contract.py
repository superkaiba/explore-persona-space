"""Durability pins for the #1224 figure-commit ordering contract (Option A).

The v2 report pipeline commits the HELD plotter figures ONCE at issue-v2
SKILL.md Step 7b (after upload-verification PASS, BEFORE assembly at 7c), so
7c splices real SHA-pinned image URLs and 7f only writes the body + parks.
These shape asserts pin that single ordering across the four prose surfaces
(SKILL.md / plotter.md / report-verifier.md / the paper-mode critics) so a
future prose edit cannot silently reintroduce the 7c-vs-7f contradiction or
the plotter's retired "rewrite the URLs after commit" third variant.

Paths resolve via git from this file's directory (worktree-safe, cwd-free).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    """The repo/worktree root of THIS checkout, resolved via git (never cwd)."""
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent,
    )
    return Path(out.stdout.strip())


def _norm(text: str) -> str:
    """Collapse whitespace so line-wrapped prose matches single-space needles."""
    return " ".join(text.split()).lower()


def _read(rel: str) -> str:
    return (_repo_root() / rel).read_text()


def _step_block(skill_norm: str, step: str, next_step: str) -> str:
    """The normalized text between ``**<step>.`` and ``**<next_step>.``."""
    m = re.search(
        re.escape(f"**{step}.") + r"(.*?)" + re.escape(f"**{next_step}."),
        skill_norm,
        re.DOTALL,
    )
    assert m, f"could not locate the {step}..{next_step} block in issue-v2 SKILL.md"
    return m.group(1)


def test_figure_commit_lives_at_7b_not_7f():
    skill = _norm(_read(".claude/skills/issue-v2/SKILL.md"))
    block_7b = _step_block(skill, "7b", "7c")
    block_7f = skill.split("**7f.", 1)[1]

    # 7b owns the figure commit: held figures committed BEFORE assembly, one SHA.
    assert "commit held figures" in skill.split("**7b.", 1)[1][:80]
    assert "commit the held plotter figures" in block_7b
    assert "before assembly" in block_7b
    assert "git add figures/issue_<n>/" in block_7b
    # 7f no longer commits figures — it only writes the body and parks.
    assert "write body + park" in block_7f[:40]
    assert "commit the held" not in block_7f
    assert "figures were committed + pinned at step 7b" in block_7f


def test_plotter_and_report_verifier_agree_on_7b():
    plotter = _norm(_read(".claude/agents/plotter.md"))
    verifier = _norm(_read(".claude/agents/report-verifier.md"))

    # plotter.md: the retired "rewrite the report's image URLs" third variant
    # is gone; the orchestrator commits at Step 7b and splices pins at 7c.
    assert "rewrites the report" not in plotter
    assert "commits the held figures at step 7b" in plotter
    assert "step-7b commit + the 7c pin splice" in plotter

    # report-verifier.md: the "around report assembly — Steps 7c/7f" hedge is
    # gone; item 2 states the 7b ordering, and check (e) verifies the DRAFT.
    assert "around report assembly" not in verifier
    assert "commits the held figures at skill step 7b" in verifier
    assert "--file <report-draft>.md --mode generation" in verifier
    assert "--expect-issue <n>" in verifier


def test_codex_paper_branch_carries_figure_read_target_rule():
    needle = (
        "the compiled pdf is the built artifact of record — on a working-tree-png "
        "vs pdf-page disagreement, review against the pdf page, note the possible "
        "stale working-tree stray, and never rest a blocker on the png alone"
    )
    codex = _norm(_read(".claude/agents/codex-clean-result-critic.md"))
    assert needle in codex, "codex paper branch lost the figure read-target rule"
    # The Claude-side paper path reads .claude/rules/clean-result-paper-review.md
    # IN FULL (clean-result-critic.md's paper branch delegates there), so the
    # same sentence must live in the rule file's read-target list.
    rule = _norm(_read(".claude/rules/clean-result-paper-review.md"))
    assert needle in rule, "clean-result-paper-review.md lost the figure read-target rule"
