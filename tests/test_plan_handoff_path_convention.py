"""Regression tests for the plan-handoff convention (issue #282 [4/4]); #1581 pins the
edit-success pre-persist gate.

CLAUDE.md documents that subagent dispatch must hand over the PATH to the
cached plan (`.claude/plans/issue-<N>.md`), not the plan body. These tests
guard the convention against drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

# Repository root (this file lives at <root>/tests/).
REPO_ROOT = Path(__file__).resolve().parent.parent

DISPATCH_AGENTS = ("experiment-implementer", "implementer", "experimenter")
PATH_PATTERN = re.compile(r"\.claude/plans/issue-")


def test_dispatch_agent_prompts_reference_plan_path() -> None:
    """Each agent that receives a plan via dispatch must reference the cached
    plan path, not infer plan content. Positive-form check (per critic C3
    round 2) — false-positive-free unlike a negative heuristic."""
    for name in DISPATCH_AGENTS:
        path = REPO_ROOT / ".claude" / "agents" / f"{name}.md"
        assert path.exists(), f"{path} missing"
        body = path.read_text()
        assert PATH_PATTERN.search(body), (
            f"{path}: dispatch agent prompt should reference "
            f"\\.claude/plans/issue-<N>.md as the plan-handoff path"
        )


def test_claude_md_contains_plan_handoff_rule() -> None:
    """Per critic C2 round 2: ensure the CLAUDE.md rule actually landed."""
    body = (REPO_ROOT / "CLAUDE.md").read_text()
    assert "Plan handoff convention" in body, (
        "CLAUDE.md must include the 'Plan handoff convention' rule"
    )


AP_SKILL = REPO_ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md"
ISSUE_SKILL = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"


def test_adversarial_planner_edit_success_gate_pinned() -> None:
    """#1581: a scripted plan edit gates its `new-plan-version` persist on
    verified edit success (edit -> verify -> persist, &&-chained; an edit
    failure aborts the persist loudly). Dropping this prose re-opens the
    #1565/#1563 gap where an AssertionError'd edit script still persisted the
    plan as an unmodified copy of the prior version."""
    ap = AP_SKILL.read_text()
    for token in (
        "Edit-success gate",
        "EDIT FAILED — not persisting",
        "edit → verify → persist",
        "never run the persist inside the same\nscript",
    ):
        assert token.replace("\n", " ") in ap.replace("\n", " "), (
            f"adversarial-planner SKILL.md lost edit-success-gate token {token!r}"
        )
    issue = issue_skill_text()
    assert "Edit-success gate" in issue, (
        "issue SKILL.md lost the edit-success-gate pointer next to its Goal-currency gate"
    )
