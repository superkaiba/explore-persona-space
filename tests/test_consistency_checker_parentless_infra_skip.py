"""Pin the parentless-kind: infra consistency-checker SKIP rule (#1732).

Grep-based prose-durability pins for the three workflow-surface files
that carry the SKIP + `epm:consistency v1` PASS-skipped marker-post
duty. Modeled on ``tests/test_issue_skill_marker_contract.py``.

The rule (from task #1732, closing the #1697 vs #1711 divergence):
when a task is ``kind: infra | batch | survey`` with no ``parent_id``
and no unrun ``epm:followup-scope v1`` marker, the consistency-checker
spawn is SKIPPED and the orchestrator posts an ``epm:consistency v1``
marker with ``**Verdict: PASS**`` whose first line is
``Skipped: kind:<X>, no parent experiment``.

Any drift in the SKIP-clause prose (a rewrite that removes the
``kind: infra | batch | survey`` predicate, a rename of the marker
channel, a removal of the marker-post duty on any surface) is a red
at Step-9c. The pin is grep-based deliberately — it costs no runtime
and its assertions are pure string containment, so it never flakes.
"""

from __future__ import annotations

from pathlib import Path

# Resolve paths against the test file's own repo root (mirrors
# tests/test_issue_skill_marker_contract.py). This keeps the pin
# gate-scope-correct in a worktree: `task_workflow.repo_root()` branch-guards
# to main, but the Step-9c gate runs pytest inside the diff-owning worktree
# and must see THAT worktree's edits — resolving via `__file__` reads the
# worktree copies directly.
ROOT = Path(__file__).resolve().parent.parent

CONSISTENCY_CHECKER = ROOT / ".claude" / "agents" / "consistency-checker.md"
ADVERSARIAL_PLANNER_SKILL = ROOT / ".claude" / "skills" / "adversarial-planner" / "SKILL.md"
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_agent_spec_carries_parentless_infra_skip_clause() -> None:
    """`.claude/agents/consistency-checker.md` names the SKIP predicate for
    parentless ``kind: infra | batch | survey`` AND names the
    ``epm:consistency v1`` marker with a ``Skipped:`` first line.
    """
    text = _read(CONSISTENCY_CHECKER)
    # The SKIP predicate must name all three kinds AND the parent-absence
    # condition (#1732).
    assert "kind: infra | batch | survey" in text, (
        f"{CONSISTENCY_CHECKER} must carry the SKIP predicate naming "
        "`kind: infra | batch | survey` for parentless non-experiment "
        "tasks (#1732). Without this predicate the checker either spawns "
        "against a task shape it cannot bind to (#1697 shape — all-'N/A' "
        "rows, no signal) or is skipped ad-hoc with no marker (#1711 "
        "shape — no auditable record on events.jsonl)."
    )
    assert "no parent" in text.lower(), (
        f"{CONSISTENCY_CHECKER} must name the parent-absence condition "
        "(no `parent_id`) as part of the SKIP predicate (#1732)."
    )
    # The marker-post duty on this surface: name the channel and the
    # Skipped: first-line convention.
    assert "epm:consistency v1" in text, (
        f"{CONSISTENCY_CHECKER} must name the `epm:consistency v1` marker "
        "channel as the record of the SKIP (#1732). Reusing the same "
        "channel the RAN branch uses collapses the reader's mental model "
        "to one lookup: an `epm:consistency` event is either present "
        "(verdict PASS/WARN/BLOCK) or present-with-`skipped` reason — "
        "never absent-and-ambiguous."
    )
    assert "Skipped: kind:" in text, (
        f"{CONSISTENCY_CHECKER} must name the `Skipped: kind:<X>, no "
        "parent experiment` first-line marker-body convention (#1732). "
        "The first-line convention is what tells a reader a SKIPPED "
        "verdict apart from a RAN PASS verdict."
    )


def test_adversarial_planner_skill_names_precondition_and_marker() -> None:
    """`.claude/skills/adversarial-planner/SKILL.md` states the
    parentless-non-experiment precondition AND points at the
    ``epm:consistency v1`` PASS-skipped marker.
    """
    text = _read(ADVERSARIAL_PLANNER_SKILL)
    # The Phase-2 spawn-batch paragraph carries the precondition (naming
    # the kind set explicitly).
    assert "kind: infra | batch | survey" in text, (
        f"{ADVERSARIAL_PLANNER_SKILL} must name the parentless "
        "`kind: infra | batch | survey` precondition on the "
        "consistency-checker spawn (#1732)."
    )
    # The parent-absence half of the predicate must survive edits — the
    # kind set alone is not sufficient (a `kind: experiment` with no
    # parent still routes to the RAN standard-baseline branch).
    assert "parent" in text.lower(), (
        f"{ADVERSARIAL_PLANNER_SKILL} must name the parent-absence "
        "condition alongside the kind set (#1732 — the SKIP applies "
        "only when there is no parent)."
    )
    # The marker-post duty is named on this surface so the pseudocode
    # skip-branch is not silent.
    assert "epm:consistency v1" in text, (
        f"{ADVERSARIAL_PLANNER_SKILL} must name the `epm:consistency v1` "
        "PASS-skipped marker as the record of the skip (#1732). Without "
        "it a reader arriving at the pseudocode `c_check = None` branch "
        "has no channel to check for the SKIP record."
    )
    # The Phase 2 spawn-table row for the consistency-checker must call
    # out SKIP for the parentless-non-experiment case (#1732).
    assert "SKIPPED" in text or "Skipped" in text, (
        f"{ADVERSARIAL_PLANNER_SKILL} Phase 2 spawn-table row for the "
        "consistency-checker must name SKIP for the parentless-non-"
        "experiment case (#1732)."
    )


def test_issue_skill_step_2b_mirrors_skipped_branch() -> None:
    """`.claude/skills/issue/SKILL.md` Step 2b documents the Skipped
    branch (matching predicate) AND names the marker-post duty.

    The Skipped branch prose must live within the Step 2b span — a
    Skipped clause outside Step 2b would not be read by an orchestrator
    resolving the Step 2b protocol.
    """
    text = _read(ISSUE_SKILL)
    # Step 2b block is bounded by its heading; check the SKIP clause
    # appears within a bounded window (~4000 chars) of the heading.
    heading = "### Step 2b: Consistency checker"
    idx = text.index(heading)
    window = text[idx : idx + 4000]
    assert "kind: infra | batch | survey" in window, (
        f"{ISSUE_SKILL} Step 2b must carry the `kind: infra | batch | survey` predicate (#1732)."
    )
    assert "epm:consistency v1" in window, (
        f"{ISSUE_SKILL} Step 2b must name the `epm:consistency v1` "
        "marker-post duty inside the Skipped branch (#1732)."
    )
    assert "Skipped" in window, (
        f"{ISSUE_SKILL} Step 2b must call out the Skipped branch explicitly (#1732)."
    )
