"""Prose pins for the #1326 Step-0 registry-blind live-driver probe.

``.claude/skills/issue/SKILL.md`` Step 0's single-orchestrator guard gained a
second leg (#1326): a marker-trail probe (``task_workflow.live_driver_evidence``)
that treats fresh compute-launch markers / ``stage-dispatch `` breadcrumbs as
evidence of a live registry-invisible driver (the CLAUDE.md inline-chat
carve-out), failing toward YIELD via the EXISTING ``step0-session-collision``
breadcrumb block — no third ``--by issue-session-guard`` block. CLAUDE.md's
inline-override clause gained the matching ``register-current`` duty (fail-soft)
plus a completion-time registration cleanup whose omission exposes the live
chat session to the watcher session-reconcile auto-stop. These tests read the
prose itself so a rewording that drops the probe, the yield default, or the
registration/cleanup duty goes red (durability pins, same style as
``tests/test_issue_skill_exit_breadcrumb.py``).
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
SKILL_MD = _REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = _REPO / "CLAUDE.md"

_PROBE_HEADING = "**Registry-blind live-driver probe"
_NEXT_HEADING = "**Stale-wake ownership re-check"


def _probe_paragraph() -> str:
    """The Step-0 probe paragraph: from its heading to the stale-wake heading."""
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.index(_PROBE_HEADING)
    end = text.index(_NEXT_HEADING, start)
    return text[start:end]


def test_step0_guard_prescribes_live_driver_probe():
    # The probe call line is load-bearing: helper name, list_events source,
    # and the explicit 30-min window pin at the call site.
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "live_driver_evidence(list_events(<N>), window_minutes=30)" in text


def test_probe_fails_toward_yield_and_reuses_collision_reason():
    para = _probe_paragraph()
    # Fail-toward-YIELD default, exiting via the EXISTING collision block
    # (referenced by reason token — not a new command block).
    assert "fail toward YIELD" in para
    assert "reason=step0-session-collision" in para
    # Redundant belt with tests/test_issue_skill_exit_breadcrumb.py: the
    # probe must NOT add a third guard-exit command block.
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.count("--by issue-session-guard") == 2


def test_inline_carveout_carries_register_current_duty():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    # Registration duty (fail-soft) at dispatch of a GPU-backed inline round.
    assert "register-current --issue <N>" in text
    assert "manual-issue-<N>.json" in text
    # The cleanup sentence names the session-reconcile auto-stop consequence
    # (a lingering registration strips the chat session's structural
    # reconcile-pass protection) NEAR the cleanup duty itself.
    cleanup_idx = text.index("rm -f ~/.eps-autonomous/manual-issue-<N>.json")
    window = text[cleanup_idx : cleanup_idx + 800]
    assert "reconcile" in window
    assert "auto-stop" in window
