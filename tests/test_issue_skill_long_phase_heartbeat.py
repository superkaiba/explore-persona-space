"""Pin the #1207 long-phase heartbeat duty + follow-up-dispatch tick re-arm.

On 2026-07-08 the autonomous-session watcher force-respawned four LIVE,
healthy sessions (#1092 x2, #825, #1112) because long detached / API-bound
waits left both staleness signals frozen. The `[long-phase-heartbeat]`
opt-in already existed on the CONSUMER side (`autonomous_session_watch.py`
90-min exemption leash + `tick_triage.py` #1051 liveness screen) but the
EMITTER-side duty was prescribed for only one wait shape. Separately, the
Step 9b cheap-band / autonomous dispatch prose claimed "the backstop cron
stays armed", which was false: CRON-TEARDOWN runs unconditionally at the
`awaiting_promotion` transition BEFORE those blocks (#1112).

These tests pin, against `.claude/skills/issue/SKILL.md`:

1. the canonical "Long-phase heartbeat duty" block exists and carries the
   literal prefix, both commands, the <=45-min wait cap, and the
   verify-first shape;
2. cross-surface prefix parity (SKILL.md duty block ==
   `autonomous_session_watch._LONG_PHASE_HEARTBEAT_PREFIX` ==
   `tick_triage.LONG_PHASE_HEARTBEAT_PREFIX`);
3. the leash / window constants the duty prose quantifies against;
4. the Step 9b C3 cheap-band + autonomous step-6 dispatch sites both
   instruct an ARM-GUARDed `/issue-tick` re-arm;
5. neither false "stays armed" phrasing survives anywhere in SKILL.md
   (the gate-auto-resolve site uses different, TRUE wording and is the
   deliberate negative control);
6. the loop-entry backstop binds BOTH session modes and the old
   "Interactive liveness backstop" anchor name is gone.

Prose assertions run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a banned literal / required phrase can span lines).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# scripts/ holds autonomous_session_watch.py and tick_triage.py; src/ holds
# the task_workflow package the watcher's label helpers lazy-import (same
# import shape as tests/test_autonomous_session_watch.py).
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import autonomous_session_watch as asw  # noqa: E402
import tick_triage  # noqa: E402

HEARTBEAT_PREFIX = "[long-phase-heartbeat]"

DUTY_HEADING_RE = re.compile(r"\*\*Long-phase heartbeat duty")


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _skill_text() -> str:
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


def _duty_block(text: str) -> str:
    """Slice the duty block: from its heading to the next ``#####`` heading."""
    m = DUTY_HEADING_RE.search(text)
    assert m is not None, "SKILL.md lacks the '**Long-phase heartbeat duty' block (#1207)"
    end = text.find("\n#####", m.start())
    assert end != -1, "duty block is not followed by a ##### heading"
    return text[m.start() : end]


def _region(text: str, start_anchor: str, end_anchor: str) -> str:
    start = text.find(start_anchor)
    assert start != -1, f"anchor {start_anchor!r} not found in SKILL.md"
    end = text.find(end_anchor, start)
    assert end != -1, f"anchor {end_anchor!r} not found after {start_anchor!r}"
    return text[start:end]


def test_heartbeat_duty_block_present() -> None:
    """The canonical duty block exists with its load-bearing pieces."""
    block = _norm(_duty_block(_skill_text()))
    assert HEARTBEAT_PREFIX in block
    assert "session_progress_report.py" in block
    # The <=45-min single-wait cap (rendered as "≤45 min" / "45-min" prose).
    assert re.search(r"(≤\s*45|45[- ]min)", block), "duty block lacks the 45-min wait cap"
    assert "post-marker" in block
    # The verify-first ban: never heartbeat without evidence.
    assert re.search(r"NEVER heartbeat blind", block, re.IGNORECASE)


def test_heartbeat_prefix_parity_across_surfaces() -> None:
    """The literal prefix is identical across SKILL.md duty block + both scripts."""
    block = _norm(_duty_block(_skill_text()))
    assert HEARTBEAT_PREFIX in block
    assert asw._LONG_PHASE_HEARTBEAT_PREFIX == HEARTBEAT_PREFIX
    assert tick_triage.LONG_PHASE_HEARTBEAT_PREFIX == HEARTBEAT_PREFIX


def test_leash_constants_match_prose() -> None:
    """The duty prose's quantitative claims track the consumer constants."""
    assert asw.LONG_PHASE_HEARTBEAT_FRESH_S_DEFAULT == 90 * 60
    assert tick_triage.LONG_PHASE_HEARTBEAT_FRESH_MIN_DEFAULT == 90.0
    # The duty prose's 60-min figures (self-report window / cadence) pin to
    # the stalled-detector signal-1 window alongside the 90-min leash.
    assert asw.STALLED_WINDOW_S_DEFAULT == 60 * 60
    block = _norm(_duty_block(_skill_text()))
    assert "90" in block, "duty block does not mention the 90-min leash"


def test_followup_dispatch_sites_rearm() -> None:
    """C3 cheap-band + autonomous step-6 both instruct the tick re-arm."""
    text = _skill_text()
    c3 = _norm(_region(text, "\nC3.", "\nC4."))
    step6 = _norm(_region(text, "Branch on the `same` partition", "**Step R"))
    for name, region in (("C3 cheap-band", c3), ("autonomous step-6", step6)):
        assert "/issue-tick" in region, f"{name} region lacks an /issue-tick re-arm"
        assert "ARM-GUARD" in region or "re-arm" in region, (
            f"{name} region lacks an ARM-GUARD / re-arm instruction"
        )


def test_no_stays_armed_false_claim() -> None:
    """Neither false 'cron stays armed' phrasing survives (wrap-tolerant).

    The gate-auto-resolve site ('stays armed and the bg-Bash poll chain
    continues') is TRUE — no teardown ran there — and deliberately survives:
    only the two dispatch-path phrasings are banned.
    """
    norm = _norm(_skill_text())
    assert "stays armed and drives the loop" not in norm
    assert "stays armed; it drives the loop" not in norm
    # Round-1 code-review sibling (routing-summary bullet): the proposer does
    # NOT fire "before CRON-TEARDOWN" — teardown already ran at the
    # awaiting_promotion transition; the dispatch paths re-arm instead.
    assert "after auto-merge, before CRON-TEARDOWN" not in norm


def test_loop_liveness_backstop_both_modes() -> None:
    """The loop-entry backstop heading binds both modes; the old name is gone."""
    text = _skill_text()
    heading_lines = [ln for ln in text.splitlines() if ln.startswith("**Loop liveness backstop")]
    assert heading_lines, "SKILL.md lacks the '**Loop liveness backstop' heading"
    assert any("BOTH session modes" in ln for ln in heading_lines), (
        "'BOTH session modes' missing from the Loop liveness backstop heading line"
    )
    # The retired anchor name must not survive anywhere (wrap-tolerant).
    assert "Interactive liveness backstop" not in _norm(text)
