"""Pin the #1875 Step 0 Monitor/TaskOutput schema preload in `/issue` SKILL.md.

Three autonomous sessions on 2026-07-29 (miners F-P6 / #1812, G-P9 / #1738,
H-P7) each burned a wasted turn on an ``InputValidationError`` calling an
unloaded deferred tool (`Monitor` / `TaskOutput`) at first use. #1875 extends
the Step 0 cron-arm block's selective preload (`CronCreate,CronList,CronDelete`)
with a `Monitor,TaskOutput` ToolSearch line INSIDE the
``EPM_AUTONOMOUS_SESSION == "1"`` branch, so every autonomous session loads
the always-needed wait/poll schemas at boot. Interactive sessions stay lazy —
the use-site loads elsewhere in SKILL.md (Long-phase heartbeat duty item 1,
Step 9a-quater LATE JOIN, Step 10d Guard 5) are deliberate keeps.

This test pins, against the Step 0 cron-arm REGION of
`.claude/skills/issue/SKILL.md` (sliced from the "MANDATORY auto-armed
backstop" heading to the "Interactive sessions" paragraph — the lazy use
sites elsewhere in the file also mention `Monitor`/`TaskOutput` loads, so a
file-global grep would pass on a gutted Step 0 block):

1. the `ToolSearch("select:Monitor,TaskOutput")` preload is present;
2. the existing Cron preload (`select:CronCreate,CronList,CronDelete`) is
   still present;
3. ordering — the preload sits AFTER the ``EPM_AUTONOMOUS_SESSION == "1"``
   line (inside/after the autonomous branch opens, not above it).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

REGION_START = "**MANDATORY auto-armed backstop for autonomous sessions — arm it NOW.**"
REGION_END = "Interactive sessions (no `EPM_AUTONOMOUS_SESSION`)"
AUTONOMOUS_BRANCH_LINE = 'if os.environ.get("EPM_AUTONOMOUS_SESSION") == "1":'
MONITOR_PRELOAD = 'ToolSearch("select:Monitor,TaskOutput")'
CRON_PRELOAD = 'ToolSearch("select:CronCreate,CronList,CronDelete")'


def _step0_cron_arm_block() -> str:
    """Slice the Step 0 cron-arm region (heading -> Interactive-sessions para)."""
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.find(REGION_START)
    assert start != -1, f"anchor {REGION_START!r} not found in SKILL.md"
    end = text.find(REGION_END, start)
    assert end != -1, f"anchor {REGION_END!r} not found after the Step 0 heading"
    return text[start:end]


def test_step0_preloads_monitor_taskoutput() -> None:
    """Step 0's autonomous branch preloads the Monitor+TaskOutput schemas (#1875)."""
    block = _step0_cron_arm_block()
    monitor_idx = block.find(MONITOR_PRELOAD)
    assert monitor_idx != -1, (
        "Step 0 cron-arm block lacks the autonomous Monitor/TaskOutput schema"
        f" preload ({MONITOR_PRELOAD!r}) — #1875"
    )
    assert CRON_PRELOAD in block, (
        f"Step 0 cron-arm block lost the existing Cron-tool preload ({CRON_PRELOAD!r})"
    )
    branch_idx = block.find(AUTONOMOUS_BRANCH_LINE)
    assert branch_idx != -1, (
        f"Step 0 cron-arm block lacks the autonomous branch line ({AUTONOMOUS_BRANCH_LINE!r})"
    )
    assert branch_idx < monitor_idx, (
        "the Monitor/TaskOutput preload must sit INSIDE the EPM_AUTONOMOUS_SESSION"
        " branch (after the branch line), not above it:"
        f" branch index {branch_idx} >= preload index {monitor_idx}"
    )
    # Round-1 review Minor: branch_idx < monitor_idx alone false-passes a
    # dedented placement BELOW the whole `if` block (unconditional — violates
    # the autonomous-only constraint). Pin the preload before the branch
    # body's first statement (`jobs = CronList()` is unique in the region;
    # `post = CronList()` does not collide).
    jobs_idx = block.find("jobs = CronList()")
    assert jobs_idx != -1, "Step 0 cron-arm block lacks the ARM-GUARD's `jobs = CronList()` line"
    assert monitor_idx < jobs_idx, (
        "the Monitor/TaskOutput preload must precede the branch body's first"
        " statement (`jobs = CronList()`) — a placement below the `if` block is"
        f" unconditional, violating the autonomous-only constraint:"
        f" preload index {monitor_idx} >= jobs index {jobs_idx}"
    )
