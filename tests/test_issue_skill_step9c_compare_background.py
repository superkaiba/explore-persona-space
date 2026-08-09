"""Pin: five gate-launch sites use the DETACHED launcher (#1893, task #2005).

The Step 9c gate + Step 10d lint gate all launch DETACHED via the § Harvest
self-harvest chaining shape: `setsid nohup bash -c '<workload>; echo $? > rc'
& echo $!`. The outer bg-Bash captures the workload pid; the inner rc-write
lives inside the SAME session-decoupled unit, so the 600s bg-Bash tool cap
can NEVER kill the pytest process. Before this fix, an inline
`timeout ... > log; echo $? > rc` chain died at the ~10-min cap BEFORE the
rc-write (#1893's "exit 144, rc file missing" signature). The old
`COMPARE_OUT=$(uv run python ...)` command-substitution shape remains banned.

The anchor for each region points AT the launcher block itself so a distant
prose-only anchor cannot mask a missing recipe token.
"""

import re
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude/skills/issue/SKILL.md"

# Each entry: (region-name, anchor string that begins the region, region-len).
# The anchor STARTS a span the pin assertions run over. Anchors are chosen so
# the setsid launcher recipe (or its documentation) sits INSIDE the span.
_REGIONS: list[tuple[str, str, int]] = [
    # Step 9c step 1b — touched-scope gate. Anchor at the launcher comment.
    (
        "step9c-1b",
        "# ONE background Bash call (run_in_background=true) captures ITS pid",
        4000,
    ),
    # Step 9c step 1c — full-scope override. Anchor at the section header
    # (the code block itself sits within 4000 chars of it).
    (
        "step9c-1c",
        "Scope override: if the plan-body frontmatter has `test_scope: full`",
        4000,
    ),
    # Step 9c step 1d — known-red-on-main compare. Anchor at the compare
    # prose (mentions the outer bg-Bash) and window covers the compare
    # launcher block (7500: the #2024 paired-invocation arithmetic merged
    # into this region puts the breadcrumb at ~6420 chars from the anchor).
    (
        "step9c-1d",
        "Run compare as a DETACHED background",
        7500,
    ),
    # Step 10d — form (i) safe case / (ii) recovery shared block. Anchor at
    # the executable-block comment; the recipe documentation sits inside.
    (
        "step10d-i-ii",
        "EXECUTABLE gate — forms (i) safe case and (ii) recovery share this block",
        4000,
    ),
    # Step 10d — form (iii) surgical additive.
    (
        "step10d-iii",
        "earlyoom-protect the gate — form (iii)",
        4000,
    ),
]

# The broken splice shape the critic caught in plan v1 (banned everywhere in
# the file, not just the launch regions; a top-level `; echo $? > <rc> & echo`
# after `2>&1` parses as three commands and defeats setsid). BOTH spellings —
# with and without a space before the `;` (`2>&1 ;` / `2>&1;`) — are banned
# (#2005 r1 m3: the old inline shape used the no-space spelling). Kept as raw
# string literals so the assertion itself never trips on incidental prose.
_BROKEN_SPLICES = ("2>&1 ; echo $? > /tmp/step9c-", "2>&1; echo $? > /tmp/step9c-")

# The inline-era completion trigger (#2005 r1 M1): under the detached launcher
# the LAUNCHER bg-Bash completes in seconds, so the harness notification
# arrives long before the rc/verdict file can exist — a reader following this
# sentence misdiagnoses a HEALTHY running gate as dead.
_STALE_INLINE_TRIGGER = "the harness notifies"


def _region(text: str, anchor: str, length: int) -> str:
    start = text.index(anchor)
    return text[start : start + length]


def test_all_five_gate_sites_use_detached_launcher():
    """Every launch site carries the setsid launcher + pid capture + harvest+rc breadcrumb."""
    text = SKILL.read_text(encoding="utf-8")
    for name, anchor, length in _REGIONS:
        region = _region(text, anchor, length)
        assert "setsid" in region, f"[{name}] missing setsid launcher"
        assert ("PYTEST_PID" in region) or ("COMPARE_PID" in region), (
            f"[{name}] missing PYTEST_PID / COMPARE_PID capture"
        )
        assert "harvest=" in region, f"[{name}] missing harvest= breadcrumb token"
        # The inner rc-write must sit inside the harvest-chained bash -c,
        # written to a /tmp/step9c-* sentinel. The escape form (`\$?`) is
        # what a Bash-tool string carries when the outer double-quoted
        # wrapper defers `$?` into the inner shell.
        assert ("echo $? > /tmp/step9c-" in region) or ("echo \\$? > /tmp/step9c-" in region), (
            f"[{name}] missing inner rc-write to /tmp/step9c-*"
        )
        # Outer bg-Bash is still a run_in_background=true call; the launcher
        # wraps the workload.
        assert "run_in_background" in region, (
            f"[{name}] missing run_in_background=true outer bg-Bash reference"
        )


def test_broken_splice_shape_is_absent():
    """NEGATIVE: the critic-caught `... 2>&1 ; echo $? > /tmp/step9c-... & echo $!`
    splice must not appear anywhere in the SKILL — it parses as three commands
    where pytest runs FOREGROUND inside the $( ) capture and the outer bg-Bash
    still dies at the 600s tool cap. The § Harvest NEVER-splice rule (this file
    is that rule's pin). Both spacings (`2>&1 ;` and `2>&1;`) are matched."""
    text = SKILL.read_text(encoding="utf-8")
    for splice in _BROKEN_SPLICES:
        assert splice not in text, (
            f"broken splice `{splice}` (top-level `;` after `2>&1` followed by "
            "` & echo $!`) is the exact #1893 failure the detached recipe fixes"
        )


def test_step1d_foreground_prescription_gone():
    """Historical: the pre-#1197 foreground compare shapes must stay gone."""
    text = SKILL.read_text(encoding="utf-8")
    assert "short/bounded foreground" not in text
    assert "COMPARE_OUT=$(uv run python" not in text  # old command substitution


def _norm(text: str) -> str:
    """Collapse all whitespace so wrapped prose sentences match one-line pins."""
    return " ".join(text.split())


def test_stale_inline_completion_trigger_gone():
    """NEGATIVE (#2005 r1 M1): the inline-era completion-trigger sentence —
    "When the background call completes (the harness notifies), read ..." —
    described the OLD shape where bg-call completion == workload completion.
    Under the detached launcher the LAUNCHER bg-Bash completes in seconds, so
    that sentence directs a premature read: rc missing -> the FATAL branch
    asserts the gate DIED -> kill-before-relaunch against a HEALTHY running
    gate (and, pre-fix, an unconditional basetemp reap under it). The phrase
    must be gone file-wide; each of the four gate completion-reads instead
    carries the detached-semantics replacement (launcher completion is NOT
    the done signal; missing rc + LIVE probe match = STILL RUNNING)."""
    text = _norm(SKILL.read_text(encoding="utf-8"))
    assert _STALE_INLINE_TRIGGER not in text, (
        "the inline-era 'harness notifies' completion trigger is the #2005 r1 M1 "
        "stale-prose bug — the detached gate's launcher completes in seconds"
    )
    # Positive replacement pins — one per gate completion-read site:
    assert text.count("is NOT the gate-done signal") == 2, "1b + 10d (i)/(ii)"
    assert text.count("is NOT the compare-done signal") == 1, "1d compare"
    assert text.count("is NOT the done signal for the sequence") == 1, "10d (iii)"
    # The still-running branch must exist at every rewritten site (plus the
    # pre-existing single-flight prose uses the same token):
    assert text.count("STILL RUNNING") >= 4


def test_step1d_compare_completion_read_pins():
    """Re-pins the two 1d coverage strings the retired inline-shape test
    carried (#2005 r1 m2): the stale compare-triplet reap at launch, and the
    missing-rc NEVER-record-PASS rule in the completion-read."""
    text = SKILL.read_text(encoding="utf-8")
    region = _region(text, "Run compare as a DETACHED background", 10000)
    assert "rm -f /tmp/step9c-compare-issue-<N>.json" in region, (
        "1d must reap the stale compare triplet before launching"
    )
    assert "NEVER record PASS" in region, "1d's missing-rc branch must forbid recording PASS"


def test_basetemp_reap_inside_rc_exists_branch():
    """The 1b completion-read's BASETEMP reap sits INSIDE the rc-exists `else`
    branch (#2005 r1 M1 part 2): a premature completion-read against a LIVE
    gate must never `rm -rf` the gate's basetemp out from under it. Pre-fix
    the reap sat unconditionally AFTER the outer `fi`, so from the reap line
    to the end of the block only ONE bare `fi` line remained; the fixed
    layout closes the reap-if AND the outer rc-exists if (two `fi` lines)."""

    text = SKILL.read_text(encoding="utf-8")
    blocks = [
        b
        for b in text.split("```")
        if "[ ! -f /tmp/step9c-rc-issue-<N> ]" in b and 'rm -rf "$BT"' in b
    ]
    assert len(blocks) == 1, "expected exactly the 1b completion-read block"
    tail = blocks[0][blocks[0].index('rm -rf "$BT"') :]
    closing_fi = re.findall(r"^\s*fi\s*$", tail, re.M)
    assert len(closing_fi) >= 2, (
        "BASETEMP reap must live inside the rc-exists else branch (reap-if fi + "
        "outer if fi both after the reap), not after the outer fi"
    )
