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
    # launcher block.
    (
        "step9c-1d",
        "Run compare as a DETACHED background",
        6000,
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
# after `2>&1` parses as three commands and defeats setsid). Kept as a raw
# string literal so the assertion itself never trips on incidental prose.
_BROKEN_SPLICE = "2>&1 ; echo $? > /tmp/step9c-"


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
    is that rule's pin)."""
    text = SKILL.read_text(encoding="utf-8")
    assert _BROKEN_SPLICE not in text, (
        f"broken splice `{_BROKEN_SPLICE}` (top-level `;` after `2>&1` followed by "
        "` & echo $!`) is the exact #1893 failure the detached recipe fixes"
    )


def test_step1d_foreground_prescription_gone():
    """Historical: the pre-#1197 foreground compare shapes must stay gone."""
    text = SKILL.read_text(encoding="utf-8")
    assert "short/bounded foreground" not in text
    assert "COMPARE_OUT=$(uv run python" not in text  # old command substitution
