"""Pin the Write-tool merged-note compose instruction (#1756).

The three #1725 `epm:merged` "VIA THE `--file` CHANNEL" sites in
`.claude/skills/issue/SKILL.md` (Step 10d safe-case Success bullet,
artifact-confirmed, surgical-additive) mandate posting via `--file` but
previously said the scratch note file was "written in the SAME shell
block" — sessions reached for a Bash heredoc, whose body re-enters
`guard_repo_root_branch.sh`'s argv-prose scan whenever the fail-closed
#1058 `strip_heredoc_bodies()` refuses (the common merged-note shape:
unquoted tag + `$( )` expansion in the body), so git-verb prose in the
note blocked the whole Bash call (incident: session 513fca53 / #1729,
2026-07-27).

This pin asserts the compose mechanism is the WRITE tool at all three
sites plus the CLAUDE.md #1722 recipe paragraph, and that the removed
"written in the SAME shell block" phrasing never returns.

Follows `tests/test_issue_skill_step2_floor.py`'s repo-root file
resolution + grep-anchored existence-check pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = ROOT / "CLAUDE.md"

FILE_CHANNEL_TOKEN = "VIA THE `--file` CHANNEL"
REMOVED_PHRASE = "written in the SAME shell block"
LOOKAHEAD_LINES = 15


def test_write_tool_compose_pins():
    """(a) removed phrase gone; (b) 'write tool' near every --file site; (c) CLAUDE.md sentence."""
    skill_text = issue_skill_text()

    # (a) The old "same shell block" compose instruction must be GONE —
    # whitespace-normalized, because the phrase was line-wrapped
    # differently at each of the three sites (a plain substring check
    # passes vacuously against the pre-edit file).
    normalized = re.sub(r"\s+", " ", skill_text)
    assert REMOVED_PHRASE not in normalized, (
        f"{REMOVED_PHRASE!r} is back in SKILL.md — the merged-note scratch "
        "file must be composed via the Write tool, never inside a Bash "
        "shell block (heredoc bodies re-enter the guard's argv-prose scan "
        "when the fail-closed #1058 strip refuses; #1756)."
    )

    # (b) Every "VIA THE `--file` CHANNEL" site carries a nearby
    # Write-tool compose mention (case-insensitive, within its bullet).
    # The window is whitespace-joined so a line-wrapped "Write / tool"
    # still matches.
    lines = skill_text.splitlines()
    site_lines = [i for i, line in enumerate(lines) if FILE_CHANNEL_TOKEN in line]
    assert len(site_lines) >= 3, (
        f"Expected >=3 {FILE_CHANNEL_TOKEN!r} sites in SKILL.md, found "
        f"{len(site_lines)} — the #1725 --file mandate sites moved or were "
        "reworded; update this pin alongside the change."
    )
    for i in site_lines:
        window = " ".join(lines[i : i + LOOKAHEAD_LINES + 1])
        assert re.search(r"write\s+tool", window, flags=re.IGNORECASE), (
            f"SKILL.md line {i + 1}: the {FILE_CHANNEL_TOKEN!r} site is not "
            f"followed within ~{LOOKAHEAD_LINES} lines by a 'Write tool' "
            "compose mention — each --file site must say the note file is "
            "composed via the Write tool, never a Bash heredoc/printf "
            "(#1756)."
        )

    # (c) The CLAUDE.md #1722 recipe paragraph carries the instruction.
    claude_text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "Compose that file via the WRITE tool" in claude_text, (
        "CLAUDE.md's #1722 marker-note recipe paragraph must instruct "
        "composing the --file note body via the WRITE tool (#1756)."
    )
