"""Pin the committed-script workload-command prescription (#1562).

Task #1562 added a paragraph to the /issue SKILL.md backend-dispatch
section (after composition rules (e)-(i)) prescribing committed branch
scripts -- never inline `python -c` one-liners -- as `--workload-cmd`
bodies for ad-hoc probe dispatches (incident #1482, 2026-07-19; sibling
of the gotchas.md #1310 inline-stdin entry). These pins keep the
prescription from silently drifting out of the file. Assertions run on
whitespace-normalized text so prose re-wrapping never breaks a pin.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import read_workflow_doc

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"


def _normalized(path: Path) -> str:
    """File text with all whitespace runs collapsed to single spaces."""
    return re.sub(r"\s+", " ", read_workflow_doc(path))


def test_committed_script_workload_cmd_prescription_pinned() -> None:
    text = _normalized(ISSUE_SKILL)
    # The prescription lead-in.
    assert "Ad-hoc probe workloads are committed scripts invoked by path" in text
    # The named anti-pattern.
    assert "never inline interpreter one-liners in `--workload-cmd`" in text
    # The incident citation.
    assert "incident #1482, 2026-07-19" in text
    # The recovery recipe.
    assert "rewrite as a committed branch script, push, re-dispatch by path" in text
    # The cross-link to the sibling gotchas entry (signature drift, not quoting).
    assert "#1310 inline-stdin entry" in text
    # The standing exception stays named (experimenter.md sentinel append).
    assert "fixed `write_completion_sentinel` append" in text
