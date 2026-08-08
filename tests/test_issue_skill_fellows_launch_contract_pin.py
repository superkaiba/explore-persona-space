"""Durability pin (#2161): SKILL.md Step 6b carries the fellows
still-waiting launch contract.

Pins the three load-bearing clauses of the free-lane (fellows QoS ladder)
exit-75 contract in ``.claude/skills/issue/SKILL.md`` Step 6b — prose a
future compaction / rewrite could silently drop while the code
(``dispatch_issue.py``'s ``free_lane_park_budget_reached`` arm +
``router.py``'s ``Lease.free_lane_park_state`` park persistence) keeps
producing the exit:

1. the literal reason token ``free_lane_park_budget_reached`` (the third
   exit-75 producer);
2. the probe-before-relaunch launch-recovery invariant: a
   ``squeue --name`` queue probe with the handle-sidecar path pattern
   (``issue-<N>-handle.json``) NEAR it (same recovery block) — the #1336
   shape: a SIGTERMed launcher left job 4684 queued with no marker, and
   the recovery is probe-then-re-run, never a blind double-submit;
3. the "never hand off to backend_poll while still_waiting" clause
   (SLURM PENDING polls as running there; the QoS ladder would stall).

Registered in ``scripts/select_step9c_tests.py`` ``WORKFLOW_INVARIANT``
(same change), so any SKILL.md edit routes here at the Step 9c gate.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

#: Chars of context around each ``squeue --name`` occurrence within which
#: the sidecar path must appear for clause (2). Sized well under the
#: distance between the PRE-#2161 stray co-occurrences (the Step 6d
#: ``reconnect_fn`` mention sits ~4.5k chars from the nearest sidecar
#: mention), so the pin fails on the base tree — demonstrated via
#: ``git show <base>:.claude/skills/issue/SKILL.md`` at implementation
#: time — and passes only with the Step 6b recovery block present.
_PROXIMITY_CHARS = 1500


def test_skillmd_carries_fellows_still_waiting_contract():
    text = SKILL_MD.read_text(encoding="utf-8")

    # (1) the third exit-75 producer's literal reason token.
    assert "free_lane_park_budget_reached" in text, (
        "SKILL.md Step 6b lost the free_lane_park_budget_reached exit-75 producer (#2161)"
    )

    # (2) probe-before-relaunch: at least one `squeue --name` occurrence
    # with the handle-sidecar path pattern within the same recovery block.
    matches = [m.start() for m in re.finditer(re.escape("squeue --name"), text)]
    assert matches, "SKILL.md lost every `squeue --name` probe mention (#2161)"
    assert any(
        "issue-<N>-handle.json" in text[max(0, pos - _PROXIMITY_CHARS) : pos + _PROXIMITY_CHARS]
        for pos in matches
    ), (
        "SKILL.md Step 6b lost the launch-recovery invariant: no "
        "`squeue --name` probe with the `issue-<N>-handle.json` sidecar "
        "probe near it (#2161 / the #1336 killed-launcher shape)"
    )

    # (3) the never-hand-off-to-backend_poll-while-still_waiting clause.
    norm = text.replace("`", "").lower()
    assert re.search(r"never hand off to\s+backend_poll", norm), (
        "SKILL.md Step 6b lost the 'NEVER hand off to backend_poll' clause (#2161)"
    )
    hand_off = norm.index("never hand off to")
    assert "still_waiting" in norm[hand_off : hand_off + 300], (
        "the backend_poll hand-off ban is no longer scoped to the still_waiting state (#2161)"
    )
