"""Pin the corrected setsid reparent-target prose in `/issue` SKILL.md (#2200).

Incident #2199/#2200: the detached-phases block asserted as fact that a
`setsid`-detached phase "reparents to PID 1 when the launching shell exits".
That is false on this VM — the orphan is adopted by the user-level `systemd`
(a `PR_SET_CHILD_SUBREAPER` process in every session's ancestry), not PID 1 —
and prose stating the PID-1 mechanism led an agent to write a predicate keyed
on `ppid == 1`, which is environment-dependent and made the Step 9c gate red
on `main` (the #2199 incident). #2200 restates the GUARANTEE (adoption
strictly ABOVE the dead session, so a ppid-tree walk down from that session
cannot reach the phase) instead of the over-specified mechanism.

Two pins, per plan #2200 §4 Edit 2 / §5:

1. NEGATIVE (whole file, primary): SKILL.md must never again claim a
   `reparent... to PID 1` target. Forbids exactly the false claim and
   constrains future rewording essentially not at all.
2. POSITIVE (detached block only): the guarantee must not be silently
   dropped — the block keeps the `subreaper` / `ppid-tree walk` / `ppid == 1`
   tokens (the last as the named-wrong-predicate warning). Matched under a
   whitespace-collapse so a future re-wrap across a line break cannot
   false-fail the pin (the `_norm()` convention from
   `tests/test_issue_skill_detached_harvest_pin.py`).

Selector reachability: this file references `.claude/skills/issue/SKILL.md`
via the path-join below, so the Step 9c selector's #1851 skills-pin
discovery arm auto-selects it whenever SKILL.md is touched (deliberately NOT
a `WORKFLOW_INVARIANT` member — plan #2200 §5.1). (The selector script's
filename is deliberately not spelled here: a literal script-name token would
enroll this file in the selector's dependency arm and break its exact-set
live-tree pin.)
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

DETACHED_HEADING = "**Detached VM-side long compute phases"
SUCCESSOR_HEADING = "**Successor / re-entry rule"

PID1_REPARENT_CLAIM = re.compile(r"reparent\w*\s+to\s+PID\s+1", re.IGNORECASE)


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _skill_text() -> str:
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


def _region(text: str, start_anchor: str, end_anchor: str) -> str:
    start = text.find(start_anchor)
    assert start != -1, f"anchor {start_anchor!r} not found in SKILL.md"
    end = text.find(end_anchor, start)
    assert end != -1, f"anchor {end_anchor!r} not found after {start_anchor!r}"
    return text[start:end]


def test_skill_md_does_not_claim_pid1_reparent_target() -> None:
    """NEGATIVE pin: no `reparent... to PID 1` claim anywhere in SKILL.md.

    The adoptive parent of a setsid-detached orphan is the nearest
    child-subreaper ancestor of the launcher (user-level `systemd` on this
    VM), else the pid-namespace init — NOT reliably PID 1. A prose claim of a
    PID-1 reparent target is the seed of the `ppid == 1` predicate bug
    (#2199); it must not return under any wording.
    """
    match = PID1_REPARENT_CLAIM.search(_skill_text())
    assert match is None, (
        f"SKILL.md re-asserts the false PID-1 reparent target: {match.group(0)!r} — "
        "state the adoption-above-the-dead-session guarantee instead (#2199, #2200)"
    )


def test_detached_block_states_the_ppid_walk_guarantee() -> None:
    """POSITIVE pin: the detached block keeps the corrected guarantee.

    Region-scoped to the detached-phases block; whitespace-normalized so a
    re-wrap of a token across a line break cannot false-fail the pin.
    """
    block = _norm(_region(_skill_text(), DETACHED_HEADING, SUCCESSOR_HEADING))
    assert "subreaper" in block, (
        "detached block dropped the child-subreaper adoption mechanism (#2200)"
    )
    assert "ppid-tree walk" in block, (
        "detached block dropped the ppid-tree-walk-cannot-reach-it guarantee (#2200)"
    )
    assert "ppid == 1" in block, (
        "detached block dropped the named-wrong-predicate warning (`ppid == 1` is "
        "environment-dependent; #2199, #2200)"
    )
