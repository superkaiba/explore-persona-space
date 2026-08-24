"""Pin a >=1500s floor on the Step 10d per-leg ``workflow_lint.py`` wedge
bounds (#2253 round 5).

The #1212 pre-push lint gate wraps each of its eight ``workflow_lint.py``
legs (baseline + gated sides x no-flags + parity legs, in BOTH the shared
gate block and the surgical block; .claude/skills/issue/steps/18-step-10d.md)
in ``timeout --kill-after=60s <N>s``. A bound kill surfaces as rc 124, which
the NO-DOWNGRADE rc fold routes into the gate's crash arm — fail CLOSED — so
an UNDER-SIZED bound silently blocks EVERY branch's Step 10d merge
fleet-wide, with the cause invisible unless someone counts verdict lines per
leg (the #931 breakage shape). #2253 round 5 measured the no-flags bundle at
747s on the branch tree under fleet load (~663s without that round's check)
against a 900s bound = 1.2x; CLAUDE.md sizes self-set fences at >=2x the
measured wall, so the floor here is 1500s (2x 747s = 1494s, rounded UP to a
stable round number that still sits below the landed 1800s bound). These
tests keep the bound from silently regressing below the x2 rule.

Follows the r4 precedent (test_issue_skill_gate_tree_pathspec.py): parse the
LOGICAL doc via ``tests.issue_skill_source.issue_skill_text()`` so the pin
binds wherever the step body lives. The TG (mapped-test) legs' own bounds are
sized from the selector's ``recommended-timeout-s`` and are deliberately OUT
of scope — the pattern below matches only timeout invocations whose wrapped
command is ``workflow_lint.py``.

NOTE for future step-doc editors: a legitimate reshaping of the gate-leg
invocation (``timeout --kill-after=<K>s <N>s uv run python
"$GT|$REPO_ROOT/scripts/workflow_lint.py"``) must update the pattern here IN
THE SAME COMMIT, or the leg-count anchor assertion goes red.
"""

from __future__ import annotations

import re

from tests.issue_skill_source import issue_skill_text

#: Floor on each per-leg bound: >=2x the 747s no-flags wall measured
#: 2026-08-21 under fleet load (#2253 r5; the CLAUDE.md x2 dispersion rule).
LINT_LEG_TIMEOUT_FLOOR_S = 1500

#: The gate runs 2 legs (no-flags + parity) x baseline/gated x shared-gate/
#: surgical-block = 8 workflow_lint timeout sites.
_EXPECTED_MIN_LEGS = 8

_LEG_BOUND = re.compile(
    r"timeout --kill-after=\d+s (\d+)s uv run python "
    r"\"\$(?:GT|REPO_ROOT)/scripts/workflow_lint\.py\""
)


def _lint_leg_bounds(text: str) -> list[int]:
    """Timeout bounds (seconds) of every workflow_lint gate leg in ``text``.

    Asserts the anchor pattern still finds all expected legs, so a reshaped
    invocation fails loud here instead of silently un-pinning the floor.
    """
    bounds = [int(m.group(1)) for m in _LEG_BOUND.finditer(text)]
    assert len(bounds) >= _EXPECTED_MIN_LEGS, (
        f"found {len(bounds)} workflow_lint timeout legs, expected >= "
        f"{_EXPECTED_MIN_LEGS} — the gate-leg invocation shape moved? "
        "Update the anchors here in the same commit."
    )
    return bounds


def test_every_lint_leg_timeout_bound_meets_floor():
    for bound in _lint_leg_bounds(issue_skill_text()):
        assert bound >= LINT_LEG_TIMEOUT_FLOOR_S, (
            f"Step 10d workflow_lint leg bound {bound}s is below the "
            f"{LINT_LEG_TIMEOUT_FLOOR_S}s floor (>=2x the 747s no-flags wall "
            "measured under fleet load, #2253 r5). An under-sized bound kills "
            "healthy legs with rc 124, which the crash arm reads as a gate "
            "crash — silently blocking every branch's Step 10d merge."
        )
