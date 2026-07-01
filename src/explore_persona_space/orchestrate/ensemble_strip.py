"""Pure decision helpers for the ensemble-review cap-hit + git-provenance strip.

These functions make the two most error-prone new predicates of the #784
policy change executable-testable, so the SKILL.md prose has a concrete
contract behind it:

- :func:`should_strip_git_provenance` — the code-review-site-only
  git-provenance strip decision (SKILL.md Step 5c-bis + code-reviewer.md
  Step 0.9). Returns True ONLY when git evidence CONFIRMS the flagged
  finding is not introduced by the round's diff. Ambiguous or
  contradictory evidence → False (leave the FAIL in place — evidence-based,
  never a blanket ignore).
- :func:`resolve_cap_hit` — the cap-5 terminal decision (SKILL.md Steps 5d /
  9a / 9a-bis). Returns the action the orchestrator takes when the review
  loop reaches the cap without a PASS: continue (all residual stripped),
  surface to the user (interactive), or block (autonomous, substantive
  residual remains).

Both functions are pure (no I/O) — the caller runs the actual git probes and
reads the mode, then hands the booleans here. This is the executable half of
the #784 contract; the prose lives in `.claude/skills/issue/SKILL.md`.
"""

from __future__ import annotations

__all__ = [
    "GIT_PROVENANCE_SUBCLASSES",
    "resolve_cap_hit",
    "should_strip_git_provenance",
]

# The three declared git-provenance subclasses a code-reviewer FAIL may carry
# on its ``**Git-provenance subclass:**`` line (code-reviewer.md Step 0.9).
GIT_PROVENANCE_SUBCLASSES: frozenset[str] = frozenset(
    {
        "pre-existing-on-trunk",
        "stale-main-or-worktree",
        "cumulative-main-head-diff",
    }
)


def should_strip_git_provenance(
    subclass: str,
    git_says_pre_existing: bool,
    git_says_round_touched_flagged_lines: bool,
) -> bool:
    """Decide whether a ``git-provenance``-tagged blocker is stripped.

    The strip fires ONLY when the read-only git probe CONFIRMS the flagged
    finding is not introduced by this round's diff. Concretely, both must
    hold:

    - ``git_says_pre_existing`` is True — the probe matching ``subclass``
      confirmed the flagged state exists independent of the round's diff
      (present on trunk / the branch never touched the file / the line is
      unchanged in the round's own commit range).
    - ``git_says_round_touched_flagged_lines`` is False — the round's OWN
      commit range did NOT touch the flagged lines. If git shows the round
      introduced the state, the strip does NOT fire (the FAIL stands).

    Ambiguous or contradictory evidence (both flags False, or both True) →
    False: the strip is evidence-based and defaults to leaving the FAIL in
    place, never a blanket ignore.

    Args:
        subclass: the declared ``**Git-provenance subclass:**`` value. Must be
            one of :data:`GIT_PROVENANCE_SUBCLASSES`; any other value (a
            malformed / absent subclass line) returns False — an
            unverifiable blocker is never stripped.
        git_says_pre_existing: probe confirmed the flagged state is not from
            this round's diff.
        git_says_round_touched_flagged_lines: the round's own commit range
            touched the flagged lines (git says the round introduced it).

    Returns:
        True iff the blocker should be stripped (git confirms pre-existence
        AND the round did not touch the flagged lines); False otherwise.
    """
    if subclass not in GIT_PROVENANCE_SUBCLASSES:
        return False
    if git_says_round_touched_flagged_lines:
        # Git says the round introduced it — the FAIL stands regardless of the
        # pre-existing flag (contradictory evidence resolves to "do not strip").
        return False
    return git_says_pre_existing


def resolve_cap_hit(all_residual_stripped: bool, autonomous: bool) -> dict[str, str]:
    """Decide the cap-5 terminal action for an ensemble review loop.

    At the cap (round 5) with a non-PASS ensemble verdict, the orchestrator
    applies the full strip once more, then:

    - ``all_residual_stripped`` True → CONTINUE: every residual blocker was a
      false positive (mechanical / git-provenance / procedural), so treat as
      PASS and advance.
    - ``all_residual_stripped`` False (a substantive residual remains) →
      SURFACE, never ship past:

      * interactive (``autonomous`` False) → present the residual to the user
        and EXIT awaiting their decision (``surface_interactive``).
      * autonomous (``autonomous`` True) → post ``epm:failure v1``
        ``failure_class: code``, set ``status: blocked``, notify, tear down
        the tick cron, and EXIT (``block_autonomous``).

    Args:
        all_residual_stripped: True iff every residual blocker at the cap was
            stripped (no substantive finding remains).
        autonomous: True in an ``EPM_AUTONOMOUS_SESSION=1`` session.

    Returns:
        A dict with ``action`` in {``continue``, ``surface_interactive``,
        ``block_autonomous``} and a one-line ``reason``.
    """
    if all_residual_stripped:
        return {
            "action": "continue",
            "reason": "cap-5: all residual blockers stripped (false-positive / "
            "mechanical / git-provenance / procedural) → treat as PASS and continue",
        }
    if autonomous:
        return {
            "action": "block_autonomous",
            "reason": "cap-5: substantive residual remains → post epm:failure v1 "
            "failure_class: code + status:blocked + notify + CRON-TEARDOWN (never ship past)",
        }
    return {
        "action": "surface_interactive",
        "reason": "cap-5: substantive residual remains → present to user + EXIT "
        "(never ship past, never pivot-loop)",
    }
