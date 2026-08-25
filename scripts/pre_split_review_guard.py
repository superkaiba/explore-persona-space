"""Pre-split completeness guard for the /issue Step 5 review-dispatch site (#2158).

One Bash call — ``uv run python scripts/pre_split_review_guard.py <N>`` — run
VM-side by the /issue orchestrator BEFORE any reviewer dispatch (the same
place ``tick_triage.py`` runs; read-only: no task mutation, no branch-guard
concern). Reads the task's events through the task-workflow library, runs
``pre_split_review_gate``, prints ONE lead-token line, and exits:

    0  REVIEW-OK              — no pre-split state in flight (or an
                                implementation marker postdates every signal);
                                the review dispatch may proceed.
    2  PRE-SPLIT-INCOMPLETE   — the latest #1810 pre-split signal (a
                                breadcrumb with a non-empty remaining list, or
                                a unit-scoped implementing stage-dispatch) has
                                no later implementation marker: do NOT
                                dispatch a review; re-dispatch the REMAINING
                                units instead.
    3  BREADCRUMB-UNPARSEABLE — a recognized breadcrumb candidate carries no
                                parseable same-line remaining field: fail
                                loud, repost the breadcrumb in the documented
                                grammar, never treat as OK.
    4  IMPLEMENTER-MARKER-MISSING — no implementation-class marker
                                (epm:experiment-implementation / epm:results)
                                exists in canonical events (#2294; incident
                                #2290 round 1): do NOT dispatch a review —
                                post the round's implementer marker from the
                                implementer's returned report FIRST, then
                                re-run the guard.

Errors (unknown task id, unreadable registry) propagate loud — a crash is
never read as REVIEW-OK. Incidents: #1336 r4 (premature Unit-A review
dispatch — 2 subagent deaths + a 2-day park), #2061 (lettered breadcrumbs
a digits-only parser fails open on), and #2290 r1 (a review dispatched with
zero implementer markers — the whole round bought only the absence finding).
"""

from __future__ import annotations

import argparse
import sys

from explore_persona_space.task_workflow import list_events, pre_split_review_gate

_EXIT_FOR_VERDICT = {
    "REVIEW-OK": 0,
    "PRE-SPLIT-INCOMPLETE": 2,
    "BREADCRUMB-UNPARSEABLE": 3,
    "IMPLEMENTER-MARKER-MISSING": 4,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``<N>``, run the gate, print one lead-token line, return 0/2/3/4."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "issue", type=int, help="task number (the integer naming tasks/<status>/<N>/)"
    )
    args = parser.parse_args(argv)
    result = pre_split_review_gate(list_events(args.issue))
    verdict = result["verdict"]
    if verdict == "PRE-SPLIT-INCOMPLETE":
        remaining = result["remaining"] or (
            "(none recorded — arm B: unit-scoped implementing dispatch in flight)"
        )
        print(
            f"PRE-SPLIT-INCOMPLETE — remaining: {remaining}; {result['reason']}; "
            "re-dispatch the REMAINING units "
            "(08-step-4.md § Pre-split multi-deliverable builds)"
        )
    elif verdict == "BREADCRUMB-UNPARSEABLE":
        print(f"BREADCRUMB-UNPARSEABLE — {result['reason']}")
    elif verdict == "IMPLEMENTER-MARKER-MISSING":
        print(
            f"IMPLEMENTER-MARKER-MISSING — {result['reason']} "
            "(09-step-5.md § Pre-split completeness guard, #2294)"
        )
    else:
        print(f"REVIEW-OK — {result['reason']}")
    return _EXIT_FOR_VERDICT[verdict]


if __name__ == "__main__":
    sys.exit(main())
