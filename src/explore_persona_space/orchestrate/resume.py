"""``/issue`` re-entry skip-ahead routing (Symphony §7.3 / §16.6 analog).

Today every ``/issue <N>`` re-invocation re-parses every ``epm:*`` marker
on the issue from Step 0 to find the resume point. For long-lived issues
with 30+ markers, that eats context budget. This module implements the
plan §5 cheap path: at re-entry, read ONLY the latest ``epm:step-completed``
marker; if it claims a clean exit and points at a step whose
``entry_status_label`` matches the current label, jump there directly.
Otherwise fall back to the existing full-replay path.

The router NEVER attempts to skip when:

1. ``status:blocked`` is the current label (a stale clean-exit marker
   must not let the skill dispatch on a manually-blocked issue).
2. No ``epm:step-completed`` marker exists (first invocation, or pre-§5
   in-flight issue).
3. The marker's ``exit_kind`` is ``parked`` or ``failure-exit``.
4. The marker references a step that's been removed from
   ``workflow.yaml`` (rename / drop in flight).
5. The current ``status:*`` label has drifted out from under the marker
   (user manually flipped the label between runs).

Rule 1 fires BEFORE the marker is even consulted — that's the
load-bearing C2.B2 fix from the plan.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepCompletedMarker:
    """Parsed shape of an ``epm:step-completed`` marker comment.

    Field names match the marker body lines verbatim (``step:``, ``at:``,
    ``next_expected_step:``, ``exit_kind:``). The skill's marker-scan
    code is responsible for parsing the comment body into this dataclass;
    this module operates on the parsed shape.
    """

    step: str
    next_expected_step: str
    exit_kind: str  # one of "clean", "parked", "failure-exit"
    at: str = ""  # informational (commit sha or ISO8601), not load-bearing
    notes: str = ""


@dataclass(frozen=True)
class WorkflowStep:
    """Subset of ``.claude/workflow.yaml`` step rows the router needs."""

    id: str
    entry_status_label: tuple[str, ...]


VALID_EXIT_KINDS: frozenset[str] = frozenset({"clean", "parked", "failure-exit"})


def latest_step_completed(
    markers: Iterable[StepCompletedMarker],
) -> StepCompletedMarker | None:
    """Return the most-recently-posted step-completed marker.

    The skill reads Sagan workflow events and walks them bottom-up; the first
    ``epm:step-completed`` marker IS the latest. This helper takes a list
    pre-filtered to that kind and returns the LAST element (or None if empty).
    Kept as a standalone function so unit tests can pass a list directly.
    """
    seq = list(markers)
    if not seq:
        return None
    return seq[-1]


def decide_entry_step(
    *,
    status_label: str,
    markers: Sequence[StepCompletedMarker],
    workflow_steps: Sequence[WorkflowStep],
) -> str | None:
    """Decide whether to skip ahead or fall back to full replay.

    Returns:
        The step ID to jump to, OR ``None`` to signal full replay.

    The precedence rules (in order):

    1. ``status:blocked`` → full replay (rule 1; load-bearing).
    2. No marker → full replay (first invocation).
    3. Non-clean ``exit_kind`` → full replay (parked / failure-exit).
    4. Marker step unknown to ``workflow.yaml`` → log + full replay.
    5. Current status not in target step's ``entry_status_label`` → full replay.
    6. Otherwise → return ``next_expected_step``.
    """
    # Rule 1: status:blocked always wins. Checked BEFORE marker lookup
    # because a manually-set status:blocked must take effect on the
    # very next /issue invocation, regardless of any stale clean-exit
    # marker. This is the C2.B2 fix from plan §5.
    if status_label == "status:blocked":
        logger.info("status:blocked present; full replay (rule 1)")
        return None

    # Rule 2: first run.
    latest = latest_step_completed(markers)
    if latest is None:
        logger.info("no step-completed marker; full replay (rule 2)")
        return None

    # Rule 3: non-clean exit.
    if latest.exit_kind not in VALID_EXIT_KINDS:
        logger.warning(
            "step-completed marker has unknown exit_kind=%r; full replay",
            latest.exit_kind,
        )
        return None
    if latest.exit_kind != "clean":
        logger.info(
            "step-completed exit_kind=%s; full replay (rule 3)",
            latest.exit_kind,
        )
        return None

    next_step = latest.next_expected_step
    by_id = {s.id: s for s in workflow_steps}

    # Rule 4: removed/renamed step.
    if next_step not in by_id:
        logger.warning(
            "step-completed marker references unknown step %r; full replay (rule 4)",
            next_step,
        )
        return None

    # Rule 5: status-marker label mismatch. entry_status_label is a list of
    # bare status names (e.g. "running", "implementing"); the live label
    # is "status:running". Compare with the prefix added.
    allowed = by_id[next_step].entry_status_label
    if not allowed:
        # workflow.yaml lint should have rejected this, but be defensive.
        logger.warning(
            "workflow step %r has empty entry_status_label; full replay",
            next_step,
        )
        return None
    if status_label not in {f"status:{s}" for s in allowed}:
        logger.info(
            "status drift: label=%s, marker expects step %s with allowed=%s; full replay (rule 5)",
            status_label,
            next_step,
            allowed,
        )
        return None

    # All checks passed.
    logger.info(
        "skip-ahead: status=%s marker step=%s exit_kind=clean; jumping to %s",
        status_label,
        latest.step,
        next_step,
    )
    return next_step
