"""TDD Phase 1 — recipe-fix port + data-prep invariants (task #397, plan v4 §5.1.0 + §14 item 5).

Plan v4 §5.1.0 is the precondition: port the three recipe-fix commits from
``task-365-recipe-fix-v1`` (commit ``32ce24ef``) before the marker/E-level
edits land. Without them, source rate sits at the floor and H1 is
structurally unmeasurable (plan v4 §1 + §10).

The three load-bearing pieces of the port:

  1. **B-suffix stripped** from training-row ``user_text`` (bare question
     only) — see ``build_user_text_strip_b_suffix``.
  2. **400 positives + 400 negatives = 800 rows per cell** — see
     ``DEFAULT_POS_PER_SOURCE`` / ``DEFAULT_NEG_PER_SOURCE``.
  3. **Marker is threaded into ``append_marker(answer, marker_text="※")``** —
     the appended token is the runtime marker, not a hard-coded default.

CPU-only; no model load.
"""

from __future__ import annotations

from explore_persona_space.experiments.factor_screen_397.data_prep import (
    DEFAULT_NEG_PER_SOURCE,
    DEFAULT_POS_PER_SOURCE,
    DEFAULT_ROWS_PER_CELL,
    append_marker,
    build_user_text_strip_b_suffix,
)


def test_default_rows_per_cell_is_800() -> None:
    """Plan v4 §3 + §4.2: 400 positives + 400 negatives = 800 rows per cell.

    The recipe-fix port bumped ``--pos-per-source`` from 200 to 400. Any
    regression to 200 would put #397 back on the #365-pre-fix floor.
    """
    assert DEFAULT_POS_PER_SOURCE == 400
    assert DEFAULT_NEG_PER_SOURCE == 400
    assert DEFAULT_ROWS_PER_CELL == 800


def test_append_marker_uses_threaded_marker_not_default_zlt() -> None:
    """Marker threaded into append_marker — the appended token is ※, not [ZLT]."""
    result = append_marker("My answer.", marker_text="※")
    assert result.endswith("\n\n※"), f"Expected ※ appended; got {result!r}"
    assert "[ZLT]" not in result


def test_append_marker_is_idempotent_when_marker_already_present() -> None:
    """If the marker is already in the answer, don't double-append."""
    answer_with_marker = "Answer text.\n\n※"
    result = append_marker(answer_with_marker, marker_text="※")
    # The trailing ※ must appear exactly once.
    assert result.count("※") == 1


def test_b_suffix_is_stripped_from_training_user_text() -> None:
    """Plan v4 §4.2 + recipe-fix: training-row user_text must NOT contain the B-suffix.

    The user supplies the bare question to the model at training time. The
    B-suffix (length-instruction string) belongs only in pool-gen prompts.
    """
    bare = build_user_text_strip_b_suffix(
        question="What do you do for a living?",
        b_suffix="Answer in roughly 1000 words.",
    )
    assert "1000 words" not in bare, (
        f"B-suffix must be stripped from training user_text; got: {bare!r}"
    )
    assert bare.strip() == "What do you do for a living?"


def test_b_suffix_empty_keeps_bare_question_unchanged() -> None:
    """B=0 cells have an empty suffix — the bare question survives intact."""
    bare = build_user_text_strip_b_suffix(
        question="What do you do for a living?",
        b_suffix="",
    )
    assert bare.strip() == "What do you do for a living?"
