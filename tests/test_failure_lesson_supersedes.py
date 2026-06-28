"""TDD pass-1 tests for the #712 failure-lesson helpers + supersedes retraction.

Three pure helpers are added to ``explore_persona_space.task_workflow`` in the
implementation pass (#712 §4e):

* ``failure_lesson_capture_eligible(block_fields, *, subsequent_distinct_failure)``
  — the AC#2 eligibility seam: a received ``epm:failure-lesson`` block is
  captured when it RESOLVED the failure (``resolved: yes``) OR when it carries
  ``root_cause_confirmed: yes`` (true REGARDLESS of a subsequent distinct
  failure / abandoned pod — the #664 L204 gap).
* ``supersedes_action(prior_ref, durable_texts, new_lesson_ref)`` — the
  locate-and-annotate seam: returns ``{path: annotated_text}`` for every
  durable entry the ``supersedes`` ref matches, prepending the CONCRETE
  ``[SUPERSEDED by <slug> — see #<task_id>] `` marker; ``{}`` on no match.
* ``apply_failure_lesson(block, durable_texts, new_lesson_ref)`` — the pure
  composer: merges the annotations (if any) over a copy of ``durable_texts``
  and APPENDS the new lesson body to the owning-agent memory file, returning
  the FINAL ``{path: text}`` map the orchestrator then writes.

All three are PURE (no I/O); these tests assert them against the production
symbols (never a test-local re-implementation). In TDD pass 1 the imports below
fail because the helpers do not exist yet — that is the expected pre-implementation
state. The tests turn green once the implementation pass lands the helpers.
"""

from __future__ import annotations

from pathlib import Path

from explore_persona_space.task_workflow import (
    apply_failure_lesson,
    failure_lesson_capture_eligible,
    supersedes_action,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "failure_lesson_append_only_pre712.txt"

# The concrete new (correcting) lesson slug + task used across the supersedes
# tests — the real slug + task id (NEVER a ``<pending>`` placeholder, per §4e2).
_NEW_SLUG = "vllm_first_generate_pod_specific_deadlock"
_NEW_TASK = "664"
_NEW_LESSON_BODY = (
    "A vLLM `generate()` deadlock on the FIRST call that reproduces across "
    "batch sizes is a pod-hardware fault, not a code bug: differential-diagnose "
    "on the SAME pod before reprovisioning."
)
_MEMORY_PATH = (
    ".claude/agent-memory/experiment-implementer/"
    "feedback_vllm_first_generate_pod_specific_deadlock.md"
)


# ---------------------------------------------------------------------------
# failure_lesson_capture_eligible — the AC#2 eligibility seam.
# ---------------------------------------------------------------------------


def test_failure_lesson_capture_eligible_resolved():
    """A block that RESOLVED the failure is eligible regardless of a
    following distinct failure (the original resolve-only trigger)."""
    assert (
        failure_lesson_capture_eligible({"resolved": "yes"}, subsequent_distinct_failure=False)
        is True
    )
    # The resolved signal stands even if a later distinct failure followed.
    assert (
        failure_lesson_capture_eligible({"resolved": "yes"}, subsequent_distinct_failure=True)
        is True
    )


def test_failure_lesson_capture_eligible_root_cause_confirmed_then_new_failure():
    """**Acceptance criterion 2.** A block carrying ``root_cause_confirmed: yes``
    is eligible EVEN when a subsequent distinct failure followed (the cause was
    confirmed though the run could not complete / the pod was abandoned in
    recovery — the #664 L204 gap)."""
    assert (
        failure_lesson_capture_eligible(
            {"root_cause_confirmed": "yes"}, subsequent_distinct_failure=True
        )
        is True
    )
    # And of course still eligible when no subsequent failure followed.
    assert (
        failure_lesson_capture_eligible(
            {"root_cause_confirmed": "yes"}, subsequent_distinct_failure=False
        )
        is True
    )


def test_failure_lesson_capture_eligible_neither_returns_false():
    """A block with neither signal (a vanilla progress note, not a lesson) is
    NOT eligible — even with a subsequent distinct failure present."""
    assert failure_lesson_capture_eligible({}, subsequent_distinct_failure=True) is False
    assert (
        failure_lesson_capture_eligible(
            {"resolved": "no", "root_cause_confirmed": "no"},
            subsequent_distinct_failure=False,
        )
        is False
    )


# ---------------------------------------------------------------------------
# apply_failure_lesson + supersedes_action — the AC#1 retraction + compose seam.
# ---------------------------------------------------------------------------


def _new_lesson_ref() -> dict[str, str]:
    return {
        "slug": _NEW_SLUG,
        "task_id": _NEW_TASK,
        "memory_path": _MEMORY_PATH,
        "lesson": _NEW_LESSON_BODY,
    }


def test_two_round_mis_then_correction():
    """**Acceptance criterion 1.** Round A captured a mis-diagnosis lesson into
    a durable file; round B emits the corrected lesson with ``supersedes:``
    pointing at the round-A slug. The composer must (a) prepend the EXACT
    concrete ``[SUPERSEDED by <slug> — see #<task_id>] `` marker to the matched
    prior entry (the full string, NOT just the ``[SUPERSEDED by`` prefix, NOT a
    ``<pending>`` placeholder), AND (b) land the corrected lesson body as a NEW
    entry in the owning-agent memory file — it lands ALONGSIDE the annotated
    prior, never replacing it."""
    prior_slug = "vllm_first_generate_is_a_code_bug"  # the round-A (wrong) slug
    gotcha_path = ".claude/rules/gotchas.md"
    matched_text = (
        f"## {prior_slug}\n\n"
        "A first-call vLLM `generate()` deadlock is a code bug — patch the "
        "eval script and relaunch on a fresh pod.\n"
    )
    unmatched_text = "## unrelated_lesson\n\nSome unrelated gotcha that mentions neither ref.\n"
    durable_texts = {
        gotcha_path: matched_text,
        _MEMORY_PATH: unmatched_text,
    }

    block = {
        "root_cause_confirmed": "yes",
        "lesson": _NEW_LESSON_BODY,
        "supersedes": prior_slug,
    }
    final = apply_failure_lesson(block, durable_texts, _new_lesson_ref())

    expected_marker = f"[SUPERSEDED by {_NEW_SLUG} — see #{_NEW_TASK}] "
    # (a) the matched prior entry is concretely annotated with the FULL string.
    assert expected_marker in final[gotcha_path], (
        f"matched entry not annotated with the concrete marker; got:\n{final[gotcha_path]!r}"
    )
    assert "<pending>" not in final[gotcha_path], (
        "annotation must be concrete, never a <pending> placeholder"
    )
    # The original matched text is preserved (annotated, not replaced).
    assert matched_text.strip() in final[gotcha_path]

    # (c) the corrected lesson body LANDS as a NEW entry in the owning-agent
    #     memory file (alongside, not replacing the prior content there).
    assert _NEW_LESSON_BODY in final[_MEMORY_PATH], (
        "corrected lesson body not appended to the owning-agent memory file; "
        f"got:\n{final[_MEMORY_PATH]!r}"
    )
    assert unmatched_text.strip() in final[_MEMORY_PATH], (
        "appending the new lesson must not drop the prior memory-file content"
    )


def test_dangling_supersedes_is_noop():
    """A ``supersedes`` ref that matches NO durable entry: ``supersedes_action``
    returns ``{}`` (no annotation), and ``apply_failure_lesson`` still lands the
    corrected lesson — a dangling ref is a logged no-op, never dropped data."""
    durable_texts = {
        ".claude/rules/gotchas.md": "## something_else\n\nUnrelated.\n",
        _MEMORY_PATH: "## prior_memory\n\nPrior memory body.\n",
    }
    # supersedes_action alone returns the empty annotated subset.
    assert (
        supersedes_action("a_ref_that_matches_nothing_anywhere", durable_texts, _new_lesson_ref())
        == {}
    )

    block = {
        "root_cause_confirmed": "yes",
        "lesson": _NEW_LESSON_BODY,
        "supersedes": "a_ref_that_matches_nothing_anywhere",
    }
    final = apply_failure_lesson(block, durable_texts, _new_lesson_ref())

    # No prior text is annotated (every prior entry stays byte-unchanged) ...
    assert "[SUPERSEDED by" not in final[".claude/rules/gotchas.md"]
    assert final[".claude/rules/gotchas.md"] == durable_texts[".claude/rules/gotchas.md"]
    # ... and the corrected lesson STILL lands in the owning-agent memory file.
    assert _NEW_LESSON_BODY in final[_MEMORY_PATH]


def test_supersedes_multi_match():
    """A ``supersedes`` ref that matches MULTIPLE durable entries annotates ALL
    of them (transparent multi-match — the transitive chain is kept, §7). Every
    matched entry gets the concrete marker prepended."""
    prior_ref = "stale_pod_lesson"
    a_path = ".claude/rules/gotchas.md"
    b_path = ".claude/agent-memory/experimenter/feedback_stale_pod_lesson.md"
    text_a = f"## {prior_ref}\n\nFirst copy of the stale lesson.\n"
    text_b = f"## {prior_ref}\n\nSecond (memory) copy of the stale lesson.\n"
    durable_texts = {a_path: text_a, b_path: text_b, _MEMORY_PATH: "## owner\n\nOwner body.\n"}

    annotated = supersedes_action(prior_ref, durable_texts, _new_lesson_ref())
    expected_marker = f"[SUPERSEDED by {_NEW_SLUG} — see #{_NEW_TASK}] "
    assert set(annotated) == {a_path, b_path}, (
        f"both matching entries should be annotated; got keys {set(annotated)}"
    )
    for path, annotated_text in annotated.items():
        assert expected_marker in annotated_text, f"{path} not concretely annotated"


def test_supersedes_absent_byte_equals_golden_fixture():
    """Backward-compat golden control: a block WITHOUT ``supersedes:`` over a
    clean (no-prior-entry) owning-agent durable text produces owning-agent text
    BYTE-FOR-BYTE EQUAL to the stored pre-#712 append-only fixture. The change
    is additive — the absent-supersedes path is the unchanged append-only path."""
    block = {
        "root_cause_confirmed": "yes",
        "lesson": _NEW_LESSON_BODY,
        # no "supersedes" key
    }
    # A clean owning-agent durable file (empty of any prior entry).
    durable_texts = {_MEMORY_PATH: ""}
    final = apply_failure_lesson(block, durable_texts, _new_lesson_ref())

    expected = _FIXTURE.read_text()
    assert final[_MEMORY_PATH] == expected, (
        "absent-supersedes append-only output drifted from the golden fixture "
        f"{_FIXTURE}; if the append-only format changed deliberately, regenerate "
        "the fixture in the same commit.\n"
        f"--- produced ---\n{final[_MEMORY_PATH]!r}\n--- fixture ---\n{expected!r}"
    )
