"""Pin the #1723 Step 10 CRON-TEARDOWN + epm:done reorder around Step 10d.

Task #1723 (2026-07-27) rewrote SKILL.md so that on the code-change path
(``kind: infra|batch|analysis|survey``) CRON-TEARDOWN + ``set-status
completed`` + ``epm:done`` fire AFTER Step 10d posts ``epm:merged v1``,
instead of before it. This closes:

* The ~33 min merge window that used to run without ``/issue-tick``
  re-drive coverage (session ``7ce3a81f`` 2026-07-26 tore down its cron
  at 08:41:20Z, began Step 10d at 08:41:52Z, and merged at 09:13:46Z).
* The ``completed_unmerged_pass`` flag class (#1540 / #1653), where the
  durable record read ``completed``+``epm:done`` on an unmerged branch
  (session ``e3b70618`` on task #1709 sat 83 min in that state).

The experiment path is unchanged — Step 9b already auto-merged the
worktree before the task parked at ``awaiting_promotion``, so by the
time Step 10 runs ``epm:merged`` is already present; Step 10 step 6's
``epm:merged``-present branch is an idempotent backstop for that arm.

Six pin tests below assert structural invariants against the SKILL.md
prose so the reorder cannot silently drift. All six read anchored
substring tokens (``### Step 10d``, ``#### Terminal teardown
(code-change path only``, ``Step 10d exit``, ``[long-phase-heartbeat]
step10d-merge``, ``` If `epm:merged` is already present ```) rather
than line numbers — the sibling
``tests/test_issue_skill_step10d_merge_form.py`` uses the same pattern
so a future re-wrap of SKILL.md that shifts line numbers does not break
these pins.
"""

from __future__ import annotations

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"


def _step10d_span() -> str:
    """SKILL.md text from the unique ``### Step 10d`` heading onward."""
    text = SKILL.read_text()
    return text[text.index("### Step 10d") :]


def _step10_step6_span() -> str:
    """SKILL.md text spanning Step 10 steps 5-7 (before ``### Step 10b``).

    The reworked step 6/7 lives between Step 10's list of destination
    branches (Step 5 tail — the ``No `type` frontmatter`` line) and the
    ``### Step 10b`` header. Anchoring on those tokens keeps this pin
    stable under Step-10 prose churn.
    """
    text = SKILL.read_text()
    start = text.index("- **No `type` frontmatter**")
    end = text.index("### Step 10b", start)
    return text[start:end]


# ── Exit-site list + parallel prose (Edits 1 and 2) ─────────────────────────


def test_step10d_exit_site_named_in_teardown_exit_list():
    """The ``It is torn down ONLY at the true terminal / park`` list must
    name Step 10d as its own exit site so the exit-site enumeration
    matches the reordered teardown behavior on the code-change path.
    """
    text = SKILL.read_text()
    anchor = "It is torn down ONLY at the true terminal / park"
    assert anchor in text, "CRON-TEARDOWN exit-site-list anchor missing"

    # Window: the anchor up to the closing paragraph. The bulleted list
    # is contiguous; 2000 chars comfortably covers it and refuses to
    # match a stray "Step 10d exit" mention elsewhere in the file.
    window = text[text.index(anchor) : text.index(anchor) + 2000]
    assert "Step 10d exit" in window, (
        "The CRON-TEARDOWN exit-site list must include an "
        "'at Step 10d exit ...' bullet (code-change path). #1723."
    )


def test_step10d_exit_site_named_in_parallel_prose():
    """The parallel prose sibling (Step 6d.3 § "On ``status=done``")
    duplicates the exit-site enumeration; it must ALSO name Step 10d so
    the two surfaces stay in sync (a drift would let one side FAIL the
    other's mandate).
    """
    # The parallel-prose enumeration lives in Step 6d.3 "On
    # ``status=done``". Anchor on its unique H5 header token and read
    # the following window so a soft-wrapped "torn\ndown only at the
    # true terminal ..." break does not defeat the substring probe.
    text = SKILL.read_text()
    anchor = "##### Step 6d.3: On `status=done`"
    assert anchor in text, "parallel-prose Step 6d.3 header missing"

    window = text[text.index(anchor) : text.index(anchor) + 2500]
    assert "true terminal / park transitions" in window, (
        "parallel-prose CRON-TEARDOWN enumeration missing in Step 6d.3"
    )
    assert "Step 10d exit" in window, (
        "The parallel prose enumeration must ALSO name Step 10d exit "
        "as its own terminal-transition entry. #1723."
    )


# ── Terminal-teardown H4 sub-section (Edit 3) ───────────────────────────────


def test_terminal_teardown_h4_present_and_after_stale_folder_guard_in_step10d():
    """The reorder plants a NEW H4 sub-section under Step 10d that runs
    the terminal-teardown sequence AFTER the ``Post-merge stale-task-
    folder guard``. The H4 heading must:

    * exist EXACTLY ONCE inside the Step 10d span (uniqueness matters —
      the sibling ``test_issue_skill_step10d_merge_form.py::
      test_experiment_rebase_rationale_current`` uses the same
      count-based idiom);
    * appear AFTER the ``Post-merge stale-task-folder guard`` heading
      inside the Step 10d span so the guard reconciles ``main`` FIRST.

    Substring-anchored + ``.index()`` positional; a reorder within the
    span fails the test.
    """
    span = _step10d_span()

    h4 = "#### Terminal teardown (code-change path only"
    assert span.count(h4) == 1, (
        f"Expected EXACTLY ONE '#### Terminal teardown (code-change path only' "
        f"H4 in Step 10d; found {span.count(h4)}. #1723."
    )

    guard = "#### Post-merge stale-task-folder guard"
    assert guard in span, "Post-merge stale-task-folder guard heading missing in Step 10d span"

    assert span.index(h4) > span.index(guard), (
        "The Terminal-teardown H4 must appear AFTER the "
        "Post-merge stale-task-folder guard heading so the guard "
        "reconciles main's task-folder state BEFORE the teardown fires. "
        "#1723."
    )


def test_terminal_teardown_fires_epm_merged_before_set_status_completed():
    """Inside the Terminal-teardown H4 span, the sequence is
    ``epm:merged v1`` -> CRON-TEARDOWN -> ``set-status ... completed``
    -> ``epm:done``. The prose must state this ordering explicitly so
    the reorder cannot be silently reversed by a later edit.

    Anchored substring positions inside the H4 span are the source of
    truth. The redundant ``AFTER `epm:merged v1` has been posted``
    check defends against a stray earlier ``epm:merged`` mention within
    the block (per code-review concern (e)).
    """
    span = _step10d_span()

    h4 = "#### Terminal teardown (code-change path only"
    h4_start = span.index(h4)

    # H4 body ends at the next H2/H3/H4 heading OR the file end. The
    # existing SKILL.md tail after this block is the '---' + '## Resume
    # semantics' section; slicing until that boundary keeps the assertions
    # scoped to the H4.
    tail = span[h4_start:]
    body_end = tail.find("\n## Resume semantics")
    if body_end == -1:
        body_end = len(tail)
    body = tail[:body_end]

    # Redundant precondition-phrase check (defends the "stray earlier
    # epm:merged" false-PASS class the code-reviewer flagged as (e)).
    assert "AFTER `epm:merged v1` has been posted" in body, (
        "Terminal-teardown H4 must state that the sequence runs "
        "AFTER `epm:merged v1` has been posted. #1723."
    )

    # Ordering inside the H4 block: CRON-TEARDOWN precedes
    # `set-status ... completed` precedes `epm:done`. `.index()`
    # positions inside the H4 body make the ordering explicit; the H4
    # body precondition above already binds the whole sequence to the
    # post-`epm:merged` slot.
    cron = "Run CRON-TEARDOWN"
    set_status = "set-status <N> completed"
    epm_done = "epm:done v1"

    for token in (cron, set_status, epm_done):
        assert token in body, f"Terminal-teardown H4 must name {token!r}. #1723."

    assert body.index(cron) < body.index(set_status) < body.index(epm_done), (
        "Terminal-teardown H4 must order the three steps as "
        "CRON-TEARDOWN -> set-status <N> completed -> epm:done v1. #1723."
    )


# ── Step 10 step 6/7 rework (branches on epm:merged presence) ───────────────


def test_step10_step6_branches_on_epm_merged_presence():
    """The reworked Step 10 step 6 must carry BOTH branches so the
    experiment path (``epm:merged`` already present from Step 9b) keeps
    its idempotent backstop AND the code-change path (``epm:merged``
    not yet present) advances to Step 10d without touching the terminal
    status yet.

    A future edit that drops either branch collapses the reorder
    (dropping the ``NOT yet present`` branch reverts to the old
    ordering; dropping the ``already present`` branch breaks the
    experiment path's idempotent backstop). Both must be present.
    """
    span = _step10_step6_span()

    experiment_branch = "If `epm:merged` is ALREADY present"
    code_change_branch = "If `epm:merged` is NOT yet present"

    assert experiment_branch in span, (
        "Step 10 step 6 must carry the experiment-path branch "
        "(`If `epm:merged` is ALREADY present`) that runs the "
        "idempotent-backstop CRON-TEARDOWN + set-status + epm:done "
        "sequence in place. #1723."
    )
    assert code_change_branch in span, (
        "Step 10 step 6 must carry the code-change-path branch "
        "(`If `epm:merged` is NOT yet present`) that DEFERS the "
        "terminal transition to Step 10d's own terminal-teardown "
        "sub-section. #1723."
    )


# ── Step 10d retry-surface heartbeats (Edit 4) ──────────────────────────────


def test_step10d_retry_surfaces_emit_long_phase_heartbeat():
    """Every retry surface inside Step 10d must emit a
    ``[long-phase-heartbeat] step10d-merge attempt=<k> shape=<S>``
    progress marker so external observers (the stalled detector,
    ``tick_triage.py``, downstream sessions) can distinguish an
    in-flight retry from a stranded merge.

    The plan names four retry surfaces — shape 0 (Base branch was
    modified), shape 2 (Pull Request has merge conflicts), shape 3
    (Head branch is out of date), and merge-conflict-recovery (the
    post-recovery `--squash` retry). Each must name the heartbeat in
    its prose. Sibling shape 1 (``can't be rebased``) falls into the
    same recovery path as shape 2's outcome-classification, so no
    dedicated heartbeat is added for it (see code-review concern (b)).
    """
    span = _step10d_span()

    # Count total hits so a duplicate insert on one surface cannot
    # false-PASS a missing sibling.
    literal = "[long-phase-heartbeat] step10d-merge"
    hits = span.count(literal)
    assert hits >= 4, (
        f"Expected at least 4 '{literal}' heartbeats in Step 10d "
        f"(shape 0 + shape 2 + shape 3 + merge-conflict recovery); "
        f"found {hits}. #1723."
    )

    # And every one of the four shape tokens must appear on its own line
    # of a heartbeat note (a bare token elsewhere in the span does not
    # count — the assertion targets the note itself).
    for shape_token in ("shape=0", "shape=2", "shape=3", "shape=conflict-recovery"):
        needle = f"[long-phase-heartbeat] step10d-merge attempt=<k> {shape_token}"
        assert needle in span, (
            f"Step 10d must emit a long-phase-heartbeat with '{shape_token}' "
            f"in its retry-surface prose. #1723."
        )
