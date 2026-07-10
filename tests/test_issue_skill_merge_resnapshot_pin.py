"""Content-invariant pins for the #1210 Step-10d merge re-snapshot recipe.

Task #1210 made the Step-10d merge robust to fleet marker churn between the
Guard-1 snapshot and the server-side merge (the #1128 incident class):

1. Guard 1 fetches `origin/main` fresh and pins every strip command to ONE
   captured `MAIN_SHA` (the movable shared ref is consumed only at the
   `rev-parse` capture).
2. A "Known failure shape 2" paragraph documents the bounded (retry-ONCE)
   re-snapshot-and-retry on the `Pull Request has merge conflicts` refusal.
3. The merge-conflict recovery merges the captured SHA, mechanically pins
   foreign `tasks/` conflicts to it, and certifies the result via a
   `diff "$MAIN_SHA" HEAD -- 'tasks/'` before pushing.

REGION-SCOPED by design: Edits (1) and (3) deliberately share byte-identical
strings (`MAIN_SHA=$(git -C "$WT" rev-parse origin/main)`, pinned checkouts),
so whole-file substring asserts would cross-satisfy and miss the deletion of
either single site. The region helpers mirror
`tests/test_step10d_guard3.py`'s `_merge_guards_region` precedent; the prose
pin (shape 2) is whitespace-normalized (the `_normalized()` precedent) so a
legitimate hard-wrap rewording does not false-fail.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

_SHA_CAPTURE = 'MAIN_SHA=$(git -C "$WT" rev-parse origin/main)'
_GUARD1_PINNED_CHECKOUT = 'git -C "$WT" checkout "$MAIN_SHA" -- "${FOREIGN_ON_MAIN[@]}"'
_RECOVERY_PINNED_CHECKOUT = 'git -C "$WT" checkout "$MAIN_SHA" -- "${RECOVERY_FOREIGN[@]}"'
_RECOVERY_CERT_DIFF = 'git -C "$WT" diff --name-only "$MAIN_SHA" HEAD -- \'tasks/\''


def _skill_text() -> str:
    return _SKILL.read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    """Whitespace-collapse for prose pins (hard-wrapped markdown re-wraps)."""
    return " ".join(text.split())


def _merge_guards_region(text: str) -> str:
    """The merge-safety-guards slice (Guard 0 + guards 1-3 live here)."""
    start_marker = "#### Merge safety guards (run before the merge commands)"
    end_marker = "#### Fast-path routing pre-check"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "Merge safety guards heading not found in SKILL.md"
    assert end != -1, "Fast-path pre-check heading not found in SKILL.md"
    assert start < end, "guards region must precede the fast-path pre-check"
    return text[start:end]


def _recovery_region(text: str) -> str:
    """The merge-conflict recovery slice (Edit C's home)."""
    start_marker = "#### Merge-conflict recovery"
    end_marker = "#### The artifact-confirmed merge procedure"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "Merge-conflict recovery heading not found in SKILL.md"
    assert end != -1, "artifact-confirmed merge heading not found in SKILL.md"
    assert start < end, "recovery region must precede the artifact-confirmed merge"
    return text[start:end]


def _shape2_paragraph(text: str) -> str:
    """The Known-failure-shape-2 slice: heading through its fenced block +
    parenthetical note, ending at the Success bullet."""
    start_marker = "**Known failure shape 2"
    end_marker = "- **Success:**"
    start = text.find(start_marker)
    assert start != -1, "Known failure shape 2 heading not found in SKILL.md"
    end = text.find(end_marker, start)
    assert end != -1, "Success bullet must follow the shape-2 paragraph"
    return text[start:end]


def test_step10d_resnapshot_recipe_present():
    """The #1210 durability pin: (i) Guard 1 captures + pins MAIN_SHA inside
    the merge-guards region; (ii) a Known-failure-shape-2 paragraph exists
    with a re-snapshot + retry-ONCE bound; (iii) the recovery pins its
    checkout to the captured SHA and certifies over tasks/ before pushing."""
    text = _skill_text()

    # (i) — inside the merge-guards region only.
    guards = _merge_guards_region(text)
    assert _SHA_CAPTURE in guards, "Guard 1 must capture MAIN_SHA via rev-parse origin/main"
    assert _GUARD1_PINNED_CHECKOUT in guards, (
        "Guard 1 must reset foreign-on-main paths to the captured MAIN_SHA snapshot"
    )

    # (ii) — whitespace-normalized prose pin.
    para = _normalized(_shape2_paragraph(text))
    assert "re-snapshot" in para, "shape 2 must prescribe a re-snapshot recovery"
    assert "retry ONCE" in para, "shape 2 must bound the retry to ONCE"
    assert "ONCE per Step 10d invocation" in para, (
        "the shape-2 block must restate the per-invocation retry bound"
    )

    # (iii) — inside the recovery region only.
    recovery = _recovery_region(text)
    assert _RECOVERY_PINNED_CHECKOUT in recovery, (
        "the recovery must pin foreign tasks/ conflicts to the captured MAIN_SHA"
    )
    assert _RECOVERY_CERT_DIFF in recovery, (
        "the recovery must certify the tasks/ tree against the captured snapshot "
        "(diff \"$MAIN_SHA\" HEAD -- 'tasks/') before pushing"
    )


def test_guard1_consumes_movable_ref_only_at_fetch_and_capture():
    """Inside the merge-guards region, the strip's COMMAND positions consume
    the pinned SHA — the old movable-ref forms must be gone (a concurrent
    session's fetch can advance origin/main mid-guard, #1128)."""
    guards = _merge_guards_region(_skill_text())
    assert 'git -C "$WT" fetch origin main --quiet' in guards, (
        "Guard 1 must fetch origin main fresh before capturing MAIN_SHA"
    )
    assert "diff --name-only \"$MAIN_SHA\" HEAD -- 'tasks/'" in guards, (
        "the Guard-1 trigger diff must run against the captured MAIN_SHA"
    )
    assert 'cat-file -e "$MAIN_SHA:$p"' in guards, (
        "the Guard-1 existence probe must run against the captured MAIN_SHA"
    )
    for stale in (
        "diff --name-only origin/main HEAD -- 'tasks/'",
        'cat-file -e "origin/main:$p"',
        'checkout origin/main -- "${FOREIGN_ON_MAIN[@]}"',
    ):
        assert stale not in guards, f"stale movable-ref strip command must be gone: {stale!r}"


def test_recovery_merges_pinned_sha_not_movable_ref():
    """Edit C: the recovery captures MAIN_SHA once post-fetch and merges THAT
    SHA — merging the movable ref re-opens the shared-ref race (#1128)."""
    recovery = _recovery_region(_skill_text())
    assert _SHA_CAPTURE in recovery, "the recovery must capture MAIN_SHA after its fetch"
    assert 'git -C "$WT" merge "$MAIN_SHA"' in recovery, (
        "the recovery must merge the captured SHA, not origin/main"
    )
    assert 'git -C "$WT" merge origin/main' not in recovery, (
        "the movable-ref merge form must be gone from the recovery"
    )
    # Fail-loud residual-foreign-diff arm: a non-empty foreign diff after
    # resolution must hard-stop the push (materialize-then-check, #1184 shape).
    assert "recovery: foreign tasks/ still differ" in recovery, (
        "the certification must fail loud when foreign tasks/ still differ"
    )
    assert "recovery: tasks/ verification diff FAILED" in recovery, (
        "a FAILED certification diff must fail loud (never read as clean)"
    )


def test_recovery_certification_arms_are_exclusive():
    """#1243: both recovery certification producers run in Guard 1's `if !`
    exclusive-arm shape — a failed producer takes the terminal echo + false
    arm and its consumer is structurally unreachable. The old
    `|| { echo "recovery:..."; false; }` form reported failure without
    halting under no-set-e / piecewise execution (the verification-diff arm
    failed OPEN into the push)."""
    recovery = _recovery_region(_skill_text())
    assert "if ! git -C \"$WT\" diff --name-only --diff-filter=U -- 'tasks/'" in recovery, (
        "the conflicted-path producer must run inside an `if !` failure arm"
    )
    assert 'if ! git -C "$WT" diff --name-only "$MAIN_SHA" HEAD -- \'tasks/\'' in recovery, (
        "the certification-diff producer must run inside an `if !` failure arm"
    )
    assert '|| { echo "recovery:' not in recovery, (
        "no non-halting `|| { echo ...; false; }` certification arm may remain"
    )
    # The residual-foreign check must be an elif arm of the SAME chain as the
    # verification diff (fused certification: a failed diff cannot vacuously pass).
    # Anchored on the verify-file path (not a bare `elif grep -Ev`) so an
    # unrelated elif-grep elsewhere in the region cannot false-satisfy it.
    cert = recovery.find('if ! git -C "$WT" diff --name-only "$MAIN_SHA" HEAD')
    residual = recovery.find(
        'elif grep -Ev "^tasks/[^/]+/<N>/" /tmp/issue-<N>-recovery-tasks-verify.txt', cert
    )
    assert -1 < cert < residual, (
        "the residual-foreign grep must be the elif work arm of the certification chain"
    )


def test_shape2_retry_gated_on_file_persisted_tip():
    """The shape-2 retry is gated on a FILE-persisted tip-changed predicate
    (fenced blocks are separate shells) with the retry in the else arm — the
    skip arm must END the block, never fall through into the push."""
    para = _shape2_paragraph(_skill_text())
    tip_persist = para.find('git -C "$WT" rev-parse HEAD > /tmp/issue-<N>-resnapshot-tip.txt')
    tip_read = para.find("TIP_BEFORE=$(cat /tmp/issue-<N>-resnapshot-tip.txt)")
    skip_echo = para.find("re-snapshot changed nothing")
    else_arm = para.find('else\n  git -C "$WT" push origin issue-<N>')
    mergeable = para.find("gh pr view <PR> --json mergeable")
    assert tip_persist != -1, "STEP 1 must persist the pre-resnapshot tip to a file"
    assert tip_read != -1, "STEP 3 must re-read the tip from the persisted file"
    assert -1 < tip_persist < tip_read, "the tip persist must precede its read-back"
    assert -1 < skip_echo < else_arm, "the skip arm must precede the else-arm push"
    assert -1 < else_arm < mergeable, (
        "the retry push must be followed by the async-mergeability re-check"
    )
