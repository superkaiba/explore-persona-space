"""Pin the #2240 Step 10d payload-aware no-PR arm in .claude/skills/issue/SKILL.md.

Task #2240 (2026-08-12) restructured the Step 10d safe-case PR routing:

- The pre-#2240 ``-z "$PR"`` arm skipped UNCONDITIONALLY ("No PR for
  issue-<N>; nothing to merge.", no marker), so a code-bearing branch whose
  Step 4a draft-PR create never fired (Step 4a runs BEFORE the implementer's
  first commit, so its else arm fires by construction) was left permanently
  unmerged with the durable record reading clean — the #456->#466
  stranded-shared-module class, invisible to the completed_unmerged_pass
  watcher flag (it keys on a marker the skip arm never posted).
- Post-#2240, BOTH no-usable-PR cases (terminal PR, #1897; no PR object at
  all, #2240/#2235) route through one shared payload-aware prelude gated on
  ``USABLE_PR``: #1897's layered fail-safe novel-payload predicate decides
  between "create a fresh PR and merge" and "genuinely nothing to merge",
  the zero-PR create carries an origin-precondition push + an rc-gated
  ``gh pr create``, the anomaly note is composed from the REALIZED outcome,
  and a novel-payload branch that cannot obtain a usable PR fails loud
  (``epm:merge-failed``) instead of printing a false nothing-to-merge line.

These tests fail the suite if a later SKILL.md editor re-introduces the
silent skip, narrows the predicate back to a ``PR_STATE != OPEN``-only arm,
drops a fail-safe layer, or removes the loud-failure routing (plan #2240
durability pins; 8 pins per plan section (D)).
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"


def _text() -> str:
    return issue_skill_text()


def _step10d_span() -> str:
    """Return the SKILL.md text from the (unique) `### Step 10d` heading onward."""
    text = _text()
    return text[text.index("### Step 10d") :]


GUARDS_COMMENT = "# Run guards 1-3 above first."
USABLE_GATE = 'if [ "$USABLE_PR" != yes ]; then'
NOVEL_GATE = 'if [ "$NOVEL_PAYLOAD" = "yes" ]; then'


def test_no_pr_arm_is_payload_aware():
    """Pin 1: the unconditional silent skip is GONE — from Step 10d and the
    Step 4a prose that promised a follow-up step which never existed."""
    text = _text()
    # The exact pre-#2240 silent-skip line (echo + post nothing) is absent.
    assert "No PR for issue-<N>; nothing to merge." not in text
    # Step 4a's false promise is gone; its else arm now names the Step 10d
    # payload-aware backstop instead (hunk B durability).
    assert "open it after the implementer commits" not in text
    step4a_region = text[: text.index("### Step 10d")]
    assert "This arm fires by construction on a fresh branch" in step4a_region
    assert "payload-aware arm (#2240) opens it at merge time" in step4a_region


def test_both_skip_arms_route_through_usable_pr_gate():
    """Pin 2: the USABLE_PR gate exists, and BOTH skip arms (loud novel-payload
    failure + quiet nothing-to-merge) live inside the routing gate, before the
    guards/merge body."""
    span = _step10d_span()
    assert span.count(USABLE_GATE) == 2  # prelude + routing gate
    idx_routing = span.index(USABLE_GATE, span.index(USABLE_GATE) + 1)
    idx_guards = span.index(GUARDS_COMMENT)
    idx_merge_failed = span.index("post-marker <N> epm:merge-failed")
    idx_echo_prior = span.index("has no novel payload vs origin/main — nothing to merge")
    idx_echo_no_pr = span.index("has no PR and no novel payload vs origin/main — nothing to merge")
    assert idx_routing < idx_merge_failed < idx_guards
    assert idx_routing < idx_echo_prior < idx_guards
    assert idx_routing < idx_echo_no_pr < idx_guards


def test_predicate_in_shared_no_usable_pr_prelude():
    """Pin 3: the novel-payload predicate lives inside the shared no-usable-PR
    prelude, NOT inside a `PR_STATE != OPEN`-only arm; the USABLE_PR
    resolution precedes the predicate's defensive NOVEL_PAYLOAD init."""
    span = _step10d_span()
    # The old scoping is gone entirely.
    assert 'if [ "$PR_STATE" != "OPEN" ]; then' not in span
    # The positive resolution replaces it.
    assert 'if [ -n "$PR" ] && [ "$PR_STATE" = "OPEN" ]; then' in span
    # Ordering: USABLE_PR is assigned before the defensive NOVEL_PAYLOAD=yes
    # init, and the bounded fetch + predicate run inside the prelude (after
    # the first USABLE_PR gate).
    assert span.index("USABLE_PR=no") < span.index("NOVEL_PAYLOAD=yes")
    idx_prelude = span.index(USABLE_GATE)
    idx_fetch = span.index('timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main')
    assert idx_prelude < idx_fetch < span.index(GUARDS_COMMENT)


def test_fail_safe_predicate_layers_verbatim():
    """Pin 4: all four #1897 fail-safe layers survive verbatim, including
    NOVEL_PAYLOAD=yes as the default and the fail-safe comments on the
    git-error paths."""
    span = _step10d_span()
    assert "NOVEL_PAYLOAD=yes" in span
    assert 'rev-list --count origin/main..issue-<N>)" -eq 0 ]' in span  # (1)
    assert 'elif CHERRY=$(git -C "$WT" cherry origin/main issue-<N>)' in span  # (2)
    assert "(a cherry FAILURE falls through — fail-safe)" in span
    assert 'OWN_FILES=$(git -C "$WT" diff --name-only origin/main...issue-<N>)' in span  # (3)
    assert 'git -C "$WT" diff --quiet origin/main issue-<N> -- $OWN_FILES' in span
    assert "(a diff ERROR keeps 'yes' — fail-safe)" in span
    flat = " ".join(span.split())
    assert "(4) else -> novel payload" in flat


def test_no_pr_anomaly_marker_present():
    """Pin 5: the zero-PR create arm exists (HAD_PRIOR_PR branch + #2240 PR
    body) and posts the [step10d-no-pr-anomaly] note composed from the
    REALIZED outcome (opened-and-proceeding vs recovery-FAILED)."""
    span = _step10d_span()
    assert "HAD_PRIOR_PR=no" in span
    assert "no PR object exists (#2240 probe)" in span
    assert span.count("[step10d-no-pr-anomaly]") == 2  # success + failure notes
    flat = " ".join(span.split())
    assert "Step 10d opened PR #$PR and is proceeding with the auto-merge (#2240)." in flat
    assert "the recovery FAILED: gh pr create did not yield an OPEN PR" in flat


def test_pr_ready_precedes_merge():
    """Pin 6: `gh pr ready "$PR"` still immediately precedes the safe-case
    merge — the draft-merge precondition for PRs created by EITHER fresh-PR
    arm — and no second ready call was added (hunk C durability)."""
    span = _step10d_span()
    assert span.count('gh pr ready "$PR"') == 1
    idx_ready = span.index('gh pr ready "$PR"')
    idx_merge = span.index('gh pr merge "$PR" $MERGE_FORM --delete-branch=false')
    assert idx_ready < idx_merge
    flat = " ".join(span.split())
    assert "Draft-merge precondition (#2240 pin)" in flat
    assert "do NOT add a second ready call elsewhere" in flat


def test_origin_precondition_precedes_rc_gated_create():
    """Pin 7: the origin-precondition (ls-remote probe + push -u) runs BEFORE
    `gh pr create`, and the create is rc-gated so a failed create can never
    fall through into the nothing-to-merge arm."""
    span = _step10d_span()
    idx_lsremote = span.index('git -C "$WT" ls-remote --heads origin issue-<N>')
    idx_push_u = span.index('git -C "$WT" push -u origin issue-<N>')
    idx_create = span.index("gh pr create --draft --head issue-<N>")
    assert idx_lsremote < idx_create
    assert idx_push_u < idx_create
    # rc-gated: the create is the condition of an `if`, never a bare command.
    assert (
        'if gh pr create --draft --head issue-<N> --title "$PR_TITLE" --body "$PR_BODY"; then'
        in span
    )
    # The fresh PR re-resolve only flips USABLE_PR on an OPEN resolve.
    assert '[ -n "$PR" ] && [ "$PR_STATE" = "OPEN" ] && USABLE_PR=yes' in span


def test_nothing_to_merge_guarded_on_novel_payload():
    """Pin 8: the nothing-to-merge echoes sit in the NOVEL_PAYLOAD else arm of
    the routing gate, and the novel-payload-but-no-usable-PR path fails loud
    with epm:merge-failed before them."""
    span = _step10d_span()
    idx_routing = span.index(USABLE_GATE, span.index(USABLE_GATE) + 1)
    idx_novel_routing = span.index(NOVEL_GATE, idx_routing)  # NOVEL conjunct inside the gate
    idx_merge_failed = span.index("post-marker <N> epm:merge-failed")
    idx_echo_prior = span.index("has no novel payload vs origin/main — nothing to merge")
    idx_echo_no_pr = span.index("has no PR and no novel payload vs origin/main — nothing to merge")
    idx_guards = span.index(GUARDS_COMMENT)
    # Routing gate -> NOVEL_PAYLOAD conjunct -> loud failure -> quiet echoes.
    assert idx_routing < idx_novel_routing < idx_merge_failed < idx_echo_prior < idx_guards
    assert idx_merge_failed < idx_echo_no_pr < idx_guards
    flat = " ".join(span.split())
    assert "NOVEL PAYLOAD ON issue-<N> COULD NOT BE MERGED" in flat
    assert "this is a stranding risk, not a no-op" in flat
