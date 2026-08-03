"""Pin the #1560 → #1714 lint/guard-family freshness sync in .claude/skills/issue/SKILL.md.

#1560 (2026-07-20) extended the Step 5a spec-freshness sync to cover the
lint/guard family (scripts/workflow_lint.py, .claude/hooks, the
test_guard_* / test_workflow_lint* pin tests), subject-scoped the
sync's per-file branch-side-edit exclusion (the Guard-3 convention), and
added a mandatory pre-gate freshness re-sync against fetched origin/main to
the Step 10d pre-push workflow-lint gate — closing the branch-era-linter
vintage-skew class that red the gate three times on 2026-07-19
(#1489 / #1482 / #1417 / #1675→#1682).

#1714 (2026-07-26) makes the sync family-atomic (a branch-side edit on
ANY member of a coupled family — workflow.yaml + markers.md,
scripts/workflow_lint.py + its pin tests, .claude/hooks + its guard
tests — widens the skip to the WHOLE family), moves the Step 10d
re-sync from PRE-gate to POST-gate (right before `gh pr merge`) so
origin/main advancement during the ~30-min gate window can no longer
red the squash merge with CONFLICTING, and adds two new pin tests
(family declaration + Step 5a/§4.6 drift guard).

#1807 (2026-07-29) adds the mechanically-gated verdict RE-BIND to the
Step 10d safe-case block (a post-gate sync commit whose cert-sha..HEAD
delta is provably origin/main-identical A/M-only re-binds the verdict
file's line 2 to the new tip instead of forcing a full gate re-run;
D/R*/C*/T/U rows and non-identical content fail CLOSED behind
REBIND_OK), moves the #1657 head-sync pre-check AFTER the re-sync +
re-bind (it must poll the FINAL tip), and adds a Step 9c step-1a
pre-gate spec-freshness re-sync — a binding REFERENCE to the Step 5a
family-atomic block, never a third inlined FAMILY_OF copy — so a
main-side spec fix landing after the Step 5a sync can no longer red the
9c gate (#1742 class). Pin tests (12) and (13) cover the two additions.

#1883 (2026-08-01) adds the `:(glob)tests/test_issue_skill_*.py`
skill-pin glob to the "workflow" family (both FAMILY_OF copies + both
SPECS lists), closing the #1824 vintage-skew residual: main's SKILL.md
synced without its paired pin test, and 3 test_issue_skill_* pin tests
red the Step 9c gate (~30-min gate re-run + manual reconciliation).
Test (14) additionally pins SPECS <-> SPECS_10D token-set equality
(SPECS_10D was previously unpinned).

#1963 (2026-08-03) adds the guard-script implementation set
`:(glob)scripts/guard_*.sh` to the "guard" family (both FAMILY_OF copies
+ both SPECS lists): the test_guard_* pin tests execute the WORKTREE
copies of scripts/guard_*.sh, so syncing the tests without the scripts
half-syncs the tree and reds main-green guard nodes on pure version
skew (incidents #1860: 3 false-red test_guard_repo_root_branch.py
nodes; #1862: 12 false-red test_1861_exitguard_* nodes). Test (1)'s
SPECS literal and test (9)'s family-membership asserts gain the new
member; tests (10) + (14) mechanically enforce the Step 10d copy +
SPECS_10D parity with no edit.

These tests fail the suite if a later SKILL.md editor drops the family
entries, the boundary-paragraph family exception, the post-gate re-sync
bullet (or reorders it before the gate's stale-verdict rm), the 9a-ter
staleness note, reintroduces the full-message commit filter the Step 5a
sync and the gate section deliberately avoid, drops the family-atomic
declaration in Step 5a, lets the Step 10d inline family-atomic block
drift from Step 5a's family definition, weakens the #1807 re-bind
stanza's fail-closed arms, or drops the 9c pre-gate re-sync reference.

NOTE for future SKILL.md editors: these assertions pin literal snippet text.
A legitimate rewording of the pinned lines in SKILL.md must update the
matching assertions here IN THE SAME COMMIT, or the suite goes red.
"""

import re
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"

# The banned full-message commit-filter literal, built by CONCATENATION so
# this test file itself never carries the token its negative asserts scan for.
_FULL_MESSAGE_FILTER = "--grep=" + "'spec-freshness'"
_FULL_MESSAGE_INVERT = _FULL_MESSAGE_FILTER + " --invert-grep"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _step5a_span(text: str) -> str:
    """The Step 5a spec-freshness block + its boundary paragraph.

    From the block's leading comment (unique) to the 429-pacing blockquote
    that follows the boundary paragraph (unique).
    """
    start = text.index("# Step 5a WANTS the WORKTREE root")
    end = text.index("429 pacing at every ensemble fan-out", start)
    return text[start:end]


def _gate_region(text: str) -> str:
    """The pre-push workflow-lint gate section: gate H4 through the auto-merge H4.

    Region-scoped deliberately: the stale-verdict rm line occurs 5x file-wide
    (Step 9c and auto-merge consumers), so whole-file index() would be vacuous
    for the ordering assert; within this region it occurs exactly once.
    """
    start = text.index("#### Pre-push workflow-lint gate")
    end = text.index("#### The auto-merge procedure", start)
    return text[start:end]


# --- (1) Step 5a SPECS family entries -------------------------------------


def test_step5a_specs_include_lint_family():
    assert (
        'SPECS=".claude/agents .claude/skills .claude/rules .claude/workflow.yaml '
        "CLAUDE.md scripts/workflow_lint.py .claude/hooks "
        ":(glob)scripts/guard_*.sh "
        "tests/test_guard_lessons_edit.py "
        "tests/test_workflow_yaml.py tests/test_autonomous_session_watch.py "
        ":(glob)tests/test_workflow_lint*.py "
        ":(glob)tests/test_guard_*.py "
        ':(glob)tests/test_issue_skill_*.py"'
    ) in _text(), (
        "Step 5a SPECS must carry the #1560 lint/guard family "
        "(workflow_lint.py, .claude/hooks, the :(glob) test_workflow_lint* "
        "and :(glob) test_guard_* pin-test families) plus the #1714 "
        "explicit importers tests/test_workflow_yaml.py and "
        "tests/test_autonomous_session_watch.py (workflow_lint symbols "
        "used outside the :(glob) test_workflow_lint* pattern) — the "
        "guard-family widening pinned by #1709 covers all "
        "tests/test_guard_*.py (vintage-skew class "
        "#1489/#1482/#1417/#1675→#1682) — plus the #1883 skill-pin glob "
        ":(glob)tests/test_issue_skill_*.py (prose-pin tests over "
        ".claude/skills content; the #1824 vintage skew) — plus the #1963 "
        "guard-script implementation set :(glob)scripts/guard_*.sh (the "
        "test_guard_* pins execute the worktree copies; syncing tests "
        "without scripts half-syncs the tree — the #1860/#1862 false-red "
        "guard nodes)"
    )


# --- (2) boundary paragraph names the family + retains the exclusion ------


def test_sync_scope_paragraph_names_family_boundary():
    span = _step5a_span(_text())
    assert "spec-coupled lint/guard family" in span, (
        "the sync-scope boundary paragraph must name the family exception"
    )
    assert "do NOT\nextend it further into `scripts/`, `tests/`, or `src/`" in span, (
        "the boundary paragraph must retain the exclusion for everything else"
    )
    assert "main's newer workflow\ntests pin behavior implemented in main's newer" in span, (
        "the boundary paragraph must retain rationale (ii): main's newer tests "
        "pin main's newer scripts/+src/"
    )
    assert "explore_persona_space.workflow" in span, (
        "the boundary paragraph must name the accepted single src seam"
    )
    assert "collection-time ImportError" in span, (
        "the cross-check-at-root operational rule must carry the "
        "collection-time-ImportError staleness symptom"
    )
    assert "`:(glob)tests/test_issue_skill_*.py`" in span, (
        "the boundary paragraph must name the #1883 skill-pin glob "
        ":(glob)tests/test_issue_skill_*.py as a workflow-family member "
        "(backtick-wrapped, so this pins the PROSE mention — the bare "
        "bash-block occurrences do not satisfy it)"
    )


# --- (3) post-gate re-sync bullet present + ordered AFTER the verdict rm --


def test_step10d_postgate_resync_present_and_ordered():
    region = _gate_region(_text())
    bullet_idx = region.index("**Post-gate freshness re-sync (#1714")
    assert "origin/main" in region[bullet_idx:], "re-sync must anchor to origin/main"
    assert 'timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet' in region, (
        "re-sync must start with the bounded fetch"
    )
    assert "[step10d] post-gate re-sync: synced <n> files (<sha>) | no drift" in region, (
        "the re-sync must end with the ran-vs-never-ran echo breadcrumb"
    )
    rm_idx = region.index("rm -f /tmp/issue-<N>-lint-verdict.txt")
    assert bullet_idx > rm_idx, (
        "the post-gate re-sync must be positioned AFTER the gate's "
        "stale-verdict rm (the bullet documents a post-gate operation "
        "executed from the auto-merge subsection below, so its prose "
        "must come after the executable gate block containing the rm)"
    )


# --- (4) 9a-ter inline-gate staleness note ---------------------------------


def test_9ater_staleness_note_present():
    text = _text()
    assert "is the stale-family class" in text, (
        "the 9a-ter inline-gate bullet must carry the #1417 staleness-class diagnostic"
    )
    assert "run the Step 5a sync (now family-inclusive)" in text, (
        "the 9a-ter staleness note must name the Step 5a sync remedy"
    )


# --- (5) no full-message grep-exclusion literal in the gate region ---------


def test_no_grep_specfreshness_in_gate_region():
    """The gate section must never carry the full-message commit-filter form
    (a commit BODY mentioning the token would launder a genuine deliverable
    at the pre-gate re-sync). The existing test_step10d_guard3.py ban covers
    only the Guard-3 span, which stops at the fast-path heading and does NOT
    reach the gate section — hence this region's own negative assert."""
    region = _gate_region(_text())
    assert _FULL_MESSAGE_FILTER not in region, (
        "the pre-push gate section must not carry a full-message "
        "grep-exclusion invocation (subject-scoped filtering only)"
    )


# --- (6) the :(glob)/never-shell-expands comment at the SPECS line ---------


def test_glob_pathspec_comment_present():
    span = _step5a_span(_text())
    assert "`:(glob)` is a git pathspec (never shell-expands" in span, (
        "the SPECS comment must document that :(glob) is a git pathspec that never shell-expands"
    )
    assert "skip grain is PER-ITEM" in span, (
        "the SPECS comment must document the per-item skip grain of the "
        "branch-side-edit guard over the :(glob) family entry"
    )


# --- (7) Step 5a exclusion is subject-scoped (the MF-B pin) -----------------


def test_step5a_exclusion_is_subject_scoped():
    span = _step5a_span(_text())
    assert "--format='%H %s'" in span, (
        "the Step 5a branch-side-edit exclusion must emit '<sha> <subject>' "
        "(subject-scoped, the Guard-3 convention)"
    )
    assert "awk 'index($0, \"sync workflow-surface specs from\") == 0'" in span, (
        "the Step 5a exclusion must filter via the subject-scoped awk index() form "
        "keyed on the prescribed sync-subject anchor (#1789)"
    )
    assert _FULL_MESSAGE_INVERT not in span, (
        "the Step 5a exclusion must NOT use the full-message form (it would "
        "wrongly exclude a deliverable whose commit BODY mentions the token)"
    )


# --- (8) the re-sync never re-derives $WT (the MF-A pin) --------------------


def test_postgate_resync_does_not_rederive_wt():
    region = _gate_region(_text())
    bullet_idx = region.index("**Post-gate freshness re-sync (#1714")
    bullet = region[bullet_idx : region.index("[step10d] post-gate re-sync:", bullet_idx)]
    assert "ALREADY-BOUND `$WT`" in bullet, (
        "the re-sync bullet must state $WT is already bound by the merge flow"
    )
    assert "WT=$(git rev-parse --show-toplevel)` line" in bullet, (
        "the re-sync bullet must name the Step 5a WT-derivation line to DROP"
    )
    assert "do NOT re-derive `$WT` at Step 10d" in bullet, (
        "the re-sync bullet must ban re-deriving $WT at Step 10d (a repo-root "
        "cwd would rebind it to the shared root)"
    )


# --- (9) family-atomic declaration in Step 5a bash (#1714 new pin) --------


def test_step5a_family_atomicity_declared_in_bash():
    """The Step 5a block must declare family membership as bash associative
    array entries (the #1714 family-atomic skip)."""
    span = _step5a_span(_text())
    assert "declare -A FAMILY_OF" in span, (
        "Step 5a must declare family membership via `declare -A FAMILY_OF` "
        "(the #1714 family-atomic transitive skip)"
    )
    assert 'FAMILY_OF[".claude/workflow.yaml"]="workflow"' in span, (
        "the workflow family must include .claude/workflow.yaml"
    )
    assert 'FAMILY_OF[".claude/skills"]="workflow"' in span, (
        "the workflow family must include .claude/skills (markers.md + SKILL.md derived tables)"
    )
    assert 'FAMILY_OF["scripts/workflow_lint.py"]="lint"' in span, (
        "the lint family must include scripts/workflow_lint.py"
    )
    assert 'FAMILY_OF[":(glob)tests/test_workflow_lint*.py"]="lint"' in span, (
        "the lint family must include :(glob)tests/test_workflow_lint*.py"
    )
    assert 'FAMILY_OF[".claude/hooks"]="guard"' in span, (
        "the guard family must include .claude/hooks"
    )
    assert 'FAMILY_OF[":(glob)scripts/guard_*.sh"]="guard"' in span, (
        "the guard family must include :(glob)scripts/guard_*.sh — the "
        "guard-script implementations the test_guard_* pins execute; "
        "syncing tests without scripts half-syncs the tree (#1860/#1862; #1963)"
    )
    assert 'FAMILY_OF[":(glob)tests/test_guard_*.py"]="guard"' in span, (
        "the guard family must include :(glob)tests/test_guard_*.py"
    )
    assert 'FAMILY_OF["tests/test_workflow_yaml.py"]="workflow"' in span, (
        "the workflow family must include tests/test_workflow_yaml.py "
        "(imports render_*_table from workflow_lint AND reads workflow.yaml data)"
    )
    assert 'FAMILY_OF[":(glob)tests/test_issue_skill_*.py"]="workflow"' in span, (
        "the workflow family must include the skill pin-test glob "
        ":(glob)tests/test_issue_skill_*.py — prose-pin tests over "
        ".claude/skills content; syncing SKILL.md without its paired pin "
        "test reds the Step 9c gate (#1824 vintage skew; #1883)"
    )
    assert 'FAMILY_OF["tests/test_autonomous_session_watch.py"]="lint"' in span, (
        "the lint family must include tests/test_autonomous_session_watch.py "
        "(imports check_asw_docstring_pass_count from workflow_lint)"
    )
    assert 'FAMILY_OF["tests/test_guard_lessons_edit.py"]="guard"' in span, (
        "the guard family must include the explicit tests/test_guard_lessons_edit.py "
        "entry (it also matches the :(glob) but is declared explicitly for clarity)"
    )
    assert "DIRTY_FAMILIES" in span, (
        "the family-atomic loop must gate the sync on a DIRTY_FAMILIES associative array"
    )


# --- (10) auto-merge post-gate re-sync matches Step 5a family (#1714 drift guard) --


def test_step10d_family_atomicity_matches_step5a():
    """The auto-merge subsection's inline family-atomic bash block must
    declare the SAME FAMILY_OF entries as the Step 5a block. #1714 §4.6
    inlines the block into `#### The auto-merge procedure` so the
    post-gate re-sync uses the same family definition; a future editor
    that adds a family entry to Step 5a but forgets the auto-merge copy
    would silently produce a divergent post-gate re-sync — this test
    catches that."""
    text = _text()
    # Auto-merge span: from the H4 to the next H4 that follows.
    merge_start = text.index("#### The auto-merge procedure")
    merge_end = text.index("#### ", merge_start + 4)
    merge_span = text[merge_start:merge_end]
    # Every FAMILY_OF entry declared in the Step 5a span must also
    # appear in the auto-merge span (identical string).
    step5a_span = _step5a_span(text)
    for line in step5a_span.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAMILY_OF[") and stripped.endswith(
            ('="workflow"', '="lint"', '="guard"')
        ):
            assert stripped in merge_span, (
                f"Step 5a declares {stripped!r} but the auto-merge "
                f"post-gate re-sync block does not — the two copies must "
                f"stay in sync (§4.6 drift guard, #1714 methodology concern 1)"
            )


# --- (11) Step 5a sources fetched origin/main (#1747 durability pin) --------

# Banned bare-local-main fragments, built by CONCATENATION so this test file
# itself never carries them (the file's _FULL_MESSAGE_FILTER convention).
_BARE_CHECKOUT_MAIN = "checkout " + "main --"
_BARE_CHECKOUT_MAIN_SPECS = _BARE_CHECKOUT_MAIN + " $SAFE_SPECS"
_BARE_DIFF_QUIET_MAIN_SPECS = "diff --quiet " + "main -- $SAFE_SPECS"
_BARE_MERGE_BASE_MAIN = "merge-base HEAD " + "main"


def test_step5a_sources_fetched_origin_main():
    """#1747: the Step 5a spec-freshness sync sources FETCHED origin/main —
    never a possibly-lagging local main (#1724 synced REGRESSED spec bytes
    from a lagging shared-root main) — and skips the ENTIRE sync body on a
    main checkout (the explicit replacement for the old vacuous-local-diff
    self-no-op, which origin/main sourcing removed)."""
    span = _step5a_span(_text())
    # (a) bounded freshness fetch (the #1289/#1714 degrade-never-wedge shape)
    assert 'timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet' in span, (
        "Step 5a must run the bounded freshness fetch before the merge-base capture"
    )
    # (b) the surgical checkout sources origin/main
    assert "checkout origin/main -- $SAFE_SPECS" in span, (
        "the Step 5a sync must check out $SAFE_SPECS from origin/main"
    )
    # (c) the merge-base is captured against origin/main
    assert "merge-base HEAD origin/main" in span, (
        "the Step 5a pass-1 scan must merge-base against origin/main"
    )
    # (d) explicit on-main skip guard covering the whole sync body
    assert "rev-parse --abbrev-ref HEAD" in span, (
        "the on-main skip guard must probe the session branch via rev-parse --abbrev-ref HEAD"
    )
    assert '" = "main" ]' in span, (
        "the on-main skip guard must compare the session branch against main"
    )
    assert "[step5a] session on main" in span, (
        "the on-main skip must announce itself with the one-line [step5a] skip echo"
    )
    # (e) NEGATIVE: no bare-local-main sync forms survive in the span
    assert _BARE_CHECKOUT_MAIN_SPECS not in span, (
        "the Step 5a sync must not check out $SAFE_SPECS from bare local main (#1724)"
    )
    assert _BARE_DIFF_QUIET_MAIN_SPECS not in span, (
        "the Step 5a sync condition must not diff against bare local main (#1724)"
    )
    assert _BARE_MERGE_BASE_MAIN not in span, (
        "the Step 5a pass-1 scan must not merge-base against bare local main (#1724)"
    )
    # (f) NEGATIVE, span-scoped: no bare checkout-from-local-main fragment
    # anywhere in the span (catches the prose/comment class — e.g. the
    # in-block :(glob) comment — that the $SAFE_SPECS-anchored bans miss).
    assert _BARE_CHECKOUT_MAIN not in span, (
        "nothing in the Step 5a span may reference a bare checkout from local main"
    )


# --- (12) Step 10d verdict re-bind stanza (#1807 pins) ----------------------


def _automerge_span(text: str) -> str:
    """The auto-merge subsection: its H4 through the next H4 (same
    extraction as test (10))."""
    start = text.index("#### The auto-merge procedure")
    end = text.index("#### ", start + 4)
    return text[start:end]


def test_step10d_verdict_rebind_present():
    """#1807: the safe-case block re-binds the SHA-bound verdict to a
    post-gate sync tip ONLY under the mechanical probe — a --name-status
    enumeration whose every row is A/M and byte-identical to fetched
    origin/main. D/R*/C*/T/U rows fail CLOSED unconditionally (critic
    round-1 Must-Fix: --name-only would read a both-sides-absent deletion
    as main-identical while the certified landing tree CONTAINED the file
    via the own-diff overlay); line 1 is COMPOSED from the existing
    verdict (sed -n 1p), never typed; an unverifiable delta routes to the
    fail-closed BLOCKED arm (stale verdict consumed, no merge), and the
    merge itself is variable-gated on REBIND_OK=yes."""
    span = _automerge_span(_text())
    assert 'git -C "$WT" diff --name-status "$CERT_SHA" HEAD' in span, (
        "the re-bind probe must enumerate the cert-sha..HEAD delta with "
        "--name-status (NOT --name-only — the deletion corner, #1807)"
    )
    assert 'A|M) git -C "$WT" diff --quiet origin/main HEAD -- "$p" || DELTA_OK=no ;;' in span, (
        "A/M rows must keep the fetched-origin/main byte-identity probe"
    )
    assert "*)   DELTA_OK=no ;;   # D / R* / C* / T / U — never sync output" in span, (
        "every non-A/M status row must fail CLOSED unconditionally "
        "(the sync's `checkout origin/main --` can only add/modify)"
    )
    assert "sed -n 1p /tmp/issue-<N>-lint-verdict.txt" in span, (
        "line 1 must be COMPOSED from the existing verdict (sed -n 1p), never typed"
    )
    assert 'if [ "$REBIND_OK" = yes ]; then' in span, (
        "gh pr ready / gh pr merge must be variable-gated on REBIND_OK=yes "
        "(never a bare mid-block false as flow control)"
    )
    fail_echo = span.index("sync delta NOT verifiable as origin/main-identical A/M-only")
    blocked_echo = span.index("BLOCKED: verdict re-bind failed", fail_echo)
    rm_after = span.find("rm -f /tmp/issue-<N>-lint-verdict.txt", blocked_echo)
    assert rm_after != -1, (
        "an unverifiable delta must end in the stale verdict being consumed "
        "(rm -f in the fail-closed BLOCKED arm) — never a merge on an "
        "unverified tip"
    )


# --- (13) Step 9c pre-gate spec-freshness re-sync (#1807 / #1742 pin) -------


def test_step9c_pregate_sync_present():
    """#1807 fix (b): Step 9c step 1a must run the Step 5a family-atomic
    spec-freshness sync — a binding REFERENCE, never a third inlined
    FAMILY_OF copy (which would escape
    test_step10d_family_atomicity_matches_step5a's drift guard) — BEFORE
    the selector computes the gate subset, closing the #1742
    stale-worktree-spec gate-red class."""
    text = _text()
    start = text.index("**9c. Test-verdict gate")
    sel_idx = text.index("select_step9c_tests.py", start)
    region = text[start:sel_idx]
    assert "Pre-gate spec-freshness re-sync (#1742)" in region, (
        "Step 9c step 1a must carry the pre-gate spec-freshness re-sync "
        "BEFORE the first select_step9c_tests.py mention"
    )
    assert "Step 5a family-atomic" in region, (
        "the 9c re-sync must reference the Step 5a family-atomic block"
    )
    assert "never inline a THIRD `FAMILY_OF` copy" in region, (
        "the 9c re-sync must bind as a reference, not a third inlined copy"
    )


# --- (14) SPECS <-> SPECS_10D token-set equality (#1883 pin) -----------------


def test_specs_and_specs_10d_token_sets_match():
    """#1883: the Step 5a SPECS list and the Step 10d auto-merge inline
    copy's SPECS_10D list must carry the SAME pathspec token set. Check
    (10) mechanically pins the FAMILY_OF entries across the two copies;
    SPECS_10D itself was previously unpinned, so an editor adding a spec
    to SPECS but not SPECS_10D would silently produce a divergent
    post-gate re-sync — this test catches that."""
    text = _text()
    m5 = re.search(r'^\s*SPECS="([^"]+)"', _step5a_span(text), flags=re.M)
    assert m5, "Step 5a must declare SPECS as a one-line double-quoted assignment"
    m10 = re.search(r'^\s*SPECS_10D="([^"]+)"', _automerge_span(text), flags=re.M)
    assert m10, (
        "the auto-merge inline block must declare SPECS_10D as a one-line double-quoted assignment"
    )
    assert set(m5.group(1).split()) == set(m10.group(1).split()), (
        "SPECS (Step 5a) and SPECS_10D (Step 10d auto-merge inline copy) "
        "must carry identical pathspec token sets (#1883; the two lists "
        "are the same sync surface — a member added to one but not the "
        "other silently diverges the post-gate re-sync)"
    )
