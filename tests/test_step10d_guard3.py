"""Content-invariant tests for the /issue Step-10d merge hardening (task #787).

Step 10d's merge-strategy / guard-routing logic lives in SKILL.md PROSE
(`.claude/skills/issue/SKILL.md`), not in importable Python — so the pragmatic
way to pin its invariants is a set of content-string assertions over the skill
file plus the new repo-root `.gitattributes`. This mirrors how other workflow
invariants are pinned when the logic lives in a `.md` skill.

The four #787 sub-fixes these tests guard:

1. `.gitattributes` (NEW) — `merge=union` on the append-only task JSONL logs.
2. Guard-1 — strip FOREIGN task folders before the merge, split by whether the
   path exists on `origin/main` (checkout vs `git rm -f` — index AND working
   tree, #1244; an index-only `rm --cached` self-reverts under the
   pathspec-limited strip commit).
3. Fast-path pre-check — a FIVE-conjunct predicate (incl. `ADDED_ONLY=yes`)
   that routes far-behind small ADDED-only workflow-fix branches straight to
   the surgical additive checkout, plus the surgical compute block's
   ADDED-only / three-dot / workflow-surface-pathspec invariants (A3-new).
4. Guard-3 — the spec-freshness exclusion matches the commit SUBJECT line only,
   keyed on the prescribed sync-subject anchor
   (`awk 'index($0, "sync workflow-surface specs from") == 0'`, #1789 — never
   the bare token, which a deliverable subject legitimately carries), never a
   subject+body `--grep`.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
_GITATTRIBUTES = _REPO_ROOT / ".gitattributes"


def _skill_text() -> str:
    return issue_skill_text()


def _guard3_region(text: str) -> str:
    """The Guard-3 slice: from the guard-3 heading to the fast-path heading.

    Scoping the Guard-3 assertions to this slice keeps them independent of the
    unrelated Step-5a spec-freshness sync (line ~1925), which #1560
    subject-scoped to the same awk index() form (see
    test_step5a_grep_filter_untouched) and which is pinned separately.
    """
    start_marker = "**Branch-content / non-`main`-base guard.**"
    end_marker = "#### Fast-path routing pre-check"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "Guard-3 heading not found in SKILL.md"
    assert end != -1, "Fast-path pre-check heading not found in SKILL.md"
    assert start < end, "Guard-3 region must precede the fast-path pre-check"
    return text[start:end]


def _surgical_compute_block(text: str) -> str:
    """The surgical additive-checkout compute block (writes the additive-files
    temp list). This is the block the A3-new invariant pins.

    Bounded to exclude the new-shared-`src/` guard's OWN
    `--diff-filter=A origin/main HEAD` scan, which is a DIFFERENT block that
    legitimately stays two-dot and `src/`-scoped and must not be conflated.
    """
    anchor = "> /tmp/issue-<N>-additive-files.txt"
    end = text.find(anchor)
    assert end != -1, "surgical additive-files compute block not found"
    # Look back to the start of that fenced compute block.
    start = text.rfind("```bash", 0, end)
    assert start != -1, "opening fence for surgical compute block not found"
    return text[start : end + len(anchor)]


# --------------------------------------------------------------------------
# Sub-fix 1 — .gitattributes
# --------------------------------------------------------------------------


def test_gitattributes_exists_at_repo_root():
    assert _GITATTRIBUTES.is_file(), f".gitattributes must exist at repo root ({_GITATTRIBUTES})"


def test_gitattributes_union_merge_for_events_and_comments():
    text = _GITATTRIBUTES.read_text(encoding="utf-8")
    assert "tasks/**/events.jsonl merge=union" in text, "events.jsonl must use union merge"
    assert "tasks/**/comments.jsonl merge=union" in text, "comments.jsonl must use union merge"


def test_gitattributes_union_merge_for_agent_memory():
    """#896: agent-memory markdown must union-merge (2026-07-02 root pull-rebase incident)."""
    text = _GITATTRIBUTES.read_text(encoding="utf-8")
    assert ".claude/agent-memory/**/*.md merge=union" in text, (
        "agent-memory .md files must use union merge (#896)"
    )


def test_gitattributes_union_merge_for_eval_results_index():
    """#1534: eval_results/INDEX.md is append-dominant (one row per completed
    experiment, many concurrent sessions) and must union-merge locally."""
    text = _GITATTRIBUTES.read_text(encoding="utf-8")
    assert "eval_results/INDEX.md merge=union" in text, (
        "eval_results/INDEX.md must use union merge (#1534)"
    )


def test_gitattributes_union_merge_for_sparse_cones():
    """#1570: tests/sparse_cones.txt is an append-only cone registry (two
    same-day merge conflicts, 2026-07-19); both consumers read it as a set."""
    text = _GITATTRIBUTES.read_text(encoding="utf-8")
    assert "tests/sparse_cones.txt merge=union" in text, (
        "tests/sparse_cones.txt must use union merge (#1570)"
    )
    # Trailing-newline pin: union is line-oriented — a registry whose last
    # line lacked "\n" could join lines on a future union concatenation.
    registry = (_REPO_ROOT / "tests" / "sparse_cones.txt").read_text(encoding="utf-8")
    assert registry.endswith("\n"), "tests/sparse_cones.txt must end with a trailing newline"


def test_gitattributes_registry_not_unioned():
    """REGISTRY.json is last-writer-wins; a union merge would break its JSON."""
    text = _GITATTRIBUTES.read_text(encoding="utf-8")
    # There must be no active (non-comment) union line for REGISTRY.json.
    active_registry_union = [
        ln
        for ln in text.splitlines()
        if "REGISTRY.json" in ln and "merge=union" in ln and not ln.lstrip().startswith("#")
    ]
    assert not active_registry_union, "REGISTRY.json must NOT carry an active merge=union line"


# --------------------------------------------------------------------------
# Sub-fix 2 — Guard-1 foreign-folder strip (present-vs-added split)
# --------------------------------------------------------------------------


def test_guard1_foreign_present_vs_added_split_present():
    text = _skill_text()
    assert "FOREIGN_ON_MAIN" in text, "Guard-1 must split FOREIGN by presence on origin/main"
    assert "FOREIGN_BRANCH_ONLY" in text, (
        "Guard-1 must track branch-only-added foreign paths separately"
    )


def test_guard1_branch_added_foreign_dropped_not_checked_out():
    """#1244: the drop must remove index AND working tree (`git rm -f`), never
    index-only (`git rm --cached`) — Guard 1's strip commit is PATHSPEC-limited
    and records WORKING-TREE content for the named paths (git-commit(1) --only
    default), so an index-only deletion is committed right back and silently
    never lands (#1210: 19 resurrected paths)."""
    text = _skill_text()
    region = _merge_guards_region(text)
    assert 'git -C "$WT" rm -f --ignore-unmatch -- "${FOREIGN_BRANCH_ONLY[@]}"' in region, (
        "Guard-1 must drop branch-added foreign paths via git rm -f "
        "--ignore-unmatch (index AND working tree; a checkout would crash with "
        "pathspec-did-not-match, and an index-only rm --cached self-reverts "
        "under the pathspec-limited strip commit — #1210/#1244)"
    )
    assert "rm --cached" not in region, (
        "index-only `git rm --cached` must not appear in the merge-guards "
        "region — the pathspec-limited strip commit records working-tree "
        "content and would resurrect the paths (#1210/#1244)"
    )


def test_guard1_own_folder_carveout_present():
    text = _skill_text()
    assert 'grep -Ev "^tasks/[^/]+/<N>/"' in text, (
        "Guard-1 must carve out THIS task's own tasks/*/<N>/ folder"
    )


# --------------------------------------------------------------------------
# Sub-fix 2 (round-2 fix) — push the Guard-1 strip commit BEFORE gh pr merge
# --------------------------------------------------------------------------
# The Guard-1 strip commit is a LOCAL worktree commit; the safe-case
# `gh pr merge $MERGE_FORM` (#1288 merge-form routing) operates on the PR
# head ref at origin/issue-<N> (server-side), so an unpushed strip commit is
# invisible to that merge and the foreign tasks/* reverts would land on main
# silently. The safe case must push the strip commit (guarded by
# STRIPPED_FOREIGN=yes) before merging, and that push must APPEAR BEFORE the
# safe-case gh pr merge line in the file.


def test_guard1_tracks_stripped_foreign_flag():
    text = _skill_text()
    assert "STRIPPED_FOREIGN=no" in text, (
        "Guard-1 must initialize a STRIPPED_FOREIGN=no flag so the safe-case "
        "push fires only when a strip commit was actually created"
    )
    assert "STRIPPED_FOREIGN=yes" in text, (
        "Guard-1 must set STRIPPED_FOREIGN=yes after committing the strip"
    )


def test_safe_case_pushes_strip_commit_gated_on_stripped_foreign():
    text = _skill_text()
    assert 'git -C "$WT" push origin issue-<N>' in text, (
        "the safe-case block must push the branch to the PR head ref before "
        "gh pr merge $MERGE_FORM (otherwise the strip commit is invisible to "
        "the server-side merge)"
    )
    assert '[ "$STRIPPED_FOREIGN" = "yes" ]' in text, (
        "the safe-case push must be gated on STRIPPED_FOREIGN=yes so it fires "
        "only when Guard-1 actually created a strip commit"
    )


def test_safe_case_push_appears_before_gh_pr_merge():
    """The push must SEQUENCE before the safe-case gh pr merge $MERGE_FORM
    call (#1288), so the server-side merge sees the stripped branch tip."""
    text = _skill_text()
    # Scope the search to the safe-case block: the #1138 canonical
    # "Bare push / merge snippets" subsection (inserted earlier in Step 10d)
    # contains the bare push literal (plus a merge snippet), so
    # first-occurrence pins would retarget it.
    base = text.find("#### The auto-merge procedure (safe case")
    assert base != -1, "safe-case auto-merge heading not found in SKILL.md"
    # "$PR" is the #1897 probe-rebound PR number (was the <PR> placeholder).
    merge_line = 'gh pr merge "$PR" $MERGE_FORM --delete-branch=false'
    push_line = 'git -C "$WT" push origin issue-<N>'
    merge_offset = text.find(merge_line, base)
    push_offset = text.find(push_line, base)
    assert merge_offset != -1, "safe-case gh pr merge line not found in SKILL.md"
    assert push_offset != -1, "safe-case strip-commit push line not found in SKILL.md"
    assert push_offset < merge_offset, (
        "the Guard-1 strip-commit push must appear BEFORE the safe-case "
        f"gh pr merge line (push@{push_offset} must precede merge@{merge_offset})"
    )


# --------------------------------------------------------------------------
# Sub-fix 3a — fast-path routing pre-check (five-conjunct predicate)
# --------------------------------------------------------------------------


def test_fast_path_pre_check_heading_present():
    text = _skill_text()
    assert "#### Fast-path routing pre-check" in text


def test_fast_path_predicate_five_conjuncts():
    text = _skill_text()
    # The load-bearing ADDED-only conjunct + the pinned thresholds.
    assert "FAST_PATH=yes" in text
    assert "ADDED_ONLY=yes" in text, "the ADDED_ONLY gate must initialize to yes"
    assert '"$ADDED_ONLY" = "yes"' in text, (
        "the fast-path predicate must include the ADDED_ONLY=yes conjunct"
    )
    assert '"$BEHIND" -gt 1000' in text, "BEHIND_THRESHOLD literal 1000 must be present"
    assert '"$N_FILES" -le 15' in text, "N_FILES_MAX literal 15 must be present"
    assert '"$TASK_KIND" = "infra"' in text, "kind:infra conjunct must be present"
    assert "grep -qw 'wf-fix'" in text, "wf-fix tag conjunct must be present"


def test_fast_path_uses_added_only_status_gate():
    text = _skill_text()
    # A modified/renamed/deleted file must flip ADDED_ONLY off.
    assert '[ "$st" = "A" ] || ADDED_ONLY=no' in text, "any non-Added status must set ADDED_ONLY=no"


# --------------------------------------------------------------------------
# Sub-fix 3b / A3-new — surgical-land invariant
# --------------------------------------------------------------------------


def test_surgical_compute_block_added_only_filter():
    block = _surgical_compute_block(_skill_text())
    assert "--diff-filter=A" in block, "surgical compute block must restrict to ADDED-only files"
    assert "--diff-filter=AM" not in block, (
        "surgical compute block must NOT use --diff-filter=AM (would clobber a "
        "modified file wholesale)"
    )


def test_surgical_compute_block_three_dot_form():
    block = _surgical_compute_block(_skill_text())
    assert "origin/main...HEAD" in block, (
        "surgical compute block must use the three-dot origin/main...HEAD form"
    )
    # The two-dot form (space, not ...) must not appear in THIS block.
    assert "--diff-filter=A origin/main HEAD" not in block, (
        "surgical compute block must NOT use the two-dot origin/main HEAD form"
    )


def test_surgical_compute_block_workflow_surface_pathspec_tokens():
    block = _surgical_compute_block(_skill_text())
    for token in (
        '".claude/"',
        '"CLAUDE.md"',
        '".gitattributes"',
        '"docs/methodology/issue_<N>.md"',
    ):
        assert token in block, f"surgical compute block must include the {token} pathspec entry"
    # The pre-existing task-own pathspec entries must remain.
    for token in (
        '"tasks/*/<N>/"',
        '"figures/issue_<N>/"',
        '"eval_results/issue_<N>/"',
        '"ood_eval_results/issue_<N>/"',
    ):
        assert token in block, f"surgical compute block must retain the existing {token} entry"


def test_new_shared_src_guard_stays_two_dot_and_src_scoped():
    """The DIFFERENT new-shared-src guard block must be left untouched: it stays
    two-dot `origin/main HEAD` and scoped to src/ (NOT converted to three-dot)."""
    text = _skill_text()
    assert (
        'git -C "$WT" diff --name-only --diff-filter=A origin/main HEAD -- \\\n'
        '  "src/explore_persona_space/" > /tmp/issue-<N>-new-src.txt' in text
    ), "the new-shared-src guard must remain two-dot origin/main HEAD, src-scoped"


# --------------------------------------------------------------------------
# Sub-fix 4 — Guard-3 spec-freshness subject-line-only exclusion
# --------------------------------------------------------------------------

# The banned full-message commit-filter literal under the #1789 sync-subject
# anchor, built by CONCATENATION so this test file itself never carries the
# form its negative asserts scan for (the test_issue_skill_lint_family_sync.py
# _FULL_MESSAGE_FILTER convention).
_FULL_MESSAGE_ANCHOR_FILTER = "--grep=" + "'sync workflow-surface specs from'"


def test_guard3_spec_freshness_subject_only_awk_filter():
    text = _skill_text()
    assert "awk 'index($0, \"sync workflow-surface specs from\") == 0'" in text, (
        "Guard-3 spec-freshness exclusion must match on the SUBJECT line only "
        "via awk index() keyed on the prescribed sync-subject anchor (#1789), "
        "not a subject+body --grep"
    )


def test_guard3_region_does_not_use_grep_spec_freshness():
    """Within the Guard-3 slice, the exclusion must NOT be a `--grep` (which
    matches the commit BODY too) — neither the bare-token form nor the #1789
    anchored form. The Step-5a sync at line ~1925 is a separate region,
    subject-scoped by #1560 to the same awk index() form (pinned by
    test_step5a_grep_filter_untouched)."""
    region = _guard3_region(_skill_text())
    assert "--grep='spec-freshness'" not in region, (
        "Guard-3 must not use --grep='spec-freshness' (over-matches commit body)"
    )
    assert _FULL_MESSAGE_ANCHOR_FILTER not in region, (
        "Guard-3 must not use the anchored full-message --grep form either "
        "(over-matches commit body; subject-scoped awk index() only, #1789)"
    )
    assert "%H %s" in region, (
        "Guard-3 must emit '<sha> <subject>' per commit for a subject-scoped filter"
    )


def test_step5a_grep_filter_untouched():
    """The Step-5a spec-freshness sync's branch-side-edit exclusion is
    SUBJECT-scoped (deliberately changed by #1560 from the old full-message
    --grep/--invert-grep form — which Guard 3's own note bans: a commit BODY
    mentioning the token would launder a genuine branch deliverable, and at
    the #1560 pre-gate re-sync position that mis-exclusion would check out
    origin/main over a reviewed deliverable). Supersedes the pre-#1560
    defensive pin that the full-message filter remained (that pin was out of
    #787 scope; #1560 deliberately subject-scoped the Step 5a filter)."""
    text = _skill_text()
    start = text.index('SPECS=".claude/agents')
    span = text[start : text.index("429 pacing at every ensemble fan-out", start)]
    assert "--format='%H %s'" in span, (
        "the Step-5a exclusion must emit '<sha> <subject>' (subject-scoped)"
    )
    assert "awk 'index($0, \"sync workflow-surface specs from\") == 0'" in span, (
        "the Step-5a exclusion must filter via the subject-scoped awk index() form "
        "keyed on the prescribed sync-subject anchor (#1789)"
    )
    assert "--grep='spec-freshness' --invert-grep" not in span, (
        "the Step-5a sync must not use the full-message --grep filter (#1560)"
    )
    assert _FULL_MESSAGE_ANCHOR_FILTER + " --invert-grep" not in span, (
        "the Step-5a sync must not use the anchored full-message --grep filter either (#1789)"
    )


# --------------------------------------------------------------------------
# Task #1047 — Step-10d pre-merge/pre-push hardening pins
# --------------------------------------------------------------------------


def _merge_guards_region(text: str) -> str:
    """The merge-safety-guards slice: from the guards heading to the fast-path
    heading (Guard 0 + guards 1-3 live here)."""
    start_marker = "#### Merge safety guards (run before the merge commands)"
    end_marker = "#### Fast-path routing pre-check"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "Merge safety guards heading not found in SKILL.md"
    assert end != -1, "Fast-path pre-check heading not found in SKILL.md"
    assert start < end, "guards region must precede the fast-path pre-check"
    return text[start:end]


def test_guard0_mem_committed_flag_in_guards_region():
    """#1047 Guard 0: the agent-memory pre-commit (flagged MEM_COMMITTED) must
    live in the guards region, and the safe-case pre-merge push condition must
    be extended by the flag — a local-only memory commit is invisible to the
    server-side rebase, same mechanism as STRIPPED_FOREIGN (#906)."""
    text = _skill_text()
    region = _merge_guards_region(text)
    assert "MEM_COMMITTED=no" in region, "Guard 0 must initialize MEM_COMMITTED=no"
    assert "MEM_COMMITTED=yes" in region, "Guard 0 must set MEM_COMMITTED=yes after committing"
    assert '[ "$MEM_COMMITTED" = "yes" ]' in text, (
        "the safe-case pre-merge push condition must include MEM_COMMITTED=yes"
    )


def _artifact_confirmed_region(text: str) -> str:
    """The artifact-confirmed merge procedure slice (the surgical additive
    checkout lands here; the round-2 gate-verdict pins scope to it)."""
    start_marker = "#### The artifact-confirmed merge procedure"
    end_marker = "#### Post-merge stale-task-folder guard"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "artifact-confirmed merge heading not found in SKILL.md"
    assert end != -1, "post-merge stale-task-folder guard heading not found"
    assert start < end, "artifact-confirmed region must precede the post-merge guard"
    return text[start:end]


def test_prepush_workflow_lint_gate_between_fast_path_and_auto_merge():
    """#1047 lint gate: the Pre-push workflow-lint gate subsection must sit
    between the fast-path pre-check and the auto-merge procedure, and be
    named at >=3 bind hooks besides its heading (safe case, recovery,
    surgical checkout) so every merge form runs it (#931). Round-2
    strengthening (Codex Minor): the load-bearing invariant is that the
    SURGICAL block computes a payload-attributed verdict — consuming
    GATED_RC + the normalized failure-line files via `comm -23` — BEFORE the
    first `git add`, and gates the stage/commit/push on an explicit
    GATE_VERDICT branch, not just a phrase count."""
    text = _skill_text()
    fast = text.find("#### Fast-path routing pre-check")
    gate = text.find("#### Pre-push workflow-lint gate")
    auto = text.find("#### The auto-merge procedure")
    assert gate != -1, "Pre-push workflow-lint gate heading not found"
    assert -1 < fast < gate < auto, (
        "the gate heading must sit between the fast-path and auto-merge sections"
    )
    assert text.count("Pre-push workflow-lint gate") >= 4, (
        "the gate must be named at its heading plus >=3 bind hooks"
    )
    region = _artifact_confirmed_region(text)
    first_add = region.find("git add --")
    assert first_add != -1, "surgical block must stage by explicit path (git add --)"
    rc_consumption = region.find('[ "$GATED_RC" -ne 0 ]')
    subtraction = region.find("comm -23")
    verdict_branch = region.find('if [ "$GATE_VERDICT" = "pass" ]')
    assert -1 < rc_consumption < first_add, (
        "GATED_RC must be consumed by a conditional BEFORE the first git add "
        "(a captured-but-unchecked RC is a hollow gate)"
    )
    assert -1 < subtraction < first_add, (
        "the failure-line-set subtraction (comm -23) must run BEFORE the first git add"
    )
    assert -1 < verdict_branch < first_add, (
        "the stage/commit/push must sit inside an explicit GATE_VERDICT=pass branch"
    )


def test_gate_executable_block_normalizes_subtracts_and_persists_verdict():
    """Round-2 (Codex M1): the gate subsection must carry a fully EXECUTABLE
    block — documented normalization of `workflow_lint:` failure lines,
    `comm -23` set subtraction, and a persisted binary verdict file the
    binding sites consume (fenced bash blocks are separate shells)."""
    text = _skill_text()
    gate = text.find("#### Pre-push workflow-lint gate")
    auto = text.find("#### The auto-merge procedure")
    region = text[gate:auto]
    assert "grep -h '^workflow_lint: '" in region, "normalization must keep workflow_lint: lines"
    assert "sed -E 's/:[0-9]+:/::/g'" in region, "normalization must blank :<line>: numbers"
    assert "comm -23" in region, "NEW = gated - baseline must use comm -23"
    assert '[ "$GATED_RC" -ne 0 ]' in region, "GATED_RC must be consumed in the gate block"
    for verdict in ("block", "pass", "skip-artifact-only"):
        assert f"echo {verdict} > /tmp/issue-<N>-lint-verdict.txt" in region, (
            f"the gate block must persist the '{verdict}' verdict to the verdict file"
        )


def test_gate_verdict_file_gates_safe_case_and_recovery():
    """Round-2 (Codex M1/M2): the persisted verdict file must be consumed by
    an explicit conditional BEFORE `gh pr ready` (safe case) and BEFORE the
    recovery `git -C "$WT" push` — a missing verdict file fails CLOSED."""
    text = _skill_text()
    probe = "grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt"
    first = text.find(probe)
    second = text.find(probe, first + 1)
    assert first != -1 and second != -1, (
        "both the safe case and the recovery must consume the persisted verdict file"
    )
    ready = text.find('gh pr ready "$PR"')  # the #1897 probe-rebound "$PR"
    assert -1 < first < ready < second, "the safe-case verdict conditional must precede gh pr ready"
    rec = text.find("#### Merge-conflict recovery")
    rec_push = text.find('git -C "$WT" push\n', rec)
    assert -1 < rec < second < rec_push, (
        "the recovery verdict conditional must precede the recovery push"
    )


def test_fast_path_lower_bound_and_no_flagless_additive_xargs():
    """Round-2 (Claude M2): the FAST_PATH conjunction must require a NON-EMPTY
    filtered own-diff (`-ge 1`), and every xargs consuming the additive-files
    list must carry `-r` (--no-run-if-empty) — an empty-input xargs checkout
    degenerates to `git checkout issue-<N> --` with NO pathspec, a branch
    SWITCH of the shared repo root."""
    text = _skill_text()
    assert '[ "$N_FILES" -ge 1 ]' in text, "FAST_PATH must require N_FILES >= 1"
    assert "xargs -a /tmp/issue-<N>-additive-files.txt" not in text, (
        "no flag-less additive-list xargs may remain (must be xargs -r -a)"
    )
    assert text.count("xargs -r -a /tmp/issue-<N>-additive-files.txt") >= 5, (
        "checkout / add / commit / restore / rm additive-list consumers must all carry -r"
    )


def test_gate_block_cleanup_recipe_hook_admitted():
    """Round-2 (Claude M1): the on-block cleanup must be the hook-admitted
    two-step — index-only `restore --staged` unstage, then plain `rm` of the
    now-untracked A-only files. The one-shot restore-with-worktree-flag form
    is mechanically BLOCKED by scripts/guard_repo_root_branch.sh's #897
    restore detector (allow requires --staged AND no worktree flag; verified
    by hook simulation 2026-07-05, exit 2)."""
    text = _skill_text()
    assert "--staged --worktree" not in text, (
        "the hook-blocked one-shot restore form must not be documented anywhere"
    )
    assert 'git -C "$REPO_ROOT" restore --staged --' in text, (
        "cleanup step 1 must be the index-only unstage (hook allow-arm)"
    )
    assert "rm -f --" in text, (
        "cleanup step 2 must remove the now-untracked A-only files via plain rm"
    )


def test_safe_case_push_rederives_unpushed_from_git_state():
    """Round-2 (Codex M3): the safe-case pre-merge push must RE-DERIVE
    'unpushed local commits exist' from git state — fenced bash blocks are
    separate shell invocations, so the Guard-0/1 flags are unset there; the
    flags remain same-block conveniences only. A missing origin/issue-<N>
    ref counts as unpushed (fails toward pushing)."""
    text = _skill_text()
    assert (
        '[ "$(git -C "$WT" rev-list --count origin/issue-<N>..HEAD 2>/dev/null'
        ' || echo 1)" -gt 0 ]' in text
    ), (
        "the push condition's load-bearing arm must be the re-derived "
        "unpushed-commit count against origin/issue-<N>"
    )


def test_post_merge_guard_push_retry_uses_sync_repo_root():
    """#1047 sync naming: the post-merge guard's push retry must route through
    scripts/sync_repo_root.py, never a hand-rolled repo-root `git pull` (#967
    `fatal: Cannot autostash`)."""
    text = _skill_text()
    start = text.find("#### Post-merge stale-task-folder guard")
    assert start != -1, "post-merge stale-task-folder guard heading not found"
    region = text[start:]
    assert "sync_repo_root.py" in region, (
        "the post-merge guard push retry must name scripts/sync_repo_root.py"
    )
    assert "|| { git pull --rebase=merges --autostash && git push origin main; }" not in region, (
        "the hand-rolled repo-root pull retry must be gone from the post-merge guard"
    )


# --------------------------------------------------------------------------
# Round 3 — lint-gate crash class (#1047 concerns: lint-gate-crash-fails-open
# BLOCKER + lint-baseline-crash-erased CONCERN)
# --------------------------------------------------------------------------


def _gate_region(text: str) -> str:
    """The Pre-push workflow-lint gate subsection (shared executable block)."""
    start = text.find("#### Pre-push workflow-lint gate")
    end = text.find("#### The auto-merge procedure")
    assert start != -1, "Pre-push workflow-lint gate heading not found"
    assert end != -1, "auto-merge procedure heading not found"
    assert start < end, "gate subsection must precede the auto-merge procedure"
    return text[start:end]


def test_gate_crash_arm_fails_closed_before_pass():
    """Round-3 (#1047 `lint-gate-crash-fails-open` BLOCKER): a workflow_lint.py
    CRASH (rc>1, or rc!=0 with ZERO normalized `workflow_lint:` failure lines)
    must fail CLOSED at BOTH gate sites — an explicit crash arm BEFORE any
    `pass` write (shared block) / before the attribution `GATE_VERDICT=block`
    arm (surgical block) — never land in the else->pass leg."""
    text = _skill_text()
    region = _gate_region(text)
    crash_arm = region.find('[ "$GATED_RC" -gt 1 ]')
    crash_write = region.find("echo crash > /tmp/issue-<N>-lint-verdict.txt")
    pass_write = region.find("echo pass > /tmp/issue-<N>-lint-verdict.txt")
    assert crash_arm != -1, "shared gate must carry an explicit GATED_RC>1 crash arm"
    assert crash_write != -1, "shared gate must persist the crash verdict to the verdict file"
    assert -1 < crash_arm < pass_write, (
        "the crash arm must be evaluated BEFORE the pass write (fail CLOSED on a crash)"
    )
    assert "[ ! -s /tmp/issue-<N>-lint-gated-norm.txt ]" in region, (
        "rc!=0 with zero normalized gated failure lines must be classified as a crash"
    )
    surg = _artifact_confirmed_region(text)
    s_crash_arm = surg.find('[ "$GATED_RC" -gt 1 ]')
    s_crash = surg.find("GATE_VERDICT=crash")
    s_block = surg.find("GATE_VERDICT=block")
    assert s_crash_arm != -1, "surgical block must carry the GATED_RC>1 crash arm"
    assert s_crash != -1, "surgical block must set GATE_VERDICT=crash on a linter crash"
    assert -1 < s_crash < s_block, "the surgical crash arm must precede the attribution block arm"


def test_gate_baseline_rc_captured_not_erased():
    """Round-3 (#1047 `lint-baseline-crash-erased` CONCERN): baseline lint legs
    must capture BASE_RC per leg (feeding the crash arm), never `|| true`-erase
    it; a baseline rc=1 WITH lines stays a legitimate red baseline (the
    subtraction handles it)."""
    text = _skill_text()
    assert "lint-baseline.txt 2>&1 || true" not in text, (
        "no baseline lint leg may erase its exit code with `|| true`"
    )
    assert text.count(_BASE_RC_FOLD) >= 4, (
        "both baseline legs at both gate sites must capture BASE_RC (2 legs x 2 sites; "
        "round-4: the capture is the no-downgrade fold, not the bare `|| BASE_RC=$?`)"
    )
    assert '[ "$BASE_RC" -gt 1 ]' in text, "a BASE_RC>1 baseline crash must feed the crash arm"
    assert "[ ! -s /tmp/issue-<N>-lint-baseline-norm.txt ]" in text, (
        "a baseline rc!=0 with zero normalized lines must be classified as a crash"
    )


# #1097 pins: the sha-equality conjunct (byte-identical at both consumers), the
# nonempty-line-2 guard conjunct (hardens the empty-vs-empty `[ "" = "" ]` cell),
# and the success-checked merge wrapper the pass-branch rm must sit inside.
_SHA_CHECK = (
    '[ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)"'
    ' = "$(git -C "$WT" rev-parse HEAD)" ]'
)
_NONEMPTY_SHA_CHECK = '[ -n "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" ]'
# #1288 merge-form routing: the safe case merges via the kind-derived
# $MERGE_FORM variable; the merge-conflict recovery block is hard-pinned to
# --squash (its just-added merge commit makes --rebase documented-doomed,
# #1041). Each consumer is pinned to ITS OWN success-checked merge form.
_MERGE_SUCCESS_IF_SAFE = 'if gh pr merge "$PR" $MERGE_FORM --delete-branch=false; then'
_MERGE_SUCCESS_IF_RECOVERY = "if gh pr merge <PR> --squash --delete-branch=false; then"


def test_gate_verdict_sha_bound_at_write_and_both_consumers():
    """#1097: the gate block must SHA-BIND the verdict (append the certified
    branch-tip sha as line 2) and BOTH file consumers (safe case + recovery)
    must accept a pass/skip verdict only while the current tip equals the
    certified sha — a hand-written `echo pass >` verdict (the #1082 move)
    lacks the sha and fails closed; any post-certification commit does too."""
    text = _skill_text()
    region = _gate_region(text)
    sha_append = 'git -C "$WT" rev-parse HEAD >> /tmp/issue-<N>-lint-verdict.txt'
    assert sha_append in region, "the gate block must append the certified sha to the verdict file"
    assert text.count(_SHA_CHECK) >= 2, (
        "both verdict-file consumers must compare line 2 against the current branch tip"
    )
    probe = "grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt"
    first = text.find(probe)
    second = text.find(probe, first + 1)
    m1 = text.find(_MERGE_SUCCESS_IF_SAFE, first)
    m2 = text.find(_MERGE_SUCCESS_IF_RECOVERY, second)
    assert -1 < first < m1, "safe case: the success-checked merge must follow its conditional"
    assert -1 < second < m2, "recovery: the success-checked merge must follow its conditional"
    assert _SHA_CHECK in text[first:m1], (
        "the SAFE-CASE consumer conditional must carry the sha-equality conjunct"
    )
    assert _SHA_CHECK in text[second:m2], (
        "the RECOVERY consumer conditional must carry the sha-equality conjunct"
    )
    assert _NONEMPTY_SHA_CHECK in text[first:m1], (
        "the safe-case conditional must guard against an empty line 2 "
        "(the empty-vs-empty [ '' = '' ] cell must fail closed)"
    )
    assert _NONEMPTY_SHA_CHECK in text[second:m2], (
        "the recovery conditional must guard against an empty line 2"
    )


def test_gate_verdict_consumed_only_after_merge_success():
    """#1097 (supersedes the round-3 consume-once pin): the pass-branch rm must
    fire only AFTER `gh pr merge` returns success, at BOTH consumers, so a
    non-lint transport failure (#1041 rebase refusal -> squash retry) stays
    certified by the same gate run; the block/crash/stale branches still rm."""
    text = _skill_text()
    rm_line = "rm -f /tmp/issue-<N>-lint-verdict.txt"
    assert text.count(rm_line) >= 4, (
        "both consumers must still rm the verdict file in the success AND fail-closed branches"
    )
    probe = "grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt"
    first = text.find(probe)
    second = text.find(probe, first + 1)
    ready = text.find('gh pr ready "$PR"')  # the #1897 probe-rebound "$PR"
    m1 = text.find(_MERGE_SUCCESS_IF_SAFE, first)
    m2 = text.find(_MERGE_SUCCESS_IF_RECOVERY, second)
    assert -1 < first < ready < m1, "safe case: conditional -> gh pr ready -> success-checked merge"
    assert text.find(rm_line, first, m1) == -1, (
        "safe case: NO rm may sit between the verdict conditional and the merge attempt "
        "(the pre-merge consume orphaned the verdict on the #1041 transport failure)"
    )
    rm1 = text.find(rm_line, m1)
    e1 = text.find('echo "MERGE FAILED', m1)
    assert -1 < m1 < rm1 < e1, (
        "safe case: the rm must sit INSIDE the merge-success branch — after the "
        "success-checked merge, before its failure echo"
    )
    assert -1 < second < m2, "recovery: the success-checked merge must follow its conditional"
    assert text.find(rm_line, second, m2) == -1, (
        "recovery: NO rm may sit between the verdict conditional and the merge attempt"
    )
    rm2 = text.find(rm_line, m2)
    e2 = text.find('echo "MERGE FAILED', m2)
    assert -1 < m2 < rm2 < e2, (
        "recovery: the rm must sit INSIDE the merge-success branch — after the "
        "success-checked merge, before its failure echo"
    )


def test_gate_trigger_diff_exit_guarded():
    """Round-3 (Codex r2 unaddressed case): a FAILED trigger `git diff
    origin/main...HEAD` must not read as an artifact-only skip — the shared
    gate materializes the own-diff with an explicit exit check routing to the
    crash verdict; the straight-into-grep pipe trigger form is gone."""
    text = _skill_text()
    region = _gate_region(text)
    assert (
        'if ! git -C "$WT" -c core.quotePath=false diff --name-only origin/main...HEAD '
        "> /tmp/issue-<N>-own-diff.txt" in region
    ), "the trigger diff must be materialized with an explicit exit check"
    assert region.count("echo crash > /tmp/issue-<N>-lint-verdict.txt") >= 2, (
        "both the failed-trigger-diff arm and the linter-crash arm must write the crash verdict"
    )
    assert (
        "cat /tmp/issue-<N>-lint-verdict.txt   "
        "# line 1: pass | block | crash | skip-artifact-only; line 2: certified branch-tip sha"
    ) in region, "the verdict enumeration must include crash and the certified-sha line"


# --------------------------------------------------------------------------
# Round 4 — reconciled v3 residuals (#1047: `lint-leg-rc-erasure-crash-masked`
# persisted BLOCKER + `surgical-additive-diff-fails-open` persisted CONCERN)
# --------------------------------------------------------------------------

# The no-downgrade (max) per-leg rc fold. Pinning the SHAPE (not just literal
# counts of `|| VAR=$?`): a gated leg-1 crash (rc=2, zero lines) must survive
# a leg-2 rc=1-with-lines — the bare last-failure-wins capture erases the
# crash and defeats the crash arm.
_BASE_RC_FOLD = '|| { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }'
_GATED_RC_FOLD = '|| { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }'


def test_lint_leg_rc_no_downgrade_fold_at_all_four_sites():
    """Round-4 (#1047 `lint-leg-rc-erasure-crash-masked` BLOCKER): every lint
    leg-pair (BASE + GATED, shared gate + surgical block = 4 sites, 8 legs)
    must capture its per-leg rc with the NO-DOWNGRADE (max) fold, and no bare
    last-failure-wins `|| VAR=$?` capture may remain anywhere in the skill."""
    text = _skill_text()
    gate = _gate_region(text)
    surg = _artifact_confirmed_region(text)
    for region, name in ((gate, "shared gate"), (surg, "surgical block")):
        assert region.count(_BASE_RC_FOLD) == 2, (
            f"{name}: both BASE legs must use the no-downgrade BASE_RC fold"
        )
        assert region.count(_GATED_RC_FOLD) == 2, (
            f"{name}: both GATED legs must use the no-downgrade GATED_RC fold"
        )
    assert "|| BASE_RC=$?" not in text, (
        "no bare last-failure-wins BASE_RC capture may remain (crash-masking erasure)"
    )
    assert "|| GATED_RC=$?" not in text, (
        "no bare last-failure-wins GATED_RC capture may remain (crash-masking erasure)"
    )


def test_surgical_additive_producer_guarded_and_empty_list_hard_stops():
    """Round-4 (#1047 `surgical-additive-diff-fails-open` CONCERN): the
    surgical additive-list producer must be materialize-then-check (a FAILED
    diff hard-stops — routed to epm:merge-failed — never read as an empty
    list), and an EMPTY additive list on the deliverables-missing landing
    must hard-stop instead of pushing + posting
    `epm:merged {surgical_checkout: true}` with nothing committed."""
    text = _skill_text()
    region = _artifact_confirmed_region(text)
    guard = region.find(
        'if ! git -C "$WT" -c core.quotePath=false diff --name-only '
        "--diff-filter=A origin/main...HEAD"
    )
    assert guard != -1, (
        "the surgical additive-list producer must check its OWN exit code "
        "(materialize-then-check, mirroring the shared gate's trigger diff)"
    )
    empty_stop = region.find("elif [ ! -s /tmp/issue-<N>-additive-files.txt ]; then")
    assert empty_stop != -1, (
        "an empty additive list on the deliverables-missing landing must be "
        "detected (phantom-success hard stop)"
    )
    assert region.count("SURGICAL ABORT") >= 2, (
        "both abort arms (failed producer diff, empty list) must fail loud"
    )
    assert region.count("epm:merge-failed") >= 2, (
        "the abort arms must route to the epm:merge-failed handling"
    )
    first_consumer = region.find("xargs -r -a /tmp/issue-<N>-additive-files.txt git")
    assert -1 < guard < empty_stop < first_consumer, (
        "the producer guard + empty-list hard stop must precede the first "
        "additive-list consumer (never proceed to checkout/stage/push)"
    )


# --------------------------------------------------------------------------
# #1085 pins (task #1105) — the guard-block recovery-contract paragraph
# (SKILL.md prose added by #1085 after the #813/#1056 guard-block incidents)
# --------------------------------------------------------------------------

_RECOVERY_HEADING = "**Guard-block recovery contract (improvised variants of this compound).**"


def _normalized(text: str) -> str:
    """Whitespace-collapse for prose pins.

    The recovery-contract paragraph is hard-wrapped markdown prose, so a
    benign re-wrap (moving line breaks) must not false-fail the anchor pins.
    The executable-block pins above stay on raw text by design (fenced bash
    never re-wraps).
    """
    return " ".join(text.split())


def _guard_block_recovery_paragraph(text: str) -> str:
    """The #1085 guard-block recovery-contract paragraph: from its bold
    heading (inside the artifact-confirmed region, right after the surgical
    additive-checkout executable block) to the `epm:merged` posting sentence
    that follows it."""
    region = _artifact_confirmed_region(text)
    start = region.find(_RECOVERY_HEADING)
    assert start != -1, (
        "guard-block recovery-contract heading not found in the artifact-confirmed region (#1085)"
    )
    end = region.find("epm:merged", start)
    assert end != -1, "the epm:merged posting sentence must follow the recovery contract"
    return region[start:end]


def test_recovery_contract_paragraph_present_and_names_the_hook():
    """#1085 pin (task #1105): the guard-block recovery contract must exist
    exactly once, inside the artifact-confirmed region, and must name its
    fencing mechanism — the scripts/guard_repo_root_branch.sh PreToolUse
    hook, its #897 checkout-pathspec + restore detectors, and the
    use-the-fence-lines-VERBATIM directive."""
    text = _skill_text()
    assert text.count(_RECOVERY_HEADING) == 1, (
        "the recovery-contract heading must appear exactly once (no stale copy drift)"
    )
    para = _normalized(_guard_block_recovery_paragraph(text))
    assert "scripts/guard_repo_root_branch.sh" in para, (
        "the contract must name the fencing hook script"
    )
    assert "PreToolUse" in para, "the contract must identify the fence as a PreToolUse hook"
    assert "#897 checkout-pathspec detector" in para, (
        "the contract must name the #897 checkout-pathspec detector"
    )
    assert "#897 restore detector" in para, "the contract must name the #897 restore detector"
    assert 'use the `-C "$REPO_ROOT"`-qualified fence lines VERBATIM' in para, (
        "the contract must direct retries to the -C-qualified fence lines verbatim"
    )


def test_recovery_contract_never_generalizes_dash_c_waiver():
    """#1085 pin (task #1105): the -C waiver scoping must survive edits — the
    waiver is admitted ONLY because both fence forms are non-destructive at
    the shared root, and the contract must forbid generalizing
    -C "$REPO_ROOT" to escape a block on any other / destructive command."""
    para = _normalized(_guard_block_recovery_paragraph(_skill_text()))
    assert "NON-DESTRUCTIVE at the shared root" in para, (
        "the waiver justification (both fence forms non-destructive) must stay"
    )
    assert 'NEVER generalize `-C "$REPO_ROOT"` to escape a block' in para, (
        "the never-generalize--C scoping rule must stay"
    )
    assert "never point `-C` at the repo root for a destructive op" in para, (
        "the hook's own destructive-op block-message rationale must stay"
    )


def test_recovery_contract_reruns_producer_before_corrected_consumer():
    """#1085 pin (task #1105): the load-bearing recovery rule — a PreToolUse
    deny rejects the ENTIRE tool call, so the producer clause writing the
    additive-files list never ran; the retry must RE-RUN the producer diff
    BEFORE the corrected consumer (incidents #813 / #1056, 2026-07-05:
    consumer-only retries died on exit 128 / `cat: ... No such file`)."""
    para = _normalized(_guard_block_recovery_paragraph(_skill_text()))
    assert "the WHOLE compound Bash call was skipped" in para, (
        "the whole-compound-skipped semantics must stay"
    )
    assert "a PreToolUse deny rejects the entire tool call" in para, (
        "the PreToolUse-deny mechanism sentence must stay"
    )
    assert "`/tmp/issue-<N>-additive-files.txt` (the producer diff above) never ran" in para, (
        "the contract must name the producer's list file as the skipped casualty"
    )
    assert "RE-RUNS the producer diff clause" in para, (
        "the producer-regeneration retry rule must stay"
    )
    assert "BEFORE re-running the corrected `-C`-qualified consumer" in para, (
        "the producer-before-consumer retry ordering must stay"
    )
    assert "exit 128" in para and "No such file" in para, (
        "the consumer-only failure signature must stay"
    )
    assert "#813" in para and "#1056" in para, (
        "the motivating incident references (#813 / #1056) must stay"
    )


# --------------------------------------------------------------------------
# Task #1184 — fail-loud producer reads (post-merge guard + Guard 1)
# --------------------------------------------------------------------------


def _post_merge_guard_region(text: str) -> str:
    """The Post-merge stale-task-folder guard span only.

    The region ends at the next `####` H4 heading (the Terminal-teardown
    sub-section, #1723 — which comes right after the guard in the
    code-change path) OR at ``## Resume semantics`` when no intervening
    H4 exists. Scoping to the H4 boundary keeps the guard's `bash`
    fence count invariant intact under Step 10d structural growth: the
    Terminal-teardown H4's own fenced bash block belongs to its own
    sub-section, not to the guard.
    """
    start = text.find("#### Post-merge stale-task-folder guard")
    assert start != -1, "post-merge stale-task-folder guard heading not found"
    tail = text[start:]
    # Prefer the earliest sibling H4 after the guard; fall back to
    # `## Resume semantics` when no such H4 exists.
    next_h4 = tail.find("\n#### ", 1)
    fallback = tail.find("## Resume semantics")
    end_rel = next_h4 if next_h4 != -1 and (fallback == -1 or next_h4 < fallback) else fallback
    assert end_rel != -1 and end_rel > 0, (
        "guard region must precede either the next H4 or the Resume semantics header"
    )
    return tail[:end_rel]


def test_post_merge_guard_materializes_lstree_and_fails_closed():
    """#1184: the guard must materialize ls-tree to a file and exit-check
    every producer (CANON / fetch / ls-tree) in TERMINAL failure arms — a
    failed producer must never read as 'no duplicates' (#644 fail-open)."""
    text = _skill_text()
    region = _post_merge_guard_region(text)
    assert "> /tmp/issue-<N>-postmerge-lstree.txt" in region
    assert 'elif ! git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main' in region
    assert 'elif ! git -C "$REPO_ROOT" fetch origin main --quiet' in region
    assert '[ -z "$CANON" ]' in region
    assert "mapfile -t DUPES < <(git" not in region, (
        "the fail-open piped ls-tree mapfile form must be gone"
    )
    # CHAIN MEMBERSHIP (stats-critic round 1): the work arm must be an
    # `elif` of the SAME if/elif chain — a detached mapfile below the `fi`
    # re-opens the fail-open (no set -e: a taken `false` arm does not halt
    # the block, so a detached mapfile would read the empty/partial file).
    assert "elif mapfile -t DUPES < <(grep -E" in region, (
        "DUPES must be filled from the materialized FILE inside the chain"
    )
    # TERMINAL `false` per failure arm (stats-critic round 1): each failure
    # echo must be immediately followed by `false` — a bare echo would be a
    # loud-log, exit-0 fail-open.
    # The shared "cannot certify..." phrase ends BOTH the fetch and ls-tree
    # arms' echoes, so the two arm-UNIQUE phrases are pinned as well —
    # otherwise removing `false` from exactly one of those two arms would
    # not trip any pin (code-review r1 Minor).
    for arm_msg in (
        "refusing to classify duplicates",
        "git fetch origin main FAILED",
        "git ls-tree origin/main FAILED",
        "cannot certify no stale task folders",
    ):
        assert re.search(re.escape(arm_msg) + r'[^\n]*"\n\s*false\b', region), (
            f"failure arm {arm_msg!r} must end in a terminal false"
        )
    assert region.find("cannot certify no stale task folders") < region.find(
        "elif mapfile -t DUPES"
    ), "failure arms must precede the DUPES work arm (fail CLOSED)"
    # Success-path byte-equivalence anchors (the exact grep programs):
    assert 'grep -E "^tasks/[^/]+/<N>$"' in region
    assert 'grep -v -F -x "$CANON"' in region


def test_guard1_materializes_foreign_diff_and_fails_closed():
    """#1184: Guard 1's foreign-tasks/ trigger diff must be materialized and
    exit-checked — a failed git diff must never read as 'no foreign files'
    (the #458 incident class would ride the merge)."""
    text = _skill_text()
    region = _merge_guards_region(text)
    assert "> /tmp/issue-<N>-guard1-tasks-diff.txt" in region
    assert "mapfile -t FOREIGN < <(git" not in region, (
        "the fail-open piped diff mapfile form must be gone"
    )
    # Chain membership + terminal false (stats-critic round 1; same
    # rationale as the post-merge-guard pins above):
    assert "elif mapfile -t FOREIGN < <(grep -Ev" in region, (
        "FOREIGN must be filled from the materialized FILE inside the chain"
    )
    # #1268 loop reshape: the diff-failure echo now records GUARD1_STATE and
    # breaks; the terminal `false` moved to the single post-loop disposition
    # arm (asserted separately below) — intent preserved, never deleted.
    assert re.search(
        r'cannot certify no foreign tasks/ paths; do NOT merge"\n\s*GUARD1_STATE=diff-failed\b',
        region,
    ), "the Guard-1 diff-failure arm must record the diff-failed state (loop shape, #1268)"
    assert re.search(
        r'if \[ "\$GUARD1_STATE" != ok \]; then\n[^\n]*\n\s*false\b',
        region,
    ), "the post-loop terminal arm must end in a terminal false on any non-ok state"
    assert region.find("cannot certify no foreign tasks/ paths") < region.find(
        "elif mapfile -t FOREIGN"
    ), "the failure arm must precede the FOREIGN work arm (fail CLOSED)"
    # Success-path byte-equivalence anchor:
    assert 'grep -Ev "^tasks/[^/]+/<N>/"' in region


def test_gate_lint_legs_run_landing_tree_copy():
    """#1212: BOTH lint leg pairs run from the ephemeral landing tree — the
    baseline pre-overlay (payload-free), the gated post-overlay — never the
    raw worktree or repo-root copies (#1112 vintage false-blocks), with
    fail-closed construction (GT_RC) and in-block teardown."""
    text = _skill_text()
    gate = text.find("#### Pre-push workflow-lint gate")
    auto = text.find("#### The auto-merge procedure")
    assert -1 < gate < auto
    region = text[gate:auto]
    assert 'git -C "$WT" archive origin/main --' in region, (
        "the gate tree must be built from origin/main's lint-scanned surface"
    )
    assert region.count('"$GT/scripts/workflow_lint.py"') >= 4, (
        "all four lint-leg invocations (2 baseline + 2 gated) must run the gate-tree copy"
    )
    assert '"$WT/scripts/workflow_lint.py"' not in region, (
        "the branch-tip lint invocation must not reappear (#1112 false-blocks)"
    )
    assert '"$REPO_ROOT/scripts/workflow_lint.py"' not in region, (
        "the baseline legs must not run the repo-root copy (root vintage/dirt asymmetry)"
    )
    overlay = region.find('git -C "$WT" show "HEAD:$p" > "$GT/$p"')
    base_legs = region.find("# BASELINE legs")
    gated_legs = region.find("# GATED legs")
    assert -1 < base_legs < overlay < gated_legs, (
        "the payload overlay must sit BETWEEN the baseline and gated lint legs"
    )
    assert (
        'git -C "$WT" -c core.quotePath=false diff --name-only --no-renames origin/main...HEAD'
        in region
    ), (
        "the overlay listing COMMAND must disable rename detection (rename SOURCES "
        "must be rm-ed); the comment's mention of --no-renames does not count"
    )
    assert '[ "$GT_RC" -ne 0 ]' in region, (
        "gate-tree construction failures must fail CLOSED via the crash arm"
    )
    assert region.count('rm -rf "$GT"') >= 2, (
        "the gate tree must be torn down AFTER the verdict too — the construction's "
        "own rm -rf (self-heal) does not satisfy the teardown pin"
    )


# --------------------------------------------------------------------------
# #1245 — background + wedge-bound the two Step 10d gate executable blocks
# (port of the Step 9c background + rc-file pattern; precedent pin:
# tests/test_issue_skill_step9c_compare_background.py, #1197)
# --------------------------------------------------------------------------


def _surgical_region(text: str) -> str:
    """The artifact-confirmed (form (iii)) merge-procedure subsection."""
    start = text.find("#### The artifact-confirmed merge procedure")
    end = text.find("#### Post-merge stale-task-folder guard")
    assert start != -1, "artifact-confirmed merge procedure heading not found"
    assert end != -1, "post-merge stale-task-folder guard heading not found"
    assert start < end, "surgical subsection must precede the post-merge guard"
    return text[start:end]


def test_gate_blocks_backgrounded_with_wedge_bounds():
    """#1245: both Step 10d gate executable blocks run as ONE background Bash
    call with per-leg wedge bounds; a missing verdict file / outcome sentinel
    after completion means the background run DIED (fail CLOSED, never a
    silent pass); the old one-fenced-foreground-invocation phrasing is gone.
    A foreground gate run is the #991/#996/#1129 600s-tool-cap kill class
    (~9-12+ min of lint + TG legs per block)."""
    text = _skill_text()
    gate = _gate_region(text)
    surgical = _surgical_region(text)
    # (i) the background prescription is present in BOTH gate regions:
    assert "run_in_background" in gate, "shared gate block must prescribe run_in_background"
    assert "run_in_background" in surgical, "surgical block must prescribe run_in_background"
    # (ii) all four lint legs per region carry the wedge bound (the sizing
    # comments deliberately do NOT quote the literal — command lines only):
    assert gate.count("timeout --kill-after=60s 1800s") >= 4, (
        "all four shared-block lint legs must carry the 1800s wedge bound (#2253 r5)"
    )
    assert surgical.count("timeout --kill-after=60s 1800s") >= 4, (
        "all four surgical-block lint legs must carry the 1800s wedge bound (#2253 r5)"
    )
    # (ii-b) the network ops inside the backgrounded blocks are bounded too:
    assert 'timeout --kill-after=30s 120s git -C "$WT" fetch origin main' in gate, (
        "the shared block's fetch must carry a 120s bound (a hung fetch wedges the bg call)"
    )
    assert "timeout --kill-after=30s 300s git push origin main" in surgical, (
        "the surgical pass-arm push must carry a 300s bound (a hung push wedges the "
        "bg call with the outcome sentinel unwritten)"
    )
    # (iii) missing-verdict death semantics — fail CLOSED (shared block):
    assert "died before writing a verdict" in gate, (
        "the shared block's completion-read must treat a missing verdict file as "
        "the background run having died (fail CLOSED)"
    )
    assert "rm -f /tmp/issue-<N>-lint-verdict.txt" in gate, (
        "the shared block must pre-rm the verdict file so a file present at "
        "completion provably came from THIS run"
    )
    # (iv) form (iii) outcome sentinel — pre-rm + all three terminal writes:
    assert "rm -f /tmp/issue-<N>-surgical-outcome.txt" in surgical
    assert "echo landed > /tmp/issue-<N>-surgical-outcome.txt" in surgical
    assert "echo push-failed > /tmp/issue-<N>-surgical-outcome.txt" in surgical
    assert "echo blocked-cleaned > /tmp/issue-<N>-surgical-outcome.txt" in surgical
    # (v) negative: the old foreground one-invocation phrasing is gone:
    assert "ONE fenced invocation" not in text, (
        "the surgical preamble's foreground 'ONE fenced invocation' phrasing must "
        "not reappear (silently reintroduces the 600s-cap kill class)"
    )
    assert "runs in ONE fenced block" not in text, (
        "the surgical 'runs in ONE fenced block' phrasing must not reappear"
    )


# --------------------------------------------------------------------------
# Task #1253 — post-merge guard work arm: sparse scratch worktree, hook-safe
# --------------------------------------------------------------------------


def test_post_merge_guard_work_arm_scratch_worktree_and_hook_safe(tmp_path):
    """#1253: the post-merge guard's WORK ARM must remove the duplicate(s) in
    a SPARSE SCRATCH WORKTREE detached at the fetched origin/main — never a
    root `git rm`, which fails pathspec whenever the local root predates the
    just-landed server-side merge and drove the improvised, hook-blocked
    checkout-pathspec fallback (session 82f5b16a, /issue 1198). Pins:
    (i) staging anchors (add flag order; cone init BEFORE set — git 2.34's
    `set --cone` is silently a literal pattern; scratch-scoped rm; old
    root-side forms gone); (ii) terminal `false` on every new failure arm;
    (iii) the STRONG pin — the fenced block, `<N>`->1198 substituted, fed to
    the LIVE scripts/guard_repo_root_branch.sh via stdin PreToolUse JSON must
    return rc=0 (a future edit reintroducing a hook-blocked shape fails
    here), plus `bash -n` syntax-cleanliness; (iv) chain-unreachability — the
    `&&`-join and populate->rm->commit ordering keep the empty-index
    delete-everything-tree commit path provably unreachable, and the region
    carries EXACTLY ONE fenced bash block so a second executable fence cannot
    escape the hook probe (extend the probe to every fence if one is ever
    deliberately added)."""
    text = _skill_text()
    region = _post_merge_guard_region(text)

    # (i) staging anchors. The add-line flag order `--detach --no-checkout` is
    # load-bearing for a bare copy (the reversed order trips the hook's
    # checkout+detach detector); cone init must precede `set`.
    assert 'worktree add --detach --no-checkout "$SCRATCH" origin/main' in region
    assert region.index("sparse-checkout init --cone") < region.index(
        'sparse-checkout set "${DUPES[@]}"'
    ), "sparse-checkout init --cone must precede set (git 2.34 ordering)"
    assert 'git -C "$SCRATCH" rm -r -q "${DUPES[@]}"' in region
    for hit in re.findall(r"git[^\n]*\brm -r\b[^\n]*", region):
        assert '-C "$SCRATCH"' in hit, f"non-scratch-scoped git rm -r: {hit!r}"
    assert 'git rm -r "${DUPES[@]}"' not in region, (
        "the old root-side `git rm` must be gone (it pathspec-fails whenever "
        "the local root predates the just-landed merge)"
    )
    assert 'cd "$REPO_ROOT"   # stay on main' not in region, (
        "the old root-cd work-arm preamble must be gone"
    )

    # (ii) every new failure arm ends in a terminal `false` (an optional
    # scratch-cleanup line may sit between the echo and the false). The two
    # verify arms share the "cannot certify the removal landed" tail, so their
    # arm-UNIQUE phrases are pinned as well (same rationale as the detection
    # arms' pins above: removing `false` from exactly one shared-tail arm must
    # trip a pin).
    cleanup = re.escape('git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null')
    for arm_msg in (
        "scratch-worktree staging FAILED",
        "did NOT land on origin/main",
        "verify fetch FAILED",
        "verify ls-tree FAILED",
        "cannot certify the removal landed",
        "STILL on origin/main after push",
        "persist after 2 root syncs",
    ):
        assert re.search(
            re.escape(arm_msg) + r'[^\n]*"\n(?:\s*' + cleanup + r"\n)?\s*false\b",
            region,
        ), f"work-arm failure arm {arm_msg!r} must end in a terminal false"

    # (iv) chain-unreachability: EXACTLY ONE fenced bash block; the staging
    # steps are one `&&`-joined chain in populate -> rm -> commit order, so a
    # failed rm can never reach commit (a commit from the add-time EMPTY index
    # would produce a delete-everything tree).
    fences = re.findall(r"```bash\n(.*?)```", region, re.DOTALL)
    assert len(fences) == 1, (
        f"the guard region must carry EXACTLY ONE fenced bash block, got {len(fences)}"
    )
    fence = fences[0]
    rm_clause = '&& git -C "$SCRATCH" rm -r -q "${DUPES[@]}"'
    commit_clause = '&& git -C "$SCRATCH" commit'
    assert rm_clause in fence, "the scratch rm must stay &&-joined into the staging chain"
    assert commit_clause in fence, "the scratch commit must stay &&-joined into the staging chain"
    assert (
        fence.index("checkout --detach origin/main")
        < fence.index(rm_clause)
        < fence.index(commit_clause)
    ), "staging order must be populate -> rm -> commit"

    # (iii) the strong pin: the `<N>`->1198-substituted block passes the LIVE
    # hook (stdin PreToolUse JSON, the tests/test_guard_repo_root_branch.py
    # `_run` convention) and `bash -n`.
    guard_script = _REPO_ROOT / "scripts" / "guard_repo_root_branch.sh"
    block = fence.replace("<N>", "1198")
    payload = json.dumps({"tool_input": {"command": block}})
    proc = subprocess.run([str(guard_script)], input=payload, text=True, capture_output=True)
    assert proc.returncode == 0, (
        f"guard hook blocked the post-merge guard block (rc={proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    script = tmp_path / "postmerge_guard_block.sh"
    script.write_text(block, encoding="utf-8")
    bn = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert bn.returncode == 0, f"bash -n failed on the substituted block:\n{bn.stderr}"


# --------------------------------------------------------------------------
# Task #1300 — unpushed-mv pre-check (canonical folder must be ON origin
# before any duplicate classification)
# --------------------------------------------------------------------------


def test_post_merge_guard_unpushed_mv_precheck_syncs_before_classify():
    """#1300: the guard must not classify duplicates while the CANONICAL
    folder is absent from origin/main — under routine local-main push lag
    origin's only copy is the OLD-status folder of a not-yet-pushed status
    mv, and deleting it left origin with ZERO folders for the task (origin
    commit 2a1a9cbc0b deleted tasks/reviewing/1291, the only 1291 folder on
    origin; recovery merge f26462fc1b). Pins: (i) POSITION — the pre-check
    is an `elif` arm of the SAME chain, between the ls-tree producer arm and
    the DUPES work arm; (ii) RECOVERY — the arm invokes
    scripts/sync_repo_root.py inside a bounded `for _ in 1 2` loop,
    re-resolves CANON via task.py find, and RE-FETCHES + REGENERATES the
    materialized ls-tree before each re-check (the ls-tree re-check is the
    arbiter — the helper's exit 0 includes the in-flight state); (iii) FAIL
    CLOSED — a still-absent CANON ends in a terminal echo + false
    (arm-unique phrase), and the arm never duplicates the classification;
    (iv) FALL THROUGH — the recovery lives in the arm's CONDITION list whose
    FINAL command is the still-absent re-test, so a successful recovery
    falls through to the DUPES classification against the regenerated
    file. The #1253 strong pin (live hook + bash -n on the substituted
    fence) covers the new arm's executability."""
    text = _skill_text()
    region = _post_merge_guard_region(text)
    fences = re.findall(r"```bash\n(.*?)```", region, re.DOTALL)
    assert len(fences) == 1, "guard region must still carry exactly one fenced bash block"
    fence = fences[0]

    precheck = fence.find('elif ! grep -qxF -- "$CANON" /tmp/issue-<N>-postmerge-lstree.txt')
    lstree_arm = fence.find('elif ! git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main')
    dupes_arm = fence.find("elif mapfile -t DUPES < <(grep -E")
    assert precheck != -1, "the unpushed-mv pre-check arm must exist (#1300)"
    assert -1 < lstree_arm < precheck < dupes_arm, (
        "the pre-check must sit BETWEEN the ls-tree producer arm and the DUPES work arm"
    )

    arm = fence[precheck:dupes_arm]
    # (ii) recovery mechanics:
    assert "for _ in 1 2; do" in arm, "the recovery must be bounded to 2 sync attempts"
    assert 'uv run python "$REPO_ROOT/scripts/sync_repo_root.py"' in arm, (
        "the recovery must land the local mv via the sanctioned root sync"
    )
    assert 'uv run python "$REPO_ROOT/scripts/task.py" find <N>' in arm, (
        "the recovery must RE-RESOLVE the canonical path after the sync "
        "(the sync pull-rebases the local root; the canonical status can "
        "change in either lag direction)"
    )
    assert 'git -C "$REPO_ROOT" fetch origin main --quiet' in arm, (
        "each attempt must re-fetch origin/main before the re-check"
    )
    assert "> /tmp/issue-<N>-postmerge-lstree.txt" in arm, (
        "each attempt must REGENERATE the materialized ls-tree file (the "
        "DUPES arm classifies against the fresh file on fall-through)"
    )
    # (iv) fall-through: the condition list's FINAL command is the
    # still-absent re-test (recovery success -> condition false -> the
    # DUPES arm runs against the regenerated file).
    assert '! grep -qxF -- "$CANON" /tmp/issue-<N>-postmerge-lstree.txt; }; then' in arm, (
        "the recovery must live in the arm's CONDITION with the still-absent "
        "re-test as its final command (successful recovery falls through)"
    )
    # (iii) terminal false on the still-absent branch, arm-unique phrase
    # (same phrase+false regex convention as the #1184/#1253 pins):
    assert re.search(
        re.escape("still ABSENT from origin/main after 2 root syncs") + r'[^\n]*"\n\s*false\b',
        arm,
    ), "the still-absent arm must end in a terminal echo + false"
    assert "mapfile -t DUPES" not in arm, (
        "the pre-check arm must not duplicate the DUPES classification "
        "(fall-through, not a copied work arm)"
    )
    assert "worktree" not in arm, (
        "the pre-check arm must never touch the scratch worktree (it deletes nothing)"
    )


# --------------------------------------------------------------------------
# Task #1792 — scratch-cone scripts/hooks + empty-dir residue disposition
# --------------------------------------------------------------------------


def test_post_merge_guard_cone_hooks_and_empty_dir_disposition():
    """#1792: two post-merge-guard recipe fixes (live incident #1780).

    (a) CONE — the scratch worktree's sparse cone must include scripts/hooks
    alongside the duplicates: the removal commit's own pre-commit gitleaks
    hook runs `bash scripts/hooks/gitleaks_scoped.sh` worktree-root-relative
    with always_run, so a duplicates-only cone exits 127 on the FIRST commit
    attempt every time (#1780; toplevel .gitleaks.toml/.gitleaksignore ride
    cone mode automatically).

    (b) EMPTY-DIR DISPOSITION — the local-residue tail gains an arm between
    the second sync attempt and the terminal fail-loud check: when the
    persisting paths hold ZERO files AND zero symlinks (one JOINT probe over
    all persisting paths), rmdir them depth-first (rmdir refuses non-empty
    dirs by construction), then RE-DERIVE STALE_LOCAL via the same
    `ls -d "${DUPES[@]}"` probe — the re-derive is the arbiter, never a
    blind clear, so late-arriving content or a failed rmdir still reaches
    the loud failure. Ordering pinned via find-from-index anchors (the
    fence carries several `ls -d "${DUPES[@]}"` occurrences). The #1253
    strong pin (live hook + bash -n on the substituted fence) covers the
    new arm's executability."""
    text = _skill_text()
    region = _post_merge_guard_region(text)
    fences = re.findall(r"```bash\n(.*?)```", region, re.DOTALL)
    assert len(fences) == 1, "guard region must still carry exactly one fenced bash block"
    fence = fences[0]

    # (a) the cone line carries scripts/hooks AFTER the duplicates.
    assert 'sparse-checkout set "${DUPES[@]}" scripts/hooks' in fence, (
        "the scratch cone must include scripts/hooks — the removal commit's "
        "own pre-commit gitleaks hook exits 127 without it (#1780)"
    )

    # (b) zero-content probe -> depth-first rmdir -> RE-DERIVE -> fail-loud,
    # in that order (find-from-index disambiguates the repeated probes).
    probe = fence.find("-type f -o -type l")
    assert probe != -1, "the zero-content probe must count files AND symlinks"
    rmdir_idx = fence.find("-depth -type d -exec rmdir {} \\;")
    assert rmdir_idx != -1, "the depth-first rmdir disposition must exist"
    assert probe < rmdir_idx, "the zero-content probe must gate the rmdir"
    rederive = fence.find(
        'STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)',
        rmdir_idx,
    )
    assert rederive != -1, (
        "STALE_LOCAL must be RE-DERIVED via the same ls -d probe AFTER the "
        "rmdir — never blind-cleared"
    )
    fail_loud = fence.find("persist after 2 root syncs")
    assert fail_loud != -1, "the terminal fail-loud echo must survive"
    assert rmdir_idx < rederive < fail_loud, (
        "ordering must be rmdir -> re-derive -> fail-loud (the re-derive is "
        "the arbiter: real content still reaches the loud failure)"
    )
    assert 'STALE_LOCAL=""' not in fence, "the arm must never blind-clear STALE_LOCAL"


# --------------------------------------------------------------------------
# Task #1573 — TG legs: selector-sized timeout + junit-node-grain subtraction
# --------------------------------------------------------------------------


def _tg_blocks(text: str) -> list[str]:
    """The two Step 10d TG executable blocks (shared-gate + surgical form
    (iii)) — same extraction convention as
    tests/test_step9c_baseline.py::test_skill_tg_blocks_pin_tmpdir_routing."""
    blocks = [b for b in text.split("```") if 'uv run pytest "${TG_TESTS[@]}"' in b]
    assert len(blocks) == 2, "expected the shared-gate + surgical TG blocks"
    return blocks


def test_tg_leg_selector_sized_timeout():
    """#1573: both TG blocks size their pytest bound from the selector's
    machine-greppable `recommended-timeout-s=` stderr line (gated map,
    /tmp/issue-<N>-tg-map-err.txt) with the fixed 600s (#1646; pre-#1573:
    300s) as the fallback floor; NO fixed `--kill-after=30s 300s` remains as a TG pytest
    bound. Since #2296 the BASELINE leg is the `mapped-baseline` helper call:
    its outer wrapper carries the TG_T-derived `$((TG_T + 420))s` bound (+420
    for scratch materialization + selection + teardown) and threads the raw
    TG_T through `--timeout-s` as the inner pytest bound, so exactly ONE
    direct `${TG_T}s` bound remains (the gated leg). The surgical pass-arm
    `git push origin main` 300s bound is a DIFFERENT command — exempted by
    its `git push` CONTENT, never by line number (it may move); the bounded
    `git ... fetch origin main` 120s line is likewise content-exempted."""
    text = _skill_text()
    for block in _tg_blocks(text):
        assert "grep -oE 'recommended-timeout-s=[0-9]+'" in block, (
            "TG block must grep the selector's sizing line"
        )
        assert "/tmp/issue-<N>-tg-map-err.txt" in block
        assert "TG_T=600" in block, "the 600s floor fallback must be present (#1646)"
        assert block.count("timeout --kill-after=30s ${TG_T}s") == 1, (
            "the GATED pytest leg carries the sized bound (the baseline leg is "
            "the #2296 mapped-baseline helper, bounded below)"
        )
        assert "timeout --kill-after=30s $((TG_T + 420))s" in block, (
            "the #2296 baseline helper call must carry the TG_T-derived +420s bound"
        )
        assert '--timeout-s "$TG_T"' in block, (
            "the baseline helper must thread the raw TG_T as its inner pytest bound"
        )
        for line in block.splitlines():
            for fixed in ("--kill-after=30s 300s", "--kill-after=30s 600s"):
                if fixed in line:
                    assert "git push" in line, (
                        f"a fixed bound must not remain on a TG pytest leg: {line!r}"
                    )


def test_tg_leg_node_grain_subtraction():
    """#1573: both TG blocks carry the junit-NODE-grain NEW-failure pipeline —
    `FAILED`/`ERROR` summary lines -> sed msg-suffix strip (NOT awk field-2:
    pytest preserves spaces in string param ids, so field-2 truncation would
    collide `test_foo[a b]` with `test_foo[a c]`) -> sort -u -> comm -23 into
    tg-new-nodes.txt — with a stale-file init and the node file OR'd into the
    verdict beside the file-grain hit set (a unit-test failure names the TEST,
    not a payload path; file-grain alone verdicts `pass` on the #1573
    founding incident)."""
    text = _skill_text()
    for block in _tg_blocks(text):
        assert ": > /tmp/issue-<N>-tg-new-nodes.txt" in block, "stale-file init missing"
        assert "grep -E '^(FAILED|ERROR) '" in block
        assert "sed -E 's/^(FAILED|ERROR) //; s/ - .*$//'" in block, (
            "the msg-suffix strip must be the sed form (never awk '{print $2}')"
        )
        assert "tg-$leg-nodes.txt" in block
        assert "comm -23 /tmp/issue-<N>-tg-gated-nodes.txt" in block
        assert "/tmp/issue-<N>-tg-baseline-nodes.txt > /tmp/issue-<N>-tg-new-nodes.txt" in block
        assert (
            "[ -s /tmp/issue-<N>-tg-new.txt ] || [ -s /tmp/issue-<N>-tg-new-nodes.txt ]" in block
        ), "the verdict must OR the node-grain file beside the file-grain hit set"


def test_tg_leg_classify_and_merge_base_pin():
    """#2348: both TG blocks (1) cut the BASELINE at a merge-base, never
    current origin/main — `--base "$TG_BASE_REF"` resolved via
    `git merge-base origin/main HEAD` with a LOUD origin/main degrade — and
    (2) run the classify-new-nodes SET-mismatch split AFTER the node-grain
    comm and BEFORE the verdict read, in-place-filtering tg-new-nodes.txt
    (the verdict operand is unchanged) with the unclassifiable stale-file
    init HOISTED beside the tg-new-nodes init. The merge-base `-C` target is
    pinned PER-BLOCK (never a shared literal): the baseline base must match
    the VINTAGE OF THE TREE THE GATED LEG RUNS — form (i)/(ii)'s gated leg
    runs the branch-tip worktree, so its merge-base resolves in `-C "$WT"`;
    form (iii)'s gated leg runs the ROOT tree (current local main +
    payload), so its merge-base resolves in `-C "$REPO_ROOT"` and must NOT
    carry `${WT` — a `${WT:-...}` form there would anchor the baseline at
    the branch fork point while the gated leg runs current main, so any
    mapped test main broke since fork would read NEW and the classify split
    would keep it blocking (its file is in the baseline selection),
    reintroducing the false-block class on the surgical fence (#2348 critic
    round-1 Must-Fix)."""
    text = _skill_text()
    blocks = _tg_blocks(text)
    for block in blocks:
        # (1) merge-base-pinned baseline + loud degrade:
        assert '--base "$TG_BASE_REF"' in block, "the baseline must consume the resolved base"
        assert "--base origin/main" not in block, "no leg may pin the baseline to origin/main"
        mb_lines = [ln for ln in block.splitlines() if "TG_BASE_REF=$(git -C " in ln]
        assert len(mb_lines) == 1, "exactly ONE merge-base resolution per block"
        assert "merge-base origin/main HEAD" in mb_lines[0]
        assert "TG_BASE_REF=origin/main" in block, (
            "the resolution-failure arm must degrade LOUDLY to origin/main"
        )
        # (2) selection-sidecar parse + classify split, ordered comm -> classify
        # -> verdict; init hoisted to the top of the block:
        assert "sed -n 's/^selected_path=//p'" in block, "TG_BASE_SELECTED parse missing"
        assert ": > /tmp/issue-<N>-tg-unclassifiable-nodes.txt" in block, (
            "the unclassifiable stale-file init must be present"
        )
        assert block.index(": > /tmp/issue-<N>-tg-unclassifiable-nodes.txt") < block.index(
            '"$TG_S9B" mapped-baseline'
        ), "the unclassifiable init must be HOISTED above the TG legs"
        classify = block.index("classify-new-nodes \\")
        comm_nodes = block.index("comm -23 /tmp/issue-<N>-tg-gated-nodes.txt")
        verdict_or = block.index(
            "[ -s /tmp/issue-<N>-tg-new.txt ] || [ -s /tmp/issue-<N>-tg-new-nodes.txt ]"
        )
        assert comm_nodes < classify < verdict_or, (
            "classify must run AFTER the node-grain comm and BEFORE the verdict read"
        )
        assert '--baseline-selected "${TG_BASE_SELECTED:-/__eps_no_selected__}"' in block, (
            "an unset TG_BASE_SELECTED must degrade via the never-matching default"
        )
        assert "--out-block /tmp/issue-<N>-tg-new-nodes.txt" in block, (
            "classify must in-place-filter the verdict operand"
        )
        assert "--out-unclassifiable /tmp/issue-<N>-tg-unclassifiable-nodes.txt" in block
        assert "WARN: classify-new-nodes failed" in block, (
            "a classify failure must be a loud status-quo WARN, never silent"
        )
    # Per-block -C target (the #2348 critic round-1 Must-Fix) — _tg_blocks
    # returns document order: shared form (i)/(ii) first, surgical form (iii)
    # second (the same order test_tg_blocks_bash_parseable relies on):
    shared, surgical = blocks
    shared_mb = next(ln for ln in shared.splitlines() if "TG_BASE_REF=$(git -C " in ln)
    assert '-C "$WT"' in shared_mb, "form (i)/(ii) must resolve the merge-base in the worktree"
    surgical_mb = next(ln for ln in surgical.splitlines() if "TG_BASE_REF=$(git -C " in ln)
    assert '-C "$REPO_ROOT"' in surgical_mb, (
        "form (iii) must resolve the merge-base in the ROOT tree its gated leg runs"
    )
    assert "${WT" not in surgical_mb, (
        "form (iii)'s merge-base must NEVER take the ${WT:-...} fallback form"
    )


def test_tg_blocks_bash_parseable(tmp_path):
    """#1847 (origin #1790): both Step 10d TG executable fences must parse
    under `bash -n`. Orchestrators execute these blocks verbatim; the
    substring pins above are structurally blind to a comment/formatting edit
    that breaks parseability (#1790: msg-strip comments spliced inside the
    node-grain pipelines broke both fences while every pin stayed green).
    Convention matches the #1253 strong pin: substitute `<N>` -> a dummy
    numeric id, write to a file, assert `bash -n` rc==0. Note: a comment
    line placed after a trailing `|` is valid bash, so the negative smoke
    is not exhaustive comment coverage."""
    text = _skill_text()
    for i, block in enumerate(_tg_blocks(text)):
        # _tg_blocks segments start with the fence language tag ("bash\n");
        # strip it so bash -n sees only the block body.
        body = block.split("\n", 1)[1] if block.startswith("bash") else block
        script = tmp_path / f"tg_block_{i}.sh"
        script.write_text(body.replace("<N>", "1790"), encoding="utf-8")
        bn = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert bn.returncode == 0, (
            f"bash -n failed on TG block {i} (the #1790 regression class):\n{bn.stderr}"
        )


# --------------------------------------------------------------------------
# Task #1753 — landing-union overlay (generalizes #1456 to every payload
# path), Guard-4 recovery ordering, landing-bytes cap-bump rule
# --------------------------------------------------------------------------


def _overlay_loop_region(text: str) -> str:
    """The payload-overlay loop slice of the pre-push gate: from the
    LANDING-UNION comment to the LINT-VINTAGE block that follows it."""
    gate = _gate_region(text)
    start = gate.index("LANDING-UNION OVERLAY (#1753")
    end = gate.index("LINT-VINTAGE 3-WAY MERGE", start)
    return gate[start:end]


def test_union_overlay_present():
    """#1753: the overlay loop 3-way-merges both-sides-modified payload
    paths (branch HEAD (ours) + merge-base + archived origin/main (theirs))
    so the gated legs certify the LANDING content — an in-loop
    `git merge-file -p` inside the overlay region, ahead of the #1456
    LINT-VINTAGE block (#1721: a branch-tip planner.md passed at 39,371 B
    while the squash union landed 40,900 B > the 40,000 cap)."""
    region = _overlay_loop_region(_skill_text())
    assert "git merge-file -p" in region
    assert "done < /tmp/issue-<N>-overlay-files.txt" in region


def test_workflow_lint_excluded_from_union_loop():
    """#1753: scripts/workflow_lint.py stays EXCLUDED from the union loop —
    the dedicated #1456 block below merges it; a double merge would feed
    the already-merged union back into merge-file as "ours"."""
    region = _overlay_loop_region(_skill_text())
    assert '[ "$p" != "scripts/workflow_lint.py" ]' in region


def test_union_fallback_is_loud():
    """#1753: a conflicted/failed per-path union merge falls back to the
    BRANCH copy with a loud per-path WARN + a counted fallback — never a
    crash (the real merge surfaces the conflict as shape 2)."""
    region = _overlay_loop_region(_skill_text())
    assert "landing-union 3-way merge conflicted/failed" in region
    assert "UNION_FALLBACK=$((UNION_FALLBACK + 1))" in region


def test_union_echo_breadcrumb():
    """#1753: the merged/fallback counters are echoed as a breadcrumb the
    `epm:merged` / `epm:merge-failed` note copies (alongside the lint/tg
    tails those notes already record)."""
    region = _gate_region(_skill_text())
    assert "[step10d] landing-union overlay:" in region


def test_guard4_recovery_ordering_present():
    """#1753 (incident #1727): Guard 4 documents commit-before-re-gate for
    the merge-of-origin/main recovery — staged-but-uncommitted merge
    content reads as dropped under the guard's `git show HEAD:"$P"`
    predicate (a false lost-update / STILL-UNMERGED read)."""
    text = _skill_text()
    start = text.index("**Lost-update refusal (shared workflow-surface files).**")
    end = text.index("#### Fast-path routing pre-check", start)
    region = text[start:end]
    assert "Recovery ordering (#1753" in region
    assert "staged-but-uncommitted" in region


def test_landing_bytes_cap_rule_present():
    """#1753 (incident #1727): the gate section documents that size-ratchet
    cap bumps are computed from landing/union bytes, never branch-tip bytes
    (#1727: cap 130,000 from a pre-merge 128,507 B tip failed post-merge)."""
    region = _gate_region(_skill_text())
    assert "Size-ratchet cap bumps are computed from landing bytes (#1753)" in region


def test_mergefile_block_untouched_order():
    """#1753 ordering pin: the #1456 LINT-VINTAGE 3-WAY MERGE block still
    FOLLOWS the overlay loop's `done` line (complements
    tests/test_issue_skill_lint_gate_mergefile.py's ordering pin, which
    anchors on the #1456-specific `git merge-file -p
    "$GT/scripts/workflow_lint.py"` invocation)."""
    gate = _gate_region(_skill_text())
    done_idx = gate.index("done < /tmp/issue-<N>-overlay-files.txt")
    lint_vintage_idx = gate.index("LINT-VINTAGE 3-WAY MERGE")
    assert done_idx < lint_vintage_idx


def test_guard_greps_carry_end_of_options_separator():
    """#1788 (incidents #1742/#1758): every VARIABLE-pattern full-line grep in
    the Step 10d guards carries the `--` end-of-options separator. Without it
    a main-added line starting with `-` (a markdown bullet — ubiquitous on the
    workflow surface Guard 4 scans) is parsed as grep OPTIONS ("invalid
    option", rc=2), which the loop miscounts as MISSING_ON_BRANCH and Guard 4
    false-fires a LOST-UPDATE refusal on a byte-identical branch."""
    text = _skill_text()
    # (a) The separator-bearing forms are present.
    assert 'grep -Fxq -- "$ADD_LINE"' in text, "Guard-4 membership grep lost its -- separator"
    assert 'grep -Fxq -- "$MB"' in text, (
        "Guard-3 first-parent (MB_FIRST_PARENT) grep lost its -- separator"
    )
    # (b) No separator-less VARIABLE-pattern form remains, in EITHER flag
    # ordering. Literal-pattern greps (e.g. -qxF 'scripts/workflow_lint.py')
    # are exempt by construction: the regex requires the pattern argument to
    # start with `"$` (a shell-variable pattern), which only a variable-
    # pattern site has; the fixed forms carry `-- ` and do not match.
    offenders = [
        f"line {i}: {line.strip()}"
        for i, line in enumerate(text.splitlines(), start=1)
        if re.search(r'grep -(?:Fxq|qxF) "\$', line)
    ]
    assert not offenders, (
        "separator-less variable-pattern grep -Fxq/-qxF site(s) in SKILL.md "
        "(add `--` before the pattern): " + "; ".join(offenders)
    )


# --------------------------------------------------------------------------
# Task #2320 — Guard-3 first-parent false-UNSAFE fix
# (D1 MB_VALID hard-stop probe · D2 unconditional content check ·
#  D3-ter retired-condition templates · D4 stranded-MODIFIED surfacing)
# --------------------------------------------------------------------------


def _guard3_probe_fence(text: str) -> str:
    """The D1 merge-base probe fence — the FIRST bash fence of the Guard-3
    region — dedented for subprocess execution."""
    region = _guard3_region(text)
    fences = re.findall(r"```bash\n(.*?)```", region, re.DOTALL)
    assert fences, "Guard-3 region carries no fenced bash block"
    fence = fences[0]
    assert "GUARD3 HARD-STOP" in fence, "first Guard-3 fence must be the MB probe block"
    return textwrap.dedent(fence)


def _guard3_own_diff_fence(text: str) -> str:
    """The own-commits three-dot content-check fence — the SECOND bash fence
    of the Guard-3 region — dedented for subprocess execution."""
    region = _guard3_region(text)
    fences = re.findall(r"```bash\n(.*?)```", region, re.DOTALL)
    assert len(fences) >= 2, "Guard-3 region must carry the own-diff fence"
    fence = fences[1]
    assert "origin/main...HEAD" in fence, "second Guard-3 fence must be the own-diff"
    return textwrap.dedent(fence)


def _git(repo: Path, *args: str) -> str:
    """Run git in a fixture repo; assert success; return stripped stdout."""
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert proc.returncode == 0, f"git {' '.join(args)} failed:\n{proc.stderr}"
    return proc.stdout.strip()


def _init_guard3_repo(parent: Path) -> Path:
    """A throwaway repo with a `main` branch and hermetic committer identity."""
    repo = parent / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "guard3-fixture@example.invalid")
    _git(repo, "config", "user.name", "guard3-fixture")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _fixture_commit(repo: Path, fname: str, msg: str) -> str:
    """Write `fname`, commit it, return the commit SHA."""
    path = repo / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(msg + "\n", encoding="utf-8")
    _git(repo, "add", fname)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _run_fence(repo: Path, fence: str, trailer: str = "") -> subprocess.CompletedProcess:
    """Drive an extracted SKILL.md fence against a fixture repo (WT bound).

    Assertions are on rc/stdout of the subprocess — the predicate is bash
    prose, not importable Python (the #1253 strong-pin convention above).
    """
    script = f'WT="{repo}"\n{fence}{trailer}'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_skillmd_guard3_probe_shape_pinned():
    """#2320 D6: the Guard-3 region carries the MB_VALID hard-stop, the
    variable-gated terminal status line, the retained first-parent DIAGNOSTIC,
    and the unconditional content-check prose; the retired first-parent
    verdict token is BANNED file-wide — BOTH the variable name AND the
    D3-ter retired-condition template phrase, which carried no such token
    and would otherwise ship untouched (round-2 blocker 4: two unambiguous
    zero-hit checks, no region carve-out)."""
    text = _skill_text()
    region = _guard3_region(text)
    fence = _guard3_probe_fence(text)
    # The hard-stop arm + the variable-gated terminal status line (the
    # documented Must-Fix convention: a `false` inside a non-final arm does
    # NOT set a fenced block's exit status — a variable-gated conditional
    # as the LAST statement is required).
    assert "GUARD3 HARD-STOP: no merge-base between HEAD and origin/main" in fence
    assert "MB_VALID=no" in fence and "MB_VALID=yes" in fence
    assert fence.rstrip().splitlines()[-1].strip() == '[ "$MB_VALID" = yes ]', (
        "the variable-gated terminal status line must be the LAST statement of the probe fence"
    )
    # The retained first-parent DIAGNOSTIC (never a verdict) keeps the exact
    # separator-bearing grep the #1788 pin asserts.
    assert 'MB_FIRST_PARENT=$(git -C "$WT" rev-list --first-parent origin/main' in fence
    assert 'grep -Fxq -- "$MB"' in fence
    # Unconditional content check: no BEHIND-threshold trigger survives.
    assert "runs on EVERY branch" in region, (
        "the content check must be documented as running on EVERY branch"
    )
    assert "TRIGGERS the own-commit" not in region, (
        "the retired BEHIND-threshold trigger sentence must be gone"
    )
    # The diagnostic has a recording carrier on BOTH landing paths
    # (safe case + artifact-confirmed) — never set-and-never-recorded.
    assert text.count("mb_first_parent: <yes|no>") >= 2, (
        "mb_first_parent must ride the epm:merged note on both landing paths"
    )
    # File-wide literal bans (unambiguous zero-hit checks).
    assert "ON_MAINLINE" not in text, (
        "the retired first-parent verdict variable must not survive anywhere in SKILL.md"
    )
    assert "not on mainline" not in text, (
        "the retired-condition template phrase must not survive anywhere in SKILL.md"
    )


def test_empty_merge_base_hard_stops():
    """#2320 D6 fail-loud pin: the extracted D1 probe exits NON-ZERO with the
    GUARD3 HARD-STOP line on (a) unrelated histories (merge-base exits 1,
    empty output) and (b) an UNFETCHED origin/main (the ref is absent, so
    merge-base dies rc=128) — the negative control proving the failure is
    not swallowed into rc=0 — and exits ZERO on a healthy fork, so the
    non-zero rc is attributable to the hard-stop arm."""
    fence = _guard3_probe_fence(_skill_text())
    scratch = Path(tempfile.mkdtemp(prefix="eps-guard3-fixture-"))
    try:
        script = scratch / "guard3_probe.sh"
        script.write_text(fence, encoding="utf-8")
        bn = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert bn.returncode == 0, f"bash -n failed on the extracted probe:\n{bn.stderr}"

        # (a) unrelated histories: origin/main resolves to a DISJOINT root.
        repo_a = _init_guard3_repo(scratch / "a")
        _fixture_commit(repo_a, "f.txt", "main root")
        _git(repo_a, "checkout", "-q", "--orphan", "other")
        _fixture_commit(repo_a, "g.txt", "disjoint root")
        other = _git(repo_a, "rev-parse", "HEAD")
        _git(repo_a, "checkout", "-q", "main")
        _git(repo_a, "update-ref", "refs/remotes/origin/main", other)
        proc = _run_fence(repo_a, fence)
        assert proc.returncode != 0, "no-merge-base must terminate the block NON-ZERO"
        assert "GUARD3 HARD-STOP: no merge-base" in proc.stdout

        # (b) negative control — UNFETCHED origin/main (ref absent entirely).
        repo_b = _init_guard3_repo(scratch / "b")
        _fixture_commit(repo_b, "f.txt", "main root")
        proc = _run_fence(repo_b, fence)
        assert proc.returncode != 0, "an unfetched origin/main must hard-stop, never rc=0"
        assert "GUARD3 HARD-STOP: no merge-base" in proc.stdout

        # Healthy control: an ordinary fork exits ZERO.
        repo_c = _init_guard3_repo(scratch / "c")
        base = _fixture_commit(repo_c, "f.txt", "base")
        _git(repo_c, "update-ref", "refs/remotes/origin/main", base)
        _fixture_commit(repo_c, "own.txt", "own work")
        proc = _run_fence(repo_c, fence)
        assert proc.returncode == 0, (
            f"healthy fork must not hard-stop:\n{proc.stdout}\n{proc.stderr}"
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_second_parent_landing_classifies_safe():
    """#2320 acceptance criterion 1 (the #2319 shape): a branch whose
    merge-base reached main ONLY as a merge commit's SECOND parent (the
    #1489/#1128 scratch-worktree merge-form landing) must NOT hard-stop —
    the probe exits 0, the first-parent DIAGNOSTIC reads no, and the
    three-dot own-diff (the content check's input) carries only the branch's
    own deliverable. Pre-fix, the first-parent verdict flagged exactly this
    shape UNSAFE."""
    text = _skill_text()
    fence = _guard3_probe_fence(text)
    own_fence = _guard3_own_diff_fence(text)
    scratch = Path(tempfile.mkdtemp(prefix="eps-guard3-fixture-"))
    try:
        repo = _init_guard3_repo(scratch / "r")
        c0 = _fixture_commit(repo, "base.txt", "main base")
        _fixture_commit(repo, "main_adv.txt", "main advance")
        _git(repo, "checkout", "-q", "-b", "markers", c0)
        w1 = _fixture_commit(repo, "marker.txt", "fleet marker commit")
        _git(repo, "checkout", "-q", "main")
        _git(repo, "merge", "-q", "--no-ff", "-m", "merge-form landing", "markers")
        landed = _git(repo, "rev-parse", "HEAD")
        _git(repo, "update-ref", "refs/remotes/origin/main", landed)
        _git(repo, "checkout", "-q", "-b", "issue-9999", w1)
        _fixture_commit(repo, "own_deliverable.txt", "own deliverable")

        # Fixture sanity: w1 IS an ancestor of origin/main (second parent only).
        ia = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", w1, landed])
        assert ia.returncode == 0, "fixture: merge-base must be an ancestor of origin/main"

        proc = _run_fence(repo, fence)
        assert proc.returncode == 0, (
            "a second-parent-landed merge-base must NOT hard-stop "
            f"(rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
        probe = _run_fence(
            repo, fence, trailer='\nprintf "MBFP=%s MB=%s\\n" "$MB_FIRST_PARENT" "$MB"'
        )
        assert f"MBFP=no MB={w1}" in probe.stdout, (
            "fixture must reproduce the #2319 shape: first-parent read no on an "
            f"origin/main-ancestor merge-base:\n{probe.stdout}"
        )
        own = _run_fence(repo, own_fence)
        assert own.returncode == 0, f"own-diff fence failed:\n{own.stderr}"
        assert own.stdout.split() == ["own_deliverable.txt"], (
            "the content check's input must carry ONLY the branch's own "
            f"deliverable (SAFE classification): {own.stdout!r}"
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_fork_off_unmerged_sibling_caught_by_content_check():
    """#2320 acceptance criterion 2 (the #479 class): a branch forked off a
    still-unmerged sibling is caught via the CONTENT CHECK — its three-dot
    own-diff carries the sibling's out-of-scope payload. The probe itself
    does NOT flag it: the merge-base is an ordinary mainline commit (the
    retired first-parent arm read yes here — false-NEGATIVE on the very
    class it claimed to catch), so the in-scope/out-of-scope judgment over
    the own-diff is the load-bearing arm."""
    text = _skill_text()
    fence = _guard3_probe_fence(text)
    own_fence = _guard3_own_diff_fence(text)
    scratch = Path(tempfile.mkdtemp(prefix="eps-guard3-fixture-"))
    try:
        repo = _init_guard3_repo(scratch / "r")
        c0 = _fixture_commit(repo, "base.txt", "main base")
        _git(repo, "update-ref", "refs/remotes/origin/main", c0)
        _git(repo, "checkout", "-q", "-b", "issue-9998")
        _fixture_commit(repo, "scripts/issue9998_tool.py", "sibling payload")
        _git(repo, "checkout", "-q", "-b", "issue-9999")
        _fixture_commit(repo, "own_deliverable.txt", "own deliverable")

        proc = _run_fence(repo, fence)
        assert proc.returncode == 0, "the probe alone must not flag the #479 class"
        probe = _run_fence(repo, fence, trailer='\nprintf "MBFP=%s\\n" "$MB_FIRST_PARENT"')
        assert "MBFP=yes" in probe.stdout, (
            "fixture sanity: the fork-off-unmerged-sibling merge-base is an "
            "ordinary mainline commit (the retired first-parent arm read yes)"
        )
        own = _run_fence(repo, own_fence)
        files = own.stdout.split()
        assert "scripts/issue9998_tool.py" in files, (
            "the three-dot own-diff must carry the unmerged sibling's "
            f"out-of-scope payload (the content check's catching input): {files}"
        )
        assert "own_deliverable.txt" in files
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_stranded_modified_scan_producer_guarded():
    """#2320 D4: the artifact-confirmed path's stranded-MODIFIED surfacing
    scan is materialize-then-check (the test_gate_trigger_diff_exit_guarded
    pattern): a FAILED producer diff sets STRANDED_STATUS=unknown — never an
    empty file read as "nothing stranded" — and the scan precedes the
    deliverables decision tree so it covers both sub-branches."""
    region = _artifact_confirmed_region(_skill_text())
    guard = region.find(
        'if ! git -C "$WT" -c core.quotePath=false diff --name-only '
        "--diff-filter=MRD origin/main...HEAD"
    )
    assert guard != -1, (
        "the stranded-MODIFIED scan must check its OWN exit code (materialize-then-check)"
    )
    assert "> /tmp/issue-<N>-stranded-modified.txt" in region
    assert "STRANDED-SCAN FAILED" in region
    assert "STRANDED_STATUS=unknown" in region and "STRANDED_STATUS=ok" in region
    # Placement: BEFORE the deliverables decision tree (covers both arms).
    verify = region.find("# Verify task deliverables resolve on origin/main.")
    assert -1 < guard < verify, "the stranded scan must precede the decision tree"
    # Note-carrier fields for both scan outcomes.
    assert "stranded_modified: [...]" in region
    assert "stranded_modified: UNKNOWN — producer diff failed" in region
