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
4. Guard-3 — the spec-freshness exclusion matches the commit SUBJECT line only
   (`awk 'index($0, "spec-freshness") == 0'`), never a subject+body `--grep`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
_GITATTRIBUTES = _REPO_ROOT / ".gitattributes"


def _skill_text() -> str:
    return _SKILL.read_text(encoding="utf-8")


def _guard3_region(text: str) -> str:
    """The Guard-3 slice: from the guard-3 heading to the fast-path heading.

    Scoping the Guard-3 assertions to this slice keeps them independent of the
    unrelated Step-5a spec-freshness sync (line ~1925), which legitimately uses
    a `--grep='spec-freshness' --invert-grep` filter and MUST NOT be touched.
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
    merge_line = "gh pr merge <PR> $MERGE_FORM --delete-branch=false"
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


def test_guard3_spec_freshness_subject_only_awk_filter():
    text = _skill_text()
    assert "awk 'index($0, \"spec-freshness\") == 0'" in text, (
        "Guard-3 spec-freshness exclusion must match on the SUBJECT line only "
        "via awk index(), not a subject+body --grep"
    )


def test_guard3_region_does_not_use_grep_spec_freshness():
    """Within the Guard-3 slice, the exclusion must NOT be a `--grep` (which
    matches the commit BODY too). The Step-5a sync at line ~1925 is a separate
    region and keeps its own legitimate `--grep --invert-grep` filter."""
    region = _guard3_region(_skill_text())
    assert "--grep='spec-freshness'" not in region, (
        "Guard-3 must not use --grep='spec-freshness' (over-matches commit body)"
    )
    assert "%H %s" in region, (
        "Guard-3 must emit '<sha> <subject>' per commit for a subject-scoped filter"
    )


def test_step5a_grep_filter_untouched():
    """Defensive: the unrelated Step-5a spec-freshness sync keeps its filter."""
    text = _skill_text()
    assert "--grep='spec-freshness' --invert-grep" in text, (
        "the Step-5a sync's --grep filter must remain (it is out of #787 scope)"
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
    ready = text.find("gh pr ready <PR>")
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
_MERGE_SUCCESS_IF_SAFE = "if gh pr merge <PR> $MERGE_FORM --delete-branch=false; then"
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
    ready = text.find("gh pr ready <PR>")
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
    start = text.find("#### Post-merge stale-task-folder guard")
    end = text.find("## Resume semantics")
    assert start != -1, "post-merge stale-task-folder guard heading not found"
    assert end != -1 and start < end, "guard region must precede Resume semantics"
    return text[start:end]


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
    assert gate.count("timeout --kill-after=60s 900s") >= 4, (
        "all four shared-block lint legs must carry the 900s wedge bound"
    )
    assert surgical.count("timeout --kill-after=60s 900s") >= 4, (
        "all four surgical-block lint legs must carry the 900s wedge bound"
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

    precheck = fence.find('elif ! grep -qxF "$CANON" /tmp/issue-<N>-postmerge-lstree.txt')
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
    assert '! grep -qxF "$CANON" /tmp/issue-<N>-postmerge-lstree.txt; }; then' in arm, (
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
