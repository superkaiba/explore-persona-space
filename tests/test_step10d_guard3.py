"""Content-invariant tests for the /issue Step-10d merge hardening (task #787).

Step 10d's merge-strategy / guard-routing logic lives in SKILL.md PROSE
(`.claude/skills/issue/SKILL.md`), not in importable Python — so the pragmatic
way to pin its invariants is a set of content-string assertions over the skill
file plus the new repo-root `.gitattributes`. This mirrors how other workflow
invariants are pinned when the logic lives in a `.md` skill.

The four #787 sub-fixes these tests guard:

1. `.gitattributes` (NEW) — `merge=union` on the append-only task JSONL logs.
2. Guard-1 — strip FOREIGN task folders before the merge, split by whether the
   path exists on `origin/main` (checkout vs `git rm --cached`).
3. Fast-path pre-check — a FIVE-conjunct predicate (incl. `ADDED_ONLY=yes`)
   that routes far-behind small ADDED-only workflow-fix branches straight to
   the surgical additive checkout, plus the surgical compute block's
   ADDED-only / three-dot / workflow-surface-pathspec invariants (A3-new).
4. Guard-3 — the spec-freshness exclusion matches the commit SUBJECT line only
   (`awk 'index($0, "spec-freshness") == 0'`), never a subject+body `--grep`.
"""

from __future__ import annotations

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
    text = _skill_text()
    assert "rm --cached -f --ignore-unmatch" in text, (
        "Guard-1 must drop branch-added foreign paths via git rm --cached -f "
        "--ignore-unmatch (a checkout would crash with pathspec-did-not-match)"
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
# `gh pr merge --rebase` rebases the PR head ref on origin/issue-<N>
# (server-side), so an unpushed strip commit is invisible to that rebase and
# the foreign tasks/* reverts would land on main silently. The safe case must
# push the strip commit (guarded by STRIPPED_FOREIGN=yes) before merging, and
# that push must APPEAR BEFORE the safe-case gh pr merge line in the file.


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
        "gh pr merge --rebase (otherwise the strip commit is invisible to the "
        "server-side rebase)"
    )
    assert '[ "$STRIPPED_FOREIGN" = "yes" ]' in text, (
        "the safe-case push must be gated on STRIPPED_FOREIGN=yes so it fires "
        "only when Guard-1 actually created a strip commit"
    )


def test_safe_case_push_appears_before_gh_pr_merge():
    """The push must SEQUENCE before the safe-case gh pr merge --rebase call,
    so the server-side rebase sees the stripped branch tip."""
    text = _skill_text()
    merge_line = "gh pr merge <PR> --rebase --delete-branch=false"
    push_line = 'git -C "$WT" push origin issue-<N>'
    merge_offset = text.find(merge_line)
    push_offset = text.find(push_line)
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
