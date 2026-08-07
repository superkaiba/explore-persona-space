"""Pin the compliant git-commit forms at the two copy-sources (#1876).

The recurring compose-time failure (piped `git push` / `git commit`
compositions guard-blocked ≥9x/day fleet-wide) is a COMPOSITION gap, not a
rules gap: agents compose from copy sources, and the two copy sources
consulted at failure time lacked the commit form / buried the compliant
one-liner. #1876 makes the compliant redirect-to-file form copy-paste
available at both moments it is needed:

1. The `/issue` SKILL.md Step 10d § "Bare push / merge snippets" section —
   the canonical copy source every composition site points at — names the
   `git commit` verb in its intro and carries a form-(5) commit-with-output
   snippet (redirect to a FILE + rc capture, pathspec-limited per CLAUDE.md
   § Concurrent repo-root committers; never a pipe — #1584/#1591 SIGPIPE).
2. The `guard_piped_git_push.sh` block message LEADS with the two compliant
   one-liners (push + pathspec commit, redirect-to-file + rc echo) BEFORE
   the rationale prose, so the blocked session can copy-paste the fix
   without re-deriving it.

Both tests are REGION-SCOPED (document-global asserts are vacuous or
false-fire: e.g. `Concurrent` also appears in the hook's header comment) —
the SKILL.md test reuses the `_snippets_region()` idiom from
tests/test_issue_skill_stash_kept_duty_pin.py; the hook test extracts the
`<<'BLOCK_MSG'`..`BLOCK_MSG` heredoc span first.

NOTE for future editors: a legitimate rewording of either surface must
update the pinned substrings below IN THE SAME DIFF. Assertions are
whitespace-normalized substring checks on stable invariants (the one-liner
shapes + their ordering), per the pin-family convention.

Paths resolve via ``Path(__file__)`` — NEVER ``task_workflow.repo_root()``,
which reads the MAIN checkout and would miss worktree edits pre-merge.
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
GUARD_HOOK = ROOT / ".claude" / "hooks" / "guard_piped_git_push.sh"

REGION_START = "#### Bare push / merge snippets"
REGION_END = "#### Merge safety guards"

# Section intro enumerates the commit verb alongside push/merge/PR.
INTRO_VERBS = "Every `git push` / `git merge` / `git commit` / `gh pr merge|create`"
# Form (5): commit output redirected to a FILE with rc capture (never a pipe).
COMMIT_FORM_REDIRECT = (
    'git commit -m "<msg>" -- <paths> > /tmp/issue-<N>-commit.log 2>&1; COMMIT_RC=$?'
)
COMMIT_FORM_RC_CHECK = '[ "$COMMIT_RC" -eq 0 ]'

# The hook's compliant one-liners (must LEAD the block message, before the
# `Concurrent` rationale reference; /tmp/push.out matches the CLAUDE.md
# § Concurrent repo-root committers canonical form).
HOOK_PUSH_ONELINER = "git push origin main > /tmp/push.out 2>&1; echo rc=$?"
HOOK_COMMIT_ONELINER = 'git commit -m "<msg>" -- <paths> > /tmp/commit.out 2>&1; echo rc=$?'
HOOK_RATIONALE_ANCHOR = "Concurrent"

HEREDOC_START = "<<'BLOCK_MSG'"
HEREDOC_END = "\nBLOCK_MSG"


def _normalized(text: str) -> str:
    """Collapse all whitespace to single spaces (wrap-tolerant substring checks)."""
    return " ".join(text.split())


def _snippets_region() -> str:
    """The normalized SKILL.md span between the Bare-push-snippets heading and
    the Merge-safety-guards heading (the canonical copy-source subsection)."""
    skill_norm = _normalized(issue_skill_text())
    start = skill_norm.find(REGION_START)
    assert start != -1, (
        f"SKILL.md lost the {REGION_START!r} heading; if the subsection was "
        "renamed, update this pin alongside it."
    )
    end = skill_norm.find(REGION_END, start)
    assert end != -1, (
        f"SKILL.md lost the {REGION_END!r} heading after {REGION_START!r}; "
        "if the subsection was renamed or reordered, update this pin alongside it."
    )
    return skill_norm[start:end]


def _block_msg_region() -> str:
    """The normalized text of the hook's `BLOCK_MSG` stderr heredoc body.

    Extracting the heredoc FIRST keeps the ordering assert honest: the
    `Concurrent` needle also appears in the hook's header comment, so a
    whole-file index() ordering check would be wrong.
    """
    hook_text = GUARD_HOOK.read_text()
    start = hook_text.find(HEREDOC_START)
    assert start != -1, (
        f"guard_piped_git_push.sh lost its quoted {HEREDOC_START!r} heredoc "
        "delimiter (the quoting is load-bearing: the compliant one-liners "
        "contain `$?`, which an unquoted delimiter would expand); if the "
        "block message was restructured, update this pin alongside it."
    )
    end = hook_text.find(HEREDOC_END, start)
    assert end != -1, (
        "guard_piped_git_push.sh's BLOCK_MSG heredoc has no terminating "
        "line-start BLOCK_MSG delimiter after the opener."
    )
    return _normalized(hook_text[start + len(HEREDOC_START) : end])


def test_commit_form_present():
    """The SKILL.md snippets section names `git commit` in its intro and
    carries the form-(5) commit-redirect shape (file redirect + rc capture)."""
    region = _snippets_region()
    assert INTRO_VERBS in region, (
        "The Bare-push-snippets intro must enumerate `git commit` alongside "
        "push/merge/PR (#1876) — the intro is what tells a composer the "
        "commit verb is governed by these forms too."
    )
    assert COMMIT_FORM_REDIRECT in region, (
        "The Bare-push-snippets fenced block must carry the form-(5) "
        "commit-with-output shape: `git commit ... -- <paths>` redirected to "
        "a FILE with rc capture (#1876; a piped hook-running commit is "
        "SIGPIPE-killed mid-pre-commit-hook, #1584/#1591)."
    )
    assert COMMIT_FORM_RC_CHECK in region, (
        "Form (5) must check the captured COMMIT_RC — the rc check is what "
        "makes the redirect form exit-code-honest (#1048 masked-exit class)."
    )


def test_hook_block_message_leads_with_compliant_forms():
    """The hook's BLOCK_MSG heredoc leads with the two compliant one-liners
    (push + pathspec commit) BEFORE the `Concurrent` rationale reference."""
    body = _block_msg_region()
    assert body.startswith("BLOCKED"), (
        "The block message must keep starting with BLOCKED (the hook tests "
        "and every downstream reader key on it)."
    )
    push_idx = body.find(HOOK_PUSH_ONELINER)
    commit_idx = body.find(HOOK_COMMIT_ONELINER)
    rationale_idx = body.find(HOOK_RATIONALE_ANCHOR)
    assert push_idx != -1, (
        "BLOCK_MSG must carry the compliant push one-liner "
        f"{HOOK_PUSH_ONELINER!r} (the CLAUDE.md § Concurrent repo-root "
        "committers canonical form, keyed on /tmp/push.out)."
    )
    assert commit_idx != -1, (
        "BLOCK_MSG must carry the compliant pathspec-commit one-liner "
        f"{HOOK_COMMIT_ONELINER!r} (#1876)."
    )
    assert rationale_idx != -1, (
        "BLOCK_MSG must keep the CLAUDE.md § Concurrent repo-root committers "
        "rationale reference (a pinned needle of "
        "tests/test_guard_piped_git_push.py)."
    )
    assert push_idx < rationale_idx and commit_idx < rationale_idx, (
        "The compliant one-liners must LEAD the block message — both before "
        "the `Concurrent` rationale prose (#1876: the copy-paste fix comes "
        f"first; push@{push_idx}, commit@{commit_idx}, "
        f"rationale@{rationale_idx})."
    )
