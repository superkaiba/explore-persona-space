"""End-to-end tests for the ``scripts/guard_repo_root_branch.sh`` PreToolUse hook.

The guard blocks any git command that would move the SHARED repo-root working
tree off ``main`` OR detach its HEAD, because the repo root is the canonical
commit target for ``scripts/task.py`` and every concurrent VM Claude session
(all assume ``HEAD==main``). A detached repo-root HEAD additionally crashes
``task_workflow``'s main-worktree resolver. As of #897 it ALSO blocks
working-tree REVERTS on the shared root (``git restore``, pathspec /
bare-path / force ``git checkout``, ``git clean -f``, ``git reset --hard``) —
the #841 incident class where a concurrent destructive op silently reverted
another session's uncommitted edits. As of #1128 it ALSO blocks branch
MERGES on the shared root (``git merge <ref>``; ``--abort``/``--quit``
recovery allowed) — the #1090 incident class where a conflicting root merge
stranded conflict markers in the shared tree until aborted. As of #1193 it
ALSO blocks the rebase family on the shared root (``git rebase <ref>`` /
``git cherry-pick <ref>``; ``--abort``/``--quit`` recovery allowed,
``--continue``/``--skip`` fail-closed). As of #1234 it ALSO blocks
``git revert <commit>`` / ``git am <mbox>`` on the shared root
(``--abort``/``--quit`` recovery allowed, ``--continue``/``--skip`` and
``am --show-current-patch`` fail-closed).

These tests drive the script exactly as the harness does: stdin PreToolUse JSON
``{"tool_input": {"command": <cmd>}}`` -> exit 2 (blocked) or exit 0 (allowed).
This mirrors the subprocess-drives-script convention of the sibling
``tests/test_*guard*`` files.

The guard's on-main gate exits 0 when the repo-root HEAD is already off ``main``
(it never traps a user recovering from an already-detached/off-main state), so
the BLOCK-path tests are guarded by ``@on_main``. The ALLOW-path and fail-soft
tests run regardless — the guard must never trap those shapes in either state.
As of #1098 the guard WAIVES, per-clause and under a fail-closed refusal ladder
(pipe/BG-producer position, expansion + here-string syntax, non-/dev/null
output redirects, ssh local-exec options, shared-repo path spellings,
``rg --pre``), clauses whose command word is ``ssh`` (remote execution) or
``grep``/``egrep``/``fgrep``/``rg`` (read-only pattern argument).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / "scripts" / "guard_repo_root_branch.sh"

# The script hardcodes REPO=/home/thomasjiralerspong/explore-persona-space and
# runs ``git -C "$REPO" ...`` for its branch/commit-ish resolution + on-main
# gate — so ref/branch/tag existence and the on-main check are read against that
# canonical checkout, NOT this worktree.
REPO = Path("/home/thomasjiralerspong/explore-persona-space")


def _run(cmd: str) -> int:
    """Feed ``cmd`` to the guard via PreToolUse JSON; return its exit code."""
    payload = json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run([str(SCRIPT)], input=payload, text=True, capture_output=True).returncode


def _run_raw(payload: str) -> int:
    """Feed a raw (possibly malformed) stdin payload; return the exit code."""
    return subprocess.run([str(SCRIPT)], input=payload, text=True, capture_output=True).returncode


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def _on_main() -> bool:
    r = _git("symbolic-ref", "--short", "HEAD")
    return r.returncode == 0 and r.stdout.strip() == "main"


def _repo_head_sha() -> str:
    """Real short SHA of the canonical repo-root HEAD (resolves via ``$REPO``)."""
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


on_main = pytest.mark.skipif(
    not _on_main(),
    reason="guard's on-main gate exits 0 when the repo-root HEAD is off main",
)


@pytest.fixture
def throwaway_branch():
    """A real local branch (in ``$REPO``) so ``refs/heads/<name>`` resolves.

    Created pointing at HEAD (no checkout, no tree change) and deleted on
    teardown, so the branch-switch-regression assertion is hermetic rather than
    depending on a repo-specific branch name existing. The name carries a
    per-run uuid suffix so concurrent test runs against the shared ``$REPO``
    never clobber each other's ref, and teardown tolerates a missing ref (a
    concurrent run already cleaned it up) rather than failing.
    """
    name = f"eps-test-throwaway-guard-796-{uuid.uuid4().hex[:8]}"
    _git("branch", "-f", name, "HEAD")
    try:
        yield name
    finally:
        # Best-effort delete: a missing ref (never created / already gone) must
        # not fail teardown. ``branch -D`` is non-raising here (``_git`` does
        # not check=True), but guard explicitly for clarity.
        _git("branch", "-D", name)


@pytest.fixture
def throwaway_tag():
    """A real local tag (in ``$REPO``) so ``<tag>^{commit}`` resolves.

    Detaching to a tag is a real detach shape; a fabricated non-existent tag
    would fail-soft (exit 0) and never exercise the block path — so the tag must
    genuinely exist. Created pointing at HEAD and deleted on teardown. The name
    carries a per-run uuid suffix (concurrent-run safe) and teardown tolerates a
    missing ref.
    """
    name = f"eps-test-throwaway-tag-796-{uuid.uuid4().hex[:8]}"
    _git("tag", "-f", name, "HEAD")
    try:
        yield name
    finally:
        _git("tag", "-d", name)


# ---------------------------------------------------------------------------
# MUST BLOCK — detach-inducing shapes (the whole point of #796)
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "template",
    [
        "git checkout {sha}",  # bare sha detach
        "git checkout --detach",  # explicit --detach, no ref
        "git checkout --detach {sha}",  # explicit --detach + ref
        "git checkout -f {sha}",  # flag-prefixed detach (-f)
        "git checkout -q {sha}",  # flag-prefixed detach (-q)
        "git checkout -p {sha}",  # flag-prefixed detach (-p)
        "git checkout -m {sha}",  # flag-prefixed detach (-m)
        "git switch --detach {sha}",  # switch --detach
        "git switch -d {sha}",  # switch -d
        "git switch -d main",  # detach AT main (branch-only detector allows `main`)
        "git checkout HEAD~1",  # relative rev
        "git checkout HEAD@{{0}}",  # reflog rev ({{0}} -> {0} after .format)
        "git checkout origin/main",  # remote-tracking ref
    ],
)
def test_detach_shapes_block(template):
    assert _run(template.format(sha=_repo_head_sha())) == 2


@on_main
def test_detach_to_real_tag_blocks(throwaway_tag):
    # A real local tag resolves to a commit-ish -> detach -> block.
    assert _run(f"git checkout {throwaway_tag}") == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — existing branch-switch regression fence
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize("cmd", ["git checkout -b feature/x", "git switch fix/foo"])
def test_branch_creation_and_switch_still_block(cmd):
    # -b/-B branch creation and `switch <non-main>` are blocked without needing
    # a resolvable ref (the detectors fire on the flag / arg shape).
    assert _run(cmd) == 2


@on_main
def test_existing_local_branch_checkout_still_blocks(throwaway_branch):
    # `git checkout <real-local-branch>` (not main) is the original blocked case.
    assert _run(f"git checkout {throwaway_branch}") == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — return-to-main, scoped, and non-git shapes. These must never be
# trapped regardless of the repo-root HEAD state, so they are NOT guarded by
# @on_main.
#
# FLIPPED to MUST-BLOCK by #897 (following the #804 `;`-latch flip precedent):
# the four file-restore rows that used to sit here — `git checkout .`,
# `git checkout -- scripts/eval.py`, `git checkout HEAD -- foo.py`,
# `git checkout -f -- scripts/eval.py` — are working-tree reverts that
# silently discard CONCURRENT sessions' uncommitted edits on the shared root
# (incident #841). They now live in test_worktree_revert_shapes_block below.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git checkout main",  # return to main
        "git -C .claude/worktrees/issue-123 checkout abc1234",  # scoped to a worktree
        "cd /tmp/foo && git checkout abc1234",  # scoped by cd /tmp
        "cd .claude/worktrees/x && git checkout abc1234",  # scoped by cd worktree
        "git status",  # not a checkout/switch
        "git commit -m x",  # not a checkout/switch
    ],
)
def test_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — QUOTED detach refs. Quoted git refs are shell-equivalent to
# unquoted ones, so they must produce the SAME exit code as their unquoted
# counterparts above. The round-1 quote-strip pre-pass erased these before the
# detectors ran (leaking the exact class this guard blocks); the revert scans
# the raw command and strips only a single surrounding quote layer on the
# classified checkout arg. Concern id: quoted-refs-bypass-detach-guard (#796).
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "template",
    [
        'git checkout "{sha}"',  # double-quoted sha -> block (was leaking, exit 0)
        "git checkout '{sha}'",  # single-quoted sha -> block
        'git checkout "HEAD~1"',  # quoted relative rev -> block
        'git checkout "origin/main"',  # quoted remote-tracking ref -> block
    ],
)
def test_quoted_detach_refs_still_block(template):
    assert _run(template.format(sha=_repo_head_sha())) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        'git switch "main"',  # quoted return-to-main -> allow (was false-positive, exit 2)
        'git checkout "main"',  # quoted return-to-main -> allow
    ],
)
def test_quoted_main_still_allows(cmd):
    # These must never be trapped regardless of repo-root HEAD state, so NOT
    # guarded by @on_main. Round-1's quote-strip false-positived `git switch
    # "main"` to exit 2; after the revert the arg-classifier strips the quotes
    # and the `main` allow-arm passes.
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — `switch main<sep>` prefixes. The switch allow-arm anchors `main`
# to the full arg (optional trailing quote, then EOL or a shell delimiter). A
# bare `\bmain\b` word boundary matched before `-` / `/` / `.` (non-word
# chars), so `git switch main-adjacent` / `main/foo` / `main.x` (and quoted
# forms) slipped through the allow-arm and leaked a branch-switch off main.
# Concern id: switch-main-prefix-allowarm-leak (#796 round 3).
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch main-adjacent",  # main-prefix branch -> block
        'git switch "main-adjacent"',  # quoted main-prefix -> block
        "git switch main/foo",  # main/ subpath -> block
        'git switch "main/foo"',  # quoted main/ subpath -> block
        "git switch main.x",  # main. suffix -> block
        "git switch main_x",  # main_ (word char) -> block (never hit the allow-arm)
        "git switch mainline",  # main-substring branch -> block
    ],
)
def test_switch_main_prefix_still_blocks(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — genuine return-to-main, including a trailing shell delimiter or
# chained command after `main`. NOT guarded by @on_main (must never trap in
# either repo-root HEAD state). Concern id: switch-main-prefix-allowarm-leak.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch main",  # bare return-to-main
        'git switch "main"',  # quoted return-to-main
        "git switch 'main'",  # single-quoted return-to-main
        "git switch main;",  # semicolon-terminated -> still return-to-main
        "git switch main | tee log.txt",  # pipe-terminated
        "git switch main && echo done",  # chained command after main
        "git switch -c main",  # create-branch named main (allow-arm tolerates -c)
    ],
)
def test_switch_return_to_main_still_allows(cmd):
    assert _run(cmd) == 0


def test_note_text_git_verb_literal_trips_guard_known_limitation():
    # KNOWN LIMITATION (documented in the script header): the guard scans the
    # RAW command and does NOT strip quoted arguments, so a quoted git-verb
    # literal buried in another command's argument (e.g. a marker note
    # discussing the guard) DOES trip it. This is the deliberate trade-off for
    # correctly parsing quoted git refs (test_quoted_detach_refs_still_block);
    # the round-1 quote-strip that "fixed" this false positive leaked real
    # quoted detach refs. The workaround is `--file` for such note text.
    # Exit 2 pins the known behavior so a future re-attempt at a quote-strip
    # (which would silently re-open the leak) trips this test.
    assert (
        _run(
            'uv run python scripts/task.py post-marker 796 epm:foo --note "test git switch string"'
        )
        == 2
    )


@on_main
def test_nonexistent_ref_fails_soft():
    # An arg that resolves to nothing (typo / non-existent ref) is NOT blocked:
    # rev-parse fails -> guard exits 0 -> git itself errors on the real call.
    assert _run("git checkout nonexistent-typo-ref-xyz-796") == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — compound-command masking (#804 / #796 r3 Codex concern).
# A later safe/scoped clause must NOT mask an earlier dangerous repo-root
# clause. Clause-local parsing classifies each clause independently; the first
# blocking clause wins. Concern id: compound-command-masking-leak.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch feature ; git switch main",  # later switch main masks earlier block
        "git switch feature && git switch main",  # same, && separator
        "git checkout HEAD~1 ; git checkout main",  # greedy-sed picked the last checkout arg
        "git switch feature ; cd .claude/worktrees/x",  # trailing cd worktree scoped earlier switch
    ],
)
def test_compound_masking_still_blocks(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — `||` / `|` do NOT scope a `cd worktree` onto the following git
# clause (#804 / #796 r3 Codex concern, the || case named explicitly).
#   `cd X || git switch f`: || runs git ONLY when cd FAILED -> cwd unchanged
#     (repo root) -> git runs off-worktree. Verified: `cd /nonexistent || pwd`
#     prints the repo-root cwd.
#   `cd X | git switch f`: | isolates cd in a subshell -> git runs in the
#     parent cwd (repo root). Verified: `cd /tmp | pwd` prints the parent cwd.
# The clause-local parser must RESET the `scoped` latch when the separator
# BEFORE a clause is || or |. Concern id: compound-command-masking-leak.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "cd .claude/worktrees/foo || git switch feature",  # || runs git on cd FAILURE (repo root)
        "cd .claude/worktrees/foo | git switch feature",  # | isolates cd (subshell), git in parent
    ],
)
def test_or_pipe_cd_scope_does_not_latch(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — bare `&` (background operator) does NOT let a later allow-arm
# clause mask an earlier dangerous one (#804 round 2). Bash `A & B` runs A in a
# background subshell (its own cwd) AND B in the foreground parent (unchanged
# cwd = repo root); BOTH execute. The `split_and_label` sed pre-pass matches
# `&&` before the single `&`, so a bare `&` becomes a BG separator that RESETS
# the `scoped` latch (like ||/|). The dangerous LHS clause therefore classifies
# on its own and blocks. Concern id: guard-single-ampersand-masking-leak.
#   `git switch feature & git switch main`  -> BG reset; switch-feature blocks.
#   `git checkout HEAD~1 & git checkout main` -> BG reset; HEAD~1 detach fires.
#   `cd .claude/worktrees/foo & git switch feature` -> a BACKGROUND cd runs in
#     its own subshell and does NOT scope the foreground git; switch blocks.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch feature & git switch main",  # BG reset; earlier switch-feature blocks
        "git checkout HEAD~1 & git checkout main",  # BG reset; HEAD~1 detach classifier fires
        "cd .claude/worktrees/foo & git switch feature",  # bg cd does not scope the fg git
    ],
)
def test_bg_ampersand_does_not_mask(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — a bare `&` between two non-dangerous clauses stays allowed: the
# background LHS is not a checkout/switch and the foreground RHS returns to
# main. Guards against the BG reset over-blocking a benign background compound.
# Concern id: guard-single-ampersand-masking-leak.
# ---------------------------------------------------------------------------
def test_bg_ampersand_benign_allows():
    # `git status` (bg) is not a checkout/switch; `git switch main` (fg) hits the
    # allow-arm. Neither clause moves off main, so the compound is allowed.
    assert _run("git status & git switch main") == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — a `;`-preceding `cd` does NOT scope the following git clause
# (#804 round 2, fail-closed). Bash runs the RHS of `cd X ; git ...` regardless
# of the `cd` exit code; a FAILED `cd` (missing target) leaves the cwd unchanged
# (repo root), so the git clause runs off-worktree. The guard cannot prove a
# `;`-preceding `cd` succeeded, so it fails CLOSED: the `scoped` latch RESETS on
# `;` (SEQ), same as ||/|/&. The v2 `cd worktree ; git switch bar` ALLOW row
# (which trusted the `;` cd-scope) is therefore removed and flips to a BLOCK.
# Concern id: guard-cd-scope-latch-when-cd-fails.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "cd .claude/worktrees/foo ; git switch feature",  # existing worktree, ; no longer latches
        "cd .claude/worktrees/missing-nonexistent-xyz ; git switch feature",  # missing: cd fails
        "cd /tmp/missing-nonexistent-xyz ; git checkout HEAD~1",  # missing /tmp: cd fails
    ],
)
def test_semicolon_cd_scope_does_not_latch(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — a `;`-preceding `cd` whose following clause is NOT a
# checkout/switch stays allowed: the git clause classifier does not fire on
# `git status`, so no block. Guards against the `;` fail-closed reset
# over-blocking a benign `cd worktree ; git status`. Concern id:
# guard-cd-scope-latch-when-cd-fails.
# ---------------------------------------------------------------------------
def test_semicolon_cd_scope_benign_git_allows():
    # `git status` is not a checkout/switch, so the clause is skipped regardless
    # of the (now-reset) latch -> allowed.
    assert _run("cd .claude/worktrees/foo ; git status") == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — a raw NEWLINE does NOT scope a `cd` onto the following git clause
# (#804 round 3). Before this fix the sed pre-pass emitted a sentinel only for
# `||`/`&&`/`;`/`|`/`&`; a raw newline produced a record with NO leading
# sentinel, so awk's `sep` inherited the STALE value from the previous line (an
# `AND` after a `&&` clause) and the `cd` scope latch leaked ACROSS the newline
# — `cd <missing> && git status\n git switch feature` returned rc=0. A
# multi-line command runs each line unconditionally (like `;`), and a FAILED
# `cd` on line N leaves line N+1 in the unchanged cwd (repo root), so the guard
# fails CLOSED: the NL sentinel resets the `scoped` latch like `;`. Concern id:
# guard-newline-after-and-scope-leak.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "cd .claude/worktrees/definitely-missing-804\ngit switch feature",  # bare cd\nswitch
        # && then NL: latch must not leak the AND across the newline
        "cd .claude/worktrees/missing-nonexistent-xyz && git status\ngit switch feature",
        # missing /tmp, && then NL: cd fails, HEAD~1 detach fires on the newline clause
        "cd /tmp/missing-nonexistent-xyz && git status\ngit checkout HEAD~1",
        # NL then glued -b: branch creation on the newline clause blocks
        "cd .claude/worktrees/missing-nonexistent-xyz && git status\ngit checkout -bfoo",
    ],
)
def test_newline_after_and_scope_does_not_latch(cmd):
    """Round 3: raw newlines reset cd-scope latch (they act as ; separators)."""
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — a raw newline between two non-dangerous clauses stays allowed:
# the NL sentinel must not over-block a benign multi-line compound. Guards
# against the NL reset trapping a `git switch main` / `git status` that never
# moves off main. NOT guarded by @on_main. Concern id:
# guard-newline-after-and-scope-leak.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch main\ngit status",  # both non-dangerous (return-to-main + status)
        "git status\ngit status",  # neither is a checkout/switch
    ],
)
def test_newline_benign_compounds_allowed(cmd):
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — a clause that itself moves off main blocks regardless of the ||
# or | connector (no cd-scoping involved; the git-switch-feature clause is
# dangerous on its own). Guards against an over-correction that disables ALL
# blocking after a || / |. Concern id: compound-command-masking-leak.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch feature || echo done",  # switch feature clause blocks; || irrelevant
        "git switch feature | tee log.txt",  # switch feature clause blocks; tee not a git-verb
    ],
)
def test_off_main_clause_blocks_under_or_pipe(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — glued short-flag branch creation (#804 / #796 r3 Claude concern).
# `(-b|-B)\b` missed the glued form `-bfoo`. Concern id:
# checkout-glued-shortflag-b-leak.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize("cmd", ["git checkout -bfoo", "git checkout -Bfoo"])
def test_glued_shortflag_branch_creation_still_blocks(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — legitimate compounds the fleet uses, including legitimate || / |
# shapes that DON'T scope a cd onto a git switch. Clause-local parsing must keep
# these passing. NOT guarded by @on_main (must never trap in either repo-root
# HEAD state).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "cd .claude/worktrees/foo && git switch bar",  # cd worktree scopes the later switch (&&)
        "cd /tmp/foo && git checkout abc1234",  # cd /tmp scopes (&&)
        "git -C .claude/worktrees/foo switch bar",  # per-clause -C scope
        "git status ; git -C .claude/worktrees/foo switch bar",  # -C scope on a compound
        "git switch main | tee log.txt",  # pipe-split: switch main allow-arm + non-git tail
        "git switch main && echo done",  # chained after return-to-main
        "git switch main || echo done",  # || chaining return-to-main + non-git recovery
        "git status || git switch main",  # || chaining a no-op status + return-to-main
    ],
)
def test_compound_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — Bash line continuations (`\<CR?><NL>`) are normalized to a single
# space at the TOP of the guard, before any parsing (#804 round 4). Bash strips
# a backslash-newline pre-execution, joining `git \<NL>checkout -bfoo` into the
# single logical command `git checkout -bfoo`; before this fix the raw-scan
# guard's newline splitter fired on the `\<NL>` and saw `git ` and
# `checkout -bfoo` as SEPARATE lines, so the joined `git checkout` invocation
# was never classified and leaked (4 of 5 probes returned rc=0). Concern id:
# guard-backslash-continuation-bypass.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "git \\\ncheckout -bfoo",  # joined -> git checkout -bfoo (branch creation)
        "git checkout \\\nHEAD~1",  # joined -> git checkout HEAD~1 (detach)
        "git checkout \\\n-bfoo",  # joined -> git checkout -bfoo (branch creation)
        "git \\\nswitch feature",  # joined -> git switch feature (off-main switch)
        "git checkout \\\n--detach abc123",  # joined -> git checkout --detach ...
        "git \\\r\nswitch feature",  # CRLF variant -> git switch feature
    ],
)
def test_backslash_newline_continuation_blocks(cmd):
    """Round 4: bash line-continuation (\\<CR?><NL>) is normalized to space before parsing."""
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — the continuation normalization does not over-block legitimate
# commands joined by a `\<NL>`. `git switch \<NL>main` (previously over-blocked
# rc=2 because the newline splitter broke the `switch main` allow-arm) now joins
# to `git switch main` and hits the allow-arm; a non-checkout/switch join stays
# allowed. NOT guarded by @on_main. Concern id:
# guard-backslash-continuation-bypass.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git switch \\\nmain",  # joined -> git switch main (return-to-main allow-arm)
        "git \\\nswitch main",  # joined -> git switch main
        "git \\\nstatus",  # joined -> git status (not a checkout/switch)
    ],
)
def test_backslash_newline_continuation_legitimate_allows(cmd):
    """Round 4: continuation normalization does not over-block legitimate commands."""
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# Fail-soft — malformed / empty / non-JSON stdin exits 0 (never traps).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    ["", "{}", '{"tool_input": {}}', "not json", '{"tool_input": {"command": ""}}'],
)
def test_fail_soft_exit0(payload):
    assert _run_raw(payload) == 0


# ===========================================================================
# #897 — working-tree reverts on the shared repo root
# ===========================================================================
# Incident #841 (2026-07-02): a concurrent session's destructive working-tree
# git op on the shared root reverted the #841 analyzer's uncommitted body.md
# mid-task and deleted untracked pre-registration + figure files. The five
# #897 detectors (restore / checkout-pathspec incl. --pathspec-from-file /
# checkout-force / clean-force / reset-hard) close this class; they use a
# TIGHT `git <verb>` bigram anchor so plain-English "restore"/"clean"/"reset"
# in `-m` messages do NOT trip. The four rows FLIPPED from
# test_allowed_shapes_exit0 appear here (the #804 `;`-latch flip precedent).
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        # -- restore detector -------------------------------------------------
        "git restore .",
        "git restore tasks/running/841/body.md",  # the exact #841 shape
        "git restore --worktree .",
        "git restore --staged --worktree foo.py",  # worktree present -> fail-closed
        "git restore -W foo.py",  # short worktree form fails closed
        "git restore -S foo.py",  # short staged form fails closed
        "git restore --source=main .",
        # -- checkout pathspec detector (incl. the 4 FLIPPED rows) ------------
        "git checkout .",  # FLIPPED from allow (#897)
        "git checkout -- scripts/eval.py",  # FLIPPED from allow (#897)
        "git checkout HEAD -- foo.py",  # FLIPPED from allow (#897)
        "git checkout -f -- scripts/eval.py",  # FLIPPED from allow (#897)
        "git checkout main -- CLAUDE.md",  # explicit ref incl. main
        "git checkout ./src",  # dot-slash positional
        "xargs -a files.txt git checkout issue-42 --",  # pins the SKILL.md -C migration
        # -- checkout --pathspec-from-file (attached + separate-value forms) --
        "git checkout HEAD --pathspec-from-file=/tmp/files",
        "git checkout HEAD --pathspec-from-file /tmp/files",
        # -- checkout force detector (discards dirty edits even AT main) ------
        "git checkout -f main",
        "git checkout --force main",
        # -- bare-pathspec existence probe (the exact #841 op) ----------------
        "git checkout CLAUDE.md",
        "git checkout scripts/eval.py CLAUDE.md",  # first positional is a real path
        # -- clean force detector ---------------------------------------------
        "git clean -f",
        "git clean -fd",
        "git clean -fdx",
        "git clean -ffdx",
        "git clean -xdf",  # f not first in the cluster
        "git clean --force",
        "git clean -e keep.me -fd",  # force flag after a valued flag
        # -- runtime reset --hard detector (#778/#815 class) -------------------
        "git reset --hard",
        "git reset --hard origin/main",
        "git reset -q --hard",
        "git reset origin/main --hard",  # ref BEFORE the flag
        "git reset --hard HEAD~1",
        "git --no-pager reset --hard",  # git-level flag before subcommand
        # -- compound / latch parity with the #804 machinery -------------------
        "git status && git clean -fd",  # AND does not mask a later dangerous clause
        "cd .claude/worktrees/x ; git restore .",  # SEQ resets the latch
        "cd .claude/worktrees/x\ngit clean -fd",  # NL resets the latch
        "git clean -fd & git switch main",  # BG does not mask
        "echo hi; git checkout .",  # later-clause classification
        "git \\\nrestore .",  # backslash-continuation normalization parity
    ],
)
def test_worktree_revert_shapes_block(cmd):
    assert _run(cmd) == 2


@on_main
def test_note_text_restore_literal_trips_guard_known_limitation():
    # KNOWN LIMITATION (documented in the script header, mirror of
    # test_note_text_git_verb_literal_trips_guard_known_limitation): a quoted
    # FULL `git restore .`-class command literal inside another command's
    # argument trips the raw scan. Workaround: `--file <path.md>` for notes,
    # `git commit -F <file>` for commit messages. Exit 2 pins the deliberate
    # trade-off so a future quote-strip re-attempt trips this test.
    assert (
        _run(
            "uv run python scripts/task.py post-marker 897 epm:x "
            '--note "run git restore . to revert"'
        )
        == 2
    )


@on_main
def test_mangled_comment_separator_split_fails_closed():
    # A comment CONTAINING a separator is mis-split by the clause splitter:
    # the tail clause (`git clean -fd`) classifies on its own and BLOCKS even
    # though bash would treat the whole line as a comment. Fail-closed is the
    # documented safe direction for the #897 comment-clause skip (§3a(iii)).
    assert _run("# note; git clean -fd") == 2


# ---------------------------------------------------------------------------
# #897 round 2 — comment-tail waiver spoof (concern id
# comment-tail-waiver-spoof, the round-1 BLOCKER). Bash never executes an
# unquoted `#` comment tail, but the raw scan previously READ it — so a
# trailing comment carrying a waiver token (`git -C ...`, `--staged`) or a
# latch shape (`cd .claude/worktrees/...`) made the hook exit 0 while bash
# executed the destructive head. The driver loop now strips the
# whitespace-anchored ` #` tail of each clause before any latch / waiver /
# gate / classification read; these rows pin the spoof shapes CLOSED.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        "git restore . # git -C /tmp status",  # comment `-C` cannot waive
        "git restore . # --staged",  # comment `--staged` cannot flip the allow-arm
        "git checkout . # git -C /tmp status",
        "git clean -fd # git -C /tmp status",
        "git reset --hard # git -C /tmp status",
        # control: a plain comment tail (no waiver token) also blocks — the
        # spoof required a waiver token in the tail, the strip removes both
        "git restore . # plain comment no waiver tokens",
        # comment-tail LATCH spoof: the `cd .claude/worktrees/x` lives in the
        # comment tail of clause 1, so it must NOT latch scope across the &&
        "echo hi # cd .claude/worktrees/x && git restore .",
    ],
)
def test_comment_tail_cannot_spoof_waiver_or_allow_arm(cmd):
    assert _run(cmd) == 2


@on_main
def test_commit_message_checkout_realpath_prose_blocks_known_limitation():
    # KNOWN LIMITATION (documented in the script header, round-2 concern id
    # header-tight-anchor-claim-overbroad): the bare-pathspec existence probe
    # rides the LEGACY loose `\bgit\b[^;&|]*\bcheckout\b` anchor, so prose
    # containing `checkout <path>` where `<path>` names a REAL repo file trips
    # WITHOUT a `git checkout` bigram — this `-m` message blocks (fails
    # CLOSED). Remediation: `git commit -F <file>` / `--file <path.md>`.
    assert _run('git commit -m "fix checkout CLAUDE.md handling"') == 2


def test_abs_path_bare_pathspec_residual_gap_allows():
    # RESIDUAL GAP (vii), documented in the script header (round-2 concern id
    # abs-path-bare-pathspec-residual-unnamed): an ABSOLUTE-path bare pathspec
    # inside the repo evades the existence probe (`cat-file -e "HEAD:/abs"`
    # fails; `[ -e "$REPO/$arg" ]` concatenates the repo prefix onto the
    # already-absolute path) so the revert is ALLOWED (fail-open) while git
    # would revert the file. Pinned as CURRENT behavior, deliberately NOT
    # closed this round; closable post-v1 by stripping a `$REPO/` prefix from
    # the arg before probing. NOT @on_main: the allow must hold in either
    # repo-root HEAD state.
    assert _run("git checkout /home/thomasjiralerspong/explore-persona-space/CLAUDE.md") == 0


# ---------------------------------------------------------------------------
# MUST ALLOW — #897 allow-side: index-only restore, dry-run clean, the safe
# stash alternative, per-clause `-C` waivers, cd-latches, tight-anchor
# non-matches, comment clauses, and unresolvable bare args. NOT guarded by
# @on_main (must never trap in either repo-root HEAD state).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git restore --staged foo.py",  # index-only
        "git restore --staged .",  # index-only, dot pathspec
        "git stash",  # the blessed safe alternative
        "git stash pop",
        "git clean -n",  # dry-run, no force
        "git clean -nd",
        "git -C .claude/worktrees/issue-1 clean -fdx",  # per-clause -C waiver
        # deliberate -C override (even pointing at the repo root — the
        # designed, auditable escape; the block message steers away from it)
        "git -C /home/thomasjiralerspong/explore-persona-space checkout -- tasks/x",
        'git -C "$WT" reset --hard origin/main',  # the documented recovery path
        'git -C "$WT" restore .',  # -C waiver symmetry for the restore detector
        "cd .claude/worktrees/x && git restore .",  # && latch
        "cd /tmp/scratch && git clean -fd",  # /tmp latch
        'git commit -m "restore the defaults"',  # tight anchor: no `git restore` bigram
        'git commit -m "clean up the sweep"',  # tight anchor
        'git commit -m "reset --hard the docs"',  # tight-anchor reset symmetry
        "git add docs/notes-clean.md",  # `clean` inside a filename, no force flag
        "uv run python scripts/clean_experiment_downloads.py 897 --apply",  # no `git` word
        "git switch main",  # unchanged allow-arm
        # -- #897 round 2: comment-tail strip allow-side parity ---------------
        "git clean -n # -f",  # comment `-f` cannot force-block a dry run
        "git -C .claude/worktrees/issue-1 clean -fdx # cleanup pass",  # waiver + comment
        "git switch main # back to main",  # comment tail no longer breaks the allow-arm
        "# git checkout .",  # comment-clause skip
        # the SKILL.md Step 10d fence shape: cd + a comment SPELLING a gated
        # form + a benign git command — the comment clause is skipped
        'cd "$REPO_ROOT"\n# `git checkout issue-9 -- <path>` below\ngit status',
    ],
)
def test_worktree_revert_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


@on_main
def test_bare_arg_resolving_to_nothing_keeps_status_quo_allow():
    # The #897 bare-pathspec existence probe fires ONLY when the positional
    # names a real branch / commit-ish / tracked-or-existing path. An arg that
    # resolves to NOTHING keeps the status-quo allow (git itself errors on the
    # real call), pinning the probe's no-new-false-positive claim.
    assert _run("git checkout nonexistent-name-xyz-897") == 0


# ===========================================================================
# #1058 — heredoc-body strip pre-pass + `$WT` cd-latch
# ===========================================================================
# Two recurring false-positive classes closed by #1058: (1) a BARE,
# unconditionally-executed clause-initial `WT=<worktree>` assignment followed
# by `cd "$WT" && git ...` provably targets a worktree, so it latches the SAME
# `scoped` machinery as the literal-path cd-latch; (2) a heredoc BODY fed to a
# non-shell consumer is stdin DATA, so document text mentioning a gated form
# is stripped before parsing — EXCEPT when bash would actually execute parts
# of it (an unquoted-tag body carrying `$(`/backtick/`${` expansion syntax, or
# a body that shells out), which stays blocked. Each pytest case id names its
# validated probe id from the #1058 plan batteries (M*/C*/I*/R*/F*).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MUST ALLOW — `$WT` cd-latch: a BARE clause-initial worktree assignment
# preceded by START / `;` (SEQ) / a raw newline (NL) arms the latch; the
# following `cd "$WT"` (exact-arg, no `..`) scopes the `&&`-chained git
# clause. Covers the recorded incident shapes (spec-sync recipes). NOT guarded
# by @on_main (allows must hold in either repo-root HEAD state).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "REPO_ROOT=/home/thomasjiralerspong/explore-persona-space\n"
            'WT="$REPO_ROOT/.claude/worktrees/issue-1058-fix"\n'
            'cd "$WT" && git checkout main -- .claude/skills/issue/SKILL.md '
            ".claude/agents/planner.md",
            id="M3h_I1-incident_spec_sync_nl",
        ),
        pytest.param(
            "REPO_ROOT=/home/thomasjiralerspong/explore-persona-space; "
            'WT="$REPO_ROOT/.claude/worktrees/issue-1058"; '
            'cd "$WT" && git checkout main -- .claude/agents/planner.md',
            id="M3g-incident_seq_sep",
        ),
        pytest.param(
            'WT=.claude/worktrees/issue-9; cd "$WT" && git checkout main -- specs.md',
            id="M3i-start_sep",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-7"; cd "${WT}" && git checkout main -- CLAUDE.md',
            id="I2-braced_wt",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-7"; cd "$WT/scripts" && git checkout main -- task.py',
            id="I3-subdir_wt_scripts",
        ),
        pytest.param(
            "WT='.claude/worktrees/issue-779'\n"
            'cd "$WT" && git checkout main -- .claude/agents/planner.md',
            id="F1-single_quoted_rhs",
        ),
        pytest.param(
            'export WT=".claude/worktrees/issue-9"\ncd "$WT" && git checkout main -- specs.md',
            id="R11-export_form",
        ),
        pytest.param(
            "WT=/tmp/other; WT=.claude/worktrees/issue-9; "
            'cd "$WT" && git checkout main -- specs.md',
            id="M3f-rearm_after_disarm",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-9"; cd "$WT" && uv run pytest -q',
            id="R17-non_git_under_latch",
        ),
    ],
)
def test_wt_variable_cd_latch_allows(cmd):
    """A bare, unconditional worktree WT= assignment + cd \"$WT\" latches scope."""
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — `$WT` cd-latch fail-closed set. The latch arms ONLY on a BARE
# (end-anchored) clause-initial worktree assignment whose preceding separator
# is START / SEQ / NL: an assignment PREFIX (`WT=<wt> true`) is a per-command
# temp env that does not persist; an AND/OR-preceded assignment is
# runtime-conditional (when skipped, `cd "$WT"` is a `cd ""` repo-root
# no-op); a PIPE-preceded assignment dies with its subshell; BG stays
# non-arming as documented conservatism. Any other clause-initial `WT=`
# DISARMS; the latch inherits the `&&`-only propagation + reset semantics of
# the literal latch; `..` never latches; the variable name is tight to `WT`.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            'WT=.claude/worktrees/x true; cd "$WT" && git restore .',
            id="M2a-prefix_assignment_true",
        ),
        pytest.param(
            "WT=.claude/worktrees/x git restore .",
            id="M2b-prefix_assignment_env_git",
        ),
        pytest.param(
            'WT=.claude/worktrees/x echo hi\ncd "$WT" && git reset --hard',
            id="M2c-prefix_assignment_echo",
        ),
        pytest.param(
            '[ -d "$X" ] && WT=.claude/worktrees/missing; cd "$WT" && git reset --hard',
            id="M3a-and_preceded_assignment",
        ),
        pytest.param(
            'false || WT=.claude/worktrees/missing; cd "$WT" && git restore .',
            id="M3b-or_preceded_assignment",
        ),
        pytest.param(
            'true | WT=.claude/worktrees/x\ncd "$WT" && git restore .',
            id="M3c-pipe_preceded_assignment",
        ),
        pytest.param(
            'sleep 1 & WT=.claude/worktrees/x; cd "$WT" && git restore .',
            id="M3j-bg_preceded_assignment_conservative",
        ),
        pytest.param(
            'WT=.claude/worktrees/a; WT=/somewhere/else; cd "$WT" && git reset --hard',
            id="M3d-nonworktree_reassign_disarms",
        ),
        pytest.param(
            'WT=.claude/worktrees/a; [ -d x ] && WT=/other; cd "$WT" && git restore .',
            id="M3e-conditional_reassign_disarms",
        ),
        pytest.param(
            'cd "$WT" && git checkout main -- .claude/agents/planner.md',
            id="B10-no_assignment_in_call",
        ),
        pytest.param(
            'WT="/home/thomasjiralerspong/explore-persona-space"\n'
            'cd "$WT" && git checkout main -- .claude/agents/planner.md',
            id="B11-nonworktree_assignment",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-9"; cd "$WT/../.." && git restore .',
            id="R13-dotdot_escape",
        ),
        pytest.param(
            'WT2=".claude/worktrees/issue-9"\ncd "$WT2" && git checkout main -- specs.md',
            id="R12-wt2_name",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-779"\n'
            'cd -P "$WT" && git checkout main -- .claude/agents/planner.md',
            id="F4-cd_dash_p",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-9"; cd "$WT"; git restore .',
            id="R14-latch_not_across_semicolon",
        ),
        pytest.param(
            'WT=".claude/worktrees/issue-9"; cd "$WT" || git restore .',
            id="R15-latch_not_across_or",
        ),
        pytest.param(
            'echo "docs: WT=.claude/worktrees/example" > /dev/null\n'
            'cd "$WT" && git checkout -b feature/x',
            id="R10_F5-prose_buried_wt",
        ),
        pytest.param(
            'cd "$WT" && git restore .\nWT=".claude/worktrees/issue-9"',
            id="R16-assignment_after_cd",
        ),
    ],
)
def test_wt_variable_cd_latch_fail_closed_blocks(cmd):
    """Unproven / conditional / disarmed WT assignments never latch scope."""
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — heredoc BODIES destined for non-shell consumers are stripped
# before parsing when provably inert, so document text that merely MENTIONS a
# gated form no longer false-blocks (incident 4 + the adjacent cat-note
# shape). Quoted/escaped-tag bodies stay strippable even with `$(git ...)`
# text (bash suppresses expansion); unquoted-tag bodies strip only when they
# carry NO expansion syntax. NOT guarded by @on_main.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "uv run python - <<'PY'\n"
            "text = 'git checkout main -- .claude/agents/planner.md'\n"
            "print(text)\n"
            "PY",
            id="I4-python_quoted_tag_incident4",
        ),
        pytest.param(
            'uv run python - <<PY\nprint("the guard blocked: git checkout main -- specs")\nPY',
            id="M1i-python_unquoted_tag_plain_prose",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\n"
            "Recovery used: git checkout main -- .claude/skills/issue/SKILL.md\n"
            "EOF",
            id="M1h-cat_note_unquoted_no_expansion",
        ),
        pytest.param(
            "cat > /tmp/note.md <<'EOF'\n$(git checkout -b evil)\nEOF",
            id="M1e-quoted_tag_dollarparen_inert_control",
        ),
        pytest.param(
            'cat > /tmp/note.md <<"EOF"\n`git checkout -b evil`\nEOF',
            id="M1f-dquoted_tag_backtick_inert_control",
        ),
        pytest.param(
            "cat > /tmp/note.md <<\\EOF\n$(git checkout -b evil)\nEOF",
            id="M1g-escaped_tag_dollarparen_inert_control",
        ),
        pytest.param(
            "cat > /tmp/note.md <<-EOF\n\tmentions git checkout main -- x\n\tEOF",
            id="R9-dash_tab_terminator",
        ),
        pytest.param(
            "uv run python - <<PY\n"
            'print("see scripts/guard_repo_root_branch.sh and git checkout main -- x")\n'
            "PY",
            id="C14-body_prose_sh_path_tail",
        ),
        pytest.param(
            'jq \'.\' <<JSON\n{"cmd": "git checkout main -- x"}\nJSON',
            id="C15-jq_quoted_dot_filter",
        ),
        pytest.param(
            "tee /tmp/note.md <<EOF\nworkaround for: git reset --hard origin/main\nEOF",
            id="E7-tee_consumer",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\nplain note, nothing gated\nEOF",
            id="R18-benign_no_git_text",
        ),
        pytest.param(
            'grep -q x <<<"restore mentioned in prose" || true',
            id="R8a-here_string_non_bigram",
        ),
    ],
)
def test_nonshell_heredoc_body_mention_allows(cmd):
    """Provably-inert heredoc bodies are stripped; gated MENTIONS no longer block."""
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — unquoted-tag heredoc bodies carrying expansion syntax. Bash
# performs command/parameter substitution on an UNQUOTED-tag body at feed
# time, so `$(git ...)` / backticks in it EXECUTE regardless of the consumer;
# the strip refuses such bodies (`${` refuses too — parameter expansion can
# nest command substitution, a documented fail-closed over-match on plain
# `${VAR}` references). The MUST-FIX-1 matrix.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "cat > /tmp/note.md <<EOF\n$(git checkout -b evil)\nEOF",
            id="M1a-dollar_paren_body",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\n`git checkout -b evil`\nEOF",
            id="M1b-backtick_body",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\n${Z:-$(git checkout -b evil)}\nEOF",
            id="M1c-nested_param_cmdsub",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\nvalue is ${SOMEVAR}\nhow to revert: git restore .\nEOF",
            id="M1d-bare_param_expansion_plus_gated_prose",
        ),
    ],
)
def test_unquoted_tag_expansion_body_blocks(cmd):
    """Unquoted-tag bodies with expansion syntax never strip (bash executes them)."""
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — heredoc shapes the strip must REFUSE: every shell-consumer /
# command-runner denylist word pinned individually (incl. the standalone-dot
# source form), pipe-to-shell, `.sh`-redirects, unterminated / multi-opener /
# mixed heredocs, shift-like `<<` openers, the continuation-join edge, gated
# text on the opener line or after the terminator, and the here-string
# full-literal raw-scan parity pin.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("bash <<EOF\ngit checkout -b evil\nEOF", id="C1-bash"),
        pytest.param("sh <<EOF\ngit restore .\nEOF", id="C2-sh"),
        pytest.param("zsh <<EOF\ngit reset --hard\nEOF", id="C3-zsh"),
        pytest.param("ksh <<EOF\ngit checkout -b evil\nEOF", id="C4-ksh"),
        pytest.param("dash <<EOF\ngit clean -fd\nEOF", id="C5-dash"),
        pytest.param('eval "$(cat)" <<EOF\ngit checkout -b evil\nEOF', id="C6-eval"),
        pytest.param("source /dev/stdin <<EOF\ngit restore .\nEOF", id="C7-source"),
        pytest.param("ssh host bash <<EOF\ngit reset --hard\nEOF", id="C8-ssh"),
        pytest.param("xargs -I{} {} <<EOF\ngit checkout -b evil\nEOF", id="C9-xargs"),
        pytest.param("parallel <<EOF\ngit restore .\nEOF", id="C10-parallel"),
        pytest.param("sudo -s <<EOF\ngit reset --hard\nEOF", id="C11-sudo"),
        pytest.param("su -c bash <<EOF\ngit checkout -b evil\nEOF", id="C12-su"),
        pytest.param(". /dev/stdin <<EOF\ngit restore .\nEOF", id="C13-dot_source"),
        pytest.param("cat <<EOF | bash\ngit checkout -b evil\nEOF", id="R1-pipe_to_bash"),
        pytest.param("cat > /tmp/run.sh <<EOF\ngit reset --hard\nEOF", id="R2-sh_redirect"),
        pytest.param("cat > /tmp/x.md <<EOF\ngit restore .", id="R3-unterminated"),
        pytest.param(
            "paste <(cat <<A) <(cat <<B)\ngit checkout -b evil\nA\nx\nB",
            id="R4-two_openers_one_line",
        ),
        pytest.param(
            'uv run python - <<PY\nprint("hi")\nPY\nbash <<EOF\ngit checkout -b evil\nEOF',
            id="C16-mixed_python_plus_bash",
        ),
        pytest.param(
            'uv run python -c "x = 1 << 2"\ngit checkout -b evil',
            id="C17-shift_like_opener_then_gated",
        ),
        pytest.param(
            "cat > /tmp/x.md <<EOF\ngit restore . \\\nEOF",
            id="C18-continuation_join_hides_terminator",
        ),
        pytest.param(
            'git checkout -b x && python - <<PY\nprint("hi")\nPY',
            id="R6-gated_on_opener_line",
        ),
        pytest.param(
            'uv run python - <<PY\nprint("hi")\nPY\ngit checkout -b evil',
            id="R7-gated_after_terminator",
        ),
        pytest.param(
            'grep -q x <<<"git restore . mentioned in prose" || true',
            id="R8b-here_string_full_literal_parity",
        ),
    ],
)
def test_shell_consumer_heredoc_still_blocks(cmd):
    """Shell-consumer / unstrippable heredoc shapes keep current (block) behavior."""
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — heredoc bodies that themselves SHELL OUT (os.system /
# subprocess / Popen / bare-or-spaced `system (` / `from os import`) may
# execute git despite a non-shell consumer, so the strip refuses them and the
# gated literal in the body still classifies. The MUST-FIX-4 matrix.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            'uv run python - <<PY\nimport os\nos.system("git checkout -b evil")\nPY',
            id="R5-os_system",
        ),
        pytest.param(
            'uv run python - <<PY\nfrom os import system\nsystem("git restore .")\nPY',
            id="M4a-from_os_import_system",
        ),
        pytest.param(
            'uv run python - <<PY\nimport os\ns = os.system\ns ("git checkout -b evil")\nPY',
            id="M4b-spaced_system_call",
        ),
        pytest.param(
            "uv run python - <<PY\n"
            "import subprocess\n"
            'subprocess.run("git restore .", shell=True)\n'
            "PY",
            id="M4c-subprocess_run_string",
        ),
        pytest.param(
            "uv run python - <<PY\n"
            "from subprocess import Popen\n"
            'Popen(["bash","-c","git reset --hard"])\n'
            "PY",
            id="M4d-popen_list",
        ),
    ],
)
def test_heredoc_shellout_body_blocks(cmd):
    """A body naming a shell-out spelling never strips; its gated text classifies."""
    assert _run(cmd) == 2


# ==== #1098 — ssh remote-command / grep-family pattern-argument clause waiver ====
#
# A clause whose COMMAND WORD is ssh (remote execution — the command string
# runs on the pod's own /workspace clone, never this VM's repo root) or
# grep/egrep/fgrep/rg (the pattern argument is data) is waived per-clause,
# IFF the fail-closed refusal ladder passes: not in pipe/BG-producer position
# (nextsep lookahead), no locally-executing expansion / here-string syntax,
# no `>`/`>>` output redirect to anything but /dev/null (the round-2 cond
# (3b) arm — closes the same-call redirect-to-file->execute channel), no ssh
# local-exec option (ProxyCommand/LocalCommand/KnownHostsCommand), no
# shared-repo path spelling (literal / $HOME/ / ~/), no rg --pre. The allow
# side is NOT @on_main (a waived clause must pass in either repo state); the
# block side pins EVERY refusal-regex alternation arm individually.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("ssh pod-779 'git reset --hard origin/main'", id="S1-incident_pod779_reset"),
        pytest.param('ssh pod-779 "git reset --hard origin/main"', id="S2-double_quoted"),
        pytest.param("ssh pod-779 git reset --hard origin/main", id="S3-unquoted_remote"),
        pytest.param(
            "ssh -p 40052 root@157.157.221.29 'git checkout HEAD -- foo.py'",
            id="S4-gotchas_diverged_pod_recovery",
        ),
        pytest.param("ssh pod-779 'git clean -fd'", id="S5-remote_clean"),
        pytest.param("ssh pod-779 'git checkout -b scratch'", id="S6-remote_branch_creation"),
        pytest.param("ssh pod-779 'git restore .'", id="S7-remote_restore"),
        pytest.param("ssh pod-779 'git reset --hard' && echo done", id="S8-compound_benign_tail"),
        pytest.param(
            "ssh pod-779 'git reset --hard && git status'",
            id="S9-gated_first_statement_mis_split",
        ),
        pytest.param('ssh pod-779 "git reset --hard $BRANCH"', id="S10-bare_dollar_var"),
        pytest.param("echo starting; ssh pod-779 'git reset --hard'", id="S11-ssh_after_seq"),
        pytest.param(
            "ssh pod-779 'git -C /workspace/explore-persona-space reset --hard origin/main'",
            id="S12-remote_dash_c_regression_pin",
        ),
        pytest.param(
            # /dev/null-target redirects are exempt from the cond (3b) redirect
            # refusal (a discard-only sink can never be re-read or executed).
            "ssh pod-779 'git reset --hard origin/main' 2>/dev/null",
            id="S13-stderr_to_dev_null_exempt",
        ),
        pytest.param(
            # SPACED spelling of the /dev/null exemption (`2> /dev/null`) —
            # the strip regex's [[:space:]]* between `>` and the target
            # covers it; pins the exemption beyond the glued S13 spelling.
            "ssh pod-779 'git reset --hard origin/main' 2> /dev/null",
            id="S14-spaced_dev_null_exempt",
        ),
    ],
)
def test_ssh_remote_git_clause_waiver_allows(cmd):
    """Single-statement ssh remote git ops are waived (the 2026-07-06 incident class)."""
    assert _run(cmd) == 0


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "grep -qE 'git reset --hard' scripts/guard_repo_root_branch.sh",
            id="G1-grep_pattern_repo_file",
        ),
        pytest.param("rg 'git clean -fd' .claude/rules/", id="G2-rg_pattern"),
        pytest.param("grep -q 'git switch feature' notes.md", id="G3-grep_switch_pattern"),
        pytest.param("egrep 'git switch feature' notes.md", id="G4-egrep"),
        pytest.param("fgrep 'git restore .' notes.md", id="G5-fgrep"),
        pytest.param("echo x | grep 'git reset --hard'", id="G6-pipe_consumer_grep"),
        pytest.param(
            # The grep-sweep convenience with a /dev/null-target redirect stays
            # waived (cond (3b) strips exact-/dev/null targets before scanning).
            "grep -rn 'git reset --hard' scripts/ 2>/dev/null",
            id="G7-sweep_stderr_to_dev_null_exempt",
        ),
        pytest.param(
            # The /dev/null exemption composes with a following same-call clause:
            # the strip leaves no `>`, so a benign AND consumer does not refuse.
            "grep -q 'git clean -fd' notes.md 2>/dev/null && echo found",
            id="G8-dev_null_exempt_with_and_consumer",
        ),
        pytest.param(
            # APPEND spelling (`2>> /dev/null`) — the strip regex's `>>?`
            # covers the double-arrow form; pins it against regression.
            "grep -rn 'git reset --hard' scripts/ 2>> /dev/null",
            id="G9-append_dev_null_exempt",
        ),
        pytest.param(
            # BARE stdout spelling (`> /dev/null`, no fd digit) — the strip
            # regex's [0-9]* matches empty; pins it against regression.
            "grep -c 'git clean -fd' notes.md > /dev/null",
            id="G10-bare_stdout_dev_null_exempt",
        ),
    ],
)
def test_grep_pattern_clause_waiver_allows(cmd):
    """grep-family pattern arguments are data; the clause is waived (incl. pipe-CONSUMER)."""
    assert _run(cmd) == 0


@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "ssh pod-779 'cd /workspace/explore-persona-space && git reset --hard origin/main'",
            id="N1-multi_statement_mis_split_residual_fp",
        ),
        pytest.param('ssh host "$(git reset --hard)"', id="N2-cmdsub_executes_locally"),
        pytest.param('ssh host "`git clean -fd`"', id="N3-backtick_executes_locally"),
        pytest.param(
            'ssh pod-779 "git reset --hard ${REF}"',
            id="N4-brace_expansion_fail_closed_overmatch",
        ),
        pytest.param(
            "ssh vm 'git --git-dir=/home/thomasjiralerspong/explore-persona-space/.git"
            " reset --hard'",
            id="N5-repo_root_path_in_remote_string",
        ),
        pytest.param(
            "git -c core.sshCommand=ssh reset --hard", id="N6-mid_clause_ssh_word_local_op"
        ),
        pytest.param(
            "ssh -o ProxyCommand='git reset --hard' host", id="N7-proxycommand_local_exec"
        ),
        pytest.param('grep -q "$(git clean -fd)" file', id="N8-grep_cmdsub"),
        pytest.param("grep -f <(git clean -fd) x", id="N9-grep_procsub_in"),
        pytest.param("rg --pre 'git reset --hard' pat file", id="N10-rg_pre_local_exec"),
        pytest.param("ssh pod-779 'git status'; git reset --hard", id="N11-no_latch_local_tail"),
        pytest.param("timeout 240 ssh pod-779 'git reset --hard'", id="N12-wrapped_ssh_not_waived"),
        pytest.param(
            "ssh -o PermitLocalCommand=yes -o LocalCommand='git reset --hard' host",
            id="N13-localcommand_local_exec",
        ),
        pytest.param("grep -q x >(git clean -fd)", id="N14-procsub_out"),
        pytest.param("rg --pre=sh 'git reset --hard' pat file", id="N15-rg_pre_equals"),
        pytest.param(
            "ssh -o KnownHostsCommand='git reset --hard' host 'git status'",
            id="N16-knownhostscommand_local_exec",
        ),
        pytest.param("ssh pod-779 'git reset --hard' <<< input", id="N17-ssh_here_string"),
        pytest.param(
            "ssh host 'echo git reset --hard' | bash", id="N18-pipe_producer_ssh_to_shell"
        ),
        pytest.param(
            "grep 'git reset --hard' recovery.txt | bash",
            id="N19-pipe_producer_grep_to_shell",
        ),
        pytest.param(
            "ssh vm 'git --work-tree=$HOME/explore-persona-space"
            " --git-dir=$HOME/explore-persona-space/.git reset --hard'",
            id="N20-home_relative_work_tree",
        ),
        pytest.param(
            "ssh vm 'git --git-dir=~/explore-persona-space/.git reset --hard'",
            id="N21-tilde_git_dir",
        ),
        pytest.param(
            # The fd-dup's single & mis-splits as a BG separator, hiding the
            # following PIPE from the lookahead — the BG refusal arm covers it
            # (implementation-round fail-closed widening; see waiver cond (2)).
            "ssh pod-779 'git reset --hard' 2>&1 | tail -5",
            id="N22-pipe_producer_benign_consumer_residual_fp",
        ),
        pytest.param(
            # Bare-pipe form: pins that the PIPE refusal is consumer-INDEPENDENT
            # (a benign `tail` consumer still refuses the waiver; gap (xiv)).
            "ssh pod-779 'git reset --hard' | tail -5",
            id="N23-bare_pipe_producer_benign_consumer",
        ),
        # N24-N29 pin the round-2 cond (3b) redirect refusal (concern id
        # redirect-file-producer-failopen): a waived producer redirecting to a
        # local FILE was rc=0 after round 1 (nextsep=SEQ/AND/NL/END never fires
        # cond (2)) while rc=2 on main — the same-call write-then-execute
        # sibling of the PIPE hole N18/N19/N23 close. All six probed rc=2 on
        # main pre-#1098 (N24-N26 by the r1 reconciler; all six by the r2
        # implementer), so the refusal is status-quo-preserving.
        pytest.param(
            "ssh host 'echo git reset --hard' > /tmp/x; bash /tmp/x",
            id="N24-redirect_seq_bash_ssh_producer",
        ),
        pytest.param(
            "grep 'git reset --hard' recovery.txt > /tmp/x && bash /tmp/x",
            id="N25-redirect_and_bash_grep_producer",
        ),
        pytest.param(
            "grep 'git clean -fd' notes.md >> /tmp/x; . /tmp/x",
            id="N26-redirect_append_source_grep_producer",
        ),
        pytest.param(
            # END-position redirect (no same-call consumer) still refuses —
            # fail-closed residual FP, gap (xiv).
            "ssh pod-779 'git reset --hard' > /tmp/pod.log",
            id="N27-redirect_end_position_residual_fp",
        ),
        pytest.param(
            # Mixed redirects: the /dev/null strip must NOT unlock a real
            # file redirect sitting beside it.
            "grep 'git reset --hard' f > /tmp/x 2>/dev/null; bash /tmp/x",
            id="N28-dev_null_strip_does_not_unlock_file_redirect",
        ),
        pytest.param(
            # A REMOTE-side redirect inside the quoted ssh string refuses too
            # (the raw scan cannot tell it from a local one) — fail-closed
            # residual FP, gap (xiv).
            "ssh pod-779 'git checkout HEAD -- app.py > /tmp/remote.log'",
            id="N29-remote_side_redirect_quote_blind_residual_fp",
        ),
        # N30-N35 pin the /dev/null exemption's REGEX BOUNDARY (concern id
        # dev-null-boundary-fixture-gap): invalid targets that share the
        # /dev/null prefix or its spelling must NOT be stripped by the cond
        # (3b) strip-then-scan. The r2 reconciler probe showed a plausible
        # boundary relaxation (`([[:space:]]|$)` -> `([^[:alnum:]]|$)`, the
        # natural shape of a fix for the gap-(xiv) quote-flush FP) kept all
        # r2 fixtures green while fail-opening the glued double-redirect
        # write-then-execute channel; N30/N31/N33 go red under exactly that
        # relaxation. N34/N35 pin the quoted-target spelling's documented
        # fail-closed behavior (the raw strip matches only unquoted targets).
        pytest.param(
            # GLUED double-redirect: the `>` boundary after /dev/null is not
            # whitespace/EOL, so NOTHING strips and the file redirect refuses
            # — the load-bearing shape from the r2 reconciler probe.
            "grep 'git reset --hard' f 2>/dev/null>/tmp/px; bash /tmp/px",
            id="N30-glued_double_redirect_grep_producer",
        ),
        pytest.param(
            "ssh host 'echo git reset --hard' 2>/dev/null>/tmp/px; bash /tmp/px",
            id="N31-glued_double_redirect_ssh_producer",
        ),
        pytest.param(
            # SUFFIX target sharing the /dev/null prefix — not a discard sink.
            "grep 'git clean -fd' f >/dev/nullX; bash /tmp/px",
            id="N32-dev_null_suffix_target_not_stripped",
        ),
        pytest.param(
            # PATH CONTINUATION under /dev/null — boundary is `/`, not
            # whitespace/EOL; a real (if bizarre) file target, never stripped.
            "grep 'git reset --hard' f > /dev/null/sub; bash /tmp/px",
            id="N33-dev_null_path_continuation_not_stripped",
        ),
        pytest.param(
            # QUOTED /dev/null target: the raw strip matches only the
            # unquoted spelling, so this refuses — pins the documented
            # fail-closed behavior (gap (xiv) quoted-target FP).
            "grep 'git reset --hard' f > \"/dev/null\"; echo ok",
            id="N34-quoted_dev_null_target_fail_closed",
        ),
        pytest.param(
            "ssh pod-779 'git reset --hard' > '/dev/null'",
            id="N35-single_quoted_dev_null_target_fail_closed",
        ),
    ],
)
def test_remote_waiver_fail_closed_blocks(cmd):
    """Every waiver refusal arm keeps its locally-executing lookalike at exit 2."""
    assert _run(cmd) == 2


# ==== #1128 — branch-merge fence (the #1090 conflict-marker incident class) ====
#
# A `git merge <ref>` at the SHARED repo root strands conflict markers in the
# shared tree on conflict (#1090: ~70s window a concurrent `git add && git
# commit` could sweep) and lands branch commits on root main outside the
# Step 10d landing path even when clean/ff. TIGHT anchor: `merge` followed by
# whitespace/end-of-clause (NOT `\b`, which would trip `git merge-base`).
# Allow-arm: `--abort`/`--quit` immediately after the verb (the sanctioned
# in-progress-merge recovery; the anchored form kills the quoted `-m "…
# --abort …"` spoof). `--continue` and `--ff-only` block fail-closed. The
# block side is @on_main (the guard's on-main gate exits 0 off-main); the
# allow side must pass in either repo state. Block tests fire on shape alone
# — the detector is pure grep, never routing to the ref-resolution probes.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("git merge issue-123", id="M1-bare_branch_merge"),
        pytest.param("git merge origin/main", id="M2-merge_origin_main"),
        pytest.param("git merge --squash somebranch", id="M3-squash"),
        pytest.param("git merge --no-commit --no-ff issue-5", id="M4-no_commit_no_ff"),
        pytest.param("git merge --continue", id="M5-continue_completes_root_merge"),
        pytest.param("git merge --ff-only origin/main", id="M6-ff_only_fail_closed"),
        pytest.param("git merge", id="M7-bare_merge_end_of_clause"),
        pytest.param("git -c core.editor=true merge x", id="M8-global_flag_prefixed"),
        pytest.param(
            "git fetch origin && git merge origin/main", id="M9-compound_fetch_then_merge"
        ),
        pytest.param("git merge issue-1 # --abort", id="M10-comment_tail_cannot_spoof_allow"),
        pytest.param(
            'git merge -m "then --abort" issue-5', id="M11-quoted_abort_in_m_msg_cannot_spoof"
        ),
    ],
)
def test_merge_shapes_block(cmd):
    assert _run(cmd) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("git merge --abort", id="A1-abort_recovery"),
        pytest.param("git merge --quit", id="A2-quit_recovery"),
        pytest.param(
            "git -C .claude/worktrees/issue-123 merge origin/main", id="A3-dash_C_worktree"
        ),
        pytest.param("git -C /tmp/scratch merge issue-123", id="A4-dash_C_scratch"),
        pytest.param("cd .claude/worktrees/x && git merge origin/main", id="A5-cd_worktree_latch"),
        pytest.param("cd /tmp/m && git merge issue-1", id="A6-cd_tmp_latch"),
        pytest.param(
            'WT=.claude/worktrees/issue-1; cd "$WT" && git merge origin/main',
            id="A7-wt_variable_latch",
        ),
        pytest.param(
            "git merge-base --is-ancestor HEAD origin/main", id="A8-merge_base_is_ancestor"
        ),
        pytest.param("git merge-base --all main HEAD", id="A9-merge_base_all"),
        pytest.param("git pull --rebase=merges --autostash", id="A10-pull_rebase_merges"),
        pytest.param("git mergetool", id="A11-mergetool"),
        pytest.param("git log --merges", id="A12-log_merges"),
        pytest.param("git branch --merged", id="A13-branch_merged"),
        pytest.param(
            "gh pr merge 123 --rebase --delete-branch=false", id="A14-gh_pr_merge_no_git_word"
        ),
        pytest.param('git commit -m "merge the eval tables"', id="A15-prose_merge_in_commit_msg"),
        pytest.param(
            "git worktree add --detach /tmp/m origin/main"
            " && git -C /tmp/m merge issue-1 && git -C /tmp/m push origin HEAD:main",
            id="A16-scratch_worktree_recipe_compound",
        ),
        pytest.param("ssh pod-779 'git merge origin/main'", id="A17-ssh_remote_merge_waiver"),
        pytest.param("grep -q 'git merge issue-1' notes.md", id="A18-grep_pattern_waiver"),
    ],
)
def test_merge_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


@on_main
def test_note_text_merge_literal_trips_guard_known_limitation():
    # KNOWN LIMITATION (header): a quoted FULL `git merge <ref>` command
    # literal in --note/-m text trips the raw scan (#1128, mirror of the
    # restore-literal pin). Workaround: --file <path.md> / git commit -F.
    assert (
        _run(
            "uv run python scripts/task.py post-marker 1128 epm:x "
            '--note "run git merge issue-1 next"'
        )
        == 2
    )


# ==== #1193 — rebase-family fence (sibling of the #1128 merge fence) =========
#
# `git rebase <ref>` / `git cherry-pick <ref>` at the SHARED repo root strand
# conflict state in the shared tree on conflict (the #1090 incident class) and
# rewrite/land commits on root main outside the sanctioned landing paths even
# when clean (a bare `git rebase` with a configured upstream genuinely RUNS,
# so the end-of-clause shape blocks too). TIGHT anchor: verb followed by
# whitespace/end-of-clause (NOT `\b`, which would trip
# `git -c rebase.autoStash=true pull` at flag-value position — RA10).
# Allow-arm: `--abort`/`--quit` immediately after the verb, ONE ARM PER VERB
# (a combined `(rebase|cherry-pick)` allow would open the R13 cross-verb
# quoted-arg spoof; a loose `[^;&|]*` allow would fail open on R12/CP9).
# `--continue`/`--skip` block fail-closed (both COMPLETE the in-progress
# operation on the root tree — the M5 decision, mirrored). The block side is
# @on_main (the guard's on-main gate exits 0 off-main); the allow side must
# pass in either repo state. Block tests fire on shape alone — the detectors
# are pure grep, never routing to the ref-resolution probes.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("git rebase issue-123", id="R1-bare_branch_rebase"),
        pytest.param("git rebase origin/main", id="R2-rebase_origin_main"),
        pytest.param("git rebase", id="R3-bare_rebase_end_of_clause"),
        pytest.param("git rebase -i HEAD~3", id="R4-interactive"),
        pytest.param("git rebase --continue", id="R5-continue_completes_root_op"),
        pytest.param("git rebase --skip", id="R6-skip_is_continue_class"),
        pytest.param("git rebase --onto main a b", id="R7-onto_form"),
        pytest.param("git -c core.editor=true rebase x", id="R8-global_flag_prefixed"),
        pytest.param(
            "git fetch origin && git rebase origin/main", id="R9-compound_fetch_then_rebase"
        ),
        pytest.param("git rebase issue-1 # --abort", id="R10-comment_tail_cannot_spoof_allow"),
        pytest.param("git rebase --autostash origin/main", id="R11-flag_then_ref_no_allow"),
        pytest.param(
            'git rebase --exec "then --abort" issue-5',
            id="R12-quoted_abort_in_exec_cannot_spoof",
        ),
        pytest.param(
            'git rebase --exec "cherry-pick --abort" main',
            id="R13-cross_verb_quoted_abort_cannot_spoof",
        ),
        pytest.param("git cherry-pick abc1234", id="CP1-bare_sha_pick"),
        pytest.param("git cherry-pick", id="CP2-bare_end_of_clause"),
        pytest.param("git cherry-pick --continue", id="CP3-continue"),
        pytest.param("git cherry-pick --skip", id="CP4-skip"),
        pytest.param("git cherry-pick -n abc1234", id="CP5-no_commit_still_mutates"),
        pytest.param("git -c core.editor=true cherry-pick x", id="CP6-global_flag_prefixed"),
        pytest.param("git cherry-pick abc1 # --abort", id="CP7-comment_tail_cannot_spoof"),
        pytest.param("git fetch origin && git cherry-pick abc1", id="CP8-compound"),
        pytest.param(
            'git cherry-pick --strategy-option "x --abort" abc1',
            id="CP9-quoted_abort_in_strategy_option_cannot_spoof",
        ),
    ],
)
def test_rebase_family_shapes_block(cmd):
    assert _run(cmd) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("git rebase --abort", id="RA1-rebase_abort_recovery"),
        pytest.param("git rebase --quit", id="RA2-rebase_quit_recovery"),
        pytest.param("git cherry-pick --abort", id="RA3-cp_abort_recovery"),
        pytest.param("git cherry-pick --quit", id="RA4-cp_quit_recovery"),
        pytest.param("git -C .claude/worktrees/issue-123 rebase main", id="RA5-dash_C_worktree"),
        pytest.param("git -C /tmp/scratch cherry-pick abc123", id="RA6-dash_C_scratch_pick"),
        pytest.param("cd .claude/worktrees/x && git rebase main", id="RA7-cd_worktree_latch"),
        pytest.param(
            'WT=.claude/worktrees/issue-1; cd "$WT" && git rebase origin/main',
            id="RA8-wt_variable_latch",
        ),
        pytest.param("git pull --rebase --autostash", id="RA9-pull_rebase_bare_flag"),
        pytest.param("git -c rebase.autoStash=true pull", id="RA10-config_value_position"),
        pytest.param("git log --cherry-pick --right-only A...B", id="RA11-log_cherry_pick_flag"),
        pytest.param("git log --cherry A...B", id="RA12-log_cherry_flag"),
        pytest.param("git cherry v1.0 main", id="RA13-git_cherry_plumbing"),
        pytest.param(
            'git commit -m "cherry-picked for illustration"',
            id="RA14-prose_cherry_picked_suffix",
        ),
        pytest.param(
            'git commit -m "rebase fence for the guard"', id="RA15-prose_rebase_in_commit_msg"
        ),
        pytest.param("ssh pod-779 'git rebase origin/main'", id="RA16-ssh_remote_rebase_waiver"),
        pytest.param("grep -q 'git cherry-pick abc' notes.md", id="RA17-grep_pattern_waiver"),
    ],
)
def test_rebase_family_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


@on_main
@pytest.mark.parametrize(
    "note_cmd",
    [
        pytest.param(
            "uv run python scripts/task.py post-marker 1193 epm:x "
            '--note "run git rebase issue-1 next"',
            id="note_rebase_literal",
        ),
        pytest.param(
            "uv run python scripts/task.py post-marker 1193 epm:x "
            '--note "then git cherry-pick abc1 onto main"',
            id="note_cherry_pick_literal",
        ),
    ],
)
def test_note_text_rebase_family_literal_trips_guard_known_limitation(note_cmd):
    # KNOWN LIMITATION (header): a quoted FULL `git rebase <ref>` /
    # `git cherry-pick <sha>` command literal in --note/-m text trips the raw
    # scan (#1193, mirror of the merge-literal pin). Workaround: --file
    # <path.md> / git commit -F.
    assert _run(note_cmd) == 2


def test_man_git_rebase_allowed():
    # `man git-rebase` passes the loose pre-filter (`\bgit\b` fires before the
    # `-`) but the tight anchors require a `git ` + space bigram, so the
    # man-page form is never classified — the known-limitation (xvii)(c)
    # remediation for the `git rebase --help` false-block parity.
    assert _run("man git-rebase") == 0


# ==== #1234 — revert/am fence (completeness siblings of the #1193 family) ====
#
# `git revert <commit>` / `git am <mbox>` at the SHARED repo root strand
# sequencer/am state + conflict markers in the shared tree on conflict (the
# #1090 incident class) and land commits on root main outside the sanctioned
# landing paths even when clean (bare `git revert` errors in git but blocks
# fail-closed — M7/CP2 parity; bare `git am` reads patches from STDIN and
# genuinely RUNS, so the end-of-clause shape is load-bearing there). TIGHT
# anchor: verb followed by whitespace/end-of-clause; prose `-m "revert foo"` /
# `-m "I am done"` never match (`commit` is a non-dash token that breaks the
# flag chain — RVA8/RVA10). Allow-arm: `--abort`/`--quit` immediately after
# the verb, ONE ARM PER VERB (a combined `(revert|am)` allow would open the
# R13 cross-verb quoted-arg spoof). `--continue`/`--skip` block fail-closed
# (both COMPLETE the in-progress op on the root tree — the M5 decision,
# mirrored), as does the read-only `git am --show-current-patch` (strict
# abort/quit-only parity, register (xviii)(a)). The block side is @on_main
# (the guard's on-main gate exits 0 off-main); the allow side must pass in
# either repo state. Block tests fire on shape alone — the detectors are pure
# grep, never routing to the ref-resolution probes.
# ---------------------------------------------------------------------------
@on_main
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("git revert HEAD", id="RV1-canonical_revert_head"),
        pytest.param("git revert", id="RV2-bare_end_of_clause"),
        pytest.param("git revert --continue", id="RV3-continue_completes_root_op"),
        pytest.param("git revert --skip", id="RV4-skip_is_continue_class"),
        pytest.param("git revert -n HEAD", id="RV5-no_commit_still_mutates_index"),
        pytest.param("git revert --no-commit HEAD~2..HEAD", id="RV6-no_commit_range"),
        pytest.param("git -c core.editor=true revert abc1", id="RV7-global_flag_prefixed"),
        pytest.param("git revert abc1 # --abort", id="RV8-comment_tail_cannot_spoof_allow"),
        pytest.param("git fetch origin && git revert HEAD", id="RV9-compound_fetch_then_revert"),
        pytest.param(
            'git revert --strategy-option "x --abort" HEAD',
            id="RV10-quoted_abort_in_strategy_option_cannot_spoof",
        ),
        pytest.param("git revert -m 1 abc123", id="RV11-mainline_flag"),
        pytest.param("git am /tmp/patch.mbox", id="AM1-canonical_mbox_apply"),
        pytest.param("git am", id="AM2-bare_end_of_clause_reads_stdin"),
        pytest.param("git am --continue", id="AM3-continue_completes_root_op"),
        pytest.param("git am --skip", id="AM4-skip_is_continue_class"),
        pytest.param("git am --3way /tmp/p.mbox", id="AM5-three_way"),
        pytest.param("git -c core.editor=true am /tmp/p.mbox", id="AM6-global_flag_prefixed"),
        pytest.param("git am /tmp/p.mbox # --abort", id="AM7-comment_tail_cannot_spoof_allow"),
        pytest.param("git am --show-current-patch", id="AM8-show_current_patch_fail_closed"),
        pytest.param(
            "curl -s http://x/p.mbox | git am", id="AM9-piped_apply_consumer_clause_classifies"
        ),
    ],
)
def test_revert_am_shapes_block(cmd):
    assert _run(cmd) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param("git revert --abort", id="RVA1-revert_abort_recovery"),
        pytest.param("git revert --quit", id="RVA2-revert_quit_recovery"),
        pytest.param("git am --abort", id="RVA3-am_abort_recovery"),
        pytest.param("git am --quit", id="RVA4-am_quit_recovery"),
        pytest.param("git -C .claude/worktrees/issue-123 revert HEAD", id="RVA5-dash_C_worktree"),
        pytest.param("git -C /tmp/scratch am /tmp/p.mbox", id="RVA6-dash_C_scratch_am"),
        pytest.param("cd .claude/worktrees/x && git revert HEAD", id="RVA7-cd_worktree_latch"),
        pytest.param('git commit -m "revert foo"', id="RVA8-prose_revert_in_commit_msg"),
        pytest.param(
            'git commit -m "reverted the earlier change"',
            id="RVA9-prose_reverted_suffix_prefilter_skip",
        ),
        pytest.param(
            'git commit -m "I am updating the plan"', id="RVA10-prose_am_flag_chain_break"
        ),
        pytest.param("git commit --amend -m fix", id="RVA11-amend_prefilter_skip"),
        pytest.param("git log --grep revert --oneline", id="RVA12-log_chain_break"),
        pytest.param("ssh pod-779 'git revert HEAD'", id="RVA13-ssh_remote_revert_waiver"),
        pytest.param("ssh pod-779 'git am /tmp/p.mbox'", id="RVA14-ssh_remote_am_waiver"),
        pytest.param("grep -q 'git revert HEAD' notes.md", id="RVA15-grep_pattern_waiver"),
        pytest.param(
            'WT=.claude/worktrees/issue-1; cd "$WT" && git revert HEAD',
            id="RVA16-wt_variable_latch",
        ),
        pytest.param(
            "git -c revert.reference=true log --oneline", id="RVA17-config_value_position"
        ),
    ],
)
def test_revert_am_allowed_shapes_exit0(cmd):
    assert _run(cmd) == 0


@on_main
@pytest.mark.parametrize(
    "note_cmd",
    [
        pytest.param(
            "uv run python scripts/task.py post-marker 1234 epm:x "
            '--note "then git revert HEAD to undo"',
            id="note_revert_literal",
        ),
        pytest.param(
            "uv run python scripts/task.py post-marker 1234 epm:x "
            '--note "run git am /tmp/p.mbox next"',
            id="note_am_literal",
        ),
    ],
)
def test_note_text_revert_am_literal_trips_guard_known_limitation(note_cmd):
    # KNOWN LIMITATION (header): a quoted FULL `git revert <sha>` /
    # `git am <path>` command literal in --note/-m text trips the raw scan
    # (#1234, mirror of the #1128/#1193 pins). Workaround: --file <path.md> /
    # git commit -F <file>.
    assert _run(note_cmd) == 2


@on_main
def test_flag_chain_valid_git_am_prose_trips_guard_known_limitation():
    # KNOWN LIMITATION (register (xviii)(b), honest wording): valid
    # global-flag-chain git with quoted prose containing a standalone `am`
    # tight-matches — the flag chain consumes `log` as `--no-pager`'s value,
    # leaving `am` in subcommand position. Accepted accidents-not-adversaries
    # FP class; bounce-only failure direction (the command is read-only, so
    # the false block costs a retry, never data).
    assert _run('git --no-pager log --since "9 am today"') == 2


def test_man_git_am_revert_allowed():
    # `man git-am` / `man git-revert` pass the loose pre-filter (`\bgit\b`
    # fires before the `-`; `\bam\b`/`\brevert\b` fire inside the hyphenated
    # page name) but the tight anchors require a `git ` + space bigram, so
    # the man-page forms are never classified — the (xviii)(c) remediation
    # for the `git am --help` / `git revert --help` false-block parity.
    assert _run("man git-am") == 0
    assert _run("man git-revert") == 0
