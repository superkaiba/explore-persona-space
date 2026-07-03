"""End-to-end tests for the ``scripts/guard_repo_root_branch.sh`` PreToolUse hook.

The guard blocks any git command that would move the SHARED repo-root working
tree off ``main`` OR detach its HEAD, because the repo root is the canonical
commit target for ``scripts/task.py`` and every concurrent VM Claude session
(all assume ``HEAD==main``). A detached repo-root HEAD additionally crashes
``task_workflow``'s main-worktree resolver. As of #897 it ALSO blocks
working-tree REVERTS on the shared root (``git restore``, pathspec /
bare-path / force ``git checkout``, ``git clean -f``, ``git reset --hard``) —
the #841 incident class where a concurrent destructive op silently reverted
another session's uncommitted edits.

These tests drive the script exactly as the harness does: stdin PreToolUse JSON
``{"tool_input": {"command": <cmd>}}`` -> exit 2 (blocked) or exit 0 (allowed).
This mirrors the subprocess-drives-script convention of the sibling
``tests/test_*guard*`` files.

The guard's on-main gate exits 0 when the repo-root HEAD is already off ``main``
(it never traps a user recovering from an already-detached/off-main state), so
the BLOCK-path tests are guarded by ``@on_main``. The ALLOW-path and fail-soft
tests run regardless — the guard must never trap those shapes in either state.
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
