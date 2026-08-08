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
``am --show-current-patch`` fail-closed). As of #1554 it ALSO declines
worktree-scoped merges of the bare LOCAL ``main`` ref — the #1530
contamination class (a worktree fast-forward onto the possibly-stale,
unpushed local ``main`` tip imports root-only commits): Arm A intercepts
``git -C <.claude/worktrees/ path | $WT spelling> ... merge ... main``
BEFORE the path-blind ``-C`` waiver; Arm B declines the same bare-main merge
shape under a WORKTREE-armed cd-latch (the ``scoped_wt`` bit; ``/tmp``
latches keep their prior disposition byte-identical). ``origin/main`` /
raw-sha / ``"$MAIN_SHA"`` merges and ``/tmp`` scratch landings stay allowed;
override via ``EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1`` (session env or
inline command prefix — a command-wide substring match, the
``EPM_ALLOW_ROOT_PULL`` sibling idiom). Residuals: hook header gap (xx).

These tests drive the script exactly as the harness does: stdin PreToolUse JSON
``{"tool_input": {"command": <cmd>}}`` -> exit 2 (blocked) or exit 0 (allowed).
This mirrors the subprocess-drives-script convention of the sibling
``tests/test_*guard*`` files.

The guard's on-main gate exits 0 when the repo-root HEAD is already off ``main``
(it never traps a user recovering from an already-detached/off-main state). The
harness pins every git-STATE read the hook makes — the ref/commit-ish resolution
probes (hook L1384-1388) and that tail on-main gate (L1777) — to a session-scoped
fixture repo that is always on ``main``, by exporting ``GIT_DIR``/``GIT_WORK_TREE``
in the hook subprocess env (git env precedence over the hook's hardcoded
``git -C "$REPO"``; the #1545 pattern, sibling:
``workflow_lint._root_guard_git_env``). Block/allow outcomes are therefore
deterministic under ANY concurrent shared-root git state — #1528: a mid-run
off-main flip (e.g. a ``sync_repo_root.py`` rebase transiently detaching HEAD)
failed 23 block tests via the tail fail-open arm, which the former import-time
``@on_main`` skipif could not see. That skipif is deleted; the fail-open arm now
has deterministic pin tests of its own (end of module).
As of #1098 the guard WAIVES, per-clause and under a fail-closed refusal ladder
(pipe/BG-producer position, expansion + here-string syntax, non-/dev/null
output redirects, ssh local-exec options, shared-repo path spellings,
``rg --pre``), clauses whose command word is ``ssh`` (remote execution) or
``grep``/``egrep``/``fgrep``/``rg`` (read-only pattern argument).
As of #1413 a pre-split masking pass (``mask_ssh_payload_separators``) merges
the CANONICAL single-quoted multi-statement ssh payload — the quoted payload
is the clause's final token and the whole candidate passes an 8-arm
fail-closed refusal predicate (R1-R8) — into ONE clause, so it reaches the
#1098 waiver whole (closes residual (xiv)'s mis-split false positive for
that shape; the former block fixture N1 moved to the masking allow list as
M9). Any refused candidate leaves the input byte-identical, keeping today's
disposition.
As of #1463 both mechanisms gain a ``gcloud compute ssh`` head (optional
literal ``timeout <num>[.frac][smhd]?`` wrapper): a
``gcloud compute ssh <instance> --command='<payload>'`` clause executes its
payload ON the GCE instance via the local ssh(1) wrapper (SDK 576.0.0 help +
a live ``--dry-run`` argv probe), so the driver-loop waiver and the masking
pre-pass treat it exactly like the clause-initial ``ssh`` head, under the
identical fail-closed refusal arms (founding incident #825, 2026-07-16; the
ssh variant was #1336, closed by #1413). As of #1859 the SAME literal
timeout wrapper is accepted on the bare ``ssh`` head too (the former
N12/NM5 asymmetry pins flipped to positive fixtures S15/M10; founding
incident: two #1769 failover-path false blocks). Everything outside the
narrow heads — release tracks, non-timeout wrappers (nohup / env-prefix /
abs-path / variable heads), ``timeout`` flag forms, redirect / expansion /
proxy-token shapes — keeps today's blocked disposition (GN- and
N36-N43-series pins).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / "scripts" / "guard_repo_root_branch.sh"

# The script hardcodes REPO=/home/thomasjiralerspong/explore-persona-space and
# runs ``git -C "$REPO" ...`` for its branch/commit-ish resolution (hook L1384
# show-ref / L1386 rev-parse / L1388 cat-file) + the tail on-main gate (L1777).
# The harness pins GIT_DIR/GIT_WORK_TREE in every hook subprocess env to a
# session-scoped always-on-main fixture repo (_build_pinned_repo/_pinned_env
# below) — git env vars take precedence over ``-C`` — so ALL of those git-state
# reads resolve in the PINNED repo, never the live shared root (#1528 flake
# class; sibling: workflow_lint._root_guard_git_env, #1545). Known residual,
# accepted: the hook's ``[ -e "$REPO/$arg" ]`` filesystem probe (L1388b) still
# reads the real root's fs — it is branch-state-independent, and dead code for
# block rows (the seeded fixture resolves their pathspecs at the cat-file site
# first). REPO below stays the REAL root: the deny-sidecar snapshot + the
# REPO-constant text pin test read it; no git STATE is read through it anymore.
REPO = Path("/home/thomasjiralerspong/explore-persona-space")

# Deny-event sidecar (#1528, #1990): the guard's default sidecar path lives
# under the CANONICAL checkout's .claude/cache/ (the guard hardcodes REPO).
# The harness pins EPM_GUARD_DENY_SIDECAR to /dev/null so the suite's hundreds
# of deny cases never create/append the production sidecar; the ROW-IDENTITY
# snapshot below backs the end-of-module production-protection test.
#
# Row-identity (not byte-size) because a foreign concurrent session may
# LEGITIMATELY append a REAL deny row during our gate window (#1876 measured
# a ~36-min window with two such rows); a byte-size predicate false-FAILs
# there. The membership predicate (every snapshot row still present at
# end-of-module) tolerates foreign appends while still proving no OBSERVED
# row was rewritten/dropped.
_PROD_SIDECAR = REPO / ".claude" / "cache" / "guard-deny-events.jsonl"


def _read_sidecar_rows(path: Path) -> list[bytes]:
    """Snapshot the raw newline-terminated rows of a deny-event sidecar file.

    Returns an empty list when the file is absent. Reads the file bytes and
    splitkeep-splits on `\\n` (drops the trailing empty element from a
    final newline), so each element is a complete row including its `\\n`.
    The bytes-in-bytes-out shape makes membership comparison exact under
    concurrent appenders: rows a snapshot observed at time T remain a
    subset of the file's rows at any later time (an append-only sidecar
    cannot rewrite or reorder existing rows).

    Used by the end-of-module production-sidecar canary to make the
    membership predicate (`snapshot rows ⊆ current rows`) tolerant of
    foreign appends from concurrent REAL denies in other sessions,
    while still refusing a suite-attributed write.
    """
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []
    rows = data.split(b"\n")
    # split() leaves a trailing empty when the file ends with `\n`; drop it.
    if rows and rows[-1] == b"":
        rows.pop()
    return [row + b"\n" for row in rows]


_PROD_SIDECAR_ROWS_AT_IMPORT = _read_sidecar_rows(_PROD_SIDECAR)


def _scrubbed_env() -> dict[str, str]:
    """``os.environ`` minus every ``GIT_*`` var (the #1545 scrub).

    Base for EVERY subprocess env in this module — hook invocations AND the
    harness's own git calls — so an ambient GIT_DIR / GIT_OBJECT_DIRECTORY /
    GIT_INDEX_FILE (e.g. a pre-commit caller) can never redirect either side.
    Built from ``os.environ`` at call time so ``monkeypatch``-based env tests
    keep working.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _build_pinned_repo(d: Path) -> None:
    """git-init a minimal always-on-main repo the hook's git reads resolve against.

    Seed <-> consumer map (every resolution-dependent fixture row):
    2 commits => ``HEAD~1`` rows resolve (hook rev-parse site); the reflog
    (written by default on non-bare init) => the ``HEAD@{0}`` row; the seeded
    ``refs/remotes/origin/main`` => the ``origin/main`` rows; the committed
    ``CLAUDE.md`` + ``scripts/eval.py`` => the bare-pathspec existence-probe
    rows resolve hermetically at the hook's ``cat-file -e HEAD:<arg>`` site
    (so the real-root fs probe is never consulted for them). Deliberately
    strictly minimal otherwise: fail-soft rows' nonexistent-token args must
    NOT resolve.
    """
    env = _scrubbed_env()
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"  # ambient gpgsign/hooks isolation (#1545)
    env["GIT_CONFIG_NOSYSTEM"] = "1"

    def run(*args: str) -> None:
        subprocess.run(args, check=True, capture_output=True, env=env, timeout=20)

    run("git", "init", "-q", "-b", "main", str(d))  # -b: git >= 2.28
    ident = ("-c", "user.name=eps-guard-test", "-c", "user.email=guard-test@eps.local")
    run("git", "-C", str(d), *ident, "commit", "-q", "--allow-empty", "-m", "c1")
    (d / "CLAUDE.md").write_text("pinned fixture\n")
    (d / "scripts").mkdir()
    (d / "scripts" / "eval.py").write_text("# pinned fixture\n")
    run("git", "-C", str(d), "add", "CLAUDE.md", "scripts/eval.py")
    run("git", "-C", str(d), *ident, "commit", "-q", "-m", "c2")  # 2 commits => HEAD~1
    run("git", "-C", str(d), "update-ref", "refs/remotes/origin/main", "HEAD")


_PINNED_REPO: Path | None = None


@pytest.fixture(scope="session", autouse=True)
def _pinned_repo_session(tmp_path_factory):
    """Build the pinned repo ONCE per pytest process; every hook env points at it.

    ``tmp_path_factory`` basetemp is per-process (numbered ``pytest-<N>`` dirs
    with locking under ``/tmp/pytest-of-<user>``), so concurrent gate runs each
    build their own fixture; pytest's retention policy cleans old basetemps.
    """
    global _PINNED_REPO
    d = tmp_path_factory.mktemp("eps-guard-pinned-repo")
    _build_pinned_repo(d)
    _PINNED_REPO = d
    yield
    _PINNED_REPO = None


def _pinned_env(sidecar: str | None) -> dict[str, str]:
    """Hook-subprocess env: GIT_* scrubbed, git-state reads pinned to the fixture.

    GIT_DIR/GIT_WORK_TREE take precedence over the hook's hardcoded
    ``git -C "$REPO"`` (git(1) ENVIRONMENT; verified on git 2.34.1 for all four
    subcommands the hook runs), redirecting every git-state read to the pinned
    always-on-main repo.
    """
    assert _PINNED_REPO is not None, "pinned-repo session fixture did not run"
    env = _scrubbed_env()
    env["GIT_DIR"] = str(_PINNED_REPO / ".git")
    env["GIT_WORK_TREE"] = str(_PINNED_REPO)
    env["EPM_GUARD_DENY_SIDECAR"] = sidecar or "/dev/null"
    return env


def _run_full(cmd: str, *, sidecar: str | None = None) -> subprocess.CompletedProcess[str]:
    """Feed ``cmd`` to the guard via PreToolUse JSON; return the full result.

    ``sidecar`` pins ``EPM_GUARD_DENY_SIDECAR`` for the subprocess; the
    ``/dev/null`` default sinks every deny row so existing tests never touch
    the production sidecar (#1528). Appending to ``/dev/null`` succeeds
    silently and ``mkdir -p /dev`` no-ops, so exit-code behavior is unchanged.
    """
    payload = json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run(
        [str(SCRIPT)], input=payload, text=True, capture_output=True, env=_pinned_env(sidecar)
    )


def _run(cmd: str, *, sidecar: str | None = None) -> int:
    """Feed ``cmd`` to the guard via PreToolUse JSON; return its exit code."""
    return _run_full(cmd, sidecar=sidecar).returncode


def _run_raw(payload: str, *, sidecar: str | None = None) -> int:
    """Feed a raw (possibly malformed) stdin payload; return the exit code."""
    return subprocess.run(
        [str(SCRIPT)], input=payload, text=True, capture_output=True, env=_pinned_env(sidecar)
    ).returncode


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git against the PINNED fixture repo (scrubbed env; never the live root)."""
    assert _PINNED_REPO is not None, "pinned-repo session fixture did not run"
    return subprocess.run(
        ["git", "-C", str(_PINNED_REPO), *args],
        capture_output=True,
        text=True,
        env=_scrubbed_env(),
        timeout=20,
    )


def _repo_head_sha() -> str:
    """Short SHA of the PINNED fixture repo's HEAD (what the hook resolves against)."""
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


@pytest.fixture
def throwaway_branch(_pinned_repo_session):
    """A real local branch (in the PINNED repo) so ``refs/heads/<name>`` resolves.

    Created pointing at HEAD (no checkout, no tree change) and deleted on
    teardown, so the branch-switch-regression assertion is hermetic rather than
    depending on a repo-specific branch name existing. The uuid suffix and the
    tolerant teardown are kept (harmless), and the ref now lives only in the
    per-process pinned fixture — never the production repo.
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
def throwaway_tag(_pinned_repo_session):
    """A real local tag (in the PINNED repo) so ``<tag>^{commit}`` resolves.

    Detaching to a tag is a real detach shape; a fabricated non-existent tag
    would fail-soft (exit 0) and never exercise the block path — so the tag must
    genuinely exist. Created pointing at HEAD and deleted on teardown. The uuid
    suffix and tolerant teardown are kept; the ref lives only in the pinned
    fixture.
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


def test_detach_to_real_tag_blocks(throwaway_tag):
    # A real local tag resolves to a commit-ish -> detach -> block.
    assert _run(f"git checkout {throwaway_tag}") == 2


# ---------------------------------------------------------------------------
# #1621 — hyphen-preceded verb tokens (a `--no-checkout` / `--checkout` FLAG)
# no longer satisfy the checkout-detach clause (ERE \b matches between `-`
# and a word char, so the flag spelling used to false-block the documented
# scratch-worktree recipe — incident 552fa84d). Every real detach spelling
# keeps a space/tab before the verb and still blocks (matrix above + the
# flag-prefixed pin below). Red-before/green-after for each allow fixture
# was confirmed at compose time against the origin/main guard (rc=2 -> 0).
# ---------------------------------------------------------------------------
_I1621_552F_SCRATCH_WORKTREE_RECIPE = (
    "set -e\n"
    "git fetch origin main\n"
    "WT=/tmp/wt1092dash\n"
    "git worktree remove --force $WT 2>/dev/null || true\n"
    "git worktree add --no-checkout --detach $WT origin/main\n"
    "git -C $WT sparse-checkout set --cone scripts tasks/awaiting_promotion/1092/artifacts\n"
    "git -C $WT checkout --quiet\n"
    "cp scripts/issue1092_divergence_dashboard.py $WT/scripts/\n"
    "mkdir -p $WT/tasks/awaiting_promotion/1092/artifacts\n"
    "cp tasks/awaiting_promotion/1092/artifacts/issue1092_divergence_dashboard.html"
    " $WT/tasks/awaiting_promotion/1092/artifacts/\n"
    "git -C $WT add scripts/issue1092_divergence_dashboard.py"
    " tasks/awaiting_promotion/1092/artifacts/issue1092_divergence_dashboard.html\n"
    "git -C $WT status --short | head -5"
)


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "git worktree add --no-checkout --detach $WT origin/main",
            id="WA1-no_checkout_detach_line_solo",
        ),
        pytest.param(
            _I1621_552F_SCRATCH_WORKTREE_RECIPE,
            id="WA2-552f_incident_verbatim",
        ),
        pytest.param(
            "git worktree add --checkout --detach /tmp/wt1092dash origin/main",
            id="WA3-sibling_checkout_flag",
        ),
    ],
)
def test_worktree_add_no_checkout_detach_allowed(cmd):
    """#1621: the checkout-detach clause requires a non-hyphen char before the
    verb token, so `--no-checkout` / `--checkout` flag spellings never match
    and the scratch-worktree recipe passes."""
    assert _run(cmd) == 0


def test_flag_prefixed_checkout_detach_still_blocks():
    """A real detach with a config flag before the verb keeps a space directly
    before `checkout`, so the #1621 `[^-]` class still matches (blocks)."""
    assert _run("git -c advice.detachedHead=false checkout --detach abc1234") == 2


# ---------------------------------------------------------------------------
# MUST BLOCK — existing branch-switch regression fence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", ["git checkout -b feature/x", "git switch fix/foo"])
def test_branch_creation_and_switch_still_block(cmd):
    # -b/-B branch creation and `switch <non-main>` are blocked without needing
    # a resolvable ref (the detectors fire on the flag / arg shape).
    assert _run(cmd) == 2


def test_existing_local_branch_checkout_still_blocks(throwaway_branch):
    # `git checkout <real-local-branch>` (not main) is the original blocked case.
    assert _run(f"git checkout {throwaway_branch}") == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — return-to-main, scoped, and non-git shapes. These must never be
# trapped regardless of the repo-root HEAD state.
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
    # These must never be trapped regardless of repo-root HEAD state.
    # Round-1's quote-strip false-positived `git switch
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
# chained command after `main`. Must never trap in either repo-root HEAD
# state. Concern id: switch-main-prefix-allowarm-leak.
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


def test_note_text_git_verb_literal_double_quoted_now_allows():
    # (#1710) The historical KNOWN LIMITATION — a git-verb literal in a
    # DOUBLE-quoted `--note` argument tripped the guard because the taskpy
    # mask covered single quotes only — is closed by the #1710 P7 extension
    # that admits double-quoted spans under a no-expansion refusal ladder.
    # A double-quoted note whose body carries no `$` / backtick / `\\`
    # tokens is byte-identical to the same content single-quoted, so the
    # taskpy mask replaces its body with the neutral __EPM_ARG_PAYLOAD__
    # sentinel BEFORE the pre-filter regex scans for git-verb literals.
    # The genuine repo-root mutation shapes still block — see
    # test_taskpy_double_quoted_masking_refusal_ladder_blocks below.
    assert (
        _run(
            'uv run python scripts/task.py post-marker 796 epm:foo --note "test git switch string"'
        )
        == 0
    )


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
# moves off main. Concern id:
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
@pytest.mark.parametrize("cmd", ["git checkout -bfoo", "git checkout -Bfoo"])
def test_glued_shortflag_branch_creation_still_blocks(cmd):
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# MUST ALLOW — legitimate compounds the fleet uses, including legitimate || / |
# shapes that DON'T scope a cd onto a git switch. Clause-local parsing must keep
# these passing. Must never trap in either repo-root HEAD state.
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
# allowed. Concern id:
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


def test_note_text_restore_literal_double_quoted_now_allows():
    # (#1710) Sibling of test_note_text_git_verb_literal_double_quoted_now_allows:
    # the taskpy mask's P7 double-quoted extension replaces a double-quoted
    # `--note` body with the neutral __EPM_ARG_PAYLOAD__ sentinel BEFORE the
    # pre-filter scans for git-verb literals. A note whose body carries no
    # `$` / backtick / `\\` tokens (byte-identical-to-single-quoted content)
    # therefore no longer trips the raw scan. The historical
    # `--file <path.md>` / `git commit -F <file>` workaround is no longer
    # required for these shapes. Genuine repo-root mutation shapes still
    # block — see test_taskpy_double_quoted_masking_refusal_ladder_blocks.
    assert (
        _run(
            "uv run python scripts/task.py post-marker 897 epm:x "
            '--note "run git restore . to revert"'
        )
        == 0
    )


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
    # the arg before probing. The allow must hold in either repo-root HEAD
    # state.
    assert _run("git checkout /home/thomasjiralerspong/explore-persona-space/CLAUDE.md") == 0


# ---------------------------------------------------------------------------
# MUST ALLOW — #897 allow-side: index-only restore, dry-run clean, the safe
# stash alternative, per-clause `-C` waivers, cd-latches, tight-anchor
# non-matches, comment clauses, and unresolvable bare args. Must never trap
# in either repo-root HEAD state.
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
# clause. Covers the recorded incident shapes (spec-sync recipes). Allows
# must hold in either repo-root HEAD state.
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
        # #1861: relocated from the BLOCK battery — name generalization makes this
        # fully-compliant arming latch; the prefix-collision half lives in
        # X16-name_mismatch_prefix_collision
        pytest.param(
            'WT2=".claude/worktrees/issue-9"\ncd "$WT2" && git checkout main -- specs.md',
            id="R12-wt2_name",
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
# carry no expansion syntax beyond plain `${NAME}` parameter references —
# check (g) deletes plain spans from a scan copy before the refusal (#1501).
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
            "cat > /tmp/note.md <<EOF\nvalue is ${SOMEVAR}\nhow to revert: git restore .\nEOF",
            id="M1d-bare_param_expansion_plus_gated_prose",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\nvalue is ${A} and ${B_2}\nhow to revert: git restore .\nEOF",
            id="M1j-multiple_plain_spans_plus_gated_prose",
        ),
        pytest.param(
            # <<- tag form: terminator (and body) genuinely TAB-indented — a
            # space-indented terminator under <<- never terminates and would
            # exercise the unterminated-refusal arm instead of check (g).
            "cat > /tmp/note.md <<-EOF\n"
            "\tvalue is ${SOMEVAR}\n"
            "\thow to revert: git restore .\n"
            "\tEOF",
            id="M1k-dash_tag_plain_span",
        ),
        pytest.param(
            # Copy-contract discriminator (#1501): line 1 = the fenced-verb
            # prose with the verb INTERRUPTED by a plain span; line 2 = a
            # (g)-refusing form. Line ORDER is load-bearing (pass 1 breaks at
            # the first refusal, so the interrupted-verb line must precede
            # it). Correct copy-scan: line 1 passes (span deleted from the
            # COPY only), line 2 refuses, buf[] emits VERBATIM, and the raw
            # scan sees the interrupted text (no verb bigram) -> ALLOW. An
            # in-place-mutation bug deletes ${A} from buf[] itself,
            # reassembling the verb bigram -> BLOCK -> this fixture fails
            # loud. Verified ALLOW both pre-fix and post-fix (2026-07-18), so
            # it discriminates ONLY the copy contract, not the narrowing.
            "cat > /tmp/note.md <<EOF\nhow to revert: git rest${A}ore .\nvalue is ${B@P}\nEOF",
            id="M1L-copy_contract_discriminator",
        ),
        pytest.param(
            # Escaped-dollar deliberate-flip pin (#1501): \${NAME} was BLOCKED
            # pre-fix (blanket ${ arm) and is ALLOWED post-fix — the deletion
            # regex matches the ${NAME} substring after the backslash; sound
            # because under an unquoted tag \$ suppresses expansion entirely
            # (literal text). Named in the script's gap-(xiii) ledger.
            "cat > /tmp/note.md <<EOF\nvalue is \\${NAME}\nhow to revert: git restore .\nEOF",
            id="M1m-escaped_dollar_plain_span",
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
# the strip refuses such bodies. Non-plain `${...}` forms refuse too
# (parameter expansion can nest command substitution, and `${V@P}` executes
# value-borne command substitution at feed time); plain `${NAME}` spans are
# deleted from a scan COPY before the refusal as of #1501 — the former
# fail-closed over-match on plain references now lives in the ALLOW matrix
# (M1d, moved there verbatim). The MUST-FIX-1 matrix.
# ---------------------------------------------------------------------------
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
            "cat > /tmp/note.md <<EOF\nvalue is ${SOMEVAR:-x}\nhow to revert: git restore .\nEOF",
            id="M1d2-param_fallback_form_still_refuses",
        ),
        pytest.param(
            # ${V@P} is the live-verified feed-time execution vector (prompt
            # expansion of the variable's VALUE; promptvars is on by default,
            # non-interactive included) — the reason check (g) keeps refusing
            # every non-plain ${...} form (#1501).
            "cat > /tmp/note.md <<EOF\nvalue is ${SOMEVAR@P}\nhow to revert: git restore .\nEOF",
            id="M1d3-prompt_transform_still_refuses",
        ),
        pytest.param(
            # Pins that plain-span deletion cannot MASK the $( refusal: a plain
            # span immediately followed by M1a's substitution text.
            "cat > /tmp/note.md <<EOF\n"
            "${A}$(git checkout -b evil)\n"
            "how to revert: git restore .\n"
            "EOF",
            id="M1d4-plain_span_adjacent_to_cmdsub",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\n"
            "value is ${SOMEVAR and more\n"
            "how to revert: git restore .\n"
            "EOF",
            id="M1d5-unclosed_brace_still_refuses",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\ncount is $((1 + 1))\nhow to revert: git restore .\nEOF",
            id="M1d6-arithmetic_still_refuses",
        ),
        pytest.param(
            "cat > /tmp/note.md <<EOF\narg is ${1}\nhow to revert: git restore .\nEOF",
            id="M1d7-positional_param_still_refuses",
        ),
        pytest.param(
            # Dash-tag block-side mirror of M1k: tag-form independence holds on
            # the refusal side too (the check-(g) edit sits inside if (!QUOTED)
            # with no tag-form consult; pinned rather than assumed).
            "cat > /tmp/note.md <<-EOF\n"
            "\tvalue is ${SOMEVAR@P}\n"
            "\thow to revert: git restore .\n"
            "\tEOF",
            id="M1d8-dash_tag_transform_still_refuses",
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


# ---------------------------------------------------------------------------
# #1621 — check (f) argv-list carve: argv-LIST-form call opens with NON-SHELL
# first elements (`subprocess.run(["git", ...`) are deleted from a per-line
# scan COPY before the shell-out refusal scan, so plan/doc heredoc bodies
# embedding argv-form subprocess git TEXT strip cleanly (guard class (xi):
# the argv form never classifies — comma-separated list tokens carry no
# `git <verb>` bigram; incident abee1289). Two fail-closed arms pin the
# loopholes the carve would otherwise open: an argv head naming a shell
# (bare / path-qualified / env, incl. the literal backslash-n bracket-gap
# spelling) refuses PRE-deletion, and a `shell=True` residual refuses
# post-deletion. Red-before/green-after for each allow fixture was confirmed
# at compose time against the origin/main guard (rc=2 -> 0).
#
# NOTE on the doc-line fixtures: they quote a BARE branch-create recipe (no
# `git -C` prefix) because the pre-existing PATH-BLIND `-C` per-clause
# waiver (#1128/#1193) waives a `-C`-prefixed checkout clause under BOTH
# guards — a `-C`-spelled doc line cannot satisfy the red-before protocol.
# ---------------------------------------------------------------------------
# The incident's refusing body line carried FOUR argv-call opens on ONE
# physical line, one with a literal backslash-n between paren and bracket
# (a plan-patch python script whose replacement text embeds a test snippet).
_I1621_ABEE_LINE51 = (
    'snippet = \'subprocess.run(["git", "-C", wt, "fetch"], check=True); '
    'subprocess.run(["git", "-C", wt, "status"], check=True); '
    'subprocess.check_call(["git", "log"]); '
    'subprocess.run(\\n    ["git", "-C", wt, "push"], check=True)\''
)
_I1621_ABEE_TRIMMED = (
    "uv run python - <<'PY'\n"
    + _I1621_ABEE_LINE51
    + "\n"
    + 'doc = f"git -C wtpath fetch origin && git checkout -B {branch} origin/main"\n'
    + "PY"
)


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "uv run python - <<'PY'\n"
            'subprocess.run(["git", "-C", wt, "push"], check=True)\n'
            'print("recipe: git checkout -B mybranch origin/main")\n'
            "PY",
            id="AV1-quoted_tag_argv_call_plus_doc_line",
        ),
        pytest.param(_I1621_ABEE_TRIMMED, id="AV2-abee_shaped_line51_multiplicity"),
        pytest.param(
            "uv run python - <<PY\n"
            'subprocess.run(["git", "-C", "/tmp/w", "push"], check=True)\n'
            'print("recipe: git checkout -B mybranch origin/main")\n'
            "PY",
            id="AV3-unquoted_tag_expansion_free",
        ),
    ],
)
def test_heredoc_argv_subprocess_body_strips(cmd):
    """#1621: argv-list-form call text with non-shell heads no longer refuses
    the strip; the stripped body's quoted recipe text never classifies."""
    assert _run(cmd) == 0


def test_heredoc_argv_shell_head_still_blocks():
    """New arm 1: an argv LIST whose first element names a shell refuses the
    strip PRE-deletion (no import line needed); the gated text classifies."""
    cmd = 'uv run python - <<\'PY\'\nPopen(["bash", "-c", "git reset --hard"])\nPY'
    assert _run(cmd) == 2


def test_heredoc_argv_shell_true_still_blocks():
    """New arm 2: a `shell=True` residual next to an argv call refuses the
    strip post-deletion; the gated single-string argv classifies."""
    cmd = "uv run python - <<'PY'\nrun([\"git checkout -b x\"], shell=True)\nPY"
    assert _run(cmd) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            'uv run python - <<\'PY\'\nrun(["/bin/bash", "-c", "git checkout -b x"])\nPY',
            id="SH1-path_qualified_head",
        ),
        pytest.param(
            'uv run python - <<\'PY\'\nrun(["env", "bash", "-c", "git checkout -b x"])\nPY',
            id="SH2-env_head",
        ),
    ],
)
def test_heredoc_argv_pathqualified_shell_head_still_blocks(cmd):
    """Arm 1 r2 widening: path-qualified ("/bin/bash") and "env" argv heads
    refuse the strip exactly like bare shell names."""
    assert _run(cmd) == 2


def test_heredoc_argv_newline_bracket_shell_head_still_blocks():
    """Arm 1 r2 widening: a literal backslash-n between bracket and quoted
    shell head cannot dodge the arm (its gap tolerance mirrors the deletion
    regex, so no shape the carve strips escapes the shell-head refusal)."""
    cmd = 'uv run python - <<\'PY\'\ns = \'Popen([\\n    "bash", "-c", "git reset --hard"])\'\nPY'
    assert _run(cmd) == 2


def test_heredoc_bare_shellout_mention_still_blocks():
    """Fail-closed boundary: a BARE shell-out-word prose mention (no argv call
    open to carve) still refuses the strip, so the co-occurring quoted recipe
    classifies (the M4b two-line value-indirection block requires this)."""
    cmd = (
        "uv run python - <<'PY'\n"
        'print("we call subprocess here")\n'
        'print("recipe: git checkout -B mybranch origin/main")\n'
        "PY"
    )
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
# side (a waived clause) must pass in either repo state; the
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
        pytest.param(
            # The former N12-wrapped_ssh_not_waived block fixture, flipped
            # by #1859: the literal `timeout <num>` wrapper is accepted on
            # the ssh head (parity with the gcloud arm's GS4).
            "timeout 240 ssh pod-779 'git reset --hard'",
            id="S15-timeout_wrapped_ssh_waived",
        ),
        pytest.param(
            # Fractional + suffixed duration — pins the full
            # `<num>[.frac][smhd]?` shape of the wrapper grammar on the
            # ssh arm (#1859).
            "timeout 1.5m ssh pod-779 'git reset --hard origin/main'",
            id="S16-timeout_fractional_suffixed_wrapped_ssh_waived",
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


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # the REAL 2026-07-18T17:00:40Z incident command, verbatim from
            # transcript 047b62be (rc=2 pre-fix; the motivating case)
            'grep -n -m 3 -A 20 "test_worktree_revert_shapes_block\\[git reset --hard\\]"'
            " /tmp/step9c-pytest-issue-1513.log | head -60",
            id="GP1-piped_grep_head_real_incident",
        ),
        pytest.param(
            # the FILED task-body shape (unpiped, trailing file) — already
            # rc=0 on main; regression-pins the body's literal Goal claim
            'grep -n -m 3 -A 20 "test_worktree_revert_shapes_block[git reset --hard]"'
            " /tmp/step9c-pytest-issue-1513.log",
            id="GP0-unpiped_trailing_file_filed_shape_pin",
        ),
        pytest.param("grep -rn 'git checkout -b' scripts/ | tail -20", id="GP2-piped_grep_tail"),
        pytest.param("grep 'git clean -fd' notes.md | wc -l", id="GP3-piped_grep_wc"),
        pytest.param(
            "grep 'git reset --hard' f.log | grep -v test | head -5",
            id="GP4-multistage_filter_chain",
        ),
        pytest.param(
            "grep -o 'git rebase' run.log | sort | uniq -c", id="GP5-sort_uniq_count_idiom"
        ),
        pytest.param("rg 'git restore' .claude/rules/ | head -30", id="GP6-piped_rg_head"),
        pytest.param(
            "grep 'git reset --hard' f.log | head -60 && echo done",
            id="GP7-chain_terminates_before_AND",
        ),
    ],
)
def test_grep_pipe_readonly_sink_chain_waived(cmd):
    """(#1538) A grep-family pattern clause piped into a VERIFIED read-only sink
    chain (allowlisted stdin->stdout text filters, no expansion / redirect /
    write-exec channel, chain ends on a non-PIPE non-BG seam) is waived."""
    assert _run(cmd) == 0


@pytest.mark.parametrize(
    "cmd",
    [
        # (#1538) Adversarial consumer-chain shapes: the pipe widening must
        # NOT waive any of these (all rc=2 both pre- and post-fix). Each pins
        # one refusal arm of _pipe_chain_is_readonly_sink() (or an outer
        # waiver arm the widening deliberately left untouched).
        pytest.param("grep 'git reset --hard' f | bash", id="GPN1-pipe_to_shell"),
        pytest.param("grep 'git reset --hard' f | head -1 | sh", id="GPN2-shell_at_second_stage"),
        pytest.param("grep 'git reset --hard' f | xargs -I{} bash -c '{}'", id="GPN3-xargs_exec"),
        pytest.param(
            "grep 'git reset --hard' f | head -1 > /tmp/x.sh", id="GPN4-consumer_output_redirect"
        ),
        pytest.param(
            "grep 'git reset --hard' f | head -n $(cat n.txt)",
            id="GPN5-consumer_command_substitution",
        ),
        pytest.param("grep 'git reset --hard' f | sort -o /tmp/x.sh", id="GPN6-sort_output_flag"),
        pytest.param(
            "grep 'git reset --hard' f | sort -ro /tmp/x.sh", id="GPN6b-sort_output_bundled_short"
        ),
        pytest.param(
            "grep 'git reset --hard' f | sort --compress-program=bash",
            id="GPN6c-sort_compress_program_exec",
        ),
        pytest.param(
            "grep 'git reset --hard' f | sort -T /tmp/spill", id="GPN6d-sort_tempdir_spill_write"
        ),
        pytest.param(
            "grep 'git reset --hard' f | uniq - /tmp/x.sh", id="GPN7-uniq_positional_output"
        ),
        pytest.param("grep 'git reset --hard' f | rg --pre bash x", id="GPN8-rg_pre_consumer"),
        pytest.param("grep 'git reset --hard' f | rg -z pattern", id="GPN8b-rg_search_zip_exec"),
        pytest.param("grep 'git reset --hard' f | tee /tmp/x.sh", id="GPN9-tee_consumer"),
        pytest.param(
            "grep safe_pattern f | head -5 && git reset --hard",
            id="GPN10-unquoted_verb_sibling_clause",
        ),
        pytest.param(
            "grep 'git reset --hard' f | head -60 &", id="GPN11-trailing_background_chain"
        ),
        pytest.param("grep 'git reset --hard' f | ", id="GPN12-trailing_empty_pipe"),
        pytest.param(
            "grep 'git reset --hard' <(cat f) | head", id="GPN13-producer_process_substitution"
        ),
        pytest.param("ssh pod-1 'git reset --hard' | head", id="GPN14-ssh_pipe_refusal_unchanged"),
        pytest.param(
            "grep 'git reset --hard' f > results.txt", id="GPN15-producer_redirect_after_pattern"
        ),
        pytest.param("grep 'git reset --hard' f | sed -n '1,5p'", id="GPN16-sed_excluded_consumer"),
        pytest.param(
            "grep 'git reset --hard' f | /usr/bin/head -5", id="GPN17-path_spelled_consumer"
        ),
        pytest.param(
            "grep 'git reset --hard' f 2>&1 | head -5", id="GPN18-fd_dup_missplit_residual"
        ),
        # --- v3 additions (Phase-2 critique fold-in) ---
        pytest.param(
            "grep 'git reset --hard' f | sort --output /tmp/x.sh", id="GPN6e-sort_output_long_form"
        ),
        pytest.param(
            "grep 'git reset --hard' f | sort --temporary-directory=/tmp/spill",
            id="GPN6f-sort_tempdir_long_form",
        ),
        pytest.param(
            "grep 'git reset --hard' f | rg --hostname-bin=bash pat",
            id="GPN8c-rg_hostname_bin_consumer",
        ),
        pytest.param(
            "grep 'git reset --hard' f | rg --search-zip pat", id="GPN8d-rg_search_zip_long_form"
        ),
        pytest.param(
            "grep 'git reset --hard' f | rg -zi pat", id="GPN8e-rg_bundled_short_flag_mid_bundle"
        ),
        pytest.param(
            "grep 'git reset --hard' f | sort $SORT_FLAGS",
            id="GPN19-consumer_bare_variable_expansion",
        ),
        pytest.param(
            "grep 'git reset --hard' f | rg 'x #' --pre bash y",
            id="GPN20-consumer_quoted_hash_before_channel_flag",
        ),
        pytest.param("grep 'git reset --hard' f | awk '{print}'", id="GPN21-awk_excluded_consumer"),
        pytest.param("grep 'git reset --hard' f | # count later", id="GPN22-comment_only_consumer"),
        pytest.param("grep 'git reset --hard' f | \"head\" -5", id="GPN23-quoted_consumer_word"),
    ],
)
def test_grep_pipe_unsafe_consumer_blocks(cmd):
    """(#1538) Any pipe chain the read-only-sink walker cannot POSITIVELY verify
    keeps blocking: shell/exec consumers, write channels (redirects + per-word
    output/exec flags in every spelling), any '$' or '#' in a consumer clause,
    off-allowlist / path-spelled / quoted consumer words, trailing BG or empty
    pipe, producer-side procsub/redirect, and the (unwidened) ssh pipe arm."""
    assert _run(cmd) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        # N1-multi_statement_mis_split_residual_fp MOVED (#1413) to the
        # masking allow list below as M9-former_N1_residual_closed: N1 was
        # the pin OF the residual-(xiv) mis-split false positive that the
        # mask_ssh_payload_separators() pre-pass closes for the canonical
        # single-quoted shape — the flip is Goal-mandated, not a regression.
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
        # N12-wrapped_ssh_not_waived MOVED (#1859) to the single-statement
        # waiver allow list above as S15-timeout_wrapped_ssh_waived: the
        # literal `timeout <num>[.frac][smhd]?` wrapper is now accepted on
        # the ssh head (parity with the #1463 gcloud arm's GS4) — the flip
        # is Goal-mandated, not a regression. Non-timeout wrappers keep
        # their blocked disposition (N36-N43 below).
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
        # N36-N43 pin the #1859 ssh-arm timeout-wrapper boundaries (mirrors
        # of the gcloud arm's GN-series pins): ONLY the literal
        # `timeout <num>[.frac][smhd]?` prefix is tolerated, and the
        # wrapper lifts NONE of the waiver's refusal arms.
        pytest.param(
            # Flag form NOT waived (mirror GN14).
            "timeout --signal=KILL 120 ssh pod-779 'git reset --hard'",
            id="N36-timeout_flag_form_not_waived",
        ),
        pytest.param(
            # `-k` flag form NOT waived (mirror GN8's beyond-timeout class).
            "timeout -k 5 240 ssh pod-779 'git reset --hard'",
            id="N37-timeout_k_flag_form_not_waived",
        ),
        pytest.param(
            # The wrapped head does NOT lift the consumer-independent
            # PIPE-producer refusal (parity with N23 / GN7).
            "timeout 240 ssh pod-779 'git reset --hard' | tail -5",
            id="N38-wrapped_head_pipe_producer_still_blocks",
        ),
        pytest.param(
            # The wrapped head does NOT lift the shared-repo-path
            # never-waive (parity with N5).
            "timeout 240 ssh vm 'git"
            " --git-dir=/home/thomasjiralerspong/explore-persona-space/.git"
            " reset --hard'",
            id="N39-wrapped_head_repo_root_path_still_blocks",
        ),
        pytest.param(
            # The wrapped head does NOT lift the ProxyCommand local-exec
            # refusal (parity with N7/N16).
            "timeout 240 ssh -o ProxyCommand='git reset --hard' host 'git status'",
            id="N40-wrapped_head_proxycommand_still_blocks",
        ),
        pytest.param(
            # The numeric group is REQUIRED — a bare `timeout ssh ...` head
            # is not the literal wrapper shape.
            "timeout ssh pod-779 'git reset --hard'",
            id="N41-timeout_without_duration_not_waived",
        ),
        pytest.param(
            "nohup ssh pod-779 'git reset --hard'",
            id="N42-nohup_wrapped_ssh_not_waived",
        ),
        pytest.param(
            "/usr/bin/ssh pod-779 'git reset --hard'",
            id="N43-abs_path_ssh_single_statement_not_waived",
        ),
    ],
)
def test_remote_waiver_fail_closed_blocks(cmd):
    """Every waiver refusal arm keeps its locally-executing lookalike at exit 2."""
    assert _run(cmd) == 2


# ==== #1413 — ssh multi-statement single-quoted payload masking =============
#
# mask_ssh_payload_separators() neutralizes the separators INSIDE the
# balanced single-quoted FINAL argument of a clause-initial ssh clause, so
# the canonical multi-statement remote string reaches the #1098 waiver as
# ONE clause instead of mis-splitting (residual (xiv)'s first arm, closed
# for that shape; founding incident #779). Masking fires ONLY when the whole
# candidate passes the 8-arm fail-closed refusal predicate — R1
# whitespace-only tail to ;/&&/||/NL/EOS; R2 no expansion/redirect/sentinel
# char; R3 no ssh local-exec option token; R4 no shared-repo path spelling;
# R5 no quote char before the candidate; R6/R7 no cd + /tmp/ or
# .claude/worktrees/ latch vocabulary in candidate/prefix; R8 no WT= text —
# and ANY refusal leaves the input byte-identical, so every refused shape
# keeps today's disposition (all 27 original NM fixtures were verified rc=2
# against the UNMODIFIED guard before the mask landed — the pre-change
# red-team gate; NM5 flipped to the M10 positive at #1859).
# The allow side (a masked-and-waived clause) must pass in
# either repo state, matching the #1098 convention.
# Where a predicate arm overlaps a #1098 ladder refusal, the block fixture
# uses an allow-arm-anchored CONTAMINATION payload: a mid-payload
# `git switch feature-x` (whose ONLY detector carries the end-anchored
# `switch main` allow-arm) followed by a payload-FINAL `git switch main` —
# under a single dropped predicate arm the merged clause reaches
# classify_clause and the trailing quote-tolerant allow-arm swallows the
# block, turning the fixture RED. A destructive payload cannot detect that
# bug class: the reset/checkout detectors span masked text via their
# [^;&|]* gaps and stay green, so those fixtures serve as disposition pins
# only.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "ssh pod-77965 'cd /workspace/explore-persona-space && git fetch origin"
            " && git checkout FETCH_HEAD -- scripts/'",
            id="M1-incident_779_fetch_head_checkout",
        ),
        pytest.param(
            "ssh pod-779 'cd /workspace/x; git reset --hard origin/main'",
            id="M2-semicolon_chain",
        ),
        pytest.param(
            "ssh pod-779 'cd /workspace/x && git checkout -b scratch'",
            id="M3-verb_in_tail_statement",
        ),
        pytest.param("ssh pod-779 'git fetch || git reset --hard'", id="M4-or_connector_payload"),
        pytest.param(
            "ssh pod-779 'git fetch origin\ngit checkout FETCH_HEAD'",
            id="M5-newline_in_payload",
        ),
        pytest.param(
            "ssh pod-779 'cd /w && git reset --hard' && echo done",
            id="M6-benign_local_tail_and",
        ),
        pytest.param(
            # Prefix is quote- and latch-clean, so the SEQ-preceded candidate
            # still masks (clause-initial tracking mirrors the splitter).
            "echo starting; ssh pod-779 'cd /w && git reset --hard'",
            id="M7-clause_initial_after_seq",
        ),
        pytest.param(
            "ssh -p 40052 root@157.157.221.29 'cd /workspace/x && git checkout FETCH_HEAD'",
            id="M8-options_host_head",
        ),
        pytest.param(
            # The former N1-multi_statement_mis_split_residual_fp block
            # fixture: its (xiv) mis-split residual is what #1413 closes.
            # /workspace/explore-persona-space matches NONE of cond (4)'s
            # covered shared-repo spellings, so the merged clause waives.
            "ssh pod-779 'cd /workspace/explore-persona-space && git reset --hard origin/main'",
            id="M9-former_N1_residual_closed",
        ),
        pytest.param(
            # The former NM5-wrapped_ssh_multi_statement block fixture,
            # flipped by #1859: the mask candidate head accepts the literal
            # `timeout <num>` wrapper on the ssh arm (parity with the
            # #1463 gcloud arm's GM fixtures).
            "timeout 240 ssh pod-779 'cd /w && git reset --hard'",
            id="M10-timeout_wrapped_multi_statement",
        ),
    ],
)
def test_ssh_multi_statement_payload_masking_allows(cmd):
    """Canonical single-quoted multi-statement ssh payloads mask and waive (#779)."""
    assert _run(cmd) == 0


@pytest.mark.parametrize(
    "cmd",
    [
        # (#1710) The historical NM1 double-quoted-ssh block is REPLACED by
        # the new NDS positive tests (double-quoted allow) + NDS_R* negative
        # tests (double-quoted refusal ladder) added below. The
        # double-quoted payload was documented as a residual under the
        # #1413 mask; Arm 1's R9 no-expansion refusal admits it now.
        pytest.param(
            # Constraint-1 ambiguity (no closing quote) stays blocked.
            "ssh pod-779 'cd /workspace/x && git reset --hard origin/main",
            id="NM2-unbalanced_quote_fail_closed",
        ),
        pytest.param(
            # R4 disposition pin: the $HOME/ shared-repo spelling never masks.
            "ssh vm 'cd $HOME/explore-persona-space && git reset --hard'",
            id="NM3-repo_path_home_spelling",
        ),
        pytest.param(
            # R4 disposition pin: the literal shared-repo spelling never masks.
            "ssh vm 'cd /home/thomasjiralerspong/explore-persona-space && git reset --hard'",
            id="NM4-repo_path_literal_spelling",
        ),
        # NM5-wrapped_ssh_multi_statement MOVED (#1859) to the masking
        # allow list above as M10-timeout_wrapped_multi_statement — the
        # literal `timeout <num>` wrapper is accepted on the ssh mask
        # candidate head (parity with the #1463 gcloud arm). Non-timeout
        # wrappers keep blocking (NM6/NM7 below; N36-N43 single-statement).
        pytest.param(
            "$SSHCMD pod-779 'cd /w && git reset --hard'",
            id="NM6-variable_ssh_multi_statement",
        ),
        pytest.param(
            "/usr/bin/ssh pod-779 'cd /w && git reset --hard'",
            id="NM7-abs_path_ssh_multi_statement",
        ),
        pytest.param(
            # R1 disposition pin: pipeline-producer position stays classifying.
            "ssh host 'echo x && git reset --hard' | bash",
            id="NM8-pipe_after_candidate",
        ),
        pytest.param(
            # R1 disposition pin: background position stays classifying.
            "ssh host 'cd /w && git reset --hard' &",
            id="NM9-bg_after_candidate",
        ),
        pytest.param(
            # Payload `>` refuses via R2 (remote-redirect residual, parity
            # with N29); the /tmp/x path alone does not trip R6's
            # conjunction — no `cd` before it — the `>` arm is what refuses.
            "ssh pod-779 'git status > /tmp/x && git checkout -b scratch'",
            id="NM10-redirect_in_payload",
        ),
        pytest.param(
            "ssh pod-779 'cd /w && git reset --hard' -v",
            id="NM11-trailing_token_after_quote",
        ),
        pytest.param(
            # Parity with N4: the deliberate ${ over-match refuses.
            "ssh pod-779 'cd /w && git reset --hard ${REF}'",
            id="NM12-brace_expansion_in_payload",
        ),
        pytest.param(
            # Parity with N17; plain input-`<` shares R2's blanket `<` branch.
            "ssh pod-779 'cd /w && git reset --hard' <<< input",
            id="NM13-here_string_tail",
        ),
        pytest.param(
            # Parity with N7: local-exec option token refuses (R3).
            "ssh -o ProxyCommand=evil host 'cd /w && git reset --hard'",
            id="NM14-proxycommand_head_multi_statement",
        ),
        pytest.param(
            # Masking is clause-local: the LOCAL tail still blocks (N11 parity).
            "ssh pod-779 'cd /w && git fetch'; git reset --hard",
            id="NM15-local_gated_clause_after_masked",
        ),
        pytest.param(
            # ~/ shared-repo spelling as a CONTAMINATION shape: today the
            # mid-payload `git switch feature-x` fragment blocks; a dropped
            # ~/ arm masks -> the merged clause repo-path-glob classifies ->
            # the trailing `switch main` allow-arm would swallow -> RED.
            "ssh vm 'cd ~/explore-persona-space; git switch feature-x; git switch main'",
            id="NM16-repo_path_tilde_contamination",
        ),
        pytest.param(
            # The LOCAL head clause still blocks (a scanner bug swallowing
            # the prefix goes red).
            "git reset --hard; ssh pod-779 'cd /workspace/x && git fetch'",
            id="NM17-local_gated_clause_before_masked",
        ),
        pytest.param(
            # Pins R2's $( branch with contamination power.
            "ssh pod-779 'echo $(hostname); git switch feature-x; git switch main'",
            id="NM18-cmdsub_payload_contamination",
        ),
        pytest.param(
            # Pins R2's backtick branch with contamination power.
            "ssh pod-779 'echo `hostname`; git switch feature-x; git switch main'",
            id="NM19-backtick_payload_contamination",
        ),
        pytest.param(
            # Today the mid-payload gated fragment blocks; a broken R1 masks
            # -> merged clause has nextsep=PIPE -> the ladder refuses the
            # waiver -> classifies MERGED text -> the trailing `switch main`
            # allow-arm would swallow -> RED (the exact contamination
            # mechanism the mask-then-let-ladder-refuse design rejects).
            "ssh host 'echo x; git switch feature-x; git switch main' | bash",
            id="NM20-pipe_position_contamination",
        ),
        pytest.param(
            # BG-position twin of NM20.
            "ssh host 'echo x; git switch feature-x; git switch main' &",
            id="NM21-bg_position_contamination",
        ),
        pytest.param(
            # Today the payload's `;` resets `scoped` and the && tail
            # BLOCKS; without R6 the merged clause would match the
            # pre-waiver `cd +/tmp/` grep -> scoped=1; continue -> the LOCAL
            # tail is skipped -> RED.
            "ssh pod-779 'cd /tmp/scratch; git fetch' && git reset --hard",
            id="NM22-latch_arming_payload_and_tail",
        ),
        pytest.param(
            # `cd` and the worktrees path in DIFFERENT payload statements:
            # without R6 the mask removes the [;&|] chars guarding
            # `cd +[^;&|]*\.claude/worktrees/`, the widened grep matches the
            # merged clause, and the local tail is skipped -> RED.
            "ssh pod-779 'cd /data && echo .claude/worktrees/x' && git reset --hard",
            id="NM23-latch_regex_widening_split_payload",
        ),
        pytest.param(
            # `scoped` armed by the LOCAL prefix; today the payload's `;`
            # resets it and the tail BLOCKS; without R7 the merged clause
            # rides scoped=1 and the tail is skipped -> RED.
            "cd /tmp/scratch && ssh pod-779 'git fetch; git status' && git reset --hard",
            id="NM24-prefix_latch_then_masked_and_tail",
        ),
        pytest.param(
            # The scanner's "payload" would be live LOCAL code between two
            # strings; today the local gated clause BLOCKS; without R5 it is
            # masked into a waived ^ssh clause -> RED.
            "echo 'x; ssh h '; git switch feature-x; echo '; y'",
            id="NM25-preopened_single_quote_swallow",
        ),
        pytest.param(
            # Double-quote variant: raw-apostrophe-parity would pass it; the
            # strict any-quote R5 refuses -> without R5, RED.
            'echo "x; ssh h \'"; git switch feature-x; echo "\';"',
            id="NM26-preopened_double_quote_swallow",
        ),
        pytest.param(
            # Today the payload's clause-initial WT=x' fragment DISARMS
            # wt_bound and the tail BLOCKS; without R8 the disarm is
            # suppressed, cd "$WT" latches, and the tail is skipped -> RED.
            "WT=.claude/worktrees/w; ssh pod-779 'echo a; WT=x'; cd \"$WT\" && git checkout -b tmp",
            id="NM27-wt_payload_disarm_suppression",
        ),
    ],
)
def test_ssh_masking_refusal_ladder_blocks(cmd):
    """Every mask-predicate refusal arm leaves its shape byte-identical (exit 2).

    All 27 fixtures were verified rc=2 against the UNMODIFIED guard before
    the mask landed (the plan's pre-change red-team gate), so each pins a
    today-blocked disposition the monotonicity invariant quantifies over.
    """
    assert _run(cmd) == 2


# ==== #1443 — clause-initial anchoring of the cd-scope latch ================
#
# The driver's literal-path cd-scope latch greps are ^-anchored (#1443): only
# a clause whose COMMAND WORD is `cd` arms `scoped`. Latch vocabulary buried
# mid-clause — a quoted ssh remote-payload fragment ahead of an internal `&&`
# (the mask's R6 arm deliberately leaves such payloads byte-identical, so
# they mis-split), or an echo'd string — must never mutate local scoping
# state. The splitter strips leading whitespace from every clause
# (split_and_label, guard L833), so post-split `&& cd /tmp/...` clauses stay
# clause-initial and keep arming. All four block fixtures were verified rc=0
# against the pre-#1443 guard (the fail-open this section closes); both
# allow fixtures were verified rc=0 pre-fix and must stay rc=0.


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "ssh pod-779 'cd /tmp/x && git reset --hard'",
            id="L1-ssh_payload_tmp_fragment_no_latch",
        ),
        pytest.param(
            "ssh pod-779 'cd .claude/worktrees/w && git checkout -b tmp'",
            id="L2-ssh_payload_worktrees_fragment_no_latch",
        ),
        pytest.param(
            "echo 'cd /tmp/x' && git reset --hard",
            id="L3-echo_quoted_tmp_text_no_latch",
        ),
        pytest.param(
            "echo 'cd .claude/worktrees/x' && git restore .",
            id="L4-echo_quoted_worktrees_text_no_latch",
        ),
    ],
)
def test_mid_clause_cd_text_does_not_latch(cmd):
    """Latch vocabulary mid-clause (payload/prose) never arms local scope."""
    assert _run(cmd) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "mkdir -p /tmp/s && cd /tmp/s && git clean -fd",
            id="L5-post_split_clause_initial_tmp_arms",
        ),
        pytest.param(
            'cd "$ROOT/.claude/worktrees/issue-9" && git restore .',
            id="L6-quoted_prefix_worktrees_arms",
        ),
    ],
)
def test_clause_initial_cd_still_arms(cmd):
    """Anchoring must not regress the intended clause-initial latch allows."""
    assert _run(cmd) == 0


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
# block side classifies against the pinned always-on-main repo; the
# allow side must pass in either repo state. Block tests fire on shape alone
# — the detector is pure grep, never routing to the ref-resolution probes.
# ---------------------------------------------------------------------------
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


def test_note_text_merge_literal_double_quoted_now_allows():
    # (#1710) Sibling of test_note_text_git_verb_literal_double_quoted_now_allows:
    # the taskpy mask's P7 double-quoted extension covers `git merge <ref>`
    # prose inside a `--note` body under the same no-expansion refusal
    # ladder. Workaround (--file / -F) still WORKS but is no longer
    # REQUIRED for these shapes.
    assert (
        _run(
            "uv run python scripts/task.py post-marker 1128 epm:x "
            '--note "run git merge issue-1 next"'
        )
        == 0
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
# operation on the root tree — the M5 decision, mirrored). The block side
# classifies against the pinned always-on-main repo; the allow side must
# pass in either repo state. Block tests fire on shape alone — the detectors
# are pure grep, never routing to the ref-resolution probes.
# ---------------------------------------------------------------------------
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
def test_note_text_rebase_family_literal_double_quoted_now_allows(note_cmd):
    # (#1710) Sibling of test_note_text_merge_literal_double_quoted_now_allows:
    # the taskpy mask's P7 double-quoted extension covers
    # `git rebase <ref>` / `git cherry-pick <sha>` prose inside a `--note`
    # body under the same no-expansion refusal ladder. Workaround (--file /
    # -F) still WORKS but is no longer REQUIRED for these shapes.
    assert _run(note_cmd) == 0


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
# abort/quit-only parity, register (xviii)(a)). The block side classifies
# against the pinned always-on-main repo; the allow side must pass in
# either repo state. Block tests fire on shape alone — the detectors are pure
# grep, never routing to the ref-resolution probes.
# ---------------------------------------------------------------------------
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
def test_note_text_revert_am_literal_double_quoted_now_allows(note_cmd):
    # (#1710) Sibling of test_note_text_rebase_family_literal_double_quoted_now_allows:
    # the taskpy mask's P7 double-quoted extension covers `git revert <sha>` /
    # `git am <path>` prose inside a `--note` body under the same
    # no-expansion refusal ladder. Workaround (--file / -F) still WORKS but
    # is no longer REQUIRED for these shapes.
    assert _run(note_cmd) == 0


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


# ==== #1463 — gcloud compute ssh remote-payload waiver + masking ============
#
# `gcloud compute ssh <instance> --command='<payload>'` is a thin wrapper
# around the local ssh(1) binary that executes its payload ON THE INSTANCE
# (SDK 576.0.0 help; probed live via --dry-run: trailing `-- ARGS`
# positionals land after the host in the constructed local ssh argv, i.e.
# they ride as the REMOTE command). As of #1463 the driver-loop waiver
# (cond (1)) and the mask pre-pass gain a `gcloud compute ssh` head — with
# an optional literal `timeout <num>[.frac][smhd]?` wrapper, extended to
# the bare `ssh` head by #1859 — routed through the SAME ssh refusal arms
# (waiver conds
# (2)/(3)/(3b)/(4); mask R1-R8). Founding incident: #825
# (2026-07-16T13:18:53Z false block); #1336 hit the ssh variant pre-#1413.
# The GN-series pins fail-closed dispositions (all verified rc=2 against
# the UNMODIFIED guard before the arm landed — the pre-change red-team
# gate); GS8 pins the pre-existing path-blind `git -C` waiver (guard's
# per-clause -C allow fires BEFORE this arm) that already allowed the
# incident session's executed sibling command.
# ---------------------------------------------------------------------------
# TEST-FIXTURE FENCE — gated command literals below are guard test INPUTS
# only (they drive the hook subprocess; nothing here executes).
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "gcloud compute ssh eps-issue-825 --configuration=eps-gcp"
            " --zone=us-central1-c --command='git checkout -b scratch'",
            id="GS1-single_statement_fenced_verb",
        ),
        pytest.param(
            'gcloud compute ssh pod --zone=us-central1-a --command="git reset --hard origin/main"',
            id="GS2-command_equals_double_quoted",
        ),
        pytest.param(
            "gcloud compute ssh pod --zone us-central1-a --command 'git clean -fd'",
            id="GS3-command_space_form_and_space_zone",
        ),
        pytest.param(
            "timeout 120 gcloud compute ssh pod --zone=us-central1-c --command='git reset --hard'",
            id="GS4-timeout_wrapped_single_statement",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='git restore .' 2>/dev/null",
            id="GS5-dev_null_redirect_exempt",
        ),
        pytest.param(
            "gcloud compute ssh pod --internal-ip --command='git reset --hard'",
            id="GS6-internal_ip_flag",
        ),
        pytest.param(
            # Probed live (--dry-run against a real instance, SDK 576.0.0):
            # the `-- ARGS` positionals land AFTER the host in the
            # constructed local ssh argv — i.e. they are the REMOTE command.
            "gcloud compute ssh pod --zone=us-central1-a -- git checkout -b scratch",
            id="GS7-passthrough_remote_positional",
        ),
        pytest.param(
            # Allowed TODAY via the path-blind `git -C` per-clause waiver
            # (fires before the #1463 arm); pins the block-message
            # remediation + the incident session's executed sibling shape.
            "gcloud compute ssh pod --command='sudo git -C /workspace/eps-issue-825"
            " merge --ff-only origin/main'",
            id="GS8-git_dash_C_payload_pre_existing_pin",
        ),
    ],
)
def test_gcloud_remote_git_clause_waiver_allows(cmd):
    """Single-statement gcloud remote payloads waive per-clause (#1463)."""
    assert _run(cmd) == 0


# TEST-FIXTURE FENCE — guard test INPUTS only.
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # The #825 13:18Z incident command, minimally remediated: same
            # repo-root cd prefix, same timeout wrapper, same head + flags,
            # same fenced merge statement; the in-payload redirects and the
            # outer local pipe dropped (GN1 pins the verbatim original as
            # still-blocked).
            "cd /home/thomasjiralerspong/explore-persona-space && timeout 120"
            " gcloud compute ssh eps-issue-825 --configuration=eps-gcp"
            " --zone=us-central1-c --command='set -e\n"
            "cd /workspace/eps-issue-825\n"
            "sudo git fetch origin issue-825\n"
            "sudo git -c safe.directory=/workspace/eps-issue-825 merge --ff-only"
            " origin/issue-825\n"
            'echo -n "HEAD="; sudo git rev-parse HEAD | head -c 12\'',
            id="GM1-incident_normalized_shape",
        ),
        pytest.param(
            "gcloud compute ssh pod --zone=us-central1-c"
            " --command='cd /workspace/x && git merge --ff-only origin/main'",
            id="GM2-amp_amp_payload",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='git merge --ff-only origin/main | tail -2'",
            id="GM3-pipe_inside_payload_masked",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='cd /w && git reset --hard' && echo done",
            id="GM4-benign_compound_tail",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='git fetch origin; git reset --hard origin/main'",
            id="GM5-seq_semicolon_payload",
        ),
    ],
)
def test_gcloud_multi_statement_payload_masking_allows(cmd):
    """Canonical single-quoted multi-statement gcloud payloads mask + waive (#1463)."""
    assert _run(cmd) == 0


# The verbatim #825 2026-07-16T13:18:53Z blocked command, recovered from the
# incident transcript's tool_use row (issue-825 session jsonl) — regenerated
# mechanically, never retyped. Stays BLOCKED: mask R2 refuses the in-payload
# `<`/`>` redirects (byte-identical -> mis-split -> the tail clause carrying
# the fenced merge statement classifies), and independently the outer
# `2>&1 | tail -20` puts the clause in BG/PIPE producer position (waiver
# cond (2)). GM1 is the sanctioned minimal remediation of this shape.
_GN1_VERBATIM_825 = (
    "cd /home/thomasjiralerspong/explore-persona-space && timeout 120 gcloud "
    "compute ssh eps-issue-825 --configuration=eps-gcp --zone=us-central1-c -"
    "-command='set -e\ncd /workspace/eps-issue-825\nsudo git fetch origin issue"
    "-825 2>&1 | tail -1\nsudo git -c safe.directory=/workspace/eps-issue-825 "
    'merge --ff-only origin/issue-825 2>&1 | tail -2\necho -n "HEAD="; sudo gi'
    't rev-parse HEAD | head -c 12; echo\necho -n "ancestry="; sudo git merge-'
    "base --is-ancestor d11695238a485a3992a49defe16180c6f6354e95 HEAD && echo"
    " OK || echo MISSING\n# capture env from frozen main process (root-only)\ns"
    'udo bash -c "tr \\"\\0\\" \\"\\n\\" < /proc/2771/environ > /workspace/eps-issu'
    "e-825/.eps-relaunch-env && chmod 600 /workspace/eps-issue-825/.eps-relau"
    'nch-env"\necho -n "env_keys="; sudo grep -cE "^(HF_TOKEN|ANTHROPIC_API_KE'
    'Y|WANDB_API_KEY|OPENAI_API_KEY|HF_XET|HF_HUB)" /workspace/eps-issue-825/'
    ".eps-relaunch-env || true\n# clear queued-not-done sentinels (NOT rollout"
    "_instruct_* — in flight under old workers)\nS=/workspace/eps-issue-825/da"
    'ta/issue_825/turn_dynamics/state\nsudo rm -f "$S/fit_gc_instruct.fail" "$'
    'S/fit_gc_instruct.queued" "$S/fit_gc_pretrained.queued" \\\n  "$S/fit_cell'
    's_armR_logged_instruct.queued" "$S/fit_cells_armR_logged_pretrained.queu'
    'ed" \\\n  "$S/fit_cells_armR_own_pretrained.queued" "$S/fit_transfer_armR_'
    'own_pretrained.queued" \\\n  "$S/fit_operators_armR_own_pretrained.queued"'
    ' "$S/fit_reach_armR_own_pretrained.queued" \\\n  "$S/upload_cap_armR_logge'
    'd_instruct.queued" "$S/upload_cap_armR_logged_pretrained.queued" \\\n  "$S'
    '/upload_cap_armR_own_pretrained.queued" "$S/upload_gen_pretrained.queued'
    '"\necho "sentinels_cleared"; sudo ls "$S" | grep -E "fail|queued" | grep '
    '-vE "rollout" || echo "(only rollout .queued remain among non-done)"\' 2>'
    "&1 | tail -20"
)


# TEST-FIXTURE FENCE — guard test INPUTS only.
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            _GN1_VERBATIM_825,
            id="GN1-verbatim_825_incident_pins_residual",
        ),
        pytest.param(
            'gcloud compute ssh pod --command="$(git reset --hard)"',
            id="GN2-cmdsub_local_exec",
        ),
        pytest.param(
            'gcloud compute ssh pod --command="git reset --hard ${REF}"',
            id="GN3-brace_expansion_overmatch",
        ),
        pytest.param(
            "gcloud compute ssh pod --ssh-flag='-o ProxyCommand=git reset --hard'"
            " --command='git status'",
            id="GN4-ssh_flag_proxycommand_local_exec",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='git status' -- -o ProxyCommand='git reset --hard'",
            id="GN5-passthrough_proxycommand_local_exec",
        ),
        pytest.param(
            # N5 mirror; uses --git-dir (NOT -C — the path-blind per-clause
            # -C waiver would fire first and mask the arm under test).
            "gcloud compute ssh cia-benchmark-vm --command='git"
            " --git-dir=/home/thomasjiralerspong/explore-persona-space/.git reset --hard'",
            id="GN6-repo_root_path_in_payload_ssh_to_self",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='git reset --hard' | tail -1",
            id="GN7-pipeline_producer_position",
        ),
        pytest.param(
            "nohup gcloud compute ssh pod --command='git reset --hard'",
            id="GN8-wrapped_beyond_timeout_not_waived",
        ),
        pytest.param(
            "gcloud beta compute ssh pod --command='git reset --hard'",
            id="GN9-release_track_not_waived",
        ),
        pytest.param(
            "gcloud compute ssh pod --command='git reset --hard' > /tmp/out.txt",
            id="GN10-file_redirect_refused",
        ),
        pytest.param(
            # mask R1 (trailing token after the payload) -> mis-split ->
            # the tail clause blocks. Remediation: put --command last.
            "gcloud compute ssh pod --command='cd /w && git reset --hard' --zone=us-central1-c",
            id="GN11-trailing_flag_after_payload_multi_statement",
        ),
        pytest.param(
            # mask R2 (redirect chars in candidate) -> mis-split -> blocks.
            "gcloud compute ssh pod --command='git fetch 2>&1 && git reset --hard'",
            id="GN12-in_payload_redirect_multi_statement",
        ),
        pytest.param(
            # mask R5 (quote before candidate) -> mis-split -> blocks.
            "echo \"note\" && gcloud compute ssh pod --command='cd /w && git reset --hard'",
            id="GN13-quoted_prefix_refuses_mask",
        ),
        pytest.param(
            # Only the literal `timeout <num>[.frac][smhd]?` wrapper is
            # tolerated; the flag forms are not (drop the flags or use the
            # canonical spelling).
            "timeout --signal=KILL 120 gcloud compute ssh pod --command='git reset --hard'",
            id="GN14-timeout_flag_form_not_waived",
        ),
        # (#1710) The historical GN15 gcloud-compute-ssh double-quoted-block
        # is REPLACED by NDS3-double_quoted_gcloud_compute_ssh (positive) +
        # NDS_R* refusal-ladder pins below. Arm 1's R9 no-expansion refusal
        # admits the double-quoted payload the same way it admits the bare
        # `ssh` head — the gcloud head is threaded through the SAME
        # mask_ssh_payload_separators branch.
    ],
)
def test_gcloud_waiver_fail_closed_blocks(cmd):
    """Every gcloud lookalike outside the narrow waiver keeps exit 2 (#1463).

    All 15 fixtures were verified rc=2 against the UNMODIFIED guard before
    the arm landed (the plan's pre-change red-team gate), so each pins a
    today-blocked disposition the additive-only claim quantifies over.
    """
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# Deny-event sidecar (#1528) — one best-effort JSON row per deny.
#
# Row schema: {ts, guard:"repo_root_branch", arm, len, head, clause_head};
# heads are printable-ASCII, masked (opaque runs >=20 -> 4-char prefix + ***)
# BEFORE the 120-char truncate. The append must NEVER change deny/allow, exit
# codes, or the stderr message. Every test pins EPM_GUARD_DENY_SIDECAR via the
# harness `sidecar` kwarg; deny-side cases classify against the pinned
# always-on-main repo (off-main the tail gate would exit 0 and no row fires).
# Templates reuse the file's existing blocked shapes (branch-switch fence /
# merge fence M1) — no new gated literals.
# ---------------------------------------------------------------------------

_SIDECAR_KEYS = {"ts", "guard", "arm", "len", "head", "clause_head"}
_TS_RE = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
_SIDECAR_BLOCKED_SWITCH = "git switch fix/foo"  # from test_branch_creation_and_switch_still_block
_SIDECAR_BLOCKED_MERGE = "git merge issue-123"  # from test_merge_shapes_block (M1)


def test_deny_writes_sidecar_row(tmp_path):
    """T1: a denied command appends exactly one valid JSON row (full schema)."""
    sidecar = tmp_path / "deny.jsonl"
    proc = _run_full(_SIDECAR_BLOCKED_SWITCH, sidecar=str(sidecar))
    assert proc.returncode == 2
    lines = sidecar.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) == _SIDECAR_KEYS
    assert row["guard"] == "repo_root_branch"
    assert row["arm"]
    assert row["len"] == len(_SIDECAR_BLOCKED_SWITCH)
    assert re.fullmatch(_TS_RE, row["ts"])


def test_deny_sidecar_arm_with_parens_valid_json(tmp_path):
    """T2: an arm label carrying spaces + parentheses still yields valid JSON."""
    sidecar = tmp_path / "deny.jsonl"
    proc = _run_full(_SIDECAR_BLOCKED_MERGE, sidecar=str(sidecar))
    assert proc.returncode == 2
    row = json.loads(sidecar.read_text().splitlines()[0])
    assert "(" in row["arm"]


def test_allow_writes_no_sidecar_row(tmp_path):
    """T3: an allowed command writes nothing (sidecar file not created)."""
    sidecar = tmp_path / "deny.jsonl"
    assert _run("git status", sidecar=str(sidecar)) == 0
    assert not sidecar.exists()


def test_sidecar_write_failure_preserves_deny_exit(tmp_path):
    """T4: a failed sidecar append leaves exit 2 + the BLOCKED stderr intact.

    The sidecar's parent is pointed at a regular FILE so both ``mkdir -p`` and
    the append fail regardless of uid (root ignores a chmod-0500 dir; a
    file-as-parent fails for root too). A writable re-run in the same test
    confirms the row WOULD have been written — isolating the failure to the
    write, not the deny logic.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    bad_sidecar = blocker / "deny.jsonl"
    proc = _run_full(_SIDECAR_BLOCKED_SWITCH, sidecar=str(bad_sidecar))
    assert proc.returncode == 2
    assert "BLOCKED:" in proc.stderr
    assert not bad_sidecar.exists()
    ok_sidecar = tmp_path / "deny.jsonl"
    proc2 = _run_full(_SIDECAR_BLOCKED_SWITCH, sidecar=str(ok_sidecar))
    assert proc2.returncode == 2
    assert len(ok_sidecar.read_text().splitlines()) == 1


def test_sidecar_head_bounded_no_full_command(tmp_path):
    """T5: no full command text in the row; heads are <=120 chars; len exact."""
    cmd = _SIDECAR_BLOCKED_SWITCH + " # " + "pad " * 60
    assert len(cmd) > 240
    sidecar = tmp_path / "deny.jsonl"
    proc = _run_full(cmd, sidecar=str(sidecar))
    assert proc.returncode == 2
    raw = sidecar.read_text()
    assert cmd not in raw
    row = json.loads(raw.splitlines()[0])
    assert len(row["head"]) <= 120
    assert len(row["clause_head"]) <= 120
    assert row["len"] == len(cmd)


def test_sidecar_secret_shaped_token_masked(tmp_path):
    """T6: a secret-shaped token (hf_ + 34 alnum) never survives into the row.

    Variant (a): token inside the first 120 chars (env-assignment prefix
    before the git verb) — masked to a 4-char prefix + ``***``.
    Variant (b): token straddling the 120-char truncation boundary — masking
    runs BEFORE the truncate, so no >=8-char fragment may survive either.
    """
    token = "hf_" + "A1b2C3d4" * 4 + "Zz"
    assert len(token) == 37
    cmd = f"HF_TOKEN={token} {_SIDECAR_BLOCKED_SWITCH}"
    sidecar = tmp_path / "deny.jsonl"
    proc = _run_full(cmd, sidecar=str(sidecar))
    assert proc.returncode == 2
    raw = sidecar.read_text()
    assert token not in raw
    row = json.loads(raw.splitlines()[0])
    assert "***" in row["head"]

    prefix = _SIDECAR_BLOCKED_SWITCH + " # " + "pad " * 21
    cmd2 = prefix + token
    assert len(prefix) < 120 < len(cmd2)
    sidecar2 = tmp_path / "deny2.jsonl"
    proc2 = _run_full(cmd2, sidecar=str(sidecar2))
    assert proc2.returncode == 2
    raw2 = sidecar2.read_text()
    for i in range(len(token) - 7):
        assert token[i : i + 8] not in raw2


# ---------------------------------------------------------------------------
# #1554: worktree-scoped LOCAL-main merge fence (Arm A: clause-initial
# `git -C <worktree>` form intercepted BEFORE the path-blind -C waiver;
# Arm B: the worktree cd-latch `scoped_wt` bit at the scoped-continue).
# Declines the #1530 contamination class — a worktree merge of the bare LOCAL
# `main` ref imports unpushed root-main commits — while `origin/main` /
# raw-sha merges, /tmp scratch landings, and every repo-root disposition are
# pinned unchanged. Escape hatch: EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1
# (session env + inline command prefix, the EPM_ALLOW_ROOT_PULL idiom).
# ---------------------------------------------------------------------------

_WT_LM_HATCH = "EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE"
_WT_LM_ARM_A_LABEL = "git -C <worktree> merge main (LOCAL-main"
_WT_LM_ARM_B_LABEL = "cd <worktree> && git merge main (LOCAL-main"
_ROOT_MERGE_ARM_LABEL = "branch merge on the shared root"


@pytest.mark.parametrize(
    "cmd",
    [
        # Literal worktree path, bare / quoted / absolute spellings.
        "git -C .claude/worktrees/issue-9 merge main",
        "git -C '.claude/worktrees/issue-9' merge main",
        (
            "git -C /home/thomasjiralerspong/explore-persona-space/"
            ".claude/worktrees/issue-9 merge main"
        ),
        # $WT spellings ($WT is the SKILL.md-conventional worktree variable;
        # blocked unconditionally — the #1530 F5 incident bound WT in a PRIOR
        # Bash call, invisible to the hook's latch).
        'git -C "$WT" merge --ff-only main',  # #1530 F5 prescription form (memory L11)
        'git -C "$WT" merge main --no-edit',  # #1530 F5 prescription form (memory L48)
        "git -C $WT merge main",
        "git -C ${WT} merge main",
        # Flag / quoting / redirect tolerance around the bare ref.
        "git -C .claude/worktrees/issue-9 merge --ff-only main",
        "git -C .claude/worktrees/issue-9 merge 'main'",
        "git -C .claude/worktrees/issue-9 merge main 2>&1",
        "git -C .claude/worktrees/issue-9 -c merge.ff=only merge main",
    ],
)
def test_worktree_local_main_merge_blocked(cmd, monkeypatch):
    """#1554 Arm A: a worktree-scoped merge of the bare LOCAL main ref declines.

    The deny label names the escape hatch (so the remediation is visible at
    deny time) and the Arm A label (so a fall-through to the #1128 root fence
    cannot masquerade as this fence).
    """
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    proc = _run_full(cmd)
    assert proc.returncode == 2, (cmd, proc.stderr)
    assert _WT_LM_HATCH in proc.stderr
    assert _WT_LM_ARM_A_LABEL in proc.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        "cd .claude/worktrees/issue-9 && git merge main",
        "cd .claude/worktrees/issue-9 && git merge --ff-only main",
        "cd .claude/worktrees/issue-9 && git merge main --no-edit",
        # The $WT cd-latch (separator-gated WT= arming) is a worktree latch.
        'WT=.claude/worktrees/issue-9; cd "$WT" && git merge --ff-only main',
        # Latch transition: scoped_wt follows the LATEST cd (tmp -> worktree).
        "cd /tmp/x && cd .claude/worktrees/z && git merge main",
        # Fail-closed by design (hook header gap (xx)): a worktree-latched
        # `git -C /tmp/...` merge of bare main declines via Arm B; the escape
        # hatch covers a deliberate need.
        "cd .claude/worktrees/issue-9 && git -C /tmp/x merge main",
        # The inline hatch is a `=1`-only substring match: a `=0` spelling
        # elsewhere in the command never disarms the fence.
        (
            "echo EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=0; "
            "cd .claude/worktrees/issue-9 && git merge main"
        ),
    ],
)
def test_worktree_cd_latch_local_main_merge_blocked(cmd, monkeypatch):
    """#1554 Arm B: a WORKTREE-armed cd-latch + bare-local-main merge declines.

    Asserts the Arm B label specifically — a broken Arm B falling through to
    the #1128 root fence would read rc==2 for the adjacent reason.
    """
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    proc = _run_full(cmd)
    assert proc.returncode == 2, (cmd, proc.stderr)
    assert _WT_LM_ARM_B_LABEL in proc.stderr
    assert _WT_LM_HATCH in proc.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        # origin/main forms — the sanctioned D7 fast-forward recipe.
        "git -C .claude/worktrees/issue-9 merge origin/main",
        'git -C "$WT" merge --ff-only origin/main',
        (
            'git -C "$WT" fetch origin "+refs/heads/main:refs/remotes/origin/main" '
            '&& git -C "$WT" merge --ff-only origin/main'
        ),
        'timeout 60 git -C "$WT" fetch origin "+refs/heads/main:refs/remotes/origin/main"',
        # Raw-sha / "$MAIN_SHA" merges (the Step 10d conflict-recovery form).
        'git -C "$WT" merge "$MAIN_SHA"',
        "git -C .claude/worktrees/issue-9 merge bf3c4711d6",
        # /tmp scratch-worktree landing (the deny text's own recipe).
        (
            "git worktree add --detach /tmp/land-1554 origin/main "
            "&& git -C /tmp/land-1554 merge issue-1554 "
            "&& git -C /tmp/land-1554 push origin HEAD:main"
        ),
        # Ref discrimination: bare-`main`-only (prefixes / lineage refs pass;
        # `main~1` is the documented gap (xx)(e) residual, pinned as allowed).
        "git -C .claude/worktrees/issue-9 merge mainline",
        'git -C "$WT" merge main~1',
        # $WT word boundary: $WTF is not the worktree variable.
        'git -C "$WTF" merge main',
        # Non-merge / merge-adjacent -C ops keep the -C waiver.
        "git -C .claude/worktrees/issue-9 merge-base main HEAD",
        "git -C .claude/worktrees/issue-9 status",
        # /tmp latch disposition byte-identical; origin/main + sha under a
        # worktree latch; latch transition worktree -> /tmp allows.
        "cd /tmp/x && git merge main",
        "cd .claude/worktrees/issue-9 && git merge origin/main",
        'cd .claude/worktrees/issue-9 && git merge "$MAIN_SHA"',
        "cd .claude/worktrees/issue-9 && cd /tmp/y && git merge main",
        (
            "cd .claude/worktrees/issue-9 "
            "&& git fetch origin +refs/heads/main:refs/remotes/origin/main"
        ),
        "cd .claude/worktrees/issue-9 && git merge-base main HEAD",
        # ^-anchor: quoted spoofs of the gated shape mid-clause never match
        # (the grep clause is ALSO covered by the #1098 waiver; the echo
        # clause by the unanchored -C waiver — both pre-existing paths).
        'grep -rn "git -C .claude/worktrees/issue-9 merge main" scripts/',
        'echo "never run git -C .claude/worktrees/x merge main"',
        # An inline `=0` prefix is an env-assignment wrapper: it evades the
        # ^-anchored Arm A head (gap (xx)(b) residual, pinned as allowed) —
        # and its value never arms the hatch either way.
        "EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=0 git -C .claude/worktrees/issue-9 merge main",
    ],
)
def test_worktree_local_main_merge_allowed(cmd, monkeypatch):
    """#1554 allow set: every sanctioned / non-bare-main / spoof shape exits 0."""
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    proc = _run_full(cmd)
    assert proc.returncode == 0, (cmd, proc.stderr)


def test_worktree_local_main_merge_escape_hatch_env(monkeypatch):
    """Session-env hatch: EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1 flips both arms to allow."""
    monkeypatch.setenv(_WT_LM_HATCH, "1")
    assert _run("git -C .claude/worktrees/issue-9 merge main") == 0
    assert _run("cd .claude/worktrees/issue-9 && git merge main") == 0


def test_worktree_local_main_merge_escape_hatch_inline(monkeypatch):
    """Inline-prefix hatch: a `<VAR>=1` spelling anywhere in the command disarms.

    Command-wide substring semantics inherited verbatim from the
    ``EPM_ALLOW_ROOT_PULL`` sibling idiom (``guard_repo_root_pull.sh``).
    """
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    assert (
        _run("EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1 git -C .claude/worktrees/issue-9 merge main")
        == 0
    )
    assert (
        _run(
            "EPM_ALLOW_WORKTREE_LOCAL_MAIN_MERGE=1 true; "
            "cd .claude/worktrees/issue-9 && git merge main"
        )
        == 0
    )


@pytest.mark.parametrize(
    "cmd",
    [
        # The repo-root merge fence (#1128) keeps its own classification.
        "git merge main",
        # `;` drops the cd-latch (fail-closed reset, #804): the merge clause
        # is UNSCOPED and takes the root fence, not the worktree arm.
        "cd .claude/worktrees/issue-9; git merge main",
    ],
)
def test_root_merge_main_classification_unchanged(cmd, monkeypatch):
    """#1554 arms neither hijack nor alter the #1128 root-fence classification."""
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    proc = _run_full(cmd)
    assert proc.returncode == 2, (cmd, proc.stderr)
    assert _ROOT_MERGE_ARM_LABEL in proc.stderr
    assert "(LOCAL-main" not in proc.stderr


def test_worktree_local_main_merge_pipe_producer(monkeypatch):
    """A piped worktree bare-main merge is still a merge (#1538 waiver is grep-family-only)."""
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    proc = _run_full("git -C .claude/worktrees/issue-9 merge main | tail -3")
    assert proc.returncode == 2, proc.stderr
    assert _WT_LM_ARM_A_LABEL in proc.stderr


# ---------------------------------------------------------------------------
# task.py quoted-argument payload masking (#1566).
#
# mask_taskpy_arg_payloads() masks balanced SINGLE-QUOTED argument payloads of
# a clause-initial `*task.py` python-script invocation to a neutral sentinel
# BEFORE the trigger-literal pre-filter, under a fail-closed P1-P6 refusal
# predicate (head shape / safe non-payload charset / exact quote parse /
# latch-vocabulary isolation incl. payload bodies / clean prefix quote-state /
# latch-clean prefix incl. WT=). ANY refusal leaves the input byte-identical
# -> today's blocked disposition with the --file workaround.
#
# Anti-vacuity contract (#1566 plan section 5): every NP* fixture embeds the
# shared _TASKPY_PAYLOAD constant — a full two-token pre-filter-matching prose
# string — and the NPB5/NPB6 twins pin exit 2 for the SAME payload when NOT in
# the maskable single-quoted shape, so each NP exit 0 is attributable to the
# mask rather than to a payload that never engaged the guard. The pre-filter
# (script: `grep -qE '\bgit\b...' || exit 0`) requires BOTH the `git` token
# AND a verb-set token, so test_taskpy_payload_matches_prefilter additionally
# pins the constant against the guard's OWN pre-filter regex (drift of either
# side fails loud).

# Mirrors the /tmp/issue-1566-repro.json incident payload: prose naming a
# branch-creation op (`git` + `checkout` = both pre-filter tokens; the
# branch-creation detector blocks it without ref resolution).
_TASKPY_PAYLOAD = "clarifier context: the guard blocks git checkout -b at the repo root"

_TASKPY = "uv run python scripts/task.py"


def test_taskpy_payload_matches_prefilter():
    """Anti-vacuity: the shared payload matches the guard's own pre-filter regex."""
    script_text = SCRIPT.read_text()
    m = re.search(r"^echo \"\$cmd\" \| grep -qE '([^']+)' \|\| exit 0$", script_text, re.M)
    assert m, "guard pre-filter line not found in script"
    assert re.search(m.group(1), _TASKPY_PAYLOAD), (m.group(1), _TASKPY_PAYLOAD)


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            f"{_TASKPY} post-marker 1566 epm:progress --note '{_TASKPY_PAYLOAD}'",
            id="NP1-incident_post_marker_note",
        ),
        pytest.param(
            f"{_TASKPY} new --kind infra --title '{_TASKPY_PAYLOAD}'",
            id="NP2-new_title",
        ),
        pytest.param(
            f"{_TASKPY} new --kind infra --origin-prompt '{_TASKPY_PAYLOAD}'",
            id="NP3-origin_prompt",
        ),
        pytest.param(
            f"{_TASKPY} new --title '{_TASKPY_PAYLOAD}' --origin-prompt '{_TASKPY_PAYLOAD}'",
            id="NP4-multi_span_one_clause",
        ),
        pytest.param(
            # set-goal takes its payload as a POSITIONAL — the flag-agnostic
            # coverage a flag-enumeration design would miss.
            f"{_TASKPY} set-goal 1566 '{_TASKPY_PAYLOAD}'",
            id="NP5-positional_payload",
        ),
        pytest.param(
            # Quote-free safe prefix clause: P5/P6 pass (no quotes, no latch
            # vocabulary — /home/user is neither /tmp/ nor a worktree path).
            f"cd /home/user && {_TASKPY} post-marker 1566 epm:progress --note '{_TASKPY_PAYLOAD}'",
            id="NP6-quote_free_safe_prefix",
        ),
        pytest.param(
            # Separator INSIDE the payload — the mis-split incident class:
            # unmasked, the `;` splits the clause and the payload tail
            # classifies clause-initial.
            f"{_TASKPY} post-marker 1566 epm:progress --note 'sync first; {_TASKPY_PAYLOAD}'",
            id="NP7-separator_in_payload_mis_split_class",
        ),
        pytest.param(
            # Absolute-path invocation head (R4 deliberately does not
            # transfer: a repo path in inert argv is prose).
            "uv run python /home/thomasjiralerspong/explore-persona-space/scripts/task.py"
            f" post-marker 1566 epm:progress --note '{_TASKPY_PAYLOAD}'",
            id="NP8-absolute_path_head",
        ),
    ],
)
def test_taskpy_arg_payload_masking_allows(cmd):
    """Canonical single-quoted task.py argument payloads mask -> exit 0 (#1566)."""
    assert _run(cmd) == 0


def test_taskpy_payload_embedded_newline_rejoin_pinned():
    """NPQ1 (#1566): literal newline INSIDE the single-quoted payload — pinned ALLOW.

    Deliberately pins the REALIZED disposition: the awk pre-pass re-joins
    stdin records with the newlines awk consumed, so P3's exact single-quote
    parse spans the embedded newline and the payload masks -> exit 0. The pin
    is the point — if the record re-join ever changed (e.g. the span scan
    stopped at a record boundary), P3 would refuse and this shape would
    revert to exit 2; this test makes that change visible, not incidental.
    """
    cmd = f"{_TASKPY} post-marker 1566 epm:progress --note 'line one\n{_TASKPY_PAYLOAD}'"
    assert _run(cmd) == 0


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # A masked benign payload must not swallow a LATER real mutation.
            f"{_TASKPY} post-marker 1566 epm:progress --note 'benign note' && git switch fix/i1566",
            id="NPB1-real_mutation_later_clause",
        ),
        pytest.param(
            # ... nor an EARLIER one in the same compound.
            f"git switch fix/i1566 && {_TASKPY} post-marker 1566 epm:progress --note 'benign note'",
            id="NPB2-real_mutation_earlier_clause",
        ),
        pytest.param(
            # Semicolon-separated twin of NPB1 (guidance item 5).
            f"{_TASKPY} post-marker 1566 epm:progress --note 'benign note'; git switch fix/i1566",
            id="NPB1b-real_mutation_later_clause_seq",
        ),
        pytest.param(
            # Command substitution as the payload — refused at `"` (P2).
            f'{_TASKPY} post-marker 1566 epm:progress --note "$(compose_note {_TASKPY_PAYLOAD})"',
            id="NPB3-command_substitution_payload",
        ),
        pytest.param(
            # Backtick variant — refused at the backtick (P2).
            f'{_TASKPY} post-marker 1566 epm:progress --note "`{_TASKPY_PAYLOAD}`"',
            id="NPB4-backtick_payload",
        ),
        # (#1710) The historical NPB5 double-quoted-shared-payload block is
        # REPLACED by NDP1-double_quoted_note_incident_3 (positive) +
        # NDP_P7_* refusal-ladder pins below. Arm 2's P7 no-expansion
        # refusal admits the double-quoted note body under the same
        # exact-parse property the single-quoted mask requires.
        pytest.param(
            # UNQUOTED note text with the shared trigger — second twin.
            f"{_TASKPY} post-marker 1566 epm:progress --note {_TASKPY_PAYLOAD}",
            id="NPB6-unquoted_shared_payload_twin",
        ),
        pytest.param(
            # ANSI-C quoting — pins the blanket `$` exclusion rationale
            # ($'...' processes escaped quotes, breaking the exact parse).
            f"{_TASKPY} post-marker 1566 epm:progress --note $'{_TASKPY_PAYLOAD}'",
            id="NPB7-ansi_c_quoted_payload",
        ),
        pytest.param(
            # Backslash-escaped-quote idiom ('it'\''s ...) — refused at the
            # backslash after the first consumed span (P2).
            f"{_TASKPY} post-marker 1566 epm:progress --note 'it'\\''s noted: {_TASKPY_PAYLOAD}'",
            id="NPB8-escaped_quote_idiom",
        ),
        pytest.param(
            # Trailing redirect after the payload — refused at `>` (P2).
            f"{_TASKPY} post-marker 1566 epm:progress --note '{_TASKPY_PAYLOAD}' > /tmp/i1566.log",
            id="NPB9-trailing_redirect",
        ),
        pytest.param(
            # Pre-opened SINGLE quote before the candidate — P5 conservatism.
            f"echo 'x' && {_TASKPY} post-marker 1566 epm:progress --note '{_TASKPY_PAYLOAD}'",
            id="NPB10-pre_opened_single_quote_prefix",
        ),
        pytest.param(
            # Clause-initial shell-consumer head lookalike: quoted args of a
            # shell consumer ARE executable code — the P1 head whitelist is
            # the boundary (the piped xargs shape is pinned separately at
            # GPN3-xargs_exec).
            f"bash -c '{_TASKPY_PAYLOAD}'",
            id="NPB11-shell_consumer_head",
        ),
        pytest.param(
            # cd-latch vocabulary INSIDE the payload — P4 covers payload
            # bodies, keeping every latch-arming shape at today's
            # disposition.
            f"{_TASKPY} post-marker 1566 epm:progress --note 'run cd"
            f" .claude/worktrees/issue-1566 first, then: {_TASKPY_PAYLOAD}'",
            id="NPB12-latch_vocab_in_payload",
        ),
        pytest.param(
            # Pre-opened DOUBLE-quote flavor of P5 (with a separator char
            # inside the double-quoted string) — the variant an
            # apostrophe-parity check would miss.
            f"echo \"a; b\" && {_TASKPY} post-marker 1566 epm:progress --note '{_TASKPY_PAYLOAD}'",
            id="NPB13-pre_opened_double_quote_prefix",
        ),
        pytest.param(
            # Prefix WT= assignment clause — pins that P6's vocabulary
            # includes WT= (deliberately STRICTER than a verbatim R7/R8
            # parity copy, which checks the candidate only).
            f"WT=.claude/worktrees/issue-1566; {_TASKPY} post-marker 1566"
            f" epm:progress --note '{_TASKPY_PAYLOAD}'",
            id="NPB14-prefix_wt_assignment",
        ),
        pytest.param(
            # Raw-NEWLINE-separated real-mutation variant of NPB1 (the
            # NL-sentinel splitter path).
            f"{_TASKPY} post-marker 1566 epm:progress --note 'benign note'\ngit switch fix/i1566",
            id="NPB15-newline_separated_real_mutation",
        ),
    ],
)
def test_taskpy_arg_payload_masking_refusals_block(cmd):
    """P1-P6 refusals keep today's blocked disposition byte-identical (#1566)."""
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# Pinned-git-context pins (#1567) — the GIT_DIR/GIT_WORK_TREE pinning itself is
# under test. All block probes REUSE existing in-file shape constants (the
# sidecar-section precedent above — no new gated literals).
# ---------------------------------------------------------------------------


def _run_against_repo(repo: Path, cmd: str) -> subprocess.CompletedProcess[str]:
    """Run the hook with its git-state reads pinned to ``repo`` (not the session fixture)."""
    env = _scrubbed_env()
    env["GIT_DIR"] = str(repo / ".git")
    env["GIT_WORK_TREE"] = str(repo)
    env["EPM_GUARD_DENY_SIDECAR"] = "/dev/null"
    payload = json.dumps({"tool_input": {"command": cmd}})
    return subprocess.run([str(SCRIPT)], input=payload, text=True, capture_output=True, env=env)


def _git_in(repo: Path, *args: str) -> None:
    """Run a checked git subcommand against a throwaway test repo (scrubbed env)."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env=_scrubbed_env(),
        timeout=20,
    )


def test_off_main_pinned_context_fail_open(tmp_path):
    """The tail fail-open arm (hook L1775-1778): off-``main`` context => exit 0.

    Self-discrimination arm FIRST: the same payload IS classified as a block
    (exit 2) against the on-main session fixture — so the off-main exit 0
    below can only come from the tail fail-open arm, not a classifier miss.
    Previously untestable without racing the fleet's shared-root state.
    """
    assert _run(_SIDECAR_BLOCKED_SWITCH) == 2
    _build_pinned_repo(tmp_path)
    _git_in(tmp_path, "checkout", "-q", "-b", "not-main")
    proc = _run_against_repo(tmp_path, _SIDECAR_BLOCKED_SWITCH)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


def test_detached_head_pinned_context_fail_open(tmp_path):
    """Detached-HEAD context (the exact #1528 mid-rebase incident state) => exit 0.

    ``rev-parse --abbrev-ref HEAD`` prints ``HEAD`` != ``main`` when detached,
    so the tail arm fails open. Same self-discrimination arm as above.
    """
    assert _run(_SIDECAR_BLOCKED_SWITCH) == 2
    _build_pinned_repo(tmp_path)
    _git_in(tmp_path, "checkout", "-q", "--detach")
    proc = _run_against_repo(tmp_path, _SIDECAR_BLOCKED_SWITCH)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


def test_block_classification_reads_pinned_repo_not_live_root():
    """Regression fence: hook classification runs in the PINNED repo, not the live root.

    The session fixture's HEAD sha does not resolve in the live root (a fresh
    unique commit), yet the bare-sha detach template still exits 2 — proving
    the hook's rev-parse ran against the pinned repo. FAILS LOUD if
    ``_run_full`` ever reverts to an unpinned ``{**os.environ}`` env: the sha
    would then resolve nowhere => fail-soft exit 0 => assertion error.
    """
    sha = _repo_head_sha()
    probe = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        capture_output=True,
        text=True,
        env=_scrubbed_env(),
        timeout=20,
    )
    assert probe.returncode != 0, f"fixture sha {sha} unexpectedly resolves in the live root"
    assert _run(f"git checkout {sha}") == 2


def test_hook_repo_constant_targets_canonical_root():
    """Text pin: the hook's ``REPO=`` constant names the canonical shared root.

    Retains the integration invariant the deleted on-main skipif implicitly
    carried (the hook points at the real shared root) with no git-state read.
    """
    match = re.search(r"^REPO=(\S+)\s*$", SCRIPT.read_text(), flags=re.MULTILINE)
    assert match, "hook REPO= constant line not found"
    assert match.group(1) == str(REPO)


def test_pinned_env_scrubs_ambient_git_vars(monkeypatch):
    """The #1545 GIT_* scrub: an ambient GIT_* var never reaches the hook.

    ``GIT_OBJECT_DIRECTORY`` is a GIT_* var ``_pinned_env`` never assigns; if
    the scrub is deleted it leaks into the hook subprocess, the hook's git
    reads fail-soft, and this block row goes red — the committed,
    always-running form of the poisoned-ambient-env acceptance run.
    """
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/nonexistent")
    assert _run(_SIDECAR_BLOCKED_SWITCH) == 2


# ============================================================================
# (#1710) Three added mask arms:
#   Arm 1 — mask_ssh_payload_separators DOUBLE-quoted branch (R9 refusal)
#   Arm 2 — mask_taskpy_arg_payloads DOUBLE-quoted span support (P7 refusal)
#   Arm 3 — mask_python_c_string_literals (NEW mask; C4-C10 refusal ladder)
#
# Every new POSITIVE param carries a paraphrased incident-shape payload (no
# destructive-command literal reproduced from the incident transcript); every
# NEGATIVE param pins one specific refusal arm as the ONLY reason that shape
# refuses (the anti-vacuity discipline the round-1 planner named).
# ============================================================================

# ---- Arm 1: SSH double-quoted payload waiver (occurrence 2 shape) -----------


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # Occurrence 2 shape: pod-side git op inside a double-quoted
            # payload — the payload runs on the POD, not the local repo.
            'ssh pod-1689 "cd /workspace/x && git reset --hard origin/issue-1689"',
            id="NDS1-double_quoted_multi_statement",
        ),
        pytest.param(
            # A different pod-side git verb inside the double-quoted payload.
            'ssh pod-1689 "cd /workspace && git switch main"',
            id="NDS2-double_quoted_pod_side_git",
        ),
        pytest.param(
            # gcloud compute ssh head + double-quoted payload — parity with
            # #1463's gcloud allowance, extended to double quotes.
            'gcloud compute ssh pod-1 --command "cd /w && git rebase main"',
            id="NDS3-double_quoted_gcloud_compute_ssh",
        ),
        pytest.param(
            # Additional pod-side git-verb variant inside the double-quoted
            # payload (pins that a second verb class also waives).
            'ssh pod-1 "cd /w && git checkout main"',
            id="NDS4-double_quoted_second_verb_class",
        ),
        pytest.param(
            # NDS_R6a anti-vacuity twin: double-quoted arm keeps the R4/R6
            # repo-path / cd-latch refusals intact. This CLOSE variant uses
            # /workspace/other (non-repo path) so it MUST pass — proving the
            # R4/R6 refusals fire only on the repo-path/cd-latch content,
            # not on any cd + double-quoted shape.
            'ssh pod-1689 "cd /workspace/other && git fetch origin"',
            id="NDS_R6a-double_quoted_repo_path_with_cd_latch",
        ),
    ],
)
def test_ssh_double_quoted_payload_masking_allows(cmd):
    """Arm 1 positive pins (#1710): double-quoted ssh payloads waive under R9."""
    assert _run(cmd) == 0


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # R9: dollar expansion in the payload — refuse. Payload uses a
            # BLOCKED verb (git reset --hard) so the fall-through classifier
            # blocks after refusal.
            'ssh pod-1 "cd $HOME && git reset --hard"',
            id="NDS_R1-double_quoted_payload_with_dollar_expansion",
        ),
        pytest.param(
            # R9: command substitution — refuse. Payload carries a BLOCKED
            # verb so fall-through classifier blocks.
            'ssh pod-1 "cd /w && $(git reset --hard)"',
            id="NDS_R2-double_quoted_payload_with_command_sub",
        ),
        pytest.param(
            # R9: backtick — refuse. Payload carries a BLOCKED verb so
            # fall-through classifier blocks.
            'ssh pod-1 "cd /w && `git reset --hard`"',
            id="NDS_R3-double_quoted_payload_with_backtick",
        ),
        pytest.param(
            # R9: any backslash in the payload — refuse (Python \xNN /
            # ANSI-C \x1b escapes resolve unpredictably).
            'ssh pod-1 "cd /w \\&& git reset --hard"',
            id="NDS_R4-double_quoted_payload_with_backslash",
        ),
        pytest.param(
            # No closing double-quote — refuse (C-parity with the
            # single-quoted branch's constraint-1).
            'ssh pod-1 "cd /w && git reset --hard',
            id="NDS_R5-double_quoted_payload_unbalanced_quote",
        ),
        pytest.param(
            # R4 repo-path spelling inside the double-quoted payload — still
            # refuses (the cd-to-repo latch is unchanged for double quotes).
            'ssh vm "cd $HOME/explore-persona-space && git reset --hard"',
            id="NDS_R6-double_quoted_payload_repo_path",
        ),
        pytest.param(
            # R1 pipe-producer position stays classifying (parity with NM8).
            'ssh host "cd /w && git reset --hard" | bash',
            id="NDS_R7-double_quoted_payload_pipe_producer",
        ),
    ],
)
def test_ssh_double_quoted_masking_refusal_ladder_blocks(cmd):
    """Arm 1 negative pins (#1710): every R9/R4/R1 refusal shape stays blocked."""
    assert _run(cmd) == 2


# ---- Arm 2: task.py double-quoted --note / --title / positional waiver ------
# (occurrence 3 shape)


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # Occurrence 3 shape: a double-quoted --note whose PROSE names a
            # destructive git phrase (`git reset --hard`) but carries no
            # actual mutation. The old --file workaround is no longer needed.
            f"{_TASKPY} post-marker 1689 epm:progress "
            f'--note "run-launched: git reset --hard to origin/issue-1689 R8 HEAD"',
            id="NDP1-double_quoted_note_incident_3",
        ),
        pytest.param(
            # Same shape via --title.
            f'{_TASKPY} set-title 1689 "git reset --hard prose only"',
            id="NDP2-double_quoted_title",
        ),
        pytest.param(
            # Same via --origin-prompt.
            f'{_TASKPY} new --kind infra --title "x" --origin-prompt "git merge issue-1"',
            id="NDP3-double_quoted_origin_prompt",
        ),
        pytest.param(
            # Positional double-quoted set-goal payload (parity with NP5).
            f'{_TASKPY} set-goal 1689 "prose citing git rebase origin/main context"',
            id="NDP4-double_quoted_positional_set_goal",
        ),
        pytest.param(
            # Multiple double-quoted spans in ONE clause (parity with NP4).
            f'{_TASKPY} post-marker 1689 epm:x --note "first prose" --title "second prose"',
            id="NDP5-multi_double_quoted_spans_one_clause",
        ),
    ],
)
def test_taskpy_double_quoted_arg_masking_allows(cmd):
    """Arm 2 positive pins (#1710): double-quoted task.py args waive under P7."""
    assert _run(cmd) == 0


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # P7: dollar expansion — refuse. Note contains a BLOCKED verb so
            # fall-through classifies as blocked.
            f'{_TASKPY} post-marker 1689 epm:x --note "$VAR git reset --hard"',
            id="NDP_P7_R1-double_quoted_dollar_expansion",
        ),
        pytest.param(
            # P7: command substitution — refuse.
            f'{_TASKPY} post-marker 1689 epm:x --note "$(git reset --hard)"',
            id="NDP_P7_R2-double_quoted_command_sub",
        ),
        pytest.param(
            # P7: backtick — refuse.
            f'{_TASKPY} post-marker 1689 epm:x --note "`git reset --hard`"',
            id="NDP_P7_R3-double_quoted_backtick",
        ),
        pytest.param(
            # P7: backslash — refuse (escapes in double-quoted strings resolve
            # at bash-parse time).
            f'{_TASKPY} post-marker 1689 epm:x --note "text with \\backslash and git reset --hard"',
            id="NDP_P7_R4-double_quoted_backslash",
        ),
        pytest.param(
            # No closing double-quote — refuse.
            f'{_TASKPY} post-marker 1689 epm:x --note "unbalanced git reset --hard',
            id="NDP_P7_R5-double_quoted_unbalanced",
        ),
    ],
)
def test_taskpy_double_quoted_masking_refusal_ladder_blocks(cmd):
    """Arm 2 negative pins (#1710): every P7 refusal shape stays blocked."""
    assert _run(cmd) == 2


# ==== #1710 — python -c string literal mask ==================================
#
# (occurrence 4 shape) A `python -c '<inert prose>'` / `uv run python -c
# "<inert prose>"` payload whose Python STRING LITERAL merely quotes a
# destructive-git phrase as PROSE (a fingerprint helper hashing a bug
# description) now masks under C1-C10. The refusal ladder is STRICTER than
# ssh/taskpy: a `python -c` payload is executable code by construction, so
# `import subprocess` / `os.system` / any function-call shape refuses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # Occurrence 4 shape: single-quoted inert string literal that
            # quotes a destructive-git phrase as prose only.
            "python -c 'phrase mentioning git reset --hard as prose only'",
            id="NPC1-python_c_single_quoted_inert_string",
        ),
        pytest.param(
            # Same, double-quoted.
            'python -c "phrase mentioning git reset --hard as prose only"',
            id="NPC2-python_c_double_quoted_inert_string",
        ),
        pytest.param(
            # The incident 4 helper shape: `uv run python -c '<inert>'`.
            "uv run python -c 'phrase mentioning git rebase origin/main as prose'",
            id="NPC3-uv_run_python_c_inert",
        ),
        pytest.param(
            # Versioned python head (parity with the taskpy mask's head).
            "python3.11 -c 'phrase mentioning git checkout -b main as prose'",
            id="NPC4-python3_11_c_inert",
        ),
        pytest.param(
            # Anti-vacuity twin: the SAME inert phrase blocks via the
            # taskpy-mask + fall-through paths WITHOUT a python -c head.
            # This proves the C1 head regex is NOT the only refusal — a
            # `bash -c` version still blocks because `bash -c` is a
            # shell-consumer head whose quoted args are executable code, and
            # the fall-through does not mask a shell-consumer.
            'bash -c "phrase mentioning git reset --hard"',
            id="NPC_anti_vacuity_bash_c_still_blocks",
        ),
    ],
)
def test_python_c_string_literal_masking_allows(cmd):
    """Arm 3 positive pins (#1710): python -c inert-prose literals waive."""
    # NPC_anti_vacuity_bash_c_still_blocks pins that bash -c does NOT get
    # the same waiver — the C1 head regex rejects it and the fall-through
    # scans the whole command; the raw `git reset --hard` in the quoted arg
    # trips the pre-filter.
    expected = 2 if "bash -c" in cmd else 0
    assert _run(cmd) == expected


@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            # C7: payload contains `subprocess` — refuse.
            "python -c 'import subprocess; git reset --hard'",
            id="NPC_C4_R1-python_c_with_subprocess_import",
        ),
        pytest.param(
            # C7: `os.system` — refuse. Payload also names a BLOCKED verb
            # so the fall-through classifier keeps it blocked.
            "python -c 'x = os.system(\"git reset --hard\")'",
            id="NPC_C4_R2-python_c_with_os_system",
        ),
        pytest.param(
            # Any function-call shape refuses (conservative arm). Payload
            # also names a BLOCKED verb so fall-through blocks.
            "python -c 'foo(x); git reset --hard'",
            id="NPC_C4_R3-python_c_with_any_function_call",
        ),
        pytest.param(
            # C5: backtick — refuse. Payload also names a BLOCKED verb.
            "python -c '`inline` and git reset --hard'",
            id="NPC_C4_R4-python_c_with_backtick",
        ),
        pytest.param(
            # C6: dollar — refuse. Payload also names a BLOCKED verb.
            "python -c 'text with $VAR and git reset --hard'",
            id="NPC_C4_R5-python_c_with_dollar",
        ),
        pytest.param(
            # C4: backslash — refuse. Payload also names a BLOCKED verb.
            "python -c 'text with \\x1b and git reset --hard'",
            id="NPC_C4_R6-python_c_with_backslash",
        ),
        pytest.param(
            # C3: no closing quote — refuse.
            "python -c 'text with git reset --hard as prose",
            id="NPC_C4_R7-python_c_unbalanced_quote",
        ),
    ],
)
def test_python_c_string_literal_masking_refusal_ladder_blocks(cmd):
    """Arm 3 negative pins (#1710): every C3-C10 refusal shape stays blocked."""
    # Note: R1 - R5 above are C-tail / C4 / C5 / C6 / C4 refusals respectively;
    # the raw command's trigger-vocab (git reset / git rebase) then hits the
    # pre-filter and the command classifies to block. On refusals that also
    # carry `subprocess` / `os.system` etc., the pre-filter's git-verb trigger
    # is what classifies; the C7-C10 refusals block the MASKING but the raw
    # command still tokenizes through the classifier normally.
    assert _run(cmd) == 2


# ---- Cross-arm compound composition — must not regress ----------------------


def test_NCX2_real_mutation_after_python_c_mask_still_blocks():
    """(#1710) NCX2: a waived python -c call followed by a bare real
    mutation — the LATER mutation still blocks (clause-by-clause classify).
    """
    cmd = "python -c 'inert prose about git rebase main' && git switch issue-42"
    assert _run(cmd) == 2


def test_NCX3_real_mutation_before_taskpy_double_still_blocks():
    """(#1710) NCX3: a bare real mutation followed by a double-quoted
    task.py note — the EARLIER mutation still blocks.
    """
    cmd = f'git switch issue-42 && {_TASKPY} post-marker 1 epm:x --note "prose"'
    assert _run(cmd) == 2


# ---------------------------------------------------------------------------
# #1861 — exit-guarded worktree cd => STICKY scope; name-generalized $VAR
# latch; arming-separator restriction + cd-clause scope invalidation.
#
# MUST ALLOW: an exit-guarded worktree cd (`|| exit N`, or the brace-group
# `|| { ...; exit N; }` form) PROVES every clause past the guard tail runs
# with cwd inside the worktree — either the cd succeeded, or the shell exited
# first — so later NL/;-separated gated ops are scoped. The `WT=` assignment
# latch arms for ANY variable name (e.g. the SKILL.md Step 4a conventional
# WORKTREE) under the same #1058 proof obligations.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "cd .claude/worktrees/issue-9 || exit 1\ngit checkout -- figures/x",
            id="S1-exitguard_bare_literal_nl",
        ),
        pytest.param(
            'WT=.claude/worktrees/issue-9; cd "$WT" || exit 1; git restore .',
            id="S2-exitguard_bare_wt_seq",
        ),
        pytest.param(
            'WORKTREE=.claude/worktrees/issue-9\ncd "$WORKTREE" || exit 1\n'
            "git checkout main -- specs.md",
            id="S3-exitguard_bare_generalized_name",
        ),
        pytest.param(
            'cd .claude/worktrees/issue-9 || { echo "FATAL: cd failed" >&2; exit 1; }\n'
            "git reset --hard origin/main",
            id="S4-exitguard_group_literal",
        ),
        pytest.param(
            'WT=.claude/worktrees/issue-9; cd "$WT" || { echo "FATAL: cd failed" >&2; exit 1; }\n'
            "git checkout -- figures/x",
            id="S5-exitguard_group_wt",
        ),
        pytest.param(
            'WORKTREE=.claude/worktrees/issue-9; cd "$WORKTREE" && git checkout -- figures/x',
            id="G1-worktree_var_name_and_chain",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || exit 1\nuv run pytest -q\ngit checkout -- x",
            id="S6-sticky_survives_benign_clause",
        ),
        pytest.param(
            "true && cd .claude/worktrees/issue-9 || exit 1\ngit restore .",
            id="S7-and_preceded_cd_with_exitguard",
        ),
    ],
)
def test_1861_exitguard_sticky_and_generalized_name_allows(cmd):
    """Exit-guarded worktree cds scope later clauses; any var name arms (#1861)."""
    assert _run(cmd) == 0


# ---------------------------------------------------------------------------
# MUST BLOCK — #1861 fail-closed set. The sticky proof requires (a) the cd
# clause provably executed in the parent shell (an OR- or PIPE-preceded cd
# never arms), (b) a provably-exiting OR-tail (bare `exit [N]`, or a brace
# group whose final pre-`}` clause is an unconditionally-reached exit — no
# `return`, no non-exiting tail, no AND-guarded exit, no nested `{`), and
# (c) a terminator not defused by a following PIPE/BG separator. The tail's
# own clauses stay UNSCOPED (they run on the cd-failure path at the root),
# and ANY later cwd-changing clause — including paren-prefixed subshell
# spellings — voids both the plain latch and the sticky scope. The
# name-generalized latch still requires the exact assigned name (no prefix
# collision) and a non-command-substitution worktree-literal RHS.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        pytest.param(
            "cd .claude/worktrees/issue-9\ngit restore .",
            id="X1-unguarded_nl_literal",
        ),
        pytest.param(
            'WT=.claude/worktrees/issue-9\ncd "$WT"\ngit restore .',
            id="X2-unguarded_nl_wt",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || echo oops\ngit restore .",
            id="X3-nonexiting_tail_echo",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || { echo FATAL >&2; }\ngit restore .",
            id="X4-group_without_exit",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || { test -f x && exit 1; }\ngit restore .",
            id="X5-and_guarded_exit_in_group",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || { git reset --hard; exit 1; }",
            id="X6-gated_op_inside_guard_group",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || return 1\ngit restore .",
            id="X7-return_tail",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || exit 1 | tee /tmp/log\ngit restore .",
            id="X8-exit_terminator_into_pipe",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || exit 1 & git restore .",
            id="X9-exit_terminator_into_bg",
        ),
        pytest.param(
            'cd .claude/worktrees/issue-9 || { echo "FATAL" >&2; exit 1; } | tee /tmp/log\n'
            "git restore .",
            id="X10-group_close_into_pipe",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || exit 1\n(cd /tmp/z && git reset --hard)",
            id="X11-paren_subshell_cd_after_sticky",
        ),
        pytest.param(
            "false || cd .claude/worktrees/issue-9 && git restore .",
            id="X12-or_preceded_cd_never_arms",
        ),
        pytest.param(
            "echo x | cd .claude/worktrees/issue-9 && git restore .",
            id="X13-pipe_preceded_cd_never_arms",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || exit 1\ncd /home/user/elsewhere\ngit restore .",
            id="X14-post_sticky_nonworktree_cd",
        ),
        pytest.param(
            'WT=.claude/worktrees/issue-9; WT=$(mktemp -d); cd "$WT" && git restore .',
            id="X15-reassign_command_substitution",
        ),
        pytest.param(
            'WT=.claude/worktrees/issue-9\ncd "$WT2" && git checkout main -- specs.md',
            id="X16-name_mismatch_prefix_collision",
        ),
        pytest.param(
            "cd .claude/worktrees/issue-9 || { { exit 1; }; }\ngit restore .",
            id="X17-nested_brace_refusal",
        ),
    ],
)
def test_1861_sticky_fail_closed_blocks(cmd):
    """Unproven guard tails / arming separators / voided scopes never allow (#1861)."""
    assert _run(cmd) == 2


def test_1861_sticky_arm_b_local_main_merge_still_blocked(monkeypatch):
    """#1554 Arm B applies unchanged under sticky scope (#1861 acceptance 5).

    A sticky worktree-scoped `git merge main` declines with the Arm B label —
    the exit-guard grant must not widen the bare-local-main merge fence.
    """
    monkeypatch.delenv(_WT_LM_HATCH, raising=False)
    proc = _run_full("cd .claude/worktrees/issue-9 || exit 1\ngit merge main")
    assert proc.returncode == 2, proc.stderr
    assert _WT_LM_ARM_B_LABEL in proc.stderr
    assert _WT_LM_HATCH in proc.stderr


def test_zz_production_sidecar_untouched_by_suite():
    """The suite must never rewrite/drop rows from the PRODUCTION deny sidecar (#1528, #1990).

    The harness pins EPM_GUARD_DENY_SIDECAR (default ``/dev/null``) on every
    guard invocation; this end-of-module check compares the production
    sidecar's ROW SET against the module-import snapshot. Runs LAST in file
    order by position.

    #1990: raw byte-size equality false-FAILed under fleet concurrency
    (#1876: 36-min gate window, two foreign concurrent production deny rows
    appended). Row-identity snapshot: any row NEW to the sidecar since
    module import is treated as foreign concurrent activity and tolerated;
    every row PRESENT at import must still be present at end-of-module (an
    append-only sidecar cannot legitimately rewrite / drop / reorder).

    Scope reduction from the byte-size shape: this predicate proves row
    OBSERVABILITY, not suite-attribution of new rows. Direct
    suite-attribution catch lives WHOLLY in the harness's
    EPM_GUARD_DENY_SIDECAR=/dev/null pin — a future pin-leak would surface
    as a real production-sidecar row appearing without a corresponding
    foreign session; the positive-control test below catches the SHAPE (a
    suite-attributable append IS observable via ``len(new_rows) == 1``),
    not the attribution.

    Membership-not-count predicate — negligible collision risk for
    timestamped JSONL denial records (each row's ``ts`` field makes
    duplicate rows vanishingly unlikely in practice).

    Residual false-fail: external rotation/truncation of the production
    sidecar mid-run (a row present at import disappears at end-of-module)
    FAILs this predicate — arguably desirable signal, though rare on the
    shared VM.
    """
    current_rows = _read_sidecar_rows(_PROD_SIDECAR)
    current_set = set(current_rows)
    for row in _PROD_SIDECAR_ROWS_AT_IMPORT:
        assert row in current_set, "production sidecar row disappeared mid-run"
    # New rows are tolerated (foreign concurrent activity is by construction
    # legal — the harness pins /dev/null on every deliberate guard subprocess
    # so we cannot have written them).


def test_zz_production_sidecar_positive_control_catches_suite_write(tmp_path):
    """Positive control: a synthetic append MUST be observable to the canary shape.

    The end-of-module production-sidecar canary observes rows on an
    append-only forensic file. A future refactor that made the canary a
    silent no-op would tolerate suite writes just as tolerantly as a
    foreign concurrent-session append — the very defect this file exists
    to prevent. This test exercises the canary's snapshot-then-compare
    logic against a SYNTHETIC scenario (its own ``tmp_path`` file, never
    the real production sidecar) so its correctness is independent of
    whether a concurrent session appended a real row this second.

    Two orthogonal assertions:

    1. Snapshot rows are a SUBSET of the current rows after a foreign
       append (the tolerance direction: an append-only file never
       rewrites or reorders earlier rows, so a snapshot's rows survive
       any later append; equally, a suite that WROTE nothing sees its
       snapshot survive unchanged).
    2. The single new row is OBSERVABLE via the current-rows minus
       snapshot-rows set difference. If the canary shape ever collapsed
       to a shape that returned early or short-circuited past its
       comparison, ``len(new_rows) == 0`` would silently satisfy the
       weaker predicate and this test would FAIL — the guard against
       future silent-no-op refactors.

    Bounds: the test uses only ``tmp_path``. It never reads or writes the
    real ``_PROD_SIDECAR`` path (which lives under the canonical checkout's
    ``.claude/cache/`` and is protected by the sibling canary above).
    """
    fake_sidecar = tmp_path / "guard-deny-events.jsonl"
    fake_sidecar.write_bytes(
        b'{"ts":"2026-01-01T00:00:00Z","guard":"x","arm":"a","len":1,'
        b'"head":"h","clause_head":"c"}\n'
    )
    snapshot = _read_sidecar_rows(fake_sidecar)
    assert len(snapshot) == 1, "snapshot precondition: exactly one row before the append"

    with fake_sidecar.open("ab") as f:
        f.write(
            b'{"ts":"2026-01-01T00:00:01Z","guard":"x","arm":"a","len":1,'
            b'"head":"h","clause_head":"c"}\n'
        )
    current_rows = _read_sidecar_rows(fake_sidecar)

    # Tolerance: snapshot rows survive the foreign append (append-only invariant).
    for row in snapshot:
        assert row in current_rows, (
            "snapshot row must survive a later append (append-only invariant)"
        )

    # Positive control: the append IS observable — the canary shape cannot
    # silently no-op past a real suite-attributable write. A future
    # refactor that made ``_read_sidecar_rows`` return early would leave
    # ``new_rows`` empty and this assertion would FAIL — the guard against
    # tolerant-to-everything regressions.
    new_rows = [row for row in current_rows if row not in snapshot]
    assert len(new_rows) == 1, (
        "positive control: canary must observe exactly one new row after "
        f"a synthetic append (got {len(new_rows)})"
    )
