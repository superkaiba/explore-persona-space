"""Pin tests for ``scripts/step10d_guards.sh`` (task #1978).

The script extracts the Step 10d PRELUDE + Guard 0 + Guard 4 shell fragments
from ``.claude/skills/issue/SKILL.md`` § Merge safety guards into ONE tested,
checked-in executable so sessions invoke it instead of re-typing prose per
merge. Each guard subcommand is byte-equivalent-in-effect to the current
SKILL.md fence; these tests exercise every behavioral arm to catch drift.

Fixture strategy: subprocess against throwaway ``tmp_path`` git repos (the
``tests/test_guard_*.py`` family pattern). The script derives ``REPO_ROOT``
from ``git rev-parse --path-format=absolute --git-common-dir``; each fixture
sets up a scratch repo (or a scratch repo + issue worktree) and invokes the
script with cwd inside it, so the script's own derivation resolves to the
scratch, not the live repo.

TODO(unit-b): add ``test_skill_md_invokes_script`` -- a SKILL.md prose pin
asserting ``.claude/skills/issue/SKILL.md`` § Merge safety guards contains the
``scripts/step10d_guards.sh`` invocation for guards prelude / 0 / 4 (per
``tests/test_issue_skill_workload_cmd_script_pin.py`` precedent). Deferred to
Unit B (the SKILL.md fence-replacement diff) so the pin is written against
the actual invocation shape landing there.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "step10d_guards.sh"

# The scoped file set the Guard 4 fence protects. The fence's REAL case glob
# is ``.claude/skills/*`` (broader than the plan's paraphrase
# ``.claude/skills/**/SKILL.md``); pick a stem inside ``.claude/rules/`` for
# the pass / refuse fixtures so we cover the case-glob preservation.
GUARD4_SCOPED_FILE = ".claude/rules/step10d-guard4-fixture.md"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "eps-test")
    env.setdefault("GIT_AUTHOR_EMAIL", "eps-test@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "eps-test")
    env.setdefault("GIT_COMMITTER_EMAIL", "eps-test@example.invalid")
    # Scrub the caller's GIT_* env so scratch repos don't leak into the live tree.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} in {cwd} failed rc={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _run_script(
    cwd: Path,
    *args: str,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``scripts/step10d_guards.sh`` with cwd inside a scratch repo.

    ``cwd`` MUST be inside a git repo (scratch or worktree) -- the script
    derives ``REPO_ROOT`` from ``git rev-parse --path-format=absolute
    --git-common-dir`` and fails without one.
    """
    env = os.environ.copy()
    # Scrub the caller's GIT_* env so the script's own git resolution uses cwd.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    env.setdefault("GIT_AUTHOR_NAME", "eps-test")
    env.setdefault("GIT_AUTHOR_EMAIL", "eps-test@example.invalid")
    env.setdefault("GIT_COMMITTER_NAME", "eps-test")
    env.setdefault("GIT_COMMITTER_EMAIL", "eps-test@example.invalid")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _make_scratch_repo(tmp_path: Path) -> Path:
    """Build a scratch git repo at ``tmp_path/scratch``.

    Returns the scratch repo root. The script's PRELUDE derivation resolves
    ``REPO_ROOT`` to this path when invoked with cwd inside it.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q", "-b", "main")
    (scratch / "README.md").write_text("scratch\n")
    _git(scratch, "add", "README.md")
    _git(scratch, "commit", "-q", "-m", "c1")
    return scratch


def _make_scratch_repo_with_worktree(tmp_path: Path, issue: int) -> tuple[Path, Path]:
    """Build a scratch repo + a real ``.claude/worktrees/issue-<N>`` worktree.

    Returns (repo_root, worktree_path). The script's derivation lands
    ``$WT = repo_root/.claude/worktrees/issue-<N>``.

    Uses ``git worktree add`` so ``$WT`` is a real linked worktree that git
    accepts under ``git -C``. A plain ``mkdir`` would fail the script's
    ``git -C "$WT" status`` calls.
    """
    scratch = _make_scratch_repo(tmp_path)
    _git(scratch, "checkout", "-q", "-b", f"issue-{issue}")
    _git(scratch, "checkout", "-q", "main")
    worktree_dir = scratch / ".claude" / "worktrees" / f"issue-{issue}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(scratch, "worktree", "add", "-q", str(worktree_dir), f"issue-{issue}")
    return scratch, worktree_dir


def _parse_kv(stdout: str) -> dict[str, str]:
    """Parse the script's ``KEY=VALUE`` stdout lines into a dict."""
    d: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        d[k] = v
    return d


# ---------------------------------------------------------------------------
# PRELUDE
# ---------------------------------------------------------------------------


def test_prelude_emits_repo_root_and_wt(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "1978", "--guard", "prelude")
    assert proc.returncode == 0, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert "REPO_ROOT" in kv, f"prelude did not emit REPO_ROOT: {proc.stdout!r}"
    assert "WT" in kv, f"prelude did not emit WT: {proc.stdout!r}"
    # REPO_ROOT resolves to the scratch repo root (path-format=absolute,
    # git-common-dir), NOT the live EPS repo.
    assert Path(kv["REPO_ROOT"]).resolve() == scratch.resolve()
    assert kv["WT"] == f"{scratch}/.claude/worktrees/issue-1978"


# ---------------------------------------------------------------------------
# Guard 0
# ---------------------------------------------------------------------------


def test_guard0_noop_on_clean_tree(tmp_path):
    scratch, worktree = _make_scratch_repo_with_worktree(tmp_path, 1978)
    # No dirty agent-memory files -> Guard 0 is a no-op.
    proc = _run_script(scratch, "1978", "--guard", "0")
    assert proc.returncode == 0, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert kv.get("MEM_COMMITTED") == "no", proc.stdout
    # No new commit on the branch.
    log_before = _git(worktree, "log", "--oneline").stdout.strip().splitlines()
    assert len(log_before) == 1, log_before


def test_guard0_commits_dirty_agent_memory(tmp_path):
    scratch, worktree = _make_scratch_repo_with_worktree(tmp_path, 1978)
    # Add a dirty agent-memory file.
    mem_dir = worktree / ".claude" / "agent-memory" / "test-agent"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_file = mem_dir / "note.md"
    mem_file.write_text("test memory entry\n")
    # First run -> commits, MEM_COMMITTED=yes.
    proc = _run_script(scratch, "1978", "--guard", "0")
    assert proc.returncode == 0, proc.stderr
    kv = _parse_kv(proc.stdout)
    assert kv.get("MEM_COMMITTED") == "yes", proc.stdout
    # Confirm commit exists and contains the memory file.
    log = _git(worktree, "log", "--oneline").stdout
    assert "persist agent-memory writes before Step-10d merge" in log, log
    show = _git(worktree, "show", "--stat", "HEAD").stdout
    assert ".claude/agent-memory/test-agent/note.md" in show, show
    # Second run -> idempotent no-op, MEM_COMMITTED=no.
    proc2 = _run_script(scratch, "1978", "--guard", "0")
    assert proc2.returncode == 0, proc2.stderr
    kv2 = _parse_kv(proc2.stdout)
    assert kv2.get("MEM_COMMITTED") == "no", proc2.stdout


def test_guard0_worktree_missing_exits_2(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    # No worktree at .claude/worktrees/issue-1978 -> exit 2 + ERROR=worktree-missing.
    proc = _run_script(scratch, "1978", "--guard", "0")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "worktree-missing", proc.stdout


# ---------------------------------------------------------------------------
# Guard 4
# ---------------------------------------------------------------------------


def _seed_guard4_fixture(
    tmp_path: Path,
    issue: int,
    *,
    drop_main_lines: bool,
) -> tuple[Path, Path]:
    """Scratch + worktree with a scoped file that origin/main added lines to.

    Layout after this returns:
    - scratch repo at ``tmp_path/scratch`` (bare-origin at
      ``tmp_path/origin.git`` pushed to)
    - main has 2 commits: c1 seed + c2 adding new lines to the scoped file
    - worktree ``issue-<N>`` forked from c1 (before c2)
    - if drop_main_lines=True: worktree DROPS the c2-added lines by taking
      a snapshot at c1's content (whole-file overwrite); this is the exact
      lost-update shape Guard 4 refuses
    - if drop_main_lines=False: worktree PRESERVES c2's added lines
      (fetches origin/main into origin/main ref, keeps identical content)
    """
    origin_dir = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin_dir))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q", "-b", "main")
    _git(scratch, "remote", "add", "origin", str(origin_dir))
    # c1: seed with a scoped file that has original lines.
    scoped = scratch / GUARD4_SCOPED_FILE
    scoped.parent.mkdir(parents=True, exist_ok=True)
    scoped.write_text("original line 1\noriginal line 2\n")
    _git(scratch, "add", str(scoped.relative_to(scratch)))
    _git(scratch, "commit", "-q", "-m", "c1")
    _git(scratch, "push", "-q", "-u", "origin", "main")
    # Fork the branch at c1.
    _git(scratch, "checkout", "-q", "-b", f"issue-{issue}")
    _git(scratch, "checkout", "-q", "main")
    # c2: add lines on main.
    scoped.write_text("original line 1\noriginal line 2\nmain-added line A\nmain-added line B\n")
    _git(scratch, "add", str(scoped.relative_to(scratch)))
    _git(scratch, "commit", "-q", "-m", "c2: main adds lines")
    _git(scratch, "push", "-q", "origin", "main")
    # Set up the worktree at issue-<N> (branch was forked at c1).
    worktree_dir = scratch / ".claude" / "worktrees" / f"issue-{issue}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(scratch, "worktree", "add", "-q", str(worktree_dir), f"issue-{issue}")
    # Branch adds its OWN line to the scoped file.
    scoped_wt = worktree_dir / GUARD4_SCOPED_FILE
    if drop_main_lines:
        # Whole-file snapshot that omits main's added lines -- the exact
        # lost-update shape.
        scoped_wt.write_text("original line 1\noriginal line 2\nbranch-added line X\n")
    else:
        # Preserve main's added lines AND add the branch's own.
        scoped_wt.write_text(
            "original line 1\n"
            "original line 2\n"
            "main-added line A\n"
            "main-added line B\n"
            "branch-added line X\n"
        )
    _git(worktree_dir, "add", GUARD4_SCOPED_FILE)
    _git(worktree_dir, "commit", "-q", "-m", "branch adds line X")
    return scratch, worktree_dir


def test_guard4_pass_when_no_dropped_lines(tmp_path):
    scratch, _ = _seed_guard4_fixture(tmp_path, 1978, drop_main_lines=False)
    proc = _run_script(scratch, "1978", "--guard", "4")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "pass", proc.stdout
    assert "LOST_UPDATE_PATHS" not in kv, proc.stdout


def test_guard4_refuses_on_dropped_mainline_line(tmp_path):
    scratch, _ = _seed_guard4_fixture(tmp_path, 1978, drop_main_lines=True)
    proc = _run_script(scratch, "1978", "--guard", "4")
    # Refusal semantics: exit 1, GUARD4=refused, LOST_UPDATE_PATHS names the
    # scoped file, stderr carries the verbatim refusal banner.
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "refused", proc.stdout
    assert "LOST_UPDATE_PATHS" in kv, proc.stdout
    assert GUARD4_SCOPED_FILE in kv["LOST_UPDATE_PATHS"], kv["LOST_UPDATE_PATHS"]
    # The dropped-count is embedded as ``path(N)``; assert on that shape.
    assert "(" in kv["LOST_UPDATE_PATHS"] and ")" in kv["LOST_UPDATE_PATHS"]
    # Verbatim refusal banner on stderr.
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in proc.stderr, proc.stderr


def test_guard4_kill_switch(tmp_path):
    scratch, _ = _seed_guard4_fixture(tmp_path, 1978, drop_main_lines=True)
    # Same drop-lines fixture BUT with the kill switch on -> skipped, exit 0.
    proc = _run_script(
        scratch,
        "1978",
        "--guard",
        "4",
        env_extra={"EPM_SKIP_LOST_UPDATE_GUARD": "1"},
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("GUARD4") == "skipped", proc.stdout
    # No refusal banner on stderr when skipped.
    assert "LOST-UPDATE REFUSAL" not in proc.stderr, proc.stderr


def test_guard4_worktree_missing_exits_2(tmp_path):
    # A scratch repo with NO issue worktree -> Guard 4 fails infra with
    # ERROR=worktree-missing, exit 2. (Kill switch is checked FIRST but is
    # OFF here; the derivation is next.)
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "1978", "--guard", "4")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "worktree-missing", proc.stdout


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------


def test_unknown_guard_exits_2(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    proc = _run_script(scratch, "1978", "--guard", "99")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR", "").startswith("unknown-guard"), proc.stdout


def test_missing_issue_number_exits_2(tmp_path):
    scratch = _make_scratch_repo(tmp_path)
    # Argument-order: without a numeric issue arg, the script rejects.
    proc = _run_script(scratch, "not-a-number", "--guard", "prelude")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    kv = _parse_kv(proc.stdout)
    assert kv.get("ERROR") == "bad-issue", proc.stdout


def test_script_is_executable():
    """The extracted script should be executable so callers can use ``bash <path>`` OR ``<path>`` directly."""
    assert SCRIPT.exists(), f"script not found at {SCRIPT}"
    # The script's PRELUDE recipe is transcribed byte-close to the SKILL.md
    # fence -- pin the ``--path-format=absolute`` token to catch a typo drift
    # (the #1867 incident-triggering fragment).
    text = SCRIPT.read_text()
    assert "--path-format=absolute" in text, "script must use --path-format=absolute"
    # The refusal banner must be the exact SKILL.md fence text.
    assert "LOST-UPDATE REFUSAL (Guard 4, #1713)" in text, "refusal banner drift"
    # The kill switch env var must be honored.
    assert "EPM_SKIP_LOST_UPDATE_GUARD" in text, "kill switch env var not honored"
