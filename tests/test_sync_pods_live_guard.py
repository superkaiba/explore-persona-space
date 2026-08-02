"""Live-run-safe fleet pod code sync (task #1893; origin incident #1776).

Pins + functional fixtures for ``scripts/pod_code_sync.sh`` and its consumers
(``sync_pods.sh``, ``sync_env.sh``, ``fleet_health.py --fix``):

* Pins: no unconditional ``origin main`` pull remains in the three sync shell
  scripts; ``sync_pods.sh`` pipes the shared body via the literal
  ``bash -s < "$SCRIPT_DIR/pod_code_sync.sh"`` composition (producer-side pin
  — the functional tests never exercise the ssh string); ``sync_env.sh``
  gates on the ``SYNC-SKIPPED (live workload`` prefix and applies the #1401
  git-auth grep to the CODE-SYNC call's captured output; ``pod_code_sync.sh``
  runs the live-workload probe BEFORE any ``git pull``; ``fleet_health.py``'s
  ``--fix`` git legs carry the live-probe + non-main-branch guards.
* Functional (tmp_path git fixtures, ``EPS_SYNC_REPO_DIR`` /
  ``EPS_SYNC_LOG_DIR`` env overrides, no ssh): (a) live pid => skip, repo
  untouched; (b) unparseable pid content => skip; (c) empty + dead pid files
  on an issue-branch clone => pulls the clone's OWN branch and a diverging
  origin ``main`` commit does NOT check out; main-branch clone keeps today's
  behavior; (d) detached HEAD => loud skip; (e) deleted origin branch =>
  NONZERO exit (loud failure, pinned so a future edit cannot silently
  downgrade it).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
POD_CODE_SYNC = SCRIPTS / "pod_code_sync.sh"
SYNC_PODS = SCRIPTS / "sync_pods.sh"
SYNC_ENV = SCRIPTS / "sync_env.sh"
FLEET_HEALTH = SCRIPTS / "fleet_health.py"

# Isolate every git invocation (fixture setup AND the script under test) from
# the developer's global/system git config (e.g. a global pull.rebase).
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


# ---------------------------------------------------------------------------
# Static pins
# ---------------------------------------------------------------------------


def _code_lines(path: Path) -> str:
    """The script's non-comment lines (ordering pins must not hit comments)."""
    return "\n".join(
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_no_unconditional_origin_main_pull() -> None:
    """The #1776 hazard: a hardcoded `origin main` pull on every pod."""
    for script in (SYNC_PODS, SYNC_ENV, POD_CODE_SYNC):
        text = script.read_text(encoding="utf-8")
        assert "pull --ff-only origin main" not in text, script.name
        assert "pull --rebase=merges origin main" not in text, script.name


def test_sync_pods_pipes_shared_script() -> None:
    text = SYNC_PODS.read_text(encoding="utf-8")
    assert 'bash -s < "$SCRIPT_DIR/pod_code_sync.sh"' in text


def test_sync_env_pipes_shared_script_first() -> None:
    text = SYNC_ENV.read_text(encoding="utf-8")
    assert 'bash -s < "$SCRIPT_DIR/pod_code_sync.sh"' in text
    # The code-sync call runs BEFORE the env heredoc.
    code = _code_lines(SYNC_ENV)
    assert code.index("pod_code_sync.sh") < code.index("uv sync --locked")


def test_sync_env_live_workload_gate() -> None:
    """A live-workload skip from the code-sync call skips the pod entirely."""
    text = SYNC_ENV.read_text(encoding="utf-8")
    assert "SYNC-SKIPPED (live workload" in text


def test_sync_env_auth_grep_applies_to_code_sync_output() -> None:
    """#1401: the git-auth classification greps the CODE-SYNC call's captured
    output — the git pull lives there now, not in the env heredoc."""
    text = SYNC_ENV.read_text(encoding="utf-8")
    auth_lines = [ln for ln in text.splitlines() if "returned error: 40[13]" in ln]
    assert auth_lines, "the #1401 git-auth classification grep is gone from sync_env.sh"
    assert all("code_output" in ln for ln in auth_lines), auth_lines


def test_pod_code_sync_probe_before_pull() -> None:
    text = POD_CODE_SYNC.read_text(encoding="utf-8")
    # Live-workload probe strictly precedes any git pull (non-comment lines).
    code = _code_lines(POD_CODE_SYNC)
    assert code.index("kill -0") < code.index("git pull")
    assert "SYNC-SKIPPED (live workload" in text
    assert "SYNC-SKIPPED (detached HEAD" in text
    # Branch-aware pull: the pulled ref is the clone's OWN branch.
    assert 'git pull --ff-only origin "$branch"' in text
    assert 'git pull --rebase=merges origin "$branch"' in text


def test_fleet_health_fix_git_legs_guarded() -> None:
    text = FLEET_HEALTH.read_text(encoding="utf-8")
    assert "def _live_workload_probe" in text
    assert "def _git_fix_skip_reason" in text
    # Skip-on-doubt predicates mirror pod_code_sync.sh.
    assert "issue-*.pid" in text
    assert "kill -0" in text
    fix_pod_src = text.split("def fix_pod(", 1)[1]
    assert "_git_fix_skip_reason(" in fix_pod_src
    # The retired unconditional checkout-main ssh command must not come back
    # (the note strings say "checkout main + pull", never the command form).
    assert 'checkout main"' not in fix_pod_src


# ---------------------------------------------------------------------------
# Functional fixtures (no ssh)
# ---------------------------------------------------------------------------


@dataclass
class RepoFixture:
    origin: Path  # bare origin repo
    seed: Path  # working repo pushing to origin
    clone: Path  # the "pod" clone the sync script runs against (on issue-999)
    logs: Path  # the EPS_SYNC_LOG_DIR pid-file dir


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **GIT_ENV},
        timeout=60,
    )


def _git_out(*args: str, cwd: Path) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **GIT_ENV},
        timeout=60,
    )
    return res.stdout.strip()


@pytest.fixture()
def repo_fixture(tmp_path: Path) -> RepoFixture:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    clone = tmp_path / "clone"
    logs = tmp_path / "logs"
    logs.mkdir()
    _git("init", "--bare", str(origin), cwd=tmp_path)
    _git("init", "-b", "main", str(seed), cwd=tmp_path)
    (seed / "file.txt").write_text("v1\n")
    _git("add", "file.txt", cwd=seed)
    _git("commit", "-q", "-m", "v1", cwd=seed)
    _git("branch", "issue-999", cwd=seed)
    _git("remote", "add", "origin", str(origin), cwd=seed)
    _git("push", "-q", "origin", "main", "issue-999", cwd=seed)
    _git("clone", "-q", "--branch", "issue-999", str(origin), str(clone), cwd=tmp_path)
    return RepoFixture(origin=origin, seed=seed, clone=clone, logs=logs)


def _advance_origin(fx: RepoFixture, branch: str, filename: str, content: str, msg: str) -> None:
    """Commit + push a new file state on `branch` of the fixture origin."""
    _git("checkout", "-q", branch, cwd=fx.seed)
    (fx.seed / filename).write_text(content)
    _git("add", filename, cwd=fx.seed)
    _git("commit", "-q", "-m", msg, cwd=fx.seed)
    _git("push", "-q", "origin", branch, cwd=fx.seed)


def _run_sync(repo_dir: Path, log_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        **GIT_ENV,
        "EPS_SYNC_REPO_DIR": str(repo_dir),
        "EPS_SYNC_LOG_DIR": str(log_dir),
    }
    return subprocess.run(
        ["bash", str(POD_CODE_SYNC)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_live_pid_skips_sync(repo_fixture: RepoFixture) -> None:
    """(a) A live registered workload pid skips the sync; no git op runs."""
    fx = repo_fixture
    _advance_origin(fx, "issue-999", "file.txt", "v2-issue\n", "v2 issue")
    proc = subprocess.Popen(["sleep", "60"])
    try:
        (fx.logs / "issue-999-run.pid").write_text(f"{proc.pid}\n")
        res = _run_sync(fx.clone, fx.logs)
    finally:
        proc.kill()
        proc.wait()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SYNC-SKIPPED (live workload" in res.stdout
    assert f"pid={proc.pid}" in res.stdout
    # The pull never ran: the clone still holds v1.
    assert (fx.clone / "file.txt").read_text() == "v1\n"


def test_unparseable_pid_content_skips(repo_fixture: RepoFixture) -> None:
    """(b) Non-empty unparseable pid content => skip-on-doubt (loud, exit 0)."""
    fx = repo_fixture
    _advance_origin(fx, "issue-999", "file.txt", "v2-issue\n", "v2 issue")
    (fx.logs / "issue-999-run.pid").write_text("not-a-pid\n")
    res = _run_sync(fx.clone, fx.logs)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SYNC-SKIPPED (live workload" in res.stdout
    assert "unparseable pid content" in res.stdout
    assert (fx.clone / "file.txt").read_text() == "v1\n"


def test_dead_pid_issue_branch_pulls_own_branch(repo_fixture: RepoFixture) -> None:
    """(c) Empty + dead-pid files proceed; the clone pulls its OWN branch and
    a diverging origin main commit does NOT check out."""
    fx = repo_fixture
    _advance_origin(fx, "issue-999", "file.txt", "v2-issue\n", "v2 issue")
    _advance_origin(fx, "main", "main_only.txt", "main\n", "main diverges")
    dead = subprocess.Popen(["true"])
    dead.wait()
    (fx.logs / "issue-999-run.pid").write_text(f"{dead.pid}\n")
    (fx.logs / "issue-998-run.pid").write_text("\n")  # empty => proceed
    res = _run_sync(fx.clone, fx.logs)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SYNC-SKIPPED" not in res.stdout
    assert _git_out("rev-parse", "--abbrev-ref", "HEAD", cwd=fx.clone) == "issue-999"
    assert (fx.clone / "file.txt").read_text() == "v2-issue\n"
    assert not (fx.clone / "main_only.txt").exists()


def test_main_branch_clone_pulls_origin_main(repo_fixture: RepoFixture) -> None:
    """A main-branch clone keeps today's behavior: the branch-aware path
    pulls origin main."""
    fx = repo_fixture
    main_clone = fx.clone.parent / "main_clone"
    _git("clone", "-q", "--branch", "main", str(fx.origin), str(main_clone), cwd=fx.clone.parent)
    _advance_origin(fx, "main", "file.txt", "v2-main\n", "v2 main")
    res = _run_sync(main_clone, fx.logs)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SYNC-SKIPPED" not in res.stdout
    assert (main_clone / "file.txt").read_text() == "v2-main\n"


def test_detached_head_skips(repo_fixture: RepoFixture) -> None:
    """(d) Detached HEAD => loud skip (exit 0), no pull attempted."""
    fx = repo_fixture
    _git("checkout", "-q", "--detach", cwd=fx.clone)
    res = _run_sync(fx.clone, fx.logs)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SYNC-SKIPPED (detached HEAD" in res.stdout


def test_deleted_origin_branch_fails_loud(repo_fixture: RepoFixture) -> None:
    """(e) Deleted origin branch => NONZERO exit — a loud FAILURE, never a
    silent skip (pinned so a future edit cannot downgrade it)."""
    fx = repo_fixture
    _git("push", "-q", "origin", "--delete", "issue-999", cwd=fx.seed)
    res = _run_sync(fx.clone, fx.logs)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "SYNC-SKIPPED" not in res.stdout


def test_missing_repo_fails_loud(tmp_path: Path) -> None:
    res = _run_sync(tmp_path / "nope", tmp_path)
    assert res.returncode == 1
    assert "SYNC-FAILED (no repo" in res.stdout
