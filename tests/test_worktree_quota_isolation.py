"""Per-task ext4 project-quota isolation tests (#681 Must-Fix #4(a)).

Two contracts:

  * The kernel EDQUOT predicate (``worktree_quota.quota_admits``) is PER-PROJECT
    — a project at its hard cap returns deny for a further write WHILE another
    project under its cap returns admit on the SAME simulated data disk. This
    certifies the per-TASK (not just per-device) bound v1 lacked: one runaway
    task EDQUOTs without starving any other task.
  * ``new_worktree.sh`` assigns project id == issue number with the
    ``EPS_ISSUE_DISK_CAP_GB`` hard cap at worktree creation (via the
    ``EPS_WORKTREE_QUOTA_CMD`` CI seam), and refuses loudly when the data-disk
    bind is required-but-absent.

The shell helper is exercised against a throwaway git repo in ``tmp_path``,
modeled on ``tests/test_sparse_worktree.py``. The quota model is loaded via
importlib like ``tests/test_vm_disk_guard.py``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
HELPER = _SCRIPTS / "new_worktree.sh"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


wq = _load("worktree_quota")

_GIB = 2**30

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=check, capture_output=True, text=True, env=_GIT_ENV
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Throwaway repo with the minimal layout new_worktree.sh needs."""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / ".gitignore").write_text(".env\n")
    (main / "CLAUDE.md").write_text("rules\n")
    (main / "src").mkdir()
    (main / "src/x.py").write_text("X = 1\n")
    _git(main, "add", ".gitignore", "CLAUDE.md", "src/x.py")
    _git(main, "commit", "-q", "-m", "seed")
    (main / ".env").write_text("KEY=1\n")
    return main


def _run_helper(
    repo: Path, wt: Path, branch: str, *extra: str, env: dict | None = None, check=True
):
    return subprocess.run(
        ["bash", str(HELPER), str(wt), branch, *extra],
        cwd=str(repo),
        check=check,
        capture_output=True,
        text=True,
        env={**_GIT_ENV, **(env or {})},
    )


# ─── the starvation predicate (per-project, the v1 gap) ──────────────────────


def test_one_issue_at_cap_cannot_starve_another():
    """Issue-A at its cap returns EDQUOT (deny) for a further write WHILE issue-B
    (separate project id, under cap) returns admit — on the SAME disk with shared
    free space. The per-TASK bound: one runaway task cannot consume another's."""
    cap = 128 * _GIB
    # Issue A (projid 658) is at its cap; a 1-byte further write is denied.
    assert wq.quota_admits(658, 1, project_used_bytes=cap, project_cap_bytes=cap) is False
    assert wq.quota_admits(658, _GIB, project_used_bytes=cap, project_cap_bytes=cap) is False
    # Issue B (projid 700) is well under its OWN cap → admitted, on the same disk.
    assert wq.quota_admits(700, _GIB, project_used_bytes=10 * _GIB, project_cap_bytes=cap) is True
    # A's at-cap state is INDEPENDENT of B's headroom (per-project, not per-device).
    assert wq.quota_admits(700, 50 * _GIB, project_used_bytes=10 * _GIB, project_cap_bytes=cap) is (
        True
    )


def test_quota_admits_boundary_exact_fit():
    cap = 128 * _GIB
    # Exactly filling the cap is admitted; one byte over is denied.
    assert wq.quota_admits(658, 8 * _GIB, project_used_bytes=120 * _GIB, project_cap_bytes=cap) is (
        True
    )
    assert (
        wq.quota_admits(658, 8 * _GIB + 1, project_used_bytes=120 * _GIB, project_cap_bytes=cap)
        is False
    )


def test_quota_admits_unbounded_default_project():
    # Project id 0 (the managed pin + tiny worktrees) is unbounded — always admits.
    assert wq.quota_admits(0, 999 * _GIB, project_used_bytes=999 * _GIB, project_cap_bytes=0) is (
        True
    )
    # A non-positive cap also means "no cap".
    assert wq.quota_admits(658, 999 * _GIB, project_used_bytes=0, project_cap_bytes=0) is True


def test_project_id_is_issue_number():
    assert wq.issue_project_id(658) == 658
    with pytest.raises(ValueError):
        wq.issue_project_id(0)


# ─── cap default + env override (shared with new_worktree.sh) ────────────────


def test_issue_cap_env_override(monkeypatch):
    monkeypatch.delenv("EPS_ISSUE_DISK_CAP_GB", raising=False)
    assert wq.issue_disk_cap_gb() == wq.DEFAULT_ISSUE_DISK_CAP_GB == 128
    monkeypatch.setenv("EPS_ISSUE_DISK_CAP_GB", "256")
    assert wq.issue_disk_cap_gb() == 256
    # Garbled / non-positive → default (fail-soft).
    monkeypatch.setenv("EPS_ISSUE_DISK_CAP_GB", "nonsense")
    assert wq.issue_disk_cap_gb() == 128
    monkeypatch.setenv("EPS_ISSUE_DISK_CAP_GB", "0")
    assert wq.issue_disk_cap_gb() == 128


# ─── new_worktree.sh assigns projid + cap via the CI seam ────────────────────


def test_new_worktree_assigns_project_id_and_cap(repo: Path, tmp_path: Path):
    """With the quota assignment opted in + the CI seam capturing the args, the
    helper assigns projid == issue number and the EPS_ISSUE_DISK_CAP_GB cap
    (in KiB)."""
    wt = tmp_path / "wt"
    capture = tmp_path / "quota-args.txt"
    # The seam receives "<projid> <cap_kb> <path>".
    env = {
        "EPS_WORKTREE_ASSIGN_QUOTA": "1",
        "EPS_WORKTREE_QUOTA_CMD": f"printf '%s %s %s\\n' >> {capture}",
        "EPS_ISSUE_DISK_CAP_GB": "128",
    }
    res = _run_helper(repo, wt, "issue-658", "--issue", "658", env=env)
    assert res.returncode == 0, res.stderr
    recorded = capture.read_text().split()
    assert recorded[0] == "658", "projid must be the issue number"
    assert recorded[1] == str(128 * 1024 * 1024), "cap_kb must be 128 GB in KiB"
    assert recorded[2] == str(wt.resolve())


def test_new_worktree_quota_cap_env_override(repo: Path, tmp_path: Path):
    wt = tmp_path / "wt"
    capture = tmp_path / "quota-args.txt"
    env = {
        "EPS_WORKTREE_ASSIGN_QUOTA": "1",
        "EPS_WORKTREE_QUOTA_CMD": f"printf '%s %s %s\\n' >> {capture}",
        "EPS_ISSUE_DISK_CAP_GB": "256",
    }
    res = _run_helper(repo, wt, "issue-660", "--issue", "660", env=env)
    assert res.returncode == 0, res.stderr
    recorded = capture.read_text().split()
    assert recorded[1] == str(256 * 1024 * 1024), "cap_kb must follow EPS_ISSUE_DISK_CAP_GB"


def test_new_worktree_no_quota_when_not_opted_in(repo: Path, tmp_path: Path):
    """Default (no EPS_WORKTREE_ASSIGN_QUOTA): the quota assignment is a no-op —
    a clean before/without-cutover state. The helper still creates the worktree."""
    wt = tmp_path / "wt"
    capture = tmp_path / "quota-args.txt"
    env = {"EPS_WORKTREE_QUOTA_CMD": f"printf '%s %s %s\\n' >> {capture}"}
    res = _run_helper(repo, wt, "issue-661", "--issue", "661", env=env)
    assert res.returncode == 0, res.stderr
    assert (wt / "src/x.py").is_file()
    assert not capture.exists(), "quota seam must NOT run when not opted in"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
