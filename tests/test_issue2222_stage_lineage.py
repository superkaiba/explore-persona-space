"""Regression tests for the issue-2222 P0 leg-A lineage gate on shallow clones.

Pre-fix, ``reassert_parent_lineage`` ran ``git log origin/main..origin/<branch>
-- <path>`` unconditionally: on the pod's depth-1 bootstrap clone the truncated
``origin/main`` ancestry excludes almost nothing from the range, so
already-MERGED commits were attributed as undispositioned branch-side edits and
the gate halted every fresh pod deterministically (flagged SHAs are ancestors
of ``origin/main`` on a full clone). These tests build tiny REAL git repos on
``tmp_path`` (an origin + a ``--depth 1`` clone) and assert:

(a) a merged-ancestor SHA flagged by the shallow range is dropped by the
    merge-base post-filter / bounded-deepen ladder (pre-fix: gate raises);
(b) a genuinely undispositioned branch-side SHA STILL raises on the same
    shallow clone (the fix must not weaken the gate);
(c)/(d) full-clone behavior is unchanged (merged passes with no shallow path
    taken; genuine branch-side raises).

Offline; seconds-fast (a handful of local git subprocesses per test).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2222_lib as lib
import issue2222_stage as stage

BRANCH = "issue-778"
WATCHED_PATH = "scripts/issue778_lib.py"
KEY = f"{BRANCH}:{WATCHED_PATH}"


def _git(cwd: Path, *args: str) -> str:
    """One fail-loud git command with a pinned test identity; returns stripped stdout."""
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=eps-test",
            "-c",
            "user.email=eps-test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "protocol.file.allow=always",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _commit_all(repo: Path, msg: str) -> str:
    """Stage everything, commit, return the new HEAD sha."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _make_origin(tmp_path: Path, *, merged: bool) -> tuple[Path, str]:
    """Origin repo: main history + an ``issue-778`` branch edit of the watched path.

    ``merged=True`` merges the branch back into main (the phantom case: the
    branch SHA is an ancestor of main on complete history); ``merged=False``
    leaves it genuinely branch-side. Filler commits land on main AFTER the
    branch point so a depth-1 clone's main tip carries none of the relevant
    ancestry (the truncation that produced the pod crash).
    """
    origin = tmp_path / "origin"
    (origin / "scripts").mkdir(parents=True)
    _git(origin, "init", "-b", "main")
    (origin / WATCHED_PATH).write_text("# v0\n")
    _commit_all(origin, "c0")
    (origin / "filler.txt").write_text("0\n")
    _commit_all(origin, "c1")
    _git(origin, "checkout", "-b", BRANCH)
    (origin / WATCHED_PATH).write_text("# v1 branch-side edit\n")
    branch_sha = _commit_all(origin, "branch edit of watched path")
    _git(origin, "checkout", "main")
    if merged:
        _git(origin, "merge", "--no-ff", "-m", "land issue-778", BRANCH)
    for i in (2, 3):
        (origin / "filler.txt").write_text(f"{i}\n")
        _commit_all(origin, f"c{i}")
    return origin, branch_sha


def _clone(origin: Path, dest: Path, *, depth1: bool) -> Path:
    """Clone origin to dest; ``depth1`` reproduces the pod bootstrap's shallow shape."""
    args = ["clone", "--quiet"]
    if depth1:
        # --no-single-branch keeps all origin/* refs resolvable (the gate probes
        # origin/<branch>) while every ref stays depth-1 truncated.
        args += ["--depth", "1", "--no-single-branch"]
    _git(origin.parent, *args, f"file://{origin}", str(dest))
    return dest


@pytest.fixture
def declared_leg_a(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the gate's disposition map to the one test key with NO declared SHAs."""
    monkeypatch.setattr(lib, "DECLARED_LEG_A", {(BRANCH, WATCHED_PATH): ()})


def test_shallow_clone_drops_merged_phantom(
    tmp_path: Path, declared_leg_a: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """(a) A merged-ancestor SHA range-flagged on a shallow clone must NOT halt the gate."""
    origin, branch_sha = _make_origin(tmp_path, merged=True)
    clone = _clone(origin, tmp_path / "shallow", depth1=True)
    assert _git(clone, "rev-parse", "--is-shallow-repository") == "true"
    # Fixture validity: the shallow range DOES mis-attribute the merged SHA as
    # branch-side (the exact pod-crash shape) before the gate runs.
    flagged = _git(
        clone,
        "log",
        "--no-merges",
        "--format=%H",
        f"origin/main..origin/{BRANCH}",
        "--",
        WATCHED_PATH,
    ).splitlines()
    assert branch_sha in flagged
    results = stage.reassert_parent_lineage(clone)  # pre-fix: raises RuntimeError here
    out = capsys.readouterr().out
    assert "shallow clone detected" in out  # the fix-engaged signal line
    assert results[KEY]["status"] == "ok"
    assert branch_sha not in results[KEY]["branch_side_shas"]


def test_shallow_clone_genuine_branch_side_still_raises(
    tmp_path: Path, declared_leg_a: None
) -> None:
    """(b) A genuinely undispositioned branch-side SHA still fails loud on a shallow clone."""
    origin, branch_sha = _make_origin(tmp_path, merged=False)
    clone = _clone(origin, tmp_path / "shallow", depth1=True)
    with pytest.raises(RuntimeError, match="leg-A re-assertion FAILED") as exc:
        stage.reassert_parent_lineage(clone)
    assert branch_sha in str(exc.value)


def test_full_clone_merged_passes_without_shallow_path(
    tmp_path: Path, declared_leg_a: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """(c) Full-clone behavior unchanged: merged history passes, shallow path never taken."""
    origin, _branch_sha = _make_origin(tmp_path, merged=True)
    clone = _clone(origin, tmp_path / "full", depth1=False)
    results = stage.reassert_parent_lineage(clone)
    assert "shallow clone detected" not in capsys.readouterr().out
    assert results[KEY]["status"] == "ok"
    assert "shallow_resolution" not in results[KEY]


def test_full_clone_genuine_branch_side_raises(tmp_path: Path, declared_leg_a: None) -> None:
    """(d) Full-clone behavior unchanged: a genuine undispositioned SHA fails loud."""
    origin, branch_sha = _make_origin(tmp_path, merged=False)
    clone = _clone(origin, tmp_path / "full", depth1=False)
    with pytest.raises(RuntimeError, match="leg-A re-assertion FAILED") as exc:
        stage.reassert_parent_lineage(clone)
    assert branch_sha in str(exc.value)
