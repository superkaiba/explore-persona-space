"""Issue #2479 U4 — axis-freeze guard branches (plan §4 Step 3, unit-tested).

Exercises `issue1345_story_char_ladder_fill.assert_axis_freeze_guard` against
THROWAWAY git repos in tmp_path (never this checkout): missing freeze /
untracked freeze / uncommitted modification / committed-pass / stale ladder
JSON (both the untracked-mtime and tracked-commit-time branches), plus the
`--pilot-outdir` + panel-cell refusal and the `--guard-selftest` wrapper.

Hermetic: tmp git repos only; no network, no stores loaded (every exercised
branch fires before any stage runs).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_story_char_ladder_fill as lf  # noqa: E402


def _git(repo: Path, *argv: str, date: str | None = None) -> str:
    env = dict(os.environ)
    if date:
        env["GIT_COMMITTER_DATE"] = date
        env["GIT_AUTHOR_DATE"] = date
    r = subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True, env=env)
    assert r.returncode == 0, (argv, r.stderr)
    return r.stdout.strip()


def _mk_repo(base: Path) -> Path:
    base.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=base, check=True)
    _git(base, "config", "user.email", "test@example.invalid")
    _git(base, "config", "user.name", "freeze-guard-test")
    (base / "README.md").write_text("fixture\n")
    _git(base, "add", "--", "README.md")
    _git(base, "commit", "-q", "-m", "init", "--", "README.md")
    return base


def _commit_freeze(repo: Path, date: str | None = None) -> str:
    freeze = repo / lf.I2479_FREEZE_REL
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps({"issue": 2479, "fixture": True}) + "\n")
    _git(repo, "add", "--", lf.I2479_FREEZE_REL)
    _git(repo, "commit", "-q", "-m", "fixture freeze", "--", lf.I2479_FREEZE_REL, date=date)
    return _git(repo, "rev-parse", "HEAD")


def test_guard_refuses_missing_freeze(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    with pytest.raises(RuntimeError, match="does not exist"):
        lf.assert_axis_freeze_guard(repo)


def test_guard_refuses_untracked_freeze(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    freeze = repo / lf.I2479_FREEZE_REL
    freeze.parent.mkdir(parents=True)
    freeze.write_text("{}\n")  # exists on disk, never committed
    with pytest.raises(RuntimeError, match="not committed"):
        lf.assert_axis_freeze_guard(repo)


def test_guard_passes_committed_freeze(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    sha = _commit_freeze(repo)
    assert lf.assert_axis_freeze_guard(repo) == sha
    assert len(sha) == 40


def test_guard_refuses_uncommitted_modification(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    _commit_freeze(repo)
    (repo / lf.I2479_FREEZE_REL).write_text('{"tampered": true}\n')
    with pytest.raises(RuntimeError, match="uncommitted modifications"):
        lf.assert_axis_freeze_guard(repo)


def test_guard_refuses_stale_untracked_ladder_by_mtime(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    sha = _commit_freeze(repo)
    freeze_ts = int(_git(repo, "show", "-s", "--format=%ct", sha))
    prod = repo / lf.I2479_PROD_OUT_REL
    prod.mkdir(parents=True)
    stale = prod / "ladder_old.json"
    stale.write_text("{}\n")
    os.utime(stale, (freeze_ts - 1000, freeze_ts - 1000))
    with pytest.raises(RuntimeError, match="predate the freeze commit"):
        lf.assert_axis_freeze_guard(repo)


def test_guard_refuses_stale_tracked_ladder_by_commit_time(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    # Ladder committed at an EARLIER committer date than the freeze commit —
    # pinned dates so the strict < comparison is deterministic (1s git grain).
    prod = repo / lf.I2479_PROD_OUT_REL
    prod.mkdir(parents=True)
    ladder_rel = f"{lf.I2479_PROD_OUT_REL}/ladder_prefreeze.json"
    (repo / ladder_rel).write_text("{}\n")
    _git(repo, "add", "--", ladder_rel)
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        "pre-freeze ladder",
        "--",
        ladder_rel,
        date="2026-01-01T00:00:00Z",
    )
    _commit_freeze(repo, date="2026-01-02T00:00:00Z")
    with pytest.raises(RuntimeError, match="predate the freeze commit"):
        lf.assert_axis_freeze_guard(repo)


def test_guard_passes_fresh_untracked_ladder(tmp_path: Path) -> None:
    repo = _mk_repo(tmp_path / "r")
    _commit_freeze(repo, date="2026-01-02T00:00:00Z")
    prod = repo / lf.I2479_PROD_OUT_REL
    prod.mkdir(parents=True)
    (prod / "ladder_fresh.json").write_text("{}\n")  # mtime = now, post-freeze
    assert lf.assert_axis_freeze_guard(repo)


def test_pilot_outdir_with_panel_cell_refused(tmp_path: Path, monkeypatch) -> None:
    # --pilot-outdir + a char_2479_* cell is a REFUSAL (argparse error, rc=2)
    # BEFORE any dir creation / store loading — pilot mode must never become a
    # panel-cell freeze-guard bypass.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1345_story_char_ladder_fill.py",
            "--pilot-outdir",
            str(tmp_path / "pilot"),
            "--stage",
            "cells",
            "--cells",
            "char_2479_iris_op",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        lf.main()
    assert exc.value.code == 2
    assert not (tmp_path / "pilot").exists()  # refusal fires before mkdir


def test_commit_and_post_commits_pushes_and_posts_marker(tmp_path: Path) -> None:
    """The --commit path REAL body: add/commit by explicit path, push to a bare
    remote, then post the axis-frozen marker via the resolved main checkout.
    The ONLY fake is at the external process boundary — a stub task.py that
    records its argv (signature-conformant by construction: it receives the
    real subprocess argv)."""
    import issue2479_freeze_axis as fz

    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(bare), str(clone)], check=True)
    _git(clone, "config", "user.email", "t@example.invalid")
    _git(clone, "config", "user.name", "freeze-test")
    (clone / "README.md").write_text("x\n")
    _git(clone, "add", "--", "README.md")
    _git(clone, "commit", "-q", "-m", "init", "--", "README.md")
    branch = _git(clone, "rev-parse", "--abbrev-ref", "HEAD")
    _git(clone, "push", "-q", "origin", branch)

    freeze = clone / lf.I2479_FREEZE_REL
    freeze.parent.mkdir(parents=True)
    freeze.write_text('{"issue": 2479, "fixture": true}\n')
    calls = tmp_path / "task_calls.jsonl"
    stub = clone / "scripts" / "task.py"
    stub.parent.mkdir()
    stub.write_text(
        f"import json, sys\nopen({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )

    sha = fz.commit_and_post(freeze, clone, issue=2479)
    assert len(sha) == 40
    # Committed at the clone tip AND pushed: the bare remote's branch tip == sha.
    assert _git(clone, "rev-parse", "HEAD") == sha
    assert _git(bare, "rev-parse", branch) == sha
    # The freeze file is what the commit carries (explicit-path commit).
    assert lf.I2479_FREEZE_REL in _git(clone, "show", "--name-only", "--format=", sha)
    # The marker posted through the stubbed main-checkout task.py with the
    # exact contract argv.
    argv = json.loads(calls.read_text().splitlines()[0])
    assert argv == ["post-marker", "2479", "epm:progress", "--note", f"axis-frozen commit={sha}"]


def test_guard_selftest_all_branches_pass(capsys) -> None:
    rc = lf._guard_selftest()
    out = capsys.readouterr().out
    assert rc == 0
    for branch in ("refuse-missing-freeze", "pass-committed-freeze", "refuse-stale-ladder"):
        assert f"[guard-selftest] branch={branch} result=PASS" in out, out
