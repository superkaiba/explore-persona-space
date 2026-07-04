"""Tests for scripts/sync_repo_root.py — the single-flight repo-root sync helper.

Fixture ``origin_and_clone`` extends the ``fake_repo`` pattern
(tests/test_task_workflow.py) with a bare origin + two clones (``local`` /
``other``) so real git divergence, untracked collisions, conflicts, stranded
autostashes, and push rejections can be reproduced end-to-end. Lock paths are
monkeypatched into ``tmp_path`` for per-test isolation (multi-process probes
receive the lock path via the child's argv, not via monkeypatch — module
monkeypatches don't survive ``spawn``).

Covers the 14 cases in plan §5 (task #904), plus the round-2 hardening cases:
exclusive rescue-dir allocation (concern ``rescue-dir-nonexclusive-overwrite``),
journal-before-action durability (concern ``sweep-before-durable-ledger``),
recorded-blob-sha rematerialization, and restore-failure containment (exit 6).
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

# ─── Import the helper module from its script path ──────────────────────────

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_repo_root.py"
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("sync_repo_root_under_test", _SCRIPT)
srr = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["sync_repo_root_under_test"] = srr
_spec.loader.exec_module(srr)  # type: ignore[union-attr]


# ─── Fixture helpers ─────────────────────────────────────────────────────────


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check
    )


def _configure(clone: Path) -> None:
    _git(clone, "config", "user.email", "test@test.test")
    _git(clone, "config", "user.name", "test")
    _git(clone, "config", "commit.gpgsign", "false")


def _write(repo: Path, rel: str, content: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _commit(repo: Path, *paths: str, msg: str = "c") -> None:
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "-q", "-m", msg)


def _backdate(path: Path, secs: float = 300.0) -> None:
    """Move a file's mtime past the sweep's fresh-mtime guard."""
    t = time.time() - secs
    os.utime(path, (t, t))


def _unmerged_paths(repo: Path) -> list[str]:
    return [
        p for p in _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.splitlines() if p
    ]


@pytest.fixture
def origin_and_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bare origin + ``local``/``other`` clones; per-test lock + rescue paths."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)],
        check=True,
        capture_output=True,
    )
    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", str(origin), str(local)], check=True, capture_output=True)
    _configure(local)
    _git(local, "commit", "-q", "--allow-empty", "-m", "init")
    _git(local, "push", "-q", "-u", "origin", "main")
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(origin), str(other)], check=True, capture_output=True)
    _configure(other)

    lock_dir = tmp_path / "locks"
    monkeypatch.setattr(srr, "ROOT_SYNC_LOCK", lock_dir / "root-sync.lock")
    monkeypatch.setattr(srr, "RESCUE_ROOT", tmp_path / "rescue")
    monkeypatch.setattr(srr.task_workflow, "LOCK_PATH", lock_dir / "task-workflow-lock")
    # A dev shell exporting a non-default husk age must not flake the husk
    # age-gate tests (per-test setenv still wins — it runs after fixture setup).
    monkeypatch.delenv("EPM_ROOT_SYNC_HUSK_AGE_S", raising=False)
    return origin, local, other


def _run(local: Path, *extra: str, capsys) -> tuple[int, dict, str]:
    """Invoke ``main`` with --json; return (exit_code, parsed report, stderr)."""
    rc = srr.main(["--repo", str(local), "--json", *extra])
    captured = capsys.readouterr()
    return rc, json.loads(captured.out), captured.err


def _worktree_digest(repo: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(repo.rglob("*")):
        if ".git" in p.parts or not p.is_file():
            continue
        out[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ─── 1. Identical collision removed, pull succeeds ──────────────────────────


def test_identical_collision_removed_pull_succeeds(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/r.json", '{"x": 1}\n')
    _commit(other, "eval_results/issue_9/r.json")
    _git(other, "push", "-q", "origin", "main")
    # Byte-identical untracked copy in local; backdated past the fresh guard.
    p = _write(local, "eval_results/issue_9/r.json", '{"x": 1}\n')
    _backdate(p)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert rep["state"] == "synced"
    assert p.read_text() == '{"x": 1}\n'
    assert "eval_results/issue_9/r.json" in _git(local, "ls-files").stdout.splitlines()
    removed = [e for e in rep["sweep"] if e["action"] == "removed"]
    assert len(removed) == 1 and removed[0]["kind"] == "identical"
    manifest = Path(rep["rescue_dir"]) / "sweep-manifest.json"
    ledger = json.loads(manifest.read_text())
    assert [e["action"] for e in ledger] == ["removed"]


# ─── 2. Differing collision rescued ──────────────────────────────────────────


def test_differing_collision_rescued(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/r.json", "ORIGIN\n")
    _commit(other, "eval_results/issue_9/r.json")
    _git(other, "push", "-q", "origin", "main")
    p = _write(local, "eval_results/issue_9/r.json", "LOCAL-DIFFERENT\n")
    _backdate(p)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert p.read_text() == "ORIGIN\n"  # tracked copy checked out
    rescued = [e for e in rep["sweep"] if e["action"] == "rescued"]
    assert len(rescued) == 1 and rescued[0]["kind"] == "differing"
    rescue_path = Path(rescued[0]["rescue_path"])
    assert rescue_path.read_text() == "LOCAL-DIFFERENT\n"
    assert rescue_path.parts[-3:] == ("eval_results", "issue_9", "r.json")


# ─── 3. Single-flight ────────────────────────────────────────────────────────


def test_single_flight_second_caller_exits_zero(origin_and_clone, capsys):
    _origin, local, _other = origin_and_clone
    srr.ROOT_SYNC_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(srr.ROOT_SYNC_LOCK, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    head_before = _git(local, "rev-parse", "HEAD").stdout
    status_before = _git(local, "status", "--porcelain=v2", "--untracked-files=all").stdout
    try:
        t0 = time.monotonic()
        rc, rep, err = _run(local, capsys=capsys)
        assert time.monotonic() - t0 < 5.0
    finally:
        os.close(fd)
    assert rc == 0
    assert rep["state"] == "in-flight"
    assert "another sync in flight — your push has NOT landed" in err
    assert _git(local, "rev-parse", "HEAD").stdout == head_before
    assert _git(local, "status", "--porcelain=v2", "--untracked-files=all").stdout == status_before


# ─── 4. Mid-rebase conflict → clean abort WITH swept-file restore ───────────


def test_conflict_clean_abort_restores_swept_files(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    _write(local, "conflict.txt", "base\n")
    _write(local, "notes.txt", "notes\n")
    _commit(local, "conflict.txt", "notes.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")

    _write(other, "conflict.txt", "origin-side\n")
    _write(other, "eval_results/issue_9/ident.json", "SAME\n")
    _write(other, "eval_results/issue_9/diff.json", "OTHER\n")
    _commit(
        other, "conflict.txt", "eval_results/issue_9/ident.json", "eval_results/issue_9/diff.json"
    )
    _git(other, "push", "-q", "origin", "main")

    _write(local, "conflict.txt", "local-side\n")
    _commit(local, "conflict.txt")
    ident = _write(local, "eval_results/issue_9/ident.json", "SAME\n")
    _backdate(ident)
    diff = _write(local, "eval_results/issue_9/diff.json", "LOCAL\n")
    _write(local, "notes.txt", "dirty-notes\n")  # uncommitted → autostash

    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 2
    gd = local / ".git"
    assert not (gd / "rebase-merge").exists()
    assert not (gd / "MERGE_HEAD").exists()
    assert "conflict.txt" in rep["conflicted_paths"]
    # Original state restored: HEAD commit + autostash-reapplied dirty file.
    assert (local / "conflict.txt").read_text() == "local-side\n"
    assert (local / "notes.txt").read_text() == "dirty-notes\n"
    # Swept files restored per the abort-restore contract.
    assert ident.read_text() == "SAME\n"  # rematerialized from origin blob
    assert diff.read_text() == "LOCAL\n"  # moved back from rescue
    assert "git worktree add --detach" in err  # scratch-worktree recipe printed


def test_conflict_report_frames_registry_conflict_as_expected(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    _write(local, "tasks/REGISTRY.json", '{"highest_id": 1}\n')
    _commit(local, "tasks/REGISTRY.json")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "tasks/REGISTRY.json", '{"highest_id": 2}\n')
    _commit(other, "tasks/REGISTRY.json")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "tasks/REGISTRY.json", '{"highest_id": 3}\n')
    _commit(local, "tasks/REGISTRY.json")

    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 2
    assert "tasks/REGISTRY.json" in rep["conflicted_paths"]
    assert "EXPECTED on incident-scale divergence" in err


# ─── 5. Stranded autostash ───────────────────────────────────────────────────


def _strand_autostash(local: Path, other: Path) -> None:
    """Verified recipe: dirty tracked file conflicting with the rebase RESULT →
    completed pull leaves ``stash@{0}: autostash`` + UU paths."""
    _write(local, "f.txt", "base\n")
    _commit(local, "f.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "f.txt", "origin\n")
    _commit(other, "f.txt")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "f.txt", "local-dirty\n")  # uncommitted
    raw = _git(local, "pull", "--rebase=merges", "--autostash", "origin", "main", check=False)
    combined = raw.stdout + raw.stderr
    assert "Applying autostash resulted in conflicts" in combined
    assert any("autostash" in line for line in _git(local, "stash", "list").stdout.splitlines())
    assert _unmerged_paths(local) == ["f.txt"]


def test_stranded_autostash_conflicting_entry_kept_and_cleared(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    _strand_autostash(local, other)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    # Rescue patch written FIRST, containing the dirty content.
    patches = list(srr.RESCUE_ROOT.glob("stash-*.patch"))
    assert len(patches) == 1
    assert "local-dirty" in patches[0].read_text()
    # Unmerged paths cleared back to HEAD; entry KEPT (apply --check dirty).
    assert _unmerged_paths(local) == []
    assert (local / "f.txt").read_text() == "origin\n"
    assert any("autostash" in line for line in _git(local, "stash", "list").stdout.splitlines())
    assert any("KEPT" in s for s in rep["stash"])


def test_stranded_autostash_clean_entry_popped(origin_and_clone, capsys):
    _origin, local, _other = origin_and_clone
    _write(local, "g.txt", "g-base\n")
    _commit(local, "g.txt")
    _git(local, "push", "-q", "origin", "main")
    _write(local, "g.txt", "g-dirty\n")
    sha = _git(local, "stash", "create").stdout.strip()
    _git(local, "stash", "store", "-m", "autostash", sha)
    _git(local, "checkout", "HEAD", "--", "g.txt")  # path-scoped reset in the fixture

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert _git(local, "stash", "list").stdout.strip() == ""  # popped
    assert (local / "g.txt").read_text() == "g-dirty\n"
    assert any("popped" in s for s in rep["stash"])
    assert list(srr.RESCUE_ROOT.glob("stash-*.patch"))  # rescue patch still on disk


def test_stranded_autostash_rescue_before_clear_seam(origin_and_clone, monkeypatch):
    """Codex-Stat MF3: an exception between the rescue-patch write and the
    path-scoped clear leaves the stash entry + patch intact."""
    _origin, local, other = origin_and_clone
    _strand_autostash(local, other)

    def _boom(repo, paths):
        raise RuntimeError("seam: injected between rescue-patch write and clear")

    monkeypatch.setattr(srr, "_clear_unmerged_paths", _boom)
    report = srr._new_report(local, dry_run=False)
    with pytest.raises(RuntimeError, match="seam"):
        srr.recover_stranded_autostash(local, report, dry_run=False, preflight_case=True)
    assert any("autostash" in line for line in _git(local, "stash", "list").stdout.splitlines())
    patches = list(srr.RESCUE_ROOT.glob("stash-*.patch"))
    assert len(patches) == 1 and "local-dirty" in patches[0].read_text()


# ─── 6. Push retry (success) ─────────────────────────────────────────────────


def _install_pre_receive(origin: Path, script: str) -> None:
    hook = origin / "hooks" / "pre-receive"
    hook.write_text(script)
    hook.chmod(0o755)


def test_push_retry_succeeds_after_one_rejection(origin_and_clone, tmp_path, capsys):
    origin, local, _other = origin_and_clone
    state = tmp_path / "rejected-once"
    _install_pre_receive(
        origin,
        f"#!/bin/sh\nif [ ! -f {state} ]; then touch {state}; "
        'echo "rejected once" >&2; exit 1; fi\nexit 0\n',
    )
    _write(local, "new.txt", "new\n")
    _commit(local, "new.txt")

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert rep["state"] == "synced"
    assert state.exists()  # the first push really was rejected
    assert _git(origin, "rev-parse", "main").stdout == _git(local, "rev-parse", "HEAD").stdout


# ─── 7. Dry-run mutates nothing ──────────────────────────────────────────────


def test_dry_run_mutates_nothing(origin_and_clone, capsys):
    origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/a.json", "A\n")
    _write(other, "eval_results/issue_9/b.json", "B\n")
    _commit(other, "eval_results/issue_9/a.json", "eval_results/issue_9/b.json")
    _git(other, "push", "-q", "origin", "main")
    a = _write(local, "eval_results/issue_9/a.json", "A\n")  # identical
    _backdate(a)
    _write(local, "eval_results/issue_9/b.json", "LOCAL-B\n")  # differing
    _write(local, "tracked-dirty.txt", "dirty\n")
    sha = _git(local, "stash", "create").stdout.strip()
    if sha:  # untracked-only trees yield no stash — keep the fixture lenient
        _git(local, "stash", "store", "-m", "autostash", sha)

    def snapshot() -> tuple:
        return (
            _git(local, "status", "--porcelain=v2", "--untracked-files=all").stdout,
            _git(local, "rev-parse", "HEAD").stdout,
            _git(local, "stash", "list").stdout,
            _worktree_digest(local),
            _git(local, "ls-remote", str(origin)).stdout,  # origin-side refs
            sorted(str(p) for p in srr.RESCUE_ROOT.rglob("*")) if srr.RESCUE_ROOT.exists() else [],
            _git(local, "config", "--list", "--local").stdout,
        )

    before = snapshot()
    rc, rep, _err = _run(local, "--dry-run", capsys=capsys)
    assert rc == 0
    assert rep["state"] == "dry-run"
    assert rep["collisions"]["identical"] == 1
    assert rep["collisions"]["differing"] == 1
    assert before == snapshot()


# ─── 8. Banned-command guard (dual) ──────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ("reset", "--hard"),
        ("reset", "--hard", "origin/main"),
        ("clean", "-f"),
        ("clean", "-fd"),
        ("clean", "-xfd"),
        ("clean", "--force"),
        ("checkout", "."),
        ("restore", "."),
        ("stash", "drop"),
        ("stash", "drop", "stash@{0}"),
    ],
)
def test_deny_list_raises_on_banned_argv(tmp_path, argv):
    with pytest.raises(srr.BannedGitInvocationError):
        srr.git(tmp_path, *argv)


def test_deny_list_allows_path_scoped_checkout():
    srr._check_banned(("checkout", "HEAD", "--", "some/path.txt"))  # must not raise
    srr._check_banned(("rebase", "--abort"))
    srr._check_banned(("stash", "pop", "stash@{0}"))


def _call_string_args(node: ast.Call) -> list[str]:
    strings: list[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            strings.append(arg.value)
        elif isinstance(arg, ast.List | ast.Tuple):
            strings.extend(
                e.value
                for e in arg.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            )
    return strings


def _banned_combo(strings: list[str]) -> str | None:
    if "reset" in strings and "--hard" in strings:
        return "reset --hard"
    if "clean" in strings and any(
        s == "--force" or (s.startswith("-") and not s.startswith("--") and "f" in s)
        for s in strings
    ):
        return "clean -f"
    for sub in ("checkout", "restore"):
        if sub in strings and "." in strings:
            return f"{sub} ."
    if "stash" in strings and "drop" in strings:
        return "stash drop"
    return None


_GIT_CALL_NAMES = {"git", "_git_argv", "run", "check_call", "check_output", "Popen", "call"}


def _scan_source_for_banned_calls(src: str) -> list[str]:
    """Argv-aware AST scan: literal banned tuples at git()/subprocess call sites."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else (func.attr if isinstance(func, ast.Attribute) else None)
        )
        if name not in _GIT_CALL_NAMES:
            continue
        combo = _banned_combo(_call_string_args(node))
        if combo:
            hits.append(f"line {node.lineno}: {combo}")
    return hits


def test_ast_scan_helper_source_has_no_banned_call(tmp_path):
    assert _scan_source_for_banned_calls(_SCRIPT.read_text()) == []


def test_ast_scan_catches_argv_form_and_skips_path_scoped():
    # Replay against synthetic banned lines — proves the argv form is caught.
    assert _scan_source_for_banned_calls('git(repo, "reset", "--hard")')
    assert _scan_source_for_banned_calls('subprocess.run(["git", "clean", "-fd"])')
    assert _scan_source_for_banned_calls('git(repo, "stash", "drop", ref)')
    # Must NOT false-fire on the legitimate path-scoped clear.
    assert not _scan_source_for_banned_calls('git(repo, "checkout", "HEAD", "--", p)')


# ─── 9. Timeout abort ────────────────────────────────────────────────────────


def test_timeout_abort_preserves_state_and_restores_swept(origin_and_clone, monkeypatch, capsys):
    _origin, local, other = origin_and_clone
    _write(local, "tracked.txt", "base\n")
    _commit(local, "tracked.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "eval_results/issue_9/ident.json", "SAME\n")
    _write(other, "eval_results/issue_9/diff.json", "OTHER\n")
    _commit(other, "eval_results/issue_9/ident.json", "eval_results/issue_9/diff.json")
    _git(other, "push", "-q", "origin", "main")
    ident = _write(local, "eval_results/issue_9/ident.json", "SAME\n")
    _backdate(ident)
    diff = _write(local, "eval_results/issue_9/diff.json", "LOCAL\n")
    _write(local, "tracked.txt", "dirty\n")  # pre-sync dirty state

    monkeypatch.setattr(srr, "_pull_argv", lambda repo: ["bash", "-c", "sleep 60"])
    rc, rep, _err = _run(local, "--timeout-s", "0.5", capsys=capsys)
    assert rc == 4
    assert not (local / ".git" / "rebase-merge").exists()
    assert (local / "tracked.txt").read_text() == "dirty\n"  # dirty state preserved
    assert ident.read_text() == "SAME\n"  # rematerialized
    assert diff.read_text() == "LOCAL\n"  # moved back
    assert rep["exit_code"] == 4


# ─── 10. Error-driven fallback sweep ─────────────────────────────────────────


def test_error_driven_fallback_sweep(origin_and_clone, monkeypatch, capsys):
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/x.json", "ORIGIN-X\n")
    _commit(other, "eval_results/issue_9/x.json")
    _git(other, "push", "-q", "origin", "main")
    x = _write(local, "eval_results/issue_9/x.json", "LOCAL-X\n")

    monkeypatch.setattr(srr, "enumerate_collisions", lambda repo: [])
    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert x.read_text() == "ORIGIN-X\n"  # tracked copy checked out after retry
    rescued = [e for e in rep["sweep"] if e["action"] == "rescued"]
    assert len(rescued) == 1
    assert Path(rescued[0]["rescue_path"]).read_text() == "LOCAL-X\n"
    assert any("fallback sweep" in m for m in rep["messages"])


def test_two_consecutive_collision_failures_exit_6(origin_and_clone, monkeypatch, capsys):
    """AC4 exit-6 coverage: the fallback sweep also failing to clear the
    collisions is the two-consecutive-failures unexpected terminal — clean
    state, both path lists reported, untracked data untouched."""
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/x.json", "ORIGIN-X\n")
    _commit(other, "eval_results/issue_9/x.json")
    _git(other, "push", "-q", "origin", "main")
    x = _write(local, "eval_results/issue_9/x.json", "LOCAL-X\n")

    # Neuter BOTH sweeps: the pull hits the real collision error twice.
    monkeypatch.setattr(srr, "sweep", lambda repo, collisions, rescue_dir, dry_run: [])
    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 6
    assert rep["exit_code"] == 6
    assert "two consecutive untracked-collision failures" in err
    assert "eval_results/issue_9/x.json" in err  # both attempts' path lists named
    assert x.read_text() == "LOCAL-X\n"  # untracked data untouched
    assert not (local / ".git" / "rebase-merge").exists()
    assert not (local / ".git" / "MERGE_HEAD").exists()


def test_parse_collision_stderr_verbatim_blob():
    blob = textwrap.dedent(
        """\
        error: The following untracked working tree files would be overwritten by checkout:
        \teval_results/issue_9/a.json
        \teval_results/issue_9/b.json
        Please move or remove them before you switch branches.
        Aborting
        error: could not detach HEAD
        """
    )
    assert srr.parse_collision_stderr(blob) == [
        "eval_results/issue_9/a.json",
        "eval_results/issue_9/b.json",
    ]


# ─── 11. Husk branches ───────────────────────────────────────────────────────


def _make_conflicted_rebase_husk(local: Path, other: Path) -> Path:
    _write(local, "c.txt", "base\n")
    _commit(local, "c.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "c.txt", "origin-side\n")
    _commit(other, "c.txt")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "c.txt", "local-side\n")
    _commit(local, "c.txt")
    raw = _git(local, "pull", "--rebase=merges", "origin", "main", check=False)
    assert raw.returncode != 0
    husk = local / ".git" / "rebase-merge"
    assert husk.exists()
    return husk


def test_stale_husk_auto_aborted_then_continue(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 7200
    os.utime(husk, (t, t))

    rc, rep, _err = _run(local, capsys=capsys)
    # Stale husk aborted + loud report; the run CONTINUES into the pull, which
    # re-hits the same genuine content conflict → clean exit-2 abort.
    assert any("STALE-HUSK ABORT" in m for m in rep["messages"])
    assert rc == 2
    assert not (local / ".git" / "rebase-merge").exists()
    assert "c.txt" in rep["conflicted_paths"]


def test_young_husk_untouched_exit_5(origin_and_clone, capsys):
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)

    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert husk.exists()  # untouched
    assert "young rebase-merge husk" in err


# ─── 11b. Head-name-less husk recovery (#971) ────────────────────────────────


def _stash_shaped_sha(local: Path) -> str:
    """Commit+push a tracked file, dirty it, ``git stash create``, restore worktree.

    Returns a stash-shaped commit sha (>= 2 parents) — the same object shape a
    crashed autostash-pull leaves behind in ``<state-dir>/autostash``.
    """
    _write(local, "dirty.txt", "base\n")
    _commit(local, "dirty.txt")
    _git(local, "push", "-q", "origin", "main")
    _write(local, "dirty.txt", "base\nuncommitted\n")
    sha = _git(local, "stash", "create", "autostash-sim").stdout.strip()
    _git(local, "checkout", "--", "dirty.txt")
    assert sha
    return sha


def _make_headnameless_husk(
    local: Path, dirname: str = "rebase-merge", autostash: str | None = None, stale: bool = True
) -> Path:
    """Synthesize the #971 incident husk: a rebase state dir with NO head-name."""
    husk = local / ".git" / dirname
    husk.mkdir()
    if autostash is not None:
        (husk / "autostash").write_text(autostash + "\n")
    if stale:
        t = time.time() - 7200
        os.utime(husk, (t, t))
    return husk


def _make_conflicted_am_state(local: Path) -> Path:
    """Genuinely conflicted ``git am`` (the MF1 fact pattern): a patch from a
    side branch applied onto diverged main leaves ``.git/rebase-apply`` with
    ``applying`` + ``patch``, and NO ``head-name``."""
    _write(local, "am.txt", "base\n")
    _commit(local, "am.txt")
    _git(local, "checkout", "-q", "-b", "patchsrc")
    _write(local, "am.txt", "patch-side\n")
    _commit(local, "am.txt")
    patch = _git(local, "format-patch", "-1", "--stdout").stdout
    _git(local, "checkout", "-q", "main")
    _write(local, "am.txt", "main-side\n")
    _commit(local, "am.txt")
    proc = subprocess.run(
        ["git", "-C", str(local), "am"], input=patch, capture_output=True, text=True, check=False
    )
    assert proc.returncode != 0
    state = local / ".git" / "rebase-apply"
    assert state.is_dir() and (state / "applying").exists()
    assert not (state / "head-name").exists()
    t = time.time() - 7200
    os.utime(state, (t, t))
    return state


def test_headnameless_husk_valid_autostash_rescued_then_archived(origin_and_clone, capsys):
    """(a) The incident repro: valid autostash rescued, husk archived, exit 0."""
    _origin, local, _other = origin_and_clone
    sha = _stash_shaped_sha(local)
    _make_headnameless_husk(local, autostash=sha)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert not (local / ".git" / "rebase-merge").exists()
    archived = list(srr.RESCUE_ROOT.glob("*/rebase-merge"))
    assert len(archived) == 1
    assert (archived[0] / "autostash").read_text().strip() == sha
    assert any("STALE-HUSK ARCHIVED" in m for m in rep["messages"])
    assert any(r.startswith("husk-rescue: stored autostash") for r in rep["stash"])
    # The stranded-autostash pass consumed the rescued entry in the SAME run:
    # the uncommitted content is back in the worktree and the stash is empty.
    assert "uncommitted" in (local / "dirty.txt").read_text()
    assert _git(local, "stash", "list").stdout.strip() == ""
    assert rep["actions_performed"] is True


@pytest.mark.parametrize("dirname", ["rebase-merge", "rebase-apply"])
@pytest.mark.parametrize("autostash", [None, ""])
def test_headnameless_husk_no_autostash_archived(origin_and_clone, capsys, dirname, autostash):
    """(b) Absent/empty autostash: husk ARCHIVED (never deleted), exit 0."""
    _origin, local, _other = origin_and_clone
    _make_headnameless_husk(local, dirname=dirname, autostash=autostash)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert not (local / ".git" / dirname).exists()
    assert len(list(srr.RESCUE_ROOT.glob(f"*/{dirname}"))) == 1
    msg = next(m for m in rep["messages"] if "STALE-HUSK ARCHIVED" in m)
    expected = "no autostash file present" if autostash is None else "autostash file empty"
    assert expected in msg
    assert _git(local, "stash", "list").stdout.strip() == ""


def test_young_headnameless_husk_exit_5(origin_and_clone, capsys):
    """(c) A YOUNG head-name-less husk still exits 5, untouched (age gate)."""
    _origin, local, _other = origin_and_clone
    husk = _make_headnameless_husk(local, stale=False)

    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert husk.exists()  # untouched
    assert "young rebase-merge husk" in err


@pytest.mark.parametrize(
    "content_kind, reason_substr",
    [
        ("non-hex", "not a 40-hex sha"),
        ("missing-object", "does not resolve to a commit"),
        ("ordinary-commit", "not stash-shaped"),
    ],
)
def test_headnameless_husk_nonstorable_autostash_preserved(
    origin_and_clone, capsys, content_kind, reason_substr
):
    """(e) Non-storable autostash content is preserved verbatim, NEVER stored."""
    _origin, local, _other = origin_and_clone
    content = {
        "non-hex": "not-a-sha",
        "missing-object": "deadbeef" + "0" * 32,
        "ordinary-commit": _git(local, "rev-parse", "HEAD").stdout.strip(),
    }[content_kind]
    _make_headnameless_husk(local, autostash=content)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert not (local / ".git" / "rebase-merge").exists()
    assert list(srr.RESCUE_ROOT.glob("*/rebase-merge"))  # archived intact
    msg = next(m for m in rep["messages"] if "STALE-HUSK ARCHIVED" in m)
    assert reason_substr in msg
    rescue_files = list(srr.RESCUE_ROOT.glob("husk-autostash-*.txt"))
    assert len(rescue_files) == 1
    assert rescue_files[0].read_text() == content + "\n"  # verbatim
    # Nothing was stored — pins the C1 downstream stash-show wedge guard.
    assert _git(local, "stash", "list").stdout.strip() == ""


def test_headnameless_husk_store_failure_blocks_archival(origin_and_clone, monkeypatch, capsys):
    """(f) A store failure on a storable commit exits 6 with the husk KEPT."""
    _origin, local, _other = origin_and_clone
    sha = _stash_shaped_sha(local)
    husk = _make_headnameless_husk(local, autostash=sha)

    real_git = srr.git

    def fake_git(repo, *args, **kwargs):
        if args[:2] == ("stash", "store"):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="simulated store failure")
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(srr, "git", fake_git)
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 6
    assert husk.exists()  # KEPT — rescue failure blocks the move
    assert (husk / "autostash").read_text().strip() == sha
    assert sha in err  # full sha named
    assert f"git stash store -m autostash {sha}" in err  # manual recipe


def test_headnameless_husk_dry_run_reports_distinctly(origin_and_clone, capsys):
    """(g) Dry-run reports the head-name-less case distinctly; mutates nothing."""
    _origin, local, _other = origin_and_clone
    sha = _stash_shaped_sha(local)
    husk = _make_headnameless_husk(local, autostash=sha)

    rc, rep, _err = _run(local, "--dry-run", capsys=capsys)
    assert rc == 0
    assert rep["state"] == "dry-run"
    msg = next(m for m in rep["messages"] if "DRY-RUN: stale head-name-less" in m)
    assert "would be rescued" in msg
    assert husk.exists()  # untouched
    assert _git(local, "stash", "list").stdout.strip() == ""
    assert not srr.RESCUE_ROOT.exists()  # rescue root untouched


def test_headnameless_husk_already_stashed_idempotent(origin_and_clone, capsys):
    """(h) Re-run after a crashed prior recovery (store succeeded, move crashed):
    the reflog containment check reports idempotently, no duplicate entry."""
    _origin, local, _other = origin_and_clone
    sha = _stash_shaped_sha(local)
    _make_headnameless_husk(local, autostash=sha)
    _git(local, "stash", "store", "-m", "autostash", sha)
    assert len(_git(local, "stash", "list").stdout.strip().splitlines()) == 1

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert not (local / ".git" / "rebase-merge").exists()
    assert list(srr.RESCUE_ROOT.glob("*/rebase-merge"))
    assert any("already present in the stash reflog" in r for r in rep["stash"])
    # Exactly one entry existed before the stranded pass consumed it (no dup).
    assert _git(local, "stash", "list").stdout.strip() == ""
    assert "uncommitted" in (local / "dirty.txt").read_text()


def test_am_state_refused_intact(origin_and_clone, capsys):
    """(i, MF1) A stale, genuinely conflicted `git am` state is REFUSED (exit 5)
    and survives byte-intact — its patch data + --continue capability kept."""
    _origin, local, _other = origin_and_clone
    state = _make_conflicted_am_state(local)
    before = {p.name: p.read_bytes() for p in state.iterdir() if p.is_file()}

    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert state.is_dir()
    assert (state / "applying").exists()
    assert (state / "patch").exists()
    after = {p.name: p.read_bytes() for p in state.iterdir() if p.is_file()}
    assert after == before  # INTACT, including the patch file
    assert "git am --continue" in err and "git am --abort" in err
    assert not srr.RESCUE_ROOT.exists()
    assert _git(local, "stash", "list").stdout.strip() == ""


def test_abort_failure_with_headname_present_refuses(origin_and_clone, monkeypatch, capsys):
    """(j, MF2) Abort fails but head-name IS present: explicit exit-6 refusal,
    husk KEPT — the one guard between recovery and a real rebase state."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 7200
    os.utime(husk, (t, t))

    real_git = srr.git

    def fake_git(repo, *args, **kwargs):
        if args[:2] == ("rebase", "--abort"):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="simulated abort failure")
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(srr, "git", fake_git)
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 6
    assert husk.exists()  # KEPT
    assert (husk / "head-name").exists()
    assert "not the known un-abortable" in err
    assert "refusing to touch it" in err


# ─── 12. HEAD ≠ main precondition ────────────────────────────────────────────


def test_head_not_main_exit_5_zero_mutations(origin_and_clone, capsys):
    _origin, local, _other = origin_and_clone
    _git(local, "checkout", "-q", "-b", "feature")
    before = (
        _git(local, "rev-parse", "HEAD").stdout,
        _git(local, "status", "--porcelain=v2", "--untracked-files=all").stdout,
        _worktree_digest(local),
    )
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert "not 'main'" in err
    assert before == (
        _git(local, "rev-parse", "HEAD").stdout,
        _git(local, "status", "--porcelain=v2", "--untracked-files=all").stdout,
        _worktree_digest(local),
    )


# ─── 13. Concurrent-writer exclusion (kill-criterion-1) ─────────────────────


def test_writer_holding_task_lock_blocks_helper_then_completes(origin_and_clone, capsys):
    _origin, local, _other = origin_and_clone
    lock2 = Path(srr.task_workflow.LOCK_PATH)
    lock2.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock2, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rc_holder: list[int] = []
    t = threading.Thread(target=lambda: rc_holder.append(srr.main(["--repo", str(local)])))
    t.start()
    time.sleep(1.0)
    assert t.is_alive()  # blocked on the task-workflow lock
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    t.join(timeout=30)
    assert rc_holder == [0]
    capsys.readouterr()  # drain the thread's report


def test_writer_holding_task_lock_past_bound_exits_5(origin_and_clone, monkeypatch, capsys):
    _origin, local, _other = origin_and_clone
    monkeypatch.setenv("EPM_ROOT_SYNC_LOCK2_WAIT_S", "1")
    lock2 = Path(srr.task_workflow.LOCK_PATH)
    lock2.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock2, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc, _rep, err = _run(local, capsys=capsys)
    finally:
        os.close(fd)
    assert rc == 5
    assert "task-workflow lock still held" in err


_PROBE_SNIPPET = """
import fcntl, os
fd = os.open({lock!r}, os.O_WRONLY | os.O_CREAT, 0o600)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("ACQUIRED")
except BlockingIOError:
    print("BLOCKED")
"""


def test_helper_mutation_window_excludes_external_writer_process(
    origin_and_clone, monkeypatch, capsys
):
    """While the helper's mutation window is open, a task_workflow._locked()-
    style writer in ANOTHER PROCESS cannot take the lock (lock path passed via
    the child's argv — module monkeypatches don't survive process boundaries)."""
    _origin, local, other = origin_and_clone
    _write(other, "z.txt", "z\n")
    _commit(other, "z.txt")
    _git(other, "push", "-q", "origin", "main")  # behind>0 so the pull leg runs
    monkeypatch.setattr(srr, "_pull_argv", lambda repo: ["bash", "-c", "sleep 2"])
    lock2 = str(srr.task_workflow.LOCK_PATH)

    rc_holder: list[int] = []
    t = threading.Thread(target=lambda: rc_holder.append(srr.main(["--repo", str(local)])))
    t.start()
    try:
        saw_blocked = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and t.is_alive():
            probe = subprocess.run(
                [sys.executable, "-c", _PROBE_SNIPPET.format(lock=lock2)],
                capture_output=True,
                text=True,
                check=True,
            )
            if probe.stdout.strip() == "BLOCKED":
                saw_blocked = True
                break
            time.sleep(0.05)
    finally:
        t.join(timeout=30)
    assert saw_blocked, "external writer was never excluded during the mutation window"
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE_SNIPPET.format(lock=lock2)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "ACQUIRED"  # released after the sync
    capsys.readouterr()


# ─── 14. Exit-3 push failure ─────────────────────────────────────────────────


def test_push_rejected_twice_exits_3(origin_and_clone, capsys):
    origin, local, _other = origin_and_clone
    _install_pre_receive(origin, '#!/bin/sh\necho "always rejected" >&2\nexit 1\n')
    _write(local, "new.txt", "new\n")
    _commit(local, "new.txt")

    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 3
    assert not (local / ".git" / "rebase-merge").exists()
    assert "push failed after the one retry" in err
    assert "always rejected" in err


# ─── Round 2: exclusive rescue-dir allocation (concern 1) ────────────────────


def test_allocate_rescue_dir_exclusive_bounded_retry(tmp_path, monkeypatch):
    """Same frozen second + same pid → mkdir(exist_ok=False) collides and the
    bounded retry appends a counter; every allocation is a DISTINCT dir."""
    monkeypatch.setattr(srr, "RESCUE_ROOT", tmp_path / "rescue")
    monkeypatch.setattr(srr, "_rescue_timestamp", lambda: "20260101T000000Z")
    dirs = [srr.allocate_rescue_dir() for _ in range(3)]
    assert len(set(dirs)) == 3
    assert all(d.is_dir() for d in dirs)
    assert all(d.name.startswith("20260101T000000Z") for d in dirs)


def test_same_second_sequential_runs_distinct_rescue_dirs(
    origin_and_clone, tmp_path, monkeypatch, capsys
):
    """Two sequential runs at a FROZEN timestamp rescuing the same rel-path:
    the first rescue's bytes survive in a DISTINCT dir; both manifests exist
    (pre-fix, run 2 reused run 1's dir and shutil.move replaced the copy)."""
    _origin, local, other = origin_and_clone
    monkeypatch.setattr(srr, "_rescue_timestamp", lambda: "20260101T000000Z")
    _write(other, "eval_results/issue_9/r.json", "ORIGIN-1\n")
    _commit(other, "eval_results/issue_9/r.json")
    _git(other, "push", "-q", "origin", "main")
    p1 = _write(local, "eval_results/issue_9/r.json", "FIRST-RESCUE\n")
    _backdate(p1)
    rc1, rep1, _err = _run(local, capsys=capsys)
    assert rc1 == 0
    dir1 = Path(rep1["rescue_dir"])

    # Run 2 (same frozen second): a fresh clone lagging at the init commit
    # gets the SAME rel-path as an untracked differing collision.
    local2 = tmp_path / "local2"
    subprocess.run(
        ["git", "clone", "-q", str(_origin), str(local2)], check=True, capture_output=True
    )
    _configure(local2)
    init_sha = _git(local2, "rev-list", "--max-parents=0", "HEAD").stdout.strip()
    _git(local2, "reset", "--hard", "-q", init_sha)  # scratch test clone, not the shared root
    p2 = _write(local2, "eval_results/issue_9/r.json", "SECOND-RESCUE\n")
    _backdate(p2)
    rc2, rep2, _err = _run(local2, capsys=capsys)
    assert rc2 == 0
    dir2 = Path(rep2["rescue_dir"])

    assert dir1 != dir2
    assert dir1.name.startswith("20260101T000000Z")
    assert dir2.name.startswith("20260101T000000Z")
    assert (dir1 / "eval_results/issue_9/r.json").read_text() == "FIRST-RESCUE\n"
    assert (dir2 / "eval_results/issue_9/r.json").read_text() == "SECOND-RESCUE\n"
    assert (dir1 / "sweep-manifest.json").exists()
    assert (dir2 / "sweep-manifest.json").exists()


# ─── Round 2: journal-before-action durability (concern 2) ───────────────────


def test_mid_sweep_crash_durable_journal_restores(origin_and_clone, monkeypatch, capsys):
    """A crash immediately after the FIRST sweep action leaves a durable
    on-disk journal (the in-memory ledger died before it was populated), and
    the exit-6 restore driven from that journal alone puts the file back."""
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/a.json", "ORIGIN-A\n")
    _write(other, "eval_results/issue_9/b.json", "SAME\n")
    _commit(other, "eval_results/issue_9/a.json", "eval_results/issue_9/b.json")
    _git(other, "push", "-q", "origin", "main")
    a = _write(local, "eval_results/issue_9/a.json", "LOCAL-A\n")  # differing → rescued first
    _backdate(a)
    b = _write(local, "eval_results/issue_9/b.json", "SAME\n")  # identical, never reached
    _backdate(b)

    real_append = srr._journal_append

    def crashing_append(rescue_dir, action, *, applied):
        real_append(rescue_dir, action, applied=applied)
        if applied:
            raise RuntimeError("seam: simulated SIGKILL immediately after the first sweep action")

    monkeypatch.setattr(srr, "_journal_append", crashing_append)
    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 6
    rescue_dir = Path(rep["rescue_dir"])
    journal = rescue_dir / "sweep-journal.jsonl"
    assert journal.exists()  # durable record written BEFORE the action
    actions = srr.load_sweep_journal(rescue_dir)
    assert [(x.path, x.action) for x in actions] == [("eval_results/issue_9/a.json", "rescued")]
    assert actions[0].rescue_path and actions[0].origin_blob_sha  # enough info to restore
    # Journal-driven restore ran on the exit-6 path: the swept file is back.
    assert a.read_text() == "LOCAL-A\n"
    assert any("moved-back eval_results/issue_9/a.json" in s for s in rep["restored"])
    assert b.read_text() == "SAME\n"  # second collision never swept (loop stopped)
    assert "seam: simulated SIGKILL" in err


def test_restore_rematerializes_from_recorded_blob_sha_not_moving_ref(origin_and_clone):
    """Minor (a): rematerialization uses the ledger's RECORDED blob sha, not
    ``origin/main:<path>`` — the ref may have moved between sweep and restore."""
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/d.json", "V1\n")
    _commit(other, "eval_results/issue_9/d.json")
    _git(other, "push", "-q", "origin", "main")
    _git(local, "fetch", "-q", "origin")
    sha_v1 = _git(local, "rev-parse", "origin/main:eval_results/issue_9/d.json").stdout.strip()
    _write(other, "eval_results/issue_9/d.json", "V2-MOVED\n")
    _commit(other, "eval_results/issue_9/d.json")
    _git(other, "push", "-q", "origin", "main")
    _git(local, "fetch", "-q", "origin")  # origin/main:<path> now resolves to V2

    row = srr.SweepAction("eval_results/issue_9/d.json", "identical", "removed", None, sha_v1)
    report = srr._new_report(local, dry_run=False)
    srr.restore_swept(local, [row], report)
    assert (local / "eval_results/issue_9/d.json").read_text() == "V1\n"
    assert any("recorded blob" in s for s in report["restored"])


def test_restore_failure_routes_to_exit_6_with_report(origin_and_clone, monkeypatch, capsys):
    """Minor (b): an exception inside restore_swept must not lose the report or
    exit outside the documented code set — it routes to exit 6, the report is
    emitted, and the rescue copies stay on disk."""
    _origin, local, other = origin_and_clone
    _write(local, "conflict.txt", "base\n")
    _commit(local, "conflict.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "conflict.txt", "origin-side\n")
    _write(other, "eval_results/issue_9/diff.json", "OTHER\n")
    _commit(other, "conflict.txt", "eval_results/issue_9/diff.json")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "conflict.txt", "local-side\n")
    _commit(local, "conflict.txt")
    _write(local, "eval_results/issue_9/diff.json", "LOCAL\n")

    def boom(repo, ledger, report):
        raise RuntimeError("boom-restore")

    monkeypatch.setattr(srr, "restore_swept", boom)
    rc, rep, err = _run(local, capsys=capsys)  # JSON report still emitted + parses
    assert rc == 6
    assert rep["exit_code"] == 6
    assert "restore_swept FAILED" in err
    assert "boom-restore" in err
    assert "content conflict" in err  # the original abort message is retained
    rescue_copy = Path(rep["rescue_dir"]) / "eval_results/issue_9/diff.json"
    assert rescue_copy.read_text() == "LOCAL\n"  # swept copy retained, not lost


# ─── Round 3: torn trailing journal line + not-applied restore report ─────────


def test_load_journal_tolerates_torn_trailing_line(tmp_path):
    """Concern ``journal-loader-torn-trailing-line`` (c): one valid row + a
    truncated trailing row → the valid action is returned without raising,
    and the tear is recorded (journal path + line number) in the report."""
    rescue_dir = tmp_path / "rescue-run"
    rescue_dir.mkdir()
    action = srr.SweepAction(
        "eval_results/issue_9/t.json",
        "differing",
        "rescued",
        str(rescue_dir / "eval_results/issue_9/t.json"),
        "0" * 40,
    )
    srr._journal_append(rescue_dir, action, applied=True)
    journal = rescue_dir / "sweep-journal.jsonl"
    with journal.open("a") as f:  # torn tail: crash mid-append, no newline
        f.write('{"path": "eval_results/issue_9/u.json", "kind": "differi')
    report = srr._new_report(tmp_path, dry_run=False)
    actions = srr.load_sweep_journal(rescue_dir, report)
    assert [(x.path, x.action) for x in actions] == [("eval_results/issue_9/t.json", "rescued")]
    assert any("torn trailing" in m and "line 2" in m for m in report["messages"])


def test_load_journal_reports_non_trailing_corruption_keeps_valid_rows(tmp_path):
    """A malformed NON-trailing row is named loudly as corruption (never
    silently ignored) and the valid rows around it are still returned."""
    rescue_dir = tmp_path / "rescue-run"
    rescue_dir.mkdir()
    a1 = srr.SweepAction("eval_results/issue_9/v.json", "identical", "removed", None, "1" * 40)
    srr._journal_append(rescue_dir, a1, applied=True)
    with (rescue_dir / "sweep-journal.jsonl").open("a") as f:
        f.write("NOT-JSON-GARBAGE\n")
    a2 = srr.SweepAction(
        "eval_results/issue_9/w.json", "differing", "rescued", str(rescue_dir / "w"), None
    )
    srr._journal_append(rescue_dir, a2, applied=False)
    report = srr._new_report(tmp_path, dry_run=False)
    actions = srr.load_sweep_journal(rescue_dir, report)
    assert [(x.path, x.action) for x in actions] == [
        ("eval_results/issue_9/v.json", "removed"),
        ("eval_results/issue_9/w.json", "rescued"),
    ]
    assert any("CORRUPT" in m and "line 2" in m for m in report["messages"])


def test_journal_append_loops_on_short_write(tmp_path, monkeypatch):
    """Concern (b): ``os.write`` may short-write; the append loops until every
    byte lands, so our own writer can never leave a torn-but-fsync'd row."""
    rescue_dir = tmp_path / "rescue-run"
    rescue_dir.mkdir()
    real_write = os.write

    def one_byte_write(fd, data):
        return real_write(fd, bytes(data)[:1])

    monkeypatch.setattr(srr.os, "write", one_byte_write)
    action = srr.SweepAction("eval_results/issue_9/s.json", "identical", "removed", None, "2" * 40)
    srr._journal_append(rescue_dir, action, applied=False)
    monkeypatch.undo()  # restore os.write before anything else runs
    rows = (rescue_dir / "sweep-journal.jsonl").read_text().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])  # the full row landed despite 1-byte writes
    assert parsed["path"] == "eval_results/issue_9/s.json"
    assert parsed["applied"] is False


def test_torn_trailing_journal_line_restore_end_to_end(origin_and_clone, monkeypatch, capsys):
    """End-to-end: a crash that tears the journal's trailing line no longer
    aborts the exit-6 restore — the complete journal rows still restore (the
    rescued file moves back) and the tear is named in the report."""
    _origin, local, other = origin_and_clone
    _write(other, "eval_results/issue_9/a.json", "ORIGIN-A\n")
    _commit(other, "eval_results/issue_9/a.json")
    _git(other, "push", "-q", "origin", "main")
    a = _write(local, "eval_results/issue_9/a.json", "LOCAL-A\n")  # differing → rescued
    _backdate(a)

    real_append = srr._journal_append

    def tearing_append(rescue_dir, action, *, applied):
        real_append(rescue_dir, action, applied=applied)
        if applied:  # crash mid-append of the NEXT row: torn partial JSON tail
            with (rescue_dir / "sweep-journal.jsonl").open("a") as f:
                f.write('{"path": "eval_results/issue_9/next.json", "kind": "differi')
            raise RuntimeError("seam: simulated SIGKILL mid-append of the next row")

    monkeypatch.setattr(srr, "_journal_append", tearing_append)
    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 6
    # Pre-fix the torn tail raised inside load_sweep_journal, skipping the
    # restore ("restore_swept FAILED"); post-fix the valid rows restore.
    assert "restore_swept FAILED" not in err
    assert a.read_text() == "LOCAL-A\n"
    assert any("moved-back eval_results/issue_9/a.json" in s for s in rep["restored"])
    assert any("torn trailing" in m for m in rep["messages"])


def test_restore_reports_intact_for_not_applied_rescue_intent(tmp_path):
    """Concern ``restore-kept-in-rescue-misreport``: a journaled-but-NOT-applied
    rescue intent (the move never ran; no rescue copy exists) must not claim a
    nonexistent rescue copy — the report states the intent was never applied
    and that the file is still present at its original path."""
    repo = tmp_path / "repo"
    p = repo / "eval_results/issue_9/n.json"
    p.parent.mkdir(parents=True)
    p.write_text("STILL-HERE\n")
    phantom = tmp_path / "rescue-never-created" / "eval_results/issue_9/n.json"
    row = srr.SweepAction("eval_results/issue_9/n.json", "differing", "rescued", str(phantom), None)
    report = srr._new_report(repo, dry_run=False)
    srr.restore_swept(repo, [row], report)
    assert p.read_text() == "STILL-HERE\n"  # untouched
    (msg,) = report["restored"]
    assert "KEPT-IN-RESCUE" not in msg
    assert "never applied" in msg
    assert "original path" in msg


def test_restore_kept_in_rescue_only_when_rescue_copy_exists(tmp_path):
    """The true KEPT-IN-RESCUE case (move applied, original path re-occupied)
    keeps its message — and the rescue copy it names actually exists."""
    repo = tmp_path / "repo"
    occupant = repo / "eval_results/issue_9/o.json"
    occupant.parent.mkdir(parents=True)
    occupant.write_text("POST-PULL-OCCUPANT\n")
    rescue_copy = tmp_path / "rescue-real" / "eval_results/issue_9/o.json"
    rescue_copy.parent.mkdir(parents=True)
    rescue_copy.write_text("SWEPT-LOCAL\n")
    row = srr.SweepAction(
        "eval_results/issue_9/o.json", "differing", "rescued", str(rescue_copy), None
    )
    report = srr._new_report(repo, dry_run=False)
    srr.restore_swept(repo, [row], report)
    (msg,) = report["restored"]
    assert msg.startswith("KEPT-IN-RESCUE")
    assert str(rescue_copy) in msg
    assert rescue_copy.read_text() == "SWEPT-LOCAL\n"  # copy retained, untouched
