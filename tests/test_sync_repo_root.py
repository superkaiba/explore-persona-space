"""Tests for scripts/sync_repo_root.py — the single-flight repo-root sync helper.

Fixture ``origin_and_clone`` extends the ``fake_repo`` pattern
(tests/test_task_workflow.py) with a bare origin + two clones (``local`` /
``other``) so real git divergence, untracked collisions, conflicts, stranded
autostashes, and push rejections can be reproduced end-to-end. Lock paths are
monkeypatched into ``tmp_path`` for per-test isolation (multi-process probes
receive the lock path via the child's argv, not via monkeypatch — module
monkeypatches don't survive ``spawn``).

Covers the 14 cases in plan §5 (task #904).
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
