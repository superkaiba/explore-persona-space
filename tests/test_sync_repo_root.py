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
from datetime import datetime
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
    monkeypatch.delenv("EPM_ROOT_SYNC_HUSK_PROBE", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_HUSK_MIN_AGE_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_RETRY_SLEEP_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_PROBE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_PROBE_BUDGET_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_ABORT_LOCK_WAIT_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_ABORT_LOCK_RETRIES", raising=False)
    # #1870: the KEPT-stash Telegram push must NEVER fire for real from a test
    # (the default script path exists on the dev VM). Per-test recorder
    # monkeypatches still win — they run after fixture setup.
    monkeypatch.setattr(srr, "_telegram_push_kept", lambda msg: False)
    monkeypatch.delenv("EPM_DISABLE_KEPT_STASH_PUSH", raising=False)
    monkeypatch.delenv("EPM_TELEGRAM_PUSH_SCRIPT", raising=False)
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
    assert "merge -s ours" in err  # the already-landed/discard variant
    assert "NEVER converge by hand" in err  # the hand-convergence warning
    # Production-path stranded-commit threading (#1525 D2): the raise site
    # mines origin/main..HEAD, echoes it into the message, AND records it in
    # the report — without these, every planned test passes with the
    # threading silently dropped.
    assert "stranded local-only commits" in err
    assert rep["stranded_local_commits"], rep
    assert rep["stranded_local_commits"][0].split()[0] in err  # ≥1 sha line


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


# ─── 4b. Merge-form defusal recipe (#1525) ───────────────────────────────────


def test_conflict_message_names_merge_defusal_and_never_hand_converge():
    """Unit pin on the exit-2 message content (#1525 acceptance criterion 1):
    every merge-form recipe element present; deny-listed/destructive tokens
    absent from the message's COMMAND lines (the warning PROSE legitimately
    contains the words ``git reset`` — presence-asserted, never absence-scanned)."""
    msg = srr._conflict_message(["eval_results/INDEX.md"], ["abc1234 add row"])
    # Presence: the merge-form scratch recipe, step by step.
    assert "git worktree add --detach" in msg
    assert "rev-parse main" in msg  # the local-tip capture
    assert 'merge "$LOCAL_TIP"' in msg
    assert "merge -s ours" in msg  # the already-landed/discard variant
    assert "ALL the stranded commits' content ALREADY landed" in msg  # -s ours scoping
    assert "push origin HEAD:main" in msg
    assert "worktree remove --force" in msg  # scratch cleanup
    assert "sync_repo_root.py" in msg  # the re-run step completing convergence
    assert "NEVER converge by hand" in msg
    assert "no `git reset` of any flavor" in msg  # the warning prose, verbatim
    # The stranded-commit list: header + the injected sha line.
    assert "stranded local-only commits (origin/main..HEAD):" in msg
    assert "abc1234 add row" in msg
    # Absence, COMMAND-LINE-SCOPED (#1525 Must-Fix 1): extract the lines whose
    # lstripped text starts with a command prefix; NO command line may carry a
    # deny-listed / reset-family token. Prose lines are never scanned — that
    # scoping is what makes criteria 1(a)/(c) jointly satisfiable.
    command_lines = [
        ln for ln in msg.splitlines() if ln.lstrip().startswith(("git ", "LOCAL_TIP=", "uv run"))
    ]
    assert command_lines, "recipe must contain command lines"
    for ln in command_lines:
        for banned in ("reset", "clean -f", "checkout .", "restore .", "stash drop"):
            assert banned not in ln, (banned, ln)
    # The stranded list is capped: 12 entries show 10 + a remainder line.
    many = [f"sha{i:04d} row {i}" for i in range(12)]
    msg2 = srr._conflict_message(["a.txt"], many)
    assert "sha0009" in msg2
    assert "sha0010" not in msg2
    assert "and 2 more" in msg2


def _build_conflict_divergence(local: Path, other: Path) -> Path:
    """Shared divergence for the defusal e2e pair: overlapping same-region
    edits to conflict.txt (the negative control DEPENDS on the overlap — a
    disjoint-region edit would rebase cleanly), plus a dirty tracked file and
    an untracked file that must survive untouched."""
    _write(local, "conflict.txt", "base\n")
    _write(local, "notes.txt", "notes\n")
    _commit(local, "conflict.txt", "notes.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "conflict.txt", "origin-side\n")
    _commit(other, "conflict.txt")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "conflict.txt", "local-side\n")
    _commit(local, "conflict.txt")
    _write(local, "notes.txt", "dirty-notes\n")  # uncommitted → autostash
    return _write(local, "scratch/untracked.txt", "keep-me\n")


def test_conflict_defusal_merge_recipe_converges_next_sync(origin_and_clone, capsys):
    """Durability pin (#1525 acceptance criterion 2): executing the printed
    MERGE-form recipe's steps mechanically makes the next helper run
    fast-forward to zero divergence under the helper's exact pull flags
    (merge ancestry ⇒ nothing replays), with dirty + untracked root files
    byte-unchanged. Pins the merge-ancestry⇒fast-forward semantics against
    git-version / pull-flag drift."""
    _origin, local, other = origin_and_clone
    untracked = _build_conflict_divergence(local, other)

    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 2
    assert "MERGE-form defusal" in err

    # Execute the recipe steps mechanically (scratch path inside tmp_path —
    # the recipe's fixed /tmp/sync-defuse is an operator convenience).
    scratch = local.parent / "sync-defuse"
    _git(local, "fetch", "origin")
    local_tip = _git(local, "rev-parse", "main").stdout.strip()
    _git(local, "worktree", "add", "--detach", str(scratch), "origin/main")
    merge = _git(scratch, "merge", local_tip, check=False)
    assert merge.returncode != 0  # conflicts, as the recipe says
    _write(scratch, "conflict.txt", "origin-side\nlocal-side\n")  # union resolution
    _git(scratch, "add", "--", "conflict.txt")
    _git(scratch, "commit", "-q", "-m", "merge: resolve root-sync conflict")
    _git(scratch, "push", "-q", "origin", "HEAD:main")
    _git(local, "worktree", "remove", "--force", str(scratch))

    rc2, rep2, _err2 = _run(local, capsys=capsys)
    assert rc2 == 0
    assert rep2["state"] == "synced"
    counts = _git(local, "rev-list", "--left-right", "--count", "origin/main...HEAD")
    assert counts.stdout.split() == ["0", "0"]  # zero divergence — defused for good
    assert (local / "conflict.txt").read_text() == "origin-side\nlocal-side\n"
    assert (local / "notes.txt").read_text() == "dirty-notes\n"  # dirty file survives
    assert untracked.read_text() == "keep-me\n"  # untracked file survives


def test_conflict_rebase_style_recovery_leaves_stranded_commit_re_conflicting(
    origin_and_clone, capsys
):
    """Negative control (#1525 acceptance criterion 3) — documents WHY the
    recipe is merge-form: a scratch REBASE-style recovery (the old recipe's
    typical realization) lands CONTENT but not ANCESTRY, so the stranded
    local commit replays and re-conflicts on the very next sync (rc 2 again).
    Depends on the fixture's same-region conflict.txt edits overlapping."""
    _origin, local, other = origin_and_clone
    _build_conflict_divergence(local, other)

    rc, _rep, _err = _run(local, capsys=capsys)
    assert rc == 2

    scratch = local.parent / "rebase-defuse"
    _git(local, "fetch", "origin")
    local_tip = _git(local, "rev-parse", "main").stdout.strip()
    _git(local, "worktree", "add", "--detach", str(scratch), local_tip)
    reb = _git(scratch, "rebase", "origin/main", check=False)
    assert reb.returncode != 0  # same-region replay conflicts
    _write(scratch, "conflict.txt", "origin-side\nlocal-side\n")  # union resolution
    _git(scratch, "add", "--", "conflict.txt")
    subprocess.run(
        ["git", "-C", str(scratch), "rebase", "--continue"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_EDITOR": "true"},
    )
    _git(scratch, "push", "-q", "origin", "HEAD:main")
    _git(local, "worktree", "remove", "--force", str(scratch))

    rc2, rep2, _err2 = _run(local, capsys=capsys)
    assert rc2 == 2  # the stranded local commit re-conflicts — content-only landing
    assert "conflict.txt" in rep2["conflicted_paths"]


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


# ─── 5b. KEPT-stash durable surfacing (#1870) ────────────────────────────────
#
# A successful sync over a stranded conflicting autostash runs BOTH recover
# passes (preflight + post-pull) with per-call ``processed`` sets, so ONE run
# yields TWO KEPT outcomes on the same sha — assertions key row counts on the
# KEPT report lines, never a hardcoded per-run constant.


def _capture_pushes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the KEPT-stash push fn with a recorder (never a real Telegram call)."""
    pushes: list[str] = []
    monkeypatch.setattr(srr, "_telegram_push_kept", lambda msg: pushes.append(msg) or True)
    return pushes


def _sidecar_rows(local: Path) -> list[dict]:
    return [json.loads(ln) for ln in srr._kept_sidecar_path(local).read_text().splitlines()]


def test_kept_outcome_appends_sidecar_row_and_report_advisory(
    origin_and_clone, capsys, monkeypatch
):
    """(a) Exactly ONE well-formed sidecar row per KEPT outcome (schema per plan
    item 2); every report line keeps the verbatim ``KEPT `` head and gains the
    ``sidecar=`` advisory; the first (new-sha) outcome fires exactly one push."""
    _origin, local, other = origin_and_clone
    pushes = _capture_pushes(monkeypatch)
    _strand_autostash(local, other)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    kept = [s for s in rep["stash"] if s.startswith("KEPT ")]
    assert kept, rep["stash"]
    sidecar = srr._kept_sidecar_path(local)
    rows = _sidecar_rows(local)
    assert len(rows) == len(kept)  # one row per KEPT outcome
    for row in rows:
        assert set(row) == {
            "ts",
            "repo",
            "ref",
            "sha",
            "sha12",
            "reason",
            "detail",
            "rescue_patch",
            "stash_list_len",
            "new_this_run",
        }
        datetime.fromisoformat(row["ts"])  # UTC ISO timestamp parses
        assert row["repo"] == str(local)
        assert row["reason"] == "apply-check-dirty"
        # Harness-stranded entry: preflight classifies it BACKLOG (#2182).
        assert row["new_this_run"] is False
        assert row["detail"] == ""
        assert len(row["sha"]) == 40 and row["sha12"] == row["sha"][:12]
        assert row["ref"].startswith("stash@{")
        assert row["rescue_patch"].endswith(f"stash-{row['sha12']}.patch")
        assert row["stash_list_len"] == 1  # entry KEPT -> still in `git stash list`
    assert all(f"; sidecar={sidecar}" in s for s in kept)
    assert len(pushes) == 1 and "#1736" in pushes[0]


def test_kept_sidecar_write_failure_fail_soft(origin_and_clone, capsys, monkeypatch):
    """(b) A sidecar write failure is FAIL-SOFT: the sync completes (exit 0,
    state machine + KEPT decision unchanged), the report line carries
    ``sidecar-write FAILED`` — the error is neither raised nor silently
    swallowed — and the push is SUPPRESSED (plan must-ask: an unsuppressed
    push would re-fire on every sync run under a persistent write failure)."""
    _origin, local, other = origin_and_clone
    pushes = _capture_pushes(monkeypatch)
    _strand_autostash(local, other)
    sidecar = srr._kept_sidecar_path(local)
    sidecar.mkdir(parents=True)  # a directory at the sidecar path -> OSError on append

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    kept = [s for s in rep["stash"] if s.startswith("KEPT ")]
    assert kept, rep["stash"]
    assert all("sidecar-write FAILED (" in s for s in kept)
    assert pushes == []
    assert any("KEPT-stash sidecar append failed" in m for m in rep["messages"])
    # Entry still KEPT — the recovery semantics are untouched by the failure.
    assert any("autostash" in ln for ln in _git(local, "stash", "list").stdout.splitlines())


def test_kept_push_dedup_second_run_same_sha_no_second_push(origin_and_clone, capsys, monkeypatch):
    """(c) Push dedup keys on the full stash-commit sha read from the sidecar
    BEFORE the append: a dry-run writes nothing (plan item 6), the first real
    sync pushes ONCE, and a second sync over the SAME kept sha appends more
    rows (one per KEPT outcome) but fires NO second push."""
    _origin, local, other = origin_and_clone
    pushes = _capture_pushes(monkeypatch)
    _strand_autostash(local, other)

    rc0, _rep0, _ = _run(local, "--dry-run", capsys=capsys)
    assert rc0 == 0
    assert not srr._kept_sidecar_path(local).exists()  # dry-run writes nothing
    assert pushes == []

    rc1, rep1, _ = _run(local, capsys=capsys)
    assert rc1 == 0
    kept1 = [s for s in rep1["stash"] if s.startswith("KEPT ")]
    assert len(pushes) == 1  # deduped even across the two same-run recover passes
    rc2, rep2, _ = _run(local, capsys=capsys)
    assert rc2 == 0
    kept2 = [s for s in rep2["stash"] if s.startswith("KEPT ")]
    assert kept2, rep2["stash"]
    rows = _sidecar_rows(local)
    assert len(rows) == len(kept1) + len(kept2)  # every KEPT outcome recorded
    assert len({r["sha"] for r in rows}) == 1
    assert len(pushes) == 1  # same sha -> no second push


def test_kept_push_kill_switch_suppresses_push(origin_and_clone, capsys, monkeypatch):
    """(c) The ``EPM_DISABLE_KEPT_STASH_PUSH=1`` kill switch suppresses the
    push entirely; sidecar recording and the report advisory are unaffected."""
    _origin, local, other = origin_and_clone
    pushes = _capture_pushes(monkeypatch)
    monkeypatch.setenv("EPM_DISABLE_KEPT_STASH_PUSH", "1")
    _strand_autostash(local, other)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    kept = [s for s in rep["stash"] if s.startswith("KEPT ")]
    assert kept and all("; sidecar=" in s for s in kept)
    assert _sidecar_rows(local)  # recording unaffected
    assert pushes == []


def test_popped_clean_entry_writes_no_sidecar_row(origin_and_clone, capsys, monkeypatch):
    """(d) The popped (clean-apply) path writes NO sidecar row and fires no push."""
    _origin, local, _other = origin_and_clone
    pushes = _capture_pushes(monkeypatch)
    _write(local, "g.txt", "g-base\n")
    _commit(local, "g.txt")
    _git(local, "push", "-q", "origin", "main")
    _write(local, "g.txt", "g-dirty\n")
    sha = _git(local, "stash", "create").stdout.strip()
    _git(local, "stash", "store", "-m", "autostash", sha)
    _git(local, "checkout", "HEAD", "--", "g.txt")

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert any("popped" in s for s in rep["stash"])
    assert not srr._kept_sidecar_path(local).exists()
    assert pushes == []


def test_kept_sidecar_known_shas_fail_soft(tmp_path):
    """(e) The dedup scan is fail-soft on ALL read errors: OSError on open
    (path is a directory / missing file) returns ``set()``; malformed JSON,
    non-dict rows, and sha-less / non-str-sha rows are skipped."""
    as_dir = tmp_path / "sidecar-as-dir"
    as_dir.mkdir()
    assert srr._kept_sidecar_known_shas(as_dir) == set()
    assert srr._kept_sidecar_known_shas(tmp_path / "missing.jsonl") == set()

    f = tmp_path / "events.jsonl"
    sha = "a" * 40
    f.write_text(
        "not-json\n"
        + json.dumps({"sha": sha})
        + "\n[1, 2]\n"
        + json.dumps({"nosha": True})
        + "\n"
        + json.dumps({"sha": 7})
        + "\n"
    )
    assert srr._kept_sidecar_known_shas(f) == {sha}


# ─── 5c. NEW-vs-BACKLOG discriminator + exit 7 (#2182) ───────────────────────
#
# An autostash stranded by the sync's OWN pull (sha absent from the pre-pull
# snapshot) is fatal — EXIT_AUTOSTASH_STRANDED = 7 with a full recovery
# message; pre-existing backlog entries (pre_pull_shas=None at preflight, or
# sha in the snapshot post-pull) stay a loud WARN at exit 0. Misclassifying
# the backlog as NEW would wedge every sync run on the shared VM.


def _setup_own_pull_strand_conditions(local: Path, other: Path) -> None:
    """The ``_strand_autostash`` SETUP only (its :420-427 half): origin
    advances f.txt, local leaves f.txt dirty-UNCOMMITTED with NO local commit
    ahead — the script's OWN pull performs the strand (completes rc=0,
    autostash reapply conflicts). Deliberately NOT ``_build_conflict_divergence``:
    that fixture COMMITS the conflicting edit, so the pull dies mid-rebase
    rc!=0 → EXIT_CONFLICT (2) and the post-pull recovery is never reached
    (plan §5 test 1, round-1 critic finding)."""
    _write(local, "f.txt", "base\n")
    _commit(local, "f.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "f.txt", "origin\n")
    _commit(other, "f.txt")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "f.txt", "local-dirty\n")  # uncommitted; no commit ahead


def _setup_backlog_plus_new(local: Path, other: Path) -> None:
    """A pre-existing backlog entry (f.txt, harness-stranded) PLUS the
    conditions for the script's OWN pull to strand a second entry (g.txt)."""
    _write(local, "g.txt", "g-base\n")
    _commit(local, "g.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _strand_autostash(local, other)  # backlog entry (f.txt) via the HARNESS pull
    _write(other, "g.txt", "g-origin\n")
    _commit(other, "g.txt")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "g.txt", "g-local-dirty\n")  # uncommitted — the script's pull strands it


def test_new_autostash_stranded_by_own_pull_exits_7(origin_and_clone, capsys):
    """Plan §5 test 1: a NEW autostash stranded by the sync's OWN pull is
    fatal (exit 7) with a self-sufficient recovery message. No literal
    swept-file restoration asserted — the pull SUCCEEDED, so restore_swept
    is occupied-path-guarded by design."""
    _origin, local, other = origin_and_clone
    _setup_own_pull_strand_conditions(local, other)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == srr.EXIT_AUTOSTASH_STRANDED == 7
    assert rep["state"] == "error"
    assert rep["exit_code"] == 7
    new = [e for e in rep["autostash_kept"] if e["new_this_run"] is True]
    assert len(new) == 1
    msg = "\n".join(rep["messages"])
    assert "stash@{0}" in msg
    assert new[0]["sha"][:12] in msg
    assert "git stash pop stash@{0}" in msg
    assert "f.txt" in msg  # the file list
    # The synced-but-unpushed framing (plan §3.2) — exit 7 is not a failed sync.
    assert "pull itself SUCCEEDED" in msg
    assert "push leg was SKIPPED" in msg
    # The pull DID succeed and the entry is KEPT, never dropped.
    assert (local / "f.txt").read_text() == "origin\n"
    assert any("autostash" in ln for ln in _git(local, "stash", "list").stdout.splitlines())
    assert any(s.startswith("KEPT (NEW THIS RUN) ") for s in rep["stash"])


def test_preexisting_backlog_entry_only_exit_0_warn(origin_and_clone, capsys):
    """Plan §5 test 2: a backlog-only run exits 0. Origin advances AGAIN after
    the strand so ``behind > 0`` and the pull pipeline actually runs — the
    post-pull ``sha in snapshot ⇒ backlog`` classification is exercised, not
    just the preflight ``pre_pull_shas=None`` path (round-1 critic concern 3)."""
    _origin, local, other = origin_and_clone
    _strand_autostash(local, other)  # backlog entry via the HARNESS pull
    _write(other, "unrelated.txt", "more\n")
    _commit(other, "unrelated.txt")
    _git(other, "push", "-q", "origin", "main")

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert rep["state"] == "synced"
    assert rep["behind"] > 0  # the pull pipeline (snapshot path) ran
    assert rep["autostash_kept"], rep
    assert all(e["new_this_run"] is False for e in rep["autostash_kept"])
    assert any("pre-existing backlog" in s for s in rep["stash"])
    # Entry still KEPT in the stash list — never dropped.
    assert any("autostash" in ln for ln in _git(local, "stash", "list").stdout.splitlines())


def test_both_classes_present_exit_7_names_only_the_new_entry(origin_and_clone, capsys):
    """Plan §5 test 3: with a backlog entry AND a new own-pull strand in the
    same run, the exit is 7 and the message names the NEW entry specifically —
    the backlog is neither the trigger nor the pop target."""
    _origin, local, other = origin_and_clone
    _setup_backlog_plus_new(local, other)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 7
    new = [e for e in rep["autostash_kept"] if e["new_this_run"]]
    backlog = [e for e in rep["autostash_kept"] if not e["new_this_run"]]
    assert len(new) == 1 and backlog
    msg = "\n".join(rep["messages"])
    assert new[0]["sha"][:12] in msg
    assert "g.txt" in msg  # the NEW entry's file
    for e in backlog:  # the backlog entry is never named in the failure message
        assert e["sha"][:12] not in msg
    # The recovery command points at the NEW entry's CURRENT ref.
    ref_by_sha = {sha: ref for ref, sha in srr._autostash_entries(local)}
    assert f"git stash pop {ref_by_sha[new[0]['sha']]}" in msg
    # Both entries still present — nothing dropped.
    stash_lines = [
        ln for ln in _git(local, "stash", "list").stdout.splitlines() if "autostash" in ln
    ]
    assert len(stash_lines) == 2


def test_sidecar_rows_carry_new_this_run_for_both_classes(origin_and_clone, capsys, monkeypatch):
    """Plan §5 test 4: every sidecar row carries a boolean ``new_this_run``,
    consistent with the report's outcome classification for BOTH classes."""
    _origin, local, other = origin_and_clone
    _capture_pushes(monkeypatch)
    _setup_backlog_plus_new(local, other)

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 7
    rows = _sidecar_rows(local)
    assert rows and all(isinstance(r["new_this_run"], bool) for r in rows)
    new_shas = {e["sha"] for e in rep["autostash_kept"] if e["new_this_run"]}
    backlog_shas = {e["sha"] for e in rep["autostash_kept"] if not e["new_this_run"]}
    assert new_shas and backlog_shas
    assert {r["sha"] for r in rows if r["new_this_run"]} == new_shas
    assert {r["sha"] for r in rows if not r["new_this_run"]} == backlog_shas


def test_triage_autostash_mutates_nothing_and_lists_entries(origin_and_clone, capsys):
    """Plan §5 test 5: ``--triage-autostash`` is read-only (worktree digest +
    ``git stash list`` + HEAD + status byte-identical; mirrors
    ``test_dry_run_mutates_nothing``), exits 0, and its stdout names every
    bare entry. The rescue-patch (re)write lands under RESCUE_ROOT, OUTSIDE
    the working tree."""
    _origin, local, other = origin_and_clone
    # Bare entry 2 FIRST via stash create/store (the :452 recipe) — after the
    # strand the tree holds unmerged paths, which would refuse a commit.
    _write(local, "g.txt", "g-base\n")
    _commit(local, "g.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(local, "g.txt", "g-dirty\n")
    sha2 = _git(local, "stash", "create").stdout.strip()
    _git(local, "stash", "store", "-m", "autostash", sha2)
    _git(local, "checkout", "HEAD", "--", "g.txt")
    _strand_autostash(local, other)  # bare entry 1 (f.txt) + UU tree state

    entries = srr._autostash_entries(local)
    assert len(entries) == 2

    def snapshot() -> tuple:
        return (
            _git(local, "status", "--porcelain=v2", "--untracked-files=all").stdout,
            _git(local, "rev-parse", "HEAD").stdout,
            _git(local, "stash", "list").stdout,
            _worktree_digest(local),
        )

    before = snapshot()
    rc = srr.main(["--repo", str(local), "--triage-autostash"])
    captured = capsys.readouterr()
    assert rc == 0
    assert snapshot() == before  # incl. `git stash list` byte-identical
    for ref, sha in entries:
        assert ref in captured.out
        assert sha[:12] in captured.out
        assert (srr.RESCUE_ROOT / f"stash-{sha[:12]}.patch").exists()
    assert "f.txt" in captured.out and "g.txt" in captured.out
    assert "apply --check:" in captured.out


def test_backlog_count_warn_line_at_preflight(origin_and_clone, capsys):
    """Plan §5 test 6: the backlog-count WARN line appears (once) when >=1
    bare entry survives preflight — in real AND dry-run modes — and is absent
    when none exist."""
    _origin, local, other = origin_and_clone
    rc0, rep0, _ = _run(local, capsys=capsys)
    assert rc0 == 0
    assert not any("autostash backlog:" in m for m in rep0["messages"])

    _strand_autostash(local, other)
    rc1, rep1, _ = _run(local, capsys=capsys)
    assert rc1 == 0
    lines = [m for m in rep1["messages"] if m.startswith("autostash backlog:")]
    assert len(lines) == 1
    assert "1 bare entry(ies) present" in lines[0]
    assert "--triage-autostash" in lines[0]

    rc2, rep2, _ = _run(local, "--dry-run", capsys=capsys)
    assert rc2 == 0
    assert any("autostash backlog:" in m for m in rep2["messages"])


def test_clean_sync_no_autostash_exit_0_empty_outcomes(origin_and_clone, capsys):
    """Plan §5 test 7 (regression anchor): a clean sync with the pull pipeline
    running (behind > 0) and no autostash anywhere stays exit 0 with an empty
    outcome list — the new fatal path cannot fire on the common clean sync."""
    _origin, local, other = origin_and_clone
    _write(other, "clean.txt", "hello\n")
    _commit(other, "clean.txt")
    _git(other, "push", "-q", "origin", "main")

    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert rep["state"] == "synced"
    assert rep["autostash_kept"] == []
    assert _git(local, "stash", "list").stdout.strip() == ""


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


# ─── 15. Multiple-branches transient retry (#1044) ───────────────────────────

_MULTI_BRANCH_FAKE_ARGV = [
    "bash",
    "-c",
    "echo 'fatal: Cannot rebase onto multiple branches.' >&2; exit 128",
]


def _make_local_behind(local: Path, other: Path) -> None:
    """Push an origin-side commit so the sync's pull pipeline actually runs."""
    _write(other, "eval_results/issue_9/y.json", "ORIGIN-Y\n")
    _commit(other, "eval_results/issue_9/y.json")
    _git(other, "push", "-q", "origin", "main")


def test_multi_branch_transient_retried_once_then_succeeds(origin_and_clone, monkeypatch, capsys):
    """First pull dies with the multiple-branches race; the ONE internal retry
    (real pull) succeeds and the sync completes end-to-end."""
    _origin, local, other = origin_and_clone
    _make_local_behind(local, other)

    real_pull_argv = srr._pull_argv
    calls: list[int] = []

    def fake_pull_argv(repo):
        calls.append(1)
        return _MULTI_BRANCH_FAKE_ARGV if len(calls) == 1 else real_pull_argv(repo)

    monkeypatch.setattr(srr, "_pull_argv", fake_pull_argv)
    monkeypatch.setenv("EPM_ROOT_SYNC_RETRY_SLEEP_S", "0")
    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 0
    assert rep["state"] == "synced"
    assert len(calls) == 2  # exactly one retry
    assert any("one retry" in m for m in rep["messages"])
    assert (local / "eval_results/issue_9/y.json").read_text() == "ORIGIN-Y\n"


def test_multi_branch_persistent_failure_surfaces_after_one_retry(
    origin_and_clone, monkeypatch, capsys
):
    """Both attempts carry the signature: exactly one retry, then the failure
    surfaces exactly as a single failure does today (exit 6, no conflict state)."""
    _origin, local, other = origin_and_clone
    _make_local_behind(local, other)
    _write(local, "tracked.txt", "base\n")
    _commit(local, "tracked.txt")
    _write(local, "tracked.txt", "dirty\n")  # pre-sync dirty state
    scratch = _write(local, "scratch.txt", "untracked\n")

    calls: list[int] = []

    def fake_pull_argv(repo):
        calls.append(1)
        return _MULTI_BRANCH_FAKE_ARGV

    monkeypatch.setattr(srr, "_pull_argv", fake_pull_argv)
    monkeypatch.setenv("EPM_ROOT_SYNC_RETRY_SLEEP_S", "0")  # no real 2s sleep in the suite
    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 6
    assert rep["exit_code"] == 6
    assert len(calls) == 2  # exactly one retry, never a loop
    assert "Cannot rebase onto multiple branches" in err
    assert not (local / ".git" / "rebase-merge").exists()
    assert (local / "tracked.txt").read_text() == "dirty\n"  # tracked state untouched
    assert scratch.read_text() == "untracked\n"  # untracked state untouched


def test_multi_branch_signature_with_rebase_state_not_retried(tmp_path, monkeypatch):
    """Belt-and-braces guard: the signature WITH rebase state present is an
    unexpected shape — returned as-is, no retry."""
    calls: list[int] = []

    def fake_pull(repo, timeout_s):
        calls.append(1)
        return srr.GitResult(128, "", "fatal: Cannot rebase onto multiple branches.\n", False)

    monkeypatch.setattr(srr, "pull_rebase", fake_pull)
    monkeypatch.setattr(srr, "_rebase_in_progress", lambda repo: True)
    report = srr._new_report(tmp_path, dry_run=False)
    result = srr._pull_with_transient_retry(tmp_path, report, 5.0)
    assert len(calls) == 1
    assert result.rc == 128
    assert any("NOT retrying" in m for m in report["messages"])


def test_multi_branch_timed_out_result_not_retried(tmp_path, monkeypatch):
    """A timed-out result is never signature-retried (timeout handling wins)."""
    calls: list[int] = []

    def fake_pull(repo, timeout_s):
        calls.append(1)
        return srr.GitResult(-9, "", "fatal: Cannot rebase onto multiple branches.\n", True)

    monkeypatch.setattr(srr, "pull_rebase", fake_pull)
    report = srr._new_report(tmp_path, dry_run=False)
    result = srr._pull_with_transient_retry(tmp_path, report, 5.0)
    assert len(calls) == 1
    assert result.timed_out is True
    assert report["messages"] == []


def test_multi_branch_needle_absent_not_retried(tmp_path, monkeypatch):
    """Negative control: an ordinary pull failure never buys a retry."""
    calls: list[int] = []

    def fake_pull(repo, timeout_s):
        calls.append(1)
        return srr.GitResult(1, "", "error: some unrelated pull failure\n", False)

    monkeypatch.setattr(srr, "pull_rebase", fake_pull)
    report = srr._new_report(tmp_path, dry_run=False)
    result = srr._pull_with_transient_retry(tmp_path, report, 5.0)
    assert len(calls) == 1
    assert result.rc == 1
    assert report["messages"] == []  # no retry message


# ─── 16. Young-husk liveness downgrade (#1044) ────────────────────────────────


def _probe_must_not_be_called(gd, proc_root=None):
    raise AssertionError("_probe_git_liveness must not be called on this path")


def test_young_husk_no_holder_past_floor_downgraded(origin_and_clone, monkeypatch, capsys):
    """A young (1800s) husk past the 600s floor with a completed no-holder scan
    is downgraded to the EXISTING stale handling (abort → continue → the run
    re-hits the genuine conflict, mirroring the stale-husk test)."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 1800
    os.utime(husk, (t, t))

    monkeypatch.setattr(srr, "_probe_git_liveness", lambda gd: srr.LivenessProbe("none", "test"))
    rc, rep, _err = _run(local, capsys=capsys)
    assert any("YOUNG-HUSK DOWNGRADE" in m for m in rep["messages"])
    assert any("STALE-HUSK ABORT" in m for m in rep["messages"])
    assert rc == 2
    assert not (local / ".git" / "rebase-merge").exists()
    assert "c.txt" in rep["conflicted_paths"]


def test_young_husk_live_holder_kept_exit_5(origin_and_clone, monkeypatch, capsys):
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 1800
    os.utime(husk, (t, t))

    monkeypatch.setattr(
        srr, "_probe_git_liveness", lambda gd: srr.LivenessProbe("holder", "live git pid 1 test")
    )
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert husk.exists()  # untouched
    assert "young rebase-merge husk" in err
    assert "live git pid 1 test" in err  # probe evidence surfaced


def test_young_husk_uncertain_probe_kept_exit_5(origin_and_clone, monkeypatch, capsys):
    """An uncertain scan keeps today's refusal AND surfaces the probe's
    reasoning (not just the verdict) in the exit-5 message."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 1800
    os.utime(husk, (t, t))

    monkeypatch.setattr(
        srr,
        "_probe_git_liveness",
        lambda gd: srr.LivenessProbe("uncertain", "pid 42: cwd unreadable (PermissionError)"),
    )
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert husk.exists()  # untouched
    assert "young rebase-merge husk" in err
    assert "pid 42: cwd unreadable" in err


def test_young_husk_below_floor_probe_not_called(origin_and_clone, monkeypatch, capsys):
    """A fresh husk (age ≈ 0 < 600s floor) keeps the existing behavior with the
    probe never invoked — sub-floor behavior stays probe-free + deterministic."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)

    monkeypatch.setattr(srr, "_probe_git_liveness", _probe_must_not_be_called)
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert husk.exists()
    assert "young rebase-merge husk" in err


def test_husk_probe_kill_switch(origin_and_clone, monkeypatch, capsys):
    """EPM_ROOT_SYNC_HUSK_PROBE=0 degrades the probe to a no-op: exactly
    today's refusal, probe never invoked."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 1800
    os.utime(husk, (t, t))

    monkeypatch.setenv("EPM_ROOT_SYNC_HUSK_PROBE", "0")
    monkeypatch.setattr(srr, "_probe_git_liveness", _probe_must_not_be_called)
    rc, _rep, err = _run(local, capsys=capsys)
    assert rc == 5
    assert husk.exists()
    assert "young rebase-merge husk" in err


def test_downgraded_headnameless_dry_run_mutates_nothing(origin_and_clone, monkeypatch, capsys):
    """A downgraded young head-name-less husk flows into the EXISTING
    mutation-free DRY-RUN stale branch."""
    _origin, local, _other = origin_and_clone
    husk = _make_headnameless_husk(local, stale=False)
    t = time.time() - 1800
    os.utime(husk, (t, t))

    monkeypatch.setattr(srr, "_probe_git_liveness", lambda gd: srr.LivenessProbe("none", "test"))
    rc, rep, _err = _run(local, "--dry-run", capsys=capsys)
    assert rc == 0
    assert any("YOUNG-HUSK DOWNGRADE" in m for m in rep["messages"])
    assert any("DRY-RUN: stale head-name-less" in m for m in rep["messages"])
    assert husk.is_dir()  # intact — dry-run mutates nothing


def test_probe_detects_live_git_process(origin_and_clone):
    """Real-/proc positive: a live git process chdir'd into the repo yields
    verdict "holder" (deterministic — a holder short-circuits regardless of
    unrelated system git processes)."""
    _origin, local, _other = origin_and_clone
    proc = subprocess.Popen(
        ["git", "-C", str(local), "hash-object", "--stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        gd = local / ".git"
        deadline = time.monotonic() + 5.0
        probe = None
        while time.monotonic() < deadline:
            probe = srr._probe_git_liveness(gd)
            if probe.verdict == "holder":
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"no holder verdict within deadline; last probe: {probe}")
        assert str(proc.pid) in probe.evidence
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


def _fake_proc(tmp_path: Path) -> Path:
    """Synthetic proc root; includes a non-pid entry like the real /proc."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "stat").write_text("")
    return proc_root


def _fake_pid(proc_root: Path, pid: int, comm: str, cwd: Path | None = None) -> Path:
    pdir = proc_root / str(pid)
    pdir.mkdir()
    (pdir / "comm").write_text(comm + "\n")
    if cwd is not None:
        os.symlink(str(cwd), pdir / "cwd")
    return pdir


def _init_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    return path


def _delenv_probe_tunables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake-proc unit tests skip origin_and_clone; guard against a dev shell
    exporting the probe tunables (same rationale as the fixture delenvs)."""
    monkeypatch.delenv("EPM_ROOT_SYNC_PROBE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("EPM_ROOT_SYNC_PROBE_BUDGET_S", raising=False)


def test_probe_fake_proc_non_git_comm_is_none(tmp_path, monkeypatch):
    """(a) Only non-git processes ⇒ a COMPLETED scan ⇒ "none" — even when the
    non-git process's cwd is inside the repo."""
    _delenv_probe_tunables(monkeypatch)
    r = _init_repo(tmp_path / "r")
    proc_root = _fake_proc(tmp_path)
    _fake_pid(proc_root, 100, "bash", cwd=r)

    probe = srr._probe_git_liveness(r / ".git", proc_root=proc_root)
    assert probe.verdict == "none"


def test_probe_fake_proc_holder_vs_other_repo(tmp_path, monkeypatch):
    """(b) A git process cwd'd in repo r is a "holder" for r's git dir and
    "none" for a DIFFERENT repo (worktree / other-repo exclusion)."""
    _delenv_probe_tunables(monkeypatch)
    r = _init_repo(tmp_path / "r")
    r2 = _init_repo(tmp_path / "r2")
    proc_root = _fake_proc(tmp_path)
    _fake_pid(proc_root, 100, "git", cwd=r)

    holder = srr._probe_git_liveness(r / ".git", proc_root=proc_root)
    assert holder.verdict == "holder"
    assert "pid 100" in holder.evidence
    other = srr._probe_git_liveness(r2 / ".git", proc_root=proc_root)
    assert other.verdict == "none"


def test_probe_fake_proc_cwd_regular_file_uncertain(tmp_path, monkeypatch):
    """(c) An unreadable cwd (readlink OSError — same branch as another user's
    EACCES) ⇒ "uncertain"."""
    _delenv_probe_tunables(monkeypatch)
    r = _init_repo(tmp_path / "r")
    proc_root = _fake_proc(tmp_path)
    pdir = _fake_pid(proc_root, 100, "git")
    (pdir / "cwd").write_text("not-a-symlink\n")  # readlink → OSError (EINVAL)

    probe = srr._probe_git_liveness(r / ".git", proc_root=proc_root)
    assert probe.verdict == "uncertain"
    assert "cwd unreadable" in probe.evidence


def test_probe_fake_proc_unattributable_nonrepo_cwd_uncertain(tmp_path, monkeypatch):
    """(d) A git process whose cwd attributes to NO git dir ⇒ "uncertain" —
    never a false "none" (the dangerous direction)."""
    _delenv_probe_tunables(monkeypatch)
    r = _init_repo(tmp_path / "r")
    nonrepo = tmp_path / "plain"
    nonrepo.mkdir()
    proc_root = _fake_proc(tmp_path)
    _fake_pid(proc_root, 100, "git", cwd=nonrepo)

    probe = srr._probe_git_liveness(r / ".git", proc_root=proc_root)
    assert probe.verdict == "uncertain"
    assert "not attributable" in probe.evidence


def test_probe_fake_proc_attribution_timeout_uncertain(tmp_path, monkeypatch):
    """(e) A hung-mount attribution (timed-out rev-parse) ⇒ "uncertain",
    pinned without a real hang via a monkeypatched _run_bounded."""
    _delenv_probe_tunables(monkeypatch)
    r = _init_repo(tmp_path / "r")
    nonrepo = tmp_path / "plain"
    nonrepo.mkdir()
    proc_root = _fake_proc(tmp_path)
    _fake_pid(proc_root, 100, "git", cwd=nonrepo)

    monkeypatch.setattr(
        srr, "_run_bounded", lambda argv, timeout_s: srr.GitResult(-9, "", "", True)
    )
    probe = srr._probe_git_liveness(r / ".git", proc_root=proc_root)
    assert probe.verdict == "uncertain"
    assert "timed out" in probe.evidence


def test_probe_fake_proc_budget_exhaustion_uncertain(tmp_path, monkeypatch):
    """(f) An exhausted total budget ⇒ "uncertain — scan incomplete"; a
    truncated scan can never yield "none"."""
    monkeypatch.delenv("EPM_ROOT_SYNC_PROBE_TIMEOUT_S", raising=False)
    monkeypatch.setenv("EPM_ROOT_SYNC_PROBE_BUDGET_S", "0")
    r = _init_repo(tmp_path / "r")
    proc_root = _fake_proc(tmp_path)
    _fake_pid(proc_root, 100, "git", cwd=r)

    probe = srr._probe_git_liveness(r / ".git", proc_root=proc_root)
    assert probe.verdict == "uncertain"
    assert "budget" in probe.evidence
    assert "scan incomplete" in probe.evidence


# ─── 20. Abort lock-race bounded retry (#1671) ───────────────────────────────


def _lock_race_stderr(lock_path: str, prefix: str = "error") -> str:
    """The measured git 2.34.1 lockfile.c EEXIST stderr (plan §2 live probes):
    ``Unable to create '<path>': File exists.`` + the hint block + the
    ``fatal: could not move back`` trailer a lock-blocked abort emits."""
    return (
        f"{prefix}: Unable to create '{lock_path}': File exists.\n"
        "\n"
        "Another git process seems to be running in this repository, e.g.\n"
        "an editor opened by 'git commit'. Please make sure all processes\n"
        "are terminated then try again. If it still fails, a git process\n"
        "may have crashed in this repository earlier:\n"
        "remove the file manually to continue.\n"
        "fatal: could not move back to 0123456789abcdef0123456789abcdef01234567\n"
    )


@pytest.mark.parametrize(
    ("stderr", "should_match"),
    [
        pytest.param(
            _lock_race_stderr("/tmp/lockprobe/local/.git/HEAD.lock"),
            True,
            id="head-lock-error-with-hint-block",
        ),
        pytest.param(
            "error: Unable to create '/tmp/lockprobe/local/.git/index.lock': File exists.\n",
            True,
            id="index-lock-error",
        ),
        pytest.param(
            "fatal: Unable to create '/tmp/lockprobe/local/.git/index.lock': File exists.\n",
            True,
            id="index-lock-fatal-merge-abort",
        ),
        pytest.param(
            "error: Unable to create '.git/HEAD.lock': File exists",
            True,
            id="incident-1645-relative-path",
        ),
        pytest.param(
            "error: cannot lock ref 'refs/heads/main': Unable to create "
            "'/x/.git/refs/heads/main.lock': File exists.\n",
            True,
            id="ref-lock-wrapper",
        ),
        pytest.param("simulated abort failure", False, id="simulated-fixture-string"),
        pytest.param(
            "fatal: Unable to create '/x/.git/index.lock': Permission denied",
            False,
            id="permission-denied-lock",
        ),
        pytest.param("fatal: No rebase in progress?", False, id="no-rebase-in-progress"),
        pytest.param("", False, id="empty"),
    ],
)
def test_lock_race_regex_shapes(stderr, should_match):
    """Pure-unit pin of the detection needle against the measured positive
    shapes (plan §2 probes + the #1645 incident form) and the known negatives
    (a non-EEXIST errno prints a different suffix and must never retry)."""
    m = srr._LOCK_RACE_RE.search(stderr)
    assert bool(m) == should_match
    if should_match:
        assert m.group(1).endswith(".lock")


def test_husk_abort_lock_race_retried_then_succeeds(origin_and_clone, monkeypatch, capsys):
    """A transient lock race on the stale-husk abort (the #1645 site) is
    retried; the abort succeeds on attempt 2 and the run proceeds to its
    normal outcome (the same genuine conflict → clean exit 2)."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 7200
    os.utime(husk, (t, t))
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", "0.01")

    real_git = srr.git
    fails = {"left": 1}
    # Lock path points at a NONEXISTENT file so the bounded poll exits at once.
    gone = str(local / ".git" / "GONE.lock")

    def fake_git(repo, *args, **kwargs):
        if args[:2] == ("rebase", "--abort") and fails["left"]:
            fails["left"] -= 1
            return subprocess.CompletedProcess(args, 128, stdout="", stderr=_lock_race_stderr(gone))
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(srr, "git", fake_git)
    rc, rep, _err = _run(local, capsys=capsys)
    assert any("transient lock race" in m for m in rep["messages"])
    assert any("succeeded on retry 1" in m for m in rep["messages"])
    assert any("STALE-HUSK ABORT" in m for m in rep["messages"])
    assert rc == 2  # run continued into the same genuine conflict
    assert not (local / ".git" / "rebase-merge").exists()


def test_husk_abort_lock_race_exhausted_exit6_lock_never_deleted(
    origin_and_clone, monkeypatch, capsys
):
    """Persistent lock → DEADLINE-BOUND exhaustion: today's exit-6 refusal
    fires with its message unchanged, the husk is kept, and the lock file is
    NEVER deleted. The abort is invoked >= 2 and <= 1 + RETRIES times — the
    inner poll consumes the wall bound before the count limit binds, so the
    exact count is deliberately NOT asserted here (the count-bound exact pin
    is test_abort_lock_retry_count_bound_exact_invocations)."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 7200
    os.utime(husk, (t, t))
    head_lock = local / ".git" / "HEAD.lock"
    head_lock.write_text("")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_WAIT_S", "0.3")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", "0.05")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_RETRIES", "2")

    real_git = srr.git
    calls = {"abort": 0}

    def fake_git(repo, *args, **kwargs):
        if args[:2] == ("rebase", "--abort"):
            calls["abort"] += 1
            return subprocess.CompletedProcess(
                args, 128, stdout="", stderr=_lock_race_stderr(str(head_lock))
            )
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(srr, "git", fake_git)
    rc, rep, err = _run(local, capsys=capsys)
    assert rc == 6
    assert "not the known un-abortable" in err  # raise-site message preserved
    assert husk.exists()  # KEPT
    assert head_lock.exists()  # NEVER deleted
    assert 2 <= calls["abort"] <= 3  # deadline-bound: count limit need not be reached
    assert any("still lock-blocked" in m for m in rep["messages"])


def test_abort_non_lock_failure_not_retried(origin_and_clone, monkeypatch, capsys):
    """The conservative predicate: a NON-lock abort failure is surfaced from
    the FIRST attempt — exactly one invocation, behavior identical to today
    (the :985 shape plus an invocation counter)."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 7200
    os.utime(husk, (t, t))

    real_git = srr.git
    calls = {"abort": 0}

    def fake_git(repo, *args, **kwargs):
        if args[:2] == ("rebase", "--abort"):
            calls["abort"] += 1
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="simulated abort failure")
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(srr, "git", fake_git)
    rc, _rep, err = _run(local, capsys=capsys)
    assert calls["abort"] == 1  # never retried
    assert rc == 6
    assert "not the known un-abortable" in err
    assert husk.exists()


def test_conflict_abort_lock_race_retried(origin_and_clone, monkeypatch, capsys):
    """check=True success-after-retry at the conflict-abort site
    (_capture_conflict_and_abort): a genuine content conflict still exits 2
    cleanly when the abort's first attempt loses a transient lock race."""
    _origin, local, other = origin_and_clone
    _write(local, "conflict.txt", "base\n")
    _commit(local, "conflict.txt")
    _git(local, "push", "-q", "origin", "main")
    _git(other, "pull", "-q", "origin", "main")
    _write(other, "conflict.txt", "origin-side\n")
    _commit(other, "conflict.txt")
    _git(other, "push", "-q", "origin", "main")
    _write(local, "conflict.txt", "local-side\n")
    _commit(local, "conflict.txt")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", "0.01")

    real_git = srr.git
    fails = {"left": 1}
    gone = str(local / ".git" / "GONE.lock")

    def fake_git(repo, *args, **kwargs):
        if args[:2] == ("rebase", "--abort") and fails["left"]:
            fails["left"] -= 1
            return subprocess.CompletedProcess(args, 128, stdout="", stderr=_lock_race_stderr(gone))
        return real_git(repo, *args, **kwargs)

    monkeypatch.setattr(srr, "git", fake_git)
    rc, rep, _err = _run(local, capsys=capsys)
    assert rc == 2
    assert "conflict.txt" in rep["conflicted_paths"]
    assert not (local / ".git" / "rebase-merge").exists()
    assert any("succeeded on retry 1" in m for m in rep["messages"])


def test_abort_with_lock_retry_check_true_exhaustion_raises(origin_and_clone, monkeypatch):
    """Direct unit pin of the check=True exhaustion contract the pull-timeout
    sites rely on: CalledProcessError carrying the FINAL attempt's fields (the
    EXIT_UNEXPECTED conversion in main is already pinned by existing tests)."""
    _origin, local, _other = origin_and_clone
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_WAIT_S", "60")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_RETRIES", "1")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", "0.01")
    stderr = _lock_race_stderr(str(local / ".git" / "GONE.lock"))

    def fake_git(repo, *args, **kwargs):
        assert args[:2] == ("rebase", "--abort")
        return subprocess.CompletedProcess(args, 128, stdout="", stderr=stderr)

    monkeypatch.setattr(srr, "git", fake_git)
    report = {"messages": []}
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        srr._abort_with_lock_retry(local, report, "rebase", "--abort")
    assert excinfo.value.returncode == 128
    assert excinfo.value.stderr == stderr  # FINAL attempt's stderr preserved
    assert excinfo.value.cmd == ["git", "-C", str(local), "rebase", "--abort"]


def test_real_lock_race_end_to_end(origin_and_clone, capsys, monkeypatch):
    """NO fake git: a REAL pre-created HEAD.lock blocks the real abort; a
    timer clears it mid-poll and the retry succeeds — validates the detection
    needle against LIVE git output end to end (the git-version canary; a
    wording change in a future git fails this loudly, not silently in prod).
    2.0s timer, generous vs the 10s wall bound, so the first abort attempt
    reliably sees the lock even on a loaded shared VM; the flake direction is
    false-RED only (a too-early clear skips the retry and fails the retry
    assertions), never a vacuous pass."""
    _origin, local, other = origin_and_clone
    husk = _make_conflicted_rebase_husk(local, other)
    t = time.time() - 7200
    os.utime(husk, (t, t))
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_WAIT_S", "10")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", "0.1")
    lock = local / ".git" / "HEAD.lock"
    lock.write_text("")
    timer = threading.Timer(2.0, lambda: lock.unlink(missing_ok=True))
    timer.start()
    try:
        rc, rep, _err = _run(local, capsys=capsys)
    finally:
        timer.join()
    assert any("transient lock race" in m for m in rep["messages"])
    assert any("succeeded on retry" in m for m in rep["messages"])
    assert not (local / ".git" / "rebase-merge").exists()
    assert rc == 2  # run continued into the same genuine conflict


def test_abort_lock_retry_count_bound_exact_invocations(origin_and_clone, monkeypatch):
    """The COUNT-BOUND exhaustion pin (the `retries >= _abort_lock_retries()`
    break — replace it with 999 and ONLY this test goes red): the named lock
    clears instantly between attempts (nonexistent path) while the abort keeps
    failing on a fresh lock needle (the two-lock-chain case), so the RETRIES
    limit is what binds and the abort runs exactly 1 + RETRIES times."""
    _origin, local, _other = origin_and_clone
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_WAIT_S", "60")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_RETRIES", "2")
    monkeypatch.setenv("EPM_ROOT_SYNC_ABORT_LOCK_POLL_S", "0.01")
    stderr = _lock_race_stderr(str(local / ".git" / "GONE.lock"))
    calls = {"abort": 0}

    def fake_git(repo, *args, **kwargs):
        assert args[:2] == ("rebase", "--abort")
        calls["abort"] += 1
        return subprocess.CompletedProcess(args, 128, stdout="", stderr=stderr)

    monkeypatch.setattr(srr, "git", fake_git)
    report = {"messages": []}
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        srr._abort_with_lock_retry(local, report, "rebase", "--abort")
    assert calls["abort"] == 3  # exactly 1 + RETRIES
    assert excinfo.value.stderr == stderr  # FINAL attempt carried

    calls["abort"] = 0  # check=False variant: same fake, same bounds, fresh counter
    res = srr._abort_with_lock_retry(local, report, "rebase", "--abort", check=False)
    assert calls["abort"] == 3
    assert res.returncode == 128
    assert res.stderr == stderr
