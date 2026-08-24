"""Functional tests for the #2328 three-source marker-presence read.

Covers ``task_workflow.list_events_head_union`` + ``task_workflow.marker_status``
(HEAD blob + working tree + deferral ledger, plus the commit/stash-window
liveness probe that gates ``absent``) and the ``scripts/task.py marker-status``
CLI (rc lattice: 0 = present-class, 4 = absent EXCLUSIVE, 5 = unknown).

Read-only contract (MF-5): every ``marker_status`` / ``list_events_head_union``
/ CLI invocation in this file runs through the ``_call_ro`` comparator — HEAD
SHA, commit count, full ``git status --porcelain=v1``, and the byte content of
every ``events.jsonl`` + ``REGISTRY.json`` + the deferral ledger are
snapshotted before/after and must be identical.
``test_comparator_self_test_detects_mutation`` proves the comparator FAILs on
deliberate mutations, so a silently-mutating read cannot pass by comparator
weakness.

Determinism: the in-flight liveness probe reads ``PRE_COMMIT_HOME`` at call
time; an autouse fixture points it at a per-test tmp dir. Pid determinism for
the MF-A sub-cases: the test process itself (via ``monkeypatch.chdir``) is the
live in-repo pid; a spawned ``sleep`` child with ``cwd=`` outside the repo is
the live OUT-of-repo pid; an already-exited child (re-spawned on the unlikely
pid recycle) is the dead pid.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import task as task_cli

# ─── Fixtures (fake_repo pattern from tests/test_task_workflow.py) ──────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tmp_path as a git repo; task_workflow's resolvers rebound onto it."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    monkeypatch.setattr(tw, "DEFERRED_COMMITS_LOG", lock_dir / "deferred-commits.jsonl")
    monkeypatch.setattr(tw, "STRANDED_COMMITS_LOG", lock_dir / "stranded-commits.jsonl")
    return tmp_path, tw


@pytest.fixture(autouse=True)
def _pre_commit_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Deterministic patch dir for the stash-window liveness probe."""
    d = tmp_path / "pre-commit-home"
    d.mkdir(exist_ok=True)
    monkeypatch.setenv("PRE_COMMIT_HOME", str(d))
    return d


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _commit_crash(paths, message):
    """Injected _git_commit failure with the real lock-collision stderr."""
    raise subprocess.CalledProcessError(
        128,
        ["git", "commit"],
        output="",
        stderr="fatal: Unable to create '/x/.git/index.lock': File exists.\n",
    )


def _git(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=env
    )


def _mk_task(tw) -> int:
    return tw.create_task(tw.NewTaskRequest(kind="experiment", title="marker-status probe"))


def _events_relpath(repo: Path, tw, task_id: int) -> str:
    return (tw.find_task_path(task_id) / "events.jsonl").relative_to(repo).as_posix()


def _snapshot(repo: Path, tw) -> dict:
    """MF-5 state snapshot: git identity + every marker-bearing byte surface."""
    snap = {
        "head": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "revcount": _git(repo, "rev-list", "--count", "HEAD").stdout.strip(),
        "status": _git(repo, "status", "--porcelain=v1").stdout,
    }
    reg = repo / "tasks" / "REGISTRY.json"
    snap["registry"] = reg.read_bytes() if reg.is_file() else None
    ledger = tw.DEFERRED_COMMITS_LOG
    snap["ledger"] = ledger.read_bytes() if ledger.is_file() else None
    tasks_root = repo / "tasks"
    snap["events"] = (
        {
            p.relative_to(repo).as_posix(): p.read_bytes()
            for p in sorted(tasks_root.rglob("events.jsonl"))
        }
        if tasks_root.exists()
        else {}
    )
    return snap


def _call_ro(repo: Path, tw, fn, *args, **kwargs):
    """MF-5 comparator: run fn, assert the call left ALL state byte-identical."""
    before = _snapshot(repo, tw)
    try:
        return fn(*args, **kwargs)
    finally:
        after = _snapshot(repo, tw)
        assert after == before, "read-only contract violated: state mutated by the call"


def _cli_marker_status(repo, tw, monkeypatch, capsys, *cli_args) -> tuple[int, str]:
    """Run the real CLI in-process (parser + dispatch + rc mapping), read-only-checked."""
    argv = ["task.py", "marker-status", *[str(a) for a in cli_args]]
    monkeypatch.setattr(sys, "argv", argv)

    def _run() -> int:
        try:
            task_cli.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    rc = _call_ro(repo, tw, _run)
    return rc, capsys.readouterr().out


def _deferred_marker_state(
    repo, tw, monkeypatch, kind="epm:results", note="round 1 results"
) -> int:
    """Committed task + a `kind` append whose commit DEFERRED (real ledger row
    via the real _commit_after_durable_append), then the unstaged append
    reverted with `git checkout --` — the #2015 stash-window / #2325
    false-destruction state: row in NEITHER tree, live deferral row present."""
    tid = _mk_task(tw)
    with monkeypatch.context() as m:
        m.setattr(tw, "_git_commit", _commit_crash)
        tw.post_event(tid, kind, note=note, by="test")
    _git(repo, "checkout", "--", _events_relpath(repo, tw, tid))
    return tid


def _orphaned_absent_state(repo, tw, monkeypatch) -> int:
    """Trees miss + ledger EMPTY (the pre-publication gap): the aggregate reads
    absent unless the in-flight window probe fires."""
    tid = _deferred_marker_state(repo, tw, monkeypatch)
    tw.DEFERRED_COMMITS_LOG.unlink()
    return tid


# ─── Case 1-2: present-committed / present-uncommitted ──────────────────────


def test_present_committed_and_cli_rc0(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    tw.post_event(tid, "epm:results", note="hello world", by="test")
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "present-committed"
    assert res["legs"]["head"]["n_matches"] == 1
    rc, out = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 0
    assert out.splitlines()[0].startswith(
        f"verdict: present-committed — task #{tid} kind=epm:results"
    )


def test_present_uncommitted_when_commit_deferred(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    with monkeypatch.context() as m:
        m.setattr(tw, "_git_commit", _commit_crash)
        payload = tw.post_event(tid, "epm:results", note="deferred row", by="test")
    assert payload["kind"] == "epm:results"
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "present-uncommitted"
    assert res["legs"]["worktree"]["n_matches"] == 1
    assert res["legs"]["head"]["n_matches"] == 0
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 0


# ─── Case 3: the #2325 reconstruction → pending-deferred, NEVER absent ───────


def test_pending_deferred_after_stash_window_revert(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _deferred_marker_state(repo, tw, monkeypatch)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "pending-deferred"
    assert res["legs"]["head"]["n_matches"] == 0
    assert res["legs"]["worktree"]["n_matches"] == 0
    assert res["legs"]["ledger"]["n_matches"] == 1
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 0  # present-class: a scripted gate can NEVER read this as absence


# ─── Case 4: absent — rc 4 reserved for the complete clean read ─────────────


def test_absent_rc4_on_complete_clean_read(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "absent"
    assert res["inflight"]["probed"] is True
    assert res["inflight"]["signals"] == []
    for leg in res["legs"].values():
        assert leg["status"] != "error"
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 4


# ─── Case 5: filters — trees filtered, ledger deliberately NOT (N7) ─────────


def test_tree_filters_version_and_note(fake_repo, monkeypatch):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    tw.post_event(tid, "epm:results", note="alpha result", by="test", version=1)
    tw.post_event(tid, "epm:results", note="beta result", by="test", version=2)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results", version=2)
    assert res["verdict"] == "present-committed"
    assert res["legs"]["head"]["n_matches"] == 1
    assert res["matches"]["head"][0]["version"] == 2
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results", note_contains="alpha")
    assert res["legs"]["head"]["n_matches"] == 1


def test_ledger_match_is_version_blind(fake_repo, monkeypatch):
    repo, tw = fake_repo
    tid = _deferred_marker_state(repo, tw, monkeypatch)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results", version=3)
    assert res["verdict"] == "pending-deferred"
    assert res["matches"]["ledger"][0]["version_blind"] is True


def test_ledger_match_ignores_note_filter_beyond_truncation(fake_repo, monkeypatch):
    repo, tw = fake_repo
    long_note = ("A" * 70) + " NEEDLE"  # NEEDLE sits past the 60-char message truncation
    tid = _deferred_marker_state(repo, tw, monkeypatch, note=long_note)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results", note_contains="NEEDLE")
    assert res["verdict"] == "pending-deferred"
    assert res["matches"]["ledger"][0]["filters_not_applied_to_ledger"] is True


# ─── Case 6: list_events_head_union ──────────────────────────────────────────


def test_list_events_head_union_order_and_dedupe(fake_repo, monkeypatch):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    tw.post_event(tid, "epm:progress", note="committed row", by="test")
    events_path = tw.find_task_path(tid) / "events.jsonl"
    manual = {"ts": "2099-01-01T00:00:00Z", "kind": "epm:manual", "by": "test", "version": 1}
    with open(events_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(manual) + "\n")
    res = _call_ro(repo, tw, tw.list_events_head_union, tid)
    head_kinds = [r["kind"] for r in res["head_rows"]]
    union_kinds = [r["kind"] for r in res["union_rows"]]
    assert "epm:manual" not in head_kinds
    assert union_kinds[: len(head_kinds)] == head_kinds  # HEAD order leads
    assert union_kinds[-1] == "epm:manual"  # worktree-only rows appended
    assert len(res["union_rows"]) == len(res["worktree_rows"])  # shared rows deduped
    assert res["head_status"] == "ok-found"
    assert res["worktree_status"] == "ok-found"


# ─── Case 7: registry fallback recovers a marker after an uncommitted move ──


def test_head_registry_fallback_recovers_marker_after_uncommitted_move(fake_repo, monkeypatch):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    tw.post_event(tid, "epm:results", note="pre-move row", by="test")
    old_dir = tw.find_task_path(tid)
    new_dir = repo / "tasks" / "running" / str(tid)
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_dir), str(new_dir))
    reg_path = repo / "tasks" / "REGISTRY.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    reg["tasks"][str(tid)]["path"] = f"tasks/running/{tid}"
    reg["tasks"][str(tid)]["status"] = "running"
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    assert tw.find_task_path(tid) == new_dir
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "present-committed"
    assert res["legs"]["head"]["registry_fallback_used"] is True
    assert res["legs"]["head"]["n_matches"] == 1


# ─── Case 8: read-only comparator self-test (MF-5) ───────────────────────────


def test_comparator_self_test_detects_mutation(fake_repo):
    repo, tw = fake_repo
    _mk_task(tw)

    def mutate_worktree():
        (repo / "newfile.txt").write_text("x", encoding="utf-8")

    with pytest.raises(AssertionError, match="read-only contract"):
        _call_ro(repo, tw, mutate_worktree)
    (repo / "newfile.txt").unlink()

    def mutate_ledger():
        tw.DEFERRED_COMMITS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(tw.DEFERRED_COMMITS_LOG, "a", encoding="utf-8") as fh:
            fh.write('{"probe": 1}\n')

    with pytest.raises(AssertionError, match="read-only contract"):
        _call_ro(repo, tw, mutate_ledger)


# ─── Case 9: non-post_event deferrals (MF-1 op→kind refinement) ─────────────


def test_set_goal_deferral_matches_goal_updated_kind(fake_repo, monkeypatch):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    with monkeypatch.context() as m:
        m.setattr(tw, "_git_commit", _commit_crash)
        tw.set_goal(tid, "measure the thing precisely", by="planner")
    _git(repo, "checkout", "--", _events_relpath(repo, tw, tid))
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:goal-updated")
    assert res["verdict"] == "pending-deferred"


def test_create_deferral_ancestor_arm(fake_repo, monkeypatch):
    """`create` records the task FOLDER (not events.jsonl) in its ledger paths —
    the folder-ancestor eligibility arm; kind refinement via the op table."""
    repo, tw = fake_repo
    with monkeypatch.context() as m:
        m.setattr(tw, "_git_commit", _commit_crash)
        tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="never committed"))
    (tw.find_task_path(tid) / "events.jsonl").unlink()  # simulate the destroyed append
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:created")
    assert res["verdict"] == "pending-deferred"
    assert res["legs"]["worktree"]["status"] == "ok-missing"
    # A DIFFERENT kind must NOT ride the create row (kind refinement): clean absent.
    res2 = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res2["verdict"] == "absent"


# ─── Case 10: in-flight window probe (MF-2/MF-A) ─────────────────────────────


def test_inflight_index_lock_blocks_absent(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _orphaned_absent_state(repo, tw, monkeypatch)
    (repo / ".git" / "index.lock").touch()
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "unknown"
    assert "commit-in-flight" in res["reasons"]
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 5


def test_inflight_live_pid_in_repo_stash_window(fake_repo, monkeypatch, _pre_commit_home):
    repo, tw = fake_repo
    tid = _orphaned_absent_state(repo, tw, monkeypatch)
    monkeypatch.chdir(repo)  # the test process IS the live in-repo pid
    (_pre_commit_home / f"patch{int(time.time())}-{os.getpid()}").touch()
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "unknown"
    assert "stash-window-live" in res["reasons"]


def test_inflight_dead_pid_fresh_patch_is_absent(fake_repo, monkeypatch, _pre_commit_home):
    repo, tw = fake_repo
    tid = _orphaned_absent_state(repo, tw, monkeypatch)
    pid = None
    for _ in range(10):  # re-spawn on the (unlikely) instant pid recycle
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        if not os.path.exists(f"/proc/{proc.pid}"):
            pid = proc.pid
            break
    assert pid is not None, "could not obtain a dead pid"
    (_pre_commit_home / f"patch{int(time.time())}-{pid}").touch()
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "absent"


def test_inflight_live_pid_out_of_repo_is_absent(
    fake_repo, monkeypatch, _pre_commit_home, tmp_path_factory
):
    repo, tw = fake_repo
    tid = _orphaned_absent_state(repo, tw, monkeypatch)
    outside = tmp_path_factory.mktemp("outside-repo")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(outside))
    try:
        (_pre_commit_home / f"patch{int(time.time())}-{proc.pid}").touch()
        res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
        assert res["verdict"] == "absent"
    finally:
        proc.terminate()
        proc.wait()


def test_inflight_stale_patch_is_absent(fake_repo, monkeypatch, _pre_commit_home):
    repo, tw = fake_repo
    tid = _orphaned_absent_state(repo, tw, monkeypatch)
    monkeypatch.chdir(repo)  # live in-repo pid — but the patch file is STALE
    p = _pre_commit_home / f"patch{int(time.time()) - 7200}-{os.getpid()}"
    p.touch()
    stale = time.time() - 7200
    os.utime(p, (stale, stale))
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "absent"


def test_inflight_unparseable_fresh_patch_is_unknown(
    fake_repo, monkeypatch, _pre_commit_home, capsys
):
    repo, tw = fake_repo
    tid = _orphaned_absent_state(repo, tw, monkeypatch)
    (_pre_commit_home / "patchgarbage").touch()
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "unknown"
    assert "in-flight-probe-error" in res["reasons"]
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 5


# ─── Case 11-12: source-read failures → unknown, NEVER absent (MF-3) ─────────


def test_ledger_read_failure_is_unknown_never_absent(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    bad = repo / "ledger-as-dir"
    bad.mkdir()
    monkeypatch.setattr(tw, "DEFERRED_COMMITS_LOG", bad)  # read raises IsADirectoryError
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "unknown"
    assert any(r.startswith("leg-error:ledger") for r in res["reasons"])
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 5


def test_git_failure_direct_leg_is_unknown(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    real = tw._marker_status_git

    def fail_events_lstree(args):
        if "ls-tree" in args and any(str(a).endswith("events.jsonl") for a in args):
            return 128, "", "injected ls-tree failure"
        return real(args)

    monkeypatch.setattr(tw, "_marker_status_git", fail_events_lstree)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "unknown"
    assert any(r.startswith("leg-error:head") for r in res["reasons"])
    rc, _ = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results")
    assert rc == 5


def test_git_failure_registry_leg_is_unknown_never_absent(fake_repo, monkeypatch):
    repo, tw = fake_repo
    with monkeypatch.context() as m:
        m.setattr(tw, "_git_commit", _commit_crash)
        tid = tw.create_task(tw.NewTaskRequest(kind="experiment", title="uncommitted"))
    real = tw._marker_status_git

    def fail_registry(args):
        if "ls-tree" in args and any("REGISTRY.json" in str(a) for a in args):
            return 128, "", "injected registry failure"
        return real(args)

    monkeypatch.setattr(tw, "_marker_status_git", fail_registry)
    # Trees miss the queried kind; the ledger row is op=create (kind-refined
    # away). WITHOUT the injected registry failure this state reads absent —
    # WITH it, the incomplete HEAD read must force unknown (rc 4 unreachable).
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "unknown"
    assert any(r.startswith("leg-error:head") for r in res["reasons"])


# ─── Case 13: staleness discriminator (N8) ───────────────────────────────────


def test_stale_ledger_row_is_unknown_with_forensic_guidance(fake_repo, monkeypatch):
    repo, tw = fake_repo
    tid = _deferred_marker_state(repo, tw, monkeypatch)
    res = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res["verdict"] == "pending-deferred"  # converse: a fresh row is live
    # A LATER commit touches events.jsonl WITHOUT carrying the row (the
    # already-swept / lost shape); committer date forced STRICTLY newer
    # (git %cI and the ledger ts are second-resolution).
    rel = _events_relpath(repo, tw, tid)
    row = {"ts": tw._utcnow_iso(), "kind": "epm:progress", "by": "test", "version": 1}
    with open(repo / rel, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    future = (datetime.now(UTC) + timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    env = {**os.environ, "GIT_COMMITTER_DATE": future, "GIT_AUTHOR_DATE": future}
    _git(repo, "add", rel, env=env)
    _git(repo, "commit", "-q", "-m", "later commit", "--", rel, env=env)
    res2 = _call_ro(repo, tw, tw.marker_status, tid, "epm:results")
    assert res2["verdict"] == "unknown"
    assert "stale-ledger-row" in res2["reasons"]
    assert "git log -p --since=" in res2["guidance"]


# ─── Case 14: query echo (N2) ────────────────────────────────────────────────


def test_cli_verdict_line_echoes_query(fake_repo, monkeypatch, capsys):
    repo, tw = fake_repo
    tid = _mk_task(tw)
    tw.post_event(tid, "epm:results", note="echo probe", by="test", version=2)
    rc, out = _cli_marker_status(
        repo,
        tw,
        monkeypatch,
        capsys,
        tid,
        "epm:results",
        "--version",
        "2",
        "--note-contains",
        "echo",
    )
    assert rc == 0
    assert out.splitlines()[0].startswith(
        f"verdict: present-committed — task #{tid} kind=epm:results "
        "version=2 note-contains=echo read-at="
    )
    rc, out = _cli_marker_status(repo, tw, monkeypatch, capsys, tid, "epm:results", "--json")
    data = json.loads(out)
    assert data["task_id"] == tid
    assert data["kind"] == "epm:results"
    assert data["verdict"] == "present-committed"
    assert data["version"] is None
    assert data["note_contains"] is None
    assert "read_at" in data
