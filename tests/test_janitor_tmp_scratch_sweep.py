"""Tests for the #2127 top-level /tmp gate/smoke SCRATCH sweep
(``clean_experiment_downloads.sweep_tmp_scratch`` + the ``vm_disk_guard``
tier-(f) wrapper + the gate-1.7 git-blob evidence branch (c)).

HERMETIC BY CONSTRUCTION (the #911 pattern): every fixture lives under
pytest's ``tmp_path`` and is passed as an EXPLICIT ``tmp_root`` /
``main_repo`` — the real ``/tmp`` and the real repo odb are never read or
written, and NOTHING destructive ever targets a real path (the brief's hard
rule: ``--apply`` against the real /tmp is forbidden; report-only smokes of
real paths live outside pytest). The "main repo" is a per-test ``git init``
fixture, so blob-existence proofs are against a THROWAWAY odb — which is
also why the branch-(c) opt-in (``git_evidence_repo``) defaults to None: an
empty file's blob exists in every real repo, and a hermetic default keeps
fixture trees from being licensed by the real odb.

Loaded via importlib like ``tests/test_janitor_noncanonical_caches.py``
(ced first — vm_disk_guard imports it by module name at load time).
"""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures / helpers ──────────────────────────────────────────────────────

AGED_TS = time.time() - 100 * 3600.0  # 100h ago — well past the 48h window

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )
    if check:
        assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r


COMMITTED_PY = "print('hello from a committed file')\n"
COMMITTED_JSON = '{"k": 1}\n'


@pytest.fixture(autouse=True)
def _clear_kill_switches(monkeypatch):
    """The sweep must run in these tests regardless of the invoking shell's
    environment (both kill-switch layers unset)."""
    monkeypatch.delenv(ced.SCRATCH_SWEEP_KILL_ENV, raising=False)
    monkeypatch.delenv(ced.NONCANONICAL_SWEEP_KILL_ENV, raising=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point both modules' repo_root at a temp dir (sidecar rows resolve under
    it) and stub the #773 consumer gate to empty (branch-(c) tests reap)."""
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    return tmp_path


@pytest.fixture
def main_repo(tmp_path):
    """A throwaway 'main repo' with two committed files — the odb every blob
    proof in this file verifies against."""
    repo = tmp_path / "mainrepo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "tracked.py").write_text(COMMITTED_PY)
    (repo / "sub").mkdir()
    (repo / "sub" / "data.json").write_text(COMMITTED_JSON)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


@pytest.fixture
def tmp_root(tmp_path):
    root = tmp_path / "faketmp"
    root.mkdir()
    return root


def _backdate(root: Path, ts: float = AGED_TS) -> None:
    for p in sorted(root.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        os.utime(p, (ts, ts), follow_symlinks=False)
    os.utime(root, (ts, ts), follow_symlinks=False)


def _reap_eligible_dir(tmp_root: Path, name: str = "issue-9990-gate2") -> Path:
    """A plain (git-less) scratch-shaped dir whose every byte is committed
    content in ``main_repo`` — fully reap-eligible once aged."""
    cand = tmp_root / name
    (cand / "nested").mkdir(parents=True)
    (cand / "copy.py").write_text(COMMITTED_PY)
    (cand / "nested" / "data.json").write_text(COMMITTED_JSON)
    _backdate(cand)
    return cand


def _scratch_worktree(main_repo: Path, tmp_root: Path, name: str) -> Path:
    """A REGISTERED detached worktree of ``main_repo`` at its HEAD, aged."""
    cand = tmp_root / name
    _git(main_repo, "worktree", "add", "--detach", str(cand))
    _backdate(cand)
    return cand


def _scratch_clone(main_repo: Path, tmp_root: Path, name: str) -> Path:
    """A full local clone of ``main_repo`` (its own in-tree odb), aged."""
    cand = tmp_root / name
    r = subprocess.run(
        ["git", "clone", str(main_repo), str(cand)],
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )
    assert r.returncode == 0, r.stderr
    _backdate(cand)
    return cand


def _sweep(tmp_root: Path, main_repo: Path, *, apply: bool, **kw) -> "ced.ScratchSweepResult":
    return ced.sweep_tmp_scratch(tmp_root, apply=apply, main_repo=main_repo, **kw)


def _row(res, name: str) -> dict:
    rows = [r for r in res.rows if r["name"] == name]
    assert rows, f"no sweep row for {name}; rows={[r['name'] for r in res.rows]}"
    return rows[0]


# ─── shape + denylist predicate ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("issue-1895-gate2", True),
        ("step9c-smoke-xyz", True),
        ("eps-main-scratch-2058", True),
        ("scratch-2171-merge2", True),
        ("mkstest-abc123", True),
        ("claude-1001", False),  # denylist — the live Claude task-output tree
        ("pytest-of-thomasjiralerspong", False),  # denylist — live gate basetemp
        ("tmux-1000", False),
        ("systemd-private-abc", False),
        ("snap-private-tmpXXXX", False),
        ("ssh-fjiewof", False),
        ("i2127_hf_dl", False),  # issue-keyed, NOT scratch-shaped (the #911 leg's)
        ("random-dir", False),
    ],
)
def test_is_tmp_scratch_name(name, expected):
    assert ced.is_tmp_scratch_name(name) is expected


def test_denylist_beats_widened_shape_globs(monkeypatch):
    """Two-layer unreachability (brief hard rule): even if a FUTURE shape glob
    matches everything, the denylist still keeps claude-*/pytest-of-* out."""
    monkeypatch.setattr(ced, "_SCRATCH_SHAPE_GLOBS", ("*",))
    assert ced.is_tmp_scratch_name("claude-1001") is False
    assert ced.is_tmp_scratch_name("pytest-of-anyone") is False
    assert ced.is_tmp_scratch_name("anything-else") is True


def test_denylisted_dirs_never_become_candidates(tmp_root, main_repo, repo):
    """§5 deny_claude + deny_pytest: reap-eligible CONTENT under a denylisted
    name is untouchable — no row, no deletion, even under --apply."""
    for name in ("claude-1001", "pytest-of-user"):
        cand = tmp_root / name
        cand.mkdir()
        (cand / "copy.py").write_text(COMMITTED_PY)
        _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    assert res.rows == []
    assert (tmp_root / "claude-1001" / "copy.py").exists()
    assert (tmp_root / "pytest-of-user" / "copy.py").exists()


# ─── the reap-eligible happy path + report mode ──────────────────────────────


def test_reap_eligible_plain_dir_is_reaped(tmp_root, main_repo, repo):
    cand = _reap_eligible_dir(tmp_root)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert "git-blob-reproducible" in row["evidence"]
    assert row["n_verified"] == 2
    assert res.bytes_freed > 0
    assert not cand.exists()


def test_report_only_mode_deletes_nothing(tmp_root, main_repo, repo):
    """§5 report_only: apply=False yields would-reap rows and ZERO mutation."""
    cand = _reap_eligible_dir(tmp_root)
    res = _sweep(tmp_root, main_repo, apply=False)
    row = _row(res, cand.name)
    assert row["disposition"] == "would-reap"
    assert cand.exists() and (cand / "copy.py").read_text() == COMMITTED_PY
    assert res.bytes_freed == 0


def test_recent_tree_kept_even_with_apply(tmp_root, main_repo, repo):
    """§5 recent_kept: a fresh-mtime tree is kept — age is only a KEEP signal."""
    cand = tmp_root / "issue-9991-smoke"
    cand.mkdir()
    (cand / "copy.py").write_text(COMMITTED_PY)  # mtime = now
    res = _sweep(tmp_root, main_repo, apply=True)
    assert _row(res, cand.name)["disposition"] == "tmp-scratch-recent-kept"
    assert cand.exists()


def test_unverified_file_keeps_tree_and_names_it(tmp_root, main_repo, repo):
    """§5 diff_not_tolerated: one uncommitted ``.diff`` blocks the whole reap
    (``.diff`` is deliberately OFF the tolerance allowlist) and is NAMED as
    first_unverified."""
    cand = _reap_eligible_dir(tmp_root, "issue-9992-gate1")
    (cand / "own.diff").write_text("--- a/x\n+++ b/x\n+uncommitted-only-copy\n")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert row["first_unverified"] == "own.diff"
    assert cand.exists() and (cand / "own.diff").exists()


def test_tolerance_only_tree_is_kept(tmp_root, main_repo, repo):
    """§5 tolerance_only_escalates: n_verified must be >= 1 — a tree holding
    ONLY tolerated small text never PASSes on tolerance alone."""
    cand = tmp_root / "issue-9993-gate3"
    cand.mkdir()
    (cand / "run.log").write_text("some uncommitted log line\n")
    (cand / "out.txt").write_text("stdout capture\n")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-tolerance-only-kept"
    assert cand.exists()


def test_tolerated_text_rides_a_verified_reap(tmp_root, main_repo, repo):
    """Small uncommitted .log/.out text beside >=1 verified file is tolerated
    (accepted without proof) and the tree reaps."""
    cand = _reap_eligible_dir(tmp_root, "issue-9994-gate4")
    (cand / "run.log").write_text("uncommitted telemetry\n")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert row["n_tolerated"] == 1
    assert not cand.exists()


def test_tolerance_never_applies_under_durable_dirs(tmp_root, main_repo, repo):
    """A .log under an ``eval_results/`` component is durable-class — it needs
    a real blob proof, so an uncommitted one keeps the tree."""
    cand = _reap_eligible_dir(tmp_root, "issue-9995-gate5")
    (cand / "eval_results").mkdir()
    (cand / "eval_results" / "scores.log").write_text("precious uncommitted numbers\n")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert row["first_unverified"] == "eval_results/scores.log"
    assert cand.exists()


def test_empty_tree_is_reaped(tmp_root, main_repo, repo):
    """§5 empty_tree_reaped: dirs-only trees carry nothing to lose (the one
    n_verified==0 carve-out)."""
    cand = tmp_root / "issue-9996-smoke"
    (cand / "a" / "b").mkdir(parents=True)
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert "empty-tree" in row["evidence"] or "empty tree" in row["evidence"]
    assert not cand.exists()


# ─── atime / hardlink recency semantics ──────────────────────────────────────


def test_verified_but_recently_read_tree_is_atime_pinned(tmp_root, main_repo, repo):
    """§5 atime_pinned: a VERIFIED tree with a fresh nlink==1 reader atime is
    kept + escalated, never reaped."""
    cand = _reap_eligible_dir(tmp_root, "issue-9997-gate6")
    os.utime(cand / "copy.py", (time.time(), AGED_TS))  # atime fresh, mtime aged
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-verified-atime-pinned"
    assert cand.exists()


def test_hardlinked_fresh_atime_is_not_reader_evidence(tmp_root, main_repo, tmp_path, repo):
    """§5 hardlink_atime_ignored: nlink>1 files share atimes with out-of-tree
    links (the uv .venv shape) — a fresh atime there must NOT pin the reap."""
    cand = _reap_eligible_dir(tmp_root, "issue-9998-gate7")
    outside = tmp_path / "outside_hardlink.py"
    os.link(cand / "copy.py", outside)  # nlink -> 2
    os.utime(cand / "copy.py", (time.time(), AGED_TS))  # fresh atime, aged mtime
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert not cand.exists()
    assert outside.exists()  # the out-of-tree link survives (rmtree, not truncate)


# ─── git-class probes: clones ────────────────────────────────────────────────


def test_dirty_clone_is_kept(tmp_root, main_repo, repo):
    """§5 dirty_clone: an untracked file in a clone blocks via the status
    probe (clone-dirty) before any blob math."""
    cand = _scratch_clone(main_repo, tmp_root, "eps-gateclone-scratch-1")
    (cand / "untracked_note.md").write_text("only copy\n")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "clone-dirty" in row["reason"]
    assert cand.exists()


def test_clone_with_own_stash_is_kept(tmp_root, main_repo, repo):
    """A clone's stash lives in ITS OWN odb and dies with the tree — kept."""
    cand = _scratch_clone(main_repo, tmp_root, "eps-stashclone-scratch-1")
    (cand / "tracked.py").write_text(COMMITTED_PY + "# local edit\n")
    _git(cand, "stash", "push", "-m", "wip")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "clone-stash" in row["reason"]
    assert cand.exists()


def test_clone_with_unpushed_commit_is_kept(tmp_root, main_repo, repo):
    """A clone ref tip not reachable from any surviving main-repo ref would
    die with the tree — kept."""
    cand = _scratch_clone(main_repo, tmp_root, "eps-aheadclone-scratch-1")
    (cand / "tracked.py").write_text(COMMITTED_PY + "# committed only here\n")
    _git(cand, "commit", "-am", "local-only work")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "unpushed" in row["reason"] or "head-unreachable" in row["reason"]
    assert cand.exists()


def test_clean_clone_is_reaped(tmp_root, main_repo, repo):
    """A pristine clone (status clean, no stash, every tip reachable from the
    main repo) is reproducible in full — reaped. Also pins that the sweep's
    OWN git probes (which rewrite the clone's in-tree .git/index) do not
    self-abort the reap re-check (the non-exempt-mtime key)."""
    cand = _scratch_clone(main_repo, tmp_root, "eps-cleanclone-scratch-1")
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert not cand.exists()


# ─── git-class probes: registered worktrees ──────────────────────────────────


def test_worktree_detached_unpushed_head_is_kept(tmp_root, main_repo, repo):
    """§5 worktree_detached_unpushed: a worktree whose detached HEAD commit is
    unreachable from every main-repo ref is kept (the commit dies on prune)."""
    cand = _scratch_worktree(main_repo, tmp_root, "eps-wt-scratch-ahead")
    (cand / "tracked.py").write_text(COMMITTED_PY + "# detached work\n")
    _git(cand, "commit", "-am", "detached-only commit")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "head-unreachable" in row["reason"]
    assert cand.exists()


def test_worktree_staged_uncommitted_is_kept(tmp_root, main_repo, repo):
    """§5 worktree_staged_uncommitted: staged-but-uncommitted state is dirty."""
    cand = _scratch_worktree(main_repo, tmp_root, "eps-wt-scratch-staged")
    (cand / "staged_only.py").write_text("never committed\n")
    _git(cand, "add", "staged_only.py")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "worktree-dirty" in row["reason"]
    assert cand.exists() and (cand / "staged_only.py").exists()


def test_clean_worktree_reaped_despite_shared_stash(tmp_root, main_repo, repo):
    """§5 worktree_clean_shared_stash + worktree_reap_prunes_registration: the
    SHARED stash lives in the main odb and survives the worktree; the reap
    goes through ``git worktree remove`` so the registration is gone too."""
    cand = _scratch_worktree(main_repo, tmp_root, "eps-wt-scratch-clean")
    # A stash entry in the MAIN repo (worktrees share it; it must survive).
    (main_repo / "tracked.py").write_text(COMMITTED_PY + "# stashed edit\n")
    _git(main_repo, "stash", "push", "-m", "shared stash entry")
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert "worktree-removed" in row["reason"]
    assert not cand.exists()
    assert str(cand) not in _git(main_repo, "worktree", "list").stdout  # unregistered
    assert "shared stash entry" in _git(main_repo, "stash", "list").stdout  # survives


def test_locked_worktree_is_kept(tmp_root, main_repo, repo):
    """§5 worktree_locked_kept: a lock present at gate time keeps via the
    class probe (its own disposition kind)."""
    cand = _scratch_worktree(main_repo, tmp_root, "eps-wt-scratch-locked")
    _git(main_repo, "worktree", "lock", str(cand))
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-worktree-locked-kept"
    assert cand.exists()


def test_lock_acquired_after_verification_keeps_tree(tmp_root, main_repo, repo, monkeypatch):
    """A lock acquired BETWEEN gate check and reap keeps the tree. Since
    review round 2 the reap-time class RE-probe catches it FIRST (one gate
    earlier than amendment 2's remove-failure path): the tree is kept and
    stays registered."""
    cand = _scratch_worktree(main_repo, tmp_root, "eps-wt-scratch-race")
    real_hit = ced._scratch_live_process_hit

    def lock_then_pass(c):
        _git(main_repo, "worktree", "lock", str(cand))
        return real_hit(c)

    monkeypatch.setattr(ced, "_scratch_live_process_hit", lock_then_pass)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reap-reprobe-kept"
    assert "worktree-locked" in row["reason"]
    assert cand.exists() and (cand / ".git").exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout  # still registered


def test_lock_acquired_after_reprobe_keeps_tree_no_rmtree_fallback(
    tmp_root, main_repo, repo, monkeypatch
):
    """Amendment 2 (pin preserved post-round-2): a lock acquired in the
    residual window AFTER the reap-time re-probe makes ``git worktree
    remove --force`` (single --force) FAIL — the failure KEEPS the tree
    (no rmtree fallback, no global prune), still registered."""
    cand = _scratch_worktree(main_repo, tmp_root, "eps-wt-scratch-race2")
    real_probe = ced._scratch_git_class_probes
    n_calls = {"n": 0}

    def probe_then_lock(c, kind, admin, *, main_repo: Path):
        out = real_probe(c, kind, admin, main_repo=main_repo)
        n_calls["n"] += 1
        if n_calls["n"] == 2 and out is None:  # the reap-time call: lock AFTER it passes
            _git(main_repo, "worktree", "lock", str(cand))
        return out

    monkeypatch.setattr(ced, "_scratch_git_class_probes", probe_then_lock)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert n_calls["n"] == 2  # verify-time + reap-time probes both ran
    assert row["disposition"] == "tmp-scratch-worktree-remove-failed"
    assert "locked" in row["reason"]
    assert cand.exists() and (cand / ".git").exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout  # still registered


def test_foreign_worktree_is_kept(tmp_root, main_repo, tmp_path, repo):
    """A registered worktree of some OTHER repo is never reasoned about
    against main's odb — kept."""
    other = tmp_path / "otherrepo"
    other.mkdir()
    _git(other, "init", "-b", "main")
    (other / "f.txt").write_text("x\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-m", "init")
    cand = tmp_root / "eps-foreign-scratch-1"
    _git(other, "worktree", "add", "--detach", str(cand))
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "foreign-worktree" in row["reason"]
    assert cand.exists()


# ─── defensive walk: FIFOs, symlinks, live processes, re-check ───────────────


def test_fifo_keeps_tree_and_never_blocks(tmp_root, main_repo, repo):
    """§5 fifo_kept (amendment 4): a FIFO anywhere keeps the tree, and the
    sweep completes without blocking — proven by running it in a worker
    thread with an explicit join timeout (no pytest-timeout dependency).
    The lstat classify-first + O_NONBLOCK open contract is what this pins."""
    cand = _reap_eligible_dir(tmp_root, "issue-9999-gate8")
    os.mkfifo(cand / "wedge.fifo")
    _backdate(cand)
    box: dict = {}

    def run():
        box["res"] = _sweep(tmp_root, main_repo, apply=True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=120)
    assert not t.is_alive(), "sweep BLOCKED on a FIFO — the non-regular gates are broken"
    row = _row(box["res"], cand.name)
    assert row["disposition"] == "tmp-scratch-nonregular-kept"
    assert cand.exists()


def test_symlinks_never_followed_and_targets_survive(tmp_root, main_repo, tmp_path, repo):
    """Symlink contract: internal, dangling, and OUT-OF-TREE links are all
    skipped (never followed, never opened, never verified); the reap removes
    the links, not their targets."""
    cand = _reap_eligible_dir(tmp_root, "issue-9989-gate9")
    outside = tmp_path / "outside_target.bin"
    outside.write_bytes(b"unique uncommitted bytes the sweep must never touch")
    os.symlink(outside, cand / "out.lnk")
    os.symlink(cand / "copy.py", cand / "self.lnk")
    os.symlink(cand / "never-existed", cand / "dangling.lnk")
    _backdate(cand)
    res = _sweep(tmp_root, main_repo, apply=True)
    row = _row(res, cand.name)
    # the out-of-tree target's content is UNCOMMITTED — if the walk followed
    # the link the tree would be unverified-kept; skipping it lets the reap
    # proceed on the two committed regular files.
    assert row["disposition"] == "tmp-scratch-reaped"
    assert not cand.exists()
    assert outside.read_bytes() == b"unique uncommitted bytes the sweep must never touch"


def test_live_process_holds_the_reap(tmp_root, main_repo, repo):
    """Amendment 1: a live process with cwd inside the candidate keeps it;
    once the process is dead the same tree reaps."""
    cand = _reap_eligible_dir(tmp_root, "issue-9988-gate10")
    proc = subprocess.Popen(["sleep", "300"], cwd=str(cand))
    try:
        res = _sweep(tmp_root, main_repo, apply=True)
        row = _row(res, cand.name)
        assert row["disposition"] == "tmp-scratch-live-process-kept"
        assert f"pid={proc.pid}" in row["reason"]
        assert cand.exists()
    finally:
        proc.kill()
        proc.wait()
    res2 = _sweep(tmp_root, main_repo, apply=True)
    assert _row(res2, cand.name)["disposition"] == "tmp-scratch-reaped"
    assert not cand.exists()


def test_reap_aborts_when_tree_changes_under_verification(tmp_root, main_repo, repo, monkeypatch):
    """§5 reap_abort_on_recheck: a write landing between verification and reap
    (simulated by an evidence stub that both PASSes and writes) trips the
    fresh re-walk and aborts — the tree survives."""
    cand = _reap_eligible_dir(tmp_root, "issue-9987-gate11")

    def pass_and_write(c, *, main_repo, full_stats, verdict_cache=None):
        late = c / "landed-during-verify.py"
        late.write_text("late write\n")
        # ext4 stamps mtimes from the kernel's COARSE clock, which can lag
        # time.time() by a few ms — pin the "landed during verification"
        # ordering explicitly so the fixture is deterministic.
        late_ts = time.time() + 10.0
        os.utime(late, (late_ts, late_ts))
        return "git-blob-reproducible: stub", {
            "reason": "pass",
            "first_unverified": None,
            "n_verified": 1,
            "n_tolerated": 0,
            "git_class": "none",
        }

    monkeypatch.setattr(ced, "_git_blob_reproducibility_evidence", pass_and_write)
    res = _sweep(tmp_root, main_repo, apply=True, now=time.time())
    row = _row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-reap-aborted-recency"
    assert cand.exists() and (cand / "landed-during-verify.py").exists()


# ─── kill switches + hermeticity ─────────────────────────────────────────────


@pytest.mark.parametrize("env_var", [ced.SCRATCH_SWEEP_KILL_ENV, ced.NONCANONICAL_SWEEP_KILL_ENV])
def test_kill_switches_disable_the_sweep(tmp_root, main_repo, repo, monkeypatch, env_var):
    """§5 kill_switch: either layer (own switch / family switch) disables the
    leg entirely — no rows, no deletion."""
    cand = _reap_eligible_dir(tmp_root, "issue-9986-gate12")
    monkeypatch.setenv(env_var, "1")
    res = _sweep(tmp_root, main_repo, apply=True)
    assert res.rows == []
    assert cand.exists()


def test_no_opt_in_means_no_sweep(main_repo):
    """Hermetic default: tmp_root=None / main_repo=None never touch anything."""
    assert ced.sweep_tmp_scratch(None, apply=True, main_repo=main_repo).rows == []
    assert ced.sweep_tmp_scratch(Path("/nonexistent-xyz"), apply=True, main_repo=None).rows == []


def test_scratch_verdict_cache_roundtrip(tmp_path):
    """Cache: definitive verdicts round-trip; reaped paths prune; corrupt
    files degrade to no-cache; transient reasons are never stored."""
    cache_file = tmp_path / "cache.json"
    cache = ced._ScratchVerdictCache(cache_file)
    cand = tmp_path / "cand"
    stats = {"newest_mtime": 1.0, "total_bytes": 42}
    cache.store(cand, stats, None, {"reason": "unverified-file", "first_unverified": "x"})
    cache.store(
        cand, {"newest_mtime": 2.0, "total_bytes": 42}, None, {"reason": "git-probe-failed"}
    )
    cache.save()
    fresh = ced._ScratchVerdictCache(cache_file)
    hit = fresh.lookup(cand, stats)
    assert hit is not None and hit[0] is None and hit[1]["reason"] == "unverified-file"
    assert fresh.lookup(cand, {"newest_mtime": 2.0, "total_bytes": 42}) is None  # not cacheable
    fresh.prune(cand)
    fresh.save()
    assert ced._ScratchVerdictCache(cache_file).lookup(cand, stats) is None
    cache_file.write_text("{corrupt json")
    assert ced._ScratchVerdictCache(cache_file).lookup(cand, stats) is None  # fail-soft


def test_verdict_cache_skips_rehash_on_unchanged_tree(
    tmp_root, main_repo, tmp_path, repo, monkeypatch
):
    """A second report-only sweep over an unchanged tree serves the verdict
    from the cache (no re-hash) — pinned by counting blob-probe calls."""
    _reap_eligible_dir(tmp_root, "issue-9985-gate13")
    cache_path = tmp_path / "verdicts.json"
    calls: list[int] = []
    real = ced._git_first_missing_blob

    def counting(main_repo_, shas):
        calls.append(len(shas))
        return real(main_repo_, shas)

    monkeypatch.setattr(ced, "_git_first_missing_blob", counting)
    _sweep(tmp_root, main_repo, apply=False, verdict_cache_path=cache_path)
    assert len(calls) == 1
    _sweep(tmp_root, main_repo, apply=False, verdict_cache_path=cache_path)
    assert len(calls) == 1  # second run: cache hit, no new probe


def test_cached_pass_never_licenses_reap_after_ref_deletion(
    tmp_root, main_repo, tmp_path, repo, monkeypatch
):
    """REGRESSION (review round 2, Finding 1): a PASS cached while a
    worktree's HEAD was ref-reachable must NOT license a later reap after
    the only containing ref is deleted with ZERO tree change (`git branch
    -D`, a pruning fetch, an upstream rewrite). The verify-time class
    probes are cache-skipped on the second run (pinned by the blob-probe
    call count going flat), so the reap-time class RE-probe in
    `_reap_scratch_tree` is the ONLY thing standing between the stale
    cached PASS and `git worktree remove --force` dropping the last
    reference to the commit chain. Fails (tree reaped) if that re-probe
    is removed."""
    cand = tmp_root / "eps-9970-scratch-reprobe"
    _git(main_repo, "worktree", "add", "--detach", str(cand))
    # A commit reachable ONLY via a side branch created in the worktree.
    _git(cand, "checkout", "-b", "side")
    (cand / "only_here.py").write_text(COMMITTED_PY)
    _git(cand, "add", "only_here.py")
    _git(cand, "commit", "-m", "side-only commit")
    _git(cand, "checkout", "--detach")  # free the branch for deletion
    _backdate(cand)

    cache_path = tmp_path / "verdicts.json"
    calls: list[int] = []
    real = ced._git_first_missing_blob

    def counting(main_repo_, shas):
        calls.append(len(shas))
        return real(main_repo_, shas)

    monkeypatch.setattr(ced, "_git_first_missing_blob", counting)

    # Sweep 1 (report-only): ref exists -> verified -> would-reap; PASS cached.
    res1 = _sweep(tmp_root, main_repo, apply=False, verdict_cache_path=cache_path)
    assert _row(res1, cand.name)["disposition"] == "would-reap"
    n_probes_run1 = len(calls)
    assert n_probes_run1 >= 1

    # Sweep 1's `git status` re-read file contents (the index is stale after
    # _backdate) and refreshed atimes; re-backdate to model the incident
    # precondition — tree quiet AND unread >=48h before the reap attempt.
    # Same timestamps => the cache key (newest_mtime|total_bytes) is intact.
    _backdate(cand)

    # External git-state flip, zero tree change: the only containing ref dies.
    _git(main_repo, "branch", "-D", "side")

    # Sweep 2 (apply): cache-hit PASS skips the verify-time probes (call
    # count flat) -> only the reap-time re-probe can catch the flip -> KEPT.
    res2 = _sweep(tmp_root, main_repo, apply=True, verdict_cache_path=cache_path)
    assert len(calls) == n_probes_run1  # cache hit: no fresh blob probe
    row = _row(res2, cand.name)
    assert row["disposition"] == "tmp-scratch-reap-reprobe-kept"
    assert "head-unreachable" in row["reason"]
    assert cand.exists()
    assert res2.bytes_freed == 0


# ─── gate-1.7 evidence branch (c) on the issue-keyed /tmp legs ───────────────


def _stub_evidence_set(monkeypatch, names: set[str] | None) -> None:
    val = frozenset(names) if names is not None else None
    monkeypatch.setattr(ced, "_data_repo_toplevel_names", lambda: val)


def test_branch_c_licenses_issue_keyed_tmp_dir(tmp_path, tmp_root, main_repo, repo, monkeypatch):
    """An aged issue-keyed /tmp dir with NO HF evidence but full git-blob
    reproducibility reaps under branch (c) (git_evidence_repo opt-in)."""
    cand = tmp_root / "i9984_gatecache"
    cand.mkdir()
    (cand / "copy.py").write_text(COMMITTED_PY)
    _backdate(cand)
    _stub_evidence_set(monkeypatch, set())
    res = ced.clean_issue_downloads(
        9984,
        apply=True,
        data_root=tmp_path / "data",
        tmp_root=tmp_root,
        git_evidence_repo=main_repo,
    )
    assert not cand.exists()
    assert any("i9984_gatecache" in r for r in res.removed)
    assert "git-blob-reproducible" in res.noncanonical_evidence[ced._rel_name(cand)]


def test_branch_c_hermetic_default_keeps(tmp_path, tmp_root, main_repo, repo, monkeypatch):
    """Without the opt-in the SAME candidate stays unverified-kept — fixture
    trees are never licensed by any real odb by default."""
    cand = tmp_root / "i9984_gatecache"
    cand.mkdir()
    (cand / "copy.py").write_text(COMMITTED_PY)
    _backdate(cand)
    _stub_evidence_set(monkeypatch, set())
    res = ced.clean_issue_downloads(
        9984, apply=True, data_root=tmp_path / "data", tmp_root=tmp_root
    )
    assert cand.exists()
    assert res.noncanonical_dispositions[ced._rel_name(cand)] == "unverified-kept"


def test_branch_c_unverified_names_first_file(tmp_path, tmp_root, main_repo, repo, monkeypatch):
    """Branch (c) failure detail (reason + first unverified file) rides the
    unverified-kept reason string."""
    cand = tmp_root / "i9983_gatecache"
    cand.mkdir()
    (cand / "own.diff").write_text("uncommitted diff\n")
    _backdate(cand)
    _stub_evidence_set(monkeypatch, set())
    res = ced.clean_issue_downloads(
        9983,
        apply=True,
        data_root=tmp_path / "data",
        tmp_root=tmp_root,
        git_evidence_repo=main_repo,
    )
    assert cand.exists()
    (reason,) = [r for n, r in res.skipped if "i9983_gatecache" in n]
    assert "git-blob branch (c)" in reason and "own.diff" in reason


def test_branch_c_refuses_registered_worktree(tmp_path, tmp_root, main_repo, repo, monkeypatch):
    """Branch (c) never licenses a REGISTERED main-repo worktree — gate 1.7's
    reap is a plain rmtree, which would strand the registration; worktree
    reaps belong to the scratch leg's worktree-aware removal."""
    cand = tmp_root / "i9982_wtcache"
    _git(main_repo, "worktree", "add", "--detach", str(cand))
    _backdate(cand)
    _stub_evidence_set(monkeypatch, set())
    res = ced.clean_issue_downloads(
        9982,
        apply=True,
        data_root=tmp_path / "data",
        tmp_root=tmp_root,
        git_evidence_repo=main_repo,
    )
    assert cand.exists() and (cand / ".git").exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout
    (reason,) = [r for n, r in res.skipped if "i9982_wtcache" in n]
    assert "registered-worktree" in reason


def test_exclude_scratch_shapes_routes_dual_matches_to_one_leg(tmp_root):
    """An issue-keyed AND scratch-shaped name (``issue9981-gate2``) is skipped
    by the #911 discovery exactly when the scratch leg owns it."""
    cand = tmp_root / "issue9981-gate2"
    cand.mkdir()
    (cand / "x.json").write_text("{}")
    both = ced.noncanonical_cache_dirs(9981, tmp_root=tmp_root)
    assert cand in both
    excluded = ced.noncanonical_cache_dirs(9981, tmp_root=tmp_root, exclude_scratch_shapes=True)
    assert cand not in excluded


# ─── vm_disk_guard tier (f) wiring ───────────────────────────────────────────


def _stub_reclaim_tiers(monkeypatch):
    """Neutralize the /-rooted + HF tiers so run_guard tests exercise only
    tier (b) + tier (f)."""
    for fn in (
        "clean_uv_cache",
        "clean_stale_logs",
        "clean_vm_workspace_hf_cache",
        "clean_home_hf_stale_revisions",
    ):
        monkeypatch.setattr(vdg, fn, lambda *a, _n=fn, **k: vdg.TierResult(name=_n))


def test_run_guard_wires_scratch_tier_and_excludes_dual_shapes(
    tmp_path, tmp_root, main_repo, repo, monkeypatch
):
    """The boot-pass wiring: tier (f) runs after tier (b); a dual-match dir
    (issue-keyed AND scratch-shaped) lands ONLY on the scratch rows; the
    scratch_candidates rows ride --json's tier dict."""
    _stub_reclaim_tiers(monkeypatch)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    dual = tmp_root / "issue9980-gate2"
    dual.mkdir()
    (dual / "copy.py").write_text(COMMITTED_PY)
    _backdate(dual)
    data_root = tmp_path / "data"
    data_root.mkdir()
    res = vdg.run_guard(
        False,
        data_root=data_root,
        tmp_root=tmp_root,
        ignore_threshold=True,
        scratch_tmp_root=tmp_root,
        scratch_main_repo=main_repo,
        git_evidence_repo=main_repo,
    )
    names = [t.name for t in res.tiers]
    assert "tmp-scratch" in names
    assert names.index("tmp-scratch") == names.index("terminal-download-caches") + 1
    tier_b = next(t for t in res.tiers if t.name == "terminal-download-caches")
    tier_f = next(t for t in res.tiers if t.name == "tmp-scratch")
    assert [r["name"] for r in tier_f.scratch_candidates] == ["issue9980-gate2"]
    assert tier_f.scratch_candidates[0]["disposition"] == "would-reap"
    assert not any("issue9980-gate2" in row["path"] for row in tier_b.noncanonical_candidates)
    assert dual.exists()  # report-only: zero mutation
    payload = vdg._result_json(res)
    tier_dicts = {t["name"]: t for t in payload["tiers"]}
    assert tier_dicts["tmp-scratch"]["scratch_candidates"][0]["name"] == "issue9980-gate2"
    json.dumps(payload)  # rows must stay JSON-serializable


def test_run_guard_scratch_off_restores_old_routing(
    tmp_path, tmp_root, main_repo, repo, monkeypatch
):
    """Kill switch set => no tmp-scratch tier AND the dual-match dir routes
    through tier (b) exactly as before #2127 (bit-identical old behavior)."""
    _stub_reclaim_tiers(monkeypatch)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    monkeypatch.setenv(ced.SCRATCH_SWEEP_KILL_ENV, "1")
    dual = tmp_root / "issue9979-gate2"
    dual.mkdir()
    (dual / "x.json").write_text("{}")
    _backdate(dual)
    data_root = tmp_path / "data"
    data_root.mkdir()
    res = vdg.run_guard(
        False,
        data_root=data_root,
        tmp_root=tmp_root,
        ignore_threshold=True,
        scratch_tmp_root=tmp_root,
        scratch_main_repo=main_repo,
        git_evidence_repo=main_repo,
    )
    assert "tmp-scratch" not in [t.name for t in res.tiers]
    tier_b = next(t for t in res.tiers if t.name == "terminal-download-caches")
    assert any("issue9979-gate2" in row["path"] for row in tier_b.noncanonical_candidates)
    assert dual.exists()


def test_clean_tmp_scratch_tier_skips_without_opt_in():
    res = vdg.clean_tmp_scratch(False, tmp_root=None, main_repo=None)
    assert res.skipped and "hermetic" in res.skip_reason
