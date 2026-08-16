"""Tests for the #2147 ``~/.eps-slurm-src/issue-<N>`` SLURM staging-tree sweep
(``clean_experiment_downloads.sweep_slurm_src`` + the ``vm_disk_guard``
tier-(g) wrapper ``clean_slurm_src`` + the D9 nested-overlay evidence class +
the D6 leg-scoped escalation dedup), plan #2147 §5 T1-T15.

HERMETIC BY CONSTRUCTION (the #2127 tier-(f) pattern): every fixture lives
under pytest's ``tmp_path`` and is passed as an EXPLICIT ``staging_root`` /
``main_repo`` — the real ``~/.eps-slurm-src``, the real /tmp, and the real
repo odb are never read or written, and NOTHING destructive ever targets a
real path. The "main repo" is a per-test ``git init`` fixture, so blob
proofs are against a THROWAWAY odb.

Loaded via importlib like ``tests/test_janitor_tmp_scratch_sweep.py``
(ced first — vm_disk_guard imports it by module name at load time).
"""

import importlib.util
import json
import os
import subprocess
import sys
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


# ─── fixtures / helpers (mirroring tests/test_janitor_tmp_scratch_sweep.py) ──

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
    """The sweep must run regardless of the invoking shell's environment
    (all three kill-switch layers unset)."""
    monkeypatch.delenv(ced.SLURM_SRC_SWEEP_KILL_ENV, raising=False)
    monkeypatch.delenv(ced.SCRATCH_SWEEP_KILL_ENV, raising=False)
    monkeypatch.delenv(ced.NONCANONICAL_SWEEP_KILL_ENV, raising=False)


@pytest.fixture(autouse=True)
def _zero_escalate_floor(monkeypatch):
    """Plan §5: escalation fixtures are tiny — zero the KEEP-row sidecar
    size floor so escalation assertions are size-independent."""
    monkeypatch.setenv("EPS_SCRATCH_ESCALATE_FLOOR_GB", "0")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point both modules' repo_root at a temp dir (sidecar rows resolve under
    it) and stub the #773 consumer gate to empty."""
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
def slurm_root(tmp_path):
    root = tmp_path / "eps-slurm-src"
    root.mkdir()
    return root


def _backdate(root: Path, ts: float = AGED_TS) -> None:
    for p in sorted(root.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        os.utime(p, (ts, ts), follow_symlinks=False)
    os.utime(root, (ts, ts), follow_symlinks=False)


def _staged_copy(slurm_root: Path, name: str = "issue-9990") -> Path:
    """A git-less staged repo copy (the ``materialize_branch_src`` shape)
    whose every byte is committed content in ``main_repo`` — fully
    reap-eligible once aged and terminal."""
    cand = slurm_root / name
    (cand / "sub").mkdir(parents=True)
    (cand / "tracked.py").write_text(COMMITTED_PY)
    (cand / "sub" / "data.json").write_text(COMMITTED_JSON)
    _backdate(cand)
    return cand


def _slurm_worktree(main_repo: Path, slurm_root: Path, name: str) -> Path:
    """A REGISTERED detached worktree of ``main_repo`` at its HEAD, aged."""
    cand = slurm_root / name
    _git(main_repo, "worktree", "add", "--detach", str(cand))
    _backdate(cand)
    return cand


def _sweep(slurm_root: Path, main_repo: Path, *, apply: bool, status: str = "completed", **kw):
    kw.setdefault("status_resolver", lambda n: status)
    kw.setdefault("terminal_statuses", vdg.TERMINAL_CACHE_REAP_STATUSES)
    return ced.sweep_slurm_src(slurm_root, apply=apply, main_repo=main_repo, **kw)


def _row(res, name: str) -> dict:
    rows = [r for r in res.rows if r["name"] == name]
    assert rows, f"no sweep row for {name}; rows={[r['name'] for r in res.rows]}"
    return rows[0]


def _sidecar_rows(repo_root_path: Path) -> list[dict]:
    p = repo_root_path / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not p.is_file():
        return []
    # split("\n"), never splitlines(): JSONL rows may carry U+2028/U+2029.
    return [json.loads(line) for line in p.read_text().split("\n") if line.strip()]


# ─── T1: pre-gates g1-g4b ────────────────────────────────────────────────────


def test_g1_unrecognized_name_kept(slurm_root, main_repo, repo):
    """T1/g1: a non-``issue-<N>``-shaped entry is kept with its own
    disposition and never escalated (not even at floor 0)."""
    cand = slurm_root / "not-an-issue-dir"
    cand.mkdir()
    (cand / "x.py").write_text(COMMITTED_PY)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unrecognized-kept"
    assert cand.exists()
    assert _sidecar_rows(repo) == []


def test_g2_unowned_entry_kept(slurm_root, main_repo, repo, monkeypatch):
    """T1/g2: an entry not owned by the current uid is kept untouched."""
    cand = _staged_copy(slurm_root, "issue-9998")
    monkeypatch.setattr(ced, "_tmp_entry_owned", lambda p: False)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-not-owned-kept"
    assert cand.exists()


def test_g3_symlink_entry_never_followed(slurm_root, main_repo, tmp_path, repo):
    """T1/g3: a symlinked ``issue-<N>`` entry is kept (containment) and its
    TARGET is never touched — the reap must never chase an escape."""
    outside = tmp_path / "outside-tree"
    outside.mkdir()
    (outside / "precious.py").write_text(COMMITTED_PY)
    link = slurm_root / "issue-9997"
    link.symlink_to(outside)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, "issue-9997")
    assert row["disposition"] == "slurm-src-containment-kept"
    assert link.is_symlink()
    assert (outside / "precious.py").is_file()


def test_g3_non_directory_entry_kept(slurm_root, main_repo, repo):
    """T1/g3: a plain FILE with an issue-shaped name is kept (containment)."""
    f = slurm_root / "issue-9996"
    f.write_text("not a directory\n")
    res = _sweep(slurm_root, main_repo, apply=True)
    assert _row(res, "issue-9996")["disposition"] == "slurm-src-containment-kept"
    assert f.is_file()


def test_g4_active_issue_kept_and_escalates(slurm_root, main_repo, repo, tmp_path):
    """T1/g4 + T7: an ACTIVE issue's tree is KEPT (escalate-only, never a
    deletion) and one sidecar row is appended in apply mode."""
    cand = _staged_copy(slurm_root, "issue-9995")
    res = _sweep(
        slurm_root,
        main_repo,
        apply=True,
        status="running",
        escalation_state_path=tmp_path / "esc.json",
    )
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-active-kept"
    assert row["status"] == "running"
    assert row["bytes"] > 0
    assert cand.exists() and (cand / "tracked.py").is_file()
    rows = _sidecar_rows(repo)
    assert [r["kind"] for r in rows] == ["slurm-src-active-kept"]
    assert res.bytes_freed == 0


def test_g4_unresolved_status_kept(slurm_root, main_repo, repo):
    """T1/g4: an UNRESOLVABLE status (resolver returns None) is treated as
    not-terminal — kept + named ``unresolved``."""
    cand = _staged_copy(slurm_root, "issue-9994")
    res = _sweep(slurm_root, main_repo, apply=True, status_resolver=lambda n: None)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-active-kept"
    assert "unresolved" in row["reason"]
    assert cand.exists()


def test_g4b_status_probe_failure_kept_and_escalates(slurm_root, main_repo, repo):
    """T1/g4b: a status probe that RAISES keeps the tree with its own
    disposition, names the exception, and escalates."""
    cand = _staged_copy(slurm_root, "issue-9993")

    def boom(n: int):
        raise RuntimeError("registry unreadable")

    res = _sweep(slurm_root, main_repo, apply=True, status_resolver=boom)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-status-probe-failed-kept"
    assert "RuntimeError" in row["reason"]
    assert cand.exists()
    assert [r["kind"] for r in _sidecar_rows(repo)] == ["slurm-src-status-probe-failed-kept"]


def test_armed_without_resolver_raises(slurm_root, main_repo):
    """Fail-fast (no silent defaults): an ARMED sweep with no status resolver
    or terminal set is a ValueError — a status-blind sweep could reap an
    ACTIVE issue's staging tree."""
    with pytest.raises(ValueError, match="status_resolver"):
        ced.sweep_slurm_src(slurm_root, apply=False, main_repo=main_repo)


# ─── T4/T5/T6/T9/T10/T11: git-class probes through the shared core ───────────


def test_terminal_gitless_staged_copy_reaped(slurm_root, main_repo, repo):
    """T4-adjacent: the production shape — a git-less ``materialize_branch_src``
    copy of committed content on a TERMINAL issue — is reaped."""
    cand = _staged_copy(slurm_root, "issue-9990")
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert not cand.exists()
    assert res.bytes_freed > 0


def test_registered_worktree_reaped_and_unregistered(slurm_root, main_repo, repo):
    """T4: a genuine ``git worktree add --detach`` fixture on a TERMINAL issue
    reaps via ``git worktree remove --force`` — tree gone AND unregistered."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9989")
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert "worktree-removed" in row["reason"]
    assert not cand.exists()
    assert str(cand) not in _git(main_repo, "worktree", "list").stdout


def test_dirty_worktree_tracked_edit_kept(slurm_root, main_repo, repo):
    """T5(a): a tracked-file modification keeps the tree (worktree-dirty)."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9988")
    (cand / "tracked.py").write_text(COMMITTED_PY + "# local edit\n")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "worktree-dirty" in row["reason"]
    assert cand.exists()


def test_dirty_worktree_untracked_file_kept(slurm_root, main_repo, repo):
    """T5(b): an untracked file keeps the tree (worktree-dirty)."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9987")
    (cand / "notes.txt").write_text("uncommitted work\n")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "worktree-dirty" in row["reason"]
    assert cand.exists() and (cand / "notes.txt").is_file()


def test_ignored_file_defeats_blob_proof(slurm_root, main_repo, repo):
    """T5(c): a gitignored file leaves ``git status`` clean but its blob is
    not in the odb — the per-file proof keeps the tree and NAMES the file."""
    (main_repo / ".gitignore").write_text("*.secret\n")
    _git(main_repo, "add", ".gitignore")
    _git(main_repo, "commit", "-m", "ignore secrets")
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9986")
    (cand / "x.secret").write_text("never committed\n")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "x.secret" in row["reason"]
    assert cand.exists() and (cand / "x.secret").is_file()


def test_locked_worktree_kept(slurm_root, main_repo, repo):
    """T9: a lock present at gate time keeps via the class probe."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9985")
    _git(main_repo, "worktree", "lock", str(cand))
    res = _sweep(slurm_root, main_repo, apply=True)
    assert _row(res, cand.name)["disposition"] == "slurm-src-worktree-locked-kept"
    assert cand.exists()


def test_detached_unpushed_head_kept(slurm_root, main_repo, repo):
    """T10: a worktree whose detached HEAD is unreachable from every main-repo
    ref is kept (the commit dies on prune)."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9984")
    (cand / "tracked.py").write_text(COMMITTED_PY + "# detached work\n")
    _git(cand, "commit", "-am", "detached-only commit")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "head-unreachable" in row["reason"]
    assert cand.exists()


def test_corrupt_git_pointer_kept(slurm_root, main_repo, repo):
    """T11: a corrupt ``.git`` pointer file (unknown git kind) fails toward
    keep through the class probes — never reaped, never raises."""
    cand = _staged_copy(slurm_root, "issue-9983")
    (cand / ".git").write_text("gitdir: /nonexistent/nowhere\n")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"].startswith("slurm-src-")
    assert "kept" in row["disposition"]
    assert cand.exists()


def test_reap_reprobe_lock_race_kept(slurm_root, main_repo, repo, monkeypatch):
    """T6: a lock acquired BETWEEN verification and reap trips the reap-time
    class RE-probe — kept, still registered (shared-core behavior under the
    slurm-src leg tag)."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9982")
    real_hit = ced._scratch_live_process_hit

    def lock_then_pass(c):
        _git(main_repo, "worktree", "lock", str(cand))
        return real_hit(c)

    monkeypatch.setattr(ced, "_scratch_live_process_hit", lock_then_pass)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "worktree-locked" in row["reason"]
    assert cand.exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout


def test_report_mode_would_reap_zero_mutation(slurm_root, main_repo, repo):
    """Report-only mode marks ``would-reap`` and deletes nothing."""
    cand = _staged_copy(slurm_root, "issue-9981")
    res = _sweep(slurm_root, main_repo, apply=False)
    assert _row(res, cand.name)["disposition"] == "would-reap"
    assert cand.exists() and (cand / "tracked.py").is_file()
    assert res.bytes_freed == 0


# ─── T3: kill switches; T2: hermeticity ──────────────────────────────────────


@pytest.mark.parametrize("env_var", [ced.SLURM_SRC_SWEEP_KILL_ENV, ced.NONCANONICAL_SWEEP_KILL_ENV])
def test_kill_switches_disable_the_sweep(slurm_root, main_repo, repo, monkeypatch, env_var):
    """T3: either layer (own switch / family switch) disables the leg —
    no rows, no deletion."""
    cand = _staged_copy(slurm_root, "issue-9980")
    monkeypatch.setenv(env_var, "1")
    res = _sweep(slurm_root, main_repo, apply=True)
    assert res.rows == []
    assert cand.exists()


def test_tier_f_kill_switch_does_not_disable_slurm_leg(slurm_root, main_repo, repo, monkeypatch):
    """T3 scope pin: tier (f)'s OWN switch (EPM_SKIP_TMP_SCRATCH_SWEEP) does
    NOT kill the slurm-src leg — only the family switch is shared."""
    _staged_copy(slurm_root, "issue-9979")
    monkeypatch.setenv(ced.SCRATCH_SWEEP_KILL_ENV, "1")
    res = _sweep(slurm_root, main_repo, apply=False)
    assert [r["name"] for r in res.rows] == ["issue-9979"]


def test_no_opt_in_means_no_sweep(main_repo, slurm_root):
    """T2: staging_root=None / main_repo=None never touch anything."""
    assert ced.sweep_slurm_src(None, apply=True, main_repo=main_repo).rows == []
    assert ced.sweep_slurm_src(slurm_root, apply=True, main_repo=None).rows == []


# ─── T12a/T12b: durable-named content is PROOF-gated, never presence-blocked ─


def test_t12a_durable_content_proven_reaps(slurm_root, main_repo, repo):
    """T12a (plan §0 POSITION_A): committed ``eval_results/`` content inside a
    staged copy is PROVEN against the odb and the tree reaps — no
    durable-path presence gate."""
    cand = slurm_root / "issue-9978"
    (cand / "eval_results" / "issue_1").mkdir(parents=True)
    (cand / "eval_results" / "issue_1" / "data.json").write_text(COMMITTED_JSON)
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    assert _row(res, cand.name)["disposition"] == "slurm-src-reaped"
    assert not cand.exists()


def test_t12b_durable_unproven_keeps_and_names_path(slurm_root, main_repo, repo):
    """T12b: an UNPROVEN file under ``eval_results/`` is denied the small-text
    tolerance (``under_durable``) — the tree is KEPT and the file named —
    while the SAME file outside a durable component is tolerated and the
    sibling tree reaps."""
    kept = slurm_root / "issue-9977"
    (kept / "eval_results" / "issue_1").mkdir(parents=True)
    (kept / "eval_results" / "issue_1" / "run.log").write_text("uncommitted telemetry\n")
    (kept / "tracked.py").write_text(COMMITTED_PY)
    reaped = slurm_root / "issue-9976"
    reaped.mkdir()
    (reaped / "run.log").write_text("uncommitted telemetry\n")
    (reaped / "tracked.py").write_text(COMMITTED_PY)
    _backdate(kept)
    _backdate(reaped)
    res = _sweep(slurm_root, main_repo, apply=True)
    kept_row = _row(res, "issue-9977")
    assert kept_row["disposition"] == "slurm-src-unverified-kept"
    assert "eval_results/issue_1/run.log" in kept_row["reason"]
    assert kept.exists() and (kept / "eval_results" / "issue_1" / "run.log").is_file()
    assert _row(res, "issue-9976")["disposition"] == "slurm-src-reaped"
    assert not reaped.exists()


# ─── T13: D6 leg-scoped escalation dedup ─────────────────────────────────────


def test_t13_escalation_dedup_one_row_then_weekly_realert(slurm_root, main_repo, repo, tmp_path):
    """T13: two apply sweeps over the same standing keep append ONE sidecar
    row; a third sweep past the 7-day re-alert window appends a second."""
    cand = _staged_copy(slurm_root, "issue-9975")
    state = tmp_path / "slurm-esc.json"
    now = time.time()
    common = dict(apply=True, status="running", escalation_state_path=state)
    _sweep(slurm_root, main_repo, now=now, **common)
    assert len(_sidecar_rows(repo)) == 1
    _sweep(slurm_root, main_repo, now=now + 3600.0, **common)
    assert len(_sidecar_rows(repo)) == 1  # deduped within the window
    _sweep(slurm_root, main_repo, now=now + 8 * 86400.0, **common)
    assert len(_sidecar_rows(repo)) == 2  # weekly re-alert
    assert cand.exists()


def test_t13_report_mode_never_writes_dedup_state(slurm_root, main_repo, repo, tmp_path):
    """D6: report-only runs never dedup and never write state — the printed
    report stays complete and the production cadence is unaffected."""
    _staged_copy(slurm_root, "issue-9974")
    state = tmp_path / "slurm-esc.json"
    _sweep(slurm_root, main_repo, apply=False, status="running", escalation_state_path=state)
    assert not state.exists()
    assert _sidecar_rows(repo) == []  # apply=False: report-only, no sidecar append


# ─── T14: D9 nested working-tree overlay evidence ────────────────────────────


def _nested_overlay_clone(cand: Path, main_repo: Path) -> Path:
    """A clean nested clone at the D9 overlay path ``external/open-instruct``."""
    nested = cand / "external" / "open-instruct"
    nested.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "clone", str(main_repo), str(nested)],
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )
    assert r.returncode == 0, r.stderr
    return nested


def test_t14_clean_nested_overlay_reaps(slurm_root, main_repo, repo):
    """T14(a): overlay files proven against the NESTED repo's own odb (clean
    tree, reachable HEAD) let the whole staged tree reap — using the DEFAULT
    overlay set (the writer's ``WORKING_TREE_OVERLAY_PATHS`` constant)."""
    cand = slurm_root / "issue-9973"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _nested_overlay_clone(cand, main_repo)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert "overlay files verified in nested odb" in row["evidence"]
    assert not cand.exists()


def test_t14_dirty_nested_overlay_keeps(slurm_root, main_repo, repo):
    """T14(b): a DIRTY nested overlay keeps the whole tree."""
    cand = slurm_root / "issue-9972"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    nested = _nested_overlay_clone(cand, main_repo)
    (nested / "tracked.py").write_text(COMMITTED_PY + "# nested local edit\n")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-dirty" in row["reason"]
    assert cand.exists()


def test_t14_overlay_not_a_repo_keeps_no_outer_fallback(slurm_root, main_repo, repo):
    """T14(c): an overlay path that is NOT a git repo KEEPs the tree even
    though its file content IS committed in the outer main repo — the D9
    contract has deliberately NO fallback to the outer-odb proof."""
    cand = slurm_root / "issue-9971"
    (cand / "external" / "open-instruct").mkdir(parents=True)
    (cand / "external" / "open-instruct" / "copy.py").write_text(COMMITTED_PY)
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-not-a-repo" in row["reason"]
    assert cand.exists()


def test_default_overlay_paths_is_writer_constant():
    """D9: the default overlay set IS the writer's constant, imported —
    never a re-typed literal."""
    from explore_persona_space.backends.slurm import WORKING_TREE_OVERLAY_PATHS

    assert ced.WORKING_TREE_OVERLAY_PATHS is WORKING_TREE_OVERLAY_PATHS
    assert "external/open-instruct" in WORKING_TREE_OVERLAY_PATHS


# ─── T15: no-loosening parity of the extracted tier-(f) core ─────────────────


def test_t15_tmp_scratch_evidence_call_shape_unchanged(tmp_path, monkeypatch, repo, main_repo):
    """T15: the extraction seam exists AND the tier-(f) leg's evidence call
    shape stays BYTE-IDENTICAL to the pre-#2147 loop — an OLD-signature
    evidence stub (no ``overlay_paths`` parameter) still works, so the
    pre-existing tier-(f) suite's stubs cannot break."""
    assert hasattr(ced, "_sweep_scratch_candidate")  # the shared-core seam
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    cand = tmp_root / "issue-9970-gate2"
    cand.mkdir()
    (cand / "copy.py").write_text(COMMITTED_PY)
    _backdate(cand)

    def old_signature_stub(c, *, main_repo, full_stats, verdict_cache=None):
        return "git-blob-reproducible: stub", {
            "reason": "pass",
            "first_unverified": None,
            "n_verified": 1,
            "n_tolerated": 0,
            "git_class": "none",
        }

    monkeypatch.setattr(ced, "_git_blob_reproducibility_evidence", old_signature_stub)
    res = ced.sweep_tmp_scratch(tmp_root, apply=False, main_repo=main_repo)
    row = next(r for r in res.rows if r["name"] == cand.name)
    assert row["leg"] == "tmp-scratch"
    assert row["disposition"] == "would-reap"


def test_t15_slurm_leg_threads_overlay_kwarg(slurm_root, main_repo, repo, monkeypatch):
    """T15 counterpart: the slurm-src leg DOES thread ``overlay_paths`` into
    the evidence call (a stub REQUIRING the kwarg receives it)."""
    _staged_copy(slurm_root, "issue-9969")
    seen: list[tuple[str, ...]] = []

    def new_signature_stub(c, *, main_repo, full_stats, verdict_cache=None, overlay_paths):
        seen.append(overlay_paths)
        return "git-blob-reproducible: stub", {
            "reason": "pass",
            "first_unverified": None,
            "n_verified": 1,
            "n_tolerated": 0,
            "git_class": "none",
        }

    monkeypatch.setattr(ced, "_git_blob_reproducibility_evidence", new_signature_stub)
    res = _sweep(slurm_root, main_repo, apply=False)
    assert [r["disposition"] for r in res.rows] == ["would-reap"]
    assert seen == [ced.WORKING_TREE_OVERLAY_PATHS]


# ─── vm_disk_guard tier (g) wiring ───────────────────────────────────────────


def _stub_reclaim_tiers(monkeypatch):
    """Neutralize the /-rooted + HF tiers so run_guard tests exercise only
    tiers (b)/(f)/(g)."""
    for fn in (
        "clean_uv_cache",
        "clean_stale_logs",
        "clean_vm_workspace_hf_cache",
        "clean_home_hf_stale_revisions",
    ):
        monkeypatch.setattr(vdg, fn, lambda *a, _n=fn, **k: vdg.TierResult(name=_n))


def test_clean_slurm_src_skips_without_opt_in():
    res = vdg.clean_slurm_src(False, staging_root=None, main_repo=None)
    assert res.skipped and "hermetic" in res.skip_reason


def test_clean_slurm_src_skips_on_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(ced.SLURM_SRC_SWEEP_KILL_ENV, "1")
    res = vdg.clean_slurm_src(False, staging_root=tmp_path, main_repo=tmp_path)
    assert res.skipped and ced.SLURM_SRC_SWEEP_KILL_ENV in res.skip_reason


def test_run_guard_library_call_appends_no_slurm_tier(tmp_path, repo, monkeypatch):
    """T2: a library run_guard call WITHOUT the slurm opt-ins never appends
    the tier — and therefore can never scan any real staging root."""
    _stub_reclaim_tiers(monkeypatch)
    data_root = tmp_path / "data"
    data_root.mkdir()
    res = vdg.run_guard(False, data_root=data_root, ignore_threshold=True)
    assert "slurm-src" not in [t.name for t in res.tiers]


def test_run_guard_wires_slurm_tier_after_tmp_scratch(
    tmp_path, slurm_root, main_repo, repo, monkeypatch
):
    """Boot-pass wiring: tier (g) runs right after tier (f); its rows ride
    ``--json``'s tier dict as ``scratch_candidates``; report mode mutates
    nothing."""
    _stub_reclaim_tiers(monkeypatch)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    cand = _staged_copy(slurm_root, "issue-9968")
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
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
        slurm_src_staging_root=slurm_root,
        slurm_src_main_repo=main_repo,
    )
    names = [t.name for t in res.tiers]
    assert names.index("slurm-src") == names.index("tmp-scratch") + 1
    tier_g = next(t for t in res.tiers if t.name == "slurm-src")
    assert [r["name"] for r in tier_g.scratch_candidates] == ["issue-9968"]
    assert tier_g.scratch_candidates[0]["disposition"] == "would-reap"
    assert cand.exists()  # report-only: zero mutation
    payload = vdg._result_json(res)
    tier_dicts = {t["name"]: t for t in payload["tiers"]}
    assert tier_dicts["slurm-src"]["scratch_candidates"][0]["name"] == "issue-9968"
    json.dumps(payload)  # rows must stay JSON-serializable


def test_run_guard_slurm_kill_switch_drops_tier(tmp_path, slurm_root, main_repo, repo, monkeypatch):
    """Kill switch set: no slurm-src tier, tree untouched."""
    _stub_reclaim_tiers(monkeypatch)
    monkeypatch.setenv(ced.SLURM_SRC_SWEEP_KILL_ENV, "1")
    cand = _staged_copy(slurm_root, "issue-9967")
    data_root = tmp_path / "data"
    data_root.mkdir()
    res = vdg.run_guard(
        False,
        data_root=data_root,
        ignore_threshold=True,
        slurm_src_staging_root=slurm_root,
        slurm_src_main_repo=main_repo,
    )
    assert "slurm-src" not in [t.name for t in res.tiers]
    assert cand.exists()


def test_run_guard_data_disk_shape_never_runs_slurm_tier(slurm_root, main_repo, repo, monkeypatch):
    """Boot-disk-only pin: the data-disk pass shape (reclaim_tiers=False)
    never runs tier (g) even when the opt-ins are (wrongly) passed."""
    _stub_reclaim_tiers(monkeypatch)
    res = vdg.run_guard(
        False,
        reclaim_tiers=False,
        ignore_threshold=True,
        slurm_src_staging_root=slurm_root,
        slurm_src_main_repo=main_repo,
    )
    assert "slurm-src" not in [t.name for t in res.tiers]


# ─── T8: the plan §6 acceptance argv, report-mode zero-mutation ──────────────


def test_t8_main_acceptance_argv_report_mode_zero_mutation(
    tmp_path, slurm_root, main_repo, repo, monkeypatch, capsys
):
    """T8: the EXACT plan §6 acceptance invocation —
    ``EPS_SCRATCH_VERDICT_CACHE=... vm_disk_guard.py --ignore-threshold
    --json`` — mutates NOTHING: the reap-eligible worktree survives and
    stays registered, no sidecar row lands, the PRODUCTION verdict cache is
    never created (the override path takes the writes), and the slurm-src
    tier rides the JSON with a ``would-reap`` row and ``bytes_freed == 0``."""
    _stub_reclaim_tiers(monkeypatch)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    monkeypatch.setattr(vdg, "_telegram_push", lambda msg, apply: False)
    monkeypatch.setattr(vdg, "_resolution_root", lambda: main_repo)
    tmp_root = tmp_path / "faketmp"
    tmp_root.mkdir()
    monkeypatch.setattr(vdg, "production_tmp_root", lambda: tmp_root)
    wt = _slurm_worktree(main_repo, slurm_root, "issue-9966")
    copy = _staged_copy(slurm_root, "issue-9965")
    override_cache = tmp_path / "issue2147-acceptance-cache.json"
    monkeypatch.setenv("EPS_SCRATCH_VERDICT_CACHE", str(override_cache))
    monkeypatch.setenv("EPS_SLURM_SRC_ROOT", str(slurm_root))
    monkeypatch.setenv("EPS_VM_DATA_DISK_PATH", str(tmp_path / "no-such-mount"))

    rc = vdg.main(["--ignore-threshold", "--json"])

    assert rc in (0, 2)  # exit 2 = the real disk is still over threshold; never a crash
    payload = json.loads(capsys.readouterr().out)
    tier_dicts = {t["name"]: t for t in payload["tiers"]}
    rows = {r["name"]: r for r in tier_dicts["slurm-src"]["scratch_candidates"]}
    assert set(rows) == {"issue-9966", "issue-9965"}
    assert all(r["disposition"] == "would-reap" for r in rows.values())
    assert tier_dicts["slurm-src"]["bytes_freed"] == 0
    # Zero mutation: trees survive, the worktree stays registered.
    assert wt.exists() and copy.exists()
    assert str(wt) in _git(main_repo, "worktree", "list").stdout
    # No sidecar row (report mode prints, never appends).
    assert _sidecar_rows(repo) == []
    assert not (main_repo / ".claude" / "cache" / "disk-guard-events.jsonl").exists()
    # The PRODUCTION verdict cache is untouched; the override took the writes.
    assert not (tmp_path / ced.SCRATCH_VERDICT_CACHE_REL).exists()
    assert not (main_repo / ced.SCRATCH_VERDICT_CACHE_REL).exists()
    assert override_cache.is_file()
