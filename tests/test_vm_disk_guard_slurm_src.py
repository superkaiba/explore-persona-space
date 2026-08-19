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
import shutil
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


# ─── T14: D9 nested working-tree overlay evidence (round 2: C2 + C3 + M3) ────

NESTED_ONLY_PY = "print('unique to the nested overlay repo, never in the outer odb')\n"


def _surviving_overlay(main_repo: Path) -> Path:
    """The SURVIVING overlay repo at ``main_repo/external/open-instruct`` —
    the round-2 C2 anchor every scratch-copy overlay proof verifies against.
    Holds a committed file whose content is deliberately ABSENT from the
    outer main odb (M3: nested blobs must not ride the outer proof)."""
    surv = main_repo / "external" / "open-instruct"
    if not (surv / ".git").exists():
        surv.mkdir(parents=True, exist_ok=True)
        _git(surv, "init", "-b", "main")
        (surv / "nested_only.py").write_text(NESTED_ONLY_PY)
        _git(surv, "add", "-A")
        _git(surv, "commit", "-m", "nested init")
    return surv


def _staged_overlay_copy(cand: Path, main_repo: Path) -> Path:
    """The ``materialize_branch_src`` overlay shape: an rsync-style FULL copy
    (``.git`` included) of the surviving overlay repo into the staged tree —
    byte-identical trees, exactly the measured production state."""
    surv = _surviving_overlay(main_repo)
    nested = cand / "external" / "open-instruct"
    nested.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(surv, nested, symlinks=True)
    return nested


def _nested_overlay_clone(cand: Path, main_repo: Path) -> Path:
    """A clean nested clone OF THE OUTER MAIN REPO at the overlay path —
    kept only for the C2 surviving-repo-missing case (its blobs live in the
    outer odb, and ``main_repo`` has no surviving overlay copy)."""
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
    """T14(a), M3 rebuild: overlay files proven against the SURVIVING overlay
    repo let the whole staged tree reap — with the nested-only blob asserted
    ABSENT from the outer odb BEFORE the sweep, so a PASS can only come from
    the overlay arm (pre-M3 the fixture cloned the outer repo, making the
    outer proof able to mask a broken overlay arm)."""
    cand = slurm_root / "issue-9973"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    nested = _staged_overlay_copy(cand, main_repo)
    # M3: blob disjointness — cat-file FAILS in the outer odb, SUCCEEDS in
    # the surviving overlay repo (and its staged copy).
    sha = _git(main_repo, "hash-object", str(nested / "nested_only.py")).stdout.strip()
    assert _git(main_repo, "cat-file", "-e", sha, check=False).returncode != 0
    surv = main_repo / "external" / "open-instruct"
    assert _git(surv, "cat-file", "-e", sha).returncode == 0
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert "overlay files verified in the surviving overlay repo" in row["evidence"]
    assert not cand.exists()
    # The C2 anchor itself is never touched by the reap.
    assert (surv / "nested_only.py").is_file()


def test_t14_dirty_nested_overlay_keeps(slurm_root, main_repo, repo):
    """T14(b): a DIRTY nested overlay keeps the whole tree."""
    cand = slurm_root / "issue-9972"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    nested = _staged_overlay_copy(cand, main_repo)
    (nested / "nested_only.py").write_text(NESTED_ONLY_PY + "# nested local edit\n")
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


def test_t14_c2_unpushed_nested_ref_keeps(slurm_root, main_repo, repo):
    """C2: a nested overlay with a CLEAN tree but a local-only branch commit
    (absent from the SURVIVING repo) KEEPs — under the round-1 nested-odb
    anchor this tree read as reapable and the commit died with it."""
    cand = slurm_root / "issue-9964"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    nested = _staged_overlay_copy(cand, main_repo)
    _git(nested, "checkout", "-b", "local-only")
    (nested / "wip.py").write_text("print('nested-only work')\n")
    _git(nested, "add", "wip.py")
    _git(nested, "commit", "-m", "nested-only commit")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-unpushed-ref" in row["reason"]
    assert cand.exists() and (nested / "wip.py").is_file()


def test_t14_c2_nested_stash_keeps(slurm_root, main_repo, repo):
    """C2: a nested overlay with a clean tree but a non-empty OWN stash
    KEEPs — stashed work dies with the tree; the round-1 probe never
    checked the stash."""
    cand = slurm_root / "issue-9963"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    nested = _staged_overlay_copy(cand, main_repo)
    (nested / "nested_only.py").write_text(NESTED_ONLY_PY + "# stashed edit\n")
    _git(nested, "stash")
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-stash" in row["reason"]
    assert cand.exists()
    assert _git(nested, "stash", "list").stdout.strip()  # the stash survives


def test_t14_c2_surviving_repo_missing_keeps(slurm_root, main_repo, repo):
    """C2: a clean, valid nested overlay whose SURVIVING anchor repo does not
    exist in the main working tree KEEPs — with no surviving copy, deletion
    would be unrecoverable regardless of the nested repo's own health."""
    cand = slurm_root / "issue-9959"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _nested_overlay_clone(cand, main_repo)  # main_repo has NO external/open-instruct
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-surviving-repo-missing" in row["reason"]
    assert cand.exists()


def test_t14_c3_overlay_with_only_tolerated_log_keeps(slurm_root, main_repo, repo):
    """C3: a NON-git overlay holding ONLY a small tolerated ``.log`` file
    KEEPs — round 1 tolerated the log, left ``overlay_entries`` empty, and
    fell back to the outer proof, silently reaping the overlay."""
    cand = slurm_root / "issue-9958"
    ov = cand / "external" / "open-instruct"
    ov.mkdir(parents=True)
    (ov / "run.log").write_text("small telemetry only\n")
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-not-a-repo" in row["reason"]
    assert cand.exists() and (ov / "run.log").is_file()


def test_t14_c3_overlay_with_only_exempt_and_symlink_content_keeps(
    slurm_root, main_repo, repo, tmp_path
):
    """C3: a NON-git overlay holding only exempt-dir content and a symlink
    (nothing the walk hashes) KEEPs via the presence-keyed validation."""
    cand = slurm_root / "issue-9957"
    ov = cand / "external" / "open-instruct"
    (ov / "__pycache__").mkdir(parents=True)
    (ov / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
    target = tmp_path / "symlink-target.py"
    target.write_text(COMMITTED_PY)
    (ov / "link.py").symlink_to(target)
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-not-a-repo" in row["reason"]
    assert cand.exists() and target.is_file()


def test_t14_c3_empty_overlay_dir_keeps(slurm_root, main_repo, repo):
    """C3: an EMPTY declared-overlay dir in an otherwise-proven tree KEEPs —
    presence of the declared path alone demands a positively-established
    nested repo before any other disposition."""
    cand = slurm_root / "issue-9956"
    (cand / "external" / "open-instruct").mkdir(parents=True)
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


# ─── round 2 C1: worktree reap dispatch keys SOLELY on the proven kind ────────


def _rmtree_boom(path, *a, **k):
    raise AssertionError(f"shutil.rmtree reached for a registered worktree: {path}")


def test_c1_transient_admin_failure_never_routes_worktree_to_rmtree(
    slurm_root, main_repo, repo, monkeypatch
):
    """C1: a TRANSIENT ``_worktree_admin_of_main`` failure AFTER the class
    probes must never route a REGISTERED worktree to ``shutil.rmtree``.

    Post-fix the admin lookup runs exactly TWICE per candidate (evidence
    class probe + reap-time class re-probe, both inside
    ``_scratch_git_class_probes``); the pre-fix reap dispatch made a THIRD
    call whose transient False fell through to rmtree. The fake succeeds for
    the two probe calls and fails afterwards; the rmtree spy turns any
    worktree-rmtree into a hard failure. Post-fix outcome: a clean checked
    ``git worktree remove`` (rmtree never consulted)."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9955")
    real = ced._worktree_admin_of_main
    calls = {"n": 0}

    def flaky(admin, mr):
        calls["n"] += 1
        if calls["n"] > 2:
            return False  # the transient registration-lookup failure
        return real(admin, mr)

    monkeypatch.setattr(ced, "_worktree_admin_of_main", flaky)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert "worktree-removed" in row["reason"]
    assert not cand.exists()
    assert str(cand) not in _git(main_repo, "worktree", "list").stdout
    assert calls["n"] == 2  # the pin: no third (dispatch-condition) lookup exists


def test_c1_worktree_remove_failure_keeps_never_rmtree(slurm_root, main_repo, repo, monkeypatch):
    """C1: when ``git worktree remove`` FAILS, the worktree-class candidate is
    KEPT (tree + registration intact) — ``shutil.rmtree`` is structurally
    unreachable for ``kind == "worktree"``."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9954")
    real_git = ced._git

    def failing_remove(args, **kw):
        if args[:2] == ["worktree", "remove"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="simulated failure")
        return real_git(args, **kw)

    monkeypatch.setattr(ced, "_git", failing_remove)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-worktree-remove-failed"
    assert "simulated failure" in row["reason"]
    assert cand.exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout


# ─── round 2 C4: staging-root contract (fail LOUD before any probe) ──────────


@pytest.mark.parametrize(
    "case", ["slash", "slash_child", "home", "home_ancestor", "repo_root", "relative"]
)
def test_c4_dangerous_staging_root_refused_before_any_probe(case, main_repo):
    """C4: a misconfigured staging root (``/``, a direct child of ``/``,
    ``$HOME`` or an ancestor, a git repo root, a relative path) aborts the
    ARMED sweep with a ValueError BEFORE any enumeration or status/evidence
    probe — the spy resolver proves no candidate was ever probed."""
    roots = {
        "slash": Path("/"),
        "slash_child": Path("/tmp"),
        "home": Path.home(),
        "home_ancestor": Path.home().parent,
        "repo_root": main_repo,
        "relative": Path("relative-staging-root"),
    }
    resolver_calls: list[int] = []

    def spy_resolver(n: int):
        resolver_calls.append(n)
        raise AssertionError("status probe reached under a dangerous staging root")

    with pytest.raises(ValueError, match="staging root"):
        ced.sweep_slurm_src(
            roots[case],
            apply=True,
            main_repo=main_repo,
            status_resolver=spy_resolver,
            terminal_statuses=frozenset({"completed"}),
        )
    assert resolver_calls == []


def test_c4_mount_point_staging_root_refused(tmp_path, main_repo, monkeypatch):
    """C4: a staging root resolving to a filesystem MOUNT POINT is refused
    (the ``/mnt/eps-data``-class whole-disk anchor), and a candidate under
    it is never touched."""
    root = tmp_path / "mnt-like"
    root.mkdir()
    cand = root / "issue-9953"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    real_ismount = os.path.ismount
    root_real = os.path.realpath(root)

    def fake_ismount(p):
        if os.path.realpath(str(p)) == root_real:
            return True
        return real_ismount(p)

    monkeypatch.setattr(ced.os.path, "ismount", fake_ismount)
    with pytest.raises(ValueError, match="mount point"):
        ced.sweep_slurm_src(
            root,
            apply=True,
            main_repo=main_repo,
            status_resolver=lambda n: "completed",
            terminal_statuses=frozenset({"completed"}),
        )
    assert cand.exists() and (cand / "tracked.py").is_file()


# ─── round 2 M1: enumeration failures are explicit, never an empty sweep ─────


def test_m1_absent_staging_root_is_explicit_skip(tmp_path, main_repo):
    """M1: a nonexistent staging root sets ``skip_reason`` — an explicit
    "did not enumerate" signal, never an indistinguishable empty result."""
    root = tmp_path / "eps-slurm-src-never-created"
    res = ced.sweep_slurm_src(
        root,
        apply=True,
        main_repo=main_repo,
        status_resolver=lambda n: "completed",
        terminal_statuses=frozenset({"completed"}),
    )
    assert res.rows == []
    assert res.skip_reason is not None and "absent" in res.skip_reason


def test_m1_unreadable_staging_root_raises(slurm_root, main_repo, monkeypatch):
    """M1: any NON-absence enumeration failure RAISES — a permission/IO
    error reported as an empty sweep would hide a broken tier forever."""
    real_listdir = os.listdir

    def denied(path=None):
        if path is not None and os.path.realpath(str(path)) == os.path.realpath(str(slurm_root)):
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path) if path is not None else real_listdir()

    monkeypatch.setattr(ced.os, "listdir", denied)
    with pytest.raises(RuntimeError, match="cannot enumerate"):
        _sweep(slurm_root, main_repo, apply=True)


def test_m1_clean_slurm_src_surfaces_absent_root_as_skipped(tmp_path, main_repo, monkeypatch):
    """M1 (tier adapter): the vm_disk_guard tier maps the sweep's
    ``skip_reason`` onto ``TierResult.skipped`` instead of reporting an
    empty tier."""
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    res = vdg.clean_slurm_src(
        True, staging_root=tmp_path / "eps-slurm-src-never-created", main_repo=main_repo
    )
    assert res.skipped
    assert res.skip_reason is not None and "absent" in res.skip_reason
    assert res.detail == [] and res.scratch_candidates == []


def test_m1_state_write_failure_is_logged_not_silent(tmp_path, capsys):
    """M1: a dedup-state write failure is fail-soft but LOGGED with the path
    and error — never ``except OSError: pass``."""
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("a file where a directory is needed\n")
    ced._save_slurm_src_escalation_state(blocked / "esc.json", {"k": {"ts": 1.0}})
    err = capsys.readouterr().err
    assert "WARNING" in err and "escalation dedup state" in err and "esc.json" in err


# ─── round 2 M2: reason slug in the dedup key; commit-after-emit ─────────────


def test_m2_reason_change_realerts_within_window(slurm_root, main_repo, repo, tmp_path):
    """M2: a materially different reason under the SAME disposition
    re-alerts WITHIN the dedup window (the plan-D6 ``(path, reason, band)``
    key) — and the same reason still dedups."""
    cand = _staged_copy(slurm_root, "issue-9952")
    state = tmp_path / "esc.json"
    now = time.time()
    common = dict(apply=True, escalation_state_path=state)
    _sweep(slurm_root, main_repo, now=now, status="running", **common)
    assert len(_sidecar_rows(repo)) == 1
    # Same disposition (slurm-src-active-kept), DIFFERENT reason slug.
    _sweep(slurm_root, main_repo, now=now + 3600.0, status="verifying", **common)
    assert len(_sidecar_rows(repo)) == 2
    # A repeat of the SAME reason within the window still dedups.
    _sweep(slurm_root, main_repo, now=now + 7200.0, status="verifying", **common)
    assert len(_sidecar_rows(repo)) == 2
    assert cand.exists()


def test_m2_failed_append_never_suppresses(slurm_root, main_repo, repo, tmp_path, monkeypatch):
    """M2: the dedup timestamp commits ONLY after the sidecar append lands —
    a FAILED append re-alerts on the next pass instead of being silently
    suppressed for the whole 7-day window."""
    cand = _staged_copy(slurm_root, "issue-9951")
    state = tmp_path / "esc.json"
    now = time.time()
    common = dict(apply=True, status="running", escalation_state_path=state)
    fail_next = {"on": True}
    real_append = ced.append_disk_guard_event

    def flaky_append(event, *, apply=True):
        if fail_next["on"]:
            return False  # the append did NOT land
        return real_append(event, apply=apply)

    monkeypatch.setattr(ced, "append_disk_guard_event", flaky_append)
    _sweep(slurm_root, main_repo, now=now, **common)
    assert _sidecar_rows(repo) == []  # nothing landed
    assert ced._load_slurm_src_escalation_state(state) == {}  # nothing committed
    fail_next["on"] = False
    _sweep(slurm_root, main_repo, now=now + 3600.0, **common)  # well within the window
    assert len(_sidecar_rows(repo)) == 1  # NOT suppressed — the failure never committed
    assert cand.exists()


# ─── round 2 M4 (T15b): REAL tier-(f) parity matrix over sweep_tmp_scratch ───
#
# The T15 tests above pin the extraction SEAM (call shapes); this matrix pins
# the tier-(f) BEHAVIOR through the real shared core — every #2127 protection
# exercised end-to-end on the tmp-scratch leg, with the exact row disposition
# AND the filesystem/registration outcome asserted per arm (plan K3: the
# refactor may not loosen tier (f)).


@pytest.fixture
def scratch_tmp_root(tmp_path):
    root = tmp_path / "faketmp"
    root.mkdir()
    return root


def _tmp_copy(scratch_tmp_root: Path, name: str = "scratch-copy") -> Path:
    cand = scratch_tmp_root / name
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _backdate(cand)
    return cand


def _tmp_sweep(scratch_tmp_root: Path, main_repo: Path, *, apply: bool, **kw):
    return ced.sweep_tmp_scratch(scratch_tmp_root, apply=apply, main_repo=main_repo, **kw)


def _tmp_row(res, name: str) -> dict:
    rows = [r for r in res.rows if r["name"] == name]
    assert rows, f"no tmp-scratch row for {name}; rows={[r['name'] for r in res.rows]}"
    return rows[0]


def test_t15b_walk_error_kept(scratch_tmp_root, main_repo, repo):
    """T15b arm 1: an unreadable subtree keeps (walk error), nothing deleted."""
    if os.geteuid() == 0:
        pytest.skip("chmod 000 does not block root")
    cand = _tmp_copy(scratch_tmp_root, "scratch-walkerr")
    sub = cand / "sub"
    sub.mkdir()
    (sub / "x.py").write_text(COMMITTED_PY)
    _backdate(cand)
    sub.chmod(0o000)
    try:
        res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
        row = _tmp_row(res, cand.name)
        assert row["disposition"] == "tmp-scratch-unverified-kept"
        assert "walk error" in row["reason"]
        assert cand.exists()
    finally:
        sub.chmod(0o755)


def test_t15b_nonregular_kept(scratch_tmp_root, main_repo, repo):
    """T15b arm 2: a FIFO anywhere in the tree keeps with its own tag."""
    cand = _tmp_copy(scratch_tmp_root, "scratch-fifo")
    os.mkfifo(cand / "pipe")
    _backdate(cand)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-nonregular-kept"
    assert cand.exists()


def test_t15b_recent_write_kept(scratch_tmp_root, main_repo, repo):
    """T15b arm 3: a fresh mtime keeps (age only ever a KEEP signal)."""
    cand = _tmp_copy(scratch_tmp_root, "scratch-recent")
    now = time.time()
    os.utime(cand / "tracked.py", (now, now))
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-recent-kept"
    assert cand.exists() and (cand / "tracked.py").is_file()


def test_t15b_unproven_blob_kept_and_named(scratch_tmp_root, main_repo, repo):
    """T15b arm 4: one uncommitted byte keeps the tree and NAMES the file."""
    cand = _tmp_copy(scratch_tmp_root, "scratch-unproven")
    (cand / "precious.py").write_text("never committed anywhere\n")
    _backdate(cand)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-unverified-kept"
    assert "precious.py" in row["reason"]
    assert cand.exists() and (cand / "precious.py").is_file()


def test_t15b_reader_atime_pins_verified_tree(scratch_tmp_root, main_repo, repo):
    """T15b arm 5: a VERIFIED tree recently READ (fresh nlink==1 atime) is
    kept + escalated as atime-pinned, never reaped."""
    cand = _tmp_copy(scratch_tmp_root, "scratch-atime")
    os.utime(cand / "tracked.py", (time.time(), AGED_TS))  # atime fresh, mtime old
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-verified-atime-pinned"
    assert cand.exists()
    assert [r["kind"] for r in _sidecar_rows(repo)] == ["tmp-scratch-verified-atime-pinned"]


def test_t15b_live_process_holds_tree(scratch_tmp_root, main_repo, repo):
    """T15b arm 6: a live process cwd'd inside the tree holds the reap."""
    cand = _tmp_copy(scratch_tmp_root, "scratch-live")
    proc = subprocess.Popen(["sleep", "60"], cwd=cand)
    try:
        res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
        row = _tmp_row(res, cand.name)
        assert row["disposition"] == "tmp-scratch-live-process-kept"
        assert cand.exists()
    finally:
        proc.kill()
        proc.wait()


def test_t15b_cached_pass_state_flip_reprobed_at_reap(scratch_tmp_root, main_repo, repo, tmp_path):
    """T15b arm 7: a CACHED PASS never licenses a deletion on flipped
    external git state — the reap-time class RE-probe catches a ref deleted
    AFTER the verdict was cached (the #2127 gc-residual contract)."""
    cand = _slurm_worktree(main_repo, scratch_tmp_root, "scratch-flip")
    cache = tmp_path / "verdict-cache.json"
    res1 = _tmp_sweep(scratch_tmp_root, main_repo, apply=False, verdict_cache_path=cache)
    assert _tmp_row(res1, cand.name)["disposition"] == "would-reap"
    assert cache.is_file()
    # Flip EXTERNAL git state with ZERO tree change: delete the only branch,
    # making the worktree's detached HEAD unreachable from every ref.
    _git(main_repo, "update-ref", "-d", "refs/heads/main")
    _backdate(cand)  # reset atimes the verification hashing refreshed
    res2 = _tmp_sweep(scratch_tmp_root, main_repo, apply=True, verdict_cache_path=cache)
    row = _tmp_row(res2, cand.name)
    # A cache MISS would keep earlier as unverified-kept; reap-reprobe-kept
    # proves the cached PASS was honored AND the destructive path re-probed.
    assert row["disposition"] == "tmp-scratch-reap-reprobe-kept"
    assert "head-unreachable" in row["reason"]
    assert cand.exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout


def test_t15b_worktree_remove_failure_kept(scratch_tmp_root, main_repo, repo, monkeypatch):
    """T15b arm 8: a failed ``git worktree remove`` keeps tree +
    registration; rmtree is never a fallback (C1 on the tier-(f) leg)."""
    cand = _slurm_worktree(main_repo, scratch_tmp_root, "scratch-rmfail")
    real_git = ced._git

    def failing_remove(args, **kw):
        if args[:2] == ["worktree", "remove"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="simulated failure")
        return real_git(args, **kw)

    monkeypatch.setattr(ced, "_git", failing_remove)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, cand.name)
    assert row["disposition"] == "tmp-scratch-worktree-remove-failed"
    assert cand.exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout


def test_t15b_registered_vs_unregistered_reap_dispatch(scratch_tmp_root, main_repo, repo):
    """T15b arm 9: a REGISTERED worktree reaps via ``git worktree remove``
    (gone AND unregistered); a plain dir reaps via rmtree — the dispatch is
    class-keyed on both legs."""
    wt = _slurm_worktree(main_repo, scratch_tmp_root, "scratch-wt")
    plain = _tmp_copy(scratch_tmp_root, "scratch-plain")
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    wt_row = _tmp_row(res, wt.name)
    assert wt_row["disposition"] == "tmp-scratch-reaped"
    assert "worktree-removed" in wt_row["reason"]
    assert not wt.exists()
    assert str(wt) not in _git(main_repo, "worktree", "list").stdout
    plain_row = _tmp_row(res, plain.name)
    assert plain_row["disposition"] == "tmp-scratch-reaped"
    assert "rmtree" in plain_row["reason"]
    assert not plain.exists()
    assert res.bytes_freed > 0


# ─── round 3 C1: positive non-registration proof before ANY rmtree ───────────


def test_r3c1_registered_worktree_paths_helper(tmp_path, main_repo, monkeypatch):
    """C1 helper: the admin-side listing contains the main working tree AND
    every linked worktree; probe failure or parse ambiguity returns None
    (fail-toward-keep), never an empty 'proof' of non-registration."""
    wt = tmp_path / "helper-wt"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    paths = ced._registered_worktree_paths(main_repo)
    assert paths is not None
    assert os.path.realpath(str(main_repo)) in paths
    assert os.path.realpath(str(wt)) in paths
    # Probe failure => None.
    monkeypatch.setattr(ced, "_git", lambda *a, **k: None)
    assert ced._registered_worktree_paths(main_repo) is None
    # Parse ambiguity (no `worktree ` lines at all) => None: a successful
    # listing always contains at least the main working tree.
    monkeypatch.setattr(
        ced,
        "_git",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="junk\n", stderr=""),
    )
    assert ced._registered_worktree_paths(main_repo) is None


def test_r3c1_registered_worktree_missing_pointer_never_rmtree(
    slurm_root, main_repo, repo, monkeypatch
):
    """C1: a REGISTERED worktree whose in-tree ``.git`` pointer file was
    DELETED classifies as ``none`` (every class probe passes), yet must KEEP —
    pre-fix it fell through to ``shutil.rmtree`` and was deleted WITHOUT
    unregistering, defeating the round-2 structural-unreachability contract
    by a different route."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9944")
    (cand / ".git").unlink()
    _backdate(cand)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "registered-path" in row["reason"]
    assert cand.exists() and (cand / "tracked.py").is_file()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout  # registration survives


def test_r3c1_registered_worktree_pointer_replaced_by_clone_never_rmtree(
    slurm_root, main_repo, repo, monkeypatch
):
    """C1 (replaced-pointer variant): the registered path's content replaced
    by a CLEAN, fully-reachable clone (class ``clone``, every clone probe
    passes) must still KEEP — the registration list, not the tree's own
    ``.git`` entry, is what proves rmtree-eligibility."""
    cand = _slurm_worktree(main_repo, slurm_root, "issue-9943")
    shutil.rmtree(cand)  # simulate the pointer/tree replacement...
    r = subprocess.run(
        ["git", "clone", str(main_repo), str(cand)],
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )
    assert r.returncode == 0, r.stderr
    _backdate(cand)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "registered-path" in row["reason"]
    assert cand.exists()
    assert str(cand) in _git(main_repo, "worktree", "list").stdout


def test_r3c1_registration_probe_failure_keeps_plain_dir(slurm_root, main_repo, repo, monkeypatch):
    """C1: a FAILED registration probe KEEPS even a plain unregistered dir —
    probe failure is ambiguity, never license (fail-toward-keep)."""
    cand = _staged_copy(slurm_root, "issue-9942")
    real_git = ced._git

    def flaky(args, **kw):
        if args[:2] == ["worktree", "list"]:
            return None
        return real_git(args, **kw)

    monkeypatch.setattr(ced, "_git", flaky)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "registration-probe-failed" in row["reason"]
    assert cand.exists()


def test_r3c1_gate17_branch_c_refuses_declassified_registered_worktree(tmp_path, main_repo):
    """C1 sibling site: gate-1.7 evidence branch (c) licenses a plain rmtree
    in the issue-keyed /tmp legs; its kind-based registered-worktree refusal
    is class-downgradeable the same way, so it too must refuse via the
    POSITIVE registration listing when the pointer is gone."""
    wt = tmp_path / "i9941_dl"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()
    _backdate(wt)
    ev, det = ced._tmp_git_evidence_branch_c(wt, main_repo=main_repo)
    assert ev is None
    assert det["reason"] == "registered-worktree"


def test_r3c1_gate17_branch_c_probe_failure_refuses(tmp_path, main_repo, monkeypatch):
    """C1 sibling site, failure arm: a failed registration probe REFUSES the
    evidence branch (fail-toward-keep) instead of proceeding to the blob
    proof."""
    d = tmp_path / "i9940_dl"
    d.mkdir()
    (d / "copy.py").write_text(COMMITTED_PY)
    _backdate(d)
    monkeypatch.setattr(ced, "_registered_worktree_paths", lambda mr: None)
    ev, det = ced._tmp_git_evidence_branch_c(d, main_repo=main_repo)
    assert ev is None
    assert det["reason"] == "registration-probe-failed"


# ─── round 3 C2: overlay proof re-run on the cached-PASS + destructive paths ─


def test_r3c2_cached_pass_surviving_anchor_deleted_keeps(slurm_root, main_repo, repo, tmp_path):
    """C2(a): a CACHED PASS must not be honored after the SURVIVING overlay
    anchor changes with ZERO candidate-tree change — deleting the anchor repo
    after a report-mode pass leaves the cache key identical, and pre-fix the
    apply run reaped the tree, destroying the only remaining overlay copy."""
    cand = slurm_root / "issue-9939"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    _staged_overlay_copy(cand, main_repo)
    _backdate(cand)
    cache = tmp_path / "verdict-cache.json"
    res1 = _sweep(slurm_root, main_repo, apply=False, verdict_cache_path=cache)
    assert _row(res1, cand.name)["disposition"] == "would-reap"
    assert cache.is_file()
    # Flip EXTERNAL overlay state with zero candidate-tree change: the
    # surviving anchor repo disappears (rebase/cleanup in the main tree).
    shutil.rmtree(main_repo / "external" / "open-instruct")
    _backdate(cand)  # reset atimes/mtimes the verification probes refreshed
    # Precondition pin: the apply run's lookup HITs the stored PASS (the
    # regression is only reachable through the cache-hit path).
    key2 = ced._ScratchVerdictCache._key(cand, ced._scratch_walk_stats(cand))
    assert key2 in json.loads(cache.read_text())
    res2 = _sweep(slurm_root, main_repo, apply=True, verdict_cache_path=cache)
    row = _row(res2, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-surviving-repo-missing" in row["reason"]
    assert cand.exists()
    assert (cand / "external" / "open-instruct" / "nested_only.py").is_file()


def test_r3c2_nested_stash_between_evidence_and_reap_keeps(
    slurm_root, main_repo, repo, monkeypatch
):
    """C2(b): nested-overlay state mutated BETWEEN evidence and reap (a stash
    created in the nested repo — its ref lives under the exempt ``.git`` dir,
    invisible to the non-exempt recency re-walk) trips the reap-time overlay
    RE-probe; pre-fix the reap re-ran only the OUTER class probes and rmtree'd
    the stash."""
    cand = slurm_root / "issue-9938"
    cand.mkdir()
    (cand / "tracked.py").write_text(COMMITTED_PY)
    nested = _staged_overlay_copy(cand, main_repo)
    _backdate(cand)
    real_hit = ced._scratch_live_process_hit

    def stash_then_pass(c):
        (nested / "nested_only.py").write_text(NESTED_ONLY_PY + "# about-to-stash edit\n")
        _git(nested, "stash")
        _backdate(cand)  # zero non-exempt mtime residue: isolate the overlay re-probe
        return real_hit(c)

    monkeypatch.setattr(ced, "_scratch_live_process_hit", stash_then_pass)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "overlay-stash" in row["reason"]
    assert cand.exists()
    assert _git(nested, "stash", "list").stdout.strip()  # the stash survives


# ─── round 3 C3: a symlinked declared-overlay path is never followed ──────────


def test_r3c3_symlinked_overlay_path_keeps(slurm_root, main_repo, repo):
    """C3: a declared overlay path that is a SYMLINK to a valid surviving
    clone must KEEP as overlay-not-a-repo — pre-fix the presence lstat
    ignored the entry's mode and ``_git_dir_kind`` followed the symlink into
    the target's ``.git``, accepting it as a nested clone."""
    surv = _surviving_overlay(main_repo)
    cand = slurm_root / "issue-9937"
    (cand / "external").mkdir(parents=True)
    (cand / "tracked.py").write_text(COMMITTED_PY)  # a verified outer file
    (cand / "external" / "open-instruct").symlink_to(surv)
    _backdate(cand)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-unverified-kept"
    assert "overlay-not-a-repo" in row["reason"]
    assert cand.exists()
    assert (surv / "nested_only.py").is_file()  # the symlink target is never touched


# ─── round 3 C4: the validated CANONICAL root is the enumerated root ──────────


def test_r3c4_root_swap_after_validation_is_inert(tmp_path, main_repo, monkeypatch):
    """C4: a symlink-target swap BETWEEN validation and enumeration must be
    inert — the sweep enumerates the validator's RETURNED canonical root, so
    the ``.git``-bearing swapped target (which validation would have refused)
    is never enumerated, never status-probed, never evidence-probed."""
    safe = tmp_path / "safe-root"
    safe.mkdir()
    danger = tmp_path / "danger-root"
    (danger / "issue-9936").mkdir(parents=True)
    (danger / "issue-9936" / "precious.py").write_text("never committed anywhere\n")
    _git(danger, "init", "-b", "main")  # a target validation would refuse (.git-bearing)
    link = tmp_path / "root-link"
    link.symlink_to(safe)
    real_validate = ced._assert_safe_slurm_src_root

    def swap_after_validate(root):
        canonical = real_validate(root)  # the REAL validator body runs
        link.unlink()
        link.symlink_to(danger)  # the TOCTOU: the raw root now points elsewhere
        return canonical

    monkeypatch.setattr(ced, "_assert_safe_slurm_src_root", swap_after_validate)
    probes: list[int] = []

    def spy_resolver(n: int):
        probes.append(n)
        raise AssertionError("status probe reached through a swapped, unvalidated root")

    res = ced.sweep_slurm_src(
        link,
        apply=True,
        main_repo=main_repo,
        status_resolver=spy_resolver,
        terminal_statuses=frozenset({"completed"}),
    )
    assert probes == []
    assert res.rows == []  # enumeration used the validated canonical (safe, empty) root
    assert (danger / "issue-9936" / "precious.py").is_file()


# ─── round 4 (R3-C1/SIB-1): record-form porcelain parse fails CLOSED ──────────


def test_r4_newline_worktree_path_returns_none(tmp_path, main_repo):
    """R4: a registered worktree whose PATH embeds a literal newline SPLITS
    its porcelain record (git 2.34.1 emits paths raw and cannot represent LF
    in a line-oriented format): the ``worktree`` line carries a TRUNCATED
    path and the remainder is an orphan continuation line. The parser must
    return None (whole listing AMBIGUOUS), never a partial set holding the
    truncated path while the REAL registered path is absent. Doubles as a
    git-upgrade probe: a future git that quotes/escapes such paths would
    parse to a non-None set and fail this test loudly."""
    wt = tmp_path / "nl\nline"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    assert wt.is_dir()
    assert ced._registered_worktree_paths(main_repo) is None


def test_r4_ambiguous_listing_keeps_reapable_candidate(
    slurm_root, main_repo, repo, tmp_path, monkeypatch
):
    """R4 end-to-end (apply=True): a fully reap-eligible staged copy is KEPT
    when the admin listing is ambiguous because SOME registered worktree's
    path embeds a newline — the positive non-registration proof cannot be
    established, so rmtree is never reached and the KEEP reason is surfaced.
    Real ambiguous listing (no monkeypatched probe); rmtree is boobytrapped
    to hard-fail the test if the fail-open path is ever reinstated."""
    cand = _staged_copy(slurm_root, "issue-9935")
    poison = tmp_path / "poison\nwt"
    _git(main_repo, "worktree", "add", "--detach", str(poison))
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "registration-probe-failed" in row["reason"]
    assert cand.exists() and (cand / "tracked.py").is_file()


def test_r4_gate17_branch_c_refuses_on_ambiguous_listing(tmp_path, main_repo):
    """R4 SIB-1: gate-1.7 evidence branch (c) consumes the same parser; under
    a REAL ambiguous listing (newline-bearing registered path) it must refuse
    the evidence branch (fail-toward-keep) instead of proceeding to the blob
    proof and licensing the generic rmtree."""
    d = tmp_path / "i9934_dl"
    d.mkdir()
    (d / "copy.py").write_text(COMMITTED_PY)
    _backdate(d)
    poison = tmp_path / "poison2\nwt"
    _git(main_repo, "worktree", "add", "--detach", str(poison))
    ev, det = ced._tmp_git_evidence_branch_c(d, main_repo=main_repo)
    assert ev is None
    assert det["reason"] == "registration-probe-failed"


def test_r4_trailing_space_path_roundtrips_exactly(tmp_path, main_repo):
    """R4: a registered path ending in a SPACE must round-trip byte-for-byte
    — the pre-fix ``.strip()`` recorded the stripped variant, so the REAL
    registered path compared unequal downstream (same fail-open consequence
    as the newline split, without any listing ambiguity)."""
    wt = tmp_path / "trail "
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    paths = ced._registered_worktree_paths(main_repo)
    assert paths is not None
    assert os.path.realpath(str(wt)) in paths
    assert os.path.realpath(str(tmp_path / "trail")) not in paths  # the stripped ghost


def test_r4_raw_special_char_paths_parse_and_reap_still_licensed(
    slurm_root, main_repo, repo, tmp_path
):
    """R4 negative control: git 2.34.1 emits space/tab/backslash/double-quote
    paths RAW on a single line (verified by live reproduction — NO C-quoting
    on this version), so a listing containing them still parses to the exact
    full set and a genuinely unregistered candidate's reap remains licensed —
    the fail-closed fix must not make every listing ambiguous (tier inert)."""
    specials = ["we ird", "ta\tb", "back\\slash", 'qu"ote']
    wts = []
    for name in specials:
        wt = tmp_path / name
        _git(main_repo, "worktree", "add", "--detach", str(wt))
        wts.append(wt)
    paths = ced._registered_worktree_paths(main_repo)
    assert paths is not None
    expected = {os.path.realpath(str(main_repo))} | {os.path.realpath(str(w)) for w in wts}
    assert paths == expected
    cand = _staged_copy(slurm_root, "issue-9933")
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert not cand.exists()
    assert res.bytes_freed > 0


# ─── round 5: flag-spoof continuation closed by the existence cross-check ─────


def test_r5_flag_spoof_continuation_returns_none(tmp_path, main_repo):
    """R5 (coordinator repro, real git, no adversary): a path embedding
    ``\\nbare`` splits into a record whose continuation line exactly spells
    the ``bare`` flag — the round-4 slot rules pass it (a genuine detached
    record simply lacks ``bare``) and the TRUNCATED path is recorded while
    the REAL registered path is absent. The existence cross-check must
    refuse: the truncated path does not exist on disk and the record is not
    ``prunable`` => the whole listing is AMBIGUOUS => None."""
    wt = tmp_path / "sp\nbare"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    assert wt.is_dir()
    assert not (tmp_path / "sp").exists()  # the spoof-truncated path
    assert ced._registered_worktree_paths(main_repo) is None


def test_r5_spoof_listing_keeps_reapable_candidate(
    slurm_root, main_repo, repo, tmp_path, monkeypatch
):
    """R5 end-to-end (apply=True): a fully reap-eligible staged copy is KEPT
    when the admin listing carries the flag-spoof record — same fail-open
    class as R3-C1, so the same KEEP reason must surface and the rmtree
    boobytrap must NOT fire."""
    cand = _staged_copy(slurm_root, "issue-9932")
    poison = tmp_path / "sp2\nbare"
    _git(main_repo, "worktree", "add", "--detach", str(poison))
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reap-reprobe-kept"
    assert "registration-probe-failed" in row["reason"]
    assert cand.exists() and (cand / "tracked.py").is_file()


def test_r5_prunable_missing_dir_parses_normally(tmp_path, main_repo):
    """R5 tolerance arm: a registered worktree whose DIRECTORY was deleted
    lists as ``prunable gitdir file points to non-existent location`` (git
    2.34.1 verified) — the one legitimate missing-dir shape. It must NOT
    force None, or a routine pruned worktree would wedge the whole tier
    into KEEP-everything."""
    gone = tmp_path / "gone-wt"
    keepd = tmp_path / "kept-wt"
    _git(main_repo, "worktree", "add", "--detach", str(gone))
    _git(main_repo, "worktree", "add", "--detach", str(keepd))
    shutil.rmtree(gone)
    paths = ced._registered_worktree_paths(main_repo)
    assert paths is not None
    assert os.path.realpath(str(main_repo)) in paths
    assert os.path.realpath(str(keepd)) in paths
    assert os.path.realpath(str(gone)) in paths  # pruned path still listed => KEEP-side


def test_r5_negative_control_locked_and_plain_listing_reap_still_licensed(
    slurm_root, main_repo, repo, tmp_path
):
    """R5 negative control: a normal listing — main + a plain detached
    worktree + a LOCKED worktree (bare ``locked`` flag line) — parses to the
    exact full set under the existence cross-check, and a genuinely
    unregistered candidate's reap remains licensed."""
    plain = tmp_path / "plain-wt"
    locked = tmp_path / "locked-wt"
    _git(main_repo, "worktree", "add", "--detach", str(plain))
    _git(main_repo, "worktree", "add", "--detach", str(locked))
    _git(main_repo, "worktree", "lock", str(locked))
    paths = ced._registered_worktree_paths(main_repo)
    assert paths is not None
    expected = {os.path.realpath(str(p)) for p in (main_repo, plain, locked)}
    assert paths == expected
    cand = _staged_copy(slurm_root, "issue-9931")
    res = _sweep(slurm_root, main_repo, apply=True)
    row = _row(res, cand.name)
    assert row["disposition"] == "slurm-src-reaped"
    assert not cand.exists()
    assert res.bytes_freed > 0


# ─── round 6: parse-free per-candidate probe + admin-side enumeration ─────────


def test_r6_intact_pointer_collision_reaps_via_worktree_remove(
    scratch_tmp_root, main_repo, repo, monkeypatch
):
    """R6 (coordinator repro, intact pointer): a registered worktree at a
    newline path whose TRUNCATION also exists as a real dir is treated as
    REGISTERED and reaps via ``git worktree remove`` — never rmtree. Pins
    the per-candidate kind dispatch (this arm was already safe pre-fix,
    stated plainly: the pre-fix failure lives in the deleted-pointer arm
    below)."""
    wt = scratch_tmp_root / "scratch-foo\nbare"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    decoy = scratch_tmp_root / "scratch-foo"
    decoy.mkdir()
    (decoy / "tracked.py").write_text(COMMITTED_PY)
    _backdate(scratch_tmp_root)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, wt.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert "worktree-removed" in row["reason"]
    assert not wt.exists()


def test_r6_deleted_pointer_collision_kept_never_rmtree(
    scratch_tmp_root, main_repo, repo, monkeypatch
):
    """R6 exploit arm (coordinator's truncation-collision, pointer DELETED —
    the R3-C1 downgrade): pre-fix the poisoned porcelain listing parsed
    "successfully" via the decoy, the registered newline worktree compared
    unequal, and rmtree DESTROYED it with the registration stranded. The
    admin-side per-record ``gitdir``-file enumeration must prove
    registration byte-exactly and KEEP (``git worktree remove`` refuses a
    pointer-deleted tree — verified rc=128 — so KEEP is the only safe
    disposition)."""
    wt = scratch_tmp_root / "scratch-foo\nbare"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()
    decoy = scratch_tmp_root / "scratch-foo"
    decoy.mkdir()
    (decoy / "tracked.py").write_text(COMMITTED_PY)
    _backdate(scratch_tmp_root)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, wt.name)
    assert row["disposition"] == "tmp-scratch-reap-reprobe-kept"
    assert "registered-path" in row["reason"]
    assert wt.exists() and (wt / "tracked.py").is_file()
    admin_set = ced._admin_registered_worktree_paths(main_repo)
    assert admin_set is not None and os.path.realpath(str(wt)) in admin_set  # registration survives


def test_r6_gate17_branch_c_deleted_pointer_collision_refuses(tmp_path, main_repo):
    """R6 exploit arm, branch (c): the deleted-pointer collision candidate
    was granted blob evidence pre-fix (licensing the generic rmtree); the
    admin-side enumeration must refuse it as registered."""
    cand = tmp_path / "i9929_x\nbare"
    _git(main_repo, "worktree", "add", "--detach", str(cand))
    (cand / ".git").unlink()
    (tmp_path / "i9929_x").mkdir()  # the decoy at the truncation
    _backdate(cand)
    ev, det = ced._tmp_git_evidence_branch_c(cand, main_repo=main_repo)
    assert ev is None
    assert det["reason"] == "registered-worktree"


def test_r6_foreign_worktree_kept_sweep_and_branch_c(scratch_tmp_root, tmp_path, main_repo, repo):
    """R6: a worktree registered to a DIFFERENT repo — with content that IS
    committed in OUR main repo, so a naive blob proof would license it —
    keeps on the sweep (foreign-worktree) and refuses at branch (c)."""
    other = tmp_path / "otherrepo"
    other.mkdir()
    _git(other, "init", "-b", "main")
    (other / "tracked.py").write_text(COMMITTED_PY)  # same bytes as main_repo's
    _git(other, "add", "-A")
    _git(other, "commit", "-m", "init")
    fwt = scratch_tmp_root / "scratch-foreign"
    _git(other, "worktree", "add", "--detach", str(fwt))
    _backdate(scratch_tmp_root)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, fwt.name)
    assert "foreign-worktree" in row["reason"]
    assert fwt.exists()
    bwt = tmp_path / "i9928_f"
    _git(other, "worktree", "add", "--detach", str(bwt))
    _backdate(bwt)
    ev, det = ced._tmp_git_evidence_branch_c(bwt, main_repo=main_repo)
    assert ev is None
    assert det["reason"] == "foreign-worktree"


def test_r6_submodule_gitfile_not_treated_as_worktree(tmp_path, main_repo):
    """R6: a submodule-style gitfile (``gitdir: …/modules/x``) is NOT a
    worktree registration — the probe classifies it ``submodule`` and
    branch (c) does not refuse it as OUR registered worktree (it falls to
    the evidence layer, whose internal class probes keep it fail-closed)."""
    mod = main_repo / ".git" / "modules" / "x"
    mod.mkdir(parents=True)
    d = tmp_path / "i9927_sub"
    d.mkdir()
    (d / "copy.py").write_text(COMMITTED_PY)
    (d / ".git").write_text(f"gitdir: {mod}\n")
    _backdate(d)
    assert ced._candidate_worktree_registration(d, main_repo) == ("submodule", None)
    ev, det = ced._tmp_git_evidence_branch_c(d, main_repo=main_repo)
    assert ev is None
    assert det["reason"] == "foreign-worktree"  # evidence-internal class probe, fail-closed
    assert det["reason"] != "registered-worktree"


def test_r6_relative_gitdir_resolves(scratch_tmp_root, main_repo, repo):
    """R6: a RELATIVE ``gitdir:`` pointer (git accepts these) resolves
    against the candidate dir — the probe classifies it ``ours`` and the
    reap still routes through ``git worktree remove``."""
    wt = scratch_tmp_root / "scratch-rel"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    admin_abs = Path((wt / ".git").read_text(encoding="utf-8")[len("gitdir:") :].strip())
    rel = os.path.relpath(admin_abs, wt)
    (wt / ".git").write_text(f"gitdir: {rel}\n")
    _backdate(scratch_tmp_root)
    reg, admin = ced._candidate_worktree_registration(wt, main_repo)
    assert reg == "ours"
    assert admin is not None and admin.is_dir()
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, wt.name)
    assert row["disposition"] == "tmp-scratch-reaped"
    assert "worktree-removed" in row["reason"]
    assert not wt.exists()


def test_r6_admin_enumeration_byte_exact_and_fail_closed(tmp_path, main_repo, monkeypatch):
    """R6 unit: the admin-side enumeration recovers a newline-bearing
    registered path BYTE-EXACTLY from the per-record ``gitdir`` file — even
    with the pointer deleted — never the truncation; includes the main
    toplevel; and fails closed (None) when the rev-parse probe fails."""
    wt = tmp_path / "adm\nwt"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()
    decoy = tmp_path / "adm"
    decoy.mkdir()
    s = ced._admin_registered_worktree_paths(main_repo)
    assert s is not None
    assert os.path.realpath(str(wt)) in s
    assert os.path.realpath(str(decoy)) not in s
    assert os.path.realpath(str(main_repo)) in s
    # Round 8: the enumeration's rev-parse seam is the binary-mode sibling
    # (``_git_bytes``) — stub the seam the function actually uses.
    monkeypatch.setattr(ced, "_git_bytes", lambda *a, **k: None)
    assert ced._admin_registered_worktree_paths(main_repo) is None


# ─── round 7 (Codex R4-1): binary gitdir read — CR/CRLF are path bytes ────────


def test_r7_cr_path_admin_enumeration_no_ghost(tmp_path, main_repo):
    """R7 unit (Codex R4-1): the admin ``gitdir`` file for a worktree at a
    CR-bearing path holds the raw bytes ``…/cr\\rX/.git\\n``; a text-mode
    read translates the CR to LF, injecting a GHOST path (``…/cr\\nX``) into
    the AUTHORITATIVE set while the REAL registration goes missing —
    fail-open in the licensing layer. The binary read must recover the real
    path byte-exactly and never the ghost."""
    wt = tmp_path / "cr\rX"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()  # hardest case: only the admin side knows
    (tmp_path / "cr\nX").mkdir()  # decoy at the NORMALIZED (ghost) name
    s = ced._admin_registered_worktree_paths(main_repo)
    assert s is not None
    assert os.path.realpath(str(wt)) in s  # the real registration, byte-exact
    assert os.path.realpath(str(tmp_path / "cr\nX")) not in s  # never the ghost


def test_r7_crlf_path_admin_enumeration_no_ghost(tmp_path, main_repo):
    """R7 CRLF variant: a literal ``\\r\\n`` inside the registered PATH is
    two path bytes; universal newlines would collapse it to one LF (ghost).
    The binary read preserves it byte-exactly."""
    wt = tmp_path / "a\r\nb"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()
    s = ced._admin_registered_worktree_paths(main_repo)
    assert s is not None
    assert os.path.realpath(str(wt)) in s
    assert os.path.realpath(str(tmp_path / "a\nb")) not in s  # the collapsed ghost


def test_r7_cr_flag_spoof_collision_kept_never_rmtree(
    scratch_tmp_root, main_repo, repo, monkeypatch
):
    """R7 e2e — the full licensing chain: a registered worktree at
    ``scratch-cr\\rbare`` with its pointer DELETED plus a decoy at the
    truncation. Pre-fix BOTH layers failed together: the text-mode admin
    read produced the LF-ghost (real path absent from the authoritative
    set), and ``_git text=True`` translated the porcelain's CR to LF,
    yielding the ``bare`` flag-spoof record whose truncated path exists
    (the decoy) — so the porcelain ALSO parsed "successfully" without the
    real path, and rmtree DESTROYED the registered worktree. Post-fix the
    binary admin read proves registration byte-exactly and KEEPs."""
    wt = scratch_tmp_root / "scratch-cr\rbare"
    _git(main_repo, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()
    decoy = scratch_tmp_root / "scratch-cr"
    decoy.mkdir()
    (decoy / "tracked.py").write_text(COMMITTED_PY)
    _backdate(scratch_tmp_root)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = _tmp_sweep(scratch_tmp_root, main_repo, apply=True)
    row = _tmp_row(res, wt.name)
    assert row["disposition"] == "tmp-scratch-reap-reprobe-kept"
    assert "registered-path" in row["reason"]
    assert wt.exists() and (wt / "tracked.py").is_file()
    admin_set = ced._admin_registered_worktree_paths(main_repo)
    assert admin_set is not None and os.path.realpath(str(wt)) in admin_set


# ─── round 8: binary rev-parse — the repo's OWN path bytes + scan-root ambiguity ──
#
# Round-5 cap residual (epm:failure, reproduced live on git 2.34.1):
# ``_admin_registered_worktree_paths`` derived its OWN scan root from
# text-mode ``_git`` stdout, so a CR (or edge whitespace) in the
# REPOSITORY'S OWN path yielded an unresolvable root whose
# ``FileNotFoundError`` was swallowed into ``entries = []`` — a
# SUCCESSFUL-LOOKING INCOMPLETE authoritative set, indistinguishable from
# "no linked worktrees", in a code path that licenses deletion.


def _init_repo(path: Path) -> Path:
    """``git init`` + one commit at ``path`` (mkdir'd here) — for round-8
    tests whose defect lives in the REPOSITORY'S OWN path bytes, which the
    fixed-name ``main_repo`` fixture cannot carry."""
    path.mkdir()
    _git(path, "init", "-b", "main")
    (path / "tracked.py").write_text(COMMITTED_PY)
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "init")
    return path


def test_r8_cr_repo_own_path_admin_enumeration_byte_exact(tmp_path):
    """R8: the repository's OWN path carries a CR; pre-fix the text-mode
    rev-parse pipe translated it to LF, the derived ``worktrees/`` scan
    root did not exist, and the swallowed ``FileNotFoundError`` returned a
    successful-looking set holding ONLY the LF-ghost toplevel — the
    registered worktree silently vanished from the AUTHORITATIVE set.
    Post-fix the binary read recovers every path byte-exactly."""
    main = _init_repo(tmp_path / "re\rpo")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "--detach", str(wt))
    s = ced._admin_registered_worktree_paths(main)
    assert s is not None
    assert os.path.realpath(str(wt)) in s  # the real registration
    assert os.path.realpath(str(main)) in s  # the real toplevel, byte-exact
    assert os.path.realpath(str(tmp_path / "re\npo")) not in s  # never the LF-ghost


def test_r8_unresolvable_scan_root_returns_none_not_empty(tmp_path, main_repo, monkeypatch):
    """R8 guard branch: a git-common-dir answer naming a NONEXISTENT
    directory with rc=0 — the shape a mangled pipe produced pre-fix, and
    the shape an external mutation between the two calls still can — is
    AMBIGUITY: ``None``, never a populated-looking or empty set. Real git
    validates the common dir at startup (probed on git 2.34.1: a doctored
    worktree ``commondir`` file dies rc=128), so the rc=0-with-ghost-path
    answer is constructible only by doctoring the ONE rev-parse seam; the
    doctored ``CompletedProcess`` is signature-real, every other call runs
    real git, and the real ``_git_bytes`` body is executed unmocked by the
    sibling r8 tests."""
    ghost = tmp_path / "no-such-common-dir"
    common_args = ["rev-parse", "--path-format=absolute", "--git-common-dir"]

    if hasattr(ced, "_git_bytes"):  # post-fix seam
        real_bytes = ced._git_bytes

        def doctored_bytes(args, *, cwd, **kw):
            if args == common_args:
                return subprocess.CompletedProcess(
                    args=["git", *args], returncode=0, stdout=f"{ghost}\n".encode(), stderr=b""
                )
            return real_bytes(args, cwd=cwd, **kw)

        monkeypatch.setattr(ced, "_git_bytes", doctored_bytes)
    real_text = ced._git  # pre-fix seam: same doctored answer, text-mode

    def doctored_text(args, *, cwd, **kw):
        if args == common_args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=0, stdout=f"{ghost}\n", stderr=""
            )
        return real_text(args, cwd=cwd, **kw)

    monkeypatch.setattr(ced, "_git", doctored_text)
    assert ced._admin_registered_worktree_paths(main_repo) is None


def test_r8_no_linked_worktrees_still_main_only_set(main_repo):
    """R8 anti-overcorrection pin: a repository with NO linked worktrees
    has a PRESENT common dir and NO ``worktrees/`` subdirectory — the one
    legitimate empty shape. It must still return the main-toplevel-only
    set, NOT ``None`` (the regression a naive "any FileNotFoundError ⇒
    None" fix would introduce, degrading every registration consumer into
    permanent keeps)."""
    assert not (main_repo / ".git" / "worktrees").exists()  # the tested shape
    s = ced._admin_registered_worktree_paths(main_repo)
    assert s == frozenset({os.path.realpath(str(main_repo))})


def test_r8_edge_whitespace_repo_path_survives_roundtrip(tmp_path):
    """R8 (.strip() arm): trailing whitespace on the repository's own FINAL
    path component sits at the string EDGE of the ``--show-toplevel``
    output — pre-fix ``.strip()`` ate it, replacing the real toplevel with
    a ghost in the authoritative set. Leading whitespace is always
    interior to an absolute path (it starts with ``/``), so the trailing
    edge is the whole exposed surface; the edge-whitespace WORKTREE path
    rides the (already byte-exact) gitdir files and pins the round-trip
    end to end."""
    main = _init_repo(tmp_path / "repo ")  # trailing space — at the stdout edge
    wt = tmp_path / " wt "
    _git(main, "worktree", "add", "--detach", str(wt))
    s = ced._admin_registered_worktree_paths(main)
    assert s is not None
    assert os.path.realpath(str(main)) in s  # trailing space survives
    assert os.path.realpath(str(wt)) in s
    assert os.path.realpath(str(tmp_path / "repo")) not in s  # never the stripped ghost


def test_r8_cr_repo_own_path_candidate_probe_classifies_ours(tmp_path):
    """R8 audit sibling (``_worktree_admin_of_main``): the same text-mode
    ``.strip()`` normalization derived the ``worktrees/`` root inside the
    layer-1 gitfile probe, so a CR in the repository's OWN path
    misclassified every genuinely-ours admin dir as ``foreign`` — a
    KEEP-direction failure, but a WRONG answer from the AUTHORITATIVE
    layer-1 probe. Byte-exact reads classify it ``ours``."""
    main = _init_repo(tmp_path / "cr\rrepo")
    wt = tmp_path / "wtx"
    _git(main, "worktree", "add", "--detach", str(wt))
    reg, admin = ced._candidate_worktree_registration(wt, main)
    assert reg == "ours"
    assert admin is not None and admin.is_dir()


def test_r8_cr_main_repo_path_full_chain_kept_never_rmtree(tmp_path, repo, monkeypatch):
    """R8 e2e — the full licensing chain with the REPOSITORY'S OWN path
    carrying a CR flag-spoof (``mr\\rbare`` + decoy at the truncation) and
    a registered worktree at a newline flag-spoof path with its pointer
    DELETED (+ decoy). Pre-fix EVERY layer failed together: layer 1 is
    structurally blind (pointer gone), the admin enumeration scanned a
    ghost root (the worktree absent from its successful-looking set), and
    the porcelain listing parsed "successfully" — both mangled records
    truncate onto EXISTING decoys with ``bare`` absorbed as a flag — so
    ``shutil.rmtree`` DESTROYED the registered worktree. Post-fix the
    byte-exact admin enumeration proves registration and KEEPs."""
    main = _init_repo(tmp_path / "mr\rbare")
    (tmp_path / "mr").mkdir()  # decoy at the CR truncation of the MAIN repo path
    scratch = tmp_path / "faketmp"
    scratch.mkdir()
    wt = scratch / "scratch-x\nbare"
    _git(main, "worktree", "add", "--detach", str(wt))
    (wt / ".git").unlink()  # layer 1 structurally blind
    decoy = scratch / "scratch-x"
    decoy.mkdir()
    (decoy / "tracked.py").write_text(COMMITTED_PY)
    _backdate(scratch)
    monkeypatch.setattr(ced.shutil, "rmtree", _rmtree_boom)
    res = ced.sweep_tmp_scratch(scratch, apply=True, main_repo=main)
    row = _tmp_row(res, wt.name)
    assert row["disposition"] == "tmp-scratch-reap-reprobe-kept"
    assert "registered-path" in row["reason"]
    assert wt.exists() and (wt / "tracked.py").is_file()
    s = ced._admin_registered_worktree_paths(main)
    assert s is not None and os.path.realpath(str(wt)) in s  # registration survives
