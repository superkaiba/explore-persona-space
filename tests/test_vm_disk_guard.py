"""Offline unit tests for the VM root-disk guard (scripts/vm_disk_guard.py)
and the download-cache cleanup helper (scripts/clean_experiment_downloads.py).

Covers — against a TEMP filesystem + mocked df, no real disk / task state:
  * the usage threshold trigger (over vs under),
  * terminal-status gating (an ACTIVE issue's caches are never deleted; a
    completed/archived/awaiting_promotion issue's caches ARE),
  * dry-run vs --apply (dry-run removes nothing on disk; apply does),
  * store/ + eval_results/ are NEVER touched,
  * stale-log age gating, and the still-over-after WARNING path.

Both scripts live under scripts/ (not an importable package), so they are
loaded via importlib the same way tests/test_worktree_audit.py loads its
target. clean_experiment_downloads is registered FIRST because vm_disk_guard
imports it by module name at load time.
"""

import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


# clean_experiment_downloads must be importable by name before vm_disk_guard
# executes its top-level `from clean_experiment_downloads import ...`.
ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures ────────────────────────────────────────────────────────────────


def _make_issue_data(data_root: Path, issue_n: int, *, prefix: str = "issue_") -> Path:
    """Build a realistic data/issue_<N>/ tree: hf_dl + g1_dl + g2_dl caches
    (each with a file), a store/ dir (KEEP), and a stray top-level json (KEEP).
    Returns the issue dir."""
    issue_dir = data_root / f"{prefix}{issue_n}"
    for cache in ("hf_dl", "g1_dl", "g2_dl"):
        d = issue_dir / cache
        d.mkdir(parents=True)
        (d / "blob.bin").write_bytes(b"x" * 1024)
    store = issue_dir / "store"
    store.mkdir(parents=True)
    (store / "generated.json").write_text('{"kept": true}')
    (issue_dir / "R_test.json").write_text('{"kept": true}')
    return issue_dir


def _setup_fake_repo(tmp_path: Path, monkeypatch) -> Path:
    """Point ced.repo_root() at tmp_path so _worktree_data_roots resolves the
    worktree tree under it. Returns the fake repo root. Both the repo-root
    data/ and the worktree data/ then live under one temp filesystem and the
    production (data_root=None) code path can be exercised offline."""
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    # Determinism pin (#924): force the on-main resolution path so a
    # hypothetical fresh-clone-on-a-branch test runner cannot flip the probe.
    monkeypatch.setattr(ced, "_off_main_checkout_root", lambda: None)
    # Sandbox the #773 active-consumer gate: REAL active tasks reference
    # data/issue_658/, so an un-sandboxed gate walks the LIVE tasks/ tree and
    # (correctly) keeps these synthetic caches — same trap the fake_repo
    # fixtures in the sibling test files exist to prevent.
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    return tmp_path


def _make_worktree_issue_data(
    repo: Path, issue_n: int, *, suffix: str = "", prefix: str = "issue_"
) -> Path:
    """Build .claude/worktrees/issue-<N>[<suffix>]/data/issue_<N>/ with the
    same cache/store layout as _make_issue_data. Returns the issue data dir."""
    wt_name = f"issue-{issue_n}{suffix}"
    wt_data = repo / ".claude" / "worktrees" / wt_name / "data"
    return _make_issue_data(wt_data, issue_n, prefix=prefix)


# ─── clean_experiment_downloads: cache discovery + keep contract ─────────────


def test_download_cache_dirs_finds_hf_and_group_caches(tmp_path):
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 658)
    caches = ced.download_cache_dirs(658, data_root=data_root)
    names = sorted(c.name for c in caches)
    assert names == ["g1_dl", "g2_dl", "hf_dl"]


def test_download_cache_dirs_never_includes_store(tmp_path):
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 658)
    caches = ced.download_cache_dirs(658, data_root=data_root)
    assert all(c.name != "store" for c in caches)


def test_issue_n_boundary_is_exact(tmp_path):
    # issue_65 must NOT pick up issue_658's caches.
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 65)
    _make_issue_data(data_root, 658)
    caches = ced.download_cache_dirs(65, data_root=data_root)
    assert all("658" not in str(c) for c in caches)
    assert len(caches) == 3  # only issue_65's three caches


def test_both_naming_conventions_matched(tmp_path):
    # data/ uses both issue_<N> and issue<N>[_slug]; both must be found.
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 333, prefix="issue")  # issue333
    (data_root / "issue333_marker" / "hf_dl").mkdir(parents=True)  # issue333_<slug>
    caches = ced.download_cache_dirs(333, data_root=data_root)
    # issue333's three caches + issue333_marker's one hf_dl
    assert len(caches) == 4


def test_clean_issue_downloads_dry_run_removes_nothing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 658)
    # Sandbox the #773 active-consumer gate: REAL active tasks reference
    # data/issue_658/, so the un-sandboxed gate walks the LIVE tasks/ tree and
    # (correctly) keeps these synthetic caches — same trap the fake_repo
    # fixtures in the sibling test files exist to prevent.
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    res = ced.clean_issue_downloads(658, apply=False, data_root=data_root)
    assert len(res.removed) == 3  # would-remove the 3 caches
    # Nothing actually deleted.
    assert (issue_dir / "hf_dl").is_dir()
    assert (issue_dir / "g1_dl").is_dir()
    assert res.bytes_freed > 0  # reported a size


def test_clean_issue_downloads_apply_deletes_caches_keeps_store(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 658)
    # Sandbox the #773 gate (live tasks reference data/issue_658/ — see above).
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    res = ced.clean_issue_downloads(658, apply=True, data_root=data_root)
    assert len(res.removed) == 3
    assert not (issue_dir / "hf_dl").exists()
    assert not (issue_dir / "g1_dl").exists()
    assert not (issue_dir / "g2_dl").exists()
    # store/ + the stray json are KEPT.
    assert (issue_dir / "store" / "generated.json").is_file()
    assert (issue_dir / "R_test.json").is_file()


def test_clean_issue_downloads_is_idempotent(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 658)
    # Sandbox the #773 gate (live tasks reference data/issue_658/ — see above);
    # without it the first reap is silently SKIPPED and idempotency is untested.
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    ced.clean_issue_downloads(658, apply=True, data_root=data_root)
    # Second run: nothing left to remove, no error.
    res2 = ced.clean_issue_downloads(658, apply=True, data_root=data_root)
    assert res2.removed == []
    assert res2.failed == []


def test_clean_issue_downloads_missing_issue_is_noop(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    res = ced.clean_issue_downloads(999, apply=True, data_root=data_root)
    assert res.removed == []
    assert res.failed == []


# ─── clean_experiment_downloads: incremental (between-phase) cleanup ──────────


def test_incremental_apply_reaps_consumed_cache_keeps_store(tmp_path, monkeypatch):
    # Between-phase cleanup: a consumed phase's hf_dl/g*_dl is reaped, store/
    # + eval-result-shaped json kept — identical contract to the end-of-run path.
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 658)
    # Sandbox the #773 gate (live tasks reference data/issue_658/ — see above).
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    res = ced.clean_issue_downloads_incremental(658, apply=True, data_root=data_root)
    assert len(res.removed) == 3
    assert not (issue_dir / "hf_dl").exists()
    assert not (issue_dir / "g1_dl").exists()
    assert not (issue_dir / "g2_dl").exists()
    # store/ + the stray json (durable, not re-downloadable) are KEPT.
    assert (issue_dir / "store" / "generated.json").is_file()
    assert (issue_dir / "R_test.json").is_file()


def test_incremental_dry_run_removes_nothing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 658)
    # Sandbox the #773 gate (live tasks reference data/issue_658/ — see above).
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    res = ced.clean_issue_downloads_incremental(658, apply=False, data_root=data_root)
    assert len(res.removed) == 3  # would-remove the 3 caches
    assert (issue_dir / "hf_dl").is_dir()  # nothing actually deleted
    assert res.bytes_freed > 0


def test_incremental_is_idempotent(tmp_path, monkeypatch):
    # A re-run after the phase's cache is already reaped is a no-op (a later
    # phase that re-downloads rebuilds the cache; absent that, nothing to do).
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 658)
    # Sandbox the #773 gate (live tasks reference data/issue_658/ — see above).
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})
    ced.clean_issue_downloads_incremental(658, apply=True, data_root=data_root)
    res2 = ced.clean_issue_downloads_incremental(658, apply=True, data_root=data_root)
    assert res2.removed == []
    assert res2.failed == []


def test_incremental_has_no_terminal_status_gate_on_active_issue(tmp_path, monkeypatch):
    # The KEY distinction from the vm_disk_guard tier-(b) backstop: incremental
    # cleanup deliberately works on an ACTIVE issue (the run is its own
    # authority that the phase is done). The vm_disk_guard tier-(b) helper would
    # KEEP this same active issue's cache; the incremental path reaps it.
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(vdg, "repo_root", lambda: repo)
    repo_issue = _make_issue_data(repo / "data", 658)

    # Sanity: the guard's tier-(b) KEEPS an active (running) issue's cache.
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    guard_res = vdg.clean_terminal_download_caches(apply=True)
    assert guard_res.bytes_freed == 0
    assert (repo_issue / "hf_dl").is_dir()  # guard left it alone

    # Incremental cleanup reaps the SAME active issue's consumed cache — it
    # never consults task status.
    inc_res = ced.clean_issue_downloads_incremental(658, apply=True)  # data_root=None
    assert inc_res.bytes_freed > 0
    assert not (repo_issue / "hf_dl").exists()
    assert (repo_issue / "store" / "generated.json").is_file()  # store still kept


def test_incremental_reaps_worktree_consumed_cache(tmp_path, monkeypatch):
    # Multi-phase runs write into the worktree (#658 evidence); the incremental
    # path sweeps repo-root + worktree copies identically to the end-of-run path.
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    wt_issue = _make_worktree_issue_data(repo, 658)
    res = ced.clean_issue_downloads_incremental(658, apply=True)  # data_root=None
    assert res.bytes_freed > 0
    assert not (wt_issue / "hf_dl").exists()
    assert not (wt_issue / "g1_dl").exists()
    assert (wt_issue / "store" / "generated.json").is_file()  # worktree store kept
    assert (wt_issue / "R_test.json").is_file()


def test_incremental_is_alias_of_clean_issue_downloads(tmp_path):
    # The incremental wrapper must produce the exact same removal set as the
    # base helper (it is a thin intent-labeling alias, same safety contract).
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 720)
    base = ced.clean_issue_downloads(720, apply=False, data_root=data_root)
    inc = ced.clean_issue_downloads_incremental(720, apply=False, data_root=data_root)
    assert sorted(base.removed) == sorted(inc.removed)
    assert base.bytes_freed == inc.bytes_freed


# ─── vm_disk_guard: threshold logic ──────────────────────────────────────────


def test_over_threshold_strict():
    assert vdg.over_threshold(85.1, 85.0) is True
    assert vdg.over_threshold(85.0, 85.0) is False  # strictly above
    assert vdg.over_threshold(50.0, 85.0) is False


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("EPS_VM_DISK_THRESHOLD", "70")
    assert vdg.threshold_pct() == 70.0


def test_threshold_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("EPS_VM_DISK_THRESHOLD", "not-a-number")
    assert vdg.threshold_pct() == vdg.DEFAULT_THRESHOLD_PCT
    monkeypatch.setenv("EPS_VM_DISK_THRESHOLD", "150")  # out of range
    assert vdg.threshold_pct() == vdg.DEFAULT_THRESHOLD_PCT


# ─── vm_disk_guard: terminal-status gating (tier b) ──────────────────────────


def test_tier_b_skips_active_issue(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 700)
    # Pretend issue 700 is RUNNING (active) — caches must be kept.
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root)
    assert res.bytes_freed == 0
    assert (issue_dir / "hf_dl").is_dir()  # NOT deleted


def test_tier_b_deletes_terminal_issue(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 701)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root)
    assert res.bytes_freed > 0
    assert not (issue_dir / "hf_dl").exists()
    assert (issue_dir / "store" / "generated.json").is_file()  # store kept


@pytest.mark.parametrize("status", ["completed", "archived", "awaiting_promotion"])
def test_tier_b_terminal_statuses_all_reap(tmp_path, monkeypatch, status):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 702)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: status)
    vdg.clean_terminal_download_caches(apply=True, data_root=data_root)
    assert not (issue_dir / "hf_dl").exists()


@pytest.mark.parametrize("status", ["running", "blocked", "interpreting", "proposed", None])
def test_tier_b_non_terminal_statuses_all_kept(tmp_path, monkeypatch, status):
    # blocked is deliberately KEPT (may resume + need its cache); unresolved
    # (None) is kept too (fail toward keep).
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 703)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: status)
    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root)
    assert res.bytes_freed == 0
    assert (issue_dir / "hf_dl").is_dir()


def test_tier_b_dry_run_keeps_terminal_caches_on_disk(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    issue_dir = _make_issue_data(data_root, 704)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    res = vdg.clean_terminal_download_caches(apply=False, data_root=data_root)
    assert res.bytes_freed > 0  # would-free reported
    assert (issue_dir / "hf_dl").is_dir()  # but NOT actually removed


# ─── vm_disk_guard: tier (c) stale logs ──────────────────────────────────────


def test_tier_c_deletes_only_old_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "old.log"
    new = logs / "new.log"
    other = logs / "keep.txt"  # not a .log
    for f in (old, new, other):
        f.write_text("x")
    now = 1_000_000_000.0
    import os

    os.utime(old, (now - 30 * 86400, now - 30 * 86400))  # 30d old
    os.utime(new, (now - 1 * 86400, now - 1 * 86400))  # 1d old
    os.utime(other, (now - 30 * 86400, now - 30 * 86400))
    res = vdg.clean_stale_logs(apply=True, max_age_days=14.0, now=now, extra_roots=[logs])
    assert not old.exists()  # 30d > 14d -> removed
    assert new.exists()  # 1d < 14d -> kept
    assert other.exists()  # not a .log -> kept
    assert res.bytes_freed > 0


def test_tier_c_dry_run_keeps_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "old.log"
    old.write_text("x")
    now = 1_000_000_000.0
    import os

    os.utime(old, (now - 30 * 86400, now - 30 * 86400))
    res = vdg.clean_stale_logs(apply=False, max_age_days=14.0, now=now, extra_roots=[logs])
    assert old.exists()
    assert res.bytes_freed > 0  # would-free reported


# ─── vm_disk_guard: run_guard orchestration ──────────────────────────────────


def _patch_disk(monkeypatch, before_pct, after_pct, free_gb=50.0):
    """Make disk_used_pct return before_pct on the 1st call and after_pct on
    subsequent calls (run_guard reads twice: pre + post cleanup)."""
    state = {"calls": 0}

    def fake_used(path="/"):
        state["calls"] += 1
        return before_pct if state["calls"] == 1 else after_pct

    monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": free_gb)


def test_run_guard_under_threshold_runs_no_tiers(tmp_path, monkeypatch):
    _patch_disk(monkeypatch, before_pct=50.0, after_pct=50.0)
    res = vdg.run_guard(apply=True, threshold=85.0, data_root=tmp_path / "data")
    assert res.triggered is False
    assert res.tiers == []
    assert res.still_over_after is False


def test_run_guard_over_threshold_runs_tiers(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    _make_issue_data(data_root, 800)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    monkeypatch.setattr(vdg, "clean_uv_cache", lambda apply: vdg.TierResult(name="uv-cache"))
    monkeypatch.setattr(vdg, "clean_stale_logs", lambda *a, **k: vdg.TierResult(name="stale-logs"))
    _patch_disk(monkeypatch, before_pct=90.0, after_pct=40.0)
    res = vdg.run_guard(apply=True, threshold=85.0, data_root=data_root)
    assert res.triggered is True
    assert {t.name for t in res.tiers} == {"uv-cache", "terminal-download-caches", "stale-logs"}
    # terminal-download-caches tier actually freed the issue-800 caches.
    assert not (data_root / "issue_800" / "hf_dl").exists()
    assert res.still_over_after is False  # 40% < 85%


def test_run_guard_still_over_after_sets_warning_flag(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(vdg, "clean_uv_cache", lambda apply: vdg.TierResult(name="uv-cache"))
    monkeypatch.setattr(vdg, "clean_stale_logs", lambda *a, **k: vdg.TierResult(name="stale-logs"))
    monkeypatch.setattr(
        vdg,
        "clean_terminal_download_caches",
        lambda *a, **k: vdg.TierResult(name="terminal-download-caches"),
    )
    _patch_disk(monkeypatch, before_pct=95.0, after_pct=92.0)
    res = vdg.run_guard(apply=True, threshold=85.0, data_root=data_root)
    assert res.triggered is True
    assert res.still_over_after is True  # 92% still > 85%


def test_terminal_reap_statuses_exclude_blocked():
    # Regression guard for the deliberate divergence from
    # task_workflow.TERMINAL_STATUSES (which includes `blocked`).
    assert "blocked" not in vdg.TERMINAL_CACHE_REAP_STATUSES
    assert "awaiting_promotion" in vdg.TERMINAL_CACHE_REAP_STATUSES
    assert {"completed", "archived", "awaiting_promotion"} == vdg.TERMINAL_CACHE_REAP_STATUSES


# ─── worktree-copy coverage (#658 evidence: live data lives in the worktree) ──


def test_download_cache_dirs_finds_worktree_copies(tmp_path, monkeypatch):
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    _make_issue_data(repo / "data", 658)  # repo-root copy
    _make_worktree_issue_data(repo, 658)  # .claude/worktrees/issue-658/data/issue_658/
    caches = ced.download_cache_dirs(658)  # data_root=None -> repo + worktree
    worktree_caches = [c for c in caches if ".claude/worktrees" in str(c)]
    assert len(worktree_caches) == 3  # hf_dl + g1_dl + g2_dl in the worktree
    # 3 repo-root + 3 worktree = 6 total
    assert len(caches) == 6


def test_download_cache_dirs_covers_suffixed_followup_worktree(tmp_path, monkeypatch):
    # issue-<N>-<suffix> (same-issue follow-up round) worktrees map to N.
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    _make_worktree_issue_data(repo, 658, suffix="-onpolicy-v2")
    caches = ced.download_cache_dirs(658)
    assert len(caches) == 3
    assert all("issue-658-onpolicy-v2" in str(c) for c in caches)


def test_worktree_boundary_is_exact(tmp_path, monkeypatch):
    # issue-65 worktree must not pick up issue-658's worktree data.
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    _make_worktree_issue_data(repo, 65)
    _make_worktree_issue_data(repo, 658)
    caches = ced.download_cache_dirs(65)
    assert all("issue-658" not in str(c) for c in caches)
    assert len(caches) == 3


def test_apply_deletes_worktree_caches_keeps_worktree_store(tmp_path, monkeypatch):
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    wt_issue = _make_worktree_issue_data(repo, 658)
    ced.clean_issue_downloads(658, apply=True)  # data_root=None
    assert not (wt_issue / "hf_dl").exists()
    assert not (wt_issue / "g1_dl").exists()
    # store/ + the stray json are KEPT in the worktree too.
    assert (wt_issue / "store" / "generated.json").is_file()
    assert (wt_issue / "R_test.json").is_file()


def test_tier_b_protects_active_issue_worktree_data(tmp_path, monkeypatch):
    # The #658-mid-analysis case: an ACTIVE issue actively writing into its
    # worktree must have BOTH its repo-root AND worktree caches protected.
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(vdg, "repo_root", lambda: repo)
    repo_issue = _make_issue_data(repo / "data", 658)
    wt_issue = _make_worktree_issue_data(repo, 658)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "followups_running")
    res = vdg.clean_terminal_download_caches(apply=True)  # data_root=None -> all roots
    assert res.bytes_freed == 0
    assert (repo_issue / "hf_dl").is_dir()  # repo-root cache kept
    assert (wt_issue / "hf_dl").is_dir()  # worktree cache kept (the key guard)


def test_tier_b_reaps_terminal_issue_worktree_data(tmp_path, monkeypatch):
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(vdg, "repo_root", lambda: repo)
    repo_issue = _make_issue_data(repo / "data", 659)
    wt_issue = _make_worktree_issue_data(repo, 659)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    res = vdg.clean_terminal_download_caches(apply=True)
    assert res.bytes_freed > 0
    assert not (repo_issue / "hf_dl").exists()  # repo-root cache reaped
    assert not (wt_issue / "hf_dl").exists()  # worktree cache reaped
    assert (wt_issue / "store" / "generated.json").is_file()  # worktree store kept


def test_tier_b_discovers_worktree_only_issue(tmp_path, monkeypatch):
    # An issue with ONLY a worktree data dir (no repo-root data/) is still
    # discovered + reaped when terminal.
    repo = _setup_fake_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(vdg, "repo_root", lambda: repo)
    (repo / "data").mkdir(parents=True)  # empty repo-root data/
    wt_issue = _make_worktree_issue_data(repo, 660)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "archived")
    res = vdg.clean_terminal_download_caches(apply=True)
    assert res.bytes_freed > 0
    assert not (wt_issue / "hf_dl").exists()


# ─── #1392: --ignore-threshold, --no-push, single-flight apply lock ──────────


def _benign_result(*, still_over_after: bool = False, apply: bool = False) -> "vdg.GuardResult":
    return vdg.GuardResult(
        used_pct_before=50.0,
        used_pct_after=50.0,
        free_gb_before=100.0,
        free_gb_after=100.0,
        threshold_pct=85.0,
        triggered=still_over_after,
        apply=apply,
        still_over_after=still_over_after,
    )


@pytest.fixture
def main_seams(tmp_path, monkeypatch):
    """Hermetic seams for main() tests: tmp lock path, telegram + sidecar
    recorders (a real push / real sidecar write must never leave pytest)."""
    monkeypatch.setattr(vdg, "_APPLY_LOCK_PATH", tmp_path / "vm-disk-guard.lock")
    pushes: list[str] = []
    monkeypatch.setattr(vdg, "_telegram_push", lambda msg, apply: (pushes.append(msg), True)[1])
    events: list[dict] = []
    monkeypatch.setattr(
        vdg, "append_disk_guard_event", lambda event, *, apply=True: events.append(event)
    )
    return {"lock_path": tmp_path / "vm-disk-guard.lock", "pushes": pushes, "events": events}


def test_ignore_threshold_triggers_tiers_under_threshold(tmp_path, monkeypatch):
    """#1392: ignore_threshold=True forces triggered under the percent gate;
    still_over_after stays computed against the REAL threshold."""
    monkeypatch.setattr(vdg, "clean_uv_cache", lambda apply: vdg.TierResult(name="uv-cache"))
    monkeypatch.setattr(vdg, "clean_stale_logs", lambda *a, **k: vdg.TierResult(name="stale-logs"))
    _patch_disk(monkeypatch, before_pct=50.0, after_pct=50.0)
    res = vdg.run_guard(apply=False, threshold=99.0, data_root=tmp_path, ignore_threshold=True)
    assert res.triggered is True
    assert {t.name for t in res.tiers} == {"uv-cache", "terminal-download-caches", "stale-logs"}
    assert res.still_over_after is False  # 50% < 99% — real-threshold semantics survive


def test_ignore_threshold_default_keeps_percent_gate(tmp_path, monkeypatch):
    _patch_disk(monkeypatch, before_pct=50.0, after_pct=50.0)
    res = vdg.run_guard(apply=False, threshold=99.0, data_root=tmp_path)
    assert res.triggered is False
    assert res.tiers == []


def test_apply_lock_single_flight(main_seams, monkeypatch, capsys):
    """#1392: a second concurrent --apply exits 0 with a pid-named skip line
    (and a sidecar row) without running any tier."""
    lock_path = main_seams["lock_path"]
    holder = open(lock_path, "w")  # noqa: SIM115  (held across the main() call; closed in finally)
    holder.write("9999")
    holder.flush()
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        monkeypatch.setattr(
            vdg, "run_guard", lambda *a, **k: pytest.fail("run_guard must not run under lock skip")
        )
        rc = vdg.main(["--apply", "--no-data-disk", "--json"])
    finally:
        holder.close()
    assert rc == 0
    captured = capsys.readouterr()
    assert "single-flight skip" in captured.err
    assert "(pid 9999)" in captured.err
    assert json.loads(captured.out.strip()) == {"skipped": "apply-lock-held"}
    assert main_seams["events"] == [
        {"kind": "vm-disk-guard-apply-skip", "skipped": "apply-lock-held"}
    ]


def test_report_only_never_takes_lock(main_seams, monkeypatch):
    """A pre-held apply lock never blocks a report-only run (tests/smokes)."""
    lock_path = main_seams["lock_path"]
    holder = open(lock_path, "w")  # noqa: SIM115  (held across the main() call; closed in finally)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        calls: list[dict] = []
        monkeypatch.setattr(
            vdg, "run_guard", lambda *a, **k: (calls.append(k), _benign_result())[1]
        )
        rc = vdg.main(["--no-data-disk"])
    finally:
        holder.close()
    assert rc == 0
    assert len(calls) == 1  # the report path ran despite the held lock


def test_no_push_suppresses_still_over_push(main_seams, monkeypatch):
    """#1392: --no-push gates the two still-over pushes (exit 2 unchanged);
    without it the push fires."""
    monkeypatch.setattr(
        vdg,
        "run_guard",
        lambda *a, **k: _benign_result(still_over_after=True, apply=True),
    )
    rc = vdg.main(["--apply", "--no-push", "--no-data-disk"])
    assert rc == 2  # exit-2 semantics survive --no-push
    assert main_seams["pushes"] == []

    rc = vdg.main(["--apply", "--no-data-disk"])
    assert rc == 2
    assert len(main_seams["pushes"]) >= 1


def test_main_threads_ignore_threshold_to_run_guard(main_seams, monkeypatch):
    """#1392 flag-threading seam (Phase-2 Statistics Must-Fix b): the argv pin
    proves the flag is PASSED; this proves main() actually HONORS it through
    run_guard — without it the flag could silently no-op today only because
    the current 945 GB geometry keeps 60 GiB free above the 85% gate."""
    calls: list[dict] = []
    monkeypatch.setattr(vdg, "run_guard", lambda *a, **k: (calls.append(k), _benign_result())[1])
    rc = vdg.main(["--apply", "--ignore-threshold", "--no-push", "--no-data-disk"])
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["ignore_threshold"] is True
