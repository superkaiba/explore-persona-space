"""Tests for the pod-side disk guard helper (scripts/pod_disk_guard.py).

Covers: intermediate-vs-final classification, reclaim dry-run listing, the --apply
guard refusing to touch non-intermediate dirs, and clear-git-lock idempotency.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_guard():
    """Load scripts/pod_disk_guard.py as an isolated module (test isolation)."""
    spec = importlib.util.spec_from_file_location(
        "pod_disk_guard_under_test", REPO_ROOT / "scripts" / "pod_disk_guard.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_disk_guard_under_test"] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _make_merged_dir(root: Path, name: str, size_bytes: int = 1024) -> Path:
    """Create a fake checkpoint dir <name> under root with a single sized file."""
    d = root / name
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\0" * size_bytes)
    return d


# ── classification ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("coupling_merged", True),
        ("midtrain_25pct_coupling_merged", True),
        ("phase1_merged", True),
        ("run3_phase1_merged_seed42", True),
        ("em_merged", False),  # final EM checkpoint — never reclaim
        ("merged", False),
        ("coupling_adapter", False),  # not a *_merged dir name at all
        ("pre_em_checkpoint", False),
    ],
)
def test_is_intermediate_merged(name, expected):
    """Only coupling_merged-substring / *phase1_merged* names are intermediate."""
    assert guard.is_intermediate_merged(name) is expected


# ── reclaim dry-run listing ─────────────────────────────────────────────────────


def test_reclaim_dry_run_lists_but_does_not_delete(tmp_path, capsys):
    """Default reclaim (no --apply) lists intermediate dirs and deletes nothing."""
    inter = _make_merged_dir(tmp_path, "coupling_merged", size_bytes=2048)
    final = _make_merged_dir(tmp_path, "em_merged", size_bytes=4096)

    rc = guard.cmd_reclaim(argparse.Namespace(root=str(tmp_path), apply=False))
    out = capsys.readouterr().out

    assert rc == 0
    assert "DRY RUN" in out
    assert "coupling_merged" in out
    assert "RECLAIM" in out
    assert "KEEP" in out  # em_merged listed as keep
    # Nothing actually deleted.
    assert inter.exists()
    assert final.exists()


def test_reclaim_find_merged_dirs_tags_intermediacy(tmp_path):
    """find_merged_dirs tags coupling/phase1 as intermediate, em as final."""
    _make_merged_dir(tmp_path, "coupling_merged")
    _make_merged_dir(tmp_path, "run_phase1_merged_x")
    _make_merged_dir(tmp_path, "em_merged")

    found = {m.path.name: m.intermediate for m in guard.find_merged_dirs(tmp_path)}
    assert found["coupling_merged"] is True
    assert found["run_phase1_merged_x"] is True
    assert found["em_merged"] is False


# ── --apply guard ───────────────────────────────────────────────────────────────


def test_reclaim_apply_deletes_only_intermediate(tmp_path, capsys):
    """--apply deletes intermediate merged dirs but leaves the final em_merged intact."""
    inter = _make_merged_dir(tmp_path, "coupling_merged", size_bytes=2048)
    inter2 = _make_merged_dir(tmp_path, "fold_phase1_merged_seed1", size_bytes=1024)
    final = _make_merged_dir(tmp_path, "em_merged", size_bytes=4096)

    rc = guard.cmd_reclaim(argparse.Namespace(root=str(tmp_path), apply=True))
    out = capsys.readouterr().out

    assert rc == 0
    assert "deleted" in out
    assert not inter.exists()
    assert not inter2.exists()
    # The final/only merged dir is NEVER touched.
    assert final.exists()


def test_reclaim_apply_with_no_intermediate_is_noop(tmp_path, capsys):
    """--apply with only a final em_merged present deletes nothing and stays intact."""
    final = _make_merged_dir(tmp_path, "em_merged", size_bytes=4096)

    rc = guard.cmd_reclaim(argparse.Namespace(root=str(tmp_path), apply=True))
    out = capsys.readouterr().out

    assert rc == 0
    assert "nothing reclaimable" in out
    assert final.exists()


def test_reclaim_missing_root_raises(tmp_path):
    """A nonexistent scan root fails loud (no silent default)."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        guard.cmd_reclaim(argparse.Namespace(root=str(missing), apply=False))


# ── clear-git-lock idempotency ──────────────────────────────────────────────────


def _make_fake_repo(root: Path) -> Path:
    """Create a minimal dir with a .git subdir to look like a repo to the guard."""
    repo = root / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def test_clear_git_lock_removes_existing_lock(tmp_path):
    """A present .git/index.lock is removed and reported as removed."""
    repo = _make_fake_repo(tmp_path)
    lock = repo / ".git" / "index.lock"
    lock.write_text("")

    removed, detail = guard.clear_git_lock(repo)
    assert removed is True
    assert not lock.exists()
    assert "removed" in detail


def test_clear_git_lock_idempotent_when_absent(tmp_path):
    """Calling clear-git-lock when no lock exists is a safe no-op (idempotent)."""
    repo = _make_fake_repo(tmp_path)
    # First call: nothing there.
    removed1, _ = guard.clear_git_lock(repo)
    assert removed1 is False
    # Second call: still a no-op, no crash.
    removed2, _ = guard.clear_git_lock(repo)
    assert removed2 is False


def test_clear_git_lock_double_call_after_create(tmp_path):
    """Create lock, clear it, clear again — second clear is idempotent."""
    repo = _make_fake_repo(tmp_path)
    lock = repo / ".git" / "index.lock"
    lock.write_text("")

    assert guard.clear_git_lock(repo)[0] is True
    assert guard.clear_git_lock(repo)[0] is False  # idempotent


def test_clear_git_lock_non_repo_raises(tmp_path):
    """Pointing clear-git-lock at a non-repo path fails loud."""
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    with pytest.raises(FileNotFoundError):
        guard.clear_git_lock(not_a_repo)


# ── report fallback path (no real quota in CI) ──────────────────────────────────


def test_probe_quota_headroom_returns_tuple(tmp_path):
    """probe_quota_headroom returns (ok, share_free_gb, detail) without raising.

    On a normal tmpfs/local FS a tiny probe succeeds; we only assert the shape and
    that share_free_gb is a positive float.
    """
    ok, share_free_gb, detail = guard.probe_quota_headroom(tmp_path, min_gb=1)
    assert isinstance(ok, bool)
    assert isinstance(share_free_gb, float)
    assert share_free_gb >= 0.0
    assert isinstance(detail, str) and detail


def test_probe_quota_headroom_rejects_nonpositive_min_gb(tmp_path):
    """min_gb must be positive (boundary assert)."""
    with pytest.raises(AssertionError):
        guard.probe_quota_headroom(tmp_path, min_gb=0)
