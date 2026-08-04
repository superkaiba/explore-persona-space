"""Tests for the pod-side disk guard helper (scripts/pod_disk_guard.py).

Covers: intermediate-vs-final classification, reclaim dry-run listing, the --apply
guard refusing to touch non-intermediate dirs, clear-git-lock idempotency, and the
#1979-class per-invocation probe-filename + widened errno fallback (siblings of
`tests/test_preflight_disk.py` for the pod-side probe).
"""

from __future__ import annotations

import argparse
import errno
import importlib.util
import os
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


# ── #1983 (#1979 sibling): per-invocation probe filename + widened errno fallback ───
#
# `probe_quota_headroom` in scripts/pod_disk_guard.py mirrors the #1979 fix
# already landed on `preflight._probe_writable_bytes` (commit 22c2ddb2d3): a
# unique per-invocation probe filename so concurrent workers on ONE shared
# filesystem cannot invalidate each other's fd mid-fallocate, and a widened
# fallback errno set (EOPNOTSUPP + ENOSYS + EINVAL + EBADF).


def test_probe_survives_sibling_interference_on_legacy_shared_name(tmp_path, monkeypatch):
    """A sibling unlinking/recreating the LEGACY fixed probe name cannot EBADF us.

    Simulates the shared-filesystem semantics of the #1979 crash class (fellows
    fallocate on a valid fd on VAST/NFS-class filesystems): a sibling worker's
    unlink/recreate of the fixed-name probe path invalidates our already-open
    fd mid-`posix_fallocate` (OSError EBADF). The fake fallocate performs that
    sibling interference against the OLD fixed name `.pod_disk_guard_probe.tmp`
    and applies shared-FS semantics: EBADF iff this fd's path no longer
    resolves to the same inode. Pre-fix (fixed name) the interference hits our
    own path and the probe raised; post-fix (unique per-invocation name) the
    probe is untouched and returns ok=True with EDQUOT still detectable.
    """

    def shared_fs_fallocate(fd, offset, length):
        fd_path = os.readlink(f"/proc/self/fd/{fd}")
        fd_stat = os.fstat(fd)
        # Sibling running the legacy fixed-name protocol: unlink + recreate + unlink.
        legacy = Path(fd_path).parent / ".pod_disk_guard_probe.tmp"
        legacy.unlink(missing_ok=True)
        legacy.touch()
        legacy.unlink()
        # Cluster-share semantics: fallocate on an fd whose path was replaced fails.
        try:
            st = os.stat(fd_path)
            same = st.st_ino == fd_stat.st_ino and st.st_dev == fd_stat.st_dev
        except FileNotFoundError:
            same = False
        if not same:
            raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(guard.os, "posix_fallocate", shared_fs_fallocate)
    ok, share_free_gb, detail = guard.probe_quota_headroom(tmp_path, min_gb=1)
    assert ok is True
    assert share_free_gb >= 0.0
    assert "unsupported" not in detail.lower()  # Uniquified name never fell back.
    assert list(tmp_path.iterdir()) == []


def test_probe_paths_unique_per_invocation(tmp_path, monkeypatch):
    """Two sequential probes use DISTINCT probe filenames (the #1983 invariant)."""
    seen: list[str] = []

    def recording_fallocate(fd, offset, length):
        seen.append(os.readlink(f"/proc/self/fd/{fd}"))

    monkeypatch.setattr(guard.os, "posix_fallocate", recording_fallocate)
    guard.probe_quota_headroom(tmp_path, min_gb=1)
    guard.probe_quota_headroom(tmp_path, min_gb=1)
    assert len(seen) == 2
    assert seen[0] != seen[1], seen
    assert list(tmp_path.iterdir()) == []


def test_probe_ebadf_falls_back(tmp_path, monkeypatch):
    """EBADF degrades to the share-level free fallback (widened errno set)."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(guard.os, "posix_fallocate", fake_fallocate)
    ok, share_free_gb, detail = guard.probe_quota_headroom(tmp_path, min_gb=1)
    # share_free_gb on any real tmp_path is >> 1 GB, so ok reflects the fallback.
    assert ok is (share_free_gb >= 1)
    assert "unsupported" in detail.lower()
    assert list(tmp_path.iterdir()) == []


def test_probe_ebadf_never_masks_edquot(tmp_path, monkeypatch):
    """EDQUOT stays the real quota signal (ok=False, no fallback) even after the
    EBADF fallback widened the caught-errno set — the MooseFS quota detection
    must never be swallowed."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(guard.os, "posix_fallocate", fake_fallocate)
    ok, _share_free_gb, detail = guard.probe_quota_headroom(tmp_path, min_gb=1)
    assert ok is False
    assert "QUOTA EXHAUSTED" in detail
