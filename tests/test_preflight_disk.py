"""Tests for the quota-aware disk preflight checks (#4 probe + #8 budget).

Covers:
- The posix_fallocate canary probe deletes its temp file (success AND refusal).
- check_disk_budget FAILs when footprint > usable headroom, PASSes when under,
  and skips when no footprint supplied.
- estimate_footprint_gb arithmetic + LoRA-only halving.
- check_disk_space wires the probe result into the go/no-go decision (mocked).
- The #8 regression: on MooseFS (TB-scale share-free + probe-success), usable
  headroom is capped at the per-pod quota so an over-quota footprint is caught
  instead of silently passing against the share-level free.
- assert_out_root_headroom (#1414, the #1333 per-phase out-root pattern): pass
  returns free GB, statvfs-below-floor raise, EDQUOT-canary raise, unsupported
  fallocate fail-soft, out-root auto-mkdir, nonpositive-need ValueError.
"""

import errno
import os
import sys
import threading
import types
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    RUNPOD_PER_POD_QUOTA_GB,
    PreflightReport,
    _probe_writable_bytes,
    _quota_aware_headroom_gb,
    assert_out_root_headroom,
    check_disk_budget,
    check_disk_space,
    estimate_footprint_gb,
)

# ── #4: canary probe deletes its file ────────────────────────────────────────


def test_probe_success_deletes_file(tmp_path):
    """A successful small-canary allocation leaves no probe file behind."""
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert ok is True
    assert fallback_reason is None
    assert not (tmp_path / ".preflight_disk_probe.tmp").exists()
    # No stray temp files of any kind.
    assert list(tmp_path.iterdir()) == []


def test_probe_refusal_deletes_file(tmp_path, monkeypatch):
    """An EDQUOT refusal still deletes the probe file and reports not-ok."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert ok is False
    assert fallback_reason is None
    assert not (tmp_path / ".preflight_disk_probe.tmp").exists()
    assert list(tmp_path.iterdir()) == []


def test_probe_unsupported_filesystem_falls_back(tmp_path, monkeypatch):
    """EOPNOTSUPP signals fallback (ok=True) and still cleans up."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert ok is True
    assert fallback_reason is not None
    assert "errno" in fallback_reason
    assert not (tmp_path / ".preflight_disk_probe.tmp").exists()


def test_probe_ebadf_falls_back(tmp_path, monkeypatch):
    """EBADF (VAST/NFS-class fallocate on a valid fd — #1902 job 16139) degrades to fallback."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert ok is True
    assert fallback_reason is not None
    assert "errno=9" in fallback_reason
    assert not (tmp_path / ".preflight_disk_probe.tmp").exists()


def test_probe_ebadf_never_masks_edquot(tmp_path, monkeypatch):
    """EDQUOT stays the real quota signal (ok=False, no fallback) even after the
    EBADF fallback widened the caught-errno set — the MooseFS quota detection
    must never be swallowed."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert ok is False
    assert fallback_reason is None


def test_probe_unexpected_errno_still_raises(tmp_path, monkeypatch):
    """An errno outside the caught set (e.g. EIO) still raises — fail fast."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    with pytest.raises(OSError):
        _probe_writable_bytes(str(tmp_path), probe_bytes=4096)


def test_probe_zero_bytes_asserts(tmp_path):
    """A zero-byte probe never exercises the quota — guard against it."""
    with pytest.raises(AssertionError):
        _probe_writable_bytes(str(tmp_path), probe_bytes=0)


# ── #2042: unwritable check_path probes a same-filesystem writable dir ───────


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_probe_unwritable_dir_uses_same_fs_candidate(tmp_path, monkeypatch):
    """An unwritable check_path (``/`` on the VM) probes a user-writable
    directory on the SAME filesystem instead of raising EACCES (#2042)."""
    ro = tmp_path / "ro"
    ro.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    ro.chmod(0o555)
    try:
        assert _probe_writable_bytes(str(ro), 4096) == (True, None)
    finally:
        ro.chmod(0o755)
    assert not list(ro.glob(".preflight_disk_probe*"))
    assert not list(home.glob(".preflight_disk_probe*"))


def test_probe_no_same_fs_candidate_falls_back(tmp_path, monkeypatch):
    """With NO user-writable location on the filesystem, the probe degrades via
    the documented fallback contract — an explicit reason, never an OSError."""
    monkeypatch.setattr(preflight, "_writable_probe_dir", lambda check_path, candidates=None: None)
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), 4096)
    assert ok is True
    assert fallback_reason is not None
    assert "no user-writable" in fallback_reason


def test_writable_probe_dir_rejects_cross_fs_candidates(tmp_path):
    """The same-st_dev pin is load-bearing: a writable candidate on a DIFFERENT
    filesystem is rejected (it would measure the wrong disk); an empty candidate
    list yields None; a same-fs writable candidate IS returned."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory modes")
    if sys.platform != "linux" or not os.path.isdir("/dev/shm"):
        pytest.skip("needs Linux /dev/shm for a genuine cross-fs candidate")
    ro = tmp_path / "ro"
    ro.mkdir()
    try:
        ro.chmod(0o555)
        if os.stat("/dev/shm").st_dev == os.stat(ro).st_dev:
            pytest.skip("/dev/shm unexpectedly on the same filesystem as tmp_path")
        # GENUINE st_dev mismatch, mock-free: /dev/shm is writable (tmpfs) but
        # sits on a different filesystem than the unwritable check_path.
        assert preflight._writable_probe_dir(str(ro), candidates=["/dev/shm"]) is None
        # No candidates at all -> None.
        assert preflight._writable_probe_dir(str(ro), candidates=[]) is None
        # A writable candidate on the SAME filesystem IS returned.
        w = tmp_path / "w"
        w.mkdir()
        assert preflight._writable_probe_dir(str(ro), candidates=[str(w)]) == str(w)
    finally:
        ro.chmod(0o755)


def test_probe_creates_missing_check_path(tmp_path):
    """A nonexistent check_path is still created — the create-if-missing
    contract ``assert_out_root_headroom`` relies on for fresh out-roots."""
    target = tmp_path / "new" / "sub"
    assert not target.exists()
    assert _probe_writable_bytes(str(target), 4096) == (True, None)
    assert target.is_dir()


def test_probe_open_edquot_refuses(tmp_path, monkeypatch):
    """EDQUOT at probe-file CREATION (``os.open`` on an already-exhausted quota)
    is a real refusal -> (False, None), matching the fallocate contract."""

    def edquot_open(path, flags, *args, **kwargs):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(preflight.os, "open", edquot_open)
    assert _probe_writable_bytes(str(tmp_path), 4096) == (False, None)
    assert list(tmp_path.iterdir()) == []

    # Negative control: EACCES at open (a post-resolver os.access lie, e.g.
    # root-squash NFS) still propagates to the caller's OSError backstop.
    def eacces_open(path, flags, *args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(preflight.os, "open", eacces_open)
    with pytest.raises(OSError) as excinfo:
        _probe_writable_bytes(str(tmp_path), 4096)
    assert excinfo.value.errno == errno.EACCES


# ── #8: disk-budget check ────────────────────────────────────────────────────


def test_budget_skipped_when_none():
    """No footprint supplied => budget check is a no-op and never FAILs."""
    report = PreflightReport()
    report.disk_probed_headroom_gb = 10.0
    check_disk_budget(report, planned_footprint_gb=None)
    assert report.ok is True
    assert report.errors == []


def test_budget_pass_under_headroom():
    """Footprint under probed headroom keeps the report OK."""
    report = PreflightReport()
    report.disk_probed_headroom_gb = 130.0
    check_disk_budget(report, planned_footprint_gb=60.0)
    assert report.ok is True
    assert report.errors == []


def test_budget_fail_over_headroom_with_ranked_remediation():
    """Footprint over usable headroom FAILs with ranked remediation guidance."""
    report = PreflightReport()
    report.disk_probed_headroom_gb = 40.0
    report.disk_headroom_basis = "per-pod quota cap (130GB)"
    check_disk_budget(report, planned_footprint_gb=120.0)
    assert report.ok is False
    assert len(report.errors) == 1
    msg = report.errors[0]
    assert "Disk budget exceeded" in msg
    # The headroom basis must be named so a quota-capped number is never
    # mislabeled as a real "probed" reservation (the #8 reviewer blocker).
    assert "probed headroom" not in msg
    assert "basis: per-pod quota cap (130GB)" in msg
    # Ranked remediation: LoRA-only first, then sequentialize, then larger volume.
    assert "LoRA-only" in msg
    assert "sequentialize" in msg
    assert "larger volume" in msg
    lora_idx = msg.index("LoRA-only")
    seq_idx = msg.index("sequentialize")
    vol_idx = msg.index("larger volume")
    assert lora_idx < seq_idx < vol_idx


# ── estimate_footprint_gb ────────────────────────────────────────────────────


def test_estimate_footprint_merged_doubles_per_cell():
    """Materializing merged adapters doubles per-cell disk."""
    base = 15.0
    merged = estimate_footprint_gb(base, n_cells=3, materialize_merged=True)
    lora_only = estimate_footprint_gb(base, n_cells=3, materialize_merged=False)
    assert merged == pytest.approx(base * 2 * 3)
    assert lora_only == pytest.approx(base * 3)
    assert merged == pytest.approx(lora_only * 2)


def test_estimate_footprint_sequential_single_cell():
    """n_cells=1 models a strictly sequential, delete-after-each run."""
    assert estimate_footprint_gb(20.0, n_cells=1, materialize_merged=True) == pytest.approx(40.0)


def test_estimate_footprint_rejects_bad_args():
    """Guards against negative model size and zero cells."""
    with pytest.raises(AssertionError):
        estimate_footprint_gb(-1.0, n_cells=1)
    with pytest.raises(AssertionError):
        estimate_footprint_gb(10.0, n_cells=0)


# ── check_disk_space: probe drives go/no-go (mocked) ─────────────────────────


def _patch_disk_usage(monkeypatch, free_gb: float):
    """Make shutil.disk_usage report a fixed share-level free size."""

    class _Usage:
        total = 200 * (1024**3)
        used = int((200 - free_gb) * (1024**3))
        free = int(free_gb * (1024**3))

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: _Usage())
    # Force the non-/workspace branch so check_path is deterministic ("/" here is fine
    # for shutil mock; the probe is mocked separately).
    monkeypatch.setattr(preflight.Path, "exists", lambda self: False)


def test_check_disk_space_probe_refusal_fails(monkeypatch):
    """Probe refusal (EDQUOT) FAILs even when share-level free looks huge."""
    _patch_disk_usage(monkeypatch, free_gb=145_000.0)  # TB-scale share free
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (False, None))
    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    assert report.ok is False
    assert report.disk_probed_headroom_gb == 0.0
    assert any("quota" in e.lower() for e in report.errors)


def test_check_disk_space_probe_success_passes(monkeypatch):
    """Probe success with ample share-level free PASSes.

    Here share-free (120GB) is below the 130GB per-pod quota, so the share-free is
    the binding headroom and the basis names it as such.
    """
    _patch_disk_usage(monkeypatch, free_gb=120.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))
    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    assert report.ok is True
    assert report.disk_free_gb == pytest.approx(120.0)
    assert report.disk_probed_headroom_gb == pytest.approx(120.0)
    assert "share-level free" in report.disk_headroom_basis
    assert "below per-pod quota cap" in report.disk_headroom_basis


def test_check_disk_space_probe_success_but_low_free_fails(monkeypatch):
    """Probe succeeds but share-level free is below the threshold => FAIL."""
    _patch_disk_usage(monkeypatch, free_gb=10.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))
    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    assert report.ok is False
    assert any("free" in e.lower() for e in report.errors)


def test_check_disk_space_fallback_when_unsupported(monkeypatch):
    """Unsupported fallocate => warn + fall back to shutil.disk_usage for go/no-go."""
    _patch_disk_usage(monkeypatch, free_gb=120.0)
    monkeypatch.setattr(
        preflight, "_probe_writable_bytes", lambda p, b: (True, "posix_fallocate unsupported")
    )
    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    assert report.ok is True
    assert report.disk_probed_headroom_gb == pytest.approx(120.0)
    assert any("fall" in w.lower() or "fallback" in w.lower() for w in report.warnings)


def test_canary_is_small_not_min_free(monkeypatch):
    """The probe canary must be probe_gb-sized, NOT min_free_gb-sized."""
    captured = {}

    def fake_probe(check_path, probe_bytes):
        captured["bytes"] = probe_bytes
        return (True, None)

    _patch_disk_usage(monkeypatch, free_gb=120.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", fake_probe)
    report = PreflightReport()
    check_disk_space(report, min_free_gb=130.0, probe_gb=1.0)
    # Canary is 1 GB, far below the 130 GB requirement.
    assert captured["bytes"] == pytest.approx(int(1.0 * (1024**3)))
    assert captured["bytes"] < int(130.0 * (1024**3))


def test_probe_real_roundtrip_in_tmp(tmp_path):
    """End-to-end real probe in a tmp dir leaves the directory clean."""
    ok, _reason = _probe_writable_bytes(str(tmp_path), probe_bytes=1024)
    # On a normal filesystem this should succeed; on an exotic FS it may report
    # fallback. Either way the temp file must be gone.
    assert isinstance(ok, bool)
    assert not (Path(tmp_path) / ".preflight_disk_probe.tmp").exists()


# ── #8 regression: quota-aware headroom catches over-quota footprints ─────────


def test_quota_aware_headroom_caps_at_quota():
    """When share-free dwarfs the quota, the quota is the binding headroom."""
    headroom, basis = _quota_aware_headroom_gb(share_free_gb=145_000.0, quota_gb=130.0)
    assert headroom == pytest.approx(130.0)
    assert "per-pod quota cap" in basis


def test_quota_aware_headroom_uses_share_free_when_smaller():
    """When share-free is below the quota, share-free is the binding headroom."""
    headroom, basis = _quota_aware_headroom_gb(share_free_gb=40.0, quota_gb=130.0)
    assert headroom == pytest.approx(40.0)
    assert "share-level free" in basis
    assert "below per-pod quota cap" in basis


def test_quota_aware_headroom_none_disables_cap_and_labels_it():
    """quota_gb=None keeps the raw share-free but flags it as quota-blind."""
    headroom, basis = _quota_aware_headroom_gb(share_free_gb=145_000.0, quota_gb=None)
    assert headroom == pytest.approx(145_000.0)
    assert "undetectable" in basis


def test_moosefs_overquota_footprint_caught_end_to_end(monkeypatch):
    """#8 REGRESSION: TB-scale share-free + probe-success must STILL catch an
    over-per-pod-quota footprint.

    This drives ``check_disk_space`` exactly as production does (no hand-set
    headroom) on a MooseFS-shaped filesystem: ``shutil.disk_usage`` reports
    145,000GB free and the small canary allocation succeeds (the pod is not yet
    exhausted). A 200GB planned footprint exceeds the ~130GB per-pod quota but is
    far under the share-level free. Before the quota cap, ``check_disk_space`` set
    ``disk_probed_headroom_gb`` to the 145,000GB share-free, so the budget check
    PASSed an over-quota sweep. With the cap, the budget check FAILs.
    """
    _patch_disk_usage(monkeypatch, free_gb=145_000.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))

    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    # check_disk_space itself still PASSes — the pod is not yet exhausted — but the
    # usable headroom is the per-pod quota, NOT the terabyte-scale share-free.
    assert report.ok is True
    assert report.disk_free_gb == pytest.approx(145_000.0)
    assert report.disk_probed_headroom_gb == pytest.approx(RUNPOD_PER_POD_QUOTA_GB)
    assert "per-pod quota cap" in report.disk_headroom_basis

    # The over-quota footprint is now caught (it was silently passed before).
    check_disk_budget(report, planned_footprint_gb=200.0)
    assert report.ok is False
    msg = report.errors[-1]
    assert "Disk budget exceeded" in msg
    assert "probed headroom" not in msg  # never mislabel the quota-capped number
    assert "per-pod quota cap" in msg


def test_moosefs_underquota_footprint_passes_end_to_end(monkeypatch):
    """The complement: a footprint comfortably under the per-pod quota PASSes."""
    _patch_disk_usage(monkeypatch, free_gb=145_000.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))

    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    check_disk_budget(report, planned_footprint_gb=60.0)
    assert report.ok is True
    assert report.errors == []


def test_check_disk_space_custom_quota_override(monkeypatch):
    """A larger explicit per-pod quota raises the headroom ceiling accordingly."""
    _patch_disk_usage(monkeypatch, free_gb=145_000.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))

    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0, quota_gb=500.0)
    assert report.disk_probed_headroom_gb == pytest.approx(500.0)
    check_disk_budget(report, planned_footprint_gb=200.0)
    assert report.ok is True  # 200GB under the 500GB explicit quota


# ── #1414: per-phase out-root headroom assert (the #1333 pattern, shared) ────


def test_assert_out_root_headroom_pass_returns_free_gb(tmp_path):
    """Healthy out-root: returns free GB as a float and leaves no stray probe file."""
    free_gb = assert_out_root_headroom(tmp_path, need_gb=0.001, canary_gb=4096 / 1e9)
    assert isinstance(free_gb, float)
    assert free_gb > 0
    assert list(tmp_path.iterdir()) == []


def test_assert_out_root_headroom_raises_below_floor(tmp_path, monkeypatch):
    """statvfs free below the §9 floor raises BEFORE the canary, naming path + mount + floor."""
    # Concrete fake with real numeric attributes (never a MagicMock): ~1.02 GB free.
    fake_stat = types.SimpleNamespace(f_bavail=1_000_000, f_frsize=1024)
    monkeypatch.setattr(preflight.os, "statvfs", lambda p: fake_stat)

    def exploding_fallocate(fd, offset, length):
        raise AssertionError("canary must not run when the statvfs floor already failed")

    # Proves the raise happens before the fallocate canary path is reached.
    monkeypatch.setattr(preflight.os, "posix_fallocate", exploding_fallocate)

    with pytest.raises(RuntimeError) as exc_info:
        assert_out_root_headroom(tmp_path, need_gb=100.0, phase="p2_train")
    msg = str(exc_info.value)
    assert str(tmp_path) in msg
    assert "GB free" in msg
    assert "100.0 GB" in msg
    assert "p2_train" in msg
    # Names SOME mount token (stub-agnostic: any non-empty token, no /proc/mounts value pinned).
    assert "(mount " in msg
    mount_token = msg.split("(mount ", 1)[1].split(")", 1)[0]
    assert mount_token.strip()


def test_assert_out_root_headroom_raises_on_quota_refusal(tmp_path, monkeypatch):
    """An EDQUOT canary refusal raises even when statvfs shows ample free space."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    with pytest.raises(RuntimeError) as exc_info:
        assert_out_root_headroom(tmp_path, need_gb=0.001, canary_gb=4096 / 1e9)
    msg = str(exc_info.value)
    assert "canary" in msg
    assert "EDQUOT" in msg or "quota" in msg
    assert not (tmp_path / ".preflight_disk_probe.tmp").exists()


def test_assert_out_root_headroom_tolerates_unsupported_fallocate(tmp_path, monkeypatch):
    """EOPNOTSUPP degrades to the statvfs-only check (fail-soft parity with the probe)."""

    def fake_fallocate(fd, offset, length):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    monkeypatch.setattr(preflight.os, "posix_fallocate", fake_fallocate)
    free_gb = assert_out_root_headroom(tmp_path, need_gb=0.001, canary_gb=4096 / 1e9)
    assert free_gb > 0


def test_assert_out_root_headroom_creates_out_root(tmp_path):
    """A nested nonexistent out-root is mkdir'd before the probe and the assert passes."""
    dest = tmp_path / "a" / "b"
    assert not dest.exists()
    free_gb = assert_out_root_headroom(dest, need_gb=0.001, canary_gb=4096 / 1e9)
    assert dest.is_dir()
    assert free_gb > 0


def test_assert_out_root_headroom_rejects_nonpositive_need(tmp_path):
    """need_gb <= 0 raises ValueError (explicit raise — asserts strip under python -O)."""
    with pytest.raises(ValueError):
        assert_out_root_headroom(tmp_path, need_gb=0)


# ── #1979 regression: concurrent probes on ONE shared dir must not collide ────


def test_probe_concurrent_workers_all_succeed_no_leftovers(tmp_path):
    """8 concurrent probes on ONE dir all succeed and leave the dir empty.

    The #1979 fellows job 16686 crash: 8 per-unit workers each ran the startup
    headroom probe against one shared out-root; the fixed probe filename made
    siblings open/fallocate/unlink one common path. Post-fix each invocation
    uses a unique per-invocation filename, so concurrent probes never touch
    each other's files.
    """
    n = 8
    results: list[tuple[bool, str | None] | None] = [None] * n
    errors: list[BaseException] = []
    barrier = threading.Barrier(n, timeout=30)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            results[i] = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"worker exceptions: {errors!r}"
    assert all(r == (True, None) for r in results), results
    # Every probe cleaned up exactly its own file — no leftovers of any kind.
    assert list(tmp_path.iterdir()) == []


def test_probe_survives_sibling_interference_on_legacy_shared_name(tmp_path, monkeypatch):
    """A sibling unlinking/recreating the LEGACY fixed probe name cannot EBADF us.

    Simulates the cluster-shared-filesystem semantics of the #1979 crash: on the
    fellows share, a sibling's unlink/recreate of the probe path invalidates a
    worker's already-open fd mid-``posix_fallocate`` (OSError EBADF — outside the
    handled errno sets, so the probe raised and the worker died rc=1). The fake
    fallocate below performs that sibling interference against the OLD fixed name
    ``.preflight_disk_probe.tmp`` and then applies shared-FS semantics: EBADF iff
    this fd's path no longer resolves to the same inode. Pre-fix (fixed name) the
    interference hits our own path and this test fails with OSError EBADF;
    post-fix (unique per-invocation name) the probe is untouched and succeeds.
    """

    def shared_fs_fallocate(fd, offset, length):
        fd_path = os.readlink(f"/proc/self/fd/{fd}")
        fd_stat = os.fstat(fd)
        # Sibling running the legacy fixed-name protocol: unlink + recreate + unlink.
        legacy = Path(fd_path).parent / ".preflight_disk_probe.tmp"
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

    monkeypatch.setattr(preflight.os, "posix_fallocate", shared_fs_fallocate)
    ok, fallback_reason = _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert ok is True
    assert fallback_reason is None
    assert list(tmp_path.iterdir()) == []


def test_probe_paths_unique_per_invocation(tmp_path, monkeypatch):
    """Two sequential probes use DISTINCT probe filenames (the #1979 invariant)."""
    seen: list[str] = []

    def recording_fallocate(fd, offset, length):
        seen.append(os.readlink(f"/proc/self/fd/{fd}"))

    monkeypatch.setattr(preflight.os, "posix_fallocate", recording_fallocate)
    _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    _probe_writable_bytes(str(tmp_path), probe_bytes=4096)
    assert len(seen) == 2
    assert seen[0] != seen[1], seen
    assert list(tmp_path.iterdir()) == []
