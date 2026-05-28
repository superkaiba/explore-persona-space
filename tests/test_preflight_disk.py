"""Tests for the quota-aware disk preflight checks (#4 probe + #8 budget).

Covers:
- The posix_fallocate canary probe deletes its temp file (success AND refusal).
- check_disk_budget FAILs when footprint > probed headroom, PASSes when under,
  and skips when no footprint supplied.
- estimate_footprint_gb arithmetic + LoRA-only halving.
- check_disk_space wires the probe result into the go/no-go decision (mocked).
"""

import errno
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    PreflightReport,
    _probe_writable_bytes,
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


def test_probe_zero_bytes_asserts(tmp_path):
    """A zero-byte probe never exercises the quota — guard against it."""
    with pytest.raises(AssertionError):
        _probe_writable_bytes(str(tmp_path), probe_bytes=0)


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
    """Footprint over probed headroom FAILs with ranked remediation guidance."""
    report = PreflightReport()
    report.disk_probed_headroom_gb = 40.0
    check_disk_budget(report, planned_footprint_gb=120.0)
    assert report.ok is False
    assert len(report.errors) == 1
    msg = report.errors[0]
    assert "Disk budget exceeded" in msg
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
    """Probe success with ample share-level free PASSes."""
    _patch_disk_usage(monkeypatch, free_gb=120.0)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))
    report = PreflightReport()
    check_disk_space(report, min_free_gb=50.0, probe_gb=1.0)
    assert report.ok is True
    assert report.disk_free_gb == pytest.approx(120.0)
    assert report.disk_probed_headroom_gb == pytest.approx(120.0)


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
