"""Tests for the VM-root free-space hard FLOOR in preflight (task #679).

The legacy ``check_disk_space`` path WARNed below ``min_free_gb`` only when the
caller passed a footprint-derived minimum. #679 promotes a configurable hard
FLOOR on the VM ROOT (``check_path == "/"``) so a fresh launch below the floor
FAILs fast (``report.ok=False``) rather than starting on a disk already near
the silent-Bash-failure regime. RunPod ``/workspace`` is exempt (its binding
signal is the EDQUOT quota probe, not free GB). An operator override
(``EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE=1``) degrades the FAIL to a WARN.

Mocks ``shutil.disk_usage`` + ``_probe_writable_bytes`` like
``tests/test_preflight_disk.py`` so no real disk is touched; ``Path.exists`` is
forced False so ``_disk_check_path()`` resolves to ``/`` (not ``/workspace``).
"""

import pytest

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    VM_ROOT_DISK_FLOOR_GB_DEFAULT,
    PreflightReport,
    check_disk_space,
)


def _patch_root_disk(monkeypatch, free_gb: float):
    """shutil.disk_usage reports free_gb; Path.exists False forces the / branch
    (not /workspace) so check_path == '/'. The canary probe succeeds (normal FS)."""

    class _Usage:
        total = 500 * (1024**3)
        used = int((500 - free_gb) * (1024**3))
        free = int(free_gb * (1024**3))

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: _Usage())
    monkeypatch.setattr(preflight.Path, "exists", lambda self: False)
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))
    # Neither cluster nor runpod -> _disk_check_path() returns "/".
    monkeypatch.setattr(preflight, "is_cluster_env", lambda: False)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: False)


# ─── default floor (40 GB) ───────────────────────────────────────────────────


def test_default_floor_is_40gb():
    assert pytest.approx(40.0) == VM_ROOT_DISK_FLOOR_GB_DEFAULT


def test_below_floor_fails(monkeypatch):
    """Below the 40 GB floor on / -> report.ok False with a floor-named error."""
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_GB", raising=False)
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", raising=False)
    _patch_root_disk(monkeypatch, free_gb=30.0)
    report = PreflightReport()
    # min_free_gb below the floor so the legacy gate does NOT pre-empt it.
    check_disk_space(report, min_free_gb=10.0, probe_gb=1.0)
    assert report.ok is False
    assert any("below floor" in e for e in report.errors)
    assert any("40GB" in e for e in report.errors)


def test_above_floor_passes(monkeypatch):
    """Comfortably above the floor -> ok, no floor error."""
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_GB", raising=False)
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", raising=False)
    _patch_root_disk(monkeypatch, free_gb=120.0)
    report = PreflightReport()
    check_disk_space(report, min_free_gb=10.0, probe_gb=1.0)
    assert report.ok is True
    assert not any("floor" in e for e in report.errors)


def test_override_degrades_to_warning(monkeypatch):
    """With the override env set, a below-floor launch PASSes with a WARN."""
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_GB", raising=False)
    monkeypatch.setenv("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", "1")
    _patch_root_disk(monkeypatch, free_gb=30.0)
    report = PreflightReport()
    check_disk_space(report, min_free_gb=10.0, probe_gb=1.0)
    assert report.ok is True  # override -> no error
    assert any("OVERRIDDEN" in w for w in report.warnings)


# ─── env override of the floor value ─────────────────────────────────────────


def test_env_raises_floor(monkeypatch):
    """EPM_PREFLIGHT_DISK_FLOOR_GB raises the floor; 55 GB free now fails an 80
    GB floor."""
    monkeypatch.setenv("EPM_PREFLIGHT_DISK_FLOOR_GB", "80")
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", raising=False)
    _patch_root_disk(monkeypatch, free_gb=55.0)
    report = PreflightReport()
    check_disk_space(report, min_free_gb=10.0, probe_gb=1.0)
    assert report.ok is False
    assert any("80GB" in e for e in report.errors)


def test_garbled_env_falls_back_to_default(monkeypatch):
    """A non-numeric floor env falls back to the 40 GB default (never crashes)."""
    monkeypatch.setenv("EPM_PREFLIGHT_DISK_FLOOR_GB", "lots")
    assert preflight._vm_root_disk_floor_gb() == pytest.approx(40.0)


# ─── RunPod /workspace is exempt ─────────────────────────────────────────────


def test_runpod_workspace_exempt_from_floor(monkeypatch):
    """On RunPod (check_path == /workspace) the floor does NOT apply — the
    EDQUOT probe is the binding signal, and a TB-scale share would otherwise
    spuriously satisfy/fail a free-GB floor."""
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_GB", raising=False)
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", raising=False)

    class _Usage:
        total = 500 * (1024**3)
        used = 470 * (1024**3)
        free = 30 * (1024**3)  # below the 40 GB floor

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: _Usage())
    monkeypatch.setattr(preflight, "_probe_writable_bytes", lambda p, b: (True, None))
    monkeypatch.setattr(preflight, "_disk_check_path", lambda: "/workspace")
    report = PreflightReport()
    # min_free_gb=10 so the legacy gate doesn't fire; the floor must NOT fire on
    # /workspace even though free (30) < floor (40).
    check_disk_space(report, min_free_gb=10.0, probe_gb=1.0)
    assert not any("below floor" in e for e in report.errors)


# ─── the _check_vm_root_floor helper directly ────────────────────────────────


def test_floor_helper_noop_off_root():
    report = PreflightReport()
    report.disk_free_gb = 5.0  # well below any floor
    preflight._check_vm_root_floor(report, "/workspace", min_free_gb=10.0)
    assert report.errors == []
    assert report.warnings == []


def test_floor_helper_dedups_against_min_free(monkeypatch):
    """When min_free_gb is the higher bar and already errored, the floor adds no
    second error (the run is already failing)."""
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_GB", raising=False)
    monkeypatch.delenv("EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE", raising=False)
    report = PreflightReport()
    report.disk_free_gb = 30.0
    # min_free_gb=50 (> floor 40) and free 30 < 50: the legacy gate would have
    # already errored, so the floor stays quiet.
    preflight._check_vm_root_floor(report, "/", min_free_gb=50.0)
    assert report.errors == []
