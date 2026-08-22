"""Unit tests for the shared-VM swap-state preflight check (#2280).

``_check_swap_state`` is WARN-only by design: earlyoom's victim-kill
condition is CONJUNCTIVE (MemAvailable AND swap free both <= 10%), so a
swap regression silently re-arms the Aug-13 selective-SIGTERM kill regime.
Two regression paths, each with its own WARN arm:

* the ``nofail`` fstab entry boots with NO swap when /mnt/eps-data fails
  to mount (SwapTotal == 0);
* swap exhaustion drives SwapFree toward earlyoom's 10% swap floor while
  SwapTotal stays healthy.

Every arm must leave ``report.ok`` unchanged (True) — the check may never
add an error. Fakes sit ONLY at the external boundary (environment
detection ``is_shared_vm_env`` + the ``_read_proc_meminfo`` filesystem
seam, both zero-arg like the real callees); the check body runs for real.
"""

from __future__ import annotations

import pytest

from explore_persona_space.orchestrate import preflight


def _meminfo(swap_total_kb: int, swap_free_kb: int) -> str:
    """A realistic /proc/meminfo slice with the two swap rows embedded."""
    return (
        "MemTotal:       131072000 kB\n"
        "MemFree:        12345678 kB\n"
        "MemAvailable:   98765432 kB\n"
        f"SwapTotal:      {swap_total_kb} kB\n"
        f"SwapFree:       {swap_free_kb} kB\n"
        "Dirty:                128 kB\n"
    )


def _run_swap_check(
    monkeypatch,
    *,
    shared_vm: bool = True,
    meminfo_text: str | None = None,
    read_error: Exception | None = None,
) -> preflight.PreflightReport:
    """Run ``_check_swap_state`` against a faked env + /proc/meminfo seam."""
    report = preflight.PreflightReport()
    monkeypatch.setattr(preflight, "is_shared_vm_env", lambda: shared_vm)
    if read_error is not None:

        def _raise() -> str:
            raise read_error

        monkeypatch.setattr(preflight, "_read_proc_meminfo", _raise)
    else:
        monkeypatch.setattr(preflight, "_read_proc_meminfo", lambda: meminfo_text)
    preflight._check_swap_state(report)
    return report


def test_healthy_swap_no_warning(monkeypatch):
    """Arm 1: healthy swap (64 GiB, ~41% free) -> fields set, zero warnings."""
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(67108860, 27647160))
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []
    assert report.swap_total_gb == pytest.approx(64.0, abs=0.01)
    assert report.swap_free_pct == pytest.approx(41.2, abs=0.1)


def test_swap_total_zero_warns_nofail_path(monkeypatch):
    """Arm 2: SwapTotal == 0 (nofail no-swap boot) -> one WARN, ok unchanged."""
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(0, 0))
    assert report.ok is True
    assert report.errors == []
    assert len(report.warnings) == 1
    warning = report.warnings[0]
    assert "SwapTotal=0" in warning
    assert "nofail" in warning
    assert "swapon /mnt/eps-data/swapfile" in warning
    assert report.swap_total_gb == 0.0
    # Ratio undefined at SwapTotal=0 — stays None by design.
    assert report.swap_free_pct is None


def test_low_swap_free_warns_exhaustion_path(monkeypatch):
    """Arm 3: SwapTotal > 0 with low SwapFree -> one WARN, ok unchanged."""
    # 64 GiB total, 10% free — under the 20% default WARN threshold.
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(67108860, 6710886))
    assert report.ok is True
    assert report.errors == []
    assert len(report.warnings) == 1
    warning = report.warnings[0]
    assert "nearly exhausted" in warning
    assert "EPM_PREFLIGHT_SWAP_FREE_WARN_PCT" in warning
    assert report.swap_total_gb == pytest.approx(64.0, abs=0.01)
    assert report.swap_free_pct == pytest.approx(10.0, abs=0.1)


def test_at_threshold_does_not_warn(monkeypatch):
    """Exactly at the WARN threshold (20%) -> no warning (strict <)."""
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(67108860, 13421772))
    assert report.ok is True
    assert report.warnings == []
    assert report.swap_free_pct == pytest.approx(20.0, abs=0.01)


def test_env_threshold_override(monkeypatch):
    """EPM_PREFLIGHT_SWAP_FREE_WARN_PCT raises the WARN bar; garbled -> default."""
    monkeypatch.setenv("EPM_PREFLIGHT_SWAP_FREE_WARN_PCT", "50")
    report = _run_swap_check(
        monkeypatch,
        meminfo_text=_meminfo(67108860, 27647160),  # ~41% free
    )
    assert report.ok is True
    assert len(report.warnings) == 1  # 41% < 50% override
    monkeypatch.setenv("EPM_PREFLIGHT_SWAP_FREE_WARN_PCT", "garbled")
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(67108860, 27647160))
    assert report.warnings == []  # falls back to the 20% default


def test_off_shared_vm_skips_clean(monkeypatch):
    """Not the shared VM (pods/GCE/SLURM) -> no fields, no rows."""
    report = _run_swap_check(monkeypatch, shared_vm=False, meminfo_text=_meminfo(0, 0))
    assert report.ok is True
    assert report.warnings == []
    assert report.swap_total_gb is None
    assert report.swap_free_pct is None


def test_read_error_degrades_to_single_warning(monkeypatch):
    """An OSError reading meminfo -> one warning, never raises, ok unchanged."""
    report = _run_swap_check(monkeypatch, read_error=OSError("boom"))
    assert report.ok is True
    assert len(report.warnings) == 1
    assert "/proc/meminfo" in report.warnings[0]
    assert report.swap_total_gb is None


def test_unparseable_meminfo_degrades_to_single_warning(monkeypatch):
    """Missing SwapTotal/SwapFree rows -> one warning, fields unset."""
    report = _run_swap_check(monkeypatch, meminfo_text="MemTotal: 1 kB\n")
    assert report.ok is True
    assert len(report.warnings) == 1
    assert "SwapTotal/SwapFree" in report.warnings[0]
    assert report.swap_total_gb is None
    assert report.swap_free_pct is None


def test_summary_renders_swap_line(monkeypatch):
    """The summary carries the swap line in both shapes (mirrors data-disk)."""
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(67108860, 27647160))
    assert "Swap: 64 GB total, 41.2% free" in report.summary()
    report = _run_swap_check(monkeypatch, meminfo_text=_meminfo(0, 0))
    assert "Swap: NONE active (SwapTotal=0)" in report.summary()
