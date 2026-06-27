"""Tests for the ACTIVE-task escalation augment in ``scripts/vm_disk_guard.py``
(task #679).

An ACTIVE task can hold a large re-downloadable cache the terminal-status gate
cannot reap. The guard must NEVER delete an active task's data — but a
band-worthy footprint is ESCALATED: a Telegram push + a row on the shared
``.claude/cache/disk-guard-events.jsonl`` sidecar naming the task, the largest
cache path, the footprint, and the SAFE reclaim command. Escalation is deduped
on (task, threshold-band), re-alerts only on >25% growth, and is suppressed by
a per-(task, band) ack sentinel.

Covers:
  * an active band-worthy cache triggers escalation (sidecar row + Telegram
    stub call) and NO deletion,
  * dedup: a second pass at the same band does NOT re-alert,
  * growth re-alert: a >25% grown cache DOES re-alert,
  * ack-sentinel suppression,
  * a small active cache (< 5 GB floor) is NOT escalated,
  * a terminal-status issue is reaped exactly as before (no escalation path).

The script is loaded via importlib like ``tests/test_vm_disk_guard.py``. The
Telegram push is redirected to a no-op stub via ``EPM_TELEGRAM_PUSH_SCRIPT``;
``clean_issue_downloads`` is stubbed to report a large footprint without
writing GBs to disk.
"""

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point both modules' ``repo_root`` at a temp dir (sidecar + state + ack
    sentinels all resolve under it) and an active task's data dir there."""
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def telegram_stub(tmp_path, monkeypatch):
    """A no-op executable stub the escalation's Telegram push invokes; records
    every call's args to a file so the test can assert it fired."""
    calls = tmp_path / "telegram_calls.txt"
    stub = tmp_path / "telegram_stub.sh"
    stub.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{calls}"\nexit 0\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(stub))
    return calls


def _make_active_cache(repo: Path, issue_n: int) -> Path:
    """A data/issue_<N>/ tree with hf_dl + store (sibling). Returns issue dir."""
    issue_dir = repo / "data" / f"issue_{issue_n}"
    (issue_dir / "hf_dl").mkdir(parents=True)
    (issue_dir / "hf_dl" / "blob.bin").write_bytes(b"x" * 1024)
    (issue_dir / "store").mkdir(parents=True)
    (issue_dir / "store" / "v0.pt").write_bytes(b"y" * 1024)
    return issue_dir


def _stub_large_cache(monkeypatch, *, bytes_freed: int, largest: str = "data/issue_700/hf_dl"):
    """Stub clean_issue_downloads to report a big footprint without writing GBs.
    Returns a CleanResult whose dry-run sizing the escalation reads."""
    cr = ced.CleanResult(issue_n=700, apply=False)
    cr.removed = [largest]
    cr.sizes_bytes = {largest: bytes_freed}
    monkeypatch.setattr(vdg, "clean_issue_downloads", lambda *a, **k: cr)


def _read_sidecar(repo: Path) -> list[dict]:
    path = repo / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ─── escalation fires for a band-worthy active cache ─────────────────────────


def test_active_band_worthy_cache_escalates_and_never_deletes(repo, telegram_stub, monkeypatch):
    """An active task holding a 60 GB re-downloadable cache is ESCALATED
    (sidecar row + Telegram push) and NOT deleted."""
    issue_dir = _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    _stub_large_cache(monkeypatch, bytes_freed=60 * 10**9)

    res = vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")

    # NOT deleted (active) — the real cache dir survives.
    assert (issue_dir / "hf_dl").exists()
    assert res.bytes_freed == 0  # escalation reclaims nothing

    rows = _read_sidecar(repo)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "active-cache-escalation"
    assert row["task"] == 700
    assert row["status"] == "running"
    assert row["bytes"] == 60 * 10**9
    assert row["band"] == 50.0  # 60 GB -> 50 GB band
    assert "clean_experiment_downloads.py 700 --apply" in row["reclaim_cmd"]

    # Telegram stub fired once.
    assert telegram_stub.is_file()
    assert "#700" in telegram_stub.read_text()


def test_active_escalation_dedups_same_band(repo, telegram_stub, monkeypatch):
    """A second pass at the SAME band (no growth) does NOT re-alert."""
    _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    _stub_large_cache(monkeypatch, bytes_freed=60 * 10**9)

    vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")
    vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")

    # Only the first pass escalated.
    assert len(_read_sidecar(repo)) == 1


def test_active_escalation_realerts_on_growth(repo, telegram_stub, monkeypatch):
    """A cache that grows >25% within the same band DOES re-alert."""
    _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")

    _stub_large_cache(monkeypatch, bytes_freed=60 * 10**9)
    vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")

    # +40% within the 50 GB band (60 -> 84 GB, still < 100 GB band).
    _stub_large_cache(monkeypatch, bytes_freed=84 * 10**9)
    vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")

    rows = _read_sidecar(repo)
    assert len(rows) == 2
    assert rows[1]["growth_pct"] == pytest.approx(40.0, abs=0.5)


def test_active_escalation_ack_sentinel_suppresses(repo, telegram_stub, monkeypatch):
    """An ack sentinel for (task, band) suppresses the escalation entirely."""
    _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    _stub_large_cache(monkeypatch, bytes_freed=60 * 10**9)

    ack = vdg._active_ack_sentinel_path(700, 50.0)
    ack.parent.mkdir(parents=True, exist_ok=True)
    ack.touch()

    vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")
    assert _read_sidecar(repo) == []


def test_small_active_cache_not_escalated(repo, telegram_stub, monkeypatch):
    """An active cache below the 5 GB floor is too small to escalate."""
    _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    _stub_large_cache(monkeypatch, bytes_freed=1 * 10**9)  # 1 GB < 5 GB floor

    vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")
    assert _read_sidecar(repo) == []


def test_terminal_issue_reaped_no_escalation(repo, telegram_stub, monkeypatch):
    """A terminal-status issue is reaped as before — the escalation path is
    not taken for it."""
    issue_dir = _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    # Do NOT stub clean_issue_downloads — let the real reap delete the tiny cache.

    res = vdg.clean_terminal_download_caches(apply=True, data_root=repo / "data")

    assert not (issue_dir / "hf_dl").exists()  # reaped
    assert (issue_dir / "store" / "v0.pt").exists()  # store kept
    assert _read_sidecar(repo) == []  # no escalation for a terminal issue
    assert res.bytes_freed > 0


def test_dry_run_escalation_reports_no_sidecar_write(repo, telegram_stub, monkeypatch):
    """In report-only (apply=False), the escalation is decided but neither the
    sidecar nor the Telegram push persists (observability is apply-gated)."""
    _make_active_cache(repo, 700)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")
    _stub_large_cache(monkeypatch, bytes_freed=60 * 10**9)

    res = vdg.clean_terminal_download_caches(apply=False, data_root=repo / "data")

    # The escalation detail line is present (it would alert), but nothing
    # persisted: no sidecar row, no telegram call, no state file.
    assert any("ESCALATED" in d for d in res.detail)
    assert _read_sidecar(repo) == []
    assert not telegram_stub.is_file()
    assert not vdg._active_escalation_state_path().is_file()


# ─── pure band helper ────────────────────────────────────────────────────────


def test_band_helper_buckets():
    assert vdg._active_escalation_band_gb(6 * 10**9) == 0.0  # below first band
    assert vdg._active_escalation_band_gb(25 * 10**9) == 20.0
    assert vdg._active_escalation_band_gb(60 * 10**9) == 50.0
    assert vdg._active_escalation_band_gb(120 * 10**9) == 120.0  # above top -> int-GB
