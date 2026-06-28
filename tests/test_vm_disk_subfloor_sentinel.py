"""Tests for the VM-disk SUB-FLOOR sentinel (task #679).

The watcher's existing alert/reclaim bands fire late (20 / 15 GiB). The
sub-floor sentinel is an EARLIER advisory band (~60 GB) that attributes the
disk pressure to the largest per-issue caches on the shared disk-guard sidecar
and signals a sooner re-check — warn-only, never deletes, no daemon. The tick
snapshot mirrors the same band labels so a cron tick surfaces the same signal.

Covers:
  * decide_subfloor pure logic (below-band first alert, dedup, drop re-alert),
  * subfloor_sentinel_pass writes a `band=sub-floor` sidecar row with top-cache
    attribution when below the band, dedups, and clears the episode on recovery,
  * dry-run writes nothing,
  * tick_triage.root_disk_band / root_disk_snapshot mirror the watcher labels.

Both modules are importable by name (the test bootstraps sys.path via
spawn_session, mirroring tests/test_autonomous_session_watch.py).
"""

import json
import sys
from pathlib import Path

import pytest

# Bootstrap sys.path the same way the watcher test does (spawn_session adds
# scripts/ to the path so the sibling scripts import by name).
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import autonomous_session_watch as asw  # noqa: E402
import tick_triage  # noqa: E402

# ─── decide_subfloor pure logic ──────────────────────────────────────────────


def test_decide_subfloor_above_band_is_false():
    above = asw.VM_DISK_SUBFLOOR_FREE_BYTES + 1
    assert asw.decide_subfloor(above, None) is False


def test_decide_subfloor_first_alert_below_band():
    below = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 1
    assert asw.decide_subfloor(below, None) is True


def test_decide_subfloor_dedups_at_stable_footprint():
    below = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 5 * 2**30
    # Same free as last alert -> no drop -> no re-alert.
    assert asw.decide_subfloor(below, below) is False


def test_decide_subfloor_realerts_on_large_drop():
    last = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 5 * 2**30
    now_free = int(last * 0.80)  # 20% drop > 10% threshold
    assert asw.decide_subfloor(now_free, last) is True


# ─── subfloor_sentinel_pass behavior ─────────────────────────────────────────


@pytest.fixture
def watcher_roots(tmp_path, monkeypatch):
    """Pin PROJECT_ROOT (sidecar + du attribution) and AUTONOMOUS_REGISTRY_DIR
    (dedup state) at a temp dir so the pass is fully offline."""
    monkeypatch.setattr(asw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    return tmp_path


def _make_cache(root: Path, issue_n: int, *, hf_kb: int) -> None:
    d = root / "data" / f"issue_{issue_n}" / "hf_dl"
    d.mkdir(parents=True)
    (d / "blob.bin").write_bytes(b"x" * hf_kb * 1024)


def _read_sidecar(root: Path) -> list[dict]:
    path = root / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_subfloor_writes_attributed_sidecar_row(watcher_roots):
    """Below the band, a band=sub-floor row is written naming the top caches."""
    _make_cache(watcher_roots, 700, hf_kb=300)
    _make_cache(watcher_roots, 701, hf_kb=100)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    wrote = asw.subfloor_sentinel_pass(dry_run=False, free_bytes=free)

    assert wrote is True
    rows = _read_sidecar(watcher_roots)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "vm-disk-subfloor"
    assert row["band"] == "sub-floor"
    assert row["recheck_sooner"] is True
    assert row["free_bytes"] == free
    paths = [e["path"] for e in row["top_cache_paths"]]
    # Largest cache first; both attributed.
    assert paths[0] == "data/issue_700/hf_dl"
    assert "data/issue_701/hf_dl" in paths


def test_subfloor_dedups_within_episode(watcher_roots):
    """A second pass at ~the same footprint does not write a second row."""
    _make_cache(watcher_roots, 700, hf_kb=50)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=free)
    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=free)

    assert len(_read_sidecar(watcher_roots)) == 1


def test_subfloor_clears_episode_on_recovery(watcher_roots):
    """Recovery above the band drops the dedup state so the next dip re-alerts."""
    _make_cache(watcher_roots, 700, hf_kb=50)
    low = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=low)
    # Recovered above the band -> episode cleared, no new row.
    asw.subfloor_sentinel_pass(
        dry_run=False, free_bytes=asw.VM_DISK_SUBFLOOR_FREE_BYTES + 5 * 2**30
    )
    assert not asw._subfloor_state_path().is_file()
    # A fresh dip re-alerts (episode reset).
    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=low)
    assert len(_read_sidecar(watcher_roots)) == 2


def test_subfloor_above_band_is_noop(watcher_roots):
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES + 5 * 2**30
    assert asw.subfloor_sentinel_pass(dry_run=False, free_bytes=free) is False
    assert _read_sidecar(watcher_roots) == []


def test_subfloor_dry_run_writes_nothing(watcher_roots):
    _make_cache(watcher_roots, 700, hf_kb=50)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    asw.subfloor_sentinel_pass(dry_run=True, free_bytes=free)
    assert _read_sidecar(watcher_roots) == []
    assert not asw._subfloor_state_path().is_file()


def test_subfloor_never_deletes(watcher_roots):
    """The sentinel is warn-only — the cache it attributes is never deleted."""
    _make_cache(watcher_roots, 700, hf_kb=50)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=free)
    assert (watcher_roots / "data" / "issue_700" / "hf_dl" / "blob.bin").exists()


# ─── tick_triage band mirror ─────────────────────────────────────────────────


def test_tick_root_disk_band_labels():
    assert tick_triage.root_disk_band(10 * 2**30) == "critical"
    assert tick_triage.root_disk_band(18 * 2**30) == "low"
    assert tick_triage.root_disk_band(40 * 2**30) == "sub-floor"
    assert tick_triage.root_disk_band(120 * 2**30) == "ok"


def test_tick_root_disk_snapshot_shape(monkeypatch):
    class _U:
        free = 40 * 2**30

    monkeypatch.setattr(tick_triage.shutil, "disk_usage", lambda _p: _U())
    snap = tick_triage.root_disk_snapshot()
    assert snap == {"band": "sub-floor", "free_gib": pytest.approx(40.0)}


def test_tick_root_disk_snapshot_none_on_error(monkeypatch):
    def _boom(_p):
        raise OSError("no such fs")

    monkeypatch.setattr(tick_triage.shutil, "disk_usage", _boom)
    assert tick_triage.root_disk_snapshot() is None
