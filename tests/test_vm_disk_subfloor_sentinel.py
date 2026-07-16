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
import subprocess
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


# ─── sub-floor RECLAIM arm (#1392) ───────────────────────────────────────────
#
# The reclaim arm launches a detached `vm_disk_guard.py --apply` while below
# the sub-floor, rate-limited on a PERSISTENT last_run_ts. A REAL launch from
# pytest would sweep the live VM, so every test here either stubs
# `_launch_guard_apply` (a recorder) or fakes at the subprocess.Popen boundary
# (the argv durability pin); the tests/conftest.py #1392 autouse guard
# (kill switch default-on + fail-loud launcher wrap) is the backstop.


NOW = 1_700_000_000.0


@pytest.fixture
def reclaim_env(watcher_roots, monkeypatch):
    """watcher_roots + the reclaim arm ENABLED (the conftest autouse guard
    sets the kill switch by default; arm tests opt back in)."""
    monkeypatch.delenv("EPM_DISABLE_SUBFLOOR_RECLAIM", raising=False)
    return watcher_roots


def _stub_launcher_recorder(monkeypatch, pid: int = 4242) -> list:
    calls: list = []
    monkeypatch.setattr(
        asw, "_launch_guard_apply", lambda log_path: (calls.append(log_path), pid)[1]
    )
    return calls


def _stub_launcher_forbidden(monkeypatch) -> None:
    def _boom(log_path):
        pytest.fail(f"_launch_guard_apply must not be called (got {log_path})")

    monkeypatch.setattr(asw, "_launch_guard_apply", _boom)


# — decide_subfloor_reclaim pure logic —


def test_decide_subfloor_reclaim_first_fire_below_floor():
    below = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 1
    assert asw.decide_subfloor_reclaim(below, None, NOW) is True


def test_decide_subfloor_reclaim_holds_within_interval():
    below = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 1
    assert asw.decide_subfloor_reclaim(below, NOW - 600.0, NOW) is False


def test_decide_subfloor_reclaim_refires_after_interval():
    below = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 1
    last = NOW - (asw.VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S + 1.0)
    assert asw.decide_subfloor_reclaim(below, last, NOW) is True


def test_decide_subfloor_reclaim_above_floor_never_fires():
    above = asw.VM_DISK_SUBFLOOR_FREE_BYTES + 1
    stale = NOW - (10 * asw.VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S)
    assert asw.decide_subfloor_reclaim(above, stale, NOW) is False


# — subfloor_reclaim_pass behavior —


def test_subfloor_reclaim_pass_dry_run_zero_subprocess(reclaim_env, monkeypatch, capsys):
    """Dry-run prints the would-launch argv and performs zero launches and
    zero writes (the #681 r3 zero-side-effect smoke contract)."""
    _stub_launcher_forbidden(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=True, free_bytes=free, now=NOW) is True

    assert "would launch:" in capsys.readouterr().out
    assert not asw._subfloor_reclaim_state_path().is_file()
    assert _read_sidecar(reclaim_env) == []


def test_subfloor_reclaim_pass_launches_detached_and_records_state(reclaim_env, monkeypatch):
    calls = _stub_launcher_recorder(monkeypatch, pid=4242)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is True

    assert len(calls) == 1
    state = json.loads(asw._subfloor_reclaim_state_path().read_text())
    assert state == {"last_run_ts": NOW, "pid": 4242}
    rows = [r for r in _read_sidecar(reclaim_env) if r["kind"] == "vm-disk-subfloor-reclaim"]
    assert len(rows) == 1
    row = rows[0]
    assert row["band"] == "sub-floor"
    assert row["action"] == "guard-apply-launched"
    assert row["pid"] == 4242
    assert row["free_bytes"] == free
    assert row["interval_s"] == asw.VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S
    assert row["log"].endswith(".log")


def test_subfloor_reclaim_pass_rate_limits_within_interval(reclaim_env, monkeypatch):
    """A second tick within the interval does NOT relaunch; past it, it does."""
    calls = _stub_launcher_recorder(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is True
    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW + 600.0) is False
    later = NOW + asw.VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S + 1.0
    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=later) is True
    assert len(calls) == 2


def test_subfloor_reclaim_state_persists_across_recovery(reclaim_env, monkeypatch):
    """The flap guard: recovery above the floor does NOT clear last_run_ts, so
    a fresh dip inside the interval stays rate-limited (plan §11 item 7)."""
    calls = _stub_launcher_recorder(monkeypatch)
    low = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    high = asw.VM_DISK_SUBFLOOR_FREE_BYTES + 10 * 2**30

    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=low, now=NOW)
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=high, now=NOW + 300.0)  # recovery
    assert asw._subfloor_reclaim_state_path().is_file()  # never cleared
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=low, now=NOW + 600.0)  # re-dip
    assert len(calls) == 1  # still rate-limited across the flap


def test_subfloor_reclaim_pass_kill_switch(reclaim_env, monkeypatch):
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    _stub_launcher_forbidden(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is False


def test_subfloor_reclaim_pass_failsoft_on_launch_error(reclaim_env, monkeypatch, capsys):
    def _boom(log_path):
        raise OSError("no such interpreter")

    monkeypatch.setattr(asw, "_launch_guard_apply", _boom)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is False

    assert "fail-soft" in capsys.readouterr().err
    assert not asw._subfloor_reclaim_state_path().is_file()
    assert _read_sidecar(reclaim_env) == []


def test_subfloor_reclaim_pass_failsoft_on_state_save_error(reclaim_env, monkeypatch):
    """#1392 concern 1: an ENOSPC in the state save (100%-full /) returns
    False cleanly — the fail-soft try/except wraps the save too."""
    _stub_launcher_recorder(monkeypatch)

    def _enospc(payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(asw, "_save_subfloor_reclaim_state", _enospc)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is False


def test_subfloor_reclaim_launch_argv_vm_root_only(reclaim_env, monkeypatch, tmp_path):
    """Durability pin (#1392): the launch argv is exactly
    [sys.executable, <repo>/scripts/vm_disk_guard.py, --apply,
    --ignore-threshold, --no-push, --no-data-disk] with
    start_new_session=True, and dry-run launches nothing. Executes the REAL
    _launch_guard_apply body with the fake at the subprocess.Popen boundary
    (autospec — signature-conformant by construction)."""
    import types
    from unittest import mock

    fake_popen = mock.create_autospec(subprocess.Popen)
    fake_popen.return_value = types.SimpleNamespace(pid=4242)
    monkeypatch.setattr(asw.subprocess, "Popen", fake_popen)

    log_path = reclaim_env / "logs" / "vm_disk_guard" / "2026-07-16.log"
    pid = asw._launch_guard_apply(log_path)

    assert pid == 4242
    assert fake_popen.call_count == 1
    argv = fake_popen.call_args.args[0]
    assert argv == [
        sys.executable,
        str(asw.PROJECT_ROOT / "scripts" / "vm_disk_guard.py"),
        "--apply",
        "--ignore-threshold",
        "--no-push",
        "--no-data-disk",
    ]
    assert fake_popen.call_args.kwargs["start_new_session"] is True
    assert fake_popen.call_args.kwargs["cwd"] == asw.PROJECT_ROOT
    # The launch header lands in the cron's dated log (one audit trail).
    assert "trigger=watch-subfloor pid=4242" in log_path.read_text()

    # Dry-run performs ZERO Popen calls for this arm.
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    asw.subfloor_reclaim_pass(dry_run=True, free_bytes=free, now=NOW)
    assert fake_popen.call_count == 1  # unchanged


def test_sentinel_row_unchanged_by_reclaim_arm(reclaim_env, monkeypatch):
    """Constraint 5 (#1392): the existing band=sub-floor sentinel row keeps
    its exact pre-change field set; the reclaim arm adds a SEPARATE row."""
    _stub_launcher_recorder(monkeypatch)
    _make_cache(reclaim_env, 700, hf_kb=50)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=free)
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW)

    rows = _read_sidecar(reclaim_env)
    sentinel = [r for r in rows if r["kind"] == "vm-disk-subfloor"]
    reclaim = [r for r in rows if r["kind"] == "vm-disk-subfloor-reclaim"]
    assert len(sentinel) == 1 and len(reclaim) == 1
    assert set(sentinel[0]) == {
        "ts",
        "kind",
        "band",
        "free_bytes",
        "free_gib",
        "top_cache_paths",
        "recheck_sooner",
    }


def test_vm_disk_pass_wires_subfloor_reclaim(reclaim_env, monkeypatch):
    """Call-site wiring seam (#1392 Phase-2 Statistics Must-Fix a): the arm
    fires from vm_disk_pass itself — all-helpers-green-but-unwired would be a
    silent no-op indistinguishable from pre-fix behavior."""
    calls = _stub_launcher_recorder(monkeypatch)
    # 40 GiB free: below the 60 GiB sub-floor, above the 20 GiB alert band —
    # the pass returns on the "ok" branch right after the two sub-floor arms.
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 40 * 2**30)

    asw.vm_disk_pass(dry_run=False, now=NOW)

    assert len(calls) == 1


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
