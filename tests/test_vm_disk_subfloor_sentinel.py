"""Tests for the VM-disk sub-floor sentinel (task #679).

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


# ─── #2141 skip diagnostics (below-floor declined ticks are loud) ────────────
#
# The reclaim arm used to decline SILENTLY, so a genuine non-fire was
# indistinguishable from a fire whose log line the reader failed to match —
# which is exactly how #2141 came to be filed against a WORKING arm (grep
# 'subfloor' = 0 on a log carrying 4 'SUB-FLOOR' fire lines). These tests pin
# the new per-skip stderr lines, the kill-switch-only throttled sidecar rows,
# the lowercase-token standardization, the merge-write state contract, the
# ok-line band note, and the D1.5 conftest sidecar-hermeticity guard.


def test_subfloor_reclaim_skip_kill_switch_below_floor_is_loud(watcher_roots, monkeypatch, capsys):
    """The below-floor declined fixture the body asked for: a kill-switched
    below-floor tick is positively observable — a stderr line naming the
    predicate plus exactly one durable action="skipped" sidecar row."""
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    _stub_launcher_forbidden(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is False

    assert "subfloor-reclaim: skipped (kill-switch" in capsys.readouterr().err
    rows = [r for r in _read_sidecar(watcher_roots) if r["kind"] == "vm-disk-subfloor-reclaim"]
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "skipped"
    assert row["reason"] == "kill-switch"
    assert row["band"] == "sub-floor"
    assert row["free_bytes"] == free
    assert row["interval_s"] == asw.VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S


def test_subfloor_reclaim_skip_kill_switch_above_floor_stays_silent(
    watcher_roots, monkeypatch, capsys
):
    """The volume-scoping rule: above the floor a kill-switched decline is the
    healthy no-op (~144 ticks/day) and stays byte-identical — no stderr, no
    sidecar row."""
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    _stub_launcher_forbidden(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES + 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is False

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
    assert _read_sidecar(watcher_roots) == []


def test_subfloor_reclaim_skip_rows_throttled(watcher_roots, monkeypatch, capsys):
    """Kill-switched below-floor ticks: a stderr line EVERY tick, sidecar rows
    throttled to <=1 per interval, and the skip-state save is a MERGE that
    never clobbers last_run_ts (the rate-limit key)."""
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    _stub_launcher_forbidden(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    # Seed prior fire state: the skip-state merge must leave it untouched.
    asw._save_subfloor_reclaim_state({"last_run_ts": NOW - 50.0, "pid": 111})
    interval = asw.VM_DISK_SUBFLOOR_RECLAIM_INTERVAL_S

    for t in (NOW, NOW + 600.0, NOW + interval + 1.0):
        assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=t) is False

    assert capsys.readouterr().err.count("skipped (kill-switch") == 3
    rows = [r for r in _read_sidecar(watcher_roots) if r.get("action") == "skipped"]
    assert len(rows) == 2  # NOW and NOW+interval+1; NOW+600 throttled
    state = json.loads(asw._subfloor_reclaim_state_path().read_text())
    assert state["last_run_ts"] == NOW - 50.0  # the merge pin: untouched
    assert state["pid"] == 111
    assert state["last_skip_row_ts"] == NOW + interval + 1.0


def test_subfloor_reclaim_skip_rate_limited_stderr_only(reclaim_env, monkeypatch, capsys):
    """A rate-limited below-floor decline gets a stderr line carrying the
    age= field (a NEGATIVE age is the corrupt/future-last_run_ts wedge
    signature) but NO sidecar row — the <=1800 s-old fire row in the same
    sidecar is already the positive evidence."""
    calls = _stub_launcher_recorder(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is True  # fire
    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW + 600.0) is False

    err = capsys.readouterr().err
    assert "skipped (rate-limited" in err
    assert "age=600" in err
    rows = _read_sidecar(reclaim_env)
    assert [r for r in rows if r.get("action") == "skipped"] == []
    assert len([r for r in rows if r.get("action") == "guard-apply-launched"]) == 1
    assert len(calls) == 1


def test_subfloor_reclaim_skip_free_unreadable_stderr_only(reclaim_env, monkeypatch, capsys):
    """free-unreadable: one stderr line naming the predicate; no rows, no
    state writes (the band condition is unevaluable — production never
    reaches this, vm_disk_pass early-returns on free is None upstream)."""
    _stub_launcher_forbidden(monkeypatch)
    monkeypatch.setattr(asw, "_root_disk_headroom", lambda: None)

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=None, now=NOW) is False

    assert "skipped (free-unreadable" in capsys.readouterr().err
    assert _read_sidecar(reclaim_env) == []
    assert not asw._subfloor_reclaim_state_path().is_file()


def test_subfloor_reclaim_skip_dry_run_zero_writes(watcher_roots, monkeypatch, capsys):
    """The #681 r3 dry-run contract extended to the skip path: stderr line +
    the [dry-run] would-append print only — NO sidecar file, NO state file."""
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    _stub_launcher_forbidden(monkeypatch)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    assert asw.subfloor_reclaim_pass(dry_run=True, free_bytes=free, now=NOW) is False

    captured = capsys.readouterr()
    assert "subfloor-reclaim: skipped (kill-switch" in captured.err
    assert "[dry-run] would append" in captured.out
    assert not (watcher_roots / ".claude" / "cache" / "disk-guard-events.jsonl").is_file()
    assert not asw._subfloor_reclaim_state_path().is_file()


def test_subfloor_lines_carry_lowercase_token(reclaim_env, monkeypatch, capsys):
    """The #2141 grep-token fix: every emitted subfloor-family line carries
    the literal lowercase `subfloor`, and the uppercase token is gone from
    the module SOURCE — closing the whole class (the data-disk print site
    included) without a mounted-data-disk fixture."""
    _stub_launcher_recorder(monkeypatch)
    below = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30

    asw.subfloor_sentinel_pass(dry_run=False, free_bytes=below)  # sentinel fire
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=below, now=NOW)  # reclaim fire
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=below, now=NOW + 1.0)  # rate-limited
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=below, now=NOW + 2.0)  # kill-switch
    monkeypatch.setattr(asw, "_root_disk_headroom", lambda: None)
    asw.subfloor_reclaim_pass(dry_run=False, free_bytes=None, now=NOW + 3.0)  # free-unreadable

    lines = [ln for ln in capsys.readouterr().err.splitlines() if "vm-disk" in ln]
    assert len(lines) >= 5  # fire x2 + skip x3
    for ln in lines:
        assert "subfloor" in ln, ln
    assert "SUB-FLOOR" not in Path(asw.__file__).read_text()


def test_vm_disk_ok_line_names_subfloor_band(watcher_roots, monkeypatch, capsys):
    """D3: the per-tick ok line names the sub-floor band when free < 60 GiB —
    `vm-disk: ok` beside a live sub-floor episode read as a contradiction in
    the #2141 false report (it reports the 20 GiB ALERT band, not the 60 GiB
    sub-floor band). Above the band the line is byte-identical to before."""
    # 40 GiB free: below the 60 GiB sub-floor, above the 20 GiB alert band.
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 40 * 2**30)
    asw.vm_disk_pass(dry_run=True, now=NOW)
    assert "vm-disk: ok (40.0 GiB free); sub-floor band" in capsys.readouterr().out

    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 120 * 2**30)
    asw.vm_disk_pass(dry_run=True, now=NOW)
    out_high = capsys.readouterr().out
    assert "vm-disk: ok (120.0 GiB free)\n" in out_high
    assert "sub-floor band" not in out_high


def test_fire_row_and_state_shape_unchanged(reclaim_env, monkeypatch):
    """After a fire with prior skip state present: the fire row keeps its
    exact #1392 field set, and the fire-path state save is a MERGE — it
    updates last_run_ts + pid while PRESERVING last_skip_row_ts (a
    whole-payload overwrite would clobber it)."""
    _stub_launcher_recorder(monkeypatch, pid=777)
    free = asw.VM_DISK_SUBFLOOR_FREE_BYTES - 10 * 2**30
    asw._save_subfloor_reclaim_state({"last_skip_row_ts": NOW - 900.0})

    assert asw.subfloor_reclaim_pass(dry_run=False, free_bytes=free, now=NOW) is True

    rows = [r for r in _read_sidecar(reclaim_env) if r["kind"] == "vm-disk-subfloor-reclaim"]
    assert len(rows) == 1
    assert set(rows[0]) == {
        "ts",
        "kind",
        "band",
        "action",
        "free_bytes",
        "free_gib",
        "pid",
        "interval_s",
        "log",
    }
    state = json.loads(asw._subfloor_reclaim_state_path().read_text())
    assert state == {"last_skip_row_ts": NOW - 900.0, "last_run_ts": NOW, "pid": 777}


def test_sidecar_path_hermetic_without_project_root_pin(tmp_path, monkeypatch):
    """The D1.5 pin: with NO PROJECT_ROOT pin the resolver must NOT point at
    the real repo root (the conftest autouse guard redirects it to pytest
    tmp); with a pinned PROJECT_ROOT it must DELEGATE to the real resolver
    (so root-pinned sidecar-content assertions keep working)."""
    real_root = asw.PROJECT_ROOT
    unpinned = asw._disk_guard_sidecar_path()
    assert not str(unpinned).startswith(str(real_root))
    assert str(unpinned).startswith(str(tmp_path))

    other_tmp = tmp_path / "other"
    monkeypatch.setattr(asw, "PROJECT_ROOT", other_tmp)
    expected = other_tmp / ".claude" / "cache" / "disk-guard-events.jsonl"
    assert asw._disk_guard_sidecar_path() == expected


def test_kill_switch_skip_from_vm_disk_pass_is_hermetic(tmp_path, monkeypatch):
    """The exact Finding-1 planting shape end-to-end: kill switch set
    (conftest default), 17 GiB free, AUTONOMOUS_REGISTRY_DIR pinned,
    PROJECT_ROOT NOT pinned, real-body vm_disk_pass(dry_run=False) — the
    REAL shared sidecar gains ZERO bytes while the tmp-redirected sidecar
    carries the sentinel row AND the new kill-switch skip row."""
    real_sidecar = asw.PROJECT_ROOT / ".claude" / "cache" / "disk-guard-events.jsonl"
    size_before = real_sidecar.stat().st_size if real_sidecar.is_file() else 0

    monkeypatch.setattr(asw, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "reg")
    monkeypatch.setattr(asw, "_vm_free_bytes", lambda: 17 * 2**30)  # low band, sub-floor active
    monkeypatch.setattr(asw, "_top_issue_cache_paths", lambda dry_run: [])
    monkeypatch.setattr(asw, "_vm_remediate_worktrees", lambda dry_run: "worktree-audit rc=0: ok")
    monkeypatch.setattr(asw, "_post_progress_marker", lambda *a, **k: None)

    asw.vm_disk_pass(dry_run=False, now=NOW)

    size_after = real_sidecar.stat().st_size if real_sidecar.is_file() else 0
    assert size_after == size_before  # ZERO rows into the real audit stream
    redirected = tmp_path / "disk-guard-events.jsonl"
    rows = [json.loads(ln) for ln in redirected.read_text().splitlines() if ln.strip()]
    kinds_actions = {(r["kind"], r.get("action")) for r in rows}
    assert ("vm-disk-subfloor", None) in kinds_actions  # sentinel row, redirected
    assert ("vm-disk-subfloor-reclaim", "skipped") in kinds_actions  # the new skip row
