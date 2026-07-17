"""Offline unit tests for the #1457 device-vs-filesystem size reconciliation
in the VM disk guard (scripts/vm_disk_guard.py).

The incident: the VM boot disk was resized to 1 TB in GCP but neither growpart
nor resize2fs ran — df showed 485G while ~500 GB sat unused on the device, and
the guard fired cleanup tiers instead of surfacing the cheapest lever. The
check compares the TOP-LEVEL block device size (sysfs) against the mounted
filesystem size (statvfs) and WARNs when the device exceeds the fs by more
than BOTH the percent threshold AND the absolute floor.

Hard constraints pinned here: surfacing ONLY (no subprocess / no mutating
block-fs command in the new code paths), fail-soft everywhere (every failure
returns state="skipped", never raises), exit codes unchanged, sidecar row on
every warn run (--no-push gates only the push+sentinel leg).

Loaded via importlib exactly like tests/test_vm_disk_guard.py.
"""

import dataclasses
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── measured geometry (this VM, 2026-07-17; incident fs from the task body) ──

INCIDENT_DEVICE_BYTES = 1_046_898_278_400  # sda: 2,044,723,200 sectors x 512
INCIDENT_FS_BYTES = 520_758_304_768  # the incident's 485G df reading
HEALTHY_BOOT_FS_BYTES = 1_014_426_468_352  # today's / statvfs (gap 3.10%)
DATA_DEVICE_BYTES = 1_099_511_627_776  # sdb: exactly 1 TiB
DATA_FS_BYTES = 1_081_106_157_568  # today's /mnt/eps-data statvfs (gap 1.67%)


@pytest.fixture(autouse=True)
def _clean_devfs_env(monkeypatch):
    """Isolate every test from ambient #1457 env knobs."""
    for var in (
        "EPS_VM_DEVICE_FS_GAP_PCT",
        "EPS_VM_DEVICE_FS_GAP_MIN_GB",
        "EPM_SKIP_DEVICE_FS_CHECK",
    ):
        monkeypatch.delenv(var, raising=False)


def _check(device_bytes, fs_bytes, **kw):
    """check_device_fs_gap against a fake resolver + fake statvfs (f_frsize=1)."""
    return vdg.check_device_fs_gap(
        "/fake",
        resolver=lambda p: (device_bytes, "sda", "sda1", "1"),
        statvfs_fn=lambda p: SimpleNamespace(f_blocks=fs_bytes, f_frsize=1),
        **kw,
    )


def _warn_chk(device_bytes=INCIDENT_DEVICE_BYTES, fs_bytes=INCIDENT_FS_BYTES):
    """A pre-built warn-state DeviceFsCheck for the alert-helper tests."""
    return vdg.DeviceFsCheck(
        path="/",
        state="warn",
        device="sda",
        partition="sda1",
        partition_number="1",
        device_bytes=device_bytes,
        fs_bytes=fs_bytes,
        gap_bytes=device_bytes - fs_bytes,
        gap_pct=round(100.0 * (device_bytes - fs_bytes) / device_bytes, 2),
        threshold_pct=5.0,
        min_gap_bytes=10 * 1024**3,
    )


def _sentinels(root: Path) -> list[Path]:
    cache = root / ".claude" / "cache"
    return sorted(cache.glob("disk-guard-devfs-ack-*")) if cache.is_dir() else []


# ─── 1-3: incident replay fires; today's healthy state is quiet ───────────────


def test_incident_replay_1tb_device_485g_fs_warns():
    """The check's own motivating incident (485G fs on the ~1TB device) WARNs
    at ~50.3% gap — the durability pin for this task."""
    res = _check(INCIDENT_DEVICE_BYTES, INCIDENT_FS_BYTES)
    assert res.state == "warn"
    assert 50.0 < res.gap_pct < 51.0
    assert res.device == "sda" and res.partition == "sda1" and res.partition_number == "1"
    assert res.gap_bytes == INCIDENT_DEVICE_BYTES - INCIDENT_FS_BYTES


def test_healthy_boot_disk_today_is_quiet():
    """Today's real / geometry (3.10% ext4-overhead gap) stays ok."""
    res = _check(INCIDENT_DEVICE_BYTES, HEALTHY_BOOT_FS_BYTES)
    assert res.state == "ok"
    assert abs(res.gap_pct - 3.10) < 0.05


def test_healthy_data_disk_today_is_quiet():
    """Today's real /mnt/eps-data geometry (1.67% gap) stays ok."""
    res = _check(DATA_DEVICE_BYTES, DATA_FS_BYTES)
    assert res.state == "ok"
    assert abs(res.gap_pct - 1.67) < 0.05


# ─── 4-7: sysfs resolver (fake trees) ─────────────────────────────────────────


def _fake_sysfs(root: Path):
    """A fake sysfs: partition dir (with `partition` + smaller `size`) whose
    PARENT disk dir carries the bigger `size`; dev_block/8:1 symlinks to the
    partition (the realpath the kernel gives for a mounted partition)."""
    disk = root / "devices" / "pci0000:00" / "block" / "sda"
    part = disk / "sda1"
    part.mkdir(parents=True)
    (part / "partition").write_text("1\n")
    (part / "size").write_text(f"{100 * 2**21}\n")  # 100 GiB partition
    (disk / "size").write_text(f"{200 * 2**21}\n")  # 200 GiB parent disk (BIGGER)
    dev_root = root / "dev_block"
    dev_root.mkdir()
    (dev_root / "8:1").symlink_to(part)
    return dev_root


def test_partition_resolves_to_parent_disk(tmp_path):
    """The comparison target is the PARENT disk's size, not the partition's —
    the un-growpart-ed-partition case the incident hinged on."""
    dev_root = _fake_sysfs(tmp_path)
    stat_fn = lambda p: SimpleNamespace(st_dev=os.makedev(8, 1))  # noqa: E731
    got = vdg._device_bytes_for_mount("/fake", sysfs_dev_root=str(dev_root), stat_fn=stat_fn)
    assert got == (200 * 2**30, "sda", "sda1", "1")


def test_wholedisk_fs_no_partition_file(tmp_path):
    """A whole-disk filesystem (the sdb layout: no `partition` file) uses its
    own size and reports empty partition fields."""
    disk = tmp_path / "devices" / "pci0000:00" / "block" / "sdb"
    disk.mkdir(parents=True)
    (disk / "size").write_text(f"{1024 * 2**21}\n")  # 1 TiB
    dev_root = tmp_path / "dev_block"
    dev_root.mkdir()
    (dev_root / "8:16").symlink_to(disk)
    stat_fn = lambda p: SimpleNamespace(st_dev=os.makedev(8, 16))  # noqa: E731
    got = vdg._device_bytes_for_mount("/fake", sysfs_dev_root=str(dev_root), stat_fn=stat_fn)
    assert got == (1024 * 2**30, "sdb", "", "")


def test_sysfs_unresolvable_skips_fail_soft(tmp_path):
    """Every resolution failure degrades to state='skipped' + a reason — the
    check NEVER raises (the cron guard is fleet-critical)."""

    def _boom(p):
        raise OSError("boom")

    # (a) stat itself fails
    got = vdg._device_bytes_for_mount("/fake", sysfs_dev_root=str(tmp_path), stat_fn=_boom)
    assert got[0] is None and "boom" in got[1]
    # (b) no sysfs node for the device numbers
    stat_fn = lambda p: SimpleNamespace(st_dev=os.makedev(8, 1))  # noqa: E731
    got = vdg._device_bytes_for_mount("/fake", sysfs_dev_root=str(tmp_path), stat_fn=stat_fn)
    assert got[0] is None and "no sysfs node" in got[1]
    # (c) anonymous major-0 device (overlay / tmpfs / btrfs)
    anon = lambda p: SimpleNamespace(st_dev=os.makedev(0, 45))  # noqa: E731
    got = vdg._device_bytes_for_mount("/fake", sysfs_dev_root=str(tmp_path), stat_fn=anon)
    assert got[0] is None and "anonymous device" in got[1]
    # A skipping resolver surfaces its reason through check_device_fs_gap...
    res = vdg.check_device_fs_gap("/fake", resolver=lambda p: (None, "why not", None, None))
    assert res.state == "skipped" and res.reason == "why not"

    # ...a statvfs failure skips too...
    def _sv_boom(p):
        raise OSError("statvfs down")

    res = vdg.check_device_fs_gap(
        "/fake", resolver=lambda p: (10 * 2**30, "sda", "", ""), statvfs_fn=_sv_boom
    )
    assert res.state == "skipped" and "statvfs" in res.reason
    # ...and even an unexpected resolver crash is caught (never raises).
    res = vdg.check_device_fs_gap("/fake", resolver=lambda p: 1 / 0)
    assert res.state == "skipped" and res.reason


def test_virtual_device_skips(tmp_path):
    """dm/loop/md/zram resolve under /devices/virtual/ — deliberately skipped
    (an LV smaller than its VG is a legitimate layout, not a resize gap)."""
    dm = tmp_path / "devices" / "virtual" / "block" / "dm-0"
    dm.mkdir(parents=True)
    (dm / "size").write_text(f"{50 * 2**21}\n")
    dev_root = tmp_path / "dev_block"
    dev_root.mkdir()
    (dev_root / "253:0").symlink_to(dm)
    stat_fn = lambda p: SimpleNamespace(st_dev=os.makedev(253, 0))  # noqa: E731
    got = vdg._device_bytes_for_mount("/fake", sysfs_dev_root=str(dev_root), stat_fn=stat_fn)
    assert got[0] is None and "virtual block device (dm-0)" in got[1]


# ─── 8 + 15: thresholds ───────────────────────────────────────────────────────


def test_small_disk_absolute_floor_suppresses():
    """A 5 GiB device with a 4.4 GiB fs is a 12% gap but only ~0.6 GiB — the
    10 GiB absolute floor keeps small-disk fixed mkfs overhead quiet."""
    res = _check(5 * 2**30, int(4.4 * 2**30))
    assert res.state == "ok"
    assert res.gap_pct > 10.0  # the percent leg alone WOULD fire


@pytest.mark.parametrize(
    ("device_bytes", "fs_bytes"),
    [
        # gap_pct == exactly 5.0 (floor cleared: 50 GiB gap) -> quiet
        (1000 * 2**30, 950 * 2**30),
        # gap == exactly 10 GiB (percent cleared: 10% gap) -> quiet
        (100 * 2**30, 90 * 2**30),
    ],
)
def test_exact_boundary_quiet(device_bytes, fs_bytes):
    """Strict > on BOTH legs: exactly-at-threshold stays quiet (pins against a
    future >= drift on either leg)."""
    res = _check(device_bytes, fs_bytes)
    assert res.state == "ok"


# ─── 9, 10, 17: push + sidecar + dedup ────────────────────────────────────────


def _patch_alert_seams(monkeypatch, tmp_path):
    """Patch repo_root + push + sidecar with recorders; return the recorders."""
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    pushes: list[tuple[str, bool]] = []

    def fake_push(msg, apply):
        pushes.append((msg, apply))
        return True

    monkeypatch.setattr(vdg, "_telegram_push", fake_push)
    sidecar: list[tuple[dict, bool]] = []
    monkeypatch.setattr(
        vdg, "append_disk_guard_event", lambda ev, apply: sidecar.append((ev, apply))
    )
    return pushes, sidecar


def test_push_dedup_once_per_resize(monkeypatch, tmp_path):
    """One push per (device, device-size) resize event; a DIFFERENT device
    size (a new resize) pushes again; the sidecar row rides EVERY warn run."""
    pushes, sidecar = _patch_alert_seams(monkeypatch, tmp_path)
    chk = _warn_chk()
    vdg._maybe_alert_device_fs(chk, True, no_push=False)
    vdg._maybe_alert_device_fs(chk, True, no_push=False)
    assert len(pushes) == 1
    assert len(_sentinels(tmp_path)) == 1
    chk2 = dataclasses.replace(chk, device_bytes=2 * INCIDENT_DEVICE_BYTES)
    vdg._maybe_alert_device_fs(chk2, True, no_push=False)
    assert len(pushes) == 2
    assert len(_sentinels(tmp_path)) == 2
    assert len(sidecar) == 3  # deduplication-free observability
    assert all(ev["kind"] == "device-fs-gap" for ev, _ in sidecar)
    # ok/skipped states never alert at all
    vdg._maybe_alert_device_fs(dataclasses.replace(chk, state="ok"), True, no_push=False)
    vdg._maybe_alert_device_fs(None, True, no_push=False)
    assert len(pushes) == 2 and len(sidecar) == 3


def test_report_only_no_sentinel_no_push(monkeypatch, tmp_path):
    """apply=False keeps the guard's report-only contract: the push helper is
    invoked with apply=False (the real one demotes to stderr), the sidecar
    helper gets apply=False (report-only prints, persists nothing), and NO
    sentinel is written."""
    pushes, sidecar = _patch_alert_seams(monkeypatch, tmp_path)
    vdg._maybe_alert_device_fs(_warn_chk(), False, no_push=False)
    assert [apply for _, apply in pushes] == [False]
    assert [apply for _, apply in sidecar] == [False]
    assert _sentinels(tmp_path) == []


def test_no_push_warn_still_writes_sidecar_row(monkeypatch, tmp_path):
    """--no-push gates ONLY the push+sentinel leg: the sidecar row still rides
    the warn run (the watcher's --no-push sub-floor launches keep the
    observability row)."""
    pushes, sidecar = _patch_alert_seams(monkeypatch, tmp_path)
    vdg._maybe_alert_device_fs(_warn_chk(), True, no_push=True)
    assert len(sidecar) == 1
    assert pushes == []
    assert _sentinels(tmp_path) == []


# ─── 11-13: run_guard / JSON / env / exit code ────────────────────────────────


def test_run_guard_carries_check_even_under_threshold(monkeypatch):
    """The check is computed BEFORE the under-threshold early return (a fresh
    resize is detectable long before disk pressure) and rides _result_json."""
    fake = vdg.DeviceFsCheck(path="/", state="ok", device="sda", gap_pct=3.1)
    monkeypatch.setattr(vdg, "check_device_fs_gap", lambda path: fake)
    monkeypatch.setattr(vdg, "disk_used_pct", lambda path="/": 10.0)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 500.0)
    res = vdg.run_guard(False, threshold=99.0)
    assert res.triggered is False
    assert res.device_fs is fake
    payload = vdg._result_json(res)
    assert payload["device_fs"]["state"] == "ok"
    assert payload["device_fs"]["device"] == "sda"


def test_env_overrides_and_kill_switch(monkeypatch):
    """EPS_VM_DEVICE_FS_GAP_PCT raises the bar (incident demotes to ok);
    EPM_SKIP_DEVICE_FS_CHECK=1 short-circuits to skipped."""
    monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_PCT", "60")
    assert _check(INCIDENT_DEVICE_BYTES, INCIDENT_FS_BYTES).state == "ok"
    monkeypatch.delenv("EPS_VM_DEVICE_FS_GAP_PCT")
    monkeypatch.setenv("EPM_SKIP_DEVICE_FS_CHECK", "1")
    res = _check(INCIDENT_DEVICE_BYTES, INCIDENT_FS_BYTES)
    assert res.state == "skipped"
    assert "EPM_SKIP_DEVICE_FS_CHECK" in res.reason
    # malformed env values fall back to the defaults (never raise)
    monkeypatch.delenv("EPM_SKIP_DEVICE_FS_CHECK")
    monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_PCT", "not-a-number")
    monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_MIN_GB", "-3")
    assert vdg.device_fs_gap_pct() == vdg.DEFAULT_DEVICE_FS_GAP_PCT
    assert vdg.device_fs_gap_min_gb() == vdg.DEFAULT_DEVICE_FS_GAP_MIN_GB


def test_nonfinite_min_gb_env_never_raises(monkeypatch):
    """EPS_VM_DEVICE_FS_GAP_MIN_GB=inf/nan must NOT raise out of
    check_device_fs_gap (pre-fix: int(inf * 1024**3) raised OverflowError past
    the env accessor's >= 0 clamp, violating the never-raises contract —
    round-1 concern devfs-nonfinite-env-overflow-escapes-failsoft). The
    accessor clamps non-finite values to the default; the incident geometry
    then clears the default 10 GiB floor and warns."""
    default_floor = int(vdg.DEFAULT_DEVICE_FS_GAP_MIN_GB * 1024**3)
    for raw in ("inf", "nan", "-inf"):
        monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_MIN_GB", raw)
        assert vdg.device_fs_gap_min_gb() == vdg.DEFAULT_DEVICE_FS_GAP_MIN_GB
        chk = _check(INCIDENT_DEVICE_BYTES, INCIDENT_FS_BYTES)
        assert chk.state == "warn"
        assert chk.min_gap_bytes == default_floor
    # Finite-but-huge (>= ~1.5e299): passes the accessor's isfinite+>=0 clamp,
    # but the GiB multiply overflows to inf — the guarded conversion inside
    # check_device_fs_gap falls back to the default floor instead of raising.
    monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_MIN_GB", "1.7e300")
    chk = _check(INCIDENT_DEVICE_BYTES, INCIDENT_FS_BYTES)
    assert chk.state == "warn"
    assert chk.min_gap_bytes == default_floor


def test_nonfinite_min_gap_gb_param_never_raises():
    """A caller-passed min_gap_gb=inf/nan bypasses the env accessor entirely —
    the conversion guard inside check_device_fs_gap must hold on its own."""
    default_floor = int(vdg.DEFAULT_DEVICE_FS_GAP_MIN_GB * 1024**3)
    for bad in (float("inf"), float("nan")):
        chk = _check(INCIDENT_DEVICE_BYTES, INCIDENT_FS_BYTES, min_gap_gb=bad)
        assert chk.state == "warn"
        assert chk.min_gap_bytes == default_floor


def test_nonfinite_pct_env_clamped_by_range_check(monkeypatch):
    """The pct knob does NOT share the vulnerability: its (0, 100] clamp
    already rejects inf/nan/-inf (nan comparisons are False; inf fails
    <= 100). Pinned so a clamp rewrite keeps non-finite values out of the
    warn comparison."""
    for raw in ("inf", "nan", "-inf", "1e400"):
        monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_PCT", raw)
        assert vdg.device_fs_gap_pct() == vdg.DEFAULT_DEVICE_FS_GAP_PCT


def test_run_guard_survives_inf_min_gb_env(monkeypatch):
    """End-to-end: run_guard with the REAL check body + the inf env knob
    completes (pre-fix the OverflowError escaped check_device_fs_gap into
    run_guard here). Only the sysfs/statvfs boundary is faked (incident
    geometry) via a wrapper that calls the real check_device_fs_gap; the
    alert leg is main()-side and out of run_guard's scope."""
    monkeypatch.setenv("EPS_VM_DEVICE_FS_GAP_MIN_GB", "inf")
    real_check = vdg.check_device_fs_gap
    monkeypatch.setattr(
        vdg,
        "check_device_fs_gap",
        lambda path: real_check(
            path,
            resolver=lambda p: (INCIDENT_DEVICE_BYTES, "sda", "sda1", "1"),
            statvfs_fn=lambda p: SimpleNamespace(f_blocks=INCIDENT_FS_BYTES, f_frsize=1),
        ),
    )
    monkeypatch.setattr(vdg, "disk_used_pct", lambda path="/": 10.0)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 500.0)
    res = vdg.run_guard(False, threshold=99.0)
    assert res.triggered is False
    assert res.device_fs.state == "warn"
    assert res.device_fs.min_gap_bytes == int(vdg.DEFAULT_DEVICE_FS_GAP_MIN_GB * 1024**3)


def test_exit_code_unchanged_on_warn(monkeypatch, tmp_path, capsys):
    """A devfs WARN never flips main()'s exit code: under-threshold disk +
    warn-state check -> rc 0 (exit 2 stays the still-over alarm channel), and
    the human-mode WARNING line lands on stderr."""
    warn = _warn_chk()
    monkeypatch.setattr(vdg, "check_device_fs_gap", lambda path: warn)
    monkeypatch.setattr(vdg, "disk_used_pct", lambda path="/": 50.0)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 500.0)
    monkeypatch.setattr(vdg, "_is_mounted", lambda path: False)  # skip data-disk pass
    _pushes, sidecar = _patch_alert_seams(monkeypatch, tmp_path)
    rc = vdg.main([])
    assert rc == 0
    assert len(sidecar) == 1
    err = capsys.readouterr().err
    assert "unexpanded" in err and "growpart" in err
    assert "NEVER auto-resizes" in err


# ─── 14: production-body test (real sysfs on this host) ───────────────────────


def test_real_sysfs_on_this_host():
    """Run the REAL resolver + check unmocked (the #906 production-body test).
    Environment-tolerant: passes on the VM (ok) and in containers (skipped)."""
    got = vdg._device_bytes_for_mount("/")
    assert isinstance(got, tuple) and len(got) == 4
    res = vdg.check_device_fs_gap("/")
    assert res.state in {"ok", "warn", "skipped"}
    if res.state != "skipped":
        assert res.device_bytes > 0 and res.fs_bytes > 0
        assert res.gap_bytes >= 0 and res.gap_pct >= 0.0
    else:
        assert res.reason


# ─── 16: surfacing-only source pin (acceptance criterion 6) ───────────────────


def test_no_subprocess_in_new_paths():
    """The new code paths invoke no subprocess/_run and never execute
    growpart/resize2fs/parted/sgdisk — surfacing ONLY (the §4.6 hard
    constraint, committed as a test)."""
    new_fns = (
        vdg._device_bytes_for_mount,
        vdg.check_device_fs_gap,
        vdg._print_device_fs_warning,
        vdg._device_fs_ack_sentinel_path,
        vdg._maybe_alert_device_fs,
    )
    all_src = "".join(inspect.getsource(fn) for fn in new_fns)
    # "subprocess." (the invocation form, dotted) — the docstrings legitimately
    # SAY "no subprocess"; the pin is on shelling out, not on the word.
    for banned in ("subprocess.", "_run(", "os.system", "Popen", "check_output", "check_call"):
        assert banned not in all_src, f"new device-fs path must not shell out: {banned}"
    # The resolver + check never even MENTION the mutating commands (the
    # printer/push advice text names them for the HUMAN to run — allowed).
    compute_src = inspect.getsource(vdg._device_bytes_for_mount) + inspect.getsource(
        vdg.check_device_fs_gap
    )
    for cmd in ("growpart", "resize2fs", "parted", "sgdisk"):
        assert cmd not in compute_src
