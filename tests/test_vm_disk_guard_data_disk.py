"""Offline unit tests for the #681 data-disk (escalate-only) pass of the VM
disk guard (scripts/vm_disk_guard.py).

The dedicated data disk (`/mnt/eps-data`) holds the relocated
`.claude/worktrees/` tree. The guard watches it in a SECOND `run_guard` call
with ``reclaim_tiers=False``: the `/`-rooted reclaim arms (tier (a) uv cache
prune, tier (c) stale-log sweep) must NOT run keyed off the data disk; only
tier (b) (terminal-cache reap + active-cache escalation) — the one
data-disk-appropriate arm — fires there.

Covers (plan §5.1):
  * run_guard(disk_path=..., reclaim_tiers=False) triggers at 96% AND skips the
    uv/log reclaim tiers,
  * a full data disk with a healthy `/` leaves escalation-only (active issue's
    cache NOT deleted; the SAFE reclaim command surfaced),
  * EPS_VM_DATA_DISK_PATH redirects the watched mount,
  * the data-disk escalation/sub-floor pass under dry_run mutates nothing.

Loaded via importlib exactly like tests/test_vm_disk_guard.py.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures ────────────────────────────────────────────────────────────────


def _make_issue_data(data_root: Path, issue_n: int, *, prefix: str = "issue_") -> Path:
    """A realistic data/issue_<N>/ tree: hf_dl + g1_dl caches (each with a file)
    + a store/ (KEEP). Returns the issue dir."""
    issue_dir = data_root / f"{prefix}{issue_n}"
    for cache in ("hf_dl", "g1_dl"):
        d = issue_dir / cache
        d.mkdir(parents=True)
        (d / "blob.bin").write_bytes(b"x" * 4096)
    store = issue_dir / "store"
    store.mkdir(parents=True)
    (store / "generated.json").write_text('{"kept": true}')
    return issue_dir


def _patch_disk(monkeypatch, before_pct, after_pct, free_gb=200.0):
    """disk_used_pct returns before_pct on the 1st call, after_pct after (run_guard
    reads pre + post)."""
    state = {"calls": 0}

    def fake_used(path="/"):
        state["calls"] += 1
        return before_pct if state["calls"] == 1 else after_pct

    monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": free_gb)


# ─── test 1: reclaim_tiers=False skips the /-rooted tiers ─────────────────────


def test_run_guard_watches_data_disk_percent(tmp_path, monkeypatch):
    """A 96%-full data disk triggers, but with reclaim_tiers=False ONLY tier (b)
    runs — the uv-cache + stale-logs reclaim tiers are NOT invoked."""
    data_root = tmp_path / "wt-data"
    _make_issue_data(data_root, 658)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")
    # Sandbox the #773 active-consumer gate: REAL active tasks reference
    # data/issue_658/, so an un-sandboxed gate walks the LIVE tasks/ tree and
    # (correctly) keeps this synthetic cache — same trap the fake_repo
    # fixtures in the sibling test files exist to prevent.
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {})

    invoked = {"uv": False, "logs": False}

    def _uv(apply):
        invoked["uv"] = True
        return vdg.TierResult(name="uv-cache")

    def _logs(*a, **k):
        invoked["logs"] = True
        return vdg.TierResult(name="stale-logs")

    monkeypatch.setattr(vdg, "clean_uv_cache", _uv)
    monkeypatch.setattr(vdg, "clean_stale_logs", _logs)
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=40.0)

    res = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
    )
    assert res.triggered is True
    # ONLY tier (b) ran — the /-rooted reclaim tiers never fired on the data disk.
    assert {t.name for t in res.tiers} == {"terminal-download-caches"}
    assert invoked["uv"] is False
    assert invoked["logs"] is False
    # tier (b) DID reap the terminal issue's cache on the data disk.
    assert not (data_root / "issue_658" / "hf_dl").exists()
    assert (data_root / "issue_658" / "store" / "generated.json").is_file()


# ─── test 2: data disk full, / healthy → escalation only, no active delete ────


def test_data_disk_full_leaves_root_escalation_only(tmp_path, monkeypatch):
    """Data disk at 96%, / at 60%: the data-disk pass escalates an ACTIVE issue's
    cache (Telegram + sidecar) but deletes NOTHING; the surfaced reclaim command
    is the SAFE clean_experiment_downloads.py <N> --apply."""
    repo = tmp_path
    monkeypatch.setattr(vdg, "repo_root", lambda: repo)
    monkeypatch.setattr(ced, "repo_root", lambda: repo)
    # Determinism pin (#924): force the on-main resolution path.
    monkeypatch.setattr(ced, "_off_main_checkout_root", lambda: None)
    data_root = repo / "wt-data"
    issue_dir = _make_issue_data(data_root, 658)
    # Make the active cache big enough to clear the escalation floor (5 GB).
    monkeypatch.setattr(vdg, "_ACTIVE_ESCALATION_MIN_BYTES", 1)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")  # ACTIVE

    pushes: list[str] = []
    monkeypatch.setattr(vdg, "_telegram_push", lambda msg, apply: pushes.append(msg) or True)
    events: list[dict] = []
    monkeypatch.setattr(vdg, "append_disk_guard_event", lambda ev, *, apply=True: events.append(ev))
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=96.0)

    res = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
    )
    assert res.triggered is True
    # The ACTIVE issue's cache is NEVER deleted — bytes_freed is 0.
    assert res.bytes_freed == 0
    assert (issue_dir / "hf_dl").is_dir()  # active cache untouched
    # An escalation row + push were emitted, naming the SAFE reclaim command.
    assert events and events[0]["task"] == 658
    assert "clean_experiment_downloads.py 658 --apply" in events[0]["reclaim_cmd"]
    assert pushes and "658" in pushes[0]


# ─── test 3: env override redirects the watched data-disk mount ───────────────


def test_env_override_data_disk_path(monkeypatch):
    monkeypatch.delenv("EPS_VM_DATA_DISK_PATH", raising=False)
    assert vdg.data_disk_path() == vdg.DEFAULT_DATA_DISK_PATH
    monkeypatch.setenv("EPS_VM_DATA_DISK_PATH", "/mnt/other-data")
    assert vdg.data_disk_path() == "/mnt/other-data"
    # A blank value falls back to the default (never an empty path).
    monkeypatch.setenv("EPS_VM_DATA_DISK_PATH", "   ")
    assert vdg.data_disk_path() == vdg.DEFAULT_DATA_DISK_PATH


# ─── test 4: dry-run mutates nothing on the data-disk pass ────────────────────


def test_data_disk_pass_dry_run_no_mutation(tmp_path, monkeypatch):
    """Drive the data-disk escalation pass with apply=False against a 96%-full
    simulated data disk: it writes NO sidecar row and deletes NOTHING — only the
    [report-only] log lines. Exercises the dry_run thread on the NEW data-disk
    path so the §8 manual --dry-run smoke cannot silently become a real
    mutation (#596/#607/#633 pattern)."""
    repo = tmp_path
    monkeypatch.setattr(vdg, "repo_root", lambda: repo)
    monkeypatch.setattr(ced, "repo_root", lambda: repo)
    # Determinism pin (#924): force the on-main resolution path.
    monkeypatch.setattr(ced, "_off_main_checkout_root", lambda: None)
    data_root = repo / "wt-data"
    issue_dir = _make_issue_data(data_root, 658)
    monkeypatch.setattr(vdg, "_ACTIVE_ESCALATION_MIN_BYTES", 1)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "running")  # ACTIVE

    # Record the `apply` flag every escalation arm receives; the dry-run pass
    # may COMPOSE a would-push / would-append, but must always pass apply=False
    # so neither performs a real side effect (the report-only branch of each
    # only prints).
    push_applies: list[bool] = []
    monkeypatch.setattr(
        vdg, "_telegram_push", lambda msg, apply: push_applies.append(apply) or False
    )
    # append_disk_guard_event is the REAL one (its apply=False path prints only,
    # writing NO sidecar file).
    sidecar = repo / ".claude" / "cache" / "disk-guard-events.jsonl"
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=96.0)

    res = vdg.run_guard(
        apply=False,  # DRY RUN
        threshold=85.0,
        data_root=data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
    )
    assert res.triggered is True
    # Nothing deleted on disk.
    assert (issue_dir / "hf_dl").is_dir()
    assert (issue_dir / "store" / "generated.json").is_file()
    # No sidecar row WRITTEN (the real append_disk_guard_event apply=False path
    # reports only — never touches the file).
    assert not sidecar.exists()
    # Every escalation arm was invoked dry (apply=False) — never a real send.
    assert all(applied is False for applied in push_applies)


# ─── #681 round-2 Major: data-disk MOUNT-presence gate (not is_dir) ───────────


def test_is_mounted_false_on_plain_dir(tmp_path):
    """``_is_mounted`` MUST return False for a plain (non-mount) directory on the
    root fs — the regression at the heart of the round-2 Major. A plain dir
    shares its parent's st_dev, so it is NOT a mount."""
    plain = tmp_path / "fake-data-disk"
    plain.mkdir()
    assert vdg._is_mounted(str(plain)) is False


def test_is_mounted_false_on_missing_path(tmp_path):
    """Fail-soft: a nonexistent path is treated as not-mounted (the pass
    cleanly no-ops, never reporting /'s usage as the data disk's)."""
    assert vdg._is_mounted(str(tmp_path / "does-not-exist")) is False


def test_is_mounted_true_on_a_real_mount():
    """A genuine mount point (here ``/proc``, present on every Linux CI host)
    has a different st_dev than its parent → True. Skips gracefully if /proc is
    somehow not a distinct mount (non-Linux CI)."""
    import os

    p = "/proc"
    if not os.path.isdir(p) or os.stat(p).st_dev == os.stat(os.path.join(p, "..")).st_dev:
        import pytest

        pytest.skip("/proc is not a distinct mount on this host")
    assert vdg._is_mounted(p) is True


def test_main_skips_data_disk_pass_for_plain_dir(tmp_path, monkeypatch):
    """PRODUCTION-PROBE (#681 round-2 Major): point the data-disk path at a plain
    (non-mount) dir on the root fs and run ``main()``. The data-disk pass MUST
    NOT run — ``main`` must call ``run_guard`` ONLY for ``/``, never with
    ``disk_path=<plain dir>`` — so it can never misread /'s statvfs as the data
    disk's. The old ``Path(dd_path).is_dir()`` gate (True for any dir) ran the
    pass against the unmounted dir; the ``_is_mounted`` gate skips it."""
    plain = tmp_path / "mnt-eps-data-not-mounted"
    plain.mkdir()  # exists as a plain root-fs dir (Phase-1 mkdir / nofail-boot state)

    calls: list[str | None] = []

    def fake_run_guard(apply, *, threshold=None, log_max_age=None, **kw):
        calls.append(kw.get("disk_path", "/"))
        return vdg.GuardResult(
            used_pct_before=40.0,
            used_pct_after=40.0,
            free_gb_before=200.0,
            free_gb_after=200.0,
            threshold_pct=85.0,
            triggered=False,
            apply=apply,
        )

    monkeypatch.setattr(vdg, "run_guard", fake_run_guard)
    rc = vdg.main(["--data-disk-path", str(plain), "--json"])
    assert rc == 0
    # Only the boot-disk (/) pass ran; the data-disk pass was correctly SKIPPED.
    assert calls == ["/"], f"data-disk pass must be skipped for a plain dir; got {calls}"


def test_main_runs_data_disk_pass_for_real_mount(tmp_path, monkeypatch):
    """Counterpart: when ``_is_mounted`` reports the data-disk path as a live
    mount, ``main()`` DOES run the second (data-disk) ``run_guard`` pass — proving
    the gate is not just always-off."""
    calls: list[str | None] = []

    def fake_run_guard(apply, *, threshold=None, log_max_age=None, **kw):
        calls.append(kw.get("disk_path", "/"))
        return vdg.GuardResult(
            used_pct_before=40.0,
            used_pct_after=40.0,
            free_gb_before=200.0,
            free_gb_after=200.0,
            threshold_pct=85.0,
            triggered=False,
            apply=apply,
        )

    monkeypatch.setattr(vdg, "run_guard", fake_run_guard)
    monkeypatch.setattr(vdg, "_is_mounted", lambda p: True)  # simulate a live mount
    rc = vdg.main(["--data-disk-path", "/mnt/eps-data", "--json"])
    assert rc == 0
    assert "/" in calls and "/mnt/eps-data" in calls, (
        f"both the boot-disk and data-disk passes must run for a live mount; got {calls}"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
