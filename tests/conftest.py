"""Shared pytest configuration for the workflow test suite."""

import logging
import os
import sys

import pytest

# Keep the suite offline-deterministic: verify_task_body.py check 4b
# (figure URL existence, incident task #507) falls back to an HTTP HEAD
# for figure URLs it cannot resolve offline via `git cat-file`. Fixture
# bodies across the suite pin synthetic SHAs (`0123456789abcdef`,
# `abc1234`, ...) that are unknown to the local object database, so
# without this fence every verify_text() call would attempt a real
# network probe (slow, flaky, and a definitive 404 would flip
# long-standing PASS fixtures to FAIL). Subprocess-based invocations of
# the verifier inherit the env var too. Tests that exercise the HTTP
# path stub `verify_task_body._http_head_status` directly (the stub
# replaces the function, bypassing this fence).
os.environ.setdefault("EPM_VERIFY_BODY_NO_HTTP", "1")

# Same offline-determinism fence for verify_task_body.py check 23 (HF Hub
# revision-pin existence, incident task #537). The check probes
# `huggingface_hub.list_repo_files(repo_id, revision=<sha>)` for HF
# `/tree/<sha>/<path>` URLs; fixture bodies pin synthetic SHAs
# (`abc123def`, ...) on real-shaped repo ids that would otherwise hit the
# live Hub API (slow, flaky, auth-gated, and a real revision-not-found
# would flip long-standing PASS fixtures to FAIL). The check is fail-soft —
# it SKIPs (PASS + `unverified` note) on this fence — so fixtures stay
# green offline. Tests that exercise the HF path stub
# `verify_task_body._hf_url_existence` directly, or monkeypatch
# `huggingface_hub.list_repo_files`, bypassing this fence.
os.environ.setdefault("EPM_VERIFY_BODY_NO_HF", "1")

# Keep the suite hermetic against an ambient developer-shell auto-lane
# override: `backends.router.route()` resolves the auto lane order from
# EPM_AUTO_LANE_ORDER when RouterConfig.lane_order is None, so a value
# exported in the invoking shell would silently reorder every auto-route
# test across test_router.py / test_issue_dispatch.py /
# test_dispatch_issue_cli.py / test_router_acceptance.py. Dropping it at
# import time makes the GCP-first STANDING DEFAULT the suite-wide
# baseline; tests that exercise the override set it explicitly via
# monkeypatch.setenv.
os.environ.pop("EPM_AUTO_LANE_ORDER", None)


# Collection-time global leaks that several test modules induce and that
# pytest's ``monkeypatch`` fixture does NOT auto-revert (they bypass
# ``monkeypatch.setenv`` / ``monkeypatch.setattr``), which makes the full
# ``uv run pytest tests/`` suite fail under alphabetical collection order
# while every offending test passes in isolation (incident #703):
#
#   * ``tests/test_issue685_{extraction,coexistence}.py`` set ``HF_HOME`` via
#     ``os.environ.setdefault`` at MODULE IMPORT (collection) time. Once
#     ``HF_HOME`` is present, ``autonomous_session_watch._vm_reclaim_hf_hub_cache``
#     takes its ``scan_cache_dir(cache_dir=_hub_cache)`` branch, which a no-arg
#     fake ``scan_cache_dir`` lambda rejects ->
#     ``test_vm_reclaim_hf_hub_cache_evicts_through_bounded_worker`` fails.
#   * an earlier module flips the root logger to ``INFO`` (via
#     ``logging.basicConfig`` / ``setLevel``), so ``issue672_validate``'s
#     ``logger.info(...)`` actually emits and ``logging.LogRecord.__init__``
#     consumes ticks of the test's monkeypatched virtual ``time.time()`` clock,
#     tripping ``test_loop_poller_detects_second_failover_in_quiet_period``.
#
# Snapshot + restore both globals around EVERY test so a collection-time leak
# can't bleed across the suite. This SNAPSHOTS the pre-test value rather than
# blanket-forcing a default, so tests that legitimately set ``HF_HOME`` inside
# their own bodies (which run after fixture setup and auto-revert) or manage
# their own log level via ``caplog`` (which uses its own handler + level) are
# unaffected. The fixture is pure in-memory (a small dict snapshot + one
# ``getLogger().level`` read), no I/O.
_HF_ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE")


@pytest.fixture(autouse=True)
def _isolate_codex_quota_sentinel(monkeypatch):
    """Point codex_task.py's quota-sentinel path at a fixed nonexistent
    location for every test (task #1126). The sibling codex test files
    invoke codex_task.main() in prompt mode with NO sentinel handling, so
    a LIVE sentinel at DISPATCH_ROOT/.claude/cache/codex-quota-exhausted-
    until (guaranteed to exist during a real org-quota outage) would flip
    their expected exit codes to 9 with zero spawns and break every
    full-suite run on this VM. A fixed nonexistent path (no tmp_path
    materialization across the ~5800-test suite) keeps the read a clean
    FileNotFoundError no-op; the quota-sentinel tests override it
    per-test to tmp_path, which also exercises the override seam."""
    monkeypatch.setenv(
        "EPM_CODEX_QUOTA_SENTINEL_PATH", "/nonexistent/eps-test-codex-quota-sentinel"
    )


@pytest.fixture(autouse=True)
def _isolate_leaky_global_state():
    saved_env = {k: os.environ.get(k) for k in _HF_ENV_KEYS}
    for k in _HF_ENV_KEYS:
        os.environ.pop(k, None)
    root = logging.getLogger()
    saved_level = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(saved_level)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ─── #1247 watcher hermeticity guards, shared (task #1265) ────────────────────
#
# The autonomous-session watcher (scripts/autonomous_session_watch.py) has two
# subprocess seams that a unit test must never reach for real:
#   * `_post_progress_marker` shells `task.py post-marker` at the canonical
#     PROJECT_ROOT — an unpatched dry_run=False call posts a JUNK marker + git
#     commit on a REAL task (the two-week #662/#663/#867 incident class).
#   * `_task_status` shells `task.py view <N> --json` against the LIVE task
#     tree — an unpatched call makes a unit test depend on a real task's live
#     status (+ ~1-2s subprocess per call).
# The guards were born as per-file autouse fixtures in the three big watcher
# test files (#1247); task #1265 moved them here so every current AND future
# watcher test module is covered with zero per-file ceremony.

# The watcher is importable under BOTH sys.modules names — scripts/ is also a
# package (tests/test_router.py imports scripts.autonomous_session_watch today).
# Each name binds a DISTINCT module object, so the guards patch EVERY live one
# (patching only one would leave the other name's seams unguarded).
_WATCHER_MODULE_NAMES = ("autonomous_session_watch", "scripts.autonomous_session_watch")


def _watcher_modules(request):
    """Return the live watcher module object(s) iff the requesting test's MODULE
    imports the watcher (the module object itself under either sys.modules name,
    or any module-level attribute whose __module__ is one of those names).
    Convention (documented here for future files): watcher test modules import
    the watcher at MODULE level; a function-body-only import dodges this
    predicate (accepted residual — see the #1265 plan §7)."""
    watchers = [m for m in (sys.modules.get(n) for n in _WATCHER_MODULE_NAMES) if m is not None]
    if not watchers:  # watcher never imported this session: cheap no-op
        return []
    try:
        mod = request.module
    except AttributeError:  # non-Python test items
        return []
    for value in vars(mod).values():
        if any(value is w for w in watchers) or (
            getattr(value, "__module__", None) in _WATCHER_MODULE_NAMES
        ):
            return watchers
    return []


@pytest.fixture(autouse=True)
def _forbid_real_marker_posts(request, monkeypatch):
    """#1247 hermeticity guard (fail-loud), shared across watcher test modules
    (task #1265). Contract: a later test-level/fixture-level monkeypatch wins;
    dry_run=True keeps the real log-only behavior; a real-BODY exercise against
    a STUBBED subprocess.run (argv-recording tests) is allowed through; only
    dry_run=False with the GENUINE subprocess.run still live fails loud."""
    watchers = _watcher_modules(request)
    if not watchers:
        return
    import functools
    import subprocess as _sp

    real_run = _sp.run

    def _make_guarded(real_post):
        # functools.wraps sets __wrapped__, so inspect.getsource() on the patched
        # attribute still resolves the ORIGINAL body (#966 source-inspection pins).
        @functools.wraps(real_post)
        def _guarded(issue, note, dry_run, *, label):
            if not dry_run and _sp.run is real_run:
                raise AssertionError(
                    f"_post_progress_marker(issue={issue}, label={label!r}, dry_run=False) "
                    "reached the #1247 autouse hermeticity guard (shared, tests/conftest.py) "
                    "with the REAL subprocess.run still live — the real body would shell "
                    "`task.py post-marker` and post a junk marker on a real task. Monkeypatch "
                    "a recorder (or stub subprocess.run) in the test."
                )
            return real_post(issue, note, dry_run, label=label)

        return _guarded

    for asw in watchers:
        monkeypatch.setattr(asw, "_post_progress_marker", _make_guarded(asw._post_progress_marker))


@pytest.fixture(autouse=True)
def _forbid_real_task_status_reads(request, monkeypatch):
    """#1247 round-2 hermeticity guard (fail-loud), shared (task #1265):
    _task_status shells `task.py view` against the LIVE task tree — a
    determinism/latency pin, not a mutation fence. A test that needs a status
    overrides with its own stub (a later monkeypatch wins), e.g.
    monkeypatch.setattr(asw, "_task_status", lambda issue: "running")."""
    watchers = _watcher_modules(request)
    if not watchers:
        return
    import functools

    def _make_guarded(real_task_status):
        @functools.wraps(real_task_status)
        def _guarded(issue):
            raise AssertionError(
                f"_task_status({issue}) reached the #1247 autouse hermeticity guard "
                "(shared, tests/conftest.py) — monkeypatch a status in the test, e.g. "
                "monkeypatch.setattr(asw, '_task_status', lambda issue: 'running')."
            )

        return _guarded

    for asw in watchers:
        monkeypatch.setattr(asw, "_task_status", _make_guarded(asw._task_status))


# ─── #1247 fleet-mutating pass stub for FULL-main() watcher tests (#1278) ────
#
# Shared home for the call-explicit stub helper formerly duplicated in
# tests/test_autonomous_session_watch.py + tests/test_stalled_detector_and_gc.py
# (the copies diverged: #1267 added boot_death_pass to only one). It lives here
# next to the #1265 autouse guards because it is the same #1247 hermeticity
# family — but it stays a PLAIN FUNCTION, never an autouse fixture: callers
# rely on call-site-relative ordering (a later monkeypatch wins), so each
# full-main() test calls it explicitly and re-patches its own recorders after.
# The caller passes ITS OWN watcher module object (`asw`) — the watcher binds
# under two sys.modules names (see _WATCHER_MODULE_NAMES above), and main()
# resolves pass names in its defining module's globals, so patching exactly
# the object the test drives is the correct scope.

_FLEET_MUTATING_PASS_NAMES = (
    # Fleet-MUTATING sweep passes (round 1).
    "proposed_infra_sweep_pass",
    "capacity_retry_pass",
    "program_orchestrator_pass",
    # #1267: the boot-death pass can STOP a real session (same
    # fleet-mutating class); its own tests stub its seams instead.
    "boot_death_pass",
    # Escalate-only observer passes against live VM state (round 2).
    "verdict_disagree_pass",
    # #1341: escalate-only too, but it runs a REAL `git status` against the
    # LIVE shared root and can write real sidecar rows + fire real pushes.
    "root_draft_pass",
    "cpu_guard_pass",
    "happy_patch_pass",
    "data_disk_pass",
    "gate_push_pass",
    "gc_pass",
    # Live ~/.task-workflow/vm-ledger.json reap (round 2).
    "vm_ledger_reap_pass",
)


def _stub_fleet_mutating_passes(asw, monkeypatch):
    """#1247 hermeticity for FULL-main() tests (shared home: task #1278):
    main() runs passes that scan the LIVE repo and can REALLY mutate fleet
    state from a unit test — ``proposed_infra_sweep_pass`` +
    ``capacity_retry_pass`` DISPATCH real ``spawn-issue --auto`` sessions via
    the live Happy daemon (observed 2026-07-10: a suite run of
    test_main_daemon_reachable_runs_both_passes spawned a REAL session for
    proposed task #1227), and ``program_orchestrator_pass``
    (daemon-INDEPENDENT) can relaunch the real #660 tmux daemon. Round 2
    (closing the test-hermeticity-residual-observer-passes concern)
    additionally stubs the escalate-only OBSERVER passes: they scan LIVE VM
    state (REGISTRY tasks + events.jsonl via task_workflow, /proc + the
    earlyoom journal, the Happy daemon bundle, statvfs on /mnt/eps-data) and
    can write REAL sidecar rows under ``.claude/cache/`` + fire real Telegram
    pushes from a unit test — plus ``vm_ledger_reap_pass``, which MUTATES the
    live ``~/.task-workflow/vm-ledger.json`` (same live-VM-state class).
    Every test that drives the full main() pass sequence calls this helper;
    a test that asserts on one of these passes re-patches its own recorder
    AFTER the helper call (a later monkeypatch wins — so call the helper
    before any recorder you need to keep), and a test OF one of these passes
    stubs its own seams instead and does not call this helper."""
    for pass_name in _FLEET_MUTATING_PASS_NAMES:
        monkeypatch.setattr(asw, pass_name, lambda *a, **kw: None)
