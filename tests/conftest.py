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
# import time makes the STANDING DEFAULT (runpod-first as of #2054; the
# module-level DEFAULT_AUTO_LANE_ORDER) the suite-wide baseline; tests
# that exercise the override set it explicitly via monkeypatch.setenv.
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
def _rewarm_task_workflow_repo_root_cache():
    """#1556 cross-test cache hygiene (the #703 leak family). Several test
    modules call ``task_workflow.invalidate_cache()`` (e.g. the ``fake_repo``
    fixtures) and never re-warm the process-wide (pid, cwd)-keyed repo-root
    ``lru_cache``. Any LATER test that (a) monkeypatches ``subprocess.run``
    globally (the poll/SSH harness convention — ``subprocess`` is a singleton
    module, so ``task_workflow``'s git probe is intercepted too) and (b)
    triggers a real task-state read (``poll_once``'s per-tick ``_marker_pid``
    / ``_issue_trigger_dense`` reads) then re-probes git THROUGH the fake and
    misreads live task state as unreadable — order-dependent failures that
    pass in isolation (the #703 shape). SETUP-side re-warm: at fixture setup
    the test's own patches are not yet applied (the harnesses patch inside
    test bodies), so a cold cache re-warms through the REAL subprocess; a
    warm cache is a dict hit (no I/O). Only fires when task_workflow is
    already imported; a genuinely unresolvable layout is left exactly as
    before this fixture existed (logged; the next consumer re-probes and
    surfaces it loudly)."""
    tw = sys.modules.get("explore_persona_space.task_workflow")
    if tw is not None:
        try:
            tw.repo_root()
        except Exception as exc:
            logging.getLogger("conftest").warning(
                "repo-root re-warm skipped (%s: %s); next consumer re-probes",
                type(exc).__name__,
                exc,
            )
    yield


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


# ---------------------------------------------------------------------------
# #2214 — artifacts CONTEXTS registry hermeticity (the #703 / #1247 leak family)
# ---------------------------------------------------------------------------
# `explore_persona_space.artifacts.context.CONTEXTS` is a process-global dict of
# 11 code-literal seed contexts. Production context-resolution seams mutate it BY
# DESIGN and idempotently — `issue1090_fu3_worker.ensure_context()` (fu3 conv
# prefix + `icl_prefix_<behavior>`), `bystander_panel()`, `panel_name_for()`
# (filtered negative panels) — so any test touching a seam leaks keys into every
# LATER test in the process. Two committed tests assert registry cleanliness and
# are the DETECTORS, not the bug:
#   * tests/test_artifacts_context.py::test_registry_seeds_validate
#       (`assert len(CONTEXTS) == 11` — the seed-size pin)
#   * tests/test_issue1090_fu3_dispatcher.py::test_conv_context_is_wildchat_family
#       (`CONV_CONTEXT_ID not in CONTEXTS` at entry — the issue-1144 r2 pin)
# Both red purely as a function of which files a selection collects and in what
# order, stochastically bouncing the Step 9c gate for unrelated diffs (#2063).
# The leak has TWO phases, which is why the baseline is taken in
# `pytest_sessionstart` (pre-collection) and restored at test SETUP:
#   (a) IMPORT time — HISTORICAL, removed by #2217. tests/test_issue1481_analysis.py
#       used to evaluate `PANEL_IDS = [c.context_id for c in fu3w.bystander_panel(BEH)]`
#       at module scope, so COLLECTION polluted before any test ran; #2217 replaced it
#       with the lazy `_panel_ids()` helper, and its `pytest_collection_finish` guard now
#       FAILS on any NEW import-time registration. The PRE-COLLECTION snapshot below is
#       kept regardless: it is what makes the baseline provably pristine. A snapshot
#       taken at the first test's setup would silently absorb whatever collection leaked,
#       so this phase being currently empty is a fact to re-verify, not to rely on.
#   (b) TEST time — bodies that call a resolution seam (fu6 / fu7 /
#       1586_read_organism / 1947_resume_matrix). Measured post-#2217: 4 polluter
#       tests across 4 files, 10 distinct CONTEXTS keys.
# Restoring at setup (not teardown) covers both with one mechanism. Production
# seams stay untouched: registration there is intended behavior and idempotent,
# so any test needing a context re-registers by calling the seam — which every
# direct `CONTEXTS[...]` subscript in the test tree already does today.
#
# SCOPE CAVEAT for future readers: this fixture is FUNCTION-scoped, so it runs
# AFTER any module/class/session-scoped fixture. A higher-scoped fixture that
# registers a context is therefore WIPED, not protected, and its registrations
# never reach the test body. ONE such fixture exists today (post-#2217): the
# module-scoped `pipeline` in tests/test_issue1481_analysis.py, which reaches
# `build_panel_fixtures` -> `_panel_ids()` -> `bystander_panel()`. It is safe
# ONLY because those tests re-register in-body and the seams are idempotent —
# not because the wipe does not happen to it. If you add another, do the same:
# register inside the test rather than relying on a higher-scoped fixture's
# registration surviving.
_CONTEXT_REGISTRY_BASELINE: dict | None = None


def pytest_sessionstart(session):
    """Snapshot the pristine seed CONTEXTS registry BEFORE collection (#2214)."""
    global _CONTEXT_REGISTRY_BASELINE
    from explore_persona_space.artifacts.context import CONTEXTS

    _CONTEXT_REGISTRY_BASELINE = dict(CONTEXTS)


@pytest.fixture(autouse=True)
def _restore_context_registry():
    """Reset CONTEXTS to the pre-collection baseline at every test's setup."""
    from explore_persona_space.artifacts.context import CONTEXTS

    baseline = _CONTEXT_REGISTRY_BASELINE
    if baseline is not None and baseline != CONTEXTS:
        CONTEXTS.clear()
        CONTEXTS.update(baseline)
    yield


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
        # Signature mirrors the real seam incl. the #2295 keyword-only `by`
        # (the attribution leg posts by="unknown") — a narrower wrapper here
        # would TypeError inside the leg's per-entry fail-soft and silently
        # skip the very markers under test.
        @functools.wraps(real_post)
        def _guarded(issue, note, dry_run, *, label, by="autonomous_session_watch"):
            if not dry_run and _sp.run is real_run:
                raise AssertionError(
                    f"_post_progress_marker(issue={issue}, label={label!r}, dry_run=False) "
                    "reached the #1247 autouse hermeticity guard (shared, tests/conftest.py) "
                    "with the REAL subprocess.run still live — the real body would shell "
                    "`task.py post-marker` and post a junk marker on a real task. Monkeypatch "
                    "a recorder (or stub subprocess.run) in the test."
                )
            return real_post(issue, note, dry_run, label=label, by=by)

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


@pytest.fixture(autouse=True)
def _forbid_real_guard_apply_launches(request, monkeypatch):
    """#1392 hermeticity guard (fail-loud), shared across watcher test modules
    — same family as the #1247/#1265 guards above. The sub-floor RECLAIM arm
    (`subfloor_reclaim_pass`) launches a DETACHED `vm_disk_guard.py --apply`;
    an unpatched launch from pytest would sweep the live VM's caches. Two
    layers: (a) the arm's kill switch is set by default, so the many
    pre-existing `vm_disk_pass(dry_run=False)` fixtures below the 60 GiB
    sub-floor never reach the launch path (a test exercising the arm
    `monkeypatch.delenv`s it); (b) `_launch_guard_apply` is wrapped fail-loud
    — a real-BODY exercise against a STUBBED subprocess.Popen (the argv
    durability pin) is allowed through; only a call with the GENUINE
    subprocess.Popen still live fails loud. A later test-level monkeypatch
    (a recorder) wins, as with the sibling guards."""
    watchers = _watcher_modules(request)
    if not watchers:
        return
    monkeypatch.setenv("EPM_DISABLE_SUBFLOOR_RECLAIM", "1")
    import functools
    import subprocess as _sp

    real_popen = _sp.Popen

    def _make_guarded(real_launch):
        @functools.wraps(real_launch)
        def _guarded(log_path):
            if _sp.Popen is real_popen:
                raise AssertionError(
                    f"_launch_guard_apply({log_path!r}) reached the #1392 autouse "
                    "hermeticity guard (shared, tests/conftest.py) with the REAL "
                    "subprocess.Popen still live — the real body would launch a detached "
                    "`vm_disk_guard.py --apply` sweep of the live VM. Monkeypatch a "
                    "recorder for _launch_guard_apply (or stub subprocess.Popen) in the "
                    "test."
                )
            return real_launch(log_path)

        return _guarded

    for asw in watchers:
        monkeypatch.setattr(asw, "_launch_guard_apply", _make_guarded(asw._launch_guard_apply))


@pytest.fixture(autouse=True)
def _sidecar_hermeticity_guard(request, monkeypatch, tmp_path):
    """#2141 hermeticity guard (redirect), shared across watcher test modules
    — same #1392 family as the guards above. No watcher-module test may
    append to the REAL ``.claude/cache/disk-guard-events.jsonl``: all watcher
    sidecar writes route through ``_disk_guard_sidecar_path()`` (single call
    site: ``_append_disk_guard_sidecar``), so this guard redirects that
    resolver to pytest tmp UNLESS the test has pinned ``PROJECT_ROOT`` itself
    (the ``watcher_roots`` convention in tests/test_vm_disk_subfloor_sentinel
    .py), in which case it DELEGATES to the real resolver — which reads the
    patched ``PROJECT_ROOT`` at call time — so root-pinned sidecar-content
    assertions keep working. One redirect covers every writer (sentinel,
    reclaim, data-disk, #2141 skip rows), present and future. Measured
    incident (#2141 Finding 1): 6,369 pytest-planted sentinel rows at
    ``free_gib: 17.0`` in the real sidecar, written by the real-body
    ``vm_disk_pass(dry_run=False)`` tests whose ``isolated_registry`` fixture
    pins only ``AUTONOMOUS_REGISTRY_DIR``. A later test-level monkeypatch of
    ``_disk_guard_sidecar_path`` wins, as with the sibling guards."""
    watchers = _watcher_modules(request)
    if not watchers:
        return
    for asw in watchers:
        real_root = asw.PROJECT_ROOT
        real_fn = asw._disk_guard_sidecar_path

        def _guarded(asw=asw, real_root=real_root, real_fn=real_fn):
            if real_root == asw.PROJECT_ROOT:  # test did NOT pin the root
                return tmp_path / "disk-guard-events.jsonl"
            return real_fn()  # root pinned to tmp -> delegate (reads patched PROJECT_ROOT)

        monkeypatch.setattr(asw, "_disk_guard_sidecar_path", _guarded)


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
    # #2058: the no-progress-respawn pass calls _stop_session on a live sid
    # when its progress fingerprint has been unchanged across N consecutive
    # tick heartbeats (same fleet-mutating class); its own tests stub the
    # pure predicate `decide_no_progress_respawn` instead.
    "no_progress_respawn_pass",
    # #1215: the orphan-wrapper sweep scans live /proc, writes real
    # ~/.eps-autonomous/wrapper-orphan-*.json state, and its opt-in stop arm
    # can SIGTERM real processes; its own tests stub its seams instead.
    "orphan_wrapper_pass",
    # Escalate-only observer passes against live VM state (round 2).
    "verdict_disagree_pass",
    # #1341: escalate-only too, but it runs a REAL `git status` against the
    # LIVE shared root and can write real sidecar rows + fire real pushes.
    "root_draft_pass",
    # #1439: report-only too, but it runs REAL task_workflow.audit() +
    # reconcile_registry(apply=False) reads against the LIVE registry and can
    # write real sidecar rows / state / pushes from a full-main() unit test.
    "registry_drift_pass",
    # #1806: escalate-only too, but it runs a REAL `git stash list` against
    # the LIVE shared root + scans the real ~/.task-workflow/root-sync-rescue/
    # dir and can write real sidecar rows / state / pushes from a full-main()
    # unit test; its own tests stub the collector / push / path seams instead.
    "stash_rescue_audit_pass",
    # #2134: escalate-only too, but it reads the LIVE registry + task
    # body.md files, runs real bounded `git log` subprocesses against the
    # live repo, and can post REAL epm:progress flag markers on live queued
    # tasks + write real sidecar rows / state / pushes from a full-main()
    # unit test; its own tests monkeypatch the collect/git seams +
    # PROJECT_ROOT / AUTONOMOUS_REGISTRY_DIR / _telegram_push instead.
    "predispatch_staleness_pass",
    # #2015: escalate-only too, but it runs a REAL `git status` against the
    # LIVE shared root and can write real sidecar rows / state / pushes from
    # a full-main() unit test; its own tests stub the collector / push / path
    # seams instead.
    "root_unstaged_audit_pass",
    # #2115: escalate-only too, but it iterates the LIVE registration dir,
    # reads real session transcripts via _transcript_tail_rows, and can write
    # real sidecar rows under .claude/cache/ + ~/.eps-autonomous/ state +
    # fire real Telegram pushes from a full-main() unit test; its own tests
    # monkeypatch PROJECT_ROOT / AUTONOMOUS_REGISTRY_DIR / the reader seams.
    "pending_call_wedge_pass",
    # #1564: flag-only too, but it sweeps the LIVE registry's completed set,
    # runs real gh/git probes, and can post REAL epm:progress markers on live
    # tasks + sidecar rows + pushes from a full-main() unit test.
    "completed_unmerged_pass",
    # #1704: escalate-only too, but it sweeps the LIVE registry's
    # non-`proposed` set, runs real scoped HF listings + real
    # `git ls-tree` against the live repo, and can write real sidecar
    # rows + fire real Telegram pushes from a full-main() unit test;
    # its own tests stub `_partial_bundle_candidate_issues` /
    # `_partial_bundle_scoped_listing` / `_committed_eval_paths` seams
    # instead.
    "partial_bundle_pass",
    # #2140: escalate-only too, but it writes real singleton state to
    # ~/.eps-autonomous/daemon-liveness.json, appends real sidecar rows under
    # .claude/cache/, and on a simulated 2-tick outage would fire a REAL
    # IMMEDIATE telegram_push.sh send from a full-main() unit test; its own
    # tests monkeypatch AUTONOMOUS_REGISTRY_DIR / PROJECT_ROOT + recorder
    # push seams instead.
    "daemon_liveness_pass",
    # #1681: the urgent-park router sweeps the LIVE tasks tree for parked
    # workflow-fix candidates and can run a real pytest subprocess, FILE +
    # dispatch a real task via file_infra_task.py, and post REAL
    # epm:workflow-fix-task-filed markers; its own tests use tmp-root
    # overrides + autospec'd subprocess seams instead.
    "urgent_wf_park_pass",
    "cpu_guard_pass",
    "happy_patch_pass",
    "data_disk_pass",
    "gate_push_pass",
    "gc_pass",
    # Live ~/.task-workflow/vm-ledger.json reap (round 2).
    "vm_ledger_reap_pass",
    # #2129: the settings model-id guard can REWRITE the real
    # ~/.claude/settings.json (+ settings.local.json) from a full-main()
    # unit test — the strongest live-HOME-mutating class here; its own
    # tests inject tmp files via the `paths=` param and monkeypatch the
    # sidecar / state / backup-dir constants + _telegram_push instead.
    "settings_model_guard_pass",
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


# ── #2217: import-time registry-mutation guard (collection-time measurement) ──
# pytest imports every COLLECTED module before running any test, so a module-
# level registration into the global CONTEXTS / NEGATIVE_PANELS registries
# poisons every other module's view for the whole run — invisible to the Step
# 9c paired-PREFIX replay when the offender sorts after the victim (#2059's
# residual blind class; incident #2217). Measure at collection: snapshot the
# key-sets at pytest_configure (== the fresh-import baseline — conftest
# imports the registry modules eagerly, before any test module; NEVER a
# hardcoded count), diff after every collector finishes (attributing ADDITION
# growth to that collector's nodeid), and snapshot again at
# pytest_collection_finish — post-collection, pre-run — so the assertion arm
# can check full key-set EQUALITY with the baseline (removals included)
# without false-positiving on RUNTIME leaks from tests that run earlier in
# the session. The assertion arm is
# tests/test_no_import_time_registry_mutation.py (a NORMAL failing test names
# offenders; a collection abort would exit rc=2 — indeterminate at the Step
# 9c compare — instead of a named NEW failure).
from explore_persona_space.artifacts.context import CONTEXTS as _GUARD_CONTEXTS  # noqa: E402
from explore_persona_space.artifacts.negatives import (  # noqa: E402
    NEGATIVE_PANELS as _GUARD_PANELS,
)

IMPORT_TIME_REGISTRY_DELTAS: dict[str, dict[str, list[str]]] = {}
_guard_baseline: dict[str, frozenset[str]] = {}
_guard_prev: dict[str, set[str]] = {}
_guard_post_collection: dict[str, frozenset[str]] = {}


def pytest_configure(config):
    """#2217 guard: snapshot the fresh-import registry key-sets (the baseline)."""
    _guard_baseline["CONTEXTS"] = frozenset(_GUARD_CONTEXTS)
    _guard_baseline["NEGATIVE_PANELS"] = frozenset(_GUARD_PANELS)
    _guard_prev["contexts"] = set(_GUARD_CONTEXTS)
    _guard_prev["panels"] = set(_GUARD_PANELS)


def pytest_collectreport(report):
    """#2217 guard: attribute registry ADDITION growth to the finishing collector."""
    ctx, pan = set(_GUARD_CONTEXTS), set(_GUARD_PANELS)
    dctx = ctx - _guard_prev["contexts"]
    dpan = pan - _guard_prev["panels"]
    if dctx or dpan:
        IMPORT_TIME_REGISTRY_DELTAS[report.nodeid] = {
            "CONTEXTS": sorted(dctx),
            "NEGATIVE_PANELS": sorted(dpan),
        }
    _guard_prev["contexts"], _guard_prev["panels"] = ctx, pan


def pytest_collection_finish(session):
    """#2217 guard: post-collection, pre-run key-set snapshot (equality anchor)."""
    _guard_post_collection["CONTEXTS"] = frozenset(_GUARD_CONTEXTS)
    _guard_post_collection["NEGATIVE_PANELS"] = frozenset(_GUARD_PANELS)


@pytest.fixture
def import_time_registry_deltas():
    """Per-collector registry ADDITION growth recorded during THIS run's
    collection (#2217). A DEEP copy — mutating it cannot corrupt the record."""
    return {
        k: {kk: list(vv) for kk, vv in v.items()} for k, v in IMPORT_TIME_REGISTRY_DELTAS.items()
    }


@pytest.fixture
def registry_collection_snapshots():
    """(configure-time fresh-import baseline, post-collection snapshot) for the
    guard test's key-set EQUALITY assert (#2217 SF1). Frozensets — immutable."""
    return dict(_guard_baseline), dict(_guard_post_collection)


@pytest.fixture
def _registry_guard_internals():
    """TEST-ONLY seam: the LIVE deltas dict + hook fns, so the guard's own
    negative control executes the real hook body (#906 one-production-body-
    test rule), never a copy."""
    return IMPORT_TIME_REGISTRY_DELTAS, pytest_collectreport, _guard_prev


@pytest.fixture
def registry_hygiene():
    """`ensure_context` -> `register_fu3_contexts()` and `panel_name_for` both
    mutate GLOBAL registries (CONTEXTS / NEGATIVE_PANELS) at runtime — correct
    in production, but test-order-poisoning for the registry-purity pins.
    Snapshot the key sets and remove anything a test added. Shared here since
    #2217 (moved verbatim from tests/test_issue1090_fu5_round.py); consumers:
    the fu5 ladder-organism tests and
    test_issue1090_fu3_dispatcher.py::test_conv_context_is_wildchat_family."""
    from explore_persona_space.artifacts.context import CONTEXTS
    from explore_persona_space.artifacts.negatives import NEGATIVE_PANELS

    ctx_before, panel_before = set(CONTEXTS), set(NEGATIVE_PANELS)
    try:
        yield
    finally:
        for k in set(CONTEXTS) - ctx_before:
            CONTEXTS.pop(k, None)
        for k in set(NEGATIVE_PANELS) - panel_before:
            NEGATIVE_PANELS.pop(k, None)
