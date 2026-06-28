"""Shared pytest configuration for the workflow test suite."""

import logging
import os

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
