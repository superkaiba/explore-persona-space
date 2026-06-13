"""Shared pytest configuration for the workflow test suite."""

import os

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
