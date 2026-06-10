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
