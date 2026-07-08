"""Tests for the codex_task.py quota-exhausted sentinel short-circuit (#1126).

Behaviors under test:

1. Reset-timestamp parsing ("try again at Aug 6th, 2026 6:26 AM") with the
   plausibility window (past / >45d / invalid calendar date -> None).
2. Sentinel WRITE in _finalize_result on terminal phase=failed + the
   usage-limit text (short-result length gate; fallback 24h TTL on parse
   failure; note carries the 'codex-quota-exhausted' token).
3. Sentinel READ short-circuit in main(): exit 9, no spawn, same
   epm:codex-task-failed marker contract; --reattach bypasses; expired /
   corrupt / implausibly far-future sentinels fail OPEN (delete + proceed);
   EPM_SKIP_CODEX_QUOTA_SENTINEL=1 disables the read only.

NOTE (intended behavior): while a sentinel is ACTIVE, an EMPTY-prompt
invocation exits 9, not 2 — the short-circuit sits BEFORE the prompt
resolution / empty-prompt check in main() (deliberate: it must also skip
the stdin read, which would block on bg dispatch).

Every test steers the sentinel path to tmp_path via
EPM_CODEX_QUOTA_SENTINEL_PATH (also exercising the override seam); the
suite-wide default is the nonexistent path set by the tests/conftest.py
autouse fixture (see test_conftest_isolation_env_is_set).
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]

# Verbatim fetched-result text of the 2026-07-08 incident artifacts
# (/tmp/codex-critic-1124-*-output.md, 317 B class).
VERBATIM_LIMIT_TEXT = (
    "You've hit your usage limit. To continue using Codex and get access to "
    "GPT-5.3-Codex, start a free trial of Plus today "
    "(https://chatgpt.com/explore/plus), or try again at Aug 6th, 2026 6:26 AM."
)


def _load_codex_task():
    """Load scripts/codex_task.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "codex_task_quota_under_test", REPO_ROOT / "scripts" / "codex_task.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["codex_task_quota_under_test"] = module
    spec.loader.exec_module(module)
    return module


codex_task = _load_codex_task()

# A deterministic "now" inside the incident window (2026-07-08 12:00 local),
# so the Aug-6-2026 reset text parses as future + plausible in every test run.
FIXED_NOW = time.mktime(datetime.datetime(2026, 7, 8, 12, 0).timetuple())


def _args(**overrides):
    """Argparse-like namespace with sane defaults for _finalize_result."""
    base = dict(
        issue=None,
        effort="high",
        write=False,
        output_file=None,
        prompt_file=None,
        prompt="do the thing",
        max_wait_secs=3600,
        poll_interval_secs=0,
        probe_error_cap=10,
        stall_detect_secs=600,
        cancelled_retry_cap=2,
        result_fetch_retry_cap=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _sentinel(monkeypatch, tmp_path) -> Path:
    """Steer the sentinel path to tmp_path; return it."""
    p = tmp_path / "sentinel"
    monkeypatch.setenv("EPM_CODEX_QUOTA_SENTINEL_PATH", str(p))
    return p


def _write_future_sentinel(path: Path, offset_secs: float = 3600.0) -> None:
    path.write_text(
        json.dumps(
            {"until_unix": time.time() + offset_secs, "until_iso": "test+1h", "parse_ok": True}
        )
    )


# ──────────────────────────────────────────────────────────────────────
# _parse_quota_reset
# ──────────────────────────────────────────────────────────────────────


def test_parse_quota_reset_verbatim_incident_text(monkeypatch):
    """The verbatim incident text parses to a local Aug 6 2026, 6:26 AM epoch."""
    monkeypatch.setattr(codex_task.time, "time", lambda: FIXED_NOW)
    ts = codex_task._parse_quota_reset(VERBATIM_LIMIT_TEXT)
    assert ts is not None
    lt = time.localtime(ts)
    assert (lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min) == (2026, 8, 6, 6, 26)


def test_parse_quota_reset_date_only_and_failures(monkeypatch):
    """Date-without-time -> local midnight; Never / past / far-future /
    invalid-calendar-date / garbage -> None (fallback-TTL path)."""
    monkeypatch.setattr(codex_task.time, "time", lambda: FIXED_NOW)

    ts = codex_task._parse_quota_reset("try again at Aug 6th, 2026.")
    assert ts is not None
    lt = time.localtime(ts)
    assert (lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min) == (8, 6, 0, 0)

    assert codex_task._parse_quota_reset("try again at Never") is None
    # Past date -> None.
    assert codex_task._parse_quota_reset("try again at Aug 6th, 2020") is None
    # Beyond the 45-day plausibility window -> None.
    assert codex_task._parse_quota_reset("try again at Aug 6th, 2027") is None
    # Invalid calendar date: the regex matches but datetime raises ValueError
    # (the except (ValueError, OverflowError) branch).
    assert codex_task._parse_quota_reset("try again at Feb 30th, 2027") is None
    assert codex_task._parse_quota_reset("garbage") is None


# ──────────────────────────────────────────────────────────────────────
# _write_quota_sentinel
# ──────────────────────────────────────────────────────────────────────


def test_write_sentinel_fallback_ttl_on_parse_failure(monkeypatch, tmp_path):
    """Limit text WITHOUT a parseable date writes parse_ok=false with the
    24h fallback TTL, so a malformed date never wedges dispatch forever."""
    p = _sentinel(monkeypatch, tmp_path)
    before = time.time()
    until_iso = codex_task._write_quota_sentinel("hit your usage limit but no date", "task-x")
    assert until_iso is not None
    data = json.loads(p.read_text())
    assert data["parse_ok"] is False
    assert data["job_id"] == "task-x"
    assert abs(data["until_unix"] - (before + codex_task.QUOTA_FALLBACK_TTL_SECS)) < 60


# ──────────────────────────────────────────────────────────────────────
# Detection in _finalize_result
# ──────────────────────────────────────────────────────────────────────


def test_finalize_failed_with_usage_limit_writes_sentinel_and_note(monkeypatch, tmp_path):
    """Terminal phase=failed + the verbatim limit text writes the sentinel
    and appends the codex-quota-exhausted token to the failure note."""
    p = _sentinel(monkeypatch, tmp_path)
    monkeypatch.setattr(
        codex_task,
        "_fetch_result_with_retry",
        lambda companion, job_id, cap: (0, VERBATIM_LIMIT_TEXT, "", "", 0),
    )
    result = codex_task._finalize_result(
        companion=Path("c"),
        job_id="task-x",
        phase="failed",
        args=_args(),
        pre_output_key=None,
        started=time.time(),
    )
    assert p.exists()
    assert result.kind == "fail"
    assert result.exit_code == 1
    assert "codex-quota-exhausted" in result.note


def test_finalize_failed_without_pattern_writes_no_sentinel(monkeypatch, tmp_path):
    """The pre-existing transient class (app-server exit) never writes the
    sentinel and keeps the unmodified note shape."""
    p = _sentinel(monkeypatch, tmp_path)
    monkeypatch.setattr(
        codex_task,
        "_fetch_result_with_retry",
        lambda companion, job_id, cap: (
            0,
            "codex app-server exited unexpectedly (exit 1).",
            "",
            "",
            0,
        ),
    )
    result = codex_task._finalize_result(
        companion=Path("c"),
        job_id="task-x",
        phase="failed",
        args=_args(),
        pre_output_key=None,
        started=time.time(),
    )
    assert not p.exists()
    assert result.kind == "fail"
    assert result.exit_code == 1
    assert "codex-quota-exhausted" not in result.note
    assert result.note.startswith("terminal phase=failed after ")


def test_finalize_failed_long_quoted_text_writes_no_sentinel(monkeypatch, tmp_path):
    """A LONG (>1500 char) phase=failed result that merely QUOTES the limit
    string (e.g. a genuine review body discussing it) must NOT arm the
    fleet-wide sentinel — the short-result length gate blocks it."""
    p = _sentinel(monkeypatch, tmp_path)
    long_text = ("review body line. " * 100) + VERBATIM_LIMIT_TEXT
    assert len(long_text) >= codex_task.QUOTA_SHORT_RESULT_MAX_CHARS
    monkeypatch.setattr(
        codex_task,
        "_fetch_result_with_retry",
        lambda companion, job_id, cap: (0, long_text, "", "", 0),
    )
    result = codex_task._finalize_result(
        companion=Path("c"),
        job_id="task-x",
        phase="failed",
        args=_args(),
        pre_output_key=None,
        started=time.time(),
    )
    assert not p.exists()
    assert "codex-quota-exhausted" not in result.note


# ──────────────────────────────────────────────────────────────────────
# Short-circuit in main()
# ──────────────────────────────────────────────────────────────────────


def _no_spawn(monkeypatch):
    """Make any companion resolution / spawn a hard test failure."""

    def _boom(*_a, **_k):
        raise AssertionError("companion invoked")

    monkeypatch.setattr(codex_task, "_resolve_companion", _boom)
    monkeypatch.setattr(codex_task, "_spawn_codex", _boom)
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)


def test_main_short_circuits_on_active_sentinel(monkeypatch, tmp_path):
    """An active future sentinel makes main() exit 9 with no companion
    resolution and no spawn."""
    p = _sentinel(monkeypatch, tmp_path)
    _write_future_sentinel(p)
    _no_spawn(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["codex_task.py", "--prompt", "x"])
    assert codex_task.main() == codex_task.EXIT_QUOTA_EXHAUSTED == 9


def test_main_short_circuit_posts_failed_marker_with_issue(monkeypatch, tmp_path):
    """With --issue, the short-circuit posts exactly one epm:codex-task-failed
    (note carrying codex-quota-exhausted) and never epm:codex-task-spawned."""
    p = _sentinel(monkeypatch, tmp_path)
    _write_future_sentinel(p)
    _no_spawn(monkeypatch)
    posted: list[tuple[int, str, str]] = []
    monkeypatch.setattr(
        codex_task, "_post_marker", lambda issue, kind, note: posted.append((issue, kind, note))
    )
    monkeypatch.setattr(sys, "argv", ["codex_task.py", "--prompt", "x", "--issue", "999"])
    assert codex_task.main() == 9
    assert len(posted) == 1
    issue, kind, note = posted[0]
    assert issue == 999
    assert kind == "epm:codex-task-failed"
    assert "codex-quota-exhausted" in note
    assert not any(k == "epm:codex-task-spawned" for _, k, _n in posted)


def test_reattach_bypasses_short_circuit(monkeypatch, tmp_path):
    """--reattach targets an already-spawned job (quota already spent); an
    active sentinel must NOT block the harvest."""
    p = _sentinel(monkeypatch, tmp_path)
    _write_future_sentinel(p)
    monkeypatch.setattr(codex_task, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(codex_task, "_resolve_companion", lambda: Path("dummy-companion"))
    monkeypatch.setattr(codex_task, "_run_reattach", lambda companion, args: 0)
    monkeypatch.setattr(
        sys, "argv", ["codex_task.py", "--reattach", "task-x", "--reattach-unbound"]
    )
    assert codex_task.main() == 0
    assert p.exists()  # sentinel untouched


# ──────────────────────────────────────────────────────────────────────
# _quota_sentinel_active fail-open behaviors + exit-code taxonomy
# ──────────────────────────────────────────────────────────────────────


def test_expired_and_corrupt_sentinels_fail_open(monkeypatch, tmp_path):
    """Expired -> deleted + None; corrupt -> deleted + None; missing -> None."""
    p = _sentinel(monkeypatch, tmp_path)

    p.write_text(json.dumps({"until_unix": time.time() - 100, "until_iso": "past"}))
    assert codex_task._quota_sentinel_active() is None
    assert not p.exists()  # self-expiry deletes

    p.write_text("not json")
    assert codex_task._quota_sentinel_active() is None
    assert not p.exists()  # corrupt deletes

    assert codex_task._quota_sentinel_active() is None  # missing


def test_far_future_sentinel_treated_as_corrupt(monkeypatch, tmp_path):
    """A hand-seeded / corrupt until_unix beyond the 45-day plausibility cap
    is deleted + ignored (read-path mirror of the parse-path cap), closing
    the far-future wedge."""
    p = _sentinel(monkeypatch, tmp_path)
    p.write_text(
        json.dumps(
            {
                "until_unix": time.time() + codex_task.QUOTA_MAX_PLAUSIBLE_SECS + 86400,
                "until_iso": "far-future",
            }
        )
    )
    assert codex_task._quota_sentinel_active() is None
    assert not p.exists()


def test_exit_code_distinct_and_env_kill_switch(monkeypatch, tmp_path):
    """Exit 9 is outside every occupied/retryable code set; the kill switch
    disables the READ only and never deletes the sentinel."""
    assert codex_task.EXIT_QUOTA_EXHAUSTED == 9
    assert 9 not in codex_task.TRANSIENT_FAIL_EXIT_CODES
    assert 9 not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 130, 143}

    p = _sentinel(monkeypatch, tmp_path)
    _write_future_sentinel(p)
    monkeypatch.setenv("EPM_SKIP_CODEX_QUOTA_SENTINEL", "1")
    assert codex_task._quota_sentinel_active() is None
    assert p.exists()  # kill switch never deletes


def test_conftest_isolation_env_is_set():
    """Guard the tests/conftest.py autouse isolation fixture: every test in
    the suite must see EPM_CODEX_QUOTA_SENTINEL_PATH pointing at a
    nonexistent path, so a LIVE outage sentinel on this VM can never flip
    the sibling codex_task tests to rc=9/0-spawns (task #1126)."""
    val = os.environ["EPM_CODEX_QUOTA_SENTINEL_PATH"]
    assert val
    assert not Path(val).exists()
