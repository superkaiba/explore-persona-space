"""Tests for scripts/cron_daily_healthcheck.sh (task #711).

The body's acceptance criterion (a): a simulated failed nightly ``/daily`` (no
``logs/daily/<date>.md`` written) triggers the heartbeat alert. We drive the
bash wrapper via subprocess with env-var overrides (daily dir, sentinel dir, log
dir, "yesterday" date, and a FAKE telegram_push.sh that records each call), then
assert:

- missing-yesterday-log → push attempted exactly once + sentinel written;
- a present+fresh yesterday-log → NO push;
- a second invocation the same day after a push → NO push (sentinel suppresses).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parent.parent / "scripts" / "cron_daily_healthcheck.sh"


@pytest.fixture
def harness(tmp_path: Path):
    """A tmp project layout + a fake telegram_push.sh that logs its message arg."""
    project = tmp_path / "project"
    daily_dir = project / "logs" / "daily"
    sentinel_dir = project / "logs" / "daily_healthcheck"
    log_dir = sentinel_dir
    daily_dir.mkdir(parents=True)
    sentinel_dir.mkdir(parents=True)

    push_log = tmp_path / "push_calls.log"
    fake_push = tmp_path / "fake_telegram_push.sh"
    fake_push.write_text(f'#!/bin/bash\necho "$1" >> "{push_log}"\nexit 0\n')
    fake_push.chmod(0o755)

    def run(yesterday: str = "2026-06-27") -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "EPS_HEALTHCHECK_PROJECT_DIR": str(project),
            "EPS_HEALTHCHECK_YESTERDAY": yesterday,
            "EPS_HEALTHCHECK_DAILY_DIR": str(daily_dir),
            "EPS_HEALTHCHECK_SENTINEL_DIR": str(sentinel_dir),
            "EPS_HEALTHCHECK_LOG_DIR": str(log_dir),
            "EPS_TELEGRAM_PUSH_SCRIPT": str(fake_push),
        }
        return subprocess.run(
            ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True, check=False
        )

    return {
        "project": project,
        "daily_dir": daily_dir,
        "sentinel_dir": sentinel_dir,
        "push_log": push_log,
        "run": run,
    }


def _push_count(push_log: Path) -> int:
    if not push_log.exists():
        return 0
    return len([ln for ln in push_log.read_text().splitlines() if ln.strip()])


def test_missing_yesterday_log_triggers_push_and_writes_sentinel(harness):
    """No logs/daily/<yesterday>.md → exactly one push + a sentinel flag."""
    yesterday = "2026-06-27"
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    # The message names the missing day.
    message = harness["push_log"].read_text()
    assert yesterday in message
    # #994: the alert carries the paste-ready backfill command (env-var name,
    # the /daily <yesterday> invocation, and the cd-into-project prefix) so a
    # future MSG edit cannot silently drop the recovery affordance.
    assert "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=" in message
    assert f"claude -p '/daily {yesterday}'" in message
    assert "backfill: cd " in message
    assert (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()


def test_second_invocation_same_day_does_not_repush(harness):
    """Sentinel suppresses a re-alert on a second run the same day."""
    yesterday = "2026-06-27"
    harness["run"](yesterday)
    assert _push_count(harness["push_log"]) == 1
    # Second run: still missing, but sentinel present → no second push.
    result2 = harness["run"](yesterday)
    assert result2.returncode == 0
    assert _push_count(harness["push_log"]) == 1


def test_present_fresh_log_no_push(harness):
    """A present + fresh logs/daily/<yesterday>.md → no push, no sentinel."""
    yesterday = "2026-06-27"
    (harness["daily_dir"] / f"{yesterday}.md").write_text("# daily brief\n")
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 0
    assert not (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()
