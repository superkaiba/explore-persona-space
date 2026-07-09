"""Tests for scripts/cron_daily_healthcheck.sh (tasks #711, #994, #1189).

The body's acceptance criterion (a): a simulated failed nightly ``/daily`` (no
``logs/daily/<date>.md`` written) triggers the heartbeat alert. We drive the
bash wrapper via subprocess with env-var overrides (daily dir, sentinel dir, log
dir, "yesterday" date, and a FAKE telegram_push.sh that records each call), then
assert:

- missing-yesterday-log → push attempted exactly once + sentinel written;
- a present+fresh ENRICHED yesterday-log → NO push;
- a second invocation the same day after a push → NO push (sentinel suppresses);
- #1189 husk arm: a present+fresh file whose ``## Applied workflow
  improvements`` section is missing or empty (the stub-first skeleton, never
  enriched) → exactly one push with a husk-classed message; the placeholder
  line counts as enriched (quiet-day false-alarm guard); the shared per-date
  sentinel suppresses across alert classes; a husk that is ALSO stale alerts
  via the stale arm first (elif precedence).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parent.parent / "scripts" / "cron_daily_healthcheck.sh"


def _skeleton(d: str) -> str:
    """The #1189 stub-first skeleton: frontmatter + the six H2 headers, bodies EMPTY."""
    return (
        f"---\nkind: daily\ndate: {d}\ntitle: Daily — {d}\nincluded_tasks: []\n"
        "visible: false\n---\n\n"
        "## What happened\n\n## Applied workflow improvements\n\n"
        "## Other problems & notes\n\n## Living-docs drift\n\n"
        "## My thoughts\n\n## Highlighted results\n"
    )


def _enriched(d: str) -> str:
    """A skeleton whose Applied section carries a real entry (the run enriched it)."""
    return _skeleton(d).replace(
        "## Applied workflow improvements\n\n",
        "## Applied workflow improvements\n\n1. fixed X (sha abc123)\n\n",
    )


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
    """A present + fresh + ENRICHED logs/daily/<yesterday>.md → no push, no sentinel.

    Fixture updated for #1189 (deliberate semantic change, not a regression):
    the prior fixture ``"# daily brief\\n"`` has no ``## Applied workflow
    improvements`` section and is a HUSK under the new content predicate, so
    the healthy-day fixture is now an enriched skeleton. Every assertion is
    unchanged.
    """
    yesterday = "2026-06-27"
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_enriched(yesterday))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 0
    assert not (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()


# ── #1189 husk arm: present + fresh but never enriched ────────────────────────


def test_stub_only_fresh_file_triggers_husk_push(harness):
    """A fresh stub-first skeleton (empty Applied section) → one husk-classed push."""
    yesterday = "2026-06-27"
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_skeleton(yesterday))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    message = harness["push_log"].read_text()
    assert yesterday in message
    assert "stub" in message
    # #994 pinned invariants: the paste-ready backfill command survives in the
    # husk-classed message too.
    assert "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=" in message
    assert f"claude -p '/daily {yesterday}'" in message
    assert "backfill: cd " in message
    assert (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()


def test_placeholder_applied_line_is_enriched_no_push(harness):
    """The quiet-day placeholder line counts as enriched → NO push (kill-criterion guard)."""
    yesterday = "2026-06-27"
    content = _skeleton(yesterday).replace(
        "## Applied workflow improvements\n\n",
        "## Applied workflow improvements\n\n- _no workflow-fixable problems found today_\n\n",
    )
    (harness["daily_dir"] / f"{yesterday}.md").write_text(content)
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 0
    assert not (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()


def test_missing_applied_h2_triggers_husk_push(harness):
    """A fresh file with prose but NO Applied H2 at all is a husk → one push."""
    yesterday = "2026-06-27"
    (harness["daily_dir"] / f"{yesterday}.md").write_text(
        "# daily brief\n\nsome prose, no Applied section\n"
    )
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    assert "stub" in harness["push_log"].read_text()
    assert (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()


def test_husk_second_invocation_sentinel_suppresses(harness):
    """A husk alerts once; a second run the same day is sentinel-suppressed."""
    yesterday = "2026-06-27"
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_skeleton(yesterday))
    harness["run"](yesterday)
    assert _push_count(harness["push_log"]) == 1
    result2 = harness["run"](yesterday)
    assert result2.returncode == 0
    assert _push_count(harness["push_log"]) == 1


def test_missing_then_husk_same_day_single_push(harness):
    """The per-date sentinel suppresses ACROSS alert classes (missing → husk)."""
    yesterday = "2026-06-27"
    # First run: file missing → 1 push (missing class) + sentinel.
    harness["run"](yesterday)
    assert _push_count(harness["push_log"]) == 1
    # A late skeleton appears (still a husk); same-day re-run → still 1 push.
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_skeleton(yesterday))
    result2 = harness["run"](yesterday)
    assert result2.returncode == 0
    assert _push_count(harness["push_log"]) == 1


def test_stale_husk_alerts_with_stale_message(harness):
    """A husk-shaped file with mtime > 25h alerts via the STALE arm (elif precedence)."""
    yesterday = "2026-06-27"
    path = harness["daily_dir"] / f"{yesterday}.md"
    path.write_text(_skeleton(yesterday))
    old = time.time() - 26 * 3600
    os.utime(path, (old, old))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    message = harness["push_log"].read_text()
    # The stale/missing message, NOT the husk-classed one.
    assert "did not land" in message
    assert "left only a stub" not in message
    assert (harness["sentinel_dir"] / f"sent-{yesterday}.flag").exists()
