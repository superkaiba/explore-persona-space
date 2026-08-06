"""Tests for scripts/cron_daily_healthcheck.sh (tasks #711, #994, #1189, #2113).

The body's acceptance criterion (a): a simulated failed nightly ``/daily`` (no
``logs/daily/<date>.md`` written) triggers the heartbeat alert. We drive the
bash wrapper via subprocess with env-var overrides (daily dir, sentinel dir, log
dir, "yesterday" date, a FAKE telegram_push.sh that records each call, and — as
of #2113 — a FAKE claude bin wired via ``EPS_HEALTHCHECK_CLAUDE_BIN``; the fake
is MANDATORY in the shared harness because the real ``$HOME/.local/bin/claude``
exists on this VM and must never be exec'd by a test), then assert:

- missing-yesterday-log → push attempted exactly once + sentinel written;
- a present+fresh ENRICHED yesterday-log → NO push;
- a second invocation the same day after a push → NO push (sentinel suppresses);
- #1189 husk arm: a present+fresh file whose ``## Applied workflow
  improvements`` section is missing or empty (the stub-first skeleton, never
  enriched) → exactly one push with a husk-classed message; the placeholder
  line counts as enriched (quiet-day false-alarm guard); the shared per-date
  sentinel suppresses across alert classes; a husk that is ALSO stale alerts
  via the stale arm first (elif precedence);
- #2113 auto-backfill arm: a missing/husk detection launches the (fake) claude
  bin detached, exactly once per date (``backfill-attempt-<date>.flag``);
  ``stale`` stays alert-only; ``EPS_HEALTHCHECK_AUTO_BACKFILL=0`` restores the
  manual-command alert form (the #994 paste-ready-command invariant is pinned
  THERE now); a later run that finds an attempted date still missing/husk
  pushes a one-time "auto-backfill FAILED" alert and never relaunches; the
  sweep skips ``$YESTERDAY``'s own in-flight attempt; a held ``backfill.lock``
  makes the detached flock-wrapped launch a no-op (single-flight).

Detached-launch assertions poll for the fake bin's marker with a <=5s deadline;
"no launch" assertions hold a shorter poll-then-assert-absent window.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parent.parent / "scripts" / "cron_daily_healthcheck.sh"

_POLL_DEADLINE_S = 5.0  # detached-launch artifacts must appear within this
_ABSENT_WINDOW_S = 1.5  # window over which a no-launch artifact must stay absent


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
    """A tmp project layout + fake telegram_push.sh + fake claude bin.

    The fake claude bin (#2113) records its argv to ``claude_calls.log`` and
    touches ``claude_ran.marker`` (in that order, so a present marker implies a
    complete calls line), then exits 0. It is wired into EVERY run via
    ``EPS_HEALTHCHECK_CLAUDE_BIN`` so no test can ever exec the real binary.
    The file is named ``claude`` inside its own dir so message substrings
    (``claude -p '/daily <date>'``) stay realistic.
    """
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

    claude_calls = tmp_path / "claude_calls.log"
    claude_marker = tmp_path / "claude_ran.marker"
    fake_claude_dir = tmp_path / "fakebin"
    fake_claude_dir.mkdir()
    fake_claude = fake_claude_dir / "claude"
    fake_claude.write_text(
        f'#!/bin/bash\necho "$@" >> "{claude_calls}"\ntouch "{claude_marker}"\nexit 0\n'
    )
    fake_claude.chmod(0o755)

    def run(
        yesterday: str = "2026-06-27", extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "EPS_HEALTHCHECK_PROJECT_DIR": str(project),
            "EPS_HEALTHCHECK_YESTERDAY": yesterday,
            "EPS_HEALTHCHECK_DAILY_DIR": str(daily_dir),
            "EPS_HEALTHCHECK_SENTINEL_DIR": str(sentinel_dir),
            "EPS_HEALTHCHECK_LOG_DIR": str(log_dir),
            "EPS_TELEGRAM_PUSH_SCRIPT": str(fake_push),
            "EPS_HEALTHCHECK_CLAUDE_BIN": str(fake_claude),
        }
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True, check=False
        )

    return {
        "project": project,
        "daily_dir": daily_dir,
        "sentinel_dir": sentinel_dir,
        "push_log": push_log,
        "claude_calls": claude_calls,
        "claude_marker": claude_marker,
        "run": run,
    }


def _push_count(push_log: Path) -> int:
    if not push_log.exists():
        return 0
    return len([ln for ln in push_log.read_text().splitlines() if ln.strip()])


def _claude_calls(claude_calls: Path) -> list[str]:
    """Non-empty argv lines the fake claude bin recorded (one per invocation)."""
    if not claude_calls.exists():
        return []
    return [ln for ln in claude_calls.read_text().splitlines() if ln.strip()]


def _poll_for(path: Path, deadline_s: float = _POLL_DEADLINE_S) -> bool:
    """Poll for a detached-launch artifact; True the moment it exists."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def _assert_stays_absent(path: Path, window_s: float = _ABSENT_WINDOW_S) -> None:
    """Assert a detached-launch artifact never appears within the window."""
    end = time.monotonic() + window_s
    while time.monotonic() < end:
        assert not path.exists(), f"{path} appeared — launch should not have happened"
        time.sleep(0.05)
    assert not path.exists(), f"{path} appeared — launch should not have happened"


def test_missing_yesterday_log_triggers_push_and_writes_sentinel(harness):
    """No logs/daily/<yesterday>.md → exactly one push + a sentinel flag."""
    yesterday = "2026-06-27"
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    # The message names the missing day.
    message = harness["push_log"].read_text()
    assert yesterday in message
    # #2113: the launch path is active by default, so the alert reports the
    # auto-backfill (log pointer included) instead of the manual paste-ready
    # command. The #994 paste-ready-command invariant now lives in
    # test_kill_switch_restores_alert_only (the manual-form path).
    assert "auto-backfill launched (attempt 1)" in message
    assert f"backfill-{yesterday}.log" in message
    assert "daily_retrospective.log" in message  # retrospective pointer kept (plan D5)
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
    # #2113: launch path active by default → the husk-classed message carries
    # the launched form too (the #994 manual-command pin moved to the
    # kill-switch test).
    assert "auto-backfill launched (attempt 1)" in message
    assert f"backfill-{yesterday}.log" in message
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


# ── #2113 auto-backfill arm: launch + kill switch + failure sweep ─────────────


def test_missing_triggers_auto_backfill_launch(harness):
    """Missing yesterday-file → the (fake) claude bin runs detached exactly once
    with ``/daily <date>``, the attempt sentinel is written, and the alert
    reports the launch."""
    yesterday = "2026-06-27"
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _poll_for(harness["claude_marker"]), "detached fake claude never ran"
    assert _claude_calls(harness["claude_calls"]) == [f"-p /daily {yesterday}"]
    assert (harness["sentinel_dir"] / f"backfill-attempt-{yesterday}.flag").exists()
    message = harness["push_log"].read_text()
    assert "auto-backfill launched (attempt 1)" in message
    assert f"backfill-{yesterday}.log" in message


def test_husk_triggers_auto_backfill_launch(harness):
    """A fresh husk → same detached launch + attempt sentinel as the missing class."""
    yesterday = "2026-06-27"
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_skeleton(yesterday))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _poll_for(harness["claude_marker"]), "detached fake claude never ran"
    assert _claude_calls(harness["claude_calls"]) == [f"-p /daily {yesterday}"]
    assert (harness["sentinel_dir"] / f"backfill-attempt-{yesterday}.flag").exists()
    assert "auto-backfill launched (attempt 1)" in harness["push_log"].read_text()


def test_stale_stays_alert_only(harness):
    """The stale class is alert-only (plan D3): push happens with the manual
    command form; NO claude launch; no attempt sentinel."""
    yesterday = "2026-06-27"
    path = harness["daily_dir"] / f"{yesterday}.md"
    path.write_text(_enriched(yesterday))
    old = time.time() - 26 * 3600
    os.utime(path, (old, old))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    message = harness["push_log"].read_text()
    assert "backfill: cd " in message
    assert f"claude -p '/daily {yesterday}'" in message
    assert "auto-backfill launched" not in message
    _assert_stays_absent(harness["claude_marker"])
    assert not (harness["sentinel_dir"] / f"backfill-attempt-{yesterday}.flag").exists()


def test_second_run_does_not_relaunch(harness):
    """The attempt sentinel gives exactly ONE auto-attempt per date: a same-day
    re-run neither relaunches (attempt flag) nor re-pushes (sent flag)."""
    yesterday = "2026-06-27"
    harness["run"](yesterday)
    assert _poll_for(harness["claude_marker"]), "first run should have launched"
    assert len(_claude_calls(harness["claude_calls"])) == 1
    result2 = harness["run"](yesterday)
    assert result2.returncode == 0
    # Poll-then-assert: the call count must STAY 1 over the absence window.
    end = time.monotonic() + _ABSENT_WINDOW_S
    while time.monotonic() < end:
        assert len(_claude_calls(harness["claude_calls"])) == 1, "second run relaunched"
        time.sleep(0.05)
    assert _push_count(harness["push_log"]) == 1


def test_kill_switch_restores_alert_only(harness):
    """EPS_HEALTHCHECK_AUTO_BACKFILL=0 disables the launch; the alert reverts to
    the current manual-command form (the #994 paste-ready-command invariant —
    env-var name, /daily invocation, cd-into-project prefix — is pinned HERE
    now that the default-path messages carry the launched form instead)."""
    yesterday = "2026-06-27"
    result = harness["run"](yesterday, extra_env={"EPS_HEALTHCHECK_AUTO_BACKFILL": "0"})
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    message = harness["push_log"].read_text()
    assert "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=" in message
    assert f"claude -p '/daily {yesterday}'" in message
    assert "backfill: cd " in message
    assert "auto-backfill launched" not in message
    _assert_stays_absent(harness["claude_marker"])
    assert not (harness["sentinel_dir"] / f"backfill-attempt-{yesterday}.flag").exists()


def test_failed_backfill_realerts_on_later_run(harness):
    """An attempt-flagged OLDER date whose file is still a husk → one distinct
    FAILED alert (own sentinel; the sent-<date> flag does not suppress it) and
    NO relaunch. Yesterday's own file is enriched so the yesterday arm stays
    quiet and the expected push count is exactly 1."""
    yesterday = "2026-06-27"
    failed_date = "2026-06-25"
    (harness["sentinel_dir"] / f"backfill-attempt-{failed_date}.flag").touch()
    # The prior auto-attempt's alert sentinel exists too — it must NOT suppress
    # the distinct FAILED alert (plan criterion 5).
    (harness["sentinel_dir"] / f"sent-{failed_date}.flag").touch()
    (harness["daily_dir"] / f"{failed_date}.md").write_text(_skeleton(failed_date))
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_enriched(yesterday))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 1
    message = harness["push_log"].read_text()
    assert f"auto-backfill for {failed_date} FAILED" in message
    assert "manual backfill:" in message
    assert f"claude -p '/daily {failed_date}'" in message
    assert (harness["sentinel_dir"] / f"backfill-failed-sent-{failed_date}.flag").exists()
    _assert_stays_absent(harness["claude_marker"])  # never relaunches


def test_failed_backfill_alert_fires_once(harness):
    """The failure alert has its own per-date sentinel: a second run does not re-push."""
    yesterday = "2026-06-27"
    failed_date = "2026-06-25"
    (harness["sentinel_dir"] / f"backfill-attempt-{failed_date}.flag").touch()
    (harness["daily_dir"] / f"{failed_date}.md").write_text(_skeleton(failed_date))
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_enriched(yesterday))
    harness["run"](yesterday)
    assert _push_count(harness["push_log"]) == 1
    result2 = harness["run"](yesterday)
    assert result2.returncode == 0
    assert _push_count(harness["push_log"]) == 1


def test_recovered_backfill_no_failure_alert(harness):
    """An attempt-flagged date whose file is now ENRICHED (the backfill worked)
    → no failure alert at all."""
    yesterday = "2026-06-27"
    recovered_date = "2026-06-25"
    (harness["sentinel_dir"] / f"backfill-attempt-{recovered_date}.flag").touch()
    (harness["daily_dir"] / f"{recovered_date}.md").write_text(_enriched(recovered_date))
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_enriched(yesterday))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    assert _push_count(harness["push_log"]) == 0
    assert not (harness["sentinel_dir"] / f"backfill-failed-sent-{recovered_date}.flag").exists()


def test_same_day_rerun_does_not_false_fail_yesterday(harness):
    """The sweep SKIPS $YESTERDAY's own attempt flag: an in-flight same-morning
    backfill (attempt flag + still-husk file) is not judged failed by a
    same-day re-run — no FAILED alert, no failure sentinel, no relaunch."""
    yesterday = "2026-06-27"
    (harness["sentinel_dir"] / f"backfill-attempt-{yesterday}.flag").touch()
    (harness["daily_dir"] / f"{yesterday}.md").write_text(_skeleton(yesterday))
    result = harness["run"](yesterday)
    assert result.returncode == 0
    message = harness["push_log"].read_text() if harness["push_log"].exists() else ""
    assert "FAILED" not in message
    assert not (harness["sentinel_dir"] / f"backfill-failed-sent-{yesterday}.flag").exists()
    # The yesterday husk arm still alerts (once), with the already-attempted
    # wording, and does not relaunch.
    assert _push_count(harness["push_log"]) == 1
    assert "auto-backfill already attempted" in message
    _assert_stays_absent(harness["claude_marker"])


def test_lock_held_skips_launch(harness):
    """Single-flight (criterion 3): with backfill.lock held, the detached
    flock-wrapped launch exits without running claude. The attempt sentinel is
    still written (plan D4 — the failure sweep catches a lock-blocked attempt
    on a later run)."""
    yesterday = "2026-06-27"
    lock = harness["sentinel_dir"] / "backfill.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "30"])
    try:
        # Wait until the holder actually owns the lock (avoid the startup race).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            probe = subprocess.run(
                ["flock", "-n", str(lock), "true"], check=False, capture_output=True
            )
            if probe.returncode != 0:
                break
            time.sleep(0.05)
        else:
            pytest.fail("background flock holder never acquired the lock")
        result = harness["run"](yesterday)
        assert result.returncode == 0
        _assert_stays_absent(harness["claude_marker"], window_s=2.0)
        assert (harness["sentinel_dir"] / f"backfill-attempt-{yesterday}.flag").exists()
    finally:
        holder.terminate()
        holder.wait(timeout=10)
