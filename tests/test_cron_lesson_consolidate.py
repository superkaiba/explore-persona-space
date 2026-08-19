"""Tests for scripts/cron_lesson_consolidate.sh's exit-3 alert arm (task #2190).

#2189 gave ``consolidate_lessons.py`` exit code 3 = "gotcha promotion refused —
appending would push .claude/rules/gotchas.md past GOTCHAS_SIZE_WARN_BYTES".
The wrapper used to capture that rc into a log nobody reads and ``exit 0`` it
away. The new arm keys STRICTLY on rc 3 and fires a Telegram push (fail-soft,
deduped by a per-date sentinel) plus one audit sidecar JSON row; rc=0 passes
stay silent and the wrapper's ``exit 0`` is unconditional.

Shape copied from tests/test_cron_daily_healthcheck.py (plan §2.7): we drive
the bash wrapper via subprocess with env overrides — a FAKE telegram_push.sh
recording ``$1``, and a STUB consolidator wired via
``EPS_LESSON_CONSOLIDATE_BIN`` whose exit code (and optional counts line) each
test chooses, so no test ever executes the real 7-day consolidation pass or
touches gotchas.md. Every run redirects ALL FIVE env overrides into tmp_path
(log dir, sentinel dir, sidecar, push script, consolidator bin) — a missing
override in any test is itself a defect (plan §6).

T1-T9 pin acceptance criteria A1-A7 plus the §4.2 sidecar-before-sentinel
ordering and the rc-discrimination guard:

- T1 ``test_rc3_pushes_once_and_writes_sentinel`` — A1; the plan §6 fail-loud
  pin: an exit-3 budget refusal is NOT silently swallowed.
- T2 ``test_rc0_no_push_no_sentinel`` — A2; the silent-path twin.
- T3 ``test_second_run_same_date_does_not_repush`` — A3.
- T4 ``test_push_failure_leaves_no_sentinel`` — A4 (next pass retries).
- T5 ``test_missing_push_script_logs_and_exits_zero`` — A5.
- T6 ``test_wrapper_exit_status_is_zero_on_rc3`` — A6.
- T7 ``test_alert_names_refused_count_and_degrades_to_unknown`` — A7; the
  count is parsed from the stub's stdout counts line via the wrapper's real
  capture route (the brace group's ``>> $LOG_FILE 2>&1`` — production's
  stderr INFO summary line rides the same redirect, plan §2.2), incl. the
  ``tail -1`` last-match choice; no counts line degrades to ``unknown``.
- T8 ``test_sidecar_row_written_even_when_sentinel_suppresses`` — §4.2
  ordering: the sidecar row lands BEFORE the sentinel check.
- T9 ``test_nonzero_rc_other_than_three_does_not_alert`` — rc 1 and rc 2
  produce zero pushes / no sentinel / no sidecar row, so a drift from
  ``-eq 3`` to ``-ne 0`` cannot ship green (critic S1).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parent.parent / "scripts" / "cron_lesson_consolidate.sh"

# A realistic #2189 stderr-INFO-summary-shaped counts line (only the
# promote_refused_budget=<n> key=value field is load-bearing for the parse).
_COUNTS_LINE_2 = "INFO consolidate_lessons: deduped=0 promoted=0 pruned=0 promote_refused_budget=2"


@pytest.fixture
def make_harness(tmp_path: Path):
    """Factory building an isolated wrapper harness (fresh dirs per call).

    All five env overrides point into tmp_path so no test touches the real
    log dir, the real .claude/cache/, the real telegram_push.sh, or
    gotchas.md. The stub consolidator records its argv to ``stub_calls.log``,
    optionally emits ``counts_line`` on stdout (captured into ``$LOG_FILE`` by
    the wrapper's brace-group redirect — the same route production's stderr
    INFO summary line takes), then exits ``stub_rc``.
    """
    counter = itertools.count()

    def _make(
        stub_rc: int = 3,
        counts_line: str | None = None,
        push_exit: int = 0,
        push_missing: bool = False,
    ) -> dict:
        root = tmp_path / f"h{next(counter)}"
        log_dir = root / "logs"
        sentinel_dir = root / "sentinels"
        cache_dir = root / "cache"
        log_dir.mkdir(parents=True)
        sentinel_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        sidecar = cache_dir / "lesson-consolidate-events.jsonl"

        push_log = root / "push_calls.log"
        fake_push = root / "fake_telegram_push.sh"
        # The failing form still records its call so tests can assert an
        # attempt happened even when the push "fails".
        fake_push.write_text(f'#!/bin/bash\necho "$1" >> "{push_log}"\nexit {push_exit}\n')
        fake_push.chmod(0o755)

        stub_calls = root / "stub_calls.log"
        stub = root / "fake_consolidator.sh"
        stub_lines = ["#!/bin/bash", f'echo "$@" >> "{stub_calls}"']
        if counts_line is not None:
            stub_lines.append(f'echo "{counts_line}"')
        stub_lines.append(f"exit {stub_rc}")
        stub.write_text("\n".join(stub_lines) + "\n")
        stub.chmod(0o755)

        push_script = root / "no_such_push.sh" if push_missing else fake_push
        env = {
            **os.environ,
            "EPS_LESSON_CONSOLIDATE_LOG_DIR": str(log_dir),
            "EPS_LESSON_CONSOLIDATE_SENTINEL_DIR": str(sentinel_dir),
            "EPS_LESSON_CONSOLIDATE_SIDECAR": str(sidecar),
            "EPS_TELEGRAM_PUSH_SCRIPT": str(push_script),
            "EPS_LESSON_CONSOLIDATE_BIN": str(stub),
        }

        def run() -> subprocess.CompletedProcess:
            return subprocess.run(
                ["bash", str(_WRAPPER)], env=env, capture_output=True, text=True, check=False
            )

        return {
            "log_dir": log_dir,
            "sentinel_dir": sentinel_dir,
            "sidecar": sidecar,
            "push_log": push_log,
            "stub_calls": stub_calls,
            "run": run,
        }

    return _make


def _push_count(push_log: Path) -> int:
    if not push_log.exists():
        return 0
    return len([ln for ln in push_log.read_text().splitlines() if ln.strip()])


def _sentinels(h: dict) -> list[Path]:
    return sorted(h["sentinel_dir"].glob("refused-*.flag"))


def _daily_log(h: dict) -> str:
    logs = sorted(h["log_dir"].glob("*.log"))
    assert len(logs) == 1, f"expected exactly one daily log, found {logs}"
    return logs[0].read_text()


def _sidecar_rows(h: dict) -> list[dict]:
    if not h["sidecar"].exists():
        return []
    return [json.loads(ln) for ln in h["sidecar"].read_text().splitlines() if ln.strip()]


# ── T1 / T6: the fail-loud pin — rc=3 alerts, and the wrapper still exits 0 ──


def test_rc3_pushes_once_and_writes_sentinel(make_harness):
    """A1 (plan §6 fail-loud pin): an exit-3 budget refusal is NOT silently
    swallowed — exactly one push is attempted and a per-date sentinel written."""
    h = make_harness(stub_rc=3, counts_line=_COUNTS_LINE_2)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 1
    message = h["push_log"].read_text()
    assert "gotchas.md" in message
    assert "refused 2 gotcha bullet(s)" in message
    # $LOG_FILE is expanded by the wrapper's shell before the message reaches
    # telegram_push.sh (which does no expansion of the body, plan §2.5).
    assert str(h["log_dir"]) in message
    assert len(_sentinels(h)) == 1


def test_wrapper_exit_status_is_zero_on_rc3(make_harness):
    """A6: the wrapper exits 0 on a child rc=3 — a propagated rc notifies
    nobody here (no MTA, crontab redirects 2>&1; plan §2.3)."""
    h = make_harness(stub_rc=3)  # no counts line — the degraded path also exits 0
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 1


# ── T2: the silent-path twin ─────────────────────────────────────────────────


def test_rc0_no_push_no_sentinel(make_harness):
    """A2: child rc=0 → zero pushes, no sentinel, no sidecar row, and the
    wrapper's observable output is structurally unchanged from today."""
    h = make_harness(stub_rc=0)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 0
    assert _sentinels(h) == []
    assert _sidecar_rows(h) == []
    log = _daily_log(h)
    assert "lesson_consolidate start" in log
    assert "exit=0" in log
    assert "ALERT" not in log
    # First run of the day still emits the daily pointer line on stdout.
    assert "per-pass output" in result.stdout


# ── T3: per-date sentinel dedup ──────────────────────────────────────────────


def test_second_run_same_date_does_not_repush(make_harness):
    """A3: a second same-date run after a pushed alert is sentinel-suppressed."""
    h = make_harness(stub_rc=3, counts_line=_COUNTS_LINE_2)
    h["run"]()
    assert _push_count(h["push_log"]) == 1
    result2 = h["run"]()
    assert result2.returncode == 0
    assert _push_count(h["push_log"]) == 1
    assert "skipping re-alert" in _daily_log(h)


# ── T4 / T5: fail-soft push handling ─────────────────────────────────────────


def test_push_failure_leaves_no_sentinel(make_harness):
    """A4: telegram_push.sh exiting non-zero → NO sentinel (so the next pass
    retries), the failure is logged, and the wrapper still exits 0."""
    h = make_harness(stub_rc=3, counts_line=_COUNTS_LINE_2, push_exit=1)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 1  # attempted (the failing fake records)
    assert _sentinels(h) == []
    log = _daily_log(h)
    assert "telegram_push.sh FAILED" in log
    assert "no sentinel written" in log


def test_missing_push_script_logs_and_exits_zero(make_harness):
    """A5: a missing / non-executable telegram_push.sh is logged, no crash,
    no sentinel, wrapper exits 0."""
    h = make_harness(stub_rc=3, counts_line=_COUNTS_LINE_2, push_missing=True)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 0
    assert _sentinels(h) == []
    assert "not executable" in _daily_log(h)


# ── T7: count parsing + the `unknown` degradation ────────────────────────────


def test_alert_names_refused_count_and_degrades_to_unknown(make_harness):
    """A7: a parseable counts line names the refused count; a missing line
    degrades to the literal `unknown` — never a crash, never an empty field;
    `tail -1` reads the CURRENT run's line past stale earlier matches."""
    # (a) Parseable counts line → the alert names the count.
    counts4 = "INFO consolidate_lessons: deduped=1 promoted=0 pruned=0 promote_refused_budget=4"
    h1 = make_harness(stub_rc=3, counts_line=counts4)
    assert h1["run"]().returncode == 0
    assert "refused 4 gotcha bullet(s)" in h1["push_log"].read_text()

    # (b) No counts line at all → the literal `unknown` token reaches the push.
    h2 = make_harness(stub_rc=3, counts_line=None)
    assert h2["run"]().returncode == 0
    msg2 = h2["push_log"].read_text()
    assert "refused unknown gotcha bullet(s)" in msg2
    assert "refused  gotcha" not in msg2  # never an empty field

    # (c) tail -1 takes the LAST match: a stale earlier line in the same daily
    #     log (e.g. a manual re-run's promote_refused_budget=0) must not win.
    h3 = make_harness(stub_rc=3, counts_line=counts4.replace("budget=4", "budget=5"))
    today = time.strftime("%Y-%m-%d")
    stale = h3["log_dir"] / f"{today}.log"
    stale.write_text("old run: promote_refused_budget=0\n")
    assert h3["run"]().returncode == 0
    assert "refused 5 gotcha bullet(s)" in h3["push_log"].read_text()


# ── T8: sidecar row ordering (before the sentinel check) ─────────────────────


def test_sidecar_row_written_even_when_sentinel_suppresses(make_harness):
    """§4.2 ordering: the audit sidecar row is appended BEFORE the sentinel
    check, so a sentinel-suppressed re-alert still leaves a row (the sentinel
    dedups the buzz, not the record)."""
    h = make_harness(stub_rc=3, counts_line=_COUNTS_LINE_2)
    h["run"]()
    rows1 = _sidecar_rows(h)
    assert len(rows1) == 1
    assert rows1[0]["event"] == "promote_refused_budget"
    assert rows1[0]["refused"] == "2"
    assert rows1[0]["rc"] == 3
    # Second same-date run: push suppressed, but a second audit row lands.
    h["run"]()
    assert _push_count(h["push_log"]) == 1
    rows2 = _sidecar_rows(h)
    assert len(rows2) == 2


# ── T9: rc discrimination — the arm keys on -eq 3, not -ne 0 ─────────────────


@pytest.mark.parametrize("other_rc", [1, 2])
def test_nonzero_rc_other_than_three_does_not_alert(make_harness, other_rc):
    """rc-discrimination (critic S1): a generic crash rc (1, 2) produces zero
    pushes, no sentinel, and no sidecar row — even with a parseable
    promote_refused_budget line sitting in the log — so an implementer drift
    from `-eq 3` to `-ne 0` cannot pass this suite."""
    h = make_harness(stub_rc=other_rc, counts_line=_COUNTS_LINE_2)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 0
    assert _sentinels(h) == []
    assert _sidecar_rows(h) == []
    assert f"exit={other_rc}" in _daily_log(h)
