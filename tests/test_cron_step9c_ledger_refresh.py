"""Tests for scripts/cron_step9c_ledger_refresh.sh (task #2114).

The wrapper runs ``scripts/step9c_baseline.py refresh --json`` nightly so
/issue sessions start each day against a fresh Step 9c known-red baseline
ledger instead of paying the ~31-40 min refresh mid-gate (#2105, #1992,
#2106). On refresh rc != 0 (rc=2 = timeout / junit-parse failure / zero
collected — NO ledger write) it appends one audit sidecar JSON row and fires
a once-per-day sentinel-gated Telegram push; rc=0 passes stay silent and the
wrapper ALWAYS exits 0.

Shape copied from tests/test_cron_lesson_consolidate.py: we drive the bash
wrapper via subprocess with env overrides — a FAKE telegram_push.sh recording
``$1``, and a STUB refresh wired via ``EPS_STEP9C_REFRESH_BIN`` whose exit
code each test chooses — so no test ever runs the real pytest universe or
touches the real ledger/Telegram/network. Every run redirects ALL FIVE env
overrides into tmp_path (log dir, sentinel dir, sidecar, push script,
refresh bin) — a missing override in any test is itself a defect.

T1-T7 pin the plan's acceptance-criterion-5 behaviors:

- T1 ``test_rc0_no_push_no_sentinel_no_sidecar`` — the silent path: log
  written, no push, no sentinel, no sidecar row, exit 0; the per-day pointer
  line reaches stdout.
- T2 ``test_rc2_pushes_once_writes_sentinel_and_sidecar`` — the fail-loud
  pin: a no-ledger-write refresh failure is NOT silently swallowed.
- T3 ``test_second_run_same_date_does_not_repush`` — per-date sentinel dedup;
  the sidecar still gains a second row (sentinel dedups the buzz, not the
  record).
- T4 ``test_push_failure_leaves_no_sentinel`` — a failed push writes NO
  sentinel (next pass retries), wrapper exits 0.
- T5 ``test_missing_push_script_logs_and_exits_zero`` — a missing/
  non-executable push script is logged, no crash, no sentinel.
- T6 ``test_daily_pointer_line_only_on_first_run_of_day`` — the pointer line
  lands on stdout exactly once per date.
- T7 ``test_stub_receives_refresh_json_args`` — the wrapper invokes the
  refresh with the exact ``refresh --json`` argv (pins the CLI shape against
  step9c_baseline.py drift).

T8-T10 pin the #2386 fail-loud log-dir guard (Pattern B — ``fatal()`` also
fires a best-effort ``TELEGRAM_PUSH``). Before #2386 an uncreatable or
unwritable log dir made the brace-group redirect fail, so the refresh never
ran AND the rc!=0 alert arm — which redirects into the same ``$LOG_FILE`` —
was double-silent, while the wrapper still exited 0. ``make_harness`` gains a
``log_dir_setup`` kwarg (``"ok"`` / ``"uncreatable"`` = the log-dir path under
a regular FILE so ``mkdir -p`` fails ENOTDIR / ``"unwritable"`` = an existing
dir at 0o555 so ``mkdir -p`` passes and the appendability probe fails; root
bypasses mode bits, hence the skipif). Only LOG_DIR varies — SENTINEL_DIR
stays writable, so the failure is unambiguously attributable to the log dir.

- T8 ``test_uncreatable_log_dir_fails_loud_and_pushes`` — rc != 0, stderr
  FATAL naming the dir, exactly one ALERT push, refresh NEVER invoked.
- T9 ``test_unwritable_log_dir_fails_loud_and_pushes`` (root-skipif) — same
  via the probe arm, distinguished by the "not appendable" message.
- T10 ``test_missing_push_script_still_exits_non_zero`` — the push leg is
  BEST-EFFORT: a missing telegram_push.sh must not swallow the failure, the
  wrapper still exits non-zero with the FATAL on stderr.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parent.parent / "scripts" / "cron_step9c_ledger_refresh.sh"

# #2386 fatal-message fragments. The mkdir arm and the probe arm are
# distinguished by these, NOT by the dated log filename (which would race a
# midnight rollover). Spelled locally rather than imported: each cron test
# file weaves the guard into its own harness idiom.
_MKDIR_FATAL = "cannot create log/sentinel dir"
_PROBE_FATAL = "not appendable"
_ALERT_PREFIX = "ALERT: step9c_ledger_refresh:"


@pytest.fixture
def make_harness(tmp_path: Path):
    """Factory building an isolated wrapper harness (fresh dirs per call).

    All five env overrides point into tmp_path so no test touches the real
    log dir, the real .claude/cache/, the real telegram_push.sh, or the real
    ledger. The stub refresh records its argv to ``stub_calls.log`` and exits
    ``stub_rc``.

    ``log_dir_setup`` (#2386) selects the LOG_DIR failure mode — ``"ok"``
    (created, the default every T1-T7 call uses), ``"uncreatable"`` (under a
    regular FILE → ``mkdir -p`` fails ENOTDIR) or ``"unwritable"`` (existing,
    0o555 → ``mkdir -p`` passes, the appendability probe fails). Every call
    gets its own ``h<N>`` root, so no arm can pollute a sibling's assertions.
    """
    counter = itertools.count()

    def _make(
        stub_rc: int = 2,
        push_exit: int = 0,
        push_missing: bool = False,
        log_dir_setup: str = "ok",
    ) -> dict:
        root = tmp_path / f"h{next(counter)}"
        sentinel_dir = root / "sentinels"
        cache_dir = root / "cache"
        # Created FIRST so `root` exists for the uncreatable blocker below;
        # SENTINEL_DIR stays writable in every mode, so a mkdir/probe failure
        # is attributable to LOG_DIR alone.
        sentinel_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        if log_dir_setup == "ok":
            log_dir = root / "logs"
            log_dir.mkdir(parents=True)
        elif log_dir_setup == "uncreatable":
            blocker = root / "blocker"
            blocker.write_text("regular file blocking mkdir -p (ENOTDIR)\n")
            log_dir = blocker / "logs"
        elif log_dir_setup == "unwritable":
            log_dir = root / "logs"
            log_dir.mkdir(parents=True)
            log_dir.chmod(0o555)
        else:  # pragma: no cover — harness misuse is a test defect
            raise ValueError(f"unknown log_dir_setup: {log_dir_setup!r}")
        sidecar = cache_dir / "step9c-refresh-cron-events.jsonl"

        push_log = root / "push_calls.log"
        fake_push = root / "fake_telegram_push.sh"
        # The failing form still records its call so tests can assert an
        # attempt happened even when the push "fails".
        fake_push.write_text(f'#!/bin/bash\necho "$1" >> "{push_log}"\nexit {push_exit}\n')
        fake_push.chmod(0o755)

        stub_calls = root / "stub_calls.log"
        stub = root / "fake_refresh.sh"
        stub.write_text(f'#!/bin/bash\necho "$@" >> "{stub_calls}"\nexit {stub_rc}\n')
        stub.chmod(0o755)

        push_script = root / "no_such_push.sh" if push_missing else fake_push
        env = {
            **os.environ,
            "EPS_STEP9C_REFRESH_LOG_DIR": str(log_dir),
            "EPS_STEP9C_REFRESH_SENTINEL_DIR": str(sentinel_dir),
            "EPS_STEP9C_REFRESH_SIDECAR": str(sidecar),
            "EPS_TELEGRAM_PUSH_SCRIPT": str(push_script),
            "EPS_STEP9C_REFRESH_BIN": str(stub),
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
    return sorted(h["sentinel_dir"].glob("failed-*.flag"))


def _daily_log(h: dict) -> str:
    logs = sorted(h["log_dir"].glob("*.log"))
    assert len(logs) == 1, f"expected exactly one daily log, found {logs}"
    return logs[0].read_text()


def _sidecar_rows(h: dict) -> list[dict]:
    if not h["sidecar"].exists():
        return []
    return [json.loads(ln) for ln in h["sidecar"].read_text().splitlines() if ln.strip()]


# ── T1: the silent rc=0 path ─────────────────────────────────────────────────


def test_rc0_no_push_no_sentinel_no_sidecar(make_harness):
    """rc=0 → zero pushes, no sentinel, no sidecar row, exit 0; the daily
    pointer line reaches stdout on the first run of the day."""
    h = make_harness(stub_rc=0)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 0
    assert _sentinels(h) == []
    assert _sidecar_rows(h) == []
    log = _daily_log(h)
    assert "step9c_ledger_refresh start" in log
    assert "exit=0" in log
    assert "ALERT" not in log
    assert "per-pass output" in result.stdout


# ── T2: the fail-loud rc=2 path ──────────────────────────────────────────────


def test_rc2_pushes_once_writes_sentinel_and_sidecar(make_harness):
    """rc=2 (refresh failure, NO ledger write) is NOT silently swallowed —
    exactly one push, a per-date sentinel, one sidecar row, wrapper exit 0."""
    h = make_harness(stub_rc=2)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 1
    message = h["push_log"].read_text()
    assert "step9c_ledger_refresh FAILED (rc=2)" in message
    # $LOG_FILE is expanded by the wrapper's shell before the message reaches
    # telegram_push.sh (which does no expansion of the body).
    assert str(h["log_dir"]) in message
    assert len(_sentinels(h)) == 1
    rows = _sidecar_rows(h)
    assert len(rows) == 1
    assert rows[0]["event"] == "refresh_failed"
    assert rows[0]["rc"] == 2


# ── T3: per-date sentinel dedup (sidecar still records) ─────────────────────


def test_second_run_same_date_does_not_repush(make_harness):
    """A second same-date run after a pushed alert is sentinel-suppressed,
    but the audit sidecar still gains a second row (the sidecar row lands
    BEFORE the sentinel check — dedup the buzz, not the record)."""
    h = make_harness(stub_rc=2)
    h["run"]()
    assert _push_count(h["push_log"]) == 1
    result2 = h["run"]()
    assert result2.returncode == 0
    assert _push_count(h["push_log"]) == 1
    assert "skipping re-alert" in _daily_log(h)
    assert len(_sidecar_rows(h)) == 2


# ── T4 / T5: fail-soft push handling ─────────────────────────────────────────


def test_push_failure_leaves_no_sentinel(make_harness):
    """telegram_push.sh exiting non-zero → NO sentinel (so the next pass
    retries), the failure is logged, and the wrapper still exits 0."""
    h = make_harness(stub_rc=2, push_exit=1)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 1  # attempted (the failing fake records)
    assert _sentinels(h) == []
    log = _daily_log(h)
    assert "telegram_push.sh FAILED" in log
    assert "no sentinel written" in log


def test_missing_push_script_logs_and_exits_zero(make_harness):
    """A missing / non-executable telegram_push.sh is logged, no crash,
    no sentinel, wrapper exits 0."""
    h = make_harness(stub_rc=2, push_missing=True)
    result = h["run"]()
    assert result.returncode == 0
    assert _push_count(h["push_log"]) == 0
    assert _sentinels(h) == []
    assert "not executable" in _daily_log(h)


# ── T6: the per-day pointer line fires once per date ─────────────────────────


def test_daily_pointer_line_only_on_first_run_of_day(make_harness):
    """The pointer line (the only stdout the crontab redirect file receives)
    lands exactly once per date — present on run 1, absent on run 2."""
    h = make_harness(stub_rc=0)
    result1 = h["run"]()
    assert "per-pass output" in result1.stdout
    result2 = h["run"]()
    assert result2.returncode == 0
    assert "per-pass output" not in result2.stdout


# ── T7: the refresh invocation shape ─────────────────────────────────────────


def test_stub_receives_refresh_json_args(make_harness):
    """The wrapper invokes the refresh bin with the exact `refresh --json`
    argv — pinning the CLI shape the production path passes to
    scripts/step9c_baseline.py."""
    h = make_harness(stub_rc=0)
    assert h["run"]().returncode == 0
    calls = [ln for ln in h["stub_calls"].read_text().splitlines() if ln.strip()]
    assert calls == ["refresh --json"]


# ── T8-T10: the #2386 fail-loud log-dir guard (Pattern B, fatal + push) ──────


def test_uncreatable_log_dir_fails_loud_and_pushes(make_harness):
    """T8: an uncreatable $LOG_DIR (path under a regular file, ENOTDIR) exits
    non-zero with a stderr FATAL naming the dir, fires exactly one ALERT push,
    and NEVER invokes the refresh — never the pre-#2386 silent skip in which
    both the refresh and its own rc!=0 alert arm vanished into the unwritable
    log while the wrapper reported success."""
    h = make_harness(stub_rc=0, log_dir_setup="uncreatable")
    result = h["run"]()

    assert result.returncode != 0, f"expected non-zero exit, stderr={result.stderr!r}"
    assert "FATAL" in result.stderr
    assert _MKDIR_FATAL in result.stderr
    assert str(h["log_dir"]) in result.stderr
    # The push leg fired: this wrapper's fatal() has a live alert channel.
    assert _push_count(h["push_log"]) == 1
    assert _ALERT_PREFIX in h["push_log"].read_text()
    assert not h["stub_calls"].exists(), "the refresh RAN despite an uncreatable log dir"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory mode bits")
def test_unwritable_log_dir_fails_loud_and_pushes(make_harness):
    """T9: a $LOG_DIR that EXISTS but is unwritable (chmod 0o555) passes
    ``mkdir -p`` and fails the appendability probe — non-zero exit, the
    probe-specific FATAL, one ALERT push, refresh never invoked."""
    h = make_harness(stub_rc=0, log_dir_setup="unwritable")
    result = h["run"]()

    assert result.returncode != 0, f"expected non-zero exit, stderr={result.stderr!r}"
    assert "FATAL" in result.stderr
    # The probe arm, not the mkdir arm — mkdir -p succeeds on an existing dir.
    assert _PROBE_FATAL in result.stderr
    assert _MKDIR_FATAL not in result.stderr
    assert str(h["log_dir"]) in result.stderr
    assert _push_count(h["push_log"]) == 1
    assert _ALERT_PREFIX in h["push_log"].read_text()
    assert not h["stub_calls"].exists(), "the refresh RAN despite an unwritable log dir"


def test_missing_push_script_still_exits_non_zero(make_harness):
    """T10: the fatal() push leg is BEST-EFFORT — a missing/non-executable
    telegram_push.sh must not swallow the infrastructure failure. No push is
    recorded, the FATAL still reaches stderr, and the wrapper still exits
    non-zero (contrast T5, where a missing push script during the ordinary
    rc!=0 alert arm deliberately leaves exit 0)."""
    h = make_harness(stub_rc=0, push_missing=True, log_dir_setup="uncreatable")
    result = h["run"]()

    assert result.returncode != 0, f"expected non-zero exit, stderr={result.stderr!r}"
    assert "FATAL" in result.stderr
    assert _MKDIR_FATAL in result.stderr
    assert _push_count(h["push_log"]) == 0
    assert not h["stub_calls"].exists(), "the refresh RAN despite an uncreatable log dir"
