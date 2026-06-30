"""Unit tests for scripts/codex_daemon_reaper.py.

Pins the selector predicate (a)-(e), the best-effort WAL truncate (f), and the
three load-bearing safety paths: WAL-checkpoint-busy (g), ps-read-failure (h),
and the PID-reuse guard at SIGTERM + SIGKILL (i). No real process is signaled,
no real cron runs; the only on-disk sqlite is a per-test temp file the test
creates and tears down.
"""

import importlib.util
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

# Load the reaper as a module (scripts/ is not an importable package).
_REAPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "codex_daemon_reaper.py"
_spec = importlib.util.spec_from_file_location("codex_daemon_reaper", _REAPER_PATH)
reaper = importlib.util.module_from_spec(_spec)
sys.modules["codex_daemon_reaper"] = reaper
_spec.loader.exec_module(reaper)

H = 3600.0
DAY_S = int(24 * H)

# Representative live argv shapes (confirmed 2026-06-30).
ARGV_APP_SERVER = "node /home/thomasjiralerspong/.codex/bin/codex app-server"
ARGV_VENDOR = "/home/thomasjiralerspong/.codex/vendor/codex-linux-x64/bin/codex app-server"
ARGV_BROKER = "node /home/thomasjiralerspong/.codex/app-server-broker.mjs serve"
ARGV_TASK_DRIVER = "uv run python scripts/codex_task.py --issue 697 --effort high"
ARGV_WORKFLOW_LINT = "uv run python scripts/workflow_lint.py --check-asks"
ARGV_WANDB_RECLAIM = "uv run python scripts/wandb_reclaim.py --apply"
ARGV_BASH_WRAPPER = (
    "bash -c CODEX_COMPANION_SESSION_ID=abc node /home/x/.codex/bin/codex app-server"
)
ARGV_SH_WRAPPER = "/bin/sh -c node /home/x/.codex/bin/codex app-server"


# ---------------------------------------------------------------------------
# Selector tests (against injectable fake (pid, etimes, argv) snapshots)
# ---------------------------------------------------------------------------
def test_a_selects_over_threshold_app_server():
    """(a) A >24h `node ... codex app-server` row lands in candidates."""
    snap = [(1001, DAY_S + 1000, ARGV_APP_SERVER)]
    candidates, sub, ps_status = reaper.enumerate_candidates(DAY_S, snapshot=snap)
    assert [c["pid"] for c in candidates] == [1001]
    assert sub == []
    assert ps_status["ok"] is True


def test_b_spares_sub_threshold_daemon():
    """(b) A <24h same-argv daemon is sub_threshold, never a candidate."""
    snap = [(1002, DAY_S - 1000, ARGV_BROKER)]
    candidates, sub, _ = reaper.enumerate_candidates(DAY_S, snapshot=snap)
    assert candidates == []
    assert [c["pid"] for c in sub] == [1002]


def test_c_never_selects_codex_task_driver_at_any_age():
    """(c) A codex_task.py review driver is hard-excluded even at 999999s."""
    snap = [(1003, 999999, ARGV_TASK_DRIVER)]
    candidates, sub, _ = reaper.enumerate_candidates(DAY_S, snapshot=snap)
    assert candidates == []
    assert sub == []


def test_d_env_and_cli_threshold_precedence(monkeypatch):
    """(d) EPS_CODEX_REAPER_MAX_AGE_H override; CLI --max-age-h wins over env."""
    # Env override: 1h threshold makes a 2h daemon a candidate.
    monkeypatch.setenv("EPS_CODEX_REAPER_MAX_AGE_H", "1")
    assert reaper._max_age_seconds(None) == pytest.approx(1 * H)
    # CLI flag wins over env.
    assert reaper._max_age_seconds(48.0) == pytest.approx(48 * H)
    # No env, no flag -> default 24h.
    monkeypatch.delenv("EPS_CODEX_REAPER_MAX_AGE_H", raising=False)
    assert reaper._max_age_seconds(None) == pytest.approx(24 * H)


def test_e_never_selects_excluded_or_shell_wrappers():
    """(e) workflow_lint / wandb_reclaim / bash -c / /bin/sh -c are all spared,
    even when the wrapper argv echoes a daemon string."""
    snap = [
        (2001, DAY_S + 1, ARGV_WORKFLOW_LINT),
        (2002, DAY_S + 1, ARGV_WANDB_RECLAIM),
        (2003, DAY_S + 1, ARGV_BASH_WRAPPER),  # echoes "codex app-server" but is a bash -c wrapper
        (2004, DAY_S + 1, ARGV_SH_WRAPPER),  # the broadened \b\S*sh -c\b exclude
    ]
    candidates, sub, _ = reaper.enumerate_candidates(DAY_S, snapshot=snap)
    assert candidates == []
    assert sub == []


def test_f_truncate_wal_best_effort(monkeypatch, tmp_path):
    """(f) truncate_wal() is best-effort: db_absent path returns ok:False without
    raising; a fresh WAL-mode temp DB returns ok:True with a 3-tuple result_row."""
    # db_absent path.
    missing = tmp_path / "nope.sqlite"
    monkeypatch.setattr(reaper, "CODEX_DB", missing)
    res = reaper.truncate_wal()
    assert res["ok"] is False
    assert res["error"] == "db_absent"
    assert res["checkpoint_busy"] is False

    # Fresh WAL-mode DB with no other reader -> clean truncate.
    db = tmp_path / "ok.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(200)])
    con.commit()
    con.close()
    monkeypatch.setattr(reaper, "CODEX_DB", db)
    res = reaper.truncate_wal()
    assert res["ok"] is True
    assert res["checkpoint_busy"] is False
    assert res["result_row"] is not None
    assert len(res["result_row"]) == 3


# ---------------------------------------------------------------------------
# Safety-path tests (round-2 additions)
# ---------------------------------------------------------------------------
def test_g_wal_checkpoint_busy_reported_as_failure(monkeypatch, tmp_path):
    """(g) A reader-pinned WAL returns ok:False / checkpoint_busy:True (NOT silent
    success), with a nonzero busy flag in result_row and the WAL bytes intact."""
    db = tmp_path / "busy.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(500)])
    con.commit()

    # A SECOND connection holding an open read transaction pins the WAL so
    # TRUNCATE is blocked (busy=1). Begin a read txn and leave it open.
    con2 = sqlite3.connect(str(db))
    con2.execute("BEGIN")
    con2.execute("SELECT count(*) FROM t").fetchone()

    monkeypatch.setattr(reaper, "CODEX_DB", db)
    try:
        res = reaper.truncate_wal()  # must NOT raise
    finally:
        con2.rollback()
        con2.close()
        con.close()

    assert res["ok"] is False
    assert res["checkpoint_busy"] is True
    assert res["result_row"] is not None
    assert res["result_row"][0] != 0  # busy flag set
    assert res["wal_bytes_after"] > 0  # WAL not reclaimed while pinned


def test_h_ps_read_failure_surfaces_and_main_exits_3(monkeypatch, capsys):
    """(h) A ps read failure (OSError AND nonzero returncode) yields ps_status.ok
    False, empty lists, and main() exit 3 with --apply taking no action."""

    # (h1) OSError path.
    def _raise_oserror(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(subprocess, "run", _raise_oserror)
    rows, status = reaper._ps_snapshot()
    assert rows == []
    assert status["ok"] is False
    assert "permission denied" in status["error"]
    cands, sub, ps_status = reaper.enumerate_candidates(DAY_S)
    assert cands == [] and sub == [] and ps_status["ok"] is False
    rc = reaper.main(["--json"])
    assert rc == 3
    out = capsys.readouterr().out
    assert '"ok": false' in out.lower()
    # --apply must take NO action on an unreadable table.
    rc_apply = reaper.main(["--apply", "--json"])
    assert rc_apply == 3
    out_apply = capsys.readouterr().out
    import json as _json

    payload = _json.loads(out_apply.strip().splitlines()[-1])
    assert payload["kill_result"] is None
    assert payload["wal"] is None

    # (h2) nonzero ps returncode path.
    def _nonzero_rc(*a, **k):
        return subprocess.CompletedProcess(
            args=a[0] if a else [], returncode=1, stdout="", stderr="ps: cannot access /proc"
        )

    monkeypatch.setattr(subprocess, "run", _nonzero_rc)
    rows2, status2 = reaper._ps_snapshot()
    assert rows2 == []
    assert status2["ok"] is False
    assert status2["returncode"] == 1
    assert reaper.main(["--json"]) == 3


def test_i_pid_reuse_guard_blocks_signal(monkeypatch):
    """(i) The reuse guard prevents SIGKILL on a pid recycled between SIGTERM and
    SIGKILL (i1) and prevents SIGTERM on a pid excluded at signal time (i2)."""
    calls: list[tuple[int, int]] = []

    def _fake_kill(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(reaper.os, "kill", _fake_kill)

    # Stateful cmdline shim keyed by pid; advances each read so the SIGKILL-time
    # read can differ from the SIGTERM-time read (the "recycled" case).
    read_counts: dict[int, int] = {}

    def _stateful_cmdline(pid):
        n = read_counts.get(pid, 0)
        read_counts[pid] = n + 1
        if pid == 9001:
            # First read (pre-SIGTERM via _still_reapable) -> daemon -> SIGTERM sent.
            # Subsequent reads (the term-wait poll + pre-SIGKILL _still_reapable)
            # -> a recycled non-daemon cmdline -> no SIGKILL.
            return ARGV_APP_SERVER if n == 0 else "/usr/bin/some-unrelated-process --flag"
        if pid == 9002:
            # Excluded at signal time -> never SIGTERM'd.
            return ARGV_TASK_DRIVER
        return None

    monkeypatch.setattr(reaper, "_read_cmdline", _stateful_cmdline)

    result = reaper.kill_candidates([9001, 9002], term_wait_s=0.05)

    # 9001: exactly one SIGTERM, never SIGKILL (recycled before escalation).
    assert (9001, signal.SIGTERM) in calls
    assert (9001, signal.SIGKILL) not in calls
    # 9002: never signaled at all, and reported as reuse-skipped.
    assert all(pid != 9002 for pid, _ in calls)
    assert 9002 in result["reuse_skipped"]
    # No stray signals to any other pid.
    assert {pid for pid, _ in calls} == {9001}
