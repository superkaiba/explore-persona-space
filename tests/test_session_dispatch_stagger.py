"""Pure-helper tests for the #1059 session-dispatch stagger.

The stagger paces `spawn-issue --auto` SESSION dispatches >=
``EPM_SESSION_DISPATCH_STAGGER_S`` (60s) apart across the watcher's two infra
dispatch loops and the file-time filer, via the shared last-writer-wins stamp
``~/.eps-autonomous/last-session-dispatch.json`` (each fresh session is a
~100K-token cold context load; the org input-TPM 429 cap climbs at minute
boundaries). These tests pin the pure decision helper, the env parsing, and
the stamp read/write round trip; the wiring tests live in
``test_autonomous_session_watch.py`` (watcher sleeps) and
``test_file_infra_task.py`` (filer defers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402


def test_stagger_delay_pure_cases():
    f = spawn_session.stagger_delay_s
    assert f(None, 60.0) == 0.0  # no prior dispatch
    assert f(70.0, 60.0) == 0.0  # window already elapsed
    assert f(60.0, 60.0) == 0.0  # boundary: exactly elapsed
    assert f(5.0, 60.0) == 55.0  # the remainder
    assert f(0.0, 60.0) == 60.0  # just recorded -> the full window
    assert f(5.0, 0.0) == 0.0  # disabled window
    assert f(-5.0, 60.0) == 60.0  # negative age clamps to <= window


def test_session_dispatch_stagger_s_env(monkeypatch):
    monkeypatch.delenv("EPM_SESSION_DISPATCH_STAGGER_S", raising=False)
    assert spawn_session.session_dispatch_stagger_s() == 60.0  # default
    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "0")
    assert spawn_session.session_dispatch_stagger_s() == 0.0  # kill switch
    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "-5")
    assert spawn_session.session_dispatch_stagger_s() == 0.0  # negative disables
    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "banana")
    assert spawn_session.session_dispatch_stagger_s() == 60.0  # malformed -> default
    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "900")
    assert spawn_session.session_dispatch_stagger_s() == 300.0  # clamp
    monkeypatch.setenv("EPM_SESSION_DISPATCH_STAGGER_S", "45")
    assert spawn_session.session_dispatch_stagger_s() == 45.0


def test_record_then_age_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    assert spawn_session.last_session_dispatch_age_s(now=1005.0) is None  # no stamp
    spawn_session.record_session_dispatch(42, "test-holder", now=1000.0)
    assert spawn_session.last_session_dispatch_age_s(now=1005.0) == pytest.approx(5.0)
    entry = json.loads((tmp_path / "last-session-dispatch.json").read_text())
    assert entry["issue"] == 42
    assert entry["holder"] == "test-holder"
    assert entry["ts"] == 1000.0


def test_garbled_stamp_falls_back_to_mtime(tmp_path, monkeypatch):
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    stamp = tmp_path / "last-session-dispatch.json"
    stamp.write_bytes(b"\x00{torn write")  # garbled content, not a crash
    mtime = stamp.stat().st_mtime
    assert spawn_session.last_session_dispatch_age_s(now=mtime + 7.0) == pytest.approx(7.0)


def test_future_dated_ts_reads_as_just_now(tmp_path, monkeypatch):
    # Clock skew / a typo'd ts must not produce a negative age (which would
    # inflate the delay past the window); it reads as "dispatched just now".
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    spawn_session.record_session_dispatch(7, "test-holder", now=2000.0)
    assert spawn_session.last_session_dispatch_age_s(now=1000.0) == 0.0


def test_stagger_sleep_seam_real_body():
    # Body coverage for the seam every watcher test stubs (code-style.md
    # "one production-body test per seam-stubbed function"): the real body is
    # a straight time.sleep passthrough — execute it with a ~10ms delay and
    # confirm wall time actually advanced (the external call was reached).
    import time as _time

    import autonomous_session_watch as asw

    t0 = _time.monotonic()
    asw._stagger_sleep(0.01)
    assert _time.monotonic() - t0 >= 0.009


def test_record_never_raises(tmp_path, monkeypatch, capsys):
    # A failed pacing record must not fail a successful spawn: loud stderr
    # warning, no exception (fail-open write).
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(spawn_session.os, "replace", _boom)
    spawn_session.record_session_dispatch(42, "test-holder", now=1000.0)  # must not raise
    assert "session-dispatch stamp write failed" in capsys.readouterr().err
