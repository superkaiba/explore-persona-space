"""Tests for the 1h billing-pod SSH-wait alarm in ``scripts/pod_lifecycle.py``
(refs #572 — pod-488 sat SSH-unreachable ~13.7h at $32/hr with only per-call
noise; the tracker escalates to a structured ``[ssh-wait-ALARM]`` line).

Covers:
- episode opens on the first unreachable observation; no alarm before the
  threshold;
- alarm fires once the cumulative unreachable span crosses the threshold on a
  RUNNING (billing) pod, and re-fires at most once per window;
- a reachable observation closes the episode (state cleared);
- EXITED pods never alarm (not billing).

The state file is redirected into ``tmp_path`` and the live-API status lookup
is stubbed, so no network and no real clock are involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402


def _redirect_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pod_lifecycle, "_SSH_WAIT_STATE_PATH", tmp_path / "ssh-wait-alarm.json"
    )


def test_no_alarm_before_threshold(monkeypatch, tmp_path, capsys):
    _redirect_state(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SSH_WAIT_ALARM_SECS", "3600")
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=1000.0
    )
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=1000.0 + 1800
    )
    assert "[ssh-wait-ALARM]" not in capsys.readouterr().err


def test_alarm_fires_after_threshold_and_dedups_within_window(
    monkeypatch, tmp_path, capsys
):
    _redirect_state(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SSH_WAIT_ALARM_SECS", "3600")
    t0 = 1000.0
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=t0
    )
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=t0 + 3700
    )
    err = capsys.readouterr().err
    assert "[ssh-wait-ALARM]" in err
    assert "pod-9" in err
    assert "--refresh-from-api pod-9" in err
    # Within the same window: deduped.
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=t0 + 3800
    )
    assert "[ssh-wait-ALARM]" not in capsys.readouterr().err
    # Next window: re-fires.
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=t0 + 3700 + 3601
    )
    assert "[ssh-wait-ALARM]" in capsys.readouterr().err


def test_reachable_clears_the_episode(monkeypatch, tmp_path, capsys):
    _redirect_state(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SSH_WAIT_ALARM_SECS", "3600")
    t0 = 1000.0
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=t0
    )
    pod_lifecycle.note_ssh_wait_outcome("pod-9", reachable=True, now=t0 + 100)
    # A new failure 2h later starts a FRESH episode — no alarm yet.
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="RUNNING", now=t0 + 7200
    )
    assert "[ssh-wait-ALARM]" not in capsys.readouterr().err


def test_exited_pod_never_alarms(monkeypatch, tmp_path, capsys):
    _redirect_state(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SSH_WAIT_ALARM_SECS", "3600")
    t0 = 1000.0
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="EXITED", now=t0
    )
    pod_lifecycle.note_ssh_wait_outcome(
        "pod-9", reachable=False, desired_status="EXITED", now=t0 + 7200
    )
    assert "[ssh-wait-ALARM]" not in capsys.readouterr().err


def test_status_lookup_falls_back_to_live_api(monkeypatch, tmp_path, capsys):
    _redirect_state(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SSH_WAIT_ALARM_SECS", "3600")
    monkeypatch.setattr(
        pod_lifecycle, "_pod_desired_status_by_name", lambda _name: "RUNNING"
    )
    t0 = 1000.0
    pod_lifecycle.note_ssh_wait_outcome("pod-9", reachable=False, now=t0)
    pod_lifecycle.note_ssh_wait_outcome("pod-9", reachable=False, now=t0 + 3700)
    err = capsys.readouterr().err
    assert "[ssh-wait-ALARM]" in err
    assert "RUNNING (BILLING)" in err


# ─── setsid detach for autonomous provisions/resumes (refs #573) ─────────────


def _record_setsid(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(pod_lifecycle.os, "setsid", lambda: calls.append(True))
    return calls


def test_setsid_fires_for_autonomous_provision(monkeypatch, capsys):
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.delenv("EPM_NO_SETSID", raising=False)
    calls = _record_setsid(monkeypatch)
    pod_lifecycle._maybe_detach_into_own_session("provision")
    assert calls == [True]
    assert "detached into own session" in capsys.readouterr().err


def test_setsid_skipped_outside_autonomous_mode(monkeypatch):
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    calls = _record_setsid(monkeypatch)
    pod_lifecycle._maybe_detach_into_own_session("provision")
    assert calls == []


def test_setsid_skipped_for_non_lifecycle_verbs(monkeypatch):
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    calls = _record_setsid(monkeypatch)
    for verb in ("stop", "terminate", "list-ephemeral", None):
        pod_lifecycle._maybe_detach_into_own_session(verb)
    assert calls == []


def test_setsid_opt_out_env(monkeypatch):
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.setenv("EPM_NO_SETSID", "1")
    calls = _record_setsid(monkeypatch)
    pod_lifecycle._maybe_detach_into_own_session("provision")
    assert calls == []


def test_setsid_failure_is_fail_soft(monkeypatch, capsys):
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    monkeypatch.delenv("EPM_NO_SETSID", raising=False)

    def _boom():
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(pod_lifecycle.os, "setsid", _boom)
    pod_lifecycle._maybe_detach_into_own_session("resume")  # must not raise
    assert "setsid failed" in capsys.readouterr().err
