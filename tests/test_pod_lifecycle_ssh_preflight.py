"""Tests for ``scripts/pod_lifecycle.py`` SSH preflight (#12) + resume supply
guard (#11).

- SSH preflight: returns True when the endpoint is reachable, attempts ONE
  ``pod.py resume`` on the first failure, and returns False (without raising)
  when still unreachable.
- Resume supply-constraint: ``cmd_resume`` raises a clear actionable SystemExit
  proposing a fresh provision (never auto-terminates / auto-provisions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
from pod_lifecycle import (  # noqa: E402
    EphemeralMetadata,
    EphemeralPod,
    _is_supply_constraint,
    ssh_preflight,
)
from runpod_api import PodInfo, RunPodError  # noqa: E402


def _info(name="pod-1", *, ssh_host="1.2.3.4", ssh_port=22000, desired_status="EXITED"):
    return PodInfo(
        pod_id=f"id-{name}",
        name=name,
        desired_status=desired_status,
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        created_at="2026-05-01T00:00:00Z",
    )


def _pod(name="pod-1", issue=1, **info_kw):
    meta = EphemeralMetadata(name=name, pod_id=f"id-{name}", issue=issue)
    return EphemeralPod(metadata=meta, info=_info(name, **info_kw))


# ---------------------------------------------------------------------------
# _is_supply_constraint
# ---------------------------------------------------------------------------


def test_is_supply_constraint_detects_null_resume():
    assert _is_supply_constraint(RunPodError("podResume returned null for id-1"))


def test_is_supply_constraint_detects_no_free_gpu():
    assert _is_supply_constraint(RunPodError("GraphQL errors: not enough free GPUs"))


def test_is_supply_constraint_false_for_other_errors():
    assert not _is_supply_constraint(RunPodError("HTTP 401 from RunPod: unauthorized"))


# ---------------------------------------------------------------------------
# ssh_preflight — reachable / unreachable / resume-once
# ---------------------------------------------------------------------------


def test_ssh_preflight_reachable_no_resume(monkeypatch):
    """Reachable endpoint → True, never tries resume."""
    monkeypatch.setattr(pod_lifecycle, "_tcp_open", lambda h, p, t: True)
    resume_calls = []
    monkeypatch.setattr(
        pod_lifecycle, "_run_resume_subprocess", lambda issue: resume_calls.append(issue) or 0
    )
    assert ssh_preflight("1.2.3.4", 22000, issue=1) is True
    assert resume_calls == []


def test_ssh_preflight_recovers_after_one_resume(monkeypatch):
    """First probe fails → one resume → second probe (fresh endpoint) succeeds."""
    probes = iter([False, True])  # initial endpoint down, post-resume up
    monkeypatch.setattr(pod_lifecycle, "_tcp_open", lambda h, p, t: next(probes))
    resume_calls = []
    monkeypatch.setattr(
        pod_lifecycle, "_run_resume_subprocess", lambda issue: resume_calls.append(issue) or 0
    )
    monkeypatch.setattr(pod_lifecycle, "_live_ssh_endpoint", lambda issue: ("9.9.9.9", 23000))
    assert ssh_preflight("1.2.3.4", 22000, issue=7) is True
    assert resume_calls == [7]  # exactly one resume attempt


def test_ssh_preflight_unreachable_after_resume_returns_false(monkeypatch):
    """Both probes fail → False (no raise), resume attempted exactly once."""
    monkeypatch.setattr(pod_lifecycle, "_tcp_open", lambda h, p, t: False)
    resume_calls = []
    monkeypatch.setattr(
        pod_lifecycle, "_run_resume_subprocess", lambda issue: resume_calls.append(issue) or 0
    )
    monkeypatch.setattr(pod_lifecycle, "_live_ssh_endpoint", lambda issue: (None, None))
    assert ssh_preflight("1.2.3.4", 22000, issue=7) is False
    assert resume_calls == [7]


def test_ssh_preflight_no_resume_when_disabled(monkeypatch):
    """allow_resume=False → unreachable returns False without any resume."""
    monkeypatch.setattr(pod_lifecycle, "_tcp_open", lambda h, p, t: False)
    resume_calls = []
    monkeypatch.setattr(
        pod_lifecycle, "_run_resume_subprocess", lambda issue: resume_calls.append(issue) or 0
    )
    assert ssh_preflight("1.2.3.4", 22000, issue=7, allow_resume=False) is False
    assert resume_calls == []


def test_ssh_preflight_resume_failure_returns_false(monkeypatch):
    """resume exits non-zero → False, no second probe needed."""
    monkeypatch.setattr(pod_lifecycle, "_tcp_open", lambda h, p, t: False)
    monkeypatch.setattr(pod_lifecycle, "_run_resume_subprocess", lambda issue: 1)
    # If _live_ssh_endpoint were called it'd blow up; assert it isn't.
    monkeypatch.setattr(
        pod_lifecycle,
        "_live_ssh_endpoint",
        lambda issue: (_ for _ in ()).throw(AssertionError("should not re-probe")),
    )
    assert ssh_preflight("1.2.3.4", 22000, issue=7) is False


def test_tcp_open_none_host_is_closed(monkeypatch):
    """A missing host/port counts as unreachable without touching the socket."""
    assert pod_lifecycle._tcp_open(None, None, 1.0) is False
    assert pod_lifecycle._tcp_open("1.2.3.4", None, 1.0) is False


# ---------------------------------------------------------------------------
# cmd_resume — supply constraint guard (#11)
# ---------------------------------------------------------------------------


def test_cmd_resume_supply_constraint_raises_actionable(monkeypatch):
    """resume_pod raising a supply error → SystemExit proposing fresh provision,
    and NO auto-terminate / auto-provision side effect."""
    pod = _pod(name="pod-5", issue=5)
    monkeypatch.setattr(pod_lifecycle, "_load_state", lambda: {"pod-5": pod})
    monkeypatch.setattr(
        pod_lifecycle, "_find_pod_in_state", lambda state, issue, name_suffix=None: pod
    )
    # Stub the account-spend guard so the test stays offline (no real API hit).
    monkeypatch.setattr(pod_lifecycle, "_assert_under_account_hourly_cap", lambda **_kw: None)

    def boom(pod_id, gpu_count):
        raise RunPodError("podResume returned null for id-pod-5")

    monkeypatch.setattr(pod_lifecycle, "resume_pod", boom)

    # Guard: these must NOT be called on a supply constraint.
    monkeypatch.setattr(
        pod_lifecycle,
        "terminate_pod",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not auto-terminate")),
    )
    monkeypatch.setattr(
        pod_lifecycle,
        "create_pod",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not auto-provision")),
    )

    # wait_for_capacity=False mirrors the parser default (_parser_resume).
    ns = argparse.Namespace(issue=5, dry_run=False, wait_for_capacity=False)
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle.cmd_resume(ns)
    msg = str(exc.value)
    assert "supply constraint" in msg.lower()
    assert "provision --issue 5" in msg


def test_cmd_resume_non_supply_error_propagates(monkeypatch):
    """A non-supply RunPodError is re-raised unchanged (not swallowed)."""
    pod = _pod(name="pod-6", issue=6)
    monkeypatch.setattr(pod_lifecycle, "_load_state", lambda: {"pod-6": pod})
    monkeypatch.setattr(
        pod_lifecycle, "_find_pod_in_state", lambda state, issue, name_suffix=None: pod
    )
    # Stub the account-spend guard so the test stays offline (no real API hit).
    monkeypatch.setattr(pod_lifecycle, "_assert_under_account_hourly_cap", lambda **_kw: None)

    def boom(pod_id, gpu_count):
        raise RunPodError("HTTP 401 from RunPod: unauthorized")

    monkeypatch.setattr(pod_lifecycle, "resume_pod", boom)

    # wait_for_capacity=False mirrors the parser default (_parser_resume).
    ns = argparse.Namespace(issue=6, dry_run=False, wait_for_capacity=False)
    with pytest.raises(RunPodError) as exc:
        pod_lifecycle.cmd_resume(ns)
    assert "401" in str(exc.value)
