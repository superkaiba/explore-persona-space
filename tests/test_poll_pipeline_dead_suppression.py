"""Tests for poll_pipeline.py's #751 dead-verdict suppression guard.

A pod that a GCP->RunPod failover (or a no-port-wedge re-provision) JUST
relaunched can be RUNNING on the live RunPod API while still ABSENT from
``pods.conf`` (the SSH/MCP config source) — the new pod's row has not landed
yet. The SSH probe then lands on an unresolvable host, ``_ssh_probe`` zeroes
``pid_alive``, and the ``elif not pid_alive: status = "dead"`` verdict would
report a FALSE ``dead`` for a healthy, still-registering pod.

The fix (``poll_once``): when the pid probes dead AND the pod is absent from
``pods.conf`` AND the live RunPod API reports it RUNNING, SUPPRESS the ``dead``
verdict (report ``running``, "still registering") for up to
``DEAD_SUPPRESSION_CAP`` CONSECUTIVE ticks, giving backend_poll's best-effort
register + the #488 self-heal time to land the row. Beyond the cap the verdict
falls through to ``dead`` so a registration that NEVER lands cannot mask a
genuinely dead pod indefinitely.

Scope guard (pinned below): a pod IN ``pods.conf`` that probes dead is NEVER
suppressed — the guard fires ONLY for absent-from-pods.conf pods. Both the
absence check and the live-RUNNING check are fail-soft (unknown state ->
no suppression -> the conservative ``dead`` verdict).

These tests stub ``_ssh_probe`` (a dead-pid probe dict), ``_drain_sentinels``,
``_marker_pid``, and the GPU-idle advisory/escalation helpers so the unit test
never touches the network — the verdict logic + the absence/live-API checks
are what's under test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_poll_pipeline():
    """Load ``scripts/poll_pipeline.py`` as an isolated module (mirrors
    ``tests/test_poll_pipeline_sentinels.py``)."""
    spec = importlib.util.spec_from_file_location(
        "poll_pipeline_dead_suppress_under_test", REPO_ROOT / "scripts" / "poll_pipeline.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["poll_pipeline_dead_suppress_under_test"] = module
    spec.loader.exec_module(module)
    return module


pp = _load_poll_pipeline()


def _dead_pid_probe() -> dict[str, str]:
    """A full ``_ssh_probe`` result dict for a pod whose pid probed DEAD via an
    SSH transport failure (the absent-from-pods.conf shape: ssh_failed, every
    pid signal zeroed). Carries every key ``poll_once`` reads off the probe."""
    return {
        "ssh_failed": "1",
        "pid_alive": "0",
        "marker_pid_alive": "0",
        "pid_file_missing": "1",
        "mtime_epoch": "0",
        "cell_mtime_epoch": "0",
        "phase_log_mtime_epoch": "0",
        "shard_log_mtime_epoch": "0",
        "pod_now_epoch": "0",
        "gpu_util": "unknown",
        "zombie_gpu_pids": "",
        "log_tail": "",
        "cell_log_tail": "",
        "session_cpu_secs": "unknown",
        "results_sentinel_present": "0",
    }


@pytest.fixture
def stub_poll_once_io(monkeypatch):
    """Stub every network-touching ``poll_once`` dependency EXCEPT the probe +
    the absence/live-API checks (which the tests drive per-case). Returns
    nothing — the tests set ``_ssh_probe`` / ``_pod_absent_from_pods_conf`` /
    ``_live_pod_running`` themselves."""
    monkeypatch.setattr(pp, "_drain_sentinels", lambda **_kw: (0, None))
    monkeypatch.setattr(pp, "_marker_pid", lambda _issue: None)
    # GPU-idle advisory/escalation post markers via task.py — no-op them so the
    # unit test never shells out. Benign return shapes matching the call sites.
    monkeypatch.setattr(pp, "_maybe_post_gpu_idle_advisory", lambda **_kw: (0, set(), False))
    monkeypatch.setattr(pp, "_maybe_escalate_gpu_idle", lambda **_kw: (set(), False))


def _run_poll_once(tmp_path, *, state_file_name="state.json"):
    return pp.poll_once(
        issue=751,
        pod="pod-697",
        log_path="/workspace/logs/issue-751.log",
        pid_file="/workspace/logs/issue-751.pid",
        state_file=tmp_path / state_file_name,
    )


# ---------------------------------------------------------------------------
# AC3: absent + live-RUNNING + counter below cap -> running (still registering)
# ---------------------------------------------------------------------------


def test_absent_and_live_running_below_cap_suppresses_dead(
    tmp_path, monkeypatch, stub_poll_once_io
):
    """Pid probed dead, pod ABSENT from pods.conf, live API says RUNNING, and the
    consecutive-suppression counter is below the cap -> status is ``running``
    (still registering), NOT ``dead``."""
    monkeypatch.setattr(pp, "_ssh_probe", lambda *a, **k: _dead_pid_probe())
    monkeypatch.setattr(pp, "_pod_absent_from_pods_conf", lambda _pod: True)
    monkeypatch.setattr(pp, "_live_pod_running", lambda _pod: True)

    result = _run_poll_once(tmp_path)
    assert result.status == "running", "a just-relaunched, still-registering pod must not be dead"


# ---------------------------------------------------------------------------
# AC3: absent but NOT live-running -> dead (genuine death, not registering)
# ---------------------------------------------------------------------------


def test_absent_but_not_live_running_routes_dead(tmp_path, monkeypatch, stub_poll_once_io):
    """Pid probed dead, pod ABSENT from pods.conf, but the live API does NOT
    report it RUNNING (gone / EXITED / API error -> fail-soft False) -> the
    suppression does NOT fire; status is ``dead`` (conservative)."""
    monkeypatch.setattr(pp, "_ssh_probe", lambda *a, **k: _dead_pid_probe())
    monkeypatch.setattr(pp, "_pod_absent_from_pods_conf", lambda _pod: True)
    monkeypatch.setattr(pp, "_live_pod_running", lambda _pod: False)

    result = _run_poll_once(tmp_path)
    assert result.status == "dead"


# ---------------------------------------------------------------------------
# Scope guard: a pod PRESENT in pods.conf that probes dead is NEVER suppressed
# ---------------------------------------------------------------------------


def test_present_in_pods_conf_dead_pod_still_dead(tmp_path, monkeypatch, stub_poll_once_io):
    """A pod that IS in pods.conf (so the SSH address was known) but probes dead
    keeps the ``dead`` verdict — the suppression is scoped to ABSENT pods only,
    even if the live API happens to still report it RUNNING (a genuinely dead
    workload on a live pod must surface)."""
    monkeypatch.setattr(pp, "_ssh_probe", lambda *a, **k: _dead_pid_probe())
    monkeypatch.setattr(pp, "_pod_absent_from_pods_conf", lambda _pod: False)
    # Even with the live API saying RUNNING, the absence gate is False -> no suppression.
    monkeypatch.setattr(pp, "_live_pod_running", lambda _pod: True)

    result = _run_poll_once(tmp_path)
    assert result.status == "dead"


# ---------------------------------------------------------------------------
# Cap: registration that NEVER lands eventually falls through to dead
# ---------------------------------------------------------------------------


def test_suppression_capped_falls_through_to_dead(tmp_path, monkeypatch, stub_poll_once_io):
    """Suppression is bounded by ``DEAD_SUPPRESSION_CAP`` CONSECUTIVE ticks. Seed
    the persisted counter AT the cap and confirm the next tick (still absent +
    live-RUNNING) falls through to ``dead`` — a registration that never lands
    cannot mask a genuinely dead pod forever. Also assert the constant (not a
    magic number) governs the bound."""
    assert pp.DEAD_SUPPRESSION_CAP == 2 * pp.SSH_FAIL_REFRESH_THRESHOLD

    monkeypatch.setattr(pp, "_ssh_probe", lambda *a, **k: _dead_pid_probe())
    monkeypatch.setattr(pp, "_pod_absent_from_pods_conf", lambda _pod: True)
    monkeypatch.setattr(pp, "_live_pod_running", lambda _pod: True)

    state_file = tmp_path / "state.json"
    # Seed the counter AT the cap (the prior tick was the last suppressed one).
    pp._save_state(state_file, 751, {"dead_suppress_count": str(pp.DEAD_SUPPRESSION_CAP)})

    result = pp.poll_once(
        issue=751,
        pod="pod-697",
        log_path="/workspace/logs/issue-751.log",
        pid_file="/workspace/logs/issue-751.pid",
        state_file=state_file,
    )
    assert result.status == "dead", "beyond the cap the verdict must fall through to dead"


def test_suppression_counter_resets_when_present(tmp_path, monkeypatch, stub_poll_once_io):
    """The counter is CONSECUTIVE: a tick where the pod is present-in-pods.conf
    (or otherwise not suppressed) RESETS it, so a later absent+running tick gets
    a full fresh suppression window. Seed the counter near the cap, run a
    present-in-pods.conf tick (resets to 0), then an absent+running tick must
    suppress again (count 1, well below cap)."""
    monkeypatch.setattr(pp, "_ssh_probe", lambda *a, **k: _dead_pid_probe())
    monkeypatch.setattr(pp, "_live_pod_running", lambda _pod: True)

    state_file = tmp_path / "state.json"
    pp._save_state(state_file, 751, {"dead_suppress_count": str(pp.DEAD_SUPPRESSION_CAP - 1)})

    # Tick 1: present in pods.conf -> dead verdict (scoped out) AND counter reset.
    monkeypatch.setattr(pp, "_pod_absent_from_pods_conf", lambda _pod: False)
    r1 = pp.poll_once(
        issue=751,
        pod="pod-697",
        log_path="/workspace/logs/issue-751.log",
        pid_file="/workspace/logs/issue-751.pid",
        state_file=state_file,
    )
    assert r1.status == "dead"

    # Tick 2: now absent + live-running. The counter was reset to 0 on tick 1,
    # so this is suppressed (count 1 << cap), proving the bound is CONSECUTIVE.
    monkeypatch.setattr(pp, "_pod_absent_from_pods_conf", lambda _pod: True)
    r2 = pp.poll_once(
        issue=751,
        pod="pod-697",
        log_path="/workspace/logs/issue-751.log",
        pid_file="/workspace/logs/issue-751.pid",
        state_file=state_file,
    )
    assert r2.status == "running"
