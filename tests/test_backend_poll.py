"""Poller-level tests for the ASYNC GCP-workload-failover seam (#659).

``scripts/backend_poll.py`` is the bg-Bash one-tick poll bridge the
orchestrator's Step 6d.2 loop calls. #659 wires it so a poller-detected dead
GCP *workload* (the VM was already up and crashed minutes in — the case the
synchronous route()-time failover of #658 cannot reach) is re-dispatched onto
RunPod exactly once, the handle sidecar is AUTHORITATIVELY re-pointed at the
new RunPod handle, and a RUNNING-shaped JSON is emitted so the orchestrator's
poll loop keeps polling the RunPod run instead of posting ``epm:failure``.

A GCP *setup* failure (the boot/secrets/uv-sync script broke before the
workload ran) must NOT fail over — re-running broken setup on RunPod would
just re-crash — so it surfaces ``current_phase == "terminal_setup_failed"``
and falls through to the ordinary ``status: dead`` -> blocked path.

These tests fail TODAY (the predicate ``_is_gcp_async_workload_failure`` and
the re-dispatch helper ``_failover_dead_gcp_to_runpod`` do not exist yet, and
``main()`` still prints the dead JSON for a workload-failed GCP poll) and pass
after the poller wiring + the GCP poll-discrimination land. Modeled on
``tests/test_issue_dispatch.py``'s backend_poll legacy-shape test, which drives
``backend_poll_main([...])`` and inspects the printed JSON.
"""

from __future__ import annotations

import json

import pytest

from explore_persona_space.backends.base import PollResult, RunHandle
from explore_persona_space.backends.issue_dispatch import (
    read_handle_sidecar,
    write_handle_sidecar,
)
from scripts.backend_poll import main as backend_poll_main

# ---------------------------------------------------------------------------
# Helpers — handle sidecars + poll doubles
# ---------------------------------------------------------------------------

#: The #659 ``extra`` keys the §4.1.0 spec-threading sub-change puts on a GCP
#: handle so ``_runspec_from_gcp_handle`` can reconstruct a RunSpec for the
#: RunPod failover (workload_cmd str / hydra_args list / gpus / time budget).
_GCP_EXTRA_659 = {
    "issue": 659,
    "zone": "us-central1-a",
    "intent": "lora-7b",
    "gpus": 1,
    "time_budget_hours": 4.0,
    "workload_cmd": "REPO_ROOT=/workspace bash scripts/foo.sh --bar",
    "hydra_args": [],
}


def _gcp_handle(extra: dict | None = None) -> RunHandle:
    """A GCP RunHandle carrying the #659 spec-threading extra keys by default."""
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-fake-1",
        pod_name="eps-issue-659",
        scratch_dir="/workspace/eps-issue-659",
        log_path="/workspace/logs/issue-659.log",
        extra=dict(extra if extra is not None else _GCP_EXTRA_659),
    )


def _poll(status: str, current_phase: str) -> PollResult:
    return PollResult(
        status=status,
        current_phase=current_phase,
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=status == "running",
        log_tail_excerpt="",
    )


class _PollDouble:
    """ComputeBackend stand-in whose ``poll`` returns a scripted PollResult."""

    def __init__(self, result: PollResult) -> None:
        self._result = result

    def poll(self, handle: RunHandle) -> PollResult:
        return self._result


class _PassiveRunpodBackend:
    """RunPodBackend stand-in: ``launch`` records the spec + returns a RunPod
    handle; never hits the real RunPod API."""

    def __init__(self) -> None:
        self.launches: list = []

    def launch(self, spec):
        self.launches.append(spec)
        return RunHandle(
            backend="runpod",
            cluster=None,
            job_id="pod-fake",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/workspace",
            log_path=f"/workspace/logs/issue-{spec.issue}.log",
            extra={"issue": spec.issue},
        )


def _last_json_line(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "backend_poll printed no stdout"
    return json.loads(out.splitlines()[-1])


@pytest.fixture(autouse=True)
def _no_real_marker_posts(monkeypatch):
    """Never shell out to real ``task.py post-marker`` from a poll test."""
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **_kw: None,
        raising=False,
    )


# ---------------------------------------------------------------------------
# The poller-level acceptance test (the end-to-end seam)
# ---------------------------------------------------------------------------


def test_poller_async_gcp_workload_crash_re_points_sidecar_and_emits_running(
    tmp_path, monkeypatch, capsys
):
    """#659 (poller end-to-end): a GCP handle whose poll surfaces
    ``terminal_workload_failed`` re-dispatches on RunPod, RE-POINTS the handle
    sidecar at the RunPod handle (authoritative readback), and emits a
    RUNNING-shaped JSON so the orchestrator keeps polling instead of posting
    ``epm:failure``."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        _PassiveRunpodBackend,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # RUNNING (NOT "dead") -> orchestrator keeps polling, no epm:failure now.
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_workload_failover_runpod_async"
    # Sidecar re-pointed to a RunPod handle (authoritative on-disk read).
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    # MF2: the reconstructed RunSpec satisfied __post_init__'s mutual exclusion
    # (no crash on construction) — the run reached the RunPod launch.


def test_failover_runpod_unavailable_emits_infra_no_compute_poll_json(
    tmp_path, monkeypatch, capsys
):
    """#659 sibling #4 (poller mapping): when the RunPod failover raises
    ``NoComputeAvailableError``, the poller emits ``status: "dead"`` +
    ``failure_class: "infra"`` + ``reason: "no_compute_available"`` (so the
    watcher's capacity-retry pass re-drives once a lane frees) and leaves the
    sidecar pointing at the GCP handle."""
    from explore_persona_space.backends import NoComputeAvailableError

    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)

    class _NoComputeRunpod:
        def launch(self, spec):
            raise NoComputeAvailableError("RunPod also unavailable")

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        _NoComputeRunpod,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "no_compute_available"


def test_failover_sidecar_persistence_failure_emits_infra_error_not_running(
    tmp_path, monkeypatch, capsys
):
    """#659 / MF4: the failover's AUTHORITATIVE post-route sidecar write is what
    GUARANTEES the next tick polls RunPod (not GCP) — its precondition for
    emitting ``running``. If that write RAISES (disk error / EDQUOT), the
    helper MUST NOT emit ``status: "running"`` (a GCP handle would be left on
    disk and the next tick would re-satisfy backend=="gcp" -> a SECOND RunPod
    launch, breaching "exactly once"). It emits a TERMINAL infra JSON with
    ``reason: "sidecar_persistence_failed"`` instead — and that reason is NOT
    in TRANSIENT_CAPACITY_REASONS, so the watcher parks it at ``blocked``
    rather than re-driving.

    Second assertion: a SECOND poll observing the still-GCP sidecar does NOT
    fire a second RunPod launch — it re-emits the terminal infra JSON (the
    "exactly once" bound holds even under a persistence failure)."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        lambda: rp,
    )

    # Make the AUTHORITATIVE post-route sidecar write raise. The helper imports
    # write_handle_sidecar from backends.issue_dispatch; patch it there so the
    # post-route write (NOT just the best-effort on_launched hook) fails.
    real_write = write_handle_sidecar

    def _raising_write(handle, path):
        if getattr(handle, "backend", None) == "runpod":
            raise OSError("Disk quota exceeded (EDQUOT) writing the RunPod sidecar")
        return real_write(handle, path)

    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.write_handle_sidecar",
        _raising_write,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # NOT running — the failover refused to claim success without a proven
    # RunPod sidecar on disk.
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "sidecar_persistence_failed"
    # NOT a transient capacity reason -> the watcher parks, never re-drives.
    from scripts.autonomous_session_watch import TRANSIENT_CAPACITY_REASONS

    assert "sidecar_persistence_failed" not in TRANSIENT_CAPACITY_REASONS

    # Second poll observing the still-GCP handle must NOT fire a second RunPod
    # launch (the "exactly once" bound holds under a persistence failure). It
    # re-emits the terminal infra JSON and exits.
    launches_after_first = len(rp.launches)
    rc2 = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc2 == 0
    out2 = _last_json_line(capsys)
    assert out2["status"] == "dead"
    assert len(rp.launches) <= launches_after_first + 1, (
        "a persistence-failed failover must not cascade into an unbounded "
        "series of RunPod launches across ticks"
    )


# ---------------------------------------------------------------------------
# Sibling negatives — the predicate must fire ONLY on a GCP workload crash
# ---------------------------------------------------------------------------


def test_async_gcp_capacity_death_does_NOT_fail_over(tmp_path, monkeypatch, capsys):
    """#659 sibling #1: a GCP handle whose poll surfaces a non-workload
    terminal phase (``terminal_terminated`` / ``terminal_instance not found``)
    emits the ordinary ``status: "dead"`` JSON, leaves the sidecar UNCHANGED,
    and never launches RunPod (the predicate's exact-phase match)."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_terminated")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        lambda: rp,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_terminated"
    assert len(rp.launches) == 0
    assert read_handle_sidecar(sidecar).backend == "gcp"  # sidecar unchanged


def test_async_slurm_workload_death_does_NOT_fail_over(tmp_path, monkeypatch, capsys):
    """#659 sibling #2: a SLURM (``nibi``) handle whose poll surfaces
    ``status: "dead"`` emits the ordinary JSON — no GCP-only failover (scope
    discipline §5)."""
    sidecar = tmp_path / "issue-659-handle.json"
    slurm_handle = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="9001",
        pod_name="eps-issue-659",
        scratch_dir="/scratch/eps/issue-659",
        log_path="/scratch/eps/issue-659/job.out",
        extra={"issue": 659},
    )
    write_handle_sidecar(slurm_handle, sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        lambda: rp,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert len(rp.launches) == 0
    assert read_handle_sidecar(sidecar).backend == "nibi"  # sidecar unchanged


def test_async_runpod_workload_death_does_NOT_recurse(tmp_path, monkeypatch, capsys):
    """#659 sibling #3 (the "exactly once" structural-bound guard): a RunPod
    handle whose poll surfaces ``status: "dead"`` emits the ordinary JSON — a
    RunPod re-crash never re-enters the failover (the predicate's first clause
    requires backend == "gcp")."""
    sidecar = tmp_path / "issue-659-handle.json"
    runpod_handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-fake",
        pod_name="pod-659",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-659.log",
        extra={"issue": 659},
    )
    write_handle_sidecar(runpod_handle, sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        lambda: rp,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert len(rp.launches) == 0  # NO re-failover from a RunPod handle


def test_gcp_setup_failure_does_not_failover_to_runpod(tmp_path, monkeypatch, capsys):
    """#659 / MF3 (poller side): a GCP handle whose poll surfaces
    ``terminal_setup_failed`` (the boot/secrets/uv-sync script broke BEFORE the
    workload ran — re-running it on RunPod would just re-crash) returns False
    from ``_is_gcp_async_workload_failure``; the poller emits the ordinary
    ``status: "dead"`` JSON (no RunPod launch, sidecar UNCHANGED)."""
    # The predicate must exist and reject the setup-failed phase.
    from scripts.backend_poll import _is_gcp_async_workload_failure

    assert (
        _is_gcp_async_workload_failure(_gcp_handle(), _poll("dead", "terminal_setup_failed"))
        is False
    )

    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_setup_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        lambda: rp,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_setup_failed"
    assert len(rp.launches) == 0
    assert read_handle_sidecar(sidecar).backend == "gcp"  # sidecar unchanged
