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
from pathlib import Path

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


@pytest.fixture(autouse=True)
def _isolate_lease_store(monkeypatch, tmp_path):
    """Redirect EVERY ``LeaseStore()`` to a per-test tmp ``~/.eps-routing/``.

    The async failover (``failover_to_runpod_after_async_workload_crash``) and
    the round-3 lease-backed idempotency check (``_lease_records_failover_of`` /
    ``_stamp_lease_failover_of``) both instantiate a bare ``LeaseStore()`` with
    NO injection seam, so they resolve to ``Path.home() / ".eps-routing"``.
    Pinning ``Path.home`` to a fresh tmp dir isolates the durable lease per
    test (no cross-test bleed) AND keeps the suite from writing to the real
    ``~/.eps-routing/`` (a pre-existing leak the round-2 tests had). Used by the
    persistence-failure test below, whose round-3 assertions depend on a clean
    lease at the start of each poll.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))


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

    # Second poll observing the still-GCP handle must NOT fire ANY new RunPod
    # launch (the "exactly once" bound holds under a persistence failure). The
    # failover wrote an idempotency sentinel before returning above; this poll
    # reads it, short-circuits BEFORE the launch, and re-emits the terminal
    # infra JSON (NOT status: "running" — the sidecar still holds the GCP
    # handle, so claiming the run is alive would be a lie).
    launches_after_first = len(rp.launches)
    rc2 = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc2 == 0
    out2 = _last_json_line(capsys)
    # (a) ZERO additional launches — the second poll re-launches nothing.
    assert len(rp.launches) == launches_after_first, (
        "a persistence-failed failover must fire RunPod EXACTLY ONCE — the "
        "repeat poll observing the unchanged GCP sidecar must launch nothing"
    )
    # (b) The repeat poll's emitted JSON is still the terminal infra
    # sidecar_persistence_failed (NOT status: "running").
    assert out2["status"] == "dead"
    assert out2["failure_class"] == "infra"
    assert out2["reason"] == "sidecar_persistence_failed"


def test_failover_both_sidecar_AND_sentinel_writes_fail_still_exactly_once(
    tmp_path, monkeypatch, capsys
):
    """#659 round-3 (the round-2 GAP — EDQUOT/persistent-disk-failure mode).

    The round-2 fix wrote an idempotency SENTINEL on the persistence-failure
    path — but the sentinel and the handle sidecar share the SAME
    ``.claude/cache/`` directory, so the canonical project failure mode (EDQUOT
    on the MooseFS per-pod quota, a read-only filesystem, out-of-inodes) that
    fails the sidecar write ALSO fails the sentinel write. With BOTH writes
    failing, round-2's "exactly once" degraded to one extra paid RunPod launch
    PER POLL TICK (the sentinel is absent on every subsequent poll, so the
    sentinel short-circuit never fires while the disk failure persists — and
    EDQUOT does not clear between ~540s polls).

    Round-3 makes the DURABLE lease at ``~/.eps-routing/`` (a DIFFERENT
    directory, stamped BEFORE the sidecar write) the authoritative idempotency
    record. This test models the round-2 gap: BOTH the RunPod sidecar write AND
    the sentinel write fail, and the SECOND poll must STILL launch nothing,
    because the lease record survived the ``.claude/cache``-wide failure. This
    is the test the round-2 fix does NOT pass."""
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

    # (1) The RunPod sidecar write fails (EDQUOT) — same as the round-2 test.
    real_write = write_handle_sidecar

    def _raising_write(handle, path):
        if getattr(handle, "backend", None) == "runpod":
            raise OSError("Disk quota exceeded (EDQUOT) writing the RunPod sidecar")
        return real_write(handle, path)

    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.write_handle_sidecar",
        _raising_write,
    )
    # (2) The SENTINEL write ALSO fails — the round-2 gap. _write_failover_sentinel
    # swallows its own OSError, so the on-disk effect of an EDQUOT'd sentinel
    # write is simply "no sentinel persisted": model that as a no-op. With BOTH
    # the sidecar AND the sentinel unwritable, the lease is the ONLY surviving
    # idempotency record.
    monkeypatch.setattr(
        "scripts.backend_poll._write_failover_sentinel",
        lambda *a, **k: None,
    )

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "sidecar_persistence_failed"
    # The first poll launched RunPod exactly once.
    launches_after_first = len(rp.launches)
    assert launches_after_first == 1

    # Confirm the sentinel really is absent on disk (the round-2 record is gone).
    from scripts.backend_poll import _failover_sentinel_path

    assert not _failover_sentinel_path(sidecar).exists(), (
        "the sentinel write was made a no-op — the round-2 record must be ABSENT, "
        "so only the round-3 lease can carry idempotency forward"
    )

    # SECOND poll: sidecar still GCP, sentinel still absent. The ONLY thing that
    # can stop a second paid launch is the durable lease. With the round-2
    # sentinel-only fix this poll fires a SECOND RunPod launch; with the round-3
    # lease check it short-circuits.
    rc2 = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc2 == 0
    out2 = _last_json_line(capsys)
    assert len(rp.launches) == launches_after_first, (
        "EDQUOT failed BOTH the sidecar AND the sentinel write, yet the repeat "
        "poll must STILL launch nothing — the durable ~/.eps-routing/ lease is "
        "the surviving idempotency record (round-3 fix)"
    )
    assert out2["status"] == "dead"
    assert out2["failure_class"] == "infra"
    assert out2["reason"] == "sidecar_persistence_failed"


def test_lease_records_failover_of_is_keyed_per_gcp_crash_not_per_issue(tmp_path, monkeypatch):
    """#659 round-3: the durable lease idempotency check matches the SPECIFIC
    crashed GCP run, so it is "exactly once PER GCP CRASH", not "per issue".

    A stamped RunPod-failover lease suppresses a repeat poll of the SAME GCP run
    (identity match) but NOT a genuinely-new GCP run on the same issue (a fresh
    dispatch -> new pod_name/job_id), which must still get its own one failover.
    Also: a lease whose backend is NOT ``runpod`` (a fresh GCP dispatch wrote it)
    never suppresses, even if a stale ``gcp_failover_of`` lingers on it."""
    from explore_persona_space.backends.router import Lease, LeaseStore
    from scripts.backend_poll import _gcp_handle_identity, _lease_records_failover_of

    store = LeaseStore(lease_dir=tmp_path / ".eps-routing")
    crashed = _gcp_handle()  # pod_name="eps-issue-659", job_id="instance-fake-1"

    # No lease at all -> no suppression.
    assert _lease_records_failover_of(659, crashed, lease_store=store) is False

    # A RunPod lease stamped with THIS GCP run's identity -> suppress.
    store.write(
        Lease(
            issue=659,
            spec_hash="deadbeef",
            attempt_id="att-1",
            backend="runpod",
            job_id="pod-fake",
            gcp_failover_of=_gcp_handle_identity(crashed),
        )
    )
    assert _lease_records_failover_of(659, crashed, lease_store=store) is True

    # A genuinely-NEW GCP run on the same issue (different pod_name/job_id) does
    # NOT match the stamp -> NOT suppressed (per-crash, not per-issue).
    new_gcp = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-fake-2",  # NEW job id
        pod_name="eps-issue-659-retry",  # NEW pod name
        scratch_dir="/workspace/eps-issue-659",
        log_path="/workspace/logs/issue-659.log",
        extra=dict(_GCP_EXTRA_659),
    )
    assert _lease_records_failover_of(659, new_gcp, lease_store=store) is False

    # A non-runpod lease (a fresh GCP dispatch re-wrote it) never suppresses,
    # even with a stale gcp_failover_of carried over.
    store.write(
        Lease(
            issue=659,
            spec_hash="cafef00d",
            attempt_id="att-2",
            backend="gcp",
            job_id="instance-fake-2",
            gcp_failover_of=_gcp_handle_identity(crashed),  # stale stamp
        )
    )
    assert _lease_records_failover_of(659, crashed, lease_store=store) is False


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


# ---------------------------------------------------------------------------
# issue #669 — poller-side hung-but-RUNNING wedge detection (the staleness
# clock + _maybe_escalate_gcp_wedge), the false-positive controls, and the
# watchdog-terminated failover, all driven end-to-end via backend_poll_main.
# ---------------------------------------------------------------------------

import time as _time  # noqa: E402  (module-level: the #669 clock tests stamp epoch seconds)


def _running_poll(phase: str, *, reachability_alarm: bool) -> PollResult:
    """A RUNNING GCP PollResult carrying the #669 typed reachability_alarm."""
    return PollResult(
        status="running",
        current_phase=phase,
        new_milestone=False,
        last_log_mtime_sec_ago=10**9,
        pid_alive=True,
        log_tail_excerpt="",
        reachability_alarm=reachability_alarm,
    )


def _gcp_handle_with_clock(*, phase: str, ts: float) -> RunHandle:
    """A GCP handle whose sidecar extra carries the staleness clock keys."""
    extra = dict(_GCP_EXTRA_659)
    extra["last_phase"] = phase
    extra["last_phase_change_ts"] = ts
    return _gcp_handle(extra=extra)


def test_poll_running_with_frozen_phase_and_drain_timeout_returns_terminal_wedged(
    tmp_path, monkeypatch, capsys
):
    """Fix 1 POSITIVE (#669, end-to-end): a RUNNING GCP poll whose non-terminal
    phase ('workload') is frozen past the 15-min floor AND carries a
    transport-class reachability alarm is escalated to status=dead /
    terminal_workload_wedged, which the async-failover predicate then matches —
    the run fails over to RunPod exactly once."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="workload", ts=_time.time() - 1000), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_running_poll("workload", reachability_alarm=True)),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # Escalated to wedged -> failed over -> RUNNING-shaped async-failover JSON.
    assert out["current_phase"] == "gcp_workload_failover_runpod_async"
    assert out["status"] == "running"
    assert len(rp.launches) == 1
    assert read_handle_sidecar(sidecar).backend == "runpod"


def test_poll_running_with_recent_phase_change_stays_running(tmp_path, monkeypatch, capsys):
    """Fix 1 NEGATIVE control A (#669): a phase that changed WITHIN the floor
    (even with a reachability alarm) stays running — no false wedge on a
    healthy run whose last phase write is recent."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="workload", ts=_time.time() - 30), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_running_poll("workload", reachability_alarm=True)),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "workload"
    assert len(rp.launches) == 0
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_poll_running_with_frozen_phase_but_no_drain_alarm_stays_running(
    tmp_path, monkeypatch, capsys
):
    """M2.1 (#669): a phase frozen past the floor but with NO reachability
    alarm (SSH works — a healthy run that simply hasn't written a phase in
    >15 min, e.g. a long training epoch) stays running. The reachability-alarm
    conjunction is what separates a wedge from a slow healthy phase."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="workload", ts=_time.time() - 1000), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_running_poll("workload", reachability_alarm=False)),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert len(rp.launches) == 0
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_poll_running_with_frozen_phase_and_sentinel_processing_alarm_stays_running(
    tmp_path, monkeypatch, capsys
):
    """M2.4 (#669, the M1 signal split): a phase frozen past the floor with a
    SENTINEL-PROCESSING-class alarm (a healthy VM with a malformed sentinel /
    transient marker-post failure) stays running — the poll's
    reachability_alarm is False on that class (set only on transport), so the
    wedge gate never fires. Modeled as the poll producing reachability_alarm
    False (the producer-side split is pinned in test_gcp_backend.py)."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="workload", ts=_time.time() - 1000), sidecar)
    rp = _PassiveRunpodBackend()
    # A sentinel-processing alarm leaves reachability_alarm=False (the producer
    # split). The poller must NOT wedge on it.
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_running_poll("workload", reachability_alarm=False)),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert len(rp.launches) == 0


def test_two_tick_phase_transition_resets_clock_and_stays_running(tmp_path, monkeypatch, capsys):
    """M2.2 (#669, cross-tick reset): tick 1 observes phase A and stamps the
    clock; tick 2 observes a DIFFERENT (advanced) phase B even though the
    sidecar's recorded change-ts is older than the floor — the phase ADVANCED,
    so the clock is re-stamped and the tick stays running. The clock resets on
    every phase transition (no false wedge after a long-but-progressing run)."""
    sidecar = tmp_path / "issue-659-handle.json"
    # Sidecar recorded phase A at a stale ts (older than the floor).
    write_handle_sidecar(_gcp_handle_with_clock(phase="phase_A", ts=_time.time() - 1000), sidecar)
    rp = _PassiveRunpodBackend()
    # The poll now observes a DIFFERENT phase (advanced), with a reachability
    # alarm — but because the phase CHANGED, the wedge must not fire.
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_running_poll("phase_B", reachability_alarm=True)),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "phase_B"
    assert len(rp.launches) == 0
    # The clock was re-stamped to the new phase with a FRESH ts.
    recovered = read_handle_sidecar(sidecar)
    assert recovered.extra["last_phase"] == "phase_B"
    assert recovered.extra["last_phase_change_ts"] > _time.time() - 60


def test_first_poll_with_no_phase_clock_stamps_and_stays_running(tmp_path, monkeypatch, capsys):
    """M2.3 (#669, first-launch missing keys): a freshly-dispatched sidecar with
    NO last_phase / last_phase_change_ts keys fails toward running on the first
    poll (even with a reachability alarm) AND initializes both clock keys —
    fresh-dispatch handles never false-wedge."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)  # no clock keys
    assert "last_phase" not in read_handle_sidecar(sidecar).extra
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_running_poll("workload", reachability_alarm=True)),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert len(rp.launches) == 0
    # Both clock keys were initialized (the first observation stamps them).
    recovered = read_handle_sidecar(sidecar)
    assert recovered.extra["last_phase"] == "workload"
    assert isinstance(recovered.extra["last_phase_change_ts"], (int, float))
    assert recovered.backend == "gcp"


def test_terminal_workload_wedged_fails_over_to_runpod_exactly_once(tmp_path, monkeypatch, capsys):
    """Fix 4 (#669): a GCP poll that surfaces status=dead / terminal_workload_wedged
    DIRECTLY (the poller-synthesized wedge phase) routes to the async failover
    and launches RunPod exactly once — the accept-set recognizes the phase."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_wedged")),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["current_phase"] == "gcp_workload_failover_runpod_async"
    assert len(rp.launches) == 1
    assert read_handle_sidecar(sidecar).backend == "runpod"


def test_terminal_wedged_terminated_fails_over_to_runpod_exactly_once(
    tmp_path, monkeypatch, capsys
):
    """Fix 4 (#669): a GCP poll that surfaces status=dead /
    terminal_wedged_terminated (the in-VM watchdog self-shutdown path, the
    Option-2 conservative phase) routes to the async failover and launches
    RunPod exactly once."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_wedged_terminated")),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["current_phase"] == "gcp_workload_failover_runpod_async"
    assert len(rp.launches) == 1
    assert read_handle_sidecar(sidecar).backend == "runpod"


# ---------------------------------------------------------------------------
# issue #669 code-review r1 (Blocker B): the 2nd triggerer's M3b short-circuit
# must PRESERVE the 1st triggerer's authoritative full RunPod sidecar — NOT
# clobber it with the minimal reconstructed handle (extra={"issue": N} only),
# which would strip expected_artifacts and FAIL downstream artifact
# verification. Sibling of test_router.py::
# test_concurrent_failover_triggers_single_runpod_launch (M2.6), which asserts
# only launch-atomicity (ONE launch), never that the FIRST launch's sidecar
# metadata survives the SECOND triggerer's return.
# ---------------------------------------------------------------------------


class _FullHandleRunpodBackend:
    """RunPodBackend stand-in whose ``launch`` returns a handle carrying the FULL
    ``extra`` a real launch declares — ``expected_artifacts`` / ``pid_file`` /
    ``runpod_attempt_id`` — so the 1st triggerer writes an AUTHORITATIVE sidecar
    a 2nd-triggerer clobber would be detectable against."""

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
            extra={
                "issue": spec.issue,
                "expected_artifacts": {
                    "issue": spec.issue,
                    "sentinel_path": f"/workspace/eval_results/issue_{spec.issue}/.sentinel.json",
                },
                "pid_file": f"/workspace/logs/issue-{spec.issue}.pid",
                "runpod_attempt_id": "att-runpod-001",
            },
        )


def test_concurrent_failover_preserves_first_runpod_sidecar(tmp_path, monkeypatch):
    """#669 code-review r1 Blocker B: two CONCURRENT triggerers of the SAME
    GCP-crash failover (the wedge classifier + the watchdog-TERMINATED path on
    one handle) both call ``_failover_dead_gcp_to_runpod``. The 1st launches
    RunPod and writes the full RunPod handle to the sidecar; the 2nd hits the
    router's M3b in-flock re-check, which returns a MINIMAL reconstructed handle
    (``extra={"issue": N}`` only) flagged ``failover_already_launched``. The
    fixed poller MUST detect that flag and PRESERVE the 1st triggerer's sidecar
    rather than overwriting it with the minimal handle.

    Pre-fix: the 2nd call's unconditional ``write_handle_sidecar`` clobbered the
    sidecar, so ``expected_artifacts`` / ``pid_file`` / ``runpod_attempt_id``
    vanished and artifact verification later read None and FAILED. Post-fix: the
    sidecar still carries the full handle from the 1st launch, byte-for-byte.

    Modelling the genuine flock race deterministically: in production BOTH
    triggerers pass the OUTSIDE-the-flock pre-check (``_lease_records_failover_of``
    / sentinel) BEFORE the 1st stamps the durable lease, then serialize on the
    per-issue flock INSIDE the router. To reach the router's in-flock short-circuit
    on the 2nd call sequentially, the 2nd call's outside pre-check must read the
    pre-stamp state — so it is pinned False here (the 1st's router transaction has
    ALREADY stamped the SHARED lease store, so the router itself still
    short-circuits). Without this pin the redundant outside pre-check would
    intercept the 2nd call first (the cross-tick idempotency path, which does NOT
    clobber); the in-flock short-circuit — and thus the Blocker-B guard — is only
    reached under true concurrent contention."""
    import scripts.backend_poll as bp
    from scripts.backend_poll import _failover_dead_gcp_to_runpod

    issue = 659
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)

    rp = _FullHandleRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    handle = _gcp_handle()
    result = _poll("dead", "terminal_workload_failed")

    # 1st triggerer — launches RunPod, writes the AUTHORITATIVE full sidecar.
    first = _failover_dead_gcp_to_runpod(issue=issue, handle=handle, result=result, sidecar=sidecar)
    assert first["status"] == "running"
    assert first["current_phase"] == "gcp_workload_failover_runpod_async"
    after_first = read_handle_sidecar(sidecar)
    assert after_first.backend == "runpod"
    assert after_first.extra.get("expected_artifacts") is not None
    full_extra_snapshot = dict(after_first.extra)

    # Model the concurrent race: the 2nd triggerer's OUTSIDE pre-check ran before
    # the 1st's durable stamp landed, so it reads a clean lease and proceeds into
    # the router (where the SHARED lease IS stamped → in-flock short-circuit).
    monkeypatch.setattr(bp, "_lease_records_failover_of", lambda *a, **k: False)

    # 2nd triggerer — same GCP handle → router M3b in-flock re-check short-circuits
    # with extra["failover_already_launched"]=True + the minimal handle.
    second = _failover_dead_gcp_to_runpod(
        issue=issue, handle=handle, result=result, sidecar=sidecar
    )
    assert second["status"] == "running"  # still "keep polling", NOT a terminal/dead

    # THE BLOCKER ASSERTION: the sidecar STILL carries the 1st triggerer's full
    # RunPod handle — expected_artifacts / pid_file / runpod_attempt_id all
    # present and UNCHANGED. NOT clobbered to a minimal extra={"issue": N}.
    after_second = read_handle_sidecar(sidecar)
    assert after_second.backend == "runpod"
    assert after_second.extra.get("expected_artifacts") is not None, (
        "2nd triggerer clobbered the 1st's sidecar — expected_artifacts is gone "
        "(the Blocker-B regression)"
    )
    assert after_second.extra.get("pid_file") is not None
    assert after_second.extra.get("runpod_attempt_id") is not None
    assert after_second.extra == full_extra_snapshot  # byte-for-byte preserved

    # M2.6 atomicity reaffirmed at the poller level: exactly ONE RunPod launch
    # across BOTH invocations.
    assert len(rp.launches) == 1
