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
    # #677: the post-#677 GCP handle carries the resolved machine's gpu_count.
    # A GPU intent is >=1, so the new CPU-exclusion conjunct in
    # _is_gcp_async_workload_failure is a no-op for these GPU failover tests
    # (they assert on PHASE, not gpu_count, so they stay green with this key
    # present). The CPU case overrides this to 0 in its own test.
    "gpu_count": 1,
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


# ---------------------------------------------------------------------------
# CPU-only GCP handle: NO async RunPod failover (#677)
# ---------------------------------------------------------------------------


def test_async_failover_skips_cpu_gcp_handle(tmp_path, monkeypatch, capsys):
    """#677: a CPU GCP handle (extra.gpu_count==0) whose poll surfaces
    terminal_workload_failed does NOT fail over to RunPod (RunPod is GPU-only).
    It emits the ordinary dead JSON; RunPodBackend is never constructed.

    RunPodBackend is monkeypatched to RAISE if constructed — the strongest
    "never touched RunPod" assertion: if the predicate fails to exclude the CPU
    handle, _failover_dead_gcp_to_runpod constructs RunPodBackend() and the test
    fails with the explicit AssertionError, not a downstream crash.
    """
    cpu_extra = dict(_GCP_EXTRA_659)
    cpu_extra["intent"] = "cpu-bigmem"
    cpu_extra["gpu_count"] = 0
    sidecar = tmp_path / "issue-677-handle.json"
    write_handle_sidecar(_gcp_handle(extra=cpu_extra), sidecar)

    def _boom(*_a, **_k):
        raise AssertionError("RunPodBackend must NOT be constructed for a CPU GCP handle (#677)")

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        _boom,
    )

    rc = backend_poll_main(["--issue", "677", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # Ordinary dead path (NOT failed over to RunPod).
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_workload_failed"
    # Sidecar unchanged (still a GCP handle — no RunPod re-point).
    assert read_handle_sidecar(sidecar).backend == "gcp"


# ---------------------------------------------------------------------------
# Cheap CPU-intent async GCP→RunPod failover relaxation (#747)
# ---------------------------------------------------------------------------
#
# NOTE: the cpu-bigmem async "does NOT fail over" guard is the EXISTING test
# above — test_async_failover_skips_cpu_gcp_handle — which uses
# intent="cpu-bigmem" + gpu_count==0. We do not duplicate it.


def _cpu_gcp_handle(intent: str) -> RunHandle:
    """A GCP RunHandle for a CPU intent (gpu_count==0 + the named intent)."""
    extra = dict(_GCP_EXTRA_659)
    extra["intent"] = intent
    extra["gpu_count"] = 0
    return _gcp_handle(extra=extra)


def test_async_cpu_small_handle_predicate_is_failover_eligible() -> None:
    """#747: the predicate _is_gcp_async_workload_failure returns True for a
    cpu-small GCP handle (gpu_count==0, intent IN RUNPOD_CPU_INSTANCE_FOR_INTENT)
    at terminal_workload_failed — the #677 CPU exclusion is RELAXED for a mapped
    intent. The companion cpu-bigmem case (NOT in the map) stays False."""
    from scripts.backend_poll import _is_gcp_async_workload_failure

    assert (
        _is_gcp_async_workload_failure(
            _cpu_gcp_handle("cpu-small"), _poll("dead", "terminal_workload_failed")
        )
        is True
    )
    # cpu-bigmem (NOT in the map) stays EXCLUDED.
    assert (
        _is_gcp_async_workload_failure(
            _cpu_gcp_handle("cpu-bigmem"), _poll("dead", "terminal_workload_failed")
        )
        is False
    )


def test_async_cpu_small_handle_fails_over_to_runpod(tmp_path, monkeypatch, capsys):
    """#747 (poller end-to-end): a cpu-small GCP handle at
    terminal_workload_failed re-dispatches on RunPod (carrying the CPU intent),
    re-points the sidecar at the RunPod handle, and emits the RUNNING-shaped
    failover JSON — symmetric with the #659 GPU async-failover path."""
    sidecar = tmp_path / "issue-747-handle.json"
    write_handle_sidecar(_cpu_gcp_handle("cpu-small"), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod.RunPodBackend",
        _PassiveRunpodBackend,
    )

    rc = backend_poll_main(["--issue", "747", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_workload_failover_runpod_async"
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"


def test_async_cpu_handle_no_intent_key_does_not_fail_over() -> None:
    """#747 fail-safe: a CPU GCP handle with gpu_count==0 and NO intent key (a
    pre-#747 shape) is treated as NOT-mapped -> EXCLUDED (returns False), the
    safe #677 behavior on a missing key. Statistics concern #3."""
    from scripts.backend_poll import _is_gcp_async_workload_failure

    extra = dict(_GCP_EXTRA_659)
    extra["gpu_count"] = 0
    extra.pop("intent", None)
    handle = _gcp_handle(extra=extra)
    assert (
        _is_gcp_async_workload_failure(handle, _poll("dead", "terminal_workload_failed")) is False
    )


# ---------------------------------------------------------------------------
# #783 — GCP FLEX_START queue-timeout → RunPod failover, end-to-end
#
# A GCP FLEX_START create can SUCCEED yet leave the instance in the capacity
# QUEUE (PENDING, polled as current_phase="pending", #782/#778) — a state the
# router's route()-time park cannot bound (the wait happens AFTER route()
# returns). The async poller ages the "pending" phase against
# EPS_GCP_QUEUE_WAIT_SECONDS and, past the floor, cancels the queued instance +
# fails over to RunPod (reason: gcp_queue_timeout_failover_runpod). These tests
# drive backend_poll_main and inspect the printed JSON + the sidecar re-point,
# modeled on the #669 wedge tests above.
# ---------------------------------------------------------------------------


class _PollDoubleWithTeardown:
    """A ComputeBackend stand-in whose ``poll`` is scripted AND whose
    ``teardown`` is recorded — the queue-timeout failover deletes the still-
    queued GCP instance via ``_resolve_backend("gcp").teardown(handle)`` before
    re-dispatch, so the poll double the queue-timeout tests patch in must expose
    a ``teardown`` method (``_PollDouble`` alone has only ``poll``)."""

    def __init__(self, result: PollResult, *, teardown_raises: BaseException | None = None) -> None:
        self._result = result
        self._teardown_raises = teardown_raises
        self.teardowns: list = []

    def poll(self, handle: RunHandle) -> PollResult:
        return self._result

    def teardown(self, handle: RunHandle) -> None:
        self.teardowns.append(handle)
        if self._teardown_raises is not None:
            raise self._teardown_raises


def _pending_poll() -> PollResult:
    """The FLEX_START capacity-queue poll: status=running / current_phase=pending
    (what ``gcp._gcp_status_to_poll_result`` maps GCE PENDING to, #782/#778)."""
    return PollResult(
        status="running",
        current_phase="pending",
        new_milestone=False,
        last_log_mtime_sec_ago=10**9,
        pid_alive=True,
        log_tail_excerpt="",
    )


def test_gcp_pending_past_timeout_fails_over_to_runpod(tmp_path, monkeypatch, capsys):
    """#783 HEADLINE acceptance (success criterion a): a GCP handle polling
    current_phase="pending" whose queue clock is older than
    EPS_GCP_QUEUE_WAIT_SECONDS is escalated to terminal_queue_timeout, the queued
    instance is torn down, and the run fails over to RunPod exactly once — the
    printed JSON carries current_phase="gcp_queue_timeout_failover_runpod",
    status="running", and the sidecar is re-pointed at the RunPod handle."""
    sidecar = tmp_path / "issue-783-handle.json"
    # Queue clock recorded "pending" at a ts older than the 600s floor.
    write_handle_sidecar(_gcp_handle_with_clock(phase="pending", ts=_time.time() - 1000), sidecar)

    teardown_backend = _PollDoubleWithTeardown(_pending_poll())
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: teardown_backend,
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # Success criterion (a) + (d): failed over with the queue-timeout reason as
    # the running-shaped JSON's current_phase.
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_queue_timeout_failover_runpod"
    # Exactly ONE RunPod launch.
    assert len(rp.launches) == 1
    # The queued GCP instance was torn down before the re-dispatch.
    assert len(teardown_backend.teardowns) == 1
    # Sidecar authoritatively re-pointed at the RunPod handle.
    assert read_handle_sidecar(sidecar).backend == "runpod"


def test_gcp_queue_timeout_failover_marker_carries_queue_timeout_reason(
    tmp_path, monkeypatch, capsys
):
    """#783 success criterion (d): the epm:backend-selected marker the failover
    posts carries reason=gcp_queue_timeout_failover_runpod in its JSON note.

    The poller wires the real router marker poster (post_marker_via_task_py); we
    capture the marker kwargs to inspect the posted reason instead of shelling
    out to task.py."""
    import scripts.backend_poll as bp

    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="pending", ts=_time.time() - 1000), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDoubleWithTeardown(_pending_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    # Capture the marker poster the failover threads into the router.
    captured: list[dict] = []
    monkeypatch.setattr(
        bp, "post_marker_via_task_py", lambda **kw: captured.append(kw), raising=False
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **kw: captured.append(kw),
        raising=False,
    )

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    backend_selected = [m for m in captured if m.get("marker") == "epm:backend-selected"]
    assert backend_selected, "no epm:backend-selected marker posted by the queue-timeout failover"
    body = json.loads(backend_selected[-1]["note"])
    assert body["reason"] == "gcp_queue_timeout_failover_runpod"


def test_gcp_queue_timeout_does_NOT_increment_gcp_attempts_today(tmp_path, monkeypatch, capsys):
    """#783 success criterion (e): a queue-timeout cancel is a CLEAN advance — it
    does NOT bump the per-day GCP attempt counter (that bumps only inside
    _attempt_one_gcp_rung's create path, which the poller never re-enters).

    Assert directly: after a full queue-timeout failover, the durable lease's
    gcp_attempts_today is UNCHANGED from its pre-failover value. The lease store
    is isolated to a per-test ~/.eps-routing by the autouse fixture."""
    from explore_persona_space.backends.router import Lease, LeaseStore

    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="pending", ts=_time.time() - 1000), sidecar)

    # Seed a lease recording the GCP attempt count already spent this run.
    store = LeaseStore()  # resolves to the tmp ~/.eps-routing (autouse fixture)
    store.write(Lease(issue=783, spec_hash="deadbeef", attempt_id="att-1", gcp_attempts_today=3))

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDoubleWithTeardown(_pending_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    assert len(rp.launches) == 1  # the failover fired
    lease_after = store.read(783)
    assert lease_after is not None
    # The queue-timeout cancel did NOT touch the GCP attempt counter.
    assert lease_after.gcp_attempts_today == 3


# ── #783 negative controls (false-positive guards) ───────────────────────────


def test_gcp_pending_within_floor_stays_running_no_failover(tmp_path, monkeypatch, capsys):
    """#783 negative control: a "pending" poll WITHIN the queue-wait floor (clock
    age < EPS_GCP_QUEUE_WAIT_SECONDS) stays running — no escalation, no
    teardown, no RunPod launch."""
    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="pending", ts=_time.time() - 30), sidecar)

    teardown_backend = _PollDoubleWithTeardown(_pending_poll())
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: teardown_backend)

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed within the queue-wait floor (#783)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "pending"
    assert len(teardown_backend.teardowns) == 0
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_running_phase_not_pending_does_not_trip_queue_timeout(tmp_path, monkeypatch, capsys):
    """#783 negative control: a GCP poll whose phase is "running" (dequeued and
    up) — NOT "pending" — is untouched even with a stale clock, because the
    queue-timeout predicate scopes to current_phase=="pending" ONLY."""
    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="running", ts=_time.time() - 1000), sidecar)

    running_poll = PollResult(
        status="running",
        current_phase="running",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
    )
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDoubleWithTeardown(running_poll),
    )

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed for a non-pending phase (#783)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "running"
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_first_pending_observation_stamps_clock_stays_running(tmp_path, monkeypatch, capsys):
    """#783 negative control: the FIRST "pending" observation (a fresh-dispatch
    handle whose sidecar carries NO queue clock, last_ts is None) re-stamps the
    clock and stays running — a freshly-queued instance is never failed over on
    its very first poll, however long the create itself took."""
    sidecar = tmp_path / "issue-783-handle.json"
    # A GCP handle with NO last_phase / last_phase_change_ts keys in extra.
    write_handle_sidecar(_gcp_handle(), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDoubleWithTeardown(_pending_poll()),
    )

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed on the first pending poll (#783)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "pending"
    # The clock was stamped so a later stale poll CAN trip — verify the sidecar
    # now carries the "pending" clock keys.
    from scripts.backend_poll import _read_phase_clock

    last_phase, last_ts = _read_phase_clock(sidecar)
    assert last_phase == "pending"
    assert last_ts is not None


def test_gcp_cpu_bigmem_pending_past_floor_does_NOT_fail_over(tmp_path, monkeypatch, capsys):
    """#783 negative control: a cpu-bigmem GCP handle (gpu_count==0, intent NOT
    in RUNPOD_CPU_INSTANCE_FOR_INTENT) stuck "pending" past the floor is escalated
    to terminal_queue_timeout by the clock BUT the _is_gcp_queue_timeout predicate
    EXCLUDES it (no RunPod CPU lane), so it falls through to the ordinary dead
    path — RunPod is never constructed."""
    cpu_extra = dict(_GCP_EXTRA_659)
    cpu_extra["intent"] = "cpu-bigmem"
    cpu_extra["gpu_count"] = 0
    cpu_extra["last_phase"] = "pending"
    cpu_extra["last_phase_change_ts"] = _time.time() - 1000
    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle(extra=cpu_extra), sidecar)

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed for a cpu-bigmem queue-stall (#783)")

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDoubleWithTeardown(_pending_poll()),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # Escalated to terminal_queue_timeout (the clock rewrote it) but the CPU
    # predicate excluded it from failover -> ordinary dead path.
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_queue_timeout"
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_cpu_small_pending_past_floor_predicate_is_eligible() -> None:
    """#783: the queue-timeout predicate _is_gcp_queue_timeout returns True for a
    cpu-small handle at terminal_queue_timeout (mapped CPU intent, #747-relaxed)
    and False for cpu-bigmem — mirroring the #659 CPU-intent guard exactly."""
    from scripts.backend_poll import _is_gcp_queue_timeout

    def _tq_poll() -> PollResult:
        return _poll("dead", "terminal_queue_timeout")

    assert _is_gcp_queue_timeout(_cpu_gcp_handle("cpu-small"), _tq_poll()) is True
    assert _is_gcp_queue_timeout(_cpu_gcp_handle("cpu-bigmem"), _tq_poll()) is False
    # A GPU handle is eligible; a non-GCP / non-dead / wrong-phase handle is not.
    assert _is_gcp_queue_timeout(_gcp_handle(), _tq_poll()) is True
    assert _is_gcp_queue_timeout(_gcp_handle(), _poll("dead", "terminal_workload_failed")) is False
    assert _is_gcp_queue_timeout(_gcp_handle(), _poll("running", "pending")) is False


def test_gcp_queue_timeout_teardown_failure_still_fails_over(tmp_path, monkeypatch, capsys):
    """#783 robustness: if the best-effort teardown of the queued instance RAISES
    (transient gcloud error), the failover STILL proceeds — the teardown is a
    cleanliness step (the stale-GCP-VM janitor is the backstop), never a
    precondition of the RunPod re-dispatch."""
    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="pending", ts=_time.time() - 1000), sidecar)

    teardown_backend = _PollDoubleWithTeardown(
        _pending_poll(), teardown_raises=RuntimeError("gcloud delete transient failure")
    )
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: teardown_backend)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_queue_timeout_failover_runpod"
    assert len(teardown_backend.teardowns) == 1  # teardown was attempted
    assert len(rp.launches) == 1  # and the failover still fired
    assert read_handle_sidecar(sidecar).backend == "runpod"


# ---------------------------------------------------------------------------
# #775 — RunPod CUDA-IMA repeat-failover, end-to-end via backend_poll_main
# ---------------------------------------------------------------------------

#: A realistic vLLM CUDA-IMA crash surface whose signature line is in the body.
_CUDA_IMA_SURFACE = (
    "step 41 forward ok\n"
    "ERROR torch.AcceleratorError: CUDA error: an illegal memory access was encountered\n"
    "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue\n"
    "(EngineCore_DP0 pid=99) Engine core proc EngineCore_DP0 died unexpectedly\n"
    "INFO shutting down client\nsubprocess returncode: 1\n"
)


def _runpod_handle_775(extra_overrides: dict | None = None) -> RunHandle:
    """A router-launched-shape RunPod handle (carries the relaunch RunSpec fields)."""
    extra = {
        "issue": 775,
        "intent": "lora-7b",
        "workload_cmd": "bash scripts/issue664_dispatch.sh --foo",
        "hydra_args": [],
        "gpus": 1,
        "time_budget_hours": 4.0,
    }
    if extra_overrides:
        extra.update(extra_overrides)
    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-fake-775",
        pod_name="pod-775",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-775.log",
        extra=extra,
    )


def _cuda_ima_dead_poll_base() -> PollResult:
    """A dead base.PollResult carrying the CUDA-IMA crash surface on crash_signature."""
    return PollResult(
        status="dead",
        current_phase="dead",
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="subprocess returncode: 1",
        crash_signature=_CUDA_IMA_SURFACE,
    )


def _seed_cuda_ima_record(sidecar: Path) -> None:
    """Write a prior-CUDA-IMA-crash record into the sidecar extra (a prior crash
    this run), so the NEXT CUDA-IMA poll is a SECOND same-signature repeat."""
    payload = json.loads(sidecar.read_text())
    payload.setdefault("extra", {})["runpod_cuda_ima_last_seen"] = {"ts": 1.0, "sig": "cuda_ima"}
    sidecar.write_text(json.dumps(payload))


def test_poller_first_cuda_ima_falls_through_to_dead(tmp_path, monkeypatch, capsys):
    """The FIRST CUDA-IMA dead poll (no prior record) records + falls through to
    the ordinary dead path — no escalation, no failover."""
    sidecar = tmp_path / "issue-775-handle.json"
    write_handle_sidecar(_runpod_handle_775(), sidecar)
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_cuda_ima_dead_poll_base()),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    from scripts.backend_poll import RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE

    rc = backend_poll_main(["--issue", "775", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] != RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE
    assert out["current_phase"] != "runpod_noport_wedge_failover_fresh_pod"
    assert len(rp.launches) == 0  # no failover on the first crash
    # The record was written so the NEXT crash is a repeat.
    payload = json.loads(sidecar.read_text())
    assert "runpod_cuda_ima_last_seen" in (payload.get("extra") or {})


def test_poller_second_cuda_ima_emits_fresh_host_failover(tmp_path, monkeypatch, capsys):
    """A SECOND same-signature CUDA-IMA dead poll (prior record seeded) drives
    main() through the fresh-host failover, emitting a RUNNING-shaped JSON and
    re-pointing the sidecar at a fresh RunPod handle."""
    sidecar = tmp_path / "issue-775-handle.json"
    write_handle_sidecar(_runpod_handle_775(), sidecar)
    _seed_cuda_ima_record(sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_cuda_ima_dead_poll_base()),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    # The crashed pod is already dead -> get_pod_by_name returns None (terminate skipped).
    import runpod_api

    monkeypatch.setattr(runpod_api, "get_pod_by_name", lambda name: None, raising=False)
    # The inputs-on-HF gate is OK (no selected cells -> nothing partial).
    monkeypatch.setattr(
        "scripts.backend_poll._issue_cells_for_handle", lambda issue, handle: [], raising=False
    )

    from scripts.backend_poll import RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE

    rc = backend_poll_main(["--issue", "775", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE
    assert len(rp.launches) == 1  # exactly one fresh-host pivot
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"


def test_poller_cuda_ima_before_noport_when_transient_no_port(tmp_path, monkeypatch, capsys):
    """M2 — a SECOND CUDA-IMA dead poll that COINCIDES with a transient
    RUNNING-no-port pod still takes the CUDA-IMA path: the no-port within-K
    status=dead->running rewrite (which runs AFTER) does NOT mask it."""
    import runpod_api

    sidecar = tmp_path / "issue-775-handle.json"
    write_handle_sidecar(_runpod_handle_775(), sidecar)
    _seed_cuda_ima_record(sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_cuda_ima_dead_poll_base()),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    # A live PodInfo that IS RUNNING-but-no-port (the transient condition the
    # no-port within-K path would rewrite to running) — but the CUDA-IMA block
    # runs FIRST, so the failover fires before the no-port path is reached.
    info = type(
        "PI",
        (),
        {
            "pod_id": "pod-id-775",
            "name": "pod-775",
            "desired_status": "RUNNING",
            "ssh_host": None,
            "ssh_port": None,
            "gpu_count": 1,
            "gpu_type_id": "H100",
            "created_at": None,
        },
    )()
    monkeypatch.setattr(runpod_api, "get_pod_by_name", lambda name: info, raising=False)
    monkeypatch.setattr(runpod_api, "terminate_pod", lambda pid: None, raising=False)
    monkeypatch.setattr(
        "scripts.backend_poll._issue_cells_for_handle", lambda issue, handle: [], raising=False
    )

    from scripts.backend_poll import RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE

    rc = backend_poll_main(["--issue", "775", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # The CUDA-IMA failover fired (CUDA-IMA fresh-host phase), NOT the no-port observed /
    # within-K running rewrite (which would have masked the dead poll).
    assert out["status"] == "running"
    assert out["current_phase"] == RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE
    assert out["current_phase"] != "runpod_no_port_observed"
    assert out["current_phase"] != "runpod_noport_wedge_failover_fresh_pod"
    assert len(rp.launches) == 1


def test_poller_cuda_ima_failover_exhausted_emits_code(tmp_path, monkeypatch, capsys):
    """M1 — after the one bounded pivot (the durable lease records a CUDA-IMA
    failover), a SECOND same-signature crash on the fresh host emits
    failure_class:code (reason=cuda_ima_repeats_after_failover), AND that marker
    is NOT a transient-capacity block (so the watcher parks it at blocked)."""
    from scripts.autonomous_session_watch import _is_transient_capacity_block

    sidecar = tmp_path / "issue-775-handle.json"
    write_handle_sidecar(_runpod_handle_775(), sidecar)
    _seed_cuda_ima_record(sidecar)

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_cuda_ima_dead_poll_base()),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    # The durable lease ALREADY has a CUDA-IMA failover stamp for this run (the
    # per-run once-more bound is spent): mock the any-non-null read to True. (The
    # real cross-identity behavior — bound fires on the FRESH pod though the stamp
    # records the OLD pod — is exercised in test_runpod_wedge_detection.py's
    # test_cuda_ima_once_more_bound_blocks_on_fresh_pod_after_stamp with REAL helpers.)
    monkeypatch.setattr(
        "scripts.backend_poll._lease_has_any_runpod_cuda_ima_failover",
        lambda *a, **k: True,
        raising=False,
    )

    rc = backend_poll_main(["--issue", "775", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["failure_class"] == "code"
    assert out["reason"] == "cuda_ima_repeats_after_failover"
    assert len(rp.launches) == 0  # NO second pivot
    # The watcher's capacity-retry gate must NOT re-drive this (it is code, not
    # transient infra) — build the marker shape _is_transient_capacity_block reads.
    marker_note = f"failure_class: {out['failure_class']}\nreason: {out['reason']}"
    retriable, _reason, _ts = _is_transient_capacity_block(
        [{"kind": "epm:failure v1", "note": marker_note, "ts": "2026-06-30T00:00:00Z"}]
    )
    assert retriable is False


# ---------------------------------------------------------------------------
# #909 — repo_branch threading through BOTH lanes' handles + reconstructors
# (AC5): the failover re-execution must sync the ISSUE branch, not `main`.
# ---------------------------------------------------------------------------


def test_production_runpod_handle_repo_branch_roundtrips_reconstructor(monkeypatch):
    """The PRODUCTION-shaped RunPod handle (via the REAL launch, provision
    no-op'd) carries ``repo_branch``, and ``_runspec_from_runpod_handle``
    round-trips it into ``spec.extra`` after the sidecar serialize/deserialize."""
    from explore_persona_space.backends import runpod as RP
    from explore_persona_space.backends.base import RunSpec
    from explore_persona_space.backends.issue_dispatch import (
        deserialize_handle,
        serialize_handle,
    )
    from scripts import backend_poll as bp

    _real_run = RP.subprocess.run

    def _selective_run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any("pod_lifecycle.py" in str(c) for c in cmd):
            return None
        return _real_run(cmd, *a, **k)

    monkeypatch.setattr(RP.subprocess, "run", _selective_run, raising=False)
    handle = RP.RunPodBackend().launch(
        RunSpec(
            issue=909,
            intent="lora-7b",
            backend="runpod",
            workload_cmd="bash scripts/issue909_dispatch.sh",
            extra={"repo_branch": "issue-909"},
        )
    )
    assert handle.extra["repo_branch"] == "issue-909"
    roundtripped = deserialize_handle(serialize_handle(handle))
    spec = bp._runspec_from_runpod_handle(roundtripped, 909)
    assert spec.extra.get("repo_branch") == "issue-909"
    assert spec.workload_cmd == "bash scripts/issue909_dispatch.sh"


def test_gcp_reconstructor_threads_repo_branch():
    from scripts import backend_poll as bp

    handle = _gcp_handle({**_GCP_EXTRA_659, "repo_branch": "issue-909"})
    spec = bp._runspec_from_gcp_handle(handle, 659)
    assert spec.extra.get("repo_branch") == "issue-909"


def test_reconstructors_tolerate_legacy_handles_without_repo_branch():
    """A pre-#909 handle (no ``repo_branch`` key) still reconstructs — and an
    empty-string value (the post-#909 unset default) does NOT thread a
    falsy branch into the spec."""
    from explore_persona_space.backends.base import RunHandle
    from scripts import backend_poll as bp

    # GCP legacy: _GCP_EXTRA_659 has no repo_branch key.
    gcp_spec = bp._runspec_from_gcp_handle(_gcp_handle(), 659)
    assert gcp_spec.extra.get("repo_branch") is None
    # GCP post-#909 unset default: "" stays out of the spec extra.
    gcp_spec_empty = bp._runspec_from_gcp_handle(
        _gcp_handle({**_GCP_EXTRA_659, "repo_branch": ""}), 659
    )
    assert "repo_branch" not in gcp_spec_empty.extra
    # RunPod legacy: hand-built pre-#909 handle without the key.
    legacy = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name="pod-689",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-689.log",
        extra={
            "issue": 689,
            "intent": "lora-7b",
            "workload_cmd": "bash scripts/x.sh",
            "hydra_args": [],
            "gpus": None,
            "time_budget_hours": None,
        },
    )
    rp_spec = bp._runspec_from_runpod_handle(legacy, 689)
    assert rp_spec.extra.get("repo_branch") is None
    assert rp_spec.workload_cmd == "bash scripts/x.sh"


def test_runspec_from_runpod_handle_forwards_boot_disk_gb(monkeypatch):
    """#1118: the RunPod-handle reconstructor forwards the footprint fields
    (``boot_disk_gb`` / ``min_ram_gb``) into the rebuilt spec extra — proven
    through the PRODUCTION handle (real launch, provision no-op'd) + the
    sidecar serialize/deserialize roundtrip, so int-survives-JSON is asserted
    — and a legacy handle without the keys reconstructs byte-identically."""
    from explore_persona_space.backends import runpod as RP
    from explore_persona_space.backends.base import RunHandle, RunSpec
    from explore_persona_space.backends.issue_dispatch import (
        deserialize_handle,
        serialize_handle,
    )
    from scripts import backend_poll as bp

    _real_run = RP.subprocess.run

    def _selective_run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any("pod_lifecycle.py" in str(c) for c in cmd):
            return None
        return _real_run(cmd, *a, **k)

    monkeypatch.setattr(RP.subprocess, "run", _selective_run, raising=False)
    handle = RP.RunPodBackend().launch(
        RunSpec(
            issue=1118,
            intent="lora-7b",
            backend="runpod",
            workload_cmd="bash scripts/issue1118_dispatch.sh",
            extra={"repo_branch": "issue-1118", "boot_disk_gb": 575, "min_ram_gb": 32},
        )
    )
    assert handle.extra["boot_disk_gb"] == 575
    roundtripped = deserialize_handle(serialize_handle(handle))
    spec = bp._runspec_from_runpod_handle(roundtripped, 1118)
    assert spec.extra.get("boot_disk_gb") == 575
    assert spec.extra.get("min_ram_gb") == 32
    assert spec.extra.get("repo_branch") == "issue-1118"

    # Legacy handle without the footprint keys: the rebuilt extra is
    # byte-identical to pre-#1118 (empty here — no repo_branch either).
    legacy_rp = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name="pod-689",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-689.log",
        extra={
            "issue": 689,
            "intent": "lora-7b",
            "workload_cmd": "bash scripts/x.sh",
            "hydra_args": [],
            "gpus": None,
            "time_budget_hours": None,
        },
    )
    legacy_spec = bp._runspec_from_runpod_handle(legacy_rp, 689)
    assert legacy_spec.extra == {}


# ---------------------------------------------------------------------------
# #934: --lane-suffix sidecar resolution
# ---------------------------------------------------------------------------


def test_poll_lane_suffix_resolves_suffixed_sidecar(tmp_path, capsys, monkeypatch) -> None:
    """`backend_poll --lane-suffix cpu` with no sidecar present emits the
    missing-sidecar terminal JSON naming the SUFFIXED path — proving the
    per-lane sidecar (not the unsuffixed lane's) was probed (#934)."""
    import explore_persona_space.backends.issue_dispatch as idp

    monkeypatch.setattr(idp, "_main_checkout_root", lambda: tmp_path)
    rc = backend_poll_main(["--issue", "9349", "--lane-suffix", "cpu"])
    assert rc == 0
    line = capsys.readouterr().out.strip()
    assert line, "backend_poll must emit a JSON line, never empty stdout"
    body = json.loads(line)
    assert body["status"] == "dead"
    assert body["reason"] == "missing_handle_sidecar"
    assert "issue-9349-cpu-handle.json" in body["log_tail_excerpt"]


def test_poll_malformed_lane_suffix_fails_loud_with_json(tmp_path, capsys, monkeypatch) -> None:
    """A malformed --lane-suffix fails LOUD (the validator raises inside
    the resolver) but still emits ONE terminal JSON line — never empty
    stdout (the bg-Bash poll loop would spin forever)."""
    import explore_persona_space.backends.issue_dispatch as idp

    monkeypatch.setattr(idp, "_main_checkout_root", lambda: tmp_path)
    rc = backend_poll_main(["--issue", "9349", "--lane-suffix", "Not_Valid"])
    assert rc == 0
    line = capsys.readouterr().out.strip()
    assert line, "backend_poll must emit a JSON line, never empty stdout"
    body = json.loads(line)
    assert body["status"] == "dead"
    assert "lane_suffix" in body["log_tail_excerpt"]


# ---------------------------------------------------------------------------
# #954 — PARTIAL RunPod failover failure: pod PROVISIONED, workload start
# FAILED. The failover legs must catch the typed RunPodWorkloadStartError,
# authoritatively re-point the sidecar at the PARTIAL handle (write+readback),
# and emit a DISTINCT terminal (runpod_workload_start_failed — NOT
# no_compute_available, whose mislabel invites the watcher's capacity-retry
# re-drive while the pod bills invisibly; the #931 incident).
# ---------------------------------------------------------------------------


class _WorkloadStartFailedRunpodBackend:
    """RunPodBackend stand-in modeling the #954 PARTIAL failure: ``launch``
    provisions (records the spec) then raises the typed error CARRYING the
    partial handle — the shape ``RunPodBackend.launch`` produces when the #909
    execution leg fails after the pod exists."""

    def __init__(self) -> None:
        self.launches: list = []

    def launch(self, spec):
        from explore_persona_space.backends.runpod import RunPodWorkloadStartError

        self.launches.append(spec)
        partial = RunHandle(
            backend="runpod",
            cluster=None,
            job_id="",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/workspace",
            log_path=f"/workspace/logs/issue-{spec.issue}.log",
            extra={
                "issue": spec.issue,
                "intent": "lora-7b",
                "workload_cmd": "bash run.sh",
                "hydra_args": [],
                "workload_executed": False,
                "workload_start_error": "branch sync timed out (ssh TimeoutExpired)",
            },
        )
        raise RunPodWorkloadStartError("branch sync timed out (ssh TimeoutExpired)", handle=partial)


def test_failover_workload_start_failure_mints_runpod_sidecar_and_emits_distinct_terminal(
    tmp_path, monkeypatch, capsys
):
    """#954 AC3 (end-to-end): a dead GCP workload whose RunPod failover
    PROVISIONS a pod but fails the workload-start leg emits a TERMINAL infra
    JSON with the DISTINCT reason, re-points the sidecar at the PARTIAL RunPod
    handle (readback-proven), and carries the recovery hints (round-1 critique
    MF1 mechanized): the live-workload/pidfile check-before-re-drive AND the
    billing-until-human-stop/terminate wording."""
    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)

    rp = _WorkloadStartFailedRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "runpod_workload_start_failed"
    # Recovery hints (MF1): pidfile-check-before-re-drive + billing exposure.
    tail = out["log_tail_excerpt"]
    assert "pidfile" in tail
    assert "BILLS until a human" in tail
    assert "stop/terminate" in tail
    # Sidecar AUTHORITATIVELY re-pointed at the PARTIAL RunPod handle.
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    assert recovered.extra["workload_executed"] is False
    assert recovered.extra["workload_start_error"]
    assert len(rp.launches) == 1


def test_failover_workload_start_failure_second_poll_no_second_launch(
    tmp_path, monkeypatch, capsys
):
    """#954 AC4-i: after the partial failover, (i) a second poll tick launches
    ZERO additional pods (the sidecar now reads runpod, so the GCP failover
    predicate cannot re-fire), and (ii) a FORCED still-GCP sidecar
    short-circuits on the DURABLE LEASE specifically (the
    sidecar_persistence_failed shape naming the lease record — the sentinel was
    cleared at the end of the first tick, so only the lease can dedup)."""
    # Deterministic scripts-dir bootstrap: backend_poll's own lazy
    # ``from runpod_api import ...`` relies on scripts/ being on sys.path;
    # mirror it here so the test is test-ordering-independent.
    import sys

    import scripts.backend_poll as bp

    scripts_dir = str(Path(bp.__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import runpod_api

    sidecar = tmp_path / "issue-659-handle.json"
    gcp_handle = _gcp_handle()
    write_handle_sidecar(gcp_handle, sidecar)

    rp = _WorkloadStartFailedRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    # The partial pod does not exist on the live API (hermetic: the RunPod
    # wedge escalation probes get_pod_by_name on any runpod-handle tick).
    monkeypatch.setattr(runpod_api, "get_pod_by_name", lambda name: None, raising=False)

    # Tick 1: partial failover -> terminal + sidecar re-pointed at RunPod.
    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    assert _last_json_line(capsys)["reason"] == "runpod_workload_start_failed"
    assert len(rp.launches) == 1
    assert read_handle_sidecar(sidecar).backend == "runpod"

    # Tick 2 (the sidecar now reads runpod): the GCP failover predicate keys on
    # backend=="gcp", so it cannot re-fire — ZERO additional launches.
    rc2 = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc2 == 0
    _last_json_line(capsys)  # drain stdout; the dead runpod poll is ordinary
    assert len(rp.launches) == 1

    # Tick 3 (FORCED still-GCP sidecar — a rollback/copy): the predicate
    # re-fires but the DURABLE LEASE short-circuit binds (the rung stamped
    # gcp_failover_of in-flock on tick 1; the sentinel was cleared) — the
    # fail-loud sidecar_persistence_failed shape, still NO launch.
    write_handle_sidecar(gcp_handle, sidecar)
    rc3 = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc3 == 0
    out3 = _last_json_line(capsys)
    assert len(rp.launches) == 1  # the exactly-once bound held
    assert out3["reason"] == "sidecar_persistence_failed"
    assert "durable lease record" in out3["log_tail_excerpt"]


def test_failover_workload_start_failure_not_transient_capacity_block(
    tmp_path, monkeypatch, capsys
):
    """#954 AC5: the emitted terminal's fields parse via the watcher's REAL
    ``_is_transient_capacity_block`` to NOT-retriable (mirrors the #775 M1
    pattern) — the capacity-retry pass never auto re-drives a run whose pod
    exists and bills."""
    import scripts.autonomous_session_watch as asw

    sidecar = tmp_path / "issue-659-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    rp = _WorkloadStartFailedRunpodBackend()
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_workload_failed")),
    )
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "659", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # The note shape the orchestrator/watcher posts from this JSON (the
    # whitespace-token shape _parse_failure_fields reads).
    note = (
        f"failure_class: {out['failure_class']} reason: {out['reason']} — {out['log_tail_excerpt']}"
    )
    synthetic_marker = {"kind": "epm:failure v1", "note": note, "ts": None}
    retriable, parsed_reason, _block_ts = asw._is_transient_capacity_block([synthetic_marker])
    assert retriable is False
    assert parsed_reason == "runpod_workload_start_failed"
    assert "runpod_workload_start_failed" not in asw.TRANSIENT_CAPACITY_REASONS


def test_queue_timeout_failover_workload_start_failure_same_contract(tmp_path, monkeypatch, capsys):
    """#954 AC3 via the #783 caller (shared-core coverage): a queued-past-floor
    GCP instance whose RunPod failover partial-fails gets the SAME contract —
    teardown-first ran BEFORE the launch (by CALL ORDER, round-1 critique
    rider 3), distinct terminal, sidecar re-pointed at the partial handle."""
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    calls: list[str] = []

    class _TeardownRecorder:
        def poll(self, handle):
            return _pending_poll()

        def teardown(self, handle):
            calls.append("teardown")

    class _WSFRunpodOrdered:
        def launch(self, spec):
            calls.append("launch")
            partial = RunHandle(
                backend="runpod",
                cluster=None,
                job_id="",
                pod_name=f"pod-{spec.issue}",
                scratch_dir="/workspace",
                log_path=f"/workspace/logs/issue-{spec.issue}.log",
                extra={
                    "issue": spec.issue,
                    "workload_executed": False,
                    "workload_start_error": "ssh TimeoutExpired",
                },
            )
            raise RunPodWorkloadStartError("ssh TimeoutExpired", handle=partial)

    sidecar = tmp_path / "issue-783-handle.json"
    write_handle_sidecar(_gcp_handle_with_clock(phase="pending", ts=_time.time() - 1000), sidecar)
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: _TeardownRecorder())
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _WSFRunpodOrdered)

    rc = backend_poll_main(["--issue", "783", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "runpod_workload_start_failed"
    # Teardown-first by CALL ORDER: the queued instance was released BEFORE
    # the RunPod launch was attempted.
    assert calls == ["teardown", "launch"]
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    assert recovered.extra["workload_executed"] is False


def test_wedge_relaunch_workload_start_failure_stamps_and_repoints_sidecar(
    tmp_path, monkeypatch, capsys
):
    """#954 AC6: ``_relaunch_fresh_runpod`` (the #692/#770 wedge + #775
    CUDA-IMA shared relaunch) on the typed-with-handle error: stamp_fn called
    with the WEDGED handle (bounds a re-fired tick), sidecar re-pointed at the
    fresh partial pod, distinct terminal; PLUS the sidecar-write-OSError
    sub-case (terminal still emitted, note carries the write failure —
    fail-loud, never swallowed)."""
    import scripts.backend_poll as bp

    rp = _WorkloadStartFailedRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    wedged = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-fake-775",
        pod_name="pod-775",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-775.log",
        extra={
            "issue": 775,
            "intent": "lora-7b",
            "workload_cmd": "bash scripts/issue664_dispatch.sh --foo",
            "hydra_args": [],
        },
    )
    sidecar = tmp_path / "issue-775-handle.json"
    write_handle_sidecar(wedged, sidecar)
    stamps: list = []

    out = bp._relaunch_fresh_runpod(
        issue=775,
        handle=wedged,
        result=_poll("dead", "terminal_runpod_no_port_wedged"),
        sidecar=sidecar,
        stamp_fn=lambda issue, handle: stamps.append((issue, handle)),
    )
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "runpod_workload_start_failed"
    tail = out["log_tail_excerpt"]
    assert "pidfile" in tail
    assert "BILLS until a human" in tail
    # stamp_fn fired with the WEDGED handle (bounds a re-fired tick).
    assert stamps == [(775, wedged)]
    # Sidecar re-pointed at the FRESH partial pod (visible to the machinery).
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    assert recovered.extra["workload_executed"] is False
    assert len(rp.launches) == 1

    # Sub-case: the sidecar write ALSO fails (OSError) — the terminal is still
    # emitted with the write failure recorded, the stamp still bounds.
    stamps2: list = []
    write_handle_sidecar(wedged, sidecar)  # reset to the wedged handle

    def _raising_write(handle, path):
        raise OSError("Disk quota exceeded (EDQUOT) writing the sidecar")

    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.write_handle_sidecar",
        _raising_write,
    )
    out2 = bp._relaunch_fresh_runpod(
        issue=775,
        handle=wedged,
        result=_poll("dead", "terminal_runpod_no_port_wedged"),
        sidecar=sidecar,
        stamp_fn=lambda issue, handle: stamps2.append((issue, handle)),
    )
    assert out2["status"] == "dead"
    assert out2["reason"] == "runpod_workload_start_failed"
    assert "sidecar write ALSO failed" in out2["log_tail_excerpt"]
    assert stamps2 == [(775, wedged)]


def test_runspec_from_gcp_handle_preserves_footprint_extra():
    """#1010: the GCP-handle reconstructor forwards the footprint fields
    (boot_disk_gb / min_ram_gb) into the rebuilt spec.extra so the RunPod
    CPU-fallback feasibility gate + container-disk threading cover the ASYNC
    failover paths (#659 crash / #783 queue timeout) — pre-#1010 only
    repo_branch survived, so the gate failed OPEN there (the #958 shape).
    A legacy handle WITHOUT the keys reconstructs byte-identically."""
    from scripts import backend_poll as bp

    handle = _gcp_handle({**_GCP_EXTRA_659, "boot_disk_gb": 80, "min_ram_gb": 32})
    spec = bp._runspec_from_gcp_handle(handle, 659)
    assert spec.extra.get("boot_disk_gb") == 80
    assert spec.extra.get("min_ram_gb") == 32

    # Legacy handle (no footprint keys, no repo_branch): pre-#1010 shape == {}.
    legacy_spec = bp._runspec_from_gcp_handle(_gcp_handle(), 659)
    assert legacy_spec.extra == {}

    # Legacy handle with only repo_branch: pre-#1010 shape preserved verbatim.
    branch_spec = bp._runspec_from_gcp_handle(
        _gcp_handle({**_GCP_EXTRA_659, "repo_branch": "issue-909"}), 659
    )
    assert branch_spec.extra == {"repo_branch": "issue-909"}


# ---------------------------------------------------------------------------
# issue #1029 — GCP pre-workload boot-loop breaker (recorder + escalation +
# failover + reset). The streak rides the durable lease (tmp-isolated by the
# autouse _isolate_lease_store fixture); each VM CREATE is a distinct
# INCARNATION (job_id), and each relaunch writes a FRESH sidecar — the tests
# model exactly that #763 shape.
# ---------------------------------------------------------------------------

#: The #1029 boot-loop test extra: a GPU GCP handle carrying the threaded
#: ladder-rung label (the streak key) on top of the #659 spec-threading keys.
_GCP_EXTRA_1029 = {**_GCP_EXTRA_659, "issue": 1029, "gcp_ladder_rung": "flexstart_l4"}


def _boot_handle(
    *,
    job_id: str = "instance-boot-1",
    extra: dict | None = None,
    launched_ts: float | None = None,
) -> RunHandle:
    """A GCP RunHandle for the boot-loop tests: distinct ``job_id`` per CREATE
    (the incarnation key), optional ``gcp_launched_ts`` for the heuristic
    branch."""
    e = dict(extra if extra is not None else _GCP_EXTRA_1029)
    if launched_ts is not None:
        e["gcp_launched_ts"] = launched_ts
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id=job_id,
        pod_name="eps-issue-1029",
        scratch_dir="/workspace/eps-issue-1029",
        log_path="/workspace/logs/issue-1029.log",
        extra=e,
    )


def _poll_death(
    monkeypatch, capsys, *, sidecar: Path, handle: RunHandle, phase: str = "terminal_setup_failed"
) -> dict:
    """Write the handle sidecar, script one dead poll at ``phase``, run main()."""
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_handle_sidecar(handle, sidecar)
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", phase)),
    )
    rc = backend_poll_main(["--issue", "1029", "--handle-file", str(sidecar)])
    assert rc == 0
    return _last_json_line(capsys)


def test_gcp_first_setup_death_records_streak_but_does_NOT_fail_over(tmp_path, monkeypatch, capsys):
    """#1029 AC-2: a SINGLE pre-workload setup death takes the ordinary dead
    path unchanged (terminal JSON as today) — but the (issue, rung) streak is
    RECORDED (0 -> 1) so the route()-side skip and the Nth-death escalation can
    key on it."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    sidecar = tmp_path / "issue-1029-handle.json"
    out = _poll_death(monkeypatch, capsys, sidecar=sidecar, handle=_boot_handle())
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_setup_failed"
    assert len(rp.launches) == 0
    assert read_handle_sidecar(sidecar).backend == "gcp"  # sidecar unchanged
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 1


def test_gcp_second_consecutive_setup_death_same_rung_fails_over_to_runpod(
    tmp_path, monkeypatch, capsys
):
    """#1029 AC-1 (headline): TWO consecutive pre-workload deaths on one rung —
    DISTINCT incarnations (job_id), each on a FRESH handle + FRESH sidecar (the
    relaunch rewrites the sidecar, so the record must survive in the durable
    lease) — produce EXACTLY ONE RunPod failover with reason
    ``gcp_boot_loop_failover_runpod``."""
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    # Death #1 (create #1) — ordinary dead path, streak 1.
    out1 = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "launch1" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="instance-boot-1"),
    )
    assert out1["status"] == "dead"
    assert len(rp.launches) == 0

    # Death #2 (create #2, FRESH sidecar + FRESH instance id) — the breaker.
    sidecar2 = tmp_path / "launch2" / "issue-1029-handle.json"
    out2 = _poll_death(
        monkeypatch,
        capsys,
        sidecar=sidecar2,
        handle=_boot_handle(job_id="instance-boot-2"),
    )
    assert out2["status"] == "running"
    assert out2["current_phase"] == "gcp_boot_loop_failover_runpod"
    assert len(rp.launches) == 1  # exactly ONE failover
    assert read_handle_sidecar(sidecar2).backend == "runpod"


def test_gcp_boot_loop_fires_with_same_attempt_id_distinct_incarnations(
    tmp_path, monkeypatch, capsys
):
    """#1029 Must-Fix (the #763-shape replay): all of #763's five creates shared
    ONE attempt_id (att-20260630-141513) with DISTINCT instance ids — so the
    idempotency key MUST be the incarnation (job_id), never attempt_id alone.
    Two deaths sharing an attempt_id but with distinct job_ids reach streak 2
    and fire the failover; attempt_id-keying would freeze the streak at 1."""
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    shared = {**_GCP_EXTRA_1029, "attempt_id": "att-20260630-141513"}

    _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "launch1" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="5583329098377891015", extra=dict(shared)),
    )
    out2 = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "launch2" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="1628930989301073651", extra=dict(shared)),
    )
    assert out2["current_phase"] == "gcp_boot_loop_failover_runpod"
    assert len(rp.launches) == 1


def test_gcp_boot_loop_repeat_poll_same_incarnation_does_not_double_increment(
    tmp_path, monkeypatch, capsys
):
    """#1029 AC-1 (re-poll control): the SAME dead instance re-polled (same
    handle, same sidecar, same job_id) does NOT double-increment — the streak
    stays 1 and no failover fires."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    sidecar = tmp_path / "issue-1029-handle.json"
    handle = _boot_handle(job_id="instance-boot-1")
    for _tick in range(2):
        out = _poll_death(monkeypatch, capsys, sidecar=sidecar, handle=handle)
        assert out["status"] == "dead"
        assert out["current_phase"] == "terminal_setup_failed"
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 1
    assert len(rp.launches) == 0


def test_gcp_boot_loop_degenerate_incarnation_key_skips_record(tmp_path, monkeypatch, capsys):
    """#1029 Must-Fix (degenerate-key guard): a pre-fix handle with NO job_id,
    NO attempt_id, and NO gcp_launched_ts has a fully-degenerate incarnation
    key — the record is SKIPPED entirely (logged, fail-open) rather than keyed
    on "" (which would dedupe every pre-fix handle's deaths together)."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    extra = dict(_GCP_EXTRA_1029)
    extra.pop("attempt_id", None)
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "issue-1029-handle.json",
        handle=_boot_handle(job_id="", extra=extra),
    )
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_setup_failed"
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 0  # no record
    assert len(rp.launches) == 0


def test_gcp_boot_loop_failover_does_NOT_increment_gcp_attempts_today(
    tmp_path, monkeypatch, capsys
):
    """#1029 (mirror of the #783 test): the boot-loop failover never re-enters
    _attempt_one_gcp_rung, so the per-day GCP attempt counter is UNCHANGED
    across the full two-death escalation + RunPod failover."""
    from explore_persona_space.backends.router import Lease, LeaseStore

    store = LeaseStore()  # resolves to the tmp ~/.eps-routing (autouse fixture)
    store.write(Lease(issue=1029, spec_hash="deadbeef", attempt_id="att-1", gcp_attempts_today=3))

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "launch1" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="instance-boot-1"),
    )
    _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "launch2" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="instance-boot-2"),
    )
    assert len(rp.launches) == 1  # the failover fired
    lease_after = store.read(1029)
    assert lease_after is not None
    assert lease_after.gcp_attempts_today == 3  # counter untouched


def test_gcp_young_terminated_death_counts_via_heuristic_branch(tmp_path, monkeypatch, capsys):
    """#1029 heuristic branch: a YOUNG ``terminal_terminated`` death (the
    TERMINATED-window observation of an attribute-unreadable boot death) counts
    toward the streak via the launch-age heuristic."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "issue-1029-handle.json",
        handle=_boot_handle(launched_ts=_time.time() - 60),
        phase="terminal_terminated",
    )
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_terminated"  # single death: unchanged
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 1


@pytest.mark.parametrize("age_seconds", [1500, 100_000])
def test_gcp_old_terminated_death_does_NOT_count(tmp_path, monkeypatch, capsys, age_seconds):
    """#1029 spot-preemption protection: a ``terminal_terminated`` death whose
    launch->observation age is AT the floor (>= semantics — exactly 1500s) or
    far above it does NOT count toward the streak — a lone mid-run spot
    preemption / max-run-duration / manual stop keeps today's behavior."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "issue-1029-handle.json",
        handle=_boot_handle(launched_ts=_time.time() - age_seconds),
        phase="terminal_terminated",
    )
    assert out["status"] == "dead"
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 0


def test_gcp_terminated_death_without_launched_ts_fails_open(tmp_path, monkeypatch, capsys):
    """#1029 pre-fix-handle guard: a ``terminal_terminated`` death on a handle
    WITHOUT ``gcp_launched_ts`` (pre-#1029 sidecar) leaves the heuristic branch
    inert — no record, ordinary dead JSON."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "issue-1029-handle.json",
        handle=_boot_handle(),  # no launched_ts
        phase="terminal_terminated",
    )
    assert out["status"] == "dead"
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 0


def test_gcp_instance_not_found_young_death_counts(tmp_path, monkeypatch, capsys):
    """#1029 heuristic branch (post-DELETE observation): a YOUNG
    ``terminal_instance not found`` death — the instance record already gone at
    describe time, the COMMON observation at the 540s poll default — counts
    toward the streak."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "issue-1029-handle.json",
        handle=_boot_handle(launched_ts=_time.time() - 60),
        phase="terminal_instance not found",
    )
    assert out["status"] == "dead"
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 1


def test_gcp_workload_crash_keeps_659_reason_not_boot_loop(tmp_path, monkeypatch, capsys):
    """#1029 AC-4: a REAL workload crash (terminal_workload_failed) still takes
    the #659 path with the #659 reason — AND resets the boot streak (the
    workload started, so boot demonstrably succeeded)."""
    from explore_persona_space.backends.router import (
        gcp_boot_death_streak,
        record_gcp_boot_death,
    )

    record_gcp_boot_death(1029, "flexstart_l4", incarnation="prior-create-1")
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 1

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "issue-1029-handle.json",
        handle=_boot_handle(job_id="instance-boot-2"),
        phase="terminal_workload_failed",
    )
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_workload_failover_runpod_async"  # the #659 reason
    assert len(rp.launches) == 1
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 0  # reset: boot was fine


@pytest.mark.parametrize(
    ("status", "phase"),
    [
        ("running", "workload"),
        ("running", "relaunched_workload"),
        ("dead", "terminal_workload_failed"),
        ("done", "workload_done"),
        ("done", "workload_done_self_poweroff"),
        ("done", "relaunched_workload_done"),
        ("done", "workload_done_finalize_failed"),  # #1055 done shape
    ],
)
def test_gcp_boot_streak_resets_on_workload_phase_observation(
    tmp_path, monkeypatch, capsys, status, phase
):
    """#1029 reset (positive direction): a POSITIVE workload signal — a running
    'workload'/'relaunched_workload' phase, a workload crash (started => boot
    fine), or a #935 done shape — clears the (issue, rung) streak so
    non-consecutive deaths never accumulate."""
    from explore_persona_space.backends.router import (
        gcp_boot_death_streak,
        record_gcp_boot_death,
    )

    record_gcp_boot_death(1029, "flexstart_l4", incarnation="prior-create-1")
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 1

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)
    sidecar = tmp_path / "issue-1029-handle.json"
    write_handle_sidecar(_boot_handle(job_id="instance-boot-2"), sidecar)
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll(status, phase)),
    )
    rc = backend_poll_main(["--issue", "1029", "--handle-file", str(sidecar)])
    assert rc == 0
    assert gcp_boot_death_streak(1029, "flexstart_l4") == 0


def test_gcp_pre_workload_running_observation_does_NOT_reset_streak(tmp_path, monkeypatch, capsys):
    """#1029 Must-Fix (the negative reset control): PRE-WORKLOAD running
    observations — the mid-boot 'startup' guest phase AND the booting-no-phase
    '' — must NOT reset the streak (positive-signal design; a blocklist-style
    reset omitting 'startup' would silently defeat the breaker in exactly the
    #763 boot-window scenario). A subsequent second death still fails over."""
    from explore_persona_space.backends.router import (
        gcp_boot_death_streak,
        record_gcp_boot_death,
    )

    record_gcp_boot_death(1029, "flexstart_l4", incarnation="instance-boot-1")
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    # Relaunch #2's first polls land in the ~5.5-min boot window: running with
    # the mid-boot 'startup' phase, then the booting-no-phase '' value.
    for boot_phase in ("startup", ""):
        sidecar = tmp_path / f"boot-{boot_phase or 'nophase'}" / "issue-1029-handle.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        write_handle_sidecar(_boot_handle(job_id="instance-boot-2"), sidecar)
        monkeypatch.setattr(
            "scripts.backend_poll._resolve_backend",
            lambda name, _p=boot_phase: _PollDouble(_poll("running", _p)),
        )
        rc = backend_poll_main(["--issue", "1029", "--handle-file", str(sidecar)])
        assert rc == 0
        assert gcp_boot_death_streak(1029, "flexstart_l4") == 1, (
            f"pre-workload running phase {boot_phase!r} must NOT reset the streak"
        )

    # Relaunch #2 then dies pre-workload -> the SECOND consecutive death fires.
    out = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "launch2" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="instance-boot-2"),
    )
    assert out["current_phase"] == "gcp_boot_loop_failover_runpod"
    assert len(rp.launches) == 1


def test_gcp_boot_loop_cpu_bigmem_records_but_never_rewrites(tmp_path, monkeypatch, capsys):
    """#1029 AC-4 (CPU guard): a cpu-bigmem boot loop RECORDS the streak (the
    route()-side skip is its only breaker) but NEVER rewrites to
    terminal_boot_loop — no RunPod lane exists for it (#677), so both deaths
    print the ordinary dead JSON. The cpu-small counterpart (mapped, #747) DOES
    rewrite and fails over."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    bigmem_extra = {
        **_GCP_EXTRA_1029,
        "gpu_count": 0,
        "intent": "cpu-bigmem",
        "gcp_ladder_rung": "ondemand_cpu",
    }
    for n, job_id in enumerate(["cpu-inst-1", "cpu-inst-2"], start=1):
        out = _poll_death(
            monkeypatch,
            capsys,
            sidecar=tmp_path / f"bigmem-{n}" / "issue-1029-handle.json",
            handle=_boot_handle(job_id=job_id, extra=dict(bigmem_extra)),
        )
        assert out["status"] == "dead"
        assert out["current_phase"] == "terminal_setup_failed"  # never rewritten
        assert gcp_boot_death_streak(1029, "ondemand_cpu") == n  # still recorded
    assert len(rp.launches) == 0

    # cpu-small counterpart: mapped in RUNPOD_CPU_INSTANCE_FOR_INTENT -> the
    # second death DOES rewrite + fail over (a distinct rung key keeps the two
    # sub-cases independent).
    small_extra = {
        **_GCP_EXTRA_1029,
        "gpu_count": 0,
        "intent": "cpu-small",
        "gcp_ladder_rung": "spot_cpu",
    }
    _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "small-1" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="cpu-small-inst-1", extra=dict(small_extra)),
    )
    out2 = _poll_death(
        monkeypatch,
        capsys,
        sidecar=tmp_path / "small-2" / "issue-1029-handle.json",
        handle=_boot_handle(job_id="cpu-small-inst-2", extra=dict(small_extra)),
    )
    assert out2["status"] == "running"
    assert out2["current_phase"] == "gcp_boot_loop_failover_runpod"
    assert len(rp.launches) == 1


# ---------------------------------------------------------------------------
# issue #1055 — the finalize-failed-but-artifacts-ok classification is a
# SUCCESS shape: the async GCP->RunPod failover predicate must never match it
# ---------------------------------------------------------------------------


def test_async_failover_excludes_finalize_failed_artifacts_ok():
    """#1055 acceptance criterion 3: the done-like classification for a
    post-deliverables finalize/tail crash (status="done",
    current_phase="workload_done_finalize_failed") fails BOTH failover
    conjuncts by construction — and DEFENSIVELY, even a hypothetical
    status="dead" poll carrying this phase is excluded, because the phase is
    not in _GCP_ASYNC_FAILOVER_PHASES. No RunPod failover, no crash-fix
    routing, for a run whose deliverables are complete on HF."""
    from scripts.backend_poll import _is_gcp_async_workload_failure

    assert (
        _is_gcp_async_workload_failure(
            _gcp_handle(), _poll("done", "workload_done_finalize_failed")
        )
        is False
    )
    assert (
        _is_gcp_async_workload_failure(
            _gcp_handle(), _poll("dead", "workload_done_finalize_failed")
        )
        is False
    )


# ---------------------------------------------------------------------------
# issue #1116 — GCP FLEX_START queue-VANISH → RunPod failover.
#
# A DWS-queued FLEX_START instance can be dropped SERVER-SIDE (create DONE, no
# delete op, the instance simply disappears from instances-list — #1112 hit
# this twice in one evening), which gcp.poll maps to status="dead" /
# current_phase="terminal_instance not found". The discriminator is the
# sidecar phase clock: last_phase=="pending" means the instance never left the
# capacity queue, so the vanish is deterministic capacity evidence and the
# poller fails over to RunPod on the FIRST occurrence (reason
# gcp_queue_vanish_failover_runpod, teardown_first=False — the record is
# already gone — and no daily-attempt burn). These tests mirror the #783 block
# above (fabricated sidecar clock, scripted dead poll, backend_poll_main
# end-to-end) plus the #1029 boot-death interaction controls.
# ---------------------------------------------------------------------------

#: The #1116 queue-vanish test extra: a GPU GCP handle carrying the threaded
#: ladder-rung label on top of the #659 spec-threading keys.
_GCP_EXTRA_1116 = {**_GCP_EXTRA_659, "issue": 1116, "gcp_ladder_rung": "flexstart_a100_80"}


def _vanish_handle(
    *,
    job_id: str = "instance-vanish-1",
    clock_phase: str | None = "pending",
    launched_ts: float | None = None,
    extra: dict | None = None,
) -> RunHandle:
    """A GCP RunHandle for the queue-vanish tests: the sidecar extra carries
    the phase clock at ``clock_phase`` (None = no clock keys, the fresh-dispatch
    / wiped-sidecar shape) and optionally ``gcp_launched_ts`` so the #1029
    heuristic branch is armed on the negative controls."""
    e = dict(extra if extra is not None else _GCP_EXTRA_1116)
    if clock_phase is not None:
        e["last_phase"] = clock_phase
        e["last_phase_change_ts"] = _time.time() - 100
    if launched_ts is not None:
        e["gcp_launched_ts"] = launched_ts
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id=job_id,
        pod_name="eps-issue-1116",
        scratch_dir="/workspace/eps-issue-1116",
        log_path="/workspace/logs/issue-1116.log",
        extra=e,
    )


def _not_found_poll() -> PollResult:
    """The post-vanish poll: gcp.poll maps a describe-404 to
    _terminal_dead_poll("instance not found") -> this exact phase."""
    return _poll("dead", "terminal_instance not found")


def test_gcp_pending_vanish_fails_over_to_runpod(tmp_path, monkeypatch, capsys):
    """#1116 HEADLINE (acceptance criterion 1): a GCP dead poll at
    terminal_instance-not-found whose sidecar clock last observed "pending" is
    rewritten to terminal_queue_vanish and failed over to RunPod on the FIRST
    occurrence — the printed JSON carries
    current_phase="gcp_queue_vanish_failover_runpod", status="running", and the
    sidecar is re-pointed at the RunPod handle."""
    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_queue_vanish_failover_runpod"
    assert len(rp.launches) == 1  # exactly ONE RunPod launch, on the FIRST vanish
    assert read_handle_sidecar(sidecar).backend == "runpod"


def test_gcp_queue_vanish_failover_marker_carries_queue_vanish_reason(
    tmp_path, monkeypatch, capsys
):
    """#1116 marker trail: the epm:backend-selected marker the failover posts
    carries reason=gcp_queue_vanish_failover_runpod, and its evidence records
    the clock discriminator (last_observed_phase="pending"), the vanish source
    (async_poller_queue_vanish), and WHICH rung's queue dropped the request."""
    import scripts.backend_poll as bp

    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    captured: list[dict] = []
    monkeypatch.setattr(
        bp, "post_marker_via_task_py", lambda **kw: captured.append(kw), raising=False
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **kw: captured.append(kw),
        raising=False,
    )

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    backend_selected = [m for m in captured if m.get("marker") == "epm:backend-selected"]
    assert backend_selected, "no epm:backend-selected marker posted by the queue-vanish failover"
    body = json.loads(backend_selected[-1]["note"])
    assert body["reason"] == "gcp_queue_vanish_failover_runpod"
    evidence = body["extra"]["gcp_workload_evidence"]
    assert evidence["source"] == "async_poller_queue_vanish"
    assert evidence["last_observed_phase"] == "pending"
    assert evidence["gcp_ladder_rung"] == "flexstart_a100_80"


def test_gcp_queue_vanish_does_NOT_increment_gcp_attempts_today(tmp_path, monkeypatch, capsys):
    """#1116 (mirror of the #783/#1029 tests, acceptance criterion 2): the
    queue-vanish failover never re-enters _attempt_one_gcp_rung, so the per-day
    GCP attempt counter is UNCHANGED across the full failover."""
    from explore_persona_space.backends.router import Lease, LeaseStore

    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(), sidecar)

    store = LeaseStore()  # resolves to the tmp ~/.eps-routing (autouse fixture)
    store.write(Lease(issue=1116, spec_hash="deadbeef", attempt_id="att-1", gcp_attempts_today=3))

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    assert len(rp.launches) == 1  # the failover fired
    lease_after = store.read(1116)
    assert lease_after is not None
    assert lease_after.gcp_attempts_today == 3  # counter untouched


# ── #1116 negative controls (false-positive guards) ──────────────────────────


def test_gcp_not_found_with_workload_clock_not_vanish(tmp_path, monkeypatch, capsys):
    """#1116 negative control (acceptance criterion 3): a dead not-found poll
    whose clock last observed a WORKLOAD phase (the instance ran, then was
    deleted) is NOT a vanish — no rewrite, no failover; the #1029 heuristic
    recorder still records the young death (streak 1), ordinary dead path."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(
        _vanish_handle(clock_phase="workload", launched_ts=_time.time() - 60), sidecar
    )

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed for a workload-clock death (#1116)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_instance not found"
    assert gcp_boot_death_streak(1116, "flexstart_a100_80") == 1  # #1029 path untouched
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_not_found_with_no_clock_record_not_vanish(tmp_path, monkeypatch, capsys):
    """#1116 negative control (fail-open): a dead not-found poll on a sidecar
    with NO phase-clock record (fresh-dispatch handle, or the #1112 shape where
    the sidecar was wiped) behaves byte-identically to today — no rewrite, no
    failover, AND the #1029 boot-death record still happens (streak 1, the
    fall-back breaker for the clock-less vanish; mirrors
    test_gcp_instance_not_found_young_death_counts)."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(clock_phase=None, launched_ts=_time.time() - 60), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed on a clock-less death (#1116)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_instance not found"
    assert gcp_boot_death_streak(1116, "flexstart_a100_80") == 1  # record still happens
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_terminated_phase_with_pending_clock_not_vanish(tmp_path, monkeypatch, capsys):
    """#1116 negative control (narrow to not-found): a dead terminal_terminated
    poll — the instance record still EXISTS server-side (a preemption / manual
    stop), NOT the vanish shape — is untouched even with a pending clock; the
    #1029 heuristic path (young terminated death -> streak 1) is byte-identical."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(launched_ts=_time.time() - 60), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_poll("dead", "terminal_terminated")),
    )

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed for a terminated instance (#1116)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_terminated"
    assert gcp_boot_death_streak(1116, "flexstart_a100_80") == 1  # #1029 path untouched
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_queue_vanish_cpu_bigmem_excluded(tmp_path, monkeypatch, capsys):
    """#1116 negative control (acceptance criterion 4): a cpu-bigmem GCP handle
    (gpu_count==0, intent NOT in RUNPOD_CPU_INSTANCE_FOR_INTENT) whose queued
    instance vanished keeps its ORDINARY dead path byte-identically — the CPU
    guard gates the REWRITE itself (no terminal_queue_vanish phase, unlike the
    #783 cpu-bigmem shape), the #1029 boot-death record still happens, and
    RunPod is never constructed."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    cpu_extra = dict(_GCP_EXTRA_1116)
    cpu_extra["intent"] = "cpu-bigmem"
    cpu_extra["gpu_count"] = 0
    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(extra=cpu_extra, launched_ts=_time.time() - 60), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )

    def _boom(*_a, **_k):
        raise AssertionError("RunPod must NOT be constructed for a cpu-bigmem vanish (#1116)")

    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", _boom)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["current_phase"] == "terminal_instance not found"  # NO rewrite
    assert gcp_boot_death_streak(1116, "flexstart_a100_80") == 1  # record still happens
    assert read_handle_sidecar(sidecar).backend == "gcp"


def test_gcp_queue_vanish_second_tick_short_circuits(tmp_path, monkeypatch, capsys):
    """#1116 exactly-once bound (the shared _failover_gcp_to_runpod core): when
    the RunPod sidecar write fails (EDQUOT), the sidecar still holds the GCP
    handle so the vanish predicate re-fires on the next tick — the lease +
    sentinel short-circuit must refuse a SECOND paid RunPod launch and re-emit
    the terminal sidecar_persistence_failed infra JSON. (On a SUCCESSFUL
    failover the sidecar is re-pointed at RunPod, so a second tick polls the
    RunPod run and the predicate cannot re-fire at all.)"""
    sidecar = tmp_path / "issue-1116-handle.json"
    write_handle_sidecar(_vanish_handle(), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    real_write = write_handle_sidecar

    def _raising_write(handle, path):
        if getattr(handle, "backend", None) == "runpod":
            raise OSError("Disk quota exceeded (EDQUOT) writing the RunPod sidecar")
        return real_write(handle, path)

    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch.write_handle_sidecar",
        _raising_write,
    )

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "dead"
    assert out["failure_class"] == "infra"
    assert out["reason"] == "sidecar_persistence_failed"
    launches_after_first = len(rp.launches)
    assert launches_after_first == 1

    # SECOND tick on the unchanged GCP sidecar: the idempotency short-circuit
    # fires BEFORE any launch — zero additional RunPod launches.
    rc2 = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc2 == 0
    out2 = _last_json_line(capsys)
    assert len(rp.launches) == launches_after_first, (
        "a queue-vanish failover must fire RunPod EXACTLY ONCE — the repeat "
        "tick observing the unchanged GCP sidecar must launch nothing"
    )
    assert out2["status"] == "dead"
    assert out2["failure_class"] == "infra"
    assert out2["reason"] == "sidecar_persistence_failed"


def test_gcp_queue_vanish_does_not_record_boot_death(tmp_path, monkeypatch, capsys):
    """#1116 ordering guarantee (the §4.1(e) main() wiring): the vanish branch
    runs BEFORE the #1029 boot-loop recorder and returns, so a young not-found
    death classified as a queue vanish never poisons the (issue, rung)
    boot-death streak — a pure capacity miss is not a boot problem."""
    from explore_persona_space.backends.router import gcp_boot_death_streak

    sidecar = tmp_path / "issue-1116-handle.json"
    # Young launched_ts: WOULD count via the #1029 heuristic if it were reached.
    write_handle_sidecar(_vanish_handle(launched_ts=_time.time() - 60), sidecar)

    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: _PollDouble(_not_found_poll()),
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "1116", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["current_phase"] == "gcp_queue_vanish_failover_runpod"
    assert len(rp.launches) == 1
    assert gcp_boot_death_streak(1116, "flexstart_a100_80") == 0  # never recorded


# ---------------------------------------------------------------------------
# #1122 — exit-75 reconnect handle rewrites preserve the workload extras
#
# Incident #1090: the exit-75 same-command RERUN reconnected via
# gcp.reconnect_or_none, whose handle carried NO workload extras; the
# on_launched sidecar overwrite then left the #783 queue-timeout failover
# unable to reconstruct a RunSpec (ValueError "pre-#659 handle?"). These
# tests pin the end-to-end fix: the REAL reconnect_or_none output survives
# the sidecar round-trip into _runspec_from_gcp_handle (T4), the both-empty
# guard names the reconnect-rewrite cause (T5), and the #783 queue-timeout
# failover completes against a reconnect-written handle (T6, seeded via the
# REAL producer — never a hand-built "reconnect-shaped" dict).
# ---------------------------------------------------------------------------

_RECONNECT_WORKLOAD_CMD = "REPO_ROOT=/workspace bash scripts/issue1090_dispatch.sh --full"


def _reconnect_handle_from_real_producer(issue: int) -> RunHandle:
    """A handle produced by the REAL ``gcp.reconnect_or_none`` for a
    workload-carrying spec (the #1090 exit-75 rerun shape). PROVISIONING
    status keeps the probe to ONE gcloud list call (no guest-phase read)."""
    from explore_persona_space.backends.base import RunSpec
    from explore_persona_space.backends.gcp import (
        GcloudRunResult,
        GcpConfig,
        reconnect_or_none,
    )

    spec = RunSpec(
        issue=issue,
        intent="lora-7b",
        backend="gcp",
        gpus=1,
        time_budget_hours=4.0,
        workload_cmd=_RECONNECT_WORKLOAD_CMD,
        extra={"repo_branch": f"issue-{issue}"},
    )
    payload = json.dumps(
        [
            {
                "name": f"eps-issue-{issue}",
                "id": "instance-fake-1",
                "status": "PROVISIONING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
            }
        ]
    )
    config = GcpConfig(project="eps-test-project", gcloud_config="eps-test-config")
    handle = reconnect_or_none(
        spec=spec, config=config, runner=lambda argv: GcloudRunResult(0, payload, "")
    )
    assert handle is not None
    return handle


def test_runspec_from_gcp_handle_reconstructs_from_reconnect_handle(tmp_path):
    """T4 (#1122, end-to-end #1090 shape): a REAL reconnect_or_none handle,
    round-tripped through the sidecar (serialize_handle/deserialize_handle),
    reconstructs a RunSpec whose workload_cmd / hydra_args / repo_branch
    equal the original spec's — the exact read the queue-timeout failover
    crashed on pre-fix."""
    from scripts.backend_poll import _runspec_from_gcp_handle

    handle = _reconnect_handle_from_real_producer(1090)
    sidecar = tmp_path / "issue-1090-handle.json"
    write_handle_sidecar(handle, sidecar)
    recovered = read_handle_sidecar(sidecar)

    spec = _runspec_from_gcp_handle(recovered, issue=1090)
    assert spec.workload_cmd == _RECONNECT_WORKLOAD_CMD  # str, verbatim (MF1)
    assert spec.hydra_args == ()  # empty branch stays empty (MF2)
    assert spec.backend == "runpod"
    assert spec.extra["repo_branch"] == "issue-1090"
    assert spec.gpus == 1
    assert spec.time_budget_hours == 4.0


def test_runspec_from_gcp_handle_both_empty_raises_with_reconnect_cause():
    """T5 (#1122): a handle whose workload pair is BOTH empty (keys present
    or absent) raises a ValueError that names the reconnect-rewrite cause
    (+ #1122) instead of mis-attributing solely to a pre-#659 handle."""
    from scripts.backend_poll import _runspec_from_gcp_handle

    # Keys PRESENT but both empty — the pre-#1122 presence-check built a
    # blank RunSpec here; the value-check now refuses loudly.
    handle = _gcp_handle(
        extra={
            "issue": 1090,
            "intent": "lora-7b",
            "workload_cmd": "",
            "hydra_args": [],
            "reconnected": True,
        }
    )
    with pytest.raises(ValueError) as excinfo:
        _runspec_from_gcp_handle(handle, issue=1090)
    msg = str(excinfo.value).lower()
    assert "reconnect" in msg
    assert "#1122" in msg
    assert "pre-#659" in msg  # legacy cause still mentioned, no longer solely
    assert "refusing" in msg

    # Keys ABSENT entirely (the true pre-#659 shape) raises the same guard.
    with pytest.raises(ValueError):
        _runspec_from_gcp_handle(_gcp_handle(extra={"intent": "lora-7b"}), issue=1090)


def test_gcp_queue_timeout_failover_reconstructs_from_reconnect_handle(
    tmp_path, monkeypatch, capsys
):
    """T6 (#1122, the #1090 acceptance shape): the #783 queue-timeout failover
    COMPLETES (no ValueError) against a sidecar seeded from the REAL
    reconnect_or_none output — the queued instance is torn down, exactly one
    RunPod launch fires, and the relaunch spec carries the original
    workload_cmd + repo_branch."""
    from dataclasses import replace

    handle = _reconnect_handle_from_real_producer(1090)
    # The queue clock is POLLER-owned sidecar state (stamped on a prior tick);
    # add it on top of the producer's own extra, older than the 600s floor.
    handle = replace(
        handle,
        extra={
            **handle.extra,
            "last_phase": "pending",
            "last_phase_change_ts": _time.time() - 1000,
        },
    )
    sidecar = tmp_path / "issue-1090-handle.json"
    write_handle_sidecar(handle, sidecar)

    teardown_backend = _PollDoubleWithTeardown(_pending_poll())
    monkeypatch.setattr(
        "scripts.backend_poll._resolve_backend",
        lambda name: teardown_backend,
    )
    rp = _PassiveRunpodBackend()
    monkeypatch.setattr("explore_persona_space.backends.runpod.RunPodBackend", lambda: rp)

    rc = backend_poll_main(["--issue", "1090", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["current_phase"] == "gcp_queue_timeout_failover_runpod"
    assert len(rp.launches) == 1
    relaunch_spec = rp.launches[0]
    assert relaunch_spec.workload_cmd == _RECONNECT_WORKLOAD_CMD
    assert relaunch_spec.hydra_args == ()
    assert relaunch_spec.extra["repo_branch"] == "issue-1090"
    # Queued GCP instance torn down; sidecar re-pointed at the RunPod handle.
    assert len(teardown_backend.teardowns) == 1
    assert read_handle_sidecar(sidecar).backend == "runpod"


# ---------------------------------------------------------------------------
# #710/#1296/#1304 — scripts-dir bootstrap before lazy scripts-local imports
# ---------------------------------------------------------------------------


def test_ensure_scripts_dir_bootstrap_resolves_runpod_api_in_module_mode(monkeypatch):
    """#1296: ``_ensure_scripts_dir_on_sys_path()`` makes a bare
    ``import runpod_api`` resolve in MODULE mode (repo root on sys.path,
    scripts/ NOT), with a built-in NEGATIVE CONTROL proving the pre-fix
    ``ModuleNotFoundError`` exists once scripts/ is scrubbed. Import only —
    no live RunPod API call is ever made."""
    import importlib
    import sys

    import scripts.backend_poll as bp

    scripts_dir = str(Path(bp.__file__).resolve().parent)
    # Scrub every sys.path entry that resolves to scripts/ (cross-test-file
    # inserts included). monkeypatch.setattr replaces the LIST OBJECT and
    # restores the original at teardown, so the helper's in-test insert (into
    # the scrubbed list) never leaks either.
    scrubbed = [p for p in sys.path if str(Path(p or ".").resolve()) != scripts_dir]
    monkeypatch.setattr(sys, "path", scrubbed)
    # delitem records + restores any PRE-test runpod_api module object.
    monkeypatch.delitem(sys.modules, "runpod_api", raising=False)

    # NEGATIVE CONTROL (fail-loud claim, plan §5 kill criterion): with
    # scripts/ scrubbed and the bootstrap not yet run, the bare import raises
    # — the exact pre-fix failure mode at the three lazy sites.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("runpod_api")

    bp._ensure_scripts_dir_on_sys_path()

    mod = importlib.import_module("runpod_api")
    try:
        assert hasattr(mod, "get_pod_by_name")
        assert hasattr(mod, "terminate_pod")
    finally:
        # Drop the module object THIS test just imported so it cannot alias
        # past teardown; monkeypatch then restores the original entry (when
        # one existed) after this pop.
        sys.modules.pop("runpod_api", None)


def _inside_type_checking_block(lines: list[str], i: int) -> bool:
    """True iff ``lines[i]`` sits INSIDE an ``if TYPE_CHECKING:`` block.

    BLOCK MEMBERSHIP, not raw-window presence (#1304 review concern): walk
    upward to the NEAREST non-blank, non-comment line at STRICTLY LOWER
    indentation — the enclosing block opener. Only a literal
    ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` opener exempts; a
    runtime lazy import that merely sits within a few lines of a
    TYPE_CHECKING mention is NOT exempt (an ``else:``/``try:``/``def`` opener
    returns False).
    """
    indent_i = len(lines[i]) - len(lines[i].lstrip())
    for j in range(i - 1, -1, -1):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent_j = len(lines[j]) - len(lines[j].lstrip())
        if indent_j < indent_i:
            return stripped.startswith(("if TYPE_CHECKING", "if typing.TYPE_CHECKING"))
    return False


def test_every_lazy_scripts_local_import_is_bootstrap_guarded():
    """#1296/#1304 durability pin, WIDENED to the scripts-local module CLASS
    (renames + supersedes the #1296
    ``test_every_lazy_runpod_api_import_is_bootstrap_guarded``, which pinned
    only ``from runpod_api import`` in backend_poll.py — a fixed module name
    is exactly what let the unguarded ``failure_classifier`` site slip past
    it, #1304): every INDENTED (function-local, lazy) import of a
    scripts-local module — any stem of ``scripts/*.py``, derived DYNAMICALLY
    so a future scripts-local module is auto-covered — in
    scripts/backend_poll.py AND scripts/pod_config.py must have the
    scripts-dir bootstrap (``_ensure_scripts_dir_on_sys_path()`` or the
    inline ``sys.path.insert(0, scripts_dir)``) within the preceding ~12
    lines, so a FUTURE bare site cannot re-introduce the module-mode
    ModuleNotFoundError landmine. Scan scope is these TWO files — the ones
    with known module-mode lazy sites (the poller + pod_config's module-mode
    consumers, e.g. ``backends/runpod.py``); a future widening starts from
    them. Both import forms are matched: ``from X import ...`` (undotted
    module names only, so module-mode-safe ``from scripts.pod_config import
    ...`` consumers are excluded by construction) and
    ``import X [as Y][, Z [as W]]``. Comment lines never satisfy the guard;
    an ``if TYPE_CHECKING:`` block member is exempt by BLOCK MEMBERSHIP only
    (``_inside_type_checking_block``)."""
    import re
    import sys

    import scripts.backend_poll as bp

    scripts_dir = Path(bp.__file__).resolve().parent
    stems = {p.stem for p in scripts_dir.glob("*.py")}
    # Sanity-guard the dynamic stems set: a scripts/*.py whose stem shadows a
    # stdlib module would make this scan flag stdlib imports (verified empty
    # today; installed-package shadowing is left unchecked by design — there
    # is no cheap authoritative enumeration of installed top-level names).
    stdlib_overlap = stems & set(sys.stdlib_module_names)
    assert not stdlib_overlap, (
        f"scripts/*.py stems shadow stdlib modules {sorted(stdlib_overlap)}; "
        f"the dynamic scripts-local stems scan cannot distinguish them"
    )

    from_re = re.compile(r"\s+from ([A-Za-z_]\w*) import\b")
    import_re = re.compile(
        r"\s+import ([A-Za-z_]\w*(?:\s+as\s+\w+)?"
        r"(?:\s*,\s*[A-Za-z_]\w*(?:\s+as\s+\w+)?)*)\s*(#.*)?$"
    )

    for fname in ("backend_poll.py", "pod_config.py"):
        lines = (scripts_dir / fname).read_text(encoding="utf-8").splitlines()
        n_flagged = 0
        for i, line in enumerate(lines):
            m = from_re.match(line)
            if m:
                names = [m.group(1)]
            else:
                m2 = import_re.match(line)
                names = [seg.split()[0] for seg in m2.group(1).split(",")] if m2 else []
            if not any(n in stems for n in names):
                continue
            if _inside_type_checking_block(lines, i):
                continue  # type-checking-only import; no runtime effect
            n_flagged += 1
            window = [
                ln.strip()
                for ln in lines[max(0, i - 12) : i]
                if ln.strip() and not ln.strip().startswith("#")
            ]
            guarded = any(
                "_ensure_scripts_dir_on_sys_path()" in ln or "sys.path.insert(0, scripts_dir)" in ln
                for ln in window
            )
            assert guarded, (
                f"scripts/{fname}:{i + 1}: lazy scripts-local import ({line.strip()!r}) "
                f"without a scripts-dir bootstrap in the preceding 12 lines — call "
                f"_ensure_scripts_dir_on_sys_path() directly above the import "
                f"(#1296/#1304)"
            )
        assert n_flagged, (
            f"expected >=1 flagged lazy scripts-local import site in scripts/{fname} "
            f"(non-vacuity: the scan went blind if this fires)"
        )
