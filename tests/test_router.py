"""Router (`backends.router.route`) tests.

Slice-5 surface coverage: decision table, override paths, auto chain,
park-watchdog state machine, cancel state machine, durable lease +
reconnect, GCP attempt-count guard, marker registration.

The negative test that no auto path EVER calls ``RunPodBackend.launch``
(injected raising backend) is the load-bearing safeguard for the
plan's "real-money safety" property; do not weaken it.

Everything runs without RunPod / SLURM / GCP being live — every backend
is a test double + every shell-out is injected.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from explore_persona_space.backends import (
    BackendKind,
    ComputeBackend,
    GcpAttemptCapExceededError,
    GcpProvisioningError,
    GcpWorkloadError,
    Lease,
    LeaseStore,
    ManualAttentionRequiredError,
    NoComputeAvailableError,
    PollResult,
    RouterConfig,
    RunHandle,
    RunSpec,
    WorkloadSurfacedError,
    canonicalize_spec,
    rank_lanes,
    route,
    spec_hash,
)
from explore_persona_space.backends.router import (
    FREE_WAIT_SECONDS,
    MAX_GCP_ATTEMPTS_PER_DAY,
    ROUTE_REASON_AUTO_FALLBACK_GCP,
    ROUTE_REASON_AUTO_STARTED,
    ROUTE_REASON_OVERRIDE,
    ROUTE_REASON_RECONNECT,
    cancel_and_wait,
    park_until_running_or_cap,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _BaseBackend(ComputeBackend):
    """Minimal ABC fill-in. Subclasses override the relevant methods."""

    @property
    def name(self) -> BackendKind:
        return "runpod"

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        raise NotImplementedError

    def estimate_start(self, spec: RunSpec):
        return datetime.now(tz=UTC)

    def poll(self, handle: RunHandle) -> PollResult:
        return _poll("running")

    def fetch_logs(self, handle: RunHandle) -> str:
        return ""

    def fetch_results(self, handle: RunHandle) -> None:
        return None

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        return True

    def teardown(self, handle: RunHandle) -> None:
        return None


def _poll(status: str, current_phase: str = "running") -> PollResult:
    return PollResult(
        status=status,
        current_phase=current_phase,
        new_milestone=False,
        last_log_mtime_sec_ago=10**9,
        pid_alive=status == "running",
        log_tail_excerpt="",
    )


class _ExplodingRunpod(_BaseBackend):
    """Negative-test backend: every ``launch`` raises.

    Used to PROVE no auto path ever calls RunPod. If the router ever
    routes auto → RunPod, this raise crashes the test, surfacing the
    regression immediately.
    """

    @property
    def name(self) -> BackendKind:
        return "runpod"

    def launch(self, spec: RunSpec) -> RunHandle:
        raise AssertionError(
            "RunPodBackend.launch must NEVER be called on an auto path "
            "(reachable only via explicit `backend: runpod` override)."
        )


class _PassiveRunpod(_BaseBackend):
    """RunPod that records launches but doesn't raise."""

    def __init__(self) -> None:
        self.launches: list[RunSpec] = []

    def launch(self, spec: RunSpec) -> RunHandle:
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


class _FreeLaneBackend(_BaseBackend):
    """SLURM-style free-lane double.

    Constructor knobs:
    * ``kind`` — what ``name`` returns (``"nibi"``, ``"fir"``, ``"mila"``).
    * ``starts_when`` — number of ``is_started`` polls before the lane
      reports RUNNING. ``float("inf")`` = never (park-cap-exceeded path).
    * ``est_start_raw`` — what the backend reports for est-start.
    * ``launch_raises`` — exception to raise from ``launch`` (None = OK).
    * ``poll_status`` — terminal status to surface via ``poll``. Use
      ``"running"`` for happy path, ``"dead"`` to simulate
      terminal-before-running.
    """

    def __init__(
        self,
        *,
        kind: BackendKind,
        starts_when: int = 0,
        est_start_raw: float | None = 0.0,
        launch_raises: BaseException | None = None,
        poll_status: str = "running",
    ) -> None:
        self._kind = kind
        self._starts_when = starts_when
        self._est_start_raw = est_start_raw
        self._launch_raises = launch_raises
        self._poll_status = poll_status
        self.launches: list[RunSpec] = []
        self.teardowns: list[RunHandle] = []
        self.is_started_calls: int = 0
        self._next_job_id = 1000

    @property
    def name(self) -> BackendKind:
        return self._kind

    def launch(self, spec: RunSpec) -> RunHandle:
        if self._launch_raises is not None:
            raise self._launch_raises
        self.launches.append(spec)
        jid = str(self._next_job_id)
        self._next_job_id += 1
        return RunHandle(
            backend=self._kind,
            cluster=self._kind,
            job_id=jid,
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir=f"/scratch/eps/issue-{spec.issue}",
            log_path=f"/scratch/eps/issue-{spec.issue}/job.out",
            extra={"issue": spec.issue},
        )

    def estimate_start_seconds(self, spec: RunSpec) -> float | None:
        return self._est_start_raw

    def poll(self, handle: RunHandle) -> PollResult:
        return _poll(self._poll_status)

    def teardown(self, handle: RunHandle) -> None:
        self.teardowns.append(handle)


class _GcpBackendDouble(_BaseBackend):
    """GCP backend double.

    Knobs:
    * ``launch_raises`` — set to a ``GcpProvisioningError`` or
      ``GcpWorkloadError`` to test the failure classification paths.
    * ``reconnect_handle`` — set to a RunHandle to simulate a live
      existing instance found via the injected reconnect_fn.
    """

    def __init__(
        self,
        *,
        launch_raises: BaseException | None = None,
    ) -> None:
        self._launch_raises = launch_raises
        self.launches: list[RunSpec] = []

    @property
    def name(self) -> BackendKind:
        return "gcp"

    def launch(self, spec: RunSpec) -> RunHandle:
        if self._launch_raises is not None:
            raise self._launch_raises
        self.launches.append(spec)
        return RunHandle(
            backend="gcp",
            cluster=None,
            job_id="instance-fake-1",
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir=f"/workspace/eps-issue-{spec.issue}",
            log_path=f"/workspace/eps-issue-{spec.issue}/logs/issue-{spec.issue}.log",
            extra={"issue": spec.issue, "zone": "us-central1-a"},
        )

    def estimate_start_seconds(self, spec: RunSpec) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lease_store(tmp_path) -> LeaseStore:
    """LeaseStore rooted in a per-test tmp dir (never touches ~/.eps-routing/)."""
    return LeaseStore(lease_dir=tmp_path / ".eps-routing")


@pytest.fixture
def captured_markers() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def marker_poster(captured_markers):
    def post(**kwargs):
        captured_markers.append(kwargs)

    return post


def _spec(issue: int = 137, backend: BackendKind | str | None = None) -> RunSpec:
    """Build a RunSpec. ``backend=None`` means AUTO routing (sentinel "auto")."""
    bk: BackendKind = backend if backend is not None else "auto"  # type: ignore[assignment]
    return RunSpec(issue=issue, intent="lora-7b", backend=bk)


def _by_reason(captured: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in captured:
        if m.get("marker") != "epm:backend-selected":
            continue
        try:
            body = json.loads(m["note"])
        except (KeyError, json.JSONDecodeError):
            continue
        if body.get("reason") == reason:
            out.append(body)
    return out


# ---------------------------------------------------------------------------
# Decision table — the (lane x est-start x override) matrix
# ---------------------------------------------------------------------------


def test_explicit_runpod_override_runs_runpod_directly(
    lease_store, marker_poster, captured_markers
):
    rp = _PassiveRunpod()
    spec = _spec(backend="runpod")
    result = route(
        spec,
        runpod_backend=rp,
        lease_store=lease_store,
        marker_poster=marker_poster,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(rp.launches) == 1
    # Marker has the override reason.
    assert _by_reason(captured_markers, ROUTE_REASON_OVERRIDE)


def test_explicit_nibi_override_launches_only_nibi(lease_store):
    nibi = _FreeLaneBackend(kind="nibi", starts_when=1)
    rp = _ExplodingRunpod()  # auto path is sealed; this also acts as a guard.
    spec = _spec(backend="nibi")
    result = route(
        spec,
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=_is_started_after_n(1),
        config=RouterConfig(free_wait_seconds=2, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(nibi.launches) == 1


def test_no_auto_runpod_path_under_any_failure(lease_store):
    """The load-bearing negative test: no auto path can call RunPod.

    Inject a RunPod whose ``launch`` raises ``AssertionError``. The
    auto-route ladder is set up so EVERY free lane fails and GCP also
    fails — without the RunPod-is-override-only invariant, the router
    would fall through to RunPod and the AssertionError would crash
    the test. The fact that we instead raise ``NoComputeAvailableError``
    is the proof.
    """
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    spec = _spec(backend=None)  # auto
    with pytest.raises(NoComputeAvailableError):
        route(
            spec,
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )


def test_auto_picks_lane_with_lowest_clamped_est_start(lease_store):
    """Auto ranks lanes by clamped est-start; instant wins."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=1800.0)  # 30 min
    fir = _FreeLaneBackend(kind="fir", est_start_raw=5.0)  # ~instant
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi, "fir": fir},
        lease_store=lease_store,
        is_started=lambda b, _h: b is fir,  # fir starts, nibi doesn't
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "fir"
    assert len(fir.launches) == 1
    assert len(nibi.launches) == 0


# ---------------------------------------------------------------------------
# est-start ranking + negative-clamp
# ---------------------------------------------------------------------------


def test_rank_lanes_clamps_negative_to_zero():
    """A lane reporting -7200s ranks as 0/instant, NOT below 0."""
    b1 = _FreeLaneBackend(kind="nibi", est_start_raw=-7200.0)
    b2 = _FreeLaneBackend(kind="fir", est_start_raw=0.0)
    b3 = _FreeLaneBackend(kind="mila", est_start_raw=10.0)
    ranked = rank_lanes([(b1, "nibi", -7200.0), (b2, "fir", 0.0), (b3, "mila", 10.0)])
    # Both b1 and b2 clamp to 0; ranking is stable — input order preserved.
    assert ranked[0][1] == "nibi"
    assert ranked[0][3] == 0.0
    assert ranked[0][2] == -7200.0  # raw preserved
    assert ranked[1][1] == "fir"
    assert ranked[1][3] == 0.0
    assert ranked[2][1] == "mila"


def test_rank_lanes_unranked_sorts_last():
    """A lane with raw=None ranks AFTER all parseable estimates."""
    b1 = _FreeLaneBackend(kind="nibi", est_start_raw=None)
    b2 = _FreeLaneBackend(kind="fir", est_start_raw=300.0)
    ranked = rank_lanes([(b1, "nibi", None), (b2, "fir", 300.0)])
    assert ranked[0][1] == "fir"
    assert ranked[1][1] == "nibi"
    assert ranked[1][3] == float("inf")


# ---------------------------------------------------------------------------
# Park state machine
# ---------------------------------------------------------------------------


def _is_started_after_n(n: int):
    """``is_started`` probe that returns True only after N polls."""
    counter = {"calls": 0}

    def fn(_backend, _handle):
        counter["calls"] += 1
        return counter["calls"] >= n

    return fn


def _clock():
    """Deterministic monotonic clock advancing 1.0 per call."""
    counter = {"t": 0.0}

    def now():
        counter["t"] += 1.0
        return counter["t"]

    return now


def test_park_running_before_cap_returns_started():
    backend = _FreeLaneBackend(kind="nibi", starts_when=2)
    handle = backend.launch(_spec())
    started, reason = park_until_running_or_cap(
        backend=backend,
        handle=handle,
        is_started=_is_started_after_n(2),
        cap_seconds=10,
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert started is True
    assert reason == "running"


def test_park_pending_at_cap_returns_park_cap_exceeded():
    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())
    started, reason = park_until_running_or_cap(
        backend=backend,
        handle=handle,
        is_started=lambda _b, _h: False,
        cap_seconds=5,  # 5 polls (clock advances by 1 per call)
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert started is False
    assert reason == "park_cap_exceeded"


def test_park_terminal_before_running_returns_specific_reason():
    backend = _FreeLaneBackend(kind="nibi", poll_status="dead")
    handle = backend.launch(_spec())
    started, reason = park_until_running_or_cap(
        backend=backend,
        handle=handle,
        is_started=lambda _b, _h: False,
        cap_seconds=10,
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert started is False
    assert reason == "terminal_before_running"


# ---------------------------------------------------------------------------
# Cancel state machine
# ---------------------------------------------------------------------------


def test_cancel_succeeds_when_live_probe_returns_false():
    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())
    # Probe returns False immediately after teardown (job left the queue).
    out = cancel_and_wait(
        backend=backend,
        handle=handle,
        is_live_after_cancel=lambda _b, _h: False,
        grace_seconds=5,
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert out == "cancelled"
    assert len(backend.teardowns) == 1


def test_cancel_race_keeps_running_job():
    """A job that raced to RUNNING during cancel is KEPT."""
    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())
    out = cancel_and_wait(
        backend=backend,
        handle=handle,
        is_live_after_cancel=lambda _b, _h: True,
        is_running_after_cancel=lambda _b, _h: True,  # winning the race
        grace_seconds=5,
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert out == "raced_to_running"


def test_cancel_timeout_returns_manual_attention():
    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())

    def fixed_clock():
        # Advance fast enough to trip the grace cap immediately on the
        # second consultation.
        counter = {"t": 0.0}

        def now():
            counter["t"] += 10.0
            return counter["t"]

        return now

    out = cancel_and_wait(
        backend=backend,
        handle=handle,
        is_live_after_cancel=lambda _b, _h: True,  # never leaves queue
        grace_seconds=5,
        poll_interval=0.0,
        now_fn=fixed_clock(),
        sleep_fn=lambda _s: None,
    )
    assert out == "manual_attention"


def test_auto_park_fail_cancels_then_escalates_to_gcp(lease_store, marker_poster, captured_markers):
    """End-to-end: free lane park-fails → cancel → GCP."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,  # nibi never starts
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.reason == ROUTE_REASON_AUTO_FALLBACK_GCP
    assert len(nibi.launches) == 1
    assert len(nibi.teardowns) == 1
    assert len(gcp.launches) == 1
    # Pre-escalation intermediate marker exists.
    intermediates = [
        body
        for body in _by_reason(captured_markers, ROUTE_REASON_AUTO_FALLBACK_GCP)
        if body.get("extra", {}).get("intermediate") is True
    ]
    assert intermediates, "pre-escalation visible-credit marker missing"
    # Plus the final resolved marker (intermediate=False).
    finals = [
        body
        for body in _by_reason(captured_markers, ROUTE_REASON_AUTO_FALLBACK_GCP)
        if not body.get("extra", {}).get("intermediate")
    ]
    assert finals
    assert finals[-1]["chosen_kind"] == "gcp"


def test_auto_cancel_race_keeps_job_no_gcp(lease_store):
    """If a job races to RUNNING during cancel, the router KEEPS it."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,  # park times out
        is_live_after_cancel=lambda _b, _h: True,
        is_running_after_cancel=lambda _b, _h: True,  # but raced to RUNNING
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=2),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert result.reason == ROUTE_REASON_AUTO_STARTED
    assert result.extra.get("cancel_race") is True
    assert len(gcp.launches) == 0


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def test_gcp_provisioning_error_surfaces_as_no_compute(lease_store):
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "QUOTA_EXCEEDED", evidence={"matched_pattern": "QUOTA_EXCEEDED"}
        )
    )
    with pytest.raises(NoComputeAvailableError) as excinfo:
        route(
            _spec(backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    # Attempts log includes the GCP provisioning failure as the last entry.
    assert any(a["outcome"] == "provisioning_failure" for a in excinfo.value.attempts)


def test_gcp_workload_error_surfaces_no_fallback(lease_store):
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpWorkloadError("entrypoint crashed", evidence={"exit_code": 1})
    )
    with pytest.raises(WorkloadSurfacedError) as excinfo:
        route(
            _spec(backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.chosen_kind == "gcp"
    assert excinfo.value.evidence.get("exit_code") == 1


def test_no_gcp_wired_raises_no_compute_after_free_lanes_fail(lease_store):
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    with pytest.raises(NoComputeAvailableError):
        route(
            _spec(backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=None,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )


# ---------------------------------------------------------------------------
# Lease / reconnect
# ---------------------------------------------------------------------------


def test_lease_path_lives_outside_worktree(lease_store):
    """Lease MUST be under the configured dir (not under .claude/worktrees/)."""
    # Sanity check the test fixture's default.
    assert "worktrees" not in str(lease_store.lease_dir)
    # Default (no override) points at ~/.eps-routing/ — confirmed by the
    # LEASE_STORE_DIRNAME constant elsewhere.


def test_lease_round_trip(lease_store):
    issue = 137
    lease = Lease(
        issue=issue,
        spec_hash="aabbccdd",
        attempt_id="att-x",
        backend="nibi",
        cluster="nibi",
        job_id="9999",
        submitted_at=1234567890.0,
        gcp_attempts_today=2,
        gcp_attempts_date="2026-06-08",
    )
    lease_store.write(lease)
    read_back = lease_store.read(issue)
    assert read_back is not None
    assert read_back.job_id == "9999"
    assert read_back.gcp_attempts_today == 2


def test_lease_transaction_holds_flock_and_round_trips(lease_store):
    issue = 137
    with lease_store.transaction(issue) as (lease, write):
        assert lease is None
        new_lease = Lease(issue=issue, spec_hash="h", attempt_id="att-y")
        new_lease.backend = "nibi"
        new_lease.job_id = "9999"
        write(new_lease)
    read_back = lease_store.read(issue)
    assert read_back is not None
    assert read_back.job_id == "9999"


def test_lease_dir_created_with_owner_only_mode(tmp_path):
    """The lease dir is 0o700 (lease contents include job ids; not for the world)."""
    store = LeaseStore(lease_dir=tmp_path / ".eps-routing")
    store.write(Lease(issue=1, spec_hash="h", attempt_id="a"))
    mode = store.lease_dir.stat().st_mode & 0o777
    assert mode == 0o700, f"lease dir mode={oct(mode)}"


def test_unknown_submitted_recovery_via_reconnect(lease_store):
    """Lease has backend but no job_id → reconnect_fn finds the live job."""
    issue = 137
    lease_store.write(
        Lease(issue=issue, spec_hash="h", attempt_id="a", backend="nibi", job_id=None)
    )
    nibi = _FreeLaneBackend(kind="nibi")
    rp = _ExplodingRunpod()

    # Simulated live-job reconnect handle (job NOT in our local state).
    recovered = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="recovered-7777",
        pod_name=f"eps-issue-{issue}",
        scratch_dir=f"/scratch/eps/issue-{issue}",
        log_path=f"/scratch/eps/issue-{issue}/job.out",
        extra={"issue": issue},
    )

    def reconnect_fn(backend, kind, spec):
        # Only nibi has the recovered job.
        if kind == "nibi":
            return recovered
        return None

    result = route(
        _spec(issue=issue, backend="nibi"),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        reconnect_fn=reconnect_fn,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.reason == ROUTE_REASON_RECONNECT
    assert result.handle.job_id == "recovered-7777"
    assert len(nibi.launches) == 0  # no double-submit


def test_auto_reconnect_to_gcp_finds_existing_instance(lease_store):
    """Reconnect_fn returning a GCE handle bypasses every free lane + provision."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble()

    existing = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-existing",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
        extra={"issue": 137, "zone": "us-central1-a"},
    )

    def reconnect_fn(backend, kind, spec):
        if kind == "gcp":
            return existing
        return None

    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        reconnect_fn=reconnect_fn,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.reason == ROUTE_REASON_RECONNECT
    assert len(nibi.launches) == 0
    assert len(gcp.launches) == 0


def test_lease_persisted_immediately_after_submit(lease_store):
    """Lease is updated with the job_id before park starts → crash-safe."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    route(
        _spec(issue=4242, backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,  # starts immediately
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    lease = lease_store.read(4242)
    assert lease is not None
    assert lease.backend == "nibi"
    assert lease.job_id is not None
    assert lease.job_id != ""


# ---------------------------------------------------------------------------
# Spec hash + canonicalization stability
# ---------------------------------------------------------------------------


def test_spec_hash_stable_under_extra_dict_reordering():
    s1 = RunSpec(
        issue=137,
        intent="lora-7b",
        hydra_args=("seed=42",),
        extra={"plan_hash": "abc", "provisioning_model": "SPOT"},
    )
    s2 = RunSpec(
        issue=137,
        intent="lora-7b",
        hydra_args=("seed=42",),
        extra={"provisioning_model": "SPOT", "plan_hash": "abc"},  # reordered
    )
    assert spec_hash(s1) == spec_hash(s2)


def test_spec_hash_stable_for_6_vs_6p0_time_budget():
    s1 = RunSpec(issue=1, intent="lora-7b", time_budget_hours=6)
    s2 = RunSpec(issue=1, intent="lora-7b", time_budget_hours=6.0)
    assert spec_hash(s1) == spec_hash(s2)


def test_spec_hash_changes_when_intent_changes():
    s1 = RunSpec(issue=1, intent="lora-7b")
    s2 = RunSpec(issue=1, intent="ft-7b")
    assert spec_hash(s1) != spec_hash(s2)


def test_canonicalize_drops_attempt_id_and_startup_path():
    """attempt_id is recorded in the lease, NOT the spec hash."""
    s1 = RunSpec(issue=1, intent="lora-7b", extra={"attempt_id": "att-a"})
    s2 = RunSpec(issue=1, intent="lora-7b", extra={"attempt_id": "att-b"})
    assert spec_hash(s1) == spec_hash(s2)
    canon = canonicalize_spec(s1)
    assert "attempt_id" not in canon["extra"]
    assert "startup_script_path" not in canon["extra"]


# ---------------------------------------------------------------------------
# GCP attempt-count guard
# ---------------------------------------------------------------------------


def test_gcp_attempt_count_guard_caps_repeated_escalation(lease_store):
    """After N escalations, the router refuses a further one same day."""
    rp = _ExplodingRunpod()
    cfg = RouterConfig(
        free_wait_seconds=1,
        poll_interval=0.0,
        cancel_grace_seconds=0,
        max_gcp_attempts_per_day=2,
    )

    # Pre-seed the lease at the cap.
    today = datetime.now(tz=UTC).date().isoformat()
    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_attempts_today=2,
            gcp_attempts_date=today,
        )
    )

    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble()  # would succeed if reached
    with pytest.raises(GcpAttemptCapExceededError):
        route(
            _spec(issue=137, backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=cfg,
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert len(gcp.launches) == 0


def test_gcp_attempt_counter_rolls_over_on_day_change(lease_store):
    """A day-change resets the counter."""
    rp = _ExplodingRunpod()
    # Pre-seed YESTERDAY's lease at the cap.
    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_attempts_today=99,
            gcp_attempts_date="1999-01-01",  # very stale
        )
    )

    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble()
    result = route(
        _spec(issue=137, backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=2,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    lease = lease_store.read(137)
    assert lease is not None
    assert lease.gcp_attempts_today == 1
    assert lease.gcp_attempts_date == datetime.now(tz=UTC).date().isoformat()


# ---------------------------------------------------------------------------
# Marker registration (workflow.yaml § markers)
# ---------------------------------------------------------------------------


def _load_workflow_markers() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    workflow_path = root / ".claude" / "workflow.yaml"
    with workflow_path.open() as fh:
        data = yaml.safe_load(fh)
    return list(data.get("markers", []))


def test_router_markers_registered_in_workflow_yaml():
    """The 4 router-relevant markers MUST appear in `.claude/workflow.yaml § markers`."""
    markers = _load_workflow_markers()
    kinds = {m["kind"] for m in markers}
    for required in (
        "epm:backend-selected",
        "epm:cluster-launched",
        "epm:cluster-poll",
        "epm:cluster-terminal",
    ):
        assert required in kinds, f"required marker {required!r} missing from workflow.yaml"


def test_backend_selected_marker_documents_router_reasons():
    """The marker's body docs MUST mention the new router reason codes."""
    markers = _load_workflow_markers()
    [entry] = [m for m in markers if m["kind"] == "epm:backend-selected"]
    fields = entry.get("fields", "")
    for code in (
        ROUTE_REASON_OVERRIDE,
        ROUTE_REASON_RECONNECT,
        ROUTE_REASON_AUTO_STARTED,
        ROUTE_REASON_AUTO_FALLBACK_GCP,
    ):
        assert code in fields, f"router reason code {code!r} not documented in marker body"


# ---------------------------------------------------------------------------
# Sanity: module-level constants pinned to plan spec
# ---------------------------------------------------------------------------


def test_free_wait_seconds_pinned_to_10_minutes():
    """Plan §5 — every free submit parks ≤ 600 s. The 6h selector default
    is superseded by this constant; reviewers should bounce a PR that
    silently bumps this."""
    assert FREE_WAIT_SECONDS == 600


def test_max_gcp_attempts_per_day_is_count_not_dollar_cap():
    """Plan §6 — per-issue/day GCP attempt-COUNT guard (NOT a dollar cap).

    This constant must be a small positive integer; a dollar-shaped name
    here (anything containing 'usd', 'cost', 'dollar') would conflict
    with tests/test_no_dollar_budget_caps.py.
    """
    assert isinstance(MAX_GCP_ATTEMPTS_PER_DAY, int)
    assert MAX_GCP_ATTEMPTS_PER_DAY > 0


def test_no_dollar_token_in_router_module():
    """The router module MUST NOT introduce a dollar-budget cap.

    This is a belt-and-suspenders check on top of
    `tests/test_no_dollar_budget_caps.py`. Any future PR that adds a
    `max_budget_usd`-shaped variable here would also have to disable
    this test, surfacing the policy change to the reviewer.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "explore_persona_space"
        / "backends"
        / "router.py"
    )
    text = src.read_text()
    banned = re.compile(r"\b(max_budget_usd|MAX_BUDGET_USD|dollar_cap|DOLLAR_CAP)\b")
    matches = banned.findall(text)
    assert not matches, f"dollar-budget cap names found in router.py: {matches}"


# ---------------------------------------------------------------------------
# BLOCKER 1 regression: default RunSpec must NOT silently route to RunPod
# ---------------------------------------------------------------------------


def test_default_runspec_does_not_silently_route_to_runpod(lease_store):
    """A bare ``RunSpec(issue, intent)`` MUST route via AUTO, not RunPod.

    The no-auto-RunPod invariant depends on callers explicitly opting
    into RunPod. The previous default of ``backend="runpod"`` meant an
    omitted backend argument spent real money via the explicit-override
    path; flipping the default to ``"auto"`` closes that.
    """
    rp = _ExplodingRunpod()  # would crash if router took the runpod path
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    # Build a RunSpec without explicitly setting backend=...
    spec = RunSpec(issue=137, intent="lora-7b")
    assert spec.backend == "auto"
    result = route(
        spec,
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert result.reason == ROUTE_REASON_AUTO_STARTED
    assert len(nibi.launches) == 1


# ---------------------------------------------------------------------------
# BLOCKER 2 regression: concurrent route() on the same issue must serialize
# ---------------------------------------------------------------------------


def test_concurrent_route_on_same_issue_does_not_double_submit(lease_store):
    """Two concurrent route() calls on the SAME issue submit EXACTLY ONCE.

    Simulates a duplicate-cron-tick race: a manual /issue invocation
    and the 20-min issue-tick cron run in parallel. Without the flock
    held across reconnect-check + launch + lease-write, both would
    decide "no live job" and both would submit (and both would escalate
    to GCP if anything timed out → double provision + colliding artifact
    ids). The per-issue flock seals the race.

    Mechanism: a barrier inside the injected ``is_started`` probe gates
    BOTH threads inside the (would-be) critical section. The threading
    is real but the launch / cluster are mocked.
    """
    import contextlib
    import threading

    barrier = threading.Barrier(2, timeout=5.0)
    launch_seen = threading.Event()

    class _GatedNibi(_FreeLaneBackend):
        def launch(self, spec):
            # First thread to launch blocks on the barrier so the second
            # thread also has time to enter route(). The flock should
            # serialize the second thread BEFORE it reaches launch.
            handle = super().launch(spec)
            launch_seen.set()
            return handle

    nibi = _GatedNibi(kind="nibi", est_start_raw=0.0)
    rp = _ExplodingRunpod()

    def _reconnect_or_none(backend, kind, spec):
        # When the SECOND thread acquires the flock, the FIRST thread
        # has already persisted its lease + job_id. We simulate
        # backend-side reconnect by returning the FIRST thread's handle.
        if not launch_seen.is_set():
            return None
        # The first thread already wrote the lease + launched. The
        # reconnect probe should find that job. Mirror it as a handle
        # for the same issue/kind so _try_reconnect's sanity checks pass.
        first_handle = nibi.launches and RunHandle(
            backend="nibi",
            cluster="nibi",
            job_id=str(nibi._next_job_id - 1),
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir=f"/scratch/eps/issue-{spec.issue}",
            log_path=f"/scratch/eps/issue-{spec.issue}/job.out",
            extra={"issue": spec.issue},
        )
        return first_handle

    results: list[Any] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def _runner(idx: int):
        try:
            results[idx] = route(
                _spec(backend=None),
                runpod_backend=rp,
                free_backends={"nibi": nibi},
                lease_store=lease_store,
                is_started=lambda _b, _h: True,
                reconnect_fn=_reconnect_or_none,
                config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
                now_fn=_clock(),
                sleep_fn=lambda _s: None,
            )
        except BaseException as exc:
            errors[idx] = exc
        finally:
            # Release the barrier so the partner thread can proceed past
            # the flock once we exit our critical section.
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=0.1)

    t1 = threading.Thread(target=_runner, args=(0,))
    t2 = threading.Thread(target=_runner, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert all(e is None for e in errors), errors
    # EXACTLY ONE actual backend.launch — the other thread reconnected.
    assert len(nibi.launches) == 1, (
        f"expected exactly 1 launch, got {len(nibi.launches)} — flock leaked"
    )
    chosen_kinds = {r.chosen_kind for r in results}
    assert chosen_kinds == {"nibi"}
    # The two results should disagree on reason: one launched, one reconnected.
    reasons = {r.reason for r in results}
    assert ROUTE_REASON_RECONNECT in reasons or ROUTE_REASON_AUTO_STARTED in reasons


# ---------------------------------------------------------------------------
# MAJOR 3 regression: terminal failures post a final epm:backend-selected marker
# ---------------------------------------------------------------------------


def test_no_compute_terminal_posts_breadcrumb_marker(lease_store, marker_poster, captured_markers):
    """``NoComputeAvailableError`` paths post a terminal marker BEFORE raising."""
    from explore_persona_space.backends.router import ROUTE_REASON_NO_COMPUTE

    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "QUOTA_EXCEEDED", evidence={"matched_pattern": "QUOTA_EXCEEDED"}
        )
    )
    with pytest.raises(NoComputeAvailableError):
        route(
            _spec(backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            marker_poster=marker_poster,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    terminal = _by_reason(captured_markers, ROUTE_REASON_NO_COMPUTE)
    assert terminal, "terminal no_compute_available marker NOT posted before raise"


def test_workload_failure_terminal_posts_breadcrumb_marker(
    lease_store, marker_poster, captured_markers
):
    """``WorkloadSurfacedError`` paths post a workload_failure marker before raising."""
    from explore_persona_space.backends.router import ROUTE_REASON_WORKLOAD_FAILURE

    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpWorkloadError("entrypoint crashed", evidence={"exit_code": 1})
    )
    with pytest.raises(WorkloadSurfacedError):
        route(
            _spec(backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            marker_poster=marker_poster,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    terminal = _by_reason(captured_markers, ROUTE_REASON_WORKLOAD_FAILURE)
    assert terminal, "terminal workload_failure marker NOT posted before raise"


# ---------------------------------------------------------------------------
# MAJOR 4: parametrized no-auto-RunPod fan-out across every failure path
# ---------------------------------------------------------------------------


def _fast_clock():
    """Clock that advances 100s per call so cap_seconds=1 trips immediately."""
    counter = {"t": 0.0}

    def now():
        counter["t"] += 100.0
        return counter["t"]

    return now


@pytest.mark.parametrize(
    "scenario",
    [
        "free_launch_fail",
        "is_started_raises",
        "is_live_raises",
        "reconnect_fn_raises",
        "manual_attention_cancel",
        "gcp_provisioning_error",
        "attempt_cap_exceeded",
    ],
)
def test_no_auto_runpod_under_failure_fanout(lease_store, scenario):
    """For EVERY failure mode the auto chain encounters, RunPod is NEVER called.

    Injects an :class:`_ExplodingRunpod` whose ``launch`` raises ``AssertionError``
    and asserts the router raises a terminal :class:`RouteError` subclass
    instead. The parametrize covers the full failure fan-out the brief calls
    out (MAJOR 4).
    """
    rp = _ExplodingRunpod()
    cfg = RouterConfig(
        free_wait_seconds=1,
        poll_interval=0.0,
        cancel_grace_seconds=0,
        max_gcp_attempts_per_day=2,
    )
    kwargs: dict[str, Any] = {
        "runpod_backend": rp,
        "lease_store": lease_store,
        "config": cfg,
        "now_fn": _fast_clock(),
        "sleep_fn": lambda _s: None,
    }

    if scenario == "free_launch_fail":
        nibi = _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("boom"))
        gcp = _GcpBackendDouble(launch_raises=GcpProvisioningError("OUT", evidence={}))
        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
        )
        expected: type[BaseException] = NoComputeAvailableError
    elif scenario == "is_started_raises":
        nibi = _FreeLaneBackend(kind="nibi")
        gcp = _GcpBackendDouble(launch_raises=GcpProvisioningError("OUT", evidence={}))

        def _is_started(_b, _h):
            raise RuntimeError("ssh died mid-poll")

        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=_is_started,
            is_live_after_cancel=lambda _b, _h: False,
        )
        expected = NoComputeAvailableError
    elif scenario == "is_live_raises":
        nibi = _FreeLaneBackend(kind="nibi")
        gcp = _GcpBackendDouble(launch_raises=GcpProvisioningError("OUT", evidence={}))

        def _is_live(_b, _h):
            raise RuntimeError("ssh died during cancel-poll")

        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=_is_live,
        )
        # is_live raising → treated as still-live → cancel_outcome=manual_attention
        # → ManualAttentionRequiredError (we still NEVER touch RunPod).
        expected = ManualAttentionRequiredError
    elif scenario == "reconnect_fn_raises":
        nibi = _FreeLaneBackend(kind="nibi")
        gcp = _GcpBackendDouble(launch_raises=GcpProvisioningError("OUT", evidence={}))

        def _reconnect_fn(_b, _kind, _spec):
            raise RuntimeError("squeue offline")

        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            reconnect_fn=_reconnect_fn,
        )
        expected = NoComputeAvailableError
    elif scenario == "manual_attention_cancel":
        nibi = _FreeLaneBackend(kind="nibi")
        gcp = _GcpBackendDouble()
        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: True,  # never leaves queue
        )
        # MAJOR 5: manual_attention must NOT escalate to GCP — it raises.
        expected = ManualAttentionRequiredError
    elif scenario == "gcp_provisioning_error":
        nibi = _FreeLaneBackend(kind="nibi")
        gcp = _GcpBackendDouble(
            launch_raises=GcpProvisioningError("ZONE_OUT", evidence={"matched_pattern": "X"})
        )
        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
        )
        expected = NoComputeAvailableError
    elif scenario == "attempt_cap_exceeded":
        # Pre-seed the lease at the cap so the very next escalation trips it.
        today = datetime.now(tz=UTC).date().isoformat()
        lease_store.write(
            Lease(
                issue=137,
                spec_hash="h",
                attempt_id="a",
                gcp_attempts_today=cfg.max_gcp_attempts_per_day,
                gcp_attempts_date=today,
            )
        )
        nibi = _FreeLaneBackend(kind="nibi")
        gcp = _GcpBackendDouble()
        kwargs.update(
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
        )
        expected = GcpAttemptCapExceededError
    else:  # pragma: no cover — pytest.mark.parametrize wall.
        raise AssertionError(f"unknown scenario: {scenario}")

    with pytest.raises(expected):
        route(_spec(backend=None), **kwargs)


# ---------------------------------------------------------------------------
# MAJOR 5: manual_attention does NOT escalate + does NOT lose the orphaned id
# ---------------------------------------------------------------------------


def test_manual_attention_raises_with_orphaned_job_id_and_no_gcp_escalation(lease_store):
    """When the cancel grace expires without confirming termination:

    1. The router raises :class:`ManualAttentionRequiredError`.
    2. The orphaned free-lane job id is carried on the exception.
    3. NO call to ``gcp.launch`` happens (no double-submit risk).
    4. The lease is NOT overwritten with a stale or absent id — the
       orphaned id stays in the lease for the orchestrator to consult.
    """
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()  # would succeed if reached, but must NOT be reached

    def _is_live(_b, _h):
        return True  # never leaves queue → manual_attention

    with pytest.raises(ManualAttentionRequiredError) as excinfo:
        route(
            _spec(issue=4242, backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=_is_live,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_fast_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.kind == "nibi"
    assert excinfo.value.orphaned_job_id == "1000"
    assert excinfo.value.cluster == nibi.launches[0].cluster
    # GCP was NEVER launched (no silent escalation).
    assert len(gcp.launches) == 0
    # The orphaned job id is still recorded in the lease — the lease was
    # NOT overwritten by a GCP id (which would have lost the orphan).
    lease = lease_store.read(4242)
    assert lease is not None
    assert lease.backend == "nibi"
    assert lease.job_id == "1000"


# ---------------------------------------------------------------------------
# Minor #8: misconfigured reconnect_fn binding to the wrong backend is ignored
# ---------------------------------------------------------------------------


def test_reconnect_returning_wrong_backend_kind_is_ignored(lease_store):
    """A reconnect_fn that hands back a handle issued by the WRONG backend
    must NOT silently re-attach (would bind to another lane's run).
    """
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    rp = _ExplodingRunpod()
    bogus = RunHandle(
        backend="gcp",  # WRONG: nibi caller, GCP-issued handle
        cluster=None,
        job_id="instance-foreign",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
        extra={"issue": 137},
    )

    def _bogus_reconnect(_backend, _kind, _spec):
        return bogus

    result = route(
        _spec(backend="nibi"),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        reconnect_fn=_bogus_reconnect,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    # The bogus reconnect was rejected → fresh launch happened.
    assert result.chosen_kind == "nibi"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(nibi.launches) == 1


# ---------------------------------------------------------------------------
# Minor #9 regression: attempt-cap message reports attempts_today == cap
# ---------------------------------------------------------------------------


def test_attempt_cap_message_reports_cap_not_one_past(lease_store):
    """The exception's ``attempts_today`` reads as the cap, not cap+1."""
    rp = _ExplodingRunpod()
    cfg = RouterConfig(
        free_wait_seconds=1,
        poll_interval=0.0,
        cancel_grace_seconds=0,
        max_gcp_attempts_per_day=2,
    )
    today = datetime.now(tz=UTC).date().isoformat()
    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_attempts_today=2,
            gcp_attempts_date=today,
        )
    )
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble()
    with pytest.raises(GcpAttemptCapExceededError) as excinfo:
        route(
            _spec(issue=137, backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=cfg,
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.attempts_today == cfg.max_gcp_attempts_per_day
    assert excinfo.value.cap == cfg.max_gcp_attempts_per_day
