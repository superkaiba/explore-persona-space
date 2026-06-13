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
from explore_persona_space.backends.gcp import QuotaHeadroom
from explore_persona_space.backends.router import (
    DEFAULT_AUTO_LANE_ORDER,
    ENV_AUTO_LANE_ORDER,
    FREE_WAIT_SECONDS,
    MAX_GCP_ATTEMPTS_PER_DAY,
    ROUTE_REASON_AUTO_FALLBACK_GCP,
    ROUTE_REASON_AUTO_STARTED,
    ROUTE_REASON_OVERRIDE,
    ROUTE_REASON_RECONNECT,
    auto_lane_order,
    cancel_and_wait,
    park_until_running_or_cap,
)

#: The pre-GCP-first auto order (free SLURM lanes first, GCP as the
#: terminal escalation). Tests that specifically exercise the
#: free→GCP ESCALATION semantics pin this order via
#: ``RouterConfig(lane_order=...)`` — the GCP-first STANDING DEFAULT
#: would otherwise resolve the route at GCP before the free-lane
#: behavior under test ever runs. New-default behavior is covered by
#: the "GCP-first auto order" test section below.
_LEGACY_FREE_FIRST_ORDER: tuple[str, ...] = ("nibi", "fir", "mila", "gcp")


@pytest.fixture(autouse=True)
def _clean_auto_lane_order_env(monkeypatch):
    """Keep every router test hermetic against an ambient env override."""
    monkeypatch.delenv(ENV_AUTO_LANE_ORDER, raising=False)


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
    * ``quota_headroom`` — scripted ``preflight_quota_headroom`` reading
      (a ``QuotaHeadroom``, ``None`` for "no opinion", or an exception
      instance to raise — the router must fail OPEN on it). Defaults to
      ``None`` so every pre-existing test proceeds exactly as before.
    """

    def __init__(
        self,
        *,
        launch_raises: BaseException | None = None,
        quota_headroom: QuotaHeadroom | BaseException | None = None,
    ) -> None:
        self._launch_raises = launch_raises
        self._quota_headroom = quota_headroom
        self.launches: list[RunSpec] = []
        self.quota_probes: list[RunSpec] = []

    def preflight_quota_headroom(self, spec: RunSpec) -> QuotaHeadroom | None:
        self.quota_probes.append(spec)
        if isinstance(self._quota_headroom, BaseException):
            raise self._quota_headroom
        return self._quota_headroom

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
            log_path=f"/workspace/logs/issue-{spec.issue}.log",
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


def test_marker_post_failure_after_launch_does_not_propagate(lease_store):
    """C1 regression: ``_post_backend_selected`` fires AFTER a successful
    launch -- a raising marker poster (flock contention, task.py crash)
    must NOT convert "launched, handle in hand" into an exception (the
    dispatch CLI would exit rc=4 with a live, billing VM/job and no
    recovery record). Markers are observability, not control flow."""

    def exploding_poster(**_kwargs):
        raise RuntimeError("task.py post-marker timed out on the workflow flock")

    rp = _PassiveRunpod()
    result = route(
        _spec(backend="runpod"),
        runpod_backend=rp,
        lease_store=lease_store,
        marker_poster=exploding_poster,
    )
    # The launch happened and the result came back whole.
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    assert result.handle is not None


def test_marker_post_failure_on_free_lane_does_not_propagate(lease_store):
    """Same C1 guard on the explicit free-lane override path (the marker
    fires after the park resolves to RUNNING)."""

    def exploding_poster(**_kwargs):
        raise RuntimeError("marker transport down")

    nibi = _FreeLaneBackend(kind="nibi", starts_when=1)
    result = route(
        _spec(backend="nibi"),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=_is_started_after_n(1),
        marker_poster=exploding_poster,
        config=RouterConfig(free_wait_seconds=2, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert len(nibi.launches) == 1


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
    started, reason, terminal_status = park_until_running_or_cap(
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
    assert terminal_status is None


def test_park_pending_at_cap_returns_park_cap_exceeded():
    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())
    started, reason, terminal_status = park_until_running_or_cap(
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
    assert terminal_status is None


def test_park_terminal_before_running_returns_specific_reason():
    backend = _FreeLaneBackend(kind="nibi", poll_status="dead")
    handle = backend.launch(_spec())
    started, reason, terminal_status = park_until_running_or_cap(
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
    # The triggering PollResult.status is threaded out so callers can
    # gate the started-evidence probe on genuinely-GONE statuses
    # (done/dead) vs possibly-live ones (stalled/gate) — round-6 M1.
    assert terminal_status == "dead"


def test_park_stalled_threads_terminal_status_for_cancel_first_routing():
    """``stalled`` covers LIVE jobs (RUNNING + stale heartbeat;
    SUSPENDED); the caller must see it so the cancel machine runs
    BEFORE any terminal classification (round-6 M1)."""
    backend = _FreeLaneBackend(kind="nibi", poll_status="stalled")
    handle = backend.launch(_spec())
    started, reason, terminal_status = park_until_running_or_cap(
        backend=backend,
        handle=handle,
        is_started=lambda _b, _h: False,
        cap_seconds=10,
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert (started, reason, terminal_status) == (False, "terminal_before_running", "stalled")


def test_park_probe_failures_exceeded_after_consecutive_failures():
    """B1: an ``is_started`` probe that keeps FAILING means the job state
    is UNKNOWN — the park must give up loudly after the consecutive
    budget instead of reading "still pending" forever (or worse,
    letting a poll-side misread classify the job terminal)."""
    from explore_persona_space.backends.router import PARK_MAX_CONSECUTIVE_PROBE_FAILURES

    calls = {"n": 0}

    def raising_probe(_b, _h):
        calls["n"] += 1
        raise RuntimeError("ssh: connect to host nibi port 22: Connection refused")

    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())
    started, reason, terminal_status = park_until_running_or_cap(
        backend=backend,
        handle=handle,
        is_started=raising_probe,
        cap_seconds=1000,  # budget must fire well before the cap
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert (started, reason, terminal_status) == (False, "probe_failures_exceeded", None)
    assert calls["n"] == PARK_MAX_CONSECUTIVE_PROBE_FAILURES


def test_park_probe_failure_counter_resets_on_success():
    """A transient blip (fail, fail, succeed, ...) must NOT accumulate
    toward the consecutive budget — the counter resets on every
    successful probe."""
    # True = raise this tick, False = probe succeeds (returns
    # not-started). Never 3 consecutive raises.
    pattern = iter([True, True, False, True, True, False, True, True, False, False])

    def flaky_probe(_b, _h):
        if next(pattern, False):
            raise RuntimeError("transient ssh blip")
        return False

    backend = _FreeLaneBackend(kind="nibi")
    handle = backend.launch(_spec())
    started, reason, terminal_status = park_until_running_or_cap(
        backend=backend,
        handle=handle,
        is_started=flaky_probe,
        cap_seconds=5,
        poll_interval=0.0,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert started is False
    assert reason == "park_cap_exceeded", (
        "two-then-reset failures must end at the park cap, not the probe budget"
    )
    assert terminal_status is None


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
    """End-to-end: free lane park-fails → cancel → GCP.

    Pinned to the legacy free-first order — under the GCP-first standing
    default the GCP double would resolve the route before the park/cancel
    chain under test ever runs."""
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
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            lane_order=_LEGACY_FREE_FIRST_ORDER,
        ),
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
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=2,
            lane_order=_LEGACY_FREE_FIRST_ORDER,
        ),
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


def test_gcp_probe_error_in_escalation_surfaces_as_no_compute(lease_store):
    """GcpBackend.launch's internal reconnect probe failing (expired
    gcloud auth) must produce the typed fail-closed NoCompute terminal,
    NOT an uncaught rc=4 crash (live auto-lane finding, issue 535)."""
    from explore_persona_space.backends.base import BackendProbeError

    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=BackendProbeError("gcloud list rc=1: Reauthentication failed")
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
            config=RouterConfig(
                free_wait_seconds=1,
                poll_interval=0.0,
                cancel_grace_seconds=0,
                lane_order=_LEGACY_FREE_FIRST_ORDER,  # escalation position under test
            ),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert any(a["outcome"] == "probe_failed" for a in excinfo.value.attempts)


def test_gcp_probe_error_on_explicit_lane_surfaces_as_no_compute(lease_store):
    """Same contract on the explicit ``backend: gcp`` override path."""
    from explore_persona_space.backends.base import BackendProbeError

    gcp = _GcpBackendDouble(
        launch_raises=BackendProbeError("gcloud list rc=1: Reauthentication failed")
    )
    with pytest.raises(NoComputeAvailableError) as excinfo:
        route(
            _spec(backend="gcp"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: True,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert any(a["outcome"] == "probe_failed" for a in excinfo.value.attempts)


def test_gcp_workload_error_surfaces_no_fallback(lease_store):
    """Under the GCP-first standing default, GCP runs in PRIMARY position
    here — a workload failure must surface immediately with NO fallback
    to the SLURM lanes (broken workload code would re-crash on every
    lane and burn queue time)."""
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
    # GCP ran FIRST (default order) — the workload failure must NOT
    # cascade to the SLURM lanes.
    assert len(nibi.launches) == 0


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
        log_path="/workspace/logs/issue-137.log",
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
        # Legacy free-first order: these tests pin the ESCALATION-position
        # cap semantics (cap-trip with nothing after GCP raises). The
        # primary-position cap behavior (skip + fall through) is covered
        # in the GCP-first section.
        lane_order=_LEGACY_FREE_FIRST_ORDER,
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
    and the 45-min issue-tick cron run in parallel. Without the flock
    held across reconnect-check + launch + lease-write, both would
    decide "no live job" and both would submit (and both would escalate
    to GCP if anything timed out → double provision + colliding artifact
    ids). The per-issue flock seals the race.

    Determinism mechanism (deliberately stronger than a finally-barrier
    or a wall-clock sleep): the injected free-lane backend's ``launch``
    blocks on a 2-party ``threading.Barrier`` BEFORE returning. Under
    a BROKEN flock, BOTH threads enter ``launch`` concurrently → the
    barrier trips immediately → ``len(nibi.launches) == 2``. Under a
    WORKING flock, only thread A enters ``launch`` → the barrier times
    out → thread A's ``launch`` catches the ``BrokenBarrierError`` and
    returns the handle anyway, and thread B reconnects via the injected
    ``reconnect_fn``. Result: EXACTLY ONE launch + EXACTLY ONE reconnect
    on the happy path, EXACTLY TWO launches under a regression. No
    single-CPU-CI dependence.
    """
    import contextlib
    import threading

    # 2-party barrier with a short timeout. The point is to FORCE both
    # threads into the critical section simultaneously IF the flock is
    # broken; the short timeout lets the working-flock path finish
    # promptly.
    launch_barrier = threading.Barrier(2, timeout=1.0)
    launch_seen = threading.Event()

    class _GatedNibi(_FreeLaneBackend):
        def launch(self, spec):
            handle = super().launch(spec)
            # Wait for the partner thread — if the flock leaks, the
            # partner will ALSO be inside launch and the barrier trips
            # immediately; both threads return handles, the test sees
            # 2 launches, and the assertion fails LOUDLY.
            # On the working-flock path only THIS thread is inside the
            # critical section, the partner is blocked on the flock, and
            # the barrier times out — suppress the expected
            # BrokenBarrierError so route() can proceed to lease-write
            # and the partner can reconnect.
            with contextlib.suppress(threading.BrokenBarrierError):
                launch_barrier.wait()
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
        if not nibi.launches:
            return None
        return RunHandle(
            backend="nibi",
            cluster="nibi",
            job_id=str(nibi._next_job_id - 1),
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir=f"/scratch/eps/issue-{spec.issue}",
            log_path=f"/scratch/eps/issue-{spec.issue}/job.out",
            extra={"issue": spec.issue},
        )

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

    t1 = threading.Thread(target=_runner, args=(0,))
    t2 = threading.Thread(target=_runner, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert all(e is None for e in errors), errors
    # EXACTLY ONE actual backend.launch — the other thread reconnected.
    # A broken flock would have BOTH threads inside launch concurrently
    # (barrier trips → 2 launches), so this assertion catches the
    # regression deterministically.
    assert len(nibi.launches) == 1, (
        f"expected exactly 1 launch, got {len(nibi.launches)} — flock leaked"
    )
    chosen_kinds = {r.chosen_kind for r in results}
    assert chosen_kinds == {"nibi"}
    # The two results should disagree on reason: one launched, one reconnected.
    reasons = {r.reason for r in results}
    assert ROUTE_REASON_AUTO_STARTED in reasons
    assert ROUTE_REASON_RECONNECT in reasons


# ---------------------------------------------------------------------------
# CONCERN-1 regression: concurrent route() on DIFFERENT issues must NOT block
# ---------------------------------------------------------------------------


def test_concurrent_route_on_DIFFERENT_issues_do_not_block(lease_store):
    """A long-held lock on issue 137 MUST NOT block routing on issue 200.

    Regression test for the global-flock bug: if ``LeaseStore`` flocks
    a shared ``<lease_dir>/.lock`` file (one lock for the whole
    directory) instead of a per-issue ``<lease_dir>/issue-<N>.lock``,
    a 600 s free-lane park INSIDE ``store.transaction(137)`` for
    issue 137 would block ANY concurrent ``route()`` on a different
    issue (e.g. issue 200) for up to 10 min. CLAUDE.md explicitly
    permits concurrent ``/issue <N>`` sessions, so this WOULD fire in
    production.

    Mechanism: thread A enters ``route(issue=137)`` and gates inside
    a fake ``launch`` that holds the per-issue flock for ~1 s. Thread
    B routes ``issue=200`` in parallel; under a per-issue flock,
    B's lock is a DIFFERENT file, so B proceeds without blocking on
    A. Under a global flock, B would be serialized behind A's 1 s
    hold. We assert B finishes within a tight wall-clock budget (and
    well before A) — under a global flock B would take >~1 s.
    """
    import threading
    import time

    a_holding_flock = threading.Event()
    a_may_finish = threading.Event()

    class _SlowNibi(_FreeLaneBackend):
        """Free-lane double that BLOCKS inside ``launch`` until released.

        Holds the per-issue flock (which spans launch + lease-write +
        park) for as long as ``launch`` is in flight.
        """

        def launch(self, spec):
            handle = super().launch(spec)
            a_holding_flock.set()
            # Block until the test releases us. Bounded so a regression
            # doesn't hang the suite indefinitely.
            a_may_finish.wait(timeout=5.0)
            return handle

    nibi_a = _SlowNibi(kind="nibi", est_start_raw=0.0)
    nibi_b = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    rp = _ExplodingRunpod()

    result_a: list[Any] = [None]
    result_b: list[Any] = [None]
    error_a: list[BaseException | None] = [None]
    error_b: list[BaseException | None] = [None]
    elapsed_b: list[float] = [0.0]

    def _route_a():
        try:
            result_a[0] = route(
                _spec(issue=137, backend=None),
                runpod_backend=rp,
                free_backends={"nibi": nibi_a},
                lease_store=lease_store,
                is_started=lambda _b, _h: True,
                config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
                now_fn=_clock(),
                sleep_fn=lambda _s: None,
            )
        except BaseException as exc:
            error_a[0] = exc

    def _route_b():
        try:
            # Wait until thread A is INSIDE the critical section (has
            # acquired the per-issue flock for issue 137 + entered
            # launch). Now race issue 200 against the held lock.
            assert a_holding_flock.wait(timeout=5.0), "thread A never reached launch"
            start = time.monotonic()
            result_b[0] = route(
                _spec(issue=200, backend=None),
                runpod_backend=rp,
                free_backends={"nibi": nibi_b},
                lease_store=lease_store,
                is_started=lambda _b, _h: True,
                config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
                now_fn=_clock(),
                sleep_fn=lambda _s: None,
            )
            elapsed_b[0] = time.monotonic() - start
        except BaseException as exc:
            error_b[0] = exc

    t_a = threading.Thread(target=_route_a)
    t_b = threading.Thread(target=_route_b)
    t_a.start()
    t_b.start()
    t_b.join(timeout=3.0)
    # B must finish even though A is still holding its flock on issue 137.
    assert error_b[0] is None, error_b
    assert result_b[0] is not None, "issue 200 route() did NOT finish while issue 137 held flock"
    assert result_b[0].chosen_kind == "nibi"
    # Tight bound: under a per-issue flock, B does NOT block on A at
    # all (it grabs a separate lock + proceeds). Allow ~500 ms for
    # thread scheduling overhead; under a global flock B would wait
    # for A's full release (~ test timeout), exceeding this bound.
    assert elapsed_b[0] < 0.5, (
        f"issue 200 routing took {elapsed_b[0]:.3f}s while issue 137 held flock "
        f"— this proves the flock is GLOBAL, not per-issue (CONCERN-1 regression)"
    )
    # Release A + assert it also finishes cleanly.
    a_may_finish.set()
    t_a.join(timeout=3.0)
    assert error_a[0] is None, error_a
    assert result_a[0] is not None
    assert result_a[0].chosen_kind == "nibi"
    # Distinct leases on disk for the two issues (sanity).
    lease_137 = lease_store.read(137)
    lease_200 = lease_store.read(200)
    assert lease_137 is not None and lease_137.issue == 137
    assert lease_200 is not None and lease_200.issue == 200


# ---------------------------------------------------------------------------
# N3 regression: empty / None / unknown backend strings are rejected at entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_backend", ["", None, "runpd", "RUNPOD", "cluster"])
def test_route_rejects_invalid_backend_string(lease_store, bad_backend):
    """Belt-and-suspenders: a stringly-typed miswire must NOT silently auto-route.

    ``BackendKind`` Literal validation only fires when ``RunSpec`` is
    *constructed*; a caller that mutates ``spec.backend`` post hoc, or
    constructs the spec with ``# type: ignore``, can sneak in ``""`` /
    ``None`` / a typo. Without the entry-time guard, the empty-string
    case falls through every override branch and into ``_auto_route``,
    silently masking a config bug. The router rejects all of these
    with a ``RouteError`` so the miswire fails LOUDLY.

    ``"cluster"`` is rejected too — slice-5 routing does NOT accept the
    legacy cluster alias; the caller must name the lane (``"nibi"`` /
    ``"fir"``) or leave ``backend`` unset to auto-route.
    """
    from explore_persona_space.backends.router import RouteError

    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    # Construct the spec normally then mutate to simulate a miswire
    # that bypassed Literal validation.
    spec = _spec(backend=None)
    object.__setattr__(spec, "backend", bad_backend)
    with pytest.raises(RouteError, match="backend"):
        route(
            spec,
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            lease_store=lease_store,
        )
    # Nothing should have launched — the guard fires BEFORE any I/O.
    assert len(nibi.launches) == 0
    # And critically, RunPod was never touched (the negative invariant).
    # _ExplodingRunpod.launch raises if called; the absence of that
    # raise is what `match="backend"` proves.


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
        # Legacy free-first order: these tests pin the ESCALATION-position
        # cap semantics (cap-trip with nothing after GCP raises). The
        # primary-position cap behavior (skip + fall through) is covered
        # in the GCP-first section.
        lane_order=_LEGACY_FREE_FIRST_ORDER,
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
            config=RouterConfig(
                free_wait_seconds=1,
                poll_interval=0.0,
                cancel_grace_seconds=0,
                lane_order=_LEGACY_FREE_FIRST_ORDER,  # nibi must be attempted first
            ),
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
        log_path="/workspace/logs/issue-137.log",
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


def test_reconnect_accepts_production_cluster_handle_shape(lease_store):
    """PRODUCTION SLURM reconnect handles use backend="cluster" + cluster=<kind>.

    Both ``SlurmBackend.launch`` and the dispatch CLI's reconnect closure
    return ``RunHandle(backend="cluster", cluster="nibi", ...)`` — NOT
    ``backend="nibi"``. Round-2 Codex Critical (task #535): the
    ``_try_reconnect`` backend cross-check rejected this shape, so a live
    Nibi/Mila job discovered by reconnect was ignored and ``route()``
    fresh-submitted a duplicate. The guard must accept the "cluster"
    alias when the concrete cluster matches the lane.
    """
    issue = 137
    nibi = _FreeLaneBackend(kind="nibi")
    rp = _ExplodingRunpod()
    live = RunHandle(
        backend="cluster",  # production shape — NOT "nibi"
        cluster="nibi",
        job_id="15931234",
        pod_name=f"eps-issue-{issue}",
        scratch_dir=f"/scratch/eps/issue-{issue}",
        log_path=f"/scratch/eps/issue-{issue}/job.out",
        extra={"issue": issue, "account": "rrg-test_gpu"},
    )

    def reconnect_fn(backend, kind, spec):
        if kind == "nibi":
            return live
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
    assert result.handle.job_id == "15931234"
    assert len(nibi.launches) == 0  # the live job was reused — no duplicate submit


def test_reconnect_cluster_handle_for_wrong_cluster_is_ignored(lease_store):
    """A backend="cluster" handle whose ``cluster`` names a DIFFERENT lane
    is still the cross-lane mismatch the guard exists for — rejected,
    fresh launch proceeds.
    """
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    rp = _ExplodingRunpod()
    foreign = RunHandle(
        backend="cluster",
        cluster="fir",  # WRONG cluster for the nibi lane
        job_id="999999",
        pod_name="eps-issue-137",
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        extra={"issue": 137},
    )

    result = route(
        _spec(backend="nibi"),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        reconnect_fn=lambda _b, _k, _s: foreign,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(nibi.launches) == 1  # foreign handle rejected → fresh launch


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
        # Legacy free-first order: these tests pin the ESCALATION-position
        # cap semantics (cap-trip with nothing after GCP raises). The
        # primary-position cap behavior (skip + fall through) is covered
        # in the GCP-first section.
        lane_order=_LEGACY_FREE_FIRST_ORDER,
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


# ---------------------------------------------------------------------------
# Slice-7: Mila gating via ``mila_socket_alive``
# ---------------------------------------------------------------------------


def test_router_skips_mila_when_socket_down(lease_store):
    """``mila_socket_alive`` returning False = Mila is NEVER launched.

    The router treats a dead socket as "skip the lane", NOT as an
    error: an instant nibi sibling still wins, no marker collateral.
    """
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    mila = _FreeLaneBackend(kind="mila", est_start_raw=0.0)
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi, "mila": mila},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        mila_socket_alive=lambda: False,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    # Nibi was the only candidate the router considered (Mila filtered
    # out before ranking) — Mila MUST NOT have been launched.
    assert result.chosen_kind == "nibi"
    assert len(mila.launches) == 0
    assert len(nibi.launches) == 1


def test_router_uses_mila_when_socket_alive_and_it_wins_estimate(lease_store):
    """When the socket is up AND Mila ranks first, the router uses Mila.

    Proves the gate doesn't silently keep Mila out of contention once
    its socket is alive — full first-class status, ranked by the same
    est-start signal every other free lane uses.
    """
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=600.0)  # 10 min queue
    mila = _FreeLaneBackend(kind="mila", est_start_raw=5.0)  # ~instant
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi, "mila": mila},
        lease_store=lease_store,
        is_started=lambda b, _h: b is mila,
        mila_socket_alive=lambda: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "mila"
    assert len(mila.launches) == 1
    assert len(nibi.launches) == 0


def test_router_socket_down_does_not_block_when_only_mila_present(lease_store):
    """Mila-only auto chain + dead socket → falls back to GCP cleanly.

    Socket-down is the designed graceful-skip path. There MUST be no
    workload error / "Mila down" exception — the router proceeds to
    the next tier as if Mila were absent from the dict.
    """
    rp = _ExplodingRunpod()
    mila = _FreeLaneBackend(kind="mila")
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"mila": mila},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        mila_socket_alive=lambda: False,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(mila.launches) == 0
    assert len(gcp.launches) == 1


def test_router_explicit_mila_override_still_runs_when_socket_alive(lease_store):
    """``backend: mila`` override targets the lane directly.

    Override is not subject to the auto-chain gate — the operator
    asked for Mila, and the socket-alive predicate is consulted ONLY
    when the override path also exercises the launch wiring (the
    gate fires inside ``_auto_route``). When the socket IS alive the
    override succeeds end-to-end.
    """
    rp = _ExplodingRunpod()
    mila = _FreeLaneBackend(kind="mila")
    result = route(
        _spec(backend="mila"),
        runpod_backend=rp,
        free_backends={"mila": mila},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        mila_socket_alive=lambda: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "mila"
    assert len(mila.launches) == 1


# ---------------------------------------------------------------------------
# prepare() chokepoint (router fix5 — live-acceptance finding, issue 535)
# ---------------------------------------------------------------------------
#
# The first live acceptance run launched a Nibi job WITHOUT the rsync repo
# sync + secrets push because SlurmBackend.prepare had zero production
# callers — every route() launch site called backend.launch directly. The
# tests below pin: (a) prepare runs BEFORE launch at every FRESH launch
# site; (b) reconnect paths never call prepare (SlurmBackend.prepare
# rsyncs with --delete and would yank code from under a RUNNING job);
# (c) a prepare failure is provision-class — next tier on auto, typed
# terminal on an explicit override.


class _PrepareRecordingLane(_FreeLaneBackend):
    """Free-lane double that records prepare/launch call order."""

    def __init__(self, *, prepare_raises: BaseException | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []
        self._prepare_raises = prepare_raises

    def prepare(self, spec: RunSpec) -> None:
        self.calls.append("prepare")
        if self._prepare_raises is not None:
            raise self._prepare_raises

    def launch(self, spec: RunSpec) -> RunHandle:
        self.calls.append("launch")
        return super().launch(spec)


class _PrepareRecordingGcp(_GcpBackendDouble):
    """GCP double that records prepare/launch call order."""

    def __init__(self, *, prepare_raises: BaseException | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []
        self._prepare_raises = prepare_raises

    def prepare(self, spec: RunSpec) -> None:
        self.calls.append("prepare")
        if self._prepare_raises is not None:
            raise self._prepare_raises

    def launch(self, spec: RunSpec) -> RunHandle:
        self.calls.append("launch")
        return super().launch(spec)


class _PrepareRecordingRunpod(_PassiveRunpod):
    """RunPod double that records prepare/launch call order."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def prepare(self, spec: RunSpec) -> None:
        self.calls.append("prepare")

    def launch(self, spec: RunSpec) -> RunHandle:
        self.calls.append("launch")
        return super().launch(spec)


def test_explicit_lane_calls_prepare_before_launch(lease_store):
    nibi = _PrepareRecordingLane(kind="nibi")
    result = route(
        _spec(backend="nibi"),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert nibi.calls == ["prepare", "launch"], (
        "fresh explicit-lane launch must run prepare (rsync + secrets) BEFORE launch; "
        f"got call order {nibi.calls}"
    )


def test_runpod_override_calls_prepare_before_launch(lease_store):
    rp = _PrepareRecordingRunpod()
    result = route(
        _spec(backend="runpod"),
        runpod_backend=rp,
        lease_store=lease_store,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert rp.calls == ["prepare", "launch"]


def test_auto_free_lane_calls_prepare_before_launch(lease_store):
    nibi = _PrepareRecordingLane(kind="nibi", est_start_raw=0.0)
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert nibi.calls == ["prepare", "launch"]


def test_gcp_escalation_calls_prepare_before_launch(lease_store):
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _PrepareRecordingGcp()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,  # nibi never starts → escalate
        is_live_after_cancel=lambda _b, _h: False,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            lane_order=_LEGACY_FREE_FIRST_ORDER,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert gcp.calls == ["prepare", "launch"]


def test_explicit_lane_reconnect_does_not_call_prepare(lease_store):
    """Reconnect re-attaches to a RUNNING job — re-preparing would rsync
    --delete the scratch out from under it. prepare must NOT run."""
    nibi = _PrepareRecordingLane(kind="nibi")
    live = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="424242",
        pod_name="eps-issue-137",
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        extra={"issue": 137},
    )
    result = route(
        _spec(backend="nibi"),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        reconnect_fn=lambda _b, _k, _s: live,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.reason == ROUTE_REASON_RECONNECT
    assert result.handle.job_id == "424242"
    assert nibi.calls == [], f"reconnect must not prepare OR launch; got {nibi.calls}"


def test_auto_reconnect_does_not_call_prepare(lease_store):
    nibi = _PrepareRecordingLane(kind="nibi", est_start_raw=0.0)
    live = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="424243",
        pod_name="eps-issue-137",
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        extra={"issue": 137},
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        lease_store=lease_store,
        reconnect_fn=lambda _b, k, _s: live if k == "nibi" else None,
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.reason == ROUTE_REASON_RECONNECT
    assert nibi.calls == [], f"auto reconnect must not prepare OR launch; got {nibi.calls}"


def test_prepare_failure_on_auto_falls_to_next_tier_never_runpod(lease_store):
    """A free-lane prepare failure (rsync/scp non-zero) is provision-class:
    next tier on auto (→ GCP), and RunPod stays unreachable."""
    import subprocess

    nibi = _PrepareRecordingLane(
        kind="nibi",
        est_start_raw=0.0,
        prepare_raises=subprocess.CalledProcessError(255, ["rsync"]),
    )
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            lane_order=_LEGACY_FREE_FIRST_ORDER,  # free-first: prepare-fail → next tier
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert nibi.calls == ["prepare"], "launch must NOT run after a failed prepare"
    assert len(gcp.launches) == 1
    prepare_attempts = [a for a in result.attempts if a.outcome == "prepare_failed"]
    assert prepare_attempts and prepare_attempts[0].kind == "nibi"


def test_prepare_failure_on_explicit_lane_raises_typed_terminal(lease_store):
    import subprocess

    from explore_persona_space.backends import BackendPrepareError

    nibi = _PrepareRecordingLane(
        kind="nibi",
        prepare_raises=subprocess.CalledProcessError(1, ["scp"]),
    )
    gcp = _GcpBackendDouble()
    with pytest.raises(BackendPrepareError) as excinfo:
        route(
            _spec(backend="nibi"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: True,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.kind == "nibi"
    assert nibi.calls == ["prepare"], "launch must NOT run after a failed prepare"
    assert len(gcp.launches) == 0, "explicit override never silently re-routes"


def test_gcp_prepare_failure_after_free_lanes_fail_raises_no_compute(lease_store):
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _PrepareRecordingGcp(prepare_raises=RuntimeError("metadata render failed"))
    with pytest.raises(NoComputeAvailableError) as excinfo:
        route(
            _spec(backend=None),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert gcp.calls == ["prepare"]
    assert any(a["outcome"] == "prepare_failed" for a in excinfo.value.attempts)


# ---------------------------------------------------------------------------
# terminal-before-running classification (router fix5, secondary)
# ---------------------------------------------------------------------------
#
# A fast-failing job (e.g. in-job preflight failure) transitions
# PD→R→exit between park polls, so it "vanishes" before being observed
# RUNNING. Pre-fix the park state machine read that as
# no_compute_available — on the auto lane that ESCALATES TO GCP, i.e. a
# workload bug burns paid credit on a doomed re-run. The
# started_evidence_probe (scratch-dir status.json / job.out read)
# distinguishes "started and FAILED" (workload failure, surface, NO
# fallback) from "never started" (genuine no-compute, escalation OK).


_EVIDENCE = {
    "phase": "preflight-failed",
    "job_out_tail": "[FAIL] secrets file not found\n[phase=preflight-failed]",
    "status_json": {},
}


def test_terminal_with_artifacts_is_workload_failure_no_gcp_on_auto(
    lease_store, marker_poster, captured_markers
):
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0, poll_status="dead")
    gcp = _GcpBackendDouble()
    with pytest.raises(WorkloadSurfacedError) as excinfo:
        route(
            _spec(backend=None),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,  # never observed RUNNING
            is_live_after_cancel=lambda _b, _h: False,
            started_evidence_probe=lambda _b, _h: dict(_EVIDENCE),
            marker_poster=marker_poster,
            config=RouterConfig(
                free_wait_seconds=5,
                poll_interval=0.0,
                cancel_grace_seconds=0,
                lane_order=_LEGACY_FREE_FIRST_ORDER,
            ),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.chosen_kind == "nibi"
    assert excinfo.value.evidence.get("phase") == "preflight-failed"
    assert len(gcp.launches) == 0, (
        "a started-then-FAILED workload must NOT escalate to GCP — that burns "
        "paid credit re-running a doomed workload"
    )
    # Terminal breadcrumb marker carries the workload_failure reason.
    failures = [
        json.loads(m["note"]) for m in captured_markers if m.get("marker") == "epm:backend-selected"
    ]
    assert any(b.get("reason") == "workload_failure" for b in failures)


def test_terminal_without_artifacts_still_escalates_to_gcp(lease_store):
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0, poll_status="dead")
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        started_evidence_probe=lambda _b, _h: None,  # no runtime artifacts
        config=RouterConfig(
            free_wait_seconds=5,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            lane_order=_LEGACY_FREE_FIRST_ORDER,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1


def test_terminal_probe_failure_falls_back_to_no_compute_and_logs(lease_store, caplog):
    def _exploding_probe(_b, _h):
        raise OSError("scp: connection refused")

    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0, poll_status="dead")
    with caplog.at_level("WARNING"), pytest.raises(NoComputeAvailableError):
        route(
            _spec(backend=None),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            gcp_backend=None,  # nothing to escalate to → no_compute terminal
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            started_evidence_probe=_exploding_probe,
            config=RouterConfig(free_wait_seconds=5, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert any("started-evidence probe FAILED" in r.message for r in caplog.records), (
        "a probe failure must be logged loud (it silently degrades classification)"
    )


def test_explicit_lane_terminal_with_artifacts_raises_workload_not_no_compute(lease_store):
    """The live-run regression shape: explicit `--backend nibi`, job fast-fails
    in preflight → must surface as a workload failure, not no_compute."""
    nibi = _FreeLaneBackend(kind="nibi", poll_status="dead")
    with pytest.raises(WorkloadSurfacedError) as excinfo:
        route(
            _spec(backend="nibi"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            started_evidence_probe=lambda _b, _h: dict(_EVIDENCE),
            config=RouterConfig(free_wait_seconds=5, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.chosen_kind == "nibi"
    assert "preflight-failed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Round-6 M1: a `stalled`-triggered terminal park covers possibly-LIVE jobs
# (RUNNING + stale heartbeat; SUSPENDED) — the evidence path must NOT fire
# for it; the job is cancelled FIRST (the issue-535 live run raised
# WorkloadSurfacedError before the cancel machine and orphaned a live job).
# ---------------------------------------------------------------------------


def test_stalled_terminal_cancels_first_and_skips_evidence_on_auto(lease_store):
    probe_calls: list[int] = []

    def recording_probe(_b, _h):
        probe_calls.append(1)
        return dict(_EVIDENCE)

    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0, poll_status="stalled")
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,  # cancel confirms gone
        started_evidence_probe=recording_probe,
        config=RouterConfig(
            free_wait_seconds=5,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            lane_order=_LEGACY_FREE_FIRST_ORDER,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert probe_calls == [], (
        "a stalled-classified job may be LIVE — the started-evidence probe "
        "must not classify it terminal before the cancel machine runs"
    )
    assert len(nibi.teardowns) == 1, "the stalled job must be scancel'd, never orphaned"
    assert result.chosen_kind == "gcp", "after a confirmed cancel the auto chain continues"


def test_stalled_terminal_on_explicit_lane_cancels_and_does_not_raise_workload(lease_store):
    nibi = _FreeLaneBackend(kind="nibi", poll_status="stalled")
    with pytest.raises(NoComputeAvailableError):
        route(
            _spec(backend="nibi"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            started_evidence_probe=lambda _b, _h: dict(_EVIDENCE),
            config=RouterConfig(free_wait_seconds=5, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert len(nibi.teardowns) == 1, "explicit lane: the stalled job must be scancel'd too"


def test_stalled_terminal_still_live_after_cancel_is_manual_attention(lease_store):
    """The full live-failure chain: stalled-classified LIVE job + a cancel
    that cannot confirm death → ManualAttentionRequiredError carrying the
    orphaned id — NEVER WorkloadSurfacedError while the job may be live."""
    from explore_persona_space.backends.router import ManualAttentionRequiredError

    nibi = _FreeLaneBackend(kind="nibi", poll_status="stalled")
    with pytest.raises(ManualAttentionRequiredError) as excinfo:
        route(
            _spec(backend="nibi"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: True,  # still live — cancel unconfirmed
            started_evidence_probe=lambda _b, _h: dict(_EVIDENCE),
            config=RouterConfig(free_wait_seconds=5, poll_interval=0.0, cancel_grace_seconds=1),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert excinfo.value.orphaned_job_id
    assert len(nibi.teardowns) == 1


# ---------------------------------------------------------------------------
# Round-6 B1: a reconnect PROBE failure (BackendProbeError) must never read
# as "no live job" — submitting blind risks a duplicate of a live job.
# ---------------------------------------------------------------------------


def test_reconnect_probe_failure_skips_lane_no_blind_submit_on_auto(lease_store):
    from explore_persona_space.backends.base import BackendProbeError

    def probing_reconnect(_backend, _kind, _spec_arg):
        raise BackendProbeError("squeue --name probe failed: rc=255 Connection refused")

    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        reconnect_fn=probing_reconnect,
        config=RouterConfig(
            free_wait_seconds=5,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            lane_order=_LEGACY_FREE_FIRST_ORDER,  # nibi probed FIRST, skip → gcp
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert nibi.launches == [], (
        "an unprobeable lane must be SKIPPED, not blind-submitted — a live "
        "job may exist behind the broken probe"
    )
    assert result.chosen_kind == "gcp"


def test_reconnect_probe_failure_on_explicit_lane_raises_typed_terminal(lease_store):
    from explore_persona_space.backends.base import BackendProbeError

    def probing_reconnect(_backend, _kind, _spec_arg):
        raise BackendProbeError("squeue --name probe failed: rc=255 Connection refused")

    nibi = _FreeLaneBackend(kind="nibi")
    with pytest.raises(NoComputeAvailableError, match="refusing to submit blind"):
        route(
            _spec(backend="nibi"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            reconnect_fn=probing_reconnect,
            config=RouterConfig(free_wait_seconds=5, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert nibi.launches == []


# ---------------------------------------------------------------------------
# Round-6 Mn1: the prepare-fail breadcrumb reason must match the typed
# terminal's `reason: backend_prepare_failed` (it previously said
# `no_compute_available`).
# ---------------------------------------------------------------------------


def test_prepare_failed_breadcrumb_reason_matches_typed_terminal(
    lease_store, marker_poster, captured_markers
):
    from explore_persona_space.backends.router import (
        ROUTE_REASON_PREPARE_FAILED,
        BackendPrepareError,
    )

    class _PrepareExploding(_FreeLaneBackend):
        def prepare(self, spec: RunSpec) -> None:
            raise OSError("rsync: connection unexpectedly closed")

    nibi = _PrepareExploding(kind="nibi")
    with pytest.raises(BackendPrepareError):
        route(
            _spec(backend="nibi"),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            marker_poster=marker_poster,
            config=RouterConfig(free_wait_seconds=5, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    breadcrumbs = _by_reason(captured_markers, ROUTE_REASON_PREPARE_FAILED)
    assert breadcrumbs, "prepare failure must post a backend_prepare_failed breadcrumb"
    assert not _by_reason(captured_markers, "no_compute_available")


# ---------------------------------------------------------------------------
# GCP-first auto order (standing default, env override, primary-lane GCP)
# ---------------------------------------------------------------------------
#
# The auto chain's STANDING DEFAULT is GCP first ("gcp", "nibi", "fir",
# "mila") so credits-backed GCP capacity is consumed BEFORE the free
# SLURM lanes. There is deliberately NO date logic — flipping back is a
# human action (EPM_AUTO_LANE_ORDER env override or a default edit),
# never a clock. RunPod remains override-only in EVERY order.


def test_default_auto_lane_order_is_gcp_first():
    """The standing default puts GCP before every free SLURM lane."""
    assert DEFAULT_AUTO_LANE_ORDER == ("gcp", "nibi", "fir", "mila")
    # With no env override, the resolver returns the default verbatim.
    assert auto_lane_order() == DEFAULT_AUTO_LANE_ORDER


def test_auto_lane_order_env_override_parsed(monkeypatch):
    """EPM_AUTO_LANE_ORDER is comma-separated; whitespace is tolerated."""
    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, " nibi , fir ,mila,gcp ")
    assert auto_lane_order() == ("nibi", "fir", "mila", "gcp")
    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "gcp")
    assert auto_lane_order() == ("gcp",)


def test_auto_lane_order_env_rejects_runpod(monkeypatch):
    """A 'runpod' entry RAISES loudly — real-money safety; NEVER silently
    dropped. RunPod stays override-only regardless of the configured order."""
    from explore_persona_space.backends.router import RouteError

    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "runpod,nibi")
    with pytest.raises(RouteError, match="runpod"):
        auto_lane_order()


@pytest.mark.parametrize("bad_lane", ["bogus", "auto", "cluster", "RUNPOD", "Nibi"])
def test_auto_lane_order_env_rejects_unknown_lane(monkeypatch, bad_lane):
    from explore_persona_space.backends.router import RouteError

    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, f"nibi,{bad_lane}")
    with pytest.raises(RouteError, match="lane"):
        auto_lane_order()


def test_auto_lane_order_env_rejects_duplicates(monkeypatch):
    from explore_persona_space.backends.router import RouteError

    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "nibi,gcp,nibi")
    with pytest.raises(RouteError, match="duplicate"):
        auto_lane_order()


def test_route_rejects_runpod_in_config_lane_order(lease_store):
    """A per-call RouterConfig.lane_order smuggling 'runpod' fails at
    route() entry, BEFORE any reconnect or submit I/O."""
    from explore_persona_space.backends.router import RouteError

    nibi = _FreeLaneBackend(kind="nibi")
    rp = _ExplodingRunpod()
    with pytest.raises(RouteError, match="runpod"):
        route(
            _spec(backend=None),
            runpod_backend=rp,
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            config=RouterConfig(lane_order=("runpod", "nibi")),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert len(nibi.launches) == 0


def test_gcp_first_default_attempts_gcp_before_free_lanes(
    lease_store, marker_poster, captured_markers
):
    """Under the standing default, a healthy GCP resolves the route with
    NO free-lane submit — and the marker's attempts trail shows GCP as
    the first (and only) attempt."""
    rp = _ExplodingRunpod()  # RunPod stays unreachable on auto
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.reason == ROUTE_REASON_AUTO_FALLBACK_GCP  # reason code kept for schema stability
    assert len(gcp.launches) == 1
    assert len(nibi.launches) == 0
    # Marker fidelity: the final marker's attempts list leads with GCP.
    finals = [
        body
        for body in _by_reason(captured_markers, ROUTE_REASON_AUTO_FALLBACK_GCP)
        if not body.get("extra", {}).get("intermediate")
    ]
    assert finals
    launched = [a for a in finals[-1]["attempts"] if a["outcome"] == "launched"]
    assert launched and launched[0]["kind"] == "gcp"


def test_gcp_primary_provision_fail_falls_through_to_free_lanes(
    lease_store, marker_poster, captured_markers
):
    """GCP capacity failure in PRIMARY position continues down the order
    to the SLURM lanes; the attempts trail reflects the actual order
    (GCP first, then the free lane that started)."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _spec(backend=None),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert result.reason == ROUTE_REASON_AUTO_STARTED
    assert len(nibi.launches) == 1
    # The attempts trail records GCP's provisioning failure BEFORE the
    # nibi launch — actual attempt order, not a free-first fiction.
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    assert outcomes.index(("gcp", "provisioning_failure")) < outcomes.index(("nibi", "launched"))
    # Same order in the posted marker body.
    finals = _by_reason(captured_markers, ROUTE_REASON_AUTO_STARTED)
    assert finals
    marker_outcomes = [(a["kind"], a["outcome"]) for a in finals[-1]["attempts"]]
    assert marker_outcomes.index(("gcp", "provisioning_failure")) < marker_outcomes.index(
        ("nibi", "launched")
    )


def test_gcp_quota_headroom_insufficient_skips_lane_without_attempt_burn(
    lease_store, marker_poster, captured_markers
):
    """A POSITIVE insufficient-headroom probe reading skips the GCP lane
    BEFORE the per-day attempt counter bumps (#608: four quota-doomed
    creates burned the cap against an exhausted regional quota) and
    continues down the auto order."""
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble(
        quota_headroom=QuotaHeadroom(
            metric="NVIDIA_A100_80GB_GPUS",
            region="us-central1",
            limit=8.0,
            usage=8.0,
            needed=4,
        )
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert gcp.launches == []  # no create was attempted
    assert gcp.quota_probes  # the probe WAS consulted
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    assert outcomes.index(("gcp", "quota_headroom_insufficient")) < outcomes.index(
        ("nibi", "launched")
    )
    skip = next(a for a in result.attempts if a.outcome == "quota_headroom_insufficient")
    assert "NVIDIA_A100_80GB_GPUS" in skip.detail
    assert "without burning a daily attempt" in skip.detail
    # The load-bearing assertion: the per-day GCP attempt counter did NOT bump.
    lease = lease_store.read(137)
    assert lease is None or lease.gcp_attempts_today == 0
    # Marker fidelity: the skip rides the attempts trail in the final marker.
    finals = _by_reason(captured_markers, ROUTE_REASON_AUTO_STARTED)
    assert finals
    assert ("gcp", "quota_headroom_insufficient") in [
        (a["kind"], a["outcome"]) for a in finals[-1]["attempts"]
    ]


def test_gcp_quota_headroom_sufficient_proceeds_to_launch(lease_store):
    """A sufficient-headroom reading proceeds to the normal launch path
    (attempt bumped, instance created)."""
    gcp = _GcpBackendDouble(
        quota_headroom=QuotaHeadroom(
            metric="NVIDIA_A100_80GB_GPUS",
            region="us-central1",
            limit=8.0,
            usage=4.0,
            needed=4,
        )
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": _FreeLaneBackend(kind="nibi")},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    lease = lease_store.read(137)
    assert lease is not None and lease.gcp_attempts_today == 1


def test_gcp_quota_preflight_fails_open_on_probe_error(lease_store):
    """A probe that RAISES fails OPEN: the launch proceeds exactly as
    before (the pre-check must never block a launch — #608 contract)."""
    gcp = _GcpBackendDouble(quota_headroom=RuntimeError("gcloud not installed"))
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": _FreeLaneBackend(kind="nibi")},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    assert gcp.quota_probes  # the probe WAS consulted, then failed open


def test_gcp_route_marker_carries_provisioning_and_quota_pool(
    lease_store, marker_poster, captured_markers
):
    """#631 D4: the SUCCESS-path GCP epm:backend-selected marker carries the
    additive provisioning_model + quota_pool fields. The default lora-7b spec
    is an A100 STANDARD launch, so the resolved values are STANDARD +
    NVIDIA_A100_80GB_GPUS (the on-demand A100 pool)."""
    gcp = _GcpBackendDouble(
        quota_headroom=QuotaHeadroom(
            metric="NVIDIA_A100_80GB_GPUS",
            region="us-central1",
            limit=8.0,
            usage=4.0,
            needed=1,
        )
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": _FreeLaneBackend(kind="nibi")},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    # The result object carries the fields.
    assert result.extra["provisioning_model"] == "STANDARD"
    assert result.extra["quota_pool"] == "NVIDIA_A100_80GB_GPUS"
    # The SUCCESS-path marker body (reason auto_fallback_gcp) carries them too.
    bodies = _by_reason(captured_markers, "auto_fallback_gcp")
    assert bodies, "no success-path GCP epm:backend-selected marker was posted"
    final = bodies[-1]
    assert final["extra"]["provisioning_model"] == "STANDARD"
    assert final["extra"]["quota_pool"] == "NVIDIA_A100_80GB_GPUS"
    # Pre-existing fields untouched (additive change).
    assert "gcp_attempts_today" in final["extra"]


def test_gcp_explicit_override_marker_carries_provisioning_and_quota_pool(
    lease_store, marker_poster, captured_markers
):
    """#631 round-3: an explicit ``backend: gcp`` fresh launch (the
    ``_override_free_or_gcp`` terminal path, reason ``override``) carries the
    additive provisioning_model + quota_pool fields on BOTH the result object
    and its epm:backend-selected marker — the GCP analogue of an explicit
    ``backend: runpod`` override, which round 1 left without the new fields."""
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend="gcp"),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.reason == ROUTE_REASON_OVERRIDE
    # The result object carries the fields.
    assert result.extra["provisioning_model"] == "STANDARD"
    assert result.extra["quota_pool"] == "NVIDIA_A100_80GB_GPUS"
    # The override-path marker body carries them too.
    bodies = _by_reason(captured_markers, ROUTE_REASON_OVERRIDE)
    assert bodies, "no explicit-override GCP epm:backend-selected marker was posted"
    final = bodies[-1]
    assert final["extra"]["provisioning_model"] == "STANDARD"
    assert final["extra"]["quota_pool"] == "NVIDIA_A100_80GB_GPUS"


def test_gcp_reconnect_marker_carries_provisioning_and_quota_pool(
    lease_store, marker_poster, captured_markers
):
    """#631 round-3: an auto-chain GCP reconnect (the ``_record_reconnect``
    terminal path, reason ``reconnect``) carries the additive
    provisioning_model + quota_pool fields on the result object and marker.
    Reconnect is the per-launch idempotency hinge — round 1 produced a
    missing-fields marker here."""
    gcp = _GcpBackendDouble()
    existing = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-existing",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"issue": 137, "zone": "us-central1-a"},
    )

    def reconnect_fn(backend, kind, spec):
        return existing if kind == "gcp" else None

    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": _FreeLaneBackend(kind="nibi")},
        gcp_backend=gcp,
        lease_store=lease_store,
        reconnect_fn=reconnect_fn,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.reason == ROUTE_REASON_RECONNECT
    assert len(gcp.launches) == 0  # reconnect, not a fresh provision
    # The result object carries the fields on the reconnect path.
    assert result.extra["provisioning_model"] == "STANDARD"
    assert result.extra["quota_pool"] == "NVIDIA_A100_80GB_GPUS"
    # The reconnect-path marker body carries them too.
    bodies = _by_reason(captured_markers, ROUTE_REASON_RECONNECT)
    assert bodies, "no GCP-reconnect epm:backend-selected marker was posted"
    final = bodies[-1]
    assert final["extra"]["provisioning_model"] == "STANDARD"
    assert final["extra"]["quota_pool"] == "NVIDIA_A100_80GB_GPUS"


def test_gcp_quota_headroom_insufficient_terminal_raises_no_compute(lease_store, monkeypatch):
    """GCP in TERMINAL position (free-first override) with insufficient
    headroom raises the typed NoCompute terminal WITHOUT burning an
    attempt — the doomed create is never issued."""
    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "nibi,gcp")
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble(
        quota_headroom=QuotaHeadroom(
            metric="NVIDIA_A100_80GB_GPUS",
            region="us-central1",
            limit=8.0,
            usage=8.0,
            needed=4,
        )
    )
    with pytest.raises(NoComputeAvailableError) as excinfo:
        route(
            _spec(backend=None),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert any(a["outcome"] == "quota_headroom_insufficient" for a in excinfo.value.attempts)
    assert gcp.launches == []
    lease = lease_store.read(137)
    assert lease is None or lease.gcp_attempts_today == 0


def test_gcp_provisioning_failure_detail_carries_stderr_tail(
    lease_store, marker_poster, captured_markers
):
    """The classified create failure's captured gcloud stderr tail rides
    the attempt detail into the marker attempts trail (#608: the reason
    said "stderr below" but no stderr followed anywhere)."""
    stderr = "Quota 'NVIDIA_A100_80GB_GPUS' exceeded.  Limit: 8.0 in region us-central1."
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "gcloud create returned 1; no known provisioning pattern (stderr below)",
            evidence={"stderr_tail": stderr, "matched_pattern": None},
        )
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    fail = next(a for a in result.attempts if a.outcome == "provisioning_failure")
    assert "NVIDIA_A100_80GB_GPUS" in fail.detail
    assert "stderr_tail:" in fail.detail
    # Marker fidelity: the stderr tail survives into the posted attempts.
    finals = _by_reason(captured_markers, ROUTE_REASON_AUTO_STARTED)
    assert finals
    marker_fail = next(a for a in finals[-1]["attempts"] if a["outcome"] == "provisioning_failure")
    assert "NVIDIA_A100_80GB_GPUS" in marker_fail["detail"]


def test_gcp_primary_prepare_fail_falls_through_to_free_lanes(lease_store):
    """A GCP prepare failure is provision-class: next lane, not terminal,
    when lanes remain after GCP."""
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _PrepareRecordingGcp(prepare_raises=RuntimeError("metadata render failed"))
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert gcp.calls == ["prepare"], "launch must NOT run after a failed prepare"
    assert any(a.outcome == "prepare_failed" and a.kind == "gcp" for a in result.attempts)


def test_gcp_primary_probe_error_falls_through_to_free_lanes(lease_store):
    """A GCP state-probe failure in primary position skips the lane and
    continues (no credit spent on unknown state; same safe reaction the
    SLURM lanes take on an unprobeable reconnect). The terminal-position
    fail-closed contract is pinned separately under the legacy order."""
    from explore_persona_space.backends.base import BackendProbeError

    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble(
        launch_raises=BackendProbeError("gcloud list rc=1: Reauthentication failed")
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert any(a.outcome == "probe_failed" and a.kind == "gcp" for a in result.attempts)


def test_gcp_primary_attempt_counts_toward_daily_cap(lease_store):
    """Primary-lane GCP attempts bump the SAME per-day counter as
    escalation attempts — the guard bounds provision attempts wherever
    GCP sits in the order."""
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    route(
        _spec(issue=137, backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    lease = lease_store.read(137)
    assert lease is not None
    assert lease.gcp_attempts_today == 1
    assert lease.gcp_attempts_date == datetime.now(tz=UTC).date().isoformat()


def test_gcp_primary_at_cap_skips_gcp_and_falls_through(lease_store):
    """At the per-day cap with lanes REMAINING after GCP, the router
    skips GCP (zero credit spent) and continues down the order instead
    of bricking the route for the day. The cap-trip RAISE is preserved
    when GCP is the LAST lane (legacy escalation position — pinned in
    test_gcp_attempt_count_guard_caps_repeated_escalation)."""
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
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()  # would succeed if (wrongly) reached
    result = route(
        _spec(issue=137, backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=2),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert len(gcp.launches) == 0, "an over-cap GCP attempt must spend ZERO credit"
    assert any(a.outcome == "attempt_cap_exceeded" and a.kind == "gcp" for a in result.attempts)
    # The on-disk counter did NOT grow past the cap.
    lease = lease_store.read(137)
    assert lease is not None
    assert lease.gcp_attempts_today == 2


def test_env_override_free_first_restores_legacy_escalation(monkeypatch, lease_store):
    """Setting EPM_AUTO_LANE_ORDER=nibi,fir,mila,gcp restores the
    free-first chain: the free lane is tried (and park-fails) BEFORE the
    GCP escalation."""
    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "nibi,fir,mila,gcp")
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,  # nibi never starts
        is_live_after_cancel=lambda _b, _h: False,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(nibi.launches) == 1, "free lane must be attempted FIRST under the override"
    assert len(gcp.launches) == 1


def test_config_lane_order_beats_env_override(monkeypatch, lease_store):
    """A per-call RouterConfig.lane_order wins over the env override."""
    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "gcp,nibi")
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, lane_order=("nibi", "gcp")),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "nibi"
    assert len(gcp.launches) == 0


def test_route_logs_resolved_auto_order(lease_store, caplog):
    """route() emits ONE INFO line stating the resolved auto order and
    its source (env override vs default) at entry to the auto path."""
    import logging

    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    with caplog.at_level(logging.INFO, logger="explore_persona_space.backends.router"):
        route(
            _spec(backend=None),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: True,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    order_lines = [r.message for r in caplog.records if "auto lane order" in r.message]
    assert order_lines, "route() must log the resolved auto order"
    assert "gcp -> nibi -> fir -> mila" in order_lines[0]
    assert "default" in order_lines[0]


def test_route_logs_env_override_source(monkeypatch, lease_store, caplog):
    import logging

    monkeypatch.setenv(ENV_AUTO_LANE_ORDER, "nibi,gcp")
    nibi = _FreeLaneBackend(kind="nibi", est_start_raw=0.0)
    with caplog.at_level(logging.INFO, logger="explore_persona_space.backends.router"):
        route(
            _spec(backend=None),
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: True,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    order_lines = [r.message for r in caplog.records if "auto lane order" in r.message]
    assert order_lines
    assert "nibi -> gcp" in order_lines[0]
    assert ENV_AUTO_LANE_ORDER in order_lines[0]


def test_no_auto_runpod_under_gcp_first_default(lease_store):
    """The load-bearing real-money invariant holds under the NEW default
    order too: GCP capacity-fails in primary position, every free lane
    fails, and RunPod is STILL never called — the chain ends in the
    typed NoComputeAvailableError."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError("QUOTA_EXCEEDED", evidence={"matched_pattern": "Q"})
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
    outcomes = [(a["kind"], a["outcome"]) for a in excinfo.value.attempts]
    assert ("gcp", "provisioning_failure") in outcomes
    assert any(kind == "nibi" for kind, _o in outcomes)


# ---------------------------------------------------------------------------
# issue #588 — spec_hash continuity for hydra-only specs
# ---------------------------------------------------------------------------


def test_spec_hash_hydra_only_matches_pre_change_recorded_hash() -> None:
    """A2 (#588): hydra-only specs must hash identically across the
    workload_cmd upgrade (lease reconnect continuity — a changed hash
    would orphan every in-flight lease).

    The recorded hash was generated from the PRE-change
    canonicalize_spec at the issue-588 merge-base (provenance in the
    fixture's JSON header).
    """
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "issue588_spec_hash_hydra_only.json").read_text()
    )
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="auto",
        hydra_args=("condition=c1_evil_wrong_em", "seed=42"),
    )
    assert spec_hash(spec) == fixture["spec_hash"]


def test_spec_hash_differs_between_hydra_and_custom_specs() -> None:
    """#588: a custom-cmd run for the same issue is a DISTINCT lease key
    (reconnect must not glue a custom dispatch onto a hydra lease), and
    the key is emitted only when non-empty (bare specs unchanged)."""
    hydra = RunSpec(issue=137, intent="lora-7b", backend="auto", hydra_args=("seed=42",))
    custom = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="auto",
        workload_cmd="bash scripts/issue588_smoke.sh",
    )
    bare = RunSpec(issue=137, intent="lora-7b", backend="auto")
    assert spec_hash(hydra) != spec_hash(custom)
    assert spec_hash(bare) != spec_hash(custom)
    assert "workload_cmd" not in canonicalize_spec(bare)
    assert "workload_cmd" not in canonicalize_spec(hydra)
    assert canonicalize_spec(custom)["workload_cmd"] == "bash scripts/issue588_smoke.sh"


def test_auto_route_workload_cmd_spec_walks_gcp_first_identically(lease_store) -> None:
    """#588: ``route()`` never introspects the workload shape — a
    workload_cmd spec walks the same GCP-first auto chain as a hydra
    spec, RunPod untouched, and the spec reaches the lane verbatim."""
    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble()
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="auto",
        workload_cmd="bash scripts/issue588_smoke.sh",
    )
    result = route(
        spec,
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        is_live_after_cancel=lambda _b, _h: False,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    # GCP-first standing default resolves at GCP; the free lane is never
    # touched and RunPod (exploding double) is provably never called.
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    assert gcp.launches[0].workload_cmd == "bash scripts/issue588_smoke.sh"
    assert len(nibi.launches) == 0
