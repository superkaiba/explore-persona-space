"""Router (`backends.router.route`) tests.

Slice-5 surface coverage: decision table, override paths, auto chain,
park-watchdog state machine, cancel state machine, durable lease +
reconnect, GCP attempt-count guard, marker registration.

#656 REVERSED the no-auto-RunPod invariant: the auto chain now reaches
RunPod as the documented TERMINAL rung, after the cost-ordered GCP ladder
(on-demand A100-80 → A100-40 → spot) AND the free SLURM lanes are all
exhausted. The load-bearing safeguard is now an ORDERING property — RunPod
is reached ONLY last, never skipping a cheaper rung — pinned by
``test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted``. The two
manual-attention paths still raise WITHOUT touching RunPod (an
unconfirmed-dead orphaned free-lane job must never trigger a second submit).

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
    ENV_SPOT_MAX_GPU_HOURS,
    FREE_WAIT_SECONDS,
    MAX_GCP_ATTEMPTS_PER_DAY,
    ROUTE_REASON_AUTO_FALLBACK_GCP,
    ROUTE_REASON_AUTO_STARTED,
    ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
    ROUTE_REASON_OVERRIDE,
    ROUTE_REASON_RECONNECT,
    ROUTE_REASON_RUNPOD_FALLBACK,
    auto_lane_order,
    cancel_and_wait,
    park_until_running_or_cap,
)
from explore_persona_space.backends.router import RouteAttempt as RouteAttempt
from explore_persona_space.backends.router import _attempt_to_dict as _attempt_to_dict
from explore_persona_space.backends.router import _gcp_ladder_specs as _ladder_specs
from explore_persona_space.backends.router import _thread_attempt_id_into as _thread_attempt_id_into

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


def _gcp_rung_key(spec: RunSpec) -> str:
    """Resolve a ``<gpu_kind>/<provisioning>`` key for the #656 ladder doubles.

    Mirrors how the production headroom pre-check + create resolve the rung's
    true machine: ``machine_for_intent`` consults the
    ``machine_spec_override`` the router threads, and
    ``resolve_provisioning_model`` reads ``extra["provisioning_model"]``. So a
    ladder rung-spec keyed on A100-80 vs A100-40 (and STANDARD vs SPOT) is
    distinguishable in the test double exactly as the real backend
    distinguishes it.
    """
    from explore_persona_space.backends.gcp import machine_for_intent, resolve_provisioning_model

    machine = machine_for_intent(spec)
    return f"{machine.gpu_kind}/{resolve_provisioning_model(spec)}"


def _gcp_rung_key_wide(spec: RunSpec) -> str:
    """Width-qualified rung key (#1121): ``<gpu_kind>x<gpu_count>/<provisioning>``.

    The #1121 wide rungs share the legacy key's ``gpu_kind`` (``A100-80``)
    with the base rungs, so per-WIDTH miss/admit scripting needs the resolved
    machine's ``gpu_count`` in the key (``A100-80x8/FLEX_START``). The doubles
    consult THIS key first and FALL BACK to the legacy width-blind key, so
    every pre-#1121 test stays byte-unmodified.
    """
    from explore_persona_space.backends.gcp import machine_for_intent, resolve_provisioning_model

    machine = machine_for_intent(spec)
    return f"{machine.gpu_kind}x{machine.gpu_count}/{resolve_provisioning_model(spec)}"


class _GcpBackendDouble(_BaseBackend):
    """GCP backend double.

    Knobs:
    * ``launch_raises`` — set to a ``GcpProvisioningError`` or
      ``GcpWorkloadError`` to test the failure classification paths
      (applies to EVERY launch regardless of rung).
    * ``launch_raises_by_rung`` — optional per-rung override keyed on the
      resolved ``<gpu_kind>/<provisioning>`` (e.g.
      ``{"A100-80/STANDARD": GcpProvisioningError(...)}``) for the #656
      ladder tests where one rung capacity-misses and a later rung
      succeeds. Falls back to ``launch_raises`` for an unkeyed rung.
    * ``quota_headroom`` — scripted ``preflight_quota_headroom`` reading
      (a ``QuotaHeadroom``, ``None`` for "no opinion", or an exception
      instance to raise — the router must fail OPEN on it). Defaults to
      ``None`` so every pre-existing test proceeds exactly as before.
    * ``quota_headroom_by_provisioning`` — optional per-provisioning-model
      override (e.g. ``{"STANDARD": <insufficient>, "SPOT": <sufficient>}``).
      When set, the probe resolves the spec's provisioning model and returns
      the matching reading.
    * ``quota_headroom_by_rung`` — optional per-rung override keyed on the
      resolved ``<gpu_kind>/<provisioning>`` (#656 ladder tests where
      A100-80 is insufficient AND A100-40 is sufficient at the SAME
      provisioning model). Takes precedence over
      ``quota_headroom_by_provisioning`` and ``quota_headroom``.
    """

    def __init__(
        self,
        *,
        launch_raises: BaseException | None = None,
        launch_raises_by_rung: dict[str, BaseException] | None = None,
        quota_headroom: QuotaHeadroom | BaseException | None = None,
        quota_headroom_by_provisioning: dict[str, QuotaHeadroom | None] | None = None,
        quota_headroom_by_rung: dict[str, QuotaHeadroom | None] | None = None,
    ) -> None:
        self._launch_raises = launch_raises
        self._launch_raises_by_rung = launch_raises_by_rung
        self._quota_headroom = quota_headroom
        self._quota_headroom_by_provisioning = quota_headroom_by_provisioning
        self._quota_headroom_by_rung = quota_headroom_by_rung
        self.launches: list[RunSpec] = []
        self.quota_probes: list[RunSpec] = []

    def preflight_quota_headroom(self, spec: RunSpec) -> QuotaHeadroom | None:
        self.quota_probes.append(spec)
        if self._quota_headroom_by_rung is not None:
            # #1121: width-qualified key first, legacy width-blind key second
            # (existing tests key only the legacy form and stay unmodified).
            wide_key = _gcp_rung_key_wide(spec)
            if wide_key in self._quota_headroom_by_rung:
                return self._quota_headroom_by_rung[wide_key]
            key = _gcp_rung_key(spec)
            if key in self._quota_headroom_by_rung:
                return self._quota_headroom_by_rung[key]
        if self._quota_headroom_by_provisioning is not None:
            from explore_persona_space.backends.gcp import resolve_provisioning_model

            provisioning = resolve_provisioning_model(spec)
            if provisioning in self._quota_headroom_by_provisioning:
                return self._quota_headroom_by_provisioning[provisioning]
        if isinstance(self._quota_headroom, BaseException):
            raise self._quota_headroom
        return self._quota_headroom

    @property
    def name(self) -> BackendKind:
        return "gcp"

    def launch(self, spec: RunSpec) -> RunHandle:
        if self._launch_raises_by_rung is not None:
            # #1121: width-qualified key first, legacy width-blind key second.
            wide_key = _gcp_rung_key_wide(spec)
            if wide_key in self._launch_raises_by_rung:
                raise self._launch_raises_by_rung[wide_key]
            key = _gcp_rung_key(spec)
            if key in self._launch_raises_by_rung:
                raise self._launch_raises_by_rung[key]
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


def _short_lora_spec(issue: int = 137) -> RunSpec:
    """A short (1 GPU-h) auto-routing lora-7b spec — all four ladder rungs apply."""
    return RunSpec(issue=issue, intent="lora-7b", backend="auto", time_budget_hours=1.0)


def test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted(
    lease_store, marker_poster, captured_markers
):
    """#656 REPLACES ``test_no_auto_runpod_path_under_any_failure``.

    The reversed invariant: RunPod is reached on the auto chain ONLY after
    EVERY cheaper GCP rung AND the free SLURM lane have failed — never first,
    never skipping a cheaper rung. We inject a ``_PassiveRunpod`` that records
    launches; the assertion is that RunPod launches EXACTLY ONCE and its
    attempt is LAST in the trail, behind every GCP rung outcome and the
    nibi park-fail. #680: a short lora-7b now walks 5 GCP rungs (spot A100-80,
    spot A100-40, flex-start A100-80, on-demand A100-80, on-demand A100-40).
    """
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    # Every GCP rung is doomed: A100-80 + A100-40, STANDARD + SPOT, all
    # capacity-miss on the create (a runtime miss, not a headroom skip — so
    # we exercise the create-failure advance path on every rung).
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=99,  # don't let the cap mask the ladder walk
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_RUNPOD_FALLBACK
    assert len(rp.launches) == 1
    # Every GCP rung was attempted (5 ladder rungs, all provisioning_failure),
    # the nibi lane failed, and RunPod is the FINAL attempt.
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    gcp_fail_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "gcp"]
    nibi_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "nibi"]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert len(gcp_fail_idxs) == 5  # all five ladder rungs attempted + failed
    assert nibi_idxs, "the free SLURM lane must have been attempted"
    assert runpod_idxs and runpod_idxs[-1] == len(outcomes) - 1  # runpod LAST
    assert max(gcp_fail_idxs) < runpod_idxs[-1]
    assert max(nibi_idxs) < runpod_idxs[-1]
    # The residual-gap marker names the exhausted lanes.
    finals = _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)
    assert finals
    assert "runpod_fallback_residual_gap" in finals[-1]["extra"]


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


def test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade(
    lease_store, marker_poster, captured_markers
):
    """Task #658 (REVERSES the pre-#658 ``no_fallback`` invariant): under
    the GCP-first standing default GCP runs in PRIMARY position; a GCP
    WORKLOAD failure now FAILS OVER TO RUNPOD — straight to the RunPod
    terminal rung, NOT cascading through the SLURM lanes (re-crashing
    broken code there burns queue time). RunPod pods persist + are
    SSH-able, the diagnosis surface GCP's delete-on-crash boot disk cannot
    give."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpWorkloadError("entrypoint crashed", evidence={"exit_code": 1})
    )
    result = route(
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
    # Failed over to RunPod (launched exactly once), labeled as the
    # workload-failover reason (distinct from the capacity-exhaustion
    # ``auto_fallback_runpod``).
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD
    assert len(rp.launches) == 1
    # No SLURM cascade — the workload failure short-circuited to RunPod.
    assert len(nibi.launches) == 0
    # The GcpWorkloadError evidence rides the failover marker for diagnosis.
    finals = _by_reason(captured_markers, ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD)
    assert finals
    assert finals[-1]["extra"].get("gcp_workload_evidence", {}).get("exit_code") == 1
    # The GCP rung's attempt is recorded as a workload_failure that failed over.
    gcp_workload = [
        a for a in result.attempts if a.kind == "gcp" and a.outcome == "workload_failure"
    ]
    assert gcp_workload and "failing over to RunPod" in (gcp_workload[-1].detail or "")


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


# ---------------------------------------------------------------------------
# Fresh attempt-id per launch (#927) — _thread_attempt_id_into mints per call;
# reconnect keeps the original id (router early-return + GCP label recovery).
# ---------------------------------------------------------------------------

_ATTEMPT_ID_RE = re.compile(r"att-\d{8}-\d{6}")


def test_thread_attempt_id_into_mints_fresh_id_when_lease_exists():
    """#927 acceptance (1), unit level: a pre-existing lease's attempt_id is
    NOT reused — a fresh ``att-YYYYMMDD-HHMMSS`` id is minted, threaded into
    the spec, AND written back to the lease (lease follows the launch)."""
    spec = _spec(backend=None)
    lease = Lease(issue=137, spec_hash="h", attempt_id="att-old")
    writes: list[Lease] = []
    new_spec, new_lease = _thread_attempt_id_into(spec, lease, writes.append)
    fresh = new_spec.extra["attempt_id"]
    assert fresh != "att-old"
    assert _ATTEMPT_ID_RE.fullmatch(fresh), fresh
    # Lease-follows-launch: the lease records the id the launch will use,
    # and the write happened inside the caller's transaction.
    assert new_lease.attempt_id == fresh
    assert writes and writes[-1].attempt_id == fresh


def test_thread_attempt_id_into_preserves_counters_and_failover_fields():
    """#927 acceptance (4): only ``attempt_id`` changes across the threading
    write — GCP attempt counters, failover-identity fields, backend, and
    job_id all survive. Called TWICE on the same lease (the per-rung ladder
    shape: each rung re-threads the ORIGINAL spec inside one transaction;
    per-rung re-mint churn is expected behavior) — fields survive BOTH."""
    today = datetime.now(tz=UTC).date().isoformat()
    lease = Lease(
        issue=137,
        spec_hash="h",
        attempt_id="att-old",
        backend="gcp",
        cluster=None,
        job_id="instance-1",
        gcp_attempts_today=3,
        gcp_attempts_date=today,
        gcp_failover_of={"pod_name": "eps-issue-1", "job_id": "z"},
        runpod_wedge_failover_of={"pod_name": "pod-1", "job_id": "w"},
        runpod_cuda_ima_failover_of={"pod_name": "pod-1", "job_id": "c"},
    )
    spec = _spec(backend=None)
    writes: list[Lease] = []
    _s1, l1 = _thread_attempt_id_into(spec, lease, writes.append)
    _s2, l2 = _thread_attempt_id_into(spec, l1, writes.append)
    assert len(writes) == 2  # one lease write per threading call
    for le in (l1, l2):
        assert le.gcp_attempts_today == 3
        assert le.gcp_attempts_date == today
        assert le.gcp_failover_of == {"pod_name": "eps-issue-1", "job_id": "z"}
        assert le.runpod_wedge_failover_of == {"pod_name": "pod-1", "job_id": "w"}
        assert le.runpod_cuda_ima_failover_of == {"pod_name": "pod-1", "job_id": "c"}
        assert le.backend == "gcp"
        assert le.job_id == "instance-1"
        assert le.attempt_id != "att-old"


def test_thread_attempt_id_into_honors_caller_pinned_extra_id():
    """A caller-pinned ``spec.extra["attempt_id"]`` wins over the fresh mint
    — even when a lease exists (explicit re-attach tooling ONLY; see the
    function docstring for why routine pinning re-creates #825) — and is
    written to the lease verbatim."""
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", extra={"attempt_id": "att-pin"})
    lease = Lease(issue=137, spec_hash="h", attempt_id="att-old")
    writes: list[Lease] = []
    new_spec, new_lease = _thread_attempt_id_into(spec, lease, writes.append)
    assert new_spec.extra["attempt_id"] == "att-pin"
    assert new_lease.attempt_id == "att-pin"
    assert writes and writes[-1].attempt_id == "att-pin"


def test_thread_attempt_id_into_creates_lease_when_absent():
    """Regression pin: the lease-None branch keeps its pre-#927 behavior —
    a fresh lease is created with the minted id and written."""
    spec = _spec(backend=None)
    writes: list[Lease] = []
    new_spec, lease = _thread_attempt_id_into(spec, None, writes.append)
    aid = new_spec.extra["attempt_id"]
    assert _ATTEMPT_ID_RE.fullmatch(aid), aid
    assert lease.issue == 137
    assert lease.spec_hash == spec_hash(spec)
    assert lease.attempt_id == aid
    assert writes and writes[0] is lease


def test_fresh_gcp_launch_after_dead_prior_attempt_uses_new_attempt_id(
    lease_store, marker_poster, captured_markers
):
    """#927 acceptance (1), route level: a fresh GCP launch with a
    pre-existing lease carrying a dead prior attempt's id launches with a
    DIFFERENT attempt_id, and the persisted lease reads back the NEW id
    (the #825 shape: three relaunches all inherited the dead attempt's
    crash-persist / sentinel namespace)."""
    gcp = _GcpBackendDouble()
    stale_spec = _spec(backend=None)
    lease_store.write(
        Lease(
            issue=137,
            spec_hash=spec_hash(stale_spec),  # same shape — the lease is NOT stale-keyed away
            attempt_id="att-dead-1",
            backend="gcp",
            job_id="instance-dead",
        )
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        # reconnect_fn omitted → reconnect disabled (the dead instance is gone).
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    launched_id = gcp.launches[0].extra["attempt_id"]
    assert launched_id != "att-dead-1"
    assert _ATTEMPT_ID_RE.fullmatch(launched_id), launched_id
    read_back = lease_store.read(137)
    assert read_back is not None
    assert read_back.attempt_id == launched_id  # lease follows the launch


def test_reconnect_keeps_prior_attempt_id_no_remint(lease_store, marker_poster, captured_markers):
    """#927 acceptance (2): a reconnect to a still-live instance early-returns
    with ``ROUTE_REASON_RECONNECT`` — no launch, no threading call — and the
    lease's attempt_id stays byte-unchanged."""
    gcp = _GcpBackendDouble()
    live_spec = _spec(backend=None)
    lease_store.write(
        Lease(
            issue=137,
            spec_hash=spec_hash(live_spec),
            attempt_id="att-orig",
            backend="gcp",
            job_id="instance-live",
        )
    )
    live = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-live",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"issue": 137, "zone": "us-central1-a", "attempt_id": "att-orig"},
    )
    result = route(
        _spec(backend=None),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        reconnect_fn=lambda _b, k, _s: live if k == "gcp" else None,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.reason == ROUTE_REASON_RECONNECT
    assert gcp.launches == []  # reconnect — never a fresh provision
    read_back = lease_store.read(137)
    assert read_back is not None
    assert read_back.attempt_id == "att-orig"  # byte-unchanged: no re-mint on reconnect


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


def test_gcp_marker_extras_unmapped_intent_returns_none_quota_pool():
    """#631 round-4: ``_gcp_marker_extras`` must not raise on an intent
    that has no ``INTENT_TO_MACHINE`` row.

    Observability code resolves ``machine_for_intent(spec)`` to look up the
    quota pool; ``machine_for_intent`` raises ``ValueError`` on an unmapped
    intent. On a reconnect path that found the live instance by NAME only,
    ``spec.intent`` can be unmapped (e.g. ``ft-70b``), so the helper must
    degrade ``quota_pool`` to ``None`` rather than crash a successful run.
    ``provisioning_model`` reads only ``spec.extra`` (no intent lookup) so
    it stays populated.
    """
    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE
    from explore_persona_space.backends.router import _gcp_marker_extras

    # Guard: the intent under test must genuinely be unmapped, so this test
    # keeps testing the degrade path even if new intents are added later.
    assert "ft-70b" not in INTENT_TO_MACHINE

    extras = _gcp_marker_extras(RunSpec(issue=137, intent="ft-70b", backend="auto"))
    assert extras["quota_pool"] is None
    assert extras["provisioning_model"]  # non-empty / non-None


def test_gcp_reconnect_marker_unmapped_intent_does_not_crash(
    lease_store, marker_poster, captured_markers
):
    """#631 round-4: a successful GCP reconnect with an unmapped intent must
    NOT crash on the ``epm:backend-selected`` marker's quota-pool lookup.

    Drives the auto-chain reconnect path (``reconnect_fn`` returns a live
    GCP handle) with ``intent="ft-70b"`` (no ``INTENT_TO_MACHINE`` row).
    ``route()`` must return ``chosen_kind == "gcp"`` and the posted marker's
    ``extra`` must carry a populated ``provisioning_model`` and a ``None``
    ``quota_pool`` (the degrade path) — not raise ``ValueError``.
    """
    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE

    assert "ft-70b" not in INTENT_TO_MACHINE

    rp = _ExplodingRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble()

    existing = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-existing-ft70b",
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
        RunSpec(issue=137, intent="ft-70b", backend="auto"),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
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
    assert len(gcp.launches) == 0  # reconnected, never provisioned

    bodies = _by_reason(captured_markers, ROUTE_REASON_RECONNECT)
    assert bodies, "expected an epm:backend-selected reconnect marker"
    extra = bodies[-1]["extra"]
    assert extra["provisioning_model"]  # populated despite unmapped intent
    assert extra["quota_pool"] is None  # degraded, not a crash


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
    """#656: at the per-day attempt cap, the GCP ladder issues NO new create
    and the chain falls through to the RunPod terminal rung (the cap no
    longer RAISES GcpAttemptCapExceededError — RunPod is the new tail). The
    cap still bounds GCP creates: ``gcp.launches`` stays empty."""
    rp = _PassiveRunpod()
    cfg = RouterConfig(
        free_wait_seconds=1,
        poll_interval=0.0,
        cancel_grace_seconds=0,
        max_gcp_attempts_per_day=2,
        # Legacy free-first order: GCP sits LAST (terminal escalation
        # position). Pre-#656 a cap-trip there RAISED; now it falls through
        # to RunPod.
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

    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble()  # would succeed if reached — but the cap blocks it
    result = route(
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
    assert result.chosen_kind == "runpod"
    assert len(gcp.launches) == 0  # cap bounded GCP creates
    assert any(a.outcome == "attempt_cap_exceeded" for a in result.attempts)


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
    """A SLURM-lane ``WorkloadSurfacedError`` path posts a workload_failure
    marker before raising.

    Repointed from the GCP lane to the SLURM lane (task #658): a GCP
    workload failure now FAILS OVER to RunPod (see
    ``test_gcp_workload_error_fails_over_to_runpod_no_slurm_cascade``)
    instead of surfacing, so the surviving ``WorkloadSurfacedError`` +
    terminal-``workload_failure``-marker path is the SLURM
    ``terminal_before_running``-with-artifacts case (a started-then-failed
    job on the lane the user explicitly asked for)."""
    from explore_persona_space.backends.router import ROUTE_REASON_WORKLOAD_FAILURE

    nibi = _FreeLaneBackend(kind="nibi", poll_status="dead")
    with pytest.raises(WorkloadSurfacedError):
        route(
            _spec(backend="nibi"),  # explicit lane → no failover, surfaces
            runpod_backend=_ExplodingRunpod(),
            free_backends={"nibi": nibi},
            lease_store=lease_store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            started_evidence_probe=lambda _b, _h: dict(_EVIDENCE),  # started-then-failed
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
    """#656 fan-out: for every no-compute failure mode, the auto chain falls
    through to the RunPod TERMINAL rung (the reversed invariant) — EXCEPT the
    two deliberate non-fallback paths (``manual_attention_cancel`` /
    ``is_live_raises``), which still RAISE ManualAttentionRequiredError
    BEFORE the chain reaches RunPod (an unconfirmed-dead orphaned free-lane
    job must never trigger a second submit anywhere).

    No-compute scenarios use a recording ``_PassiveRunpod`` and assert
    ``chosen_kind == "runpod"``; the two manual-attention scenarios use an
    ``_ExplodingRunpod`` and assert the raise — proving RunPod is NOT reached
    on those paths.
    """
    # The two manual-attention paths must NEVER reach RunPod → exploding
    # double proves it. Every other path falls through to RunPod → passive
    # double records the terminal launch.
    manual_attention_scenarios = {"is_live_raises", "manual_attention_cancel"}
    rp: ComputeBackend = (
        _ExplodingRunpod() if scenario in manual_attention_scenarios else _PassiveRunpod()
    )
    cfg = RouterConfig(
        free_wait_seconds=1,
        poll_interval=0.0,
        cancel_grace_seconds=0,
        max_gcp_attempts_per_day=2,
        # Legacy free-first order: GCP sits LAST (the terminal escalation
        # position), so the whole free→GCP→RunPod chain is exercised.
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
        expected: type[BaseException] | str = "runpod"
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
        expected = "runpod"
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
        expected = "runpod"
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
        expected = "runpod"
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
        expected = "runpod"
    else:  # pragma: no cover — pytest.mark.parametrize wall.
        raise AssertionError(f"unknown scenario: {scenario}")

    if expected == "runpod":
        # #656: the no-compute fan-out now falls through to the RunPod
        # terminal rung (passive double records the launch).
        result = route(_spec(backend=None), **kwargs)
        assert result.chosen_kind == "runpod"
        assert result.reason == ROUTE_REASON_RUNPOD_FALLBACK
        assert len(rp.launches) == 1  # type: ignore[attr-defined]
    else:
        # The two manual-attention paths still RAISE before reaching RunPod
        # (exploding double proves RunPod is never touched).
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
    """#656: the cap-hit attempt detail reports the cap value (not cap+1) and
    the chain falls through to RunPod (the cap no longer raises
    GcpAttemptCapExceededError)."""
    rp = _PassiveRunpod()
    cfg = RouterConfig(
        free_wait_seconds=1,
        poll_interval=0.0,
        cancel_grace_seconds=0,
        max_gcp_attempts_per_day=2,
        lane_order=_LEGACY_FREE_FIRST_ORDER,  # GCP last (terminal position)
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
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble()
    result = route(
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
    assert result.chosen_kind == "runpod"
    cap_attempt = next(a for a in result.attempts if a.outcome == "attempt_cap_exceeded")
    # The detail reports the cap value (2), not cap+1.
    assert f"cap {cfg.max_gcp_attempts_per_day}" in cap_attempt.detail


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


def test_gcp_prepare_failure_after_free_lanes_fail_falls_to_runpod(lease_store):
    """#656: a GCP prepare failure on every applicable rung + a never-starting
    free lane falls through to the RunPod terminal rung (no longer a hard
    NoComputeAvailableError). #680: the nobudget lora-7b is unknown-length =>
    the long branch (flex-80 -> ondemand-80 -> ondemand-40, NO spot), so
    prepare is attempted on all THREE before the fall-through."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9, est_start_raw=0.0)
    gcp = _PrepareRecordingGcp(prepare_raises=RuntimeError("metadata render failed"))
    result = route(
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
    assert result.chosen_kind == "runpod"
    # prepare attempted on all three long-branch rungs (flex-80, ondemand-80,
    # ondemand-40), NO spot rung (unknown length).
    assert gcp.calls == ["prepare", "prepare", "prepare"]
    assert any(a.outcome == "prepare_failed" for a in result.attempts)


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


# ---------------------------------------------------------------------------
# #656 — GCP cost-ordered fallback ladder (on-demand A100-80 → A100-40 →
# SPOT → RunPod terminal rung). These REPLACE the four #537
# EPS_GCP_SPOT_FALLBACK tests (the env-gated SPOT fallback is subsumed by the
# ladder: a SPOT rung now fires by DEFAULT for any short job).
# ---------------------------------------------------------------------------

#: A100-80 STANDARD insufficient (1 needed, 0 free).
_A100_80_STD_SHORT = QuotaHeadroom(
    metric="NVIDIA_A100_80GB_GPUS", region="us-central1", limit=8.0, usage=8.0, needed=1
)
#: A100-40 STANDARD ample (8 free).
_A100_40_STD_AMPLE = QuotaHeadroom(
    metric="NVIDIA_A100_GPUS", region="us-central1", limit=8.0, usage=0.0, needed=1
)
#: A100-40 STANDARD insufficient.
_A100_40_STD_SHORT = QuotaHeadroom(
    metric="NVIDIA_A100_GPUS", region="us-central1", limit=8.0, usage=8.0, needed=1
)
#: A100-80 SPOT ample (8 free).
_A100_80_SPOT_AMPLE = QuotaHeadroom(
    metric="PREEMPTIBLE_NVIDIA_A100_80GB_GPUS",
    region="us-central1",
    limit=16.0,
    usage=8.0,
    needed=1,
)


def test_ladder_a100_80_full_long_lora_routes_to_a100_40(
    lease_store, marker_poster, captured_markers
):
    """T1 (#680: re-scoped to a LONG lora-7b so the on-demand rungs lead — a
    short job would land on the spot rung first under the new spot-first
    order). A100-80 FLEX_START + A100-80 STANDARD both insufficient, A100-40
    STANDARD sufficient → lands on the smaller-GPU A100-40 rescue, NOT a hard
    block, and RunPod (exploding) is never called."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/FLEX_START": _A100_80_STD_SHORT,
            "A100-80/STANDARD": _A100_80_STD_SHORT,
            "A100-40/STANDARD": _A100_40_STD_AMPLE,
        }
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", time_budget_hours=10.0)  # long
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": _FreeLaneBackend(kind="nibi")},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    assert gcp.launches[0].extra["machine_kind_tag"] == "A100-40"
    assert gcp.launches[0].extra["provisioning_model"] == "STANDARD"
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    # flex-80 + ondemand-80 headroom-skip BEFORE the A100-40 launch; NO spot rung.
    assert ("gcp", "quota_headroom_insufficient") in outcomes
    assert ("gcp", "launched") in outcomes
    assert not any("spot" in (a.detail or "") for a in result.attempts if a.kind == "gcp")
    assert result.extra["gcp_ladder_rung"] == "ondemand_a100_40"


def test_ladder_short_job_spot_leads(lease_store, marker_poster, captured_markers):
    """T2 (#680: spot now LEADS for a short job). A100-80 SPOT sufficient →
    lands on SPOT A100-80 as the FIRST rung, with no on-demand headroom-skip
    preamble (the on-demand rungs are never reached on a clean spot land)."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/SPOT": _A100_80_SPOT_AMPLE,
        }
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": _FreeLaneBackend(kind="nibi")},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: True,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert gcp.launches[0].extra["provisioning_model"] == "SPOT"
    assert gcp.launches[0].extra["machine_kind_tag"] == "A100-80"
    assert result.extra["gcp_ladder_rung"] == "spot_a100_80"
    # Spot is the FIRST GCP attempt — no on-demand rung was tried first.
    assert not any("ondemand" in (a.detail or "") for a in result.attempts if a.kind == "gcp")


def test_ladder_long_job_a100_80_full_a100_40_unusable_routes_to_runpod_skipping_spot(
    lease_store, marker_poster, captured_markers
):
    """T3: a long ft-7b (no A100-40 fallback, not short) with both GCP rungs
    full → RunPod, skipping every spot rung (none exist for ft-7b). #680: the
    long branch is flex-80 -> ondemand-80, so TWO provisioning failures."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    spec = RunSpec(issue=137, intent="ft-7b", backend="auto")  # long, no A100-40 fallback
    result = route(
        spec,
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    assert not any("spot" in (a.detail or "") for a in result.attempts if a.kind == "gcp"), (
        "no spot rung for a long / no-fit job"
    )
    # Two GCP provisioning failures: flex-start A100-80 then on-demand A100-80.
    assert sum(1 for k, o in outcomes if k == "gcp" and o == "provisioning_failure") == 2
    finals = _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)
    assert finals and "runpod_fallback_residual_gap" in finals[-1]["extra"]


def test_ladder_long_lora_over_threshold_skips_spot_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """T3b: a long lora-7b (10 GPU-h > 2 threshold) with all GCP rungs full →
    the A100-40 rung IS attempted (fits 40GB) but spot rungs are ABSENT (over
    threshold) → RunPod. #680: the long branch is flex-80 -> ondemand-80 ->
    ondemand-40, so THREE provisioning failures."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "QUOTA_EXCEEDED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", time_budget_hours=10.0)
    result = route(
        spec,
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    # Three rung failures (flex-80, ondemand-80, ondemand-40), NO spot rung.
    gcp_fails = [
        a for a in result.attempts if a.kind == "gcp" and a.outcome == "provisioning_failure"
    ]
    assert len(gcp_fails) == 3
    assert not any("spot" in (a.detail or "") for a in gcp_fails)


def test_explicit_gcp_pin_gets_full_ladder(lease_store, marker_poster, captured_markers):
    """T4 (#680: re-scoped to a LONG lora-7b so the on-demand A100-40 landing
    is preserved — a short pin would land on the spot rung first). An explicit
    ``backend: gcp`` pin with flex-80 + ondemand-80 full + A100-40 sufficient
    lands on A100-40 — NOT a hard NoComputeAvailableError (the #654
    regression). Same ladder as auto."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/FLEX_START": _A100_80_STD_SHORT,
            "A100-80/STANDARD": _A100_80_STD_SHORT,
            "A100-40/STANDARD": _A100_40_STD_AMPLE,
        }
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="gcp", time_budget_hours=10.0)  # long
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.requested_kind == "gcp"  # the explicit pin
    assert gcp.launches[0].extra["machine_kind_tag"] == "A100-40"


def test_explicit_gcp_pin_full_ladder_exhausted_falls_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """T4b: an explicit gcp pin with EVERY rung insufficient falls through to
    the RunPod terminal rung (never a hard block)."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "QUOTA_EXCEEDED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="gcp", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_RUNPOD_FALLBACK
    assert len(rp.launches) == 1
    finals = _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)
    assert finals and "explicit gcp pin" in finals[-1]["extra"]["runpod_fallback_residual_gap"]


def test_explicit_gcp_pin_is_cap_exempt(lease_store, marker_poster, captured_markers):
    """REGRESSION (#656): the explicit ``backend: gcp`` pin is EXEMPT from the
    per-day GCP attempt cap — an explicit user ask attempts regardless of the
    auto-escalation counter (the pre-#656 explicit-gcp path never touched it).
    Pre-seed the lease AT the cap; the explicit pin STILL launches on GCP and
    the counter is NOT bumped further.

    Without the ``count_attempt_cap=False`` guard on the explicit-gcp path,
    this falls through to RunPod instead (the regression that broke
    test_launch_spot_tolerant_threads_to_spec_extra when the shared CLI lease
    accumulated 5 auto attempts)."""
    today = datetime.now(tz=UTC).date().isoformat()
    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_attempts_today=5,  # AT the default cap
            gcp_attempts_date=today,
        )
    )
    gcp = _GcpBackendDouble()  # would launch the first rung (spot A100-80) if reached
    spec = RunSpec(issue=137, intent="lora-7b", backend="gcp", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),  # RunPod must NOT be reached
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=5),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    # The explicit pin launched on GCP despite the cap being reached.
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    # #680: a short job leads with the spot A100-80 rung under the new order.
    assert result.extra["gcp_ladder_rung"] == "spot_a100_80"
    # The per-day counter was NOT bumped past the cap (cap-exempt path).
    lease = lease_store.read(137)
    assert lease is not None and lease.gcp_attempts_today == 5
    # No cap-exceeded attempt was recorded.
    assert not any(a.outcome == "attempt_cap_exceeded" for a in result.attempts)


def test_gcp_provisioning_error_capacity_miss_advances_ladder(
    lease_store, marker_poster, captured_markers
):
    """T5b (#680: re-scoped to a LONG lora-7b so the ladder is
    flex-80 -> ondemand-80 -> ondemand-40 with no spot rung). RUNTIME capacity
    misses (GcpProvisioningError on the create, not a headroom skip) on the
    flex-80 + ondemand-80 rungs advance to A100-40, which succeeds — the ladder
    handles BOTH the headroom-skip path and the create-failure path."""
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80/FLEX_START": GcpProvisioningError(
                "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
            ),
            "A100-80/STANDARD": GcpProvisioningError(
                "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
            ),
        }
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", time_budget_hours=10.0)  # long
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "ondemand_a100_40"
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    # flex-80 + ondemand-80 creates FAILED, THEN A100-40 launched; no spot rung.
    assert ("gcp", "provisioning_failure") in outcomes
    assert ("gcp", "launched") in outcomes
    assert not any("spot" in (a.detail or "") for a in result.attempts if a.kind == "gcp")


# ---------------------------------------------------------------------------
# #680: length-aware spot-first (short) / flex-first (long) ladder
# ---------------------------------------------------------------------------


def test_ladder_short_job_spot_before_ondemand(lease_store, marker_poster, captured_markers):
    """#680 MF3 (headline form): a short lora-7b with ample spot lands on the
    spot A100-80 rung — the FIRST GCP attempt label is ``spot_a100_80`` (flex /
    on-demand are never attempted on a clean spot land, so there is no later
    rung to index-compare against here)."""
    gcp = _GcpBackendDouble(quota_headroom_by_rung={"A100-80/SPOT": _A100_80_SPOT_AMPLE})
    result = route(
        _short_lora_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "spot_a100_80"
    first_gcp = next(a for a in result.attempts if a.kind == "gcp")
    assert "rung spot_a100_80" in (first_gcp.detail or "")


def test_ladder_short_job_spot_miss_then_ondemand_order(
    lease_store, marker_poster, captured_markers
):
    """#680 MF3 (sister form): the spot + flex rungs capacity-miss so on-demand
    IS attempted; assert spot is attempted BEFORE on-demand in the trail (the
    flex rung sits between them, so it too must miss for on-demand to be
    reached and the index comparison to be well-defined)."""
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80/SPOT": GcpProvisioningError("spot OUT", evidence={}),
            "A100-40/SPOT": GcpProvisioningError("spot40 OUT", evidence={}),
            "A100-80/FLEX_START": GcpProvisioningError("flex OUT", evidence={}),
        }
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    details = [a.detail or "" for a in result.attempts if a.kind == "gcp"]

    def _idx(label: str) -> int:
        return next(i for i, d in enumerate(details) if f"rung {label}" in d)

    assert _idx("spot_a100_80") < _idx("ondemand_a100_80")
    # The job lands on the on-demand A100-80 rung (the next rung after the spot +
    # flex misses).
    assert result.extra["gcp_ladder_rung"] == "ondemand_a100_80"


def test_ladder_short_job_full_rung_order(lease_store, marker_poster, captured_markers):
    """#680: a short lora-7b with EVERY rung capacity-missing → RunPod; the rung
    labels appear in the canonical order spot-80, spot-40, flex-80, ondemand-80,
    ondemand-40, then RunPod last (all-miss, so every index is well-defined)."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    details = [a.detail or "" for a in result.attempts if a.kind == "gcp"]
    expected = [
        "spot_a100_80",
        "spot_a100_40",
        "flexstart_a100_80",
        "ondemand_a100_80",
        "ondemand_a100_40",
    ]

    def _idx(label: str) -> int:
        return next(i for i, d in enumerate(details) if f"rung {label}" in d)

    idxs = [_idx(label) for label in expected]
    assert idxs == sorted(idxs), f"rung order wrong: {details}"
    assert len(idxs) == 5


def test_ladder_long_job_flexstart_before_ondemand_no_spot(
    lease_store, marker_poster, captured_markers
):
    """#680 (option (b) first-attempt-label): a long ft-7b with ample flex
    lands on ``flexstart_a100_80`` as the FIRST GCP attempt AND no ``spot_*``
    rung appears (a clean flex land stops the ladder, so the "flex leads" claim
    is carried by the first-attempt label + the no-spot assertion)."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/FLEX_START": QuotaHeadroom(
                metric="PREEMPTIBLE_NVIDIA_A100_80GB_GPUS",
                region="us-central1",
                limit=16.0,
                usage=0.0,
                needed=1,
            )
        }
    )
    spec = RunSpec(issue=137, intent="ft-7b", backend="auto")  # long, no A100-40 fallback
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "flexstart_a100_80"
    first_gcp = next(a for a in result.attempts if a.kind == "gcp")
    assert "rung flexstart_a100_80" in (first_gcp.detail or "")
    assert not any("spot" in (a.detail or "") for a in result.attempts if a.kind == "gcp")


def test_ladder_long_job_flexstart_miss_then_ondemand_order(
    lease_store, marker_poster, captured_markers
):
    """#680 MF3 (sister form, option (a)): a long ft-7b whose flex rung
    capacity-misses → on-demand IS attempted; assert flex precedes on-demand in
    the trail AND no spot rung appears."""
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80/FLEX_START": GcpProvisioningError("flex OUT", evidence={}),
        }
    )
    spec = RunSpec(issue=137, intent="ft-7b", backend="auto")  # long, no A100-40 fallback
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "ondemand_a100_80"
    details = [a.detail or "" for a in result.attempts if a.kind == "gcp"]

    def _idx(label: str) -> int:
        return next(i for i, d in enumerate(details) if f"rung {label}" in d)

    assert _idx("flexstart_a100_80") < _idx("ondemand_a100_80")
    assert not any("spot" in d for d in details)


def test_ladder_unknown_length_takes_long_branch(lease_store, marker_poster, captured_markers):
    """#680: an unknown-length spec (no time budget) takes the LONG branch —
    flex-first, no spot. With every rung capacity-missing, the FIRST GCP
    attempt label is ``flexstart_a100_80`` and no ``spot_*`` rung appears."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _spec(backend=None),  # unknown length
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    first_gcp = next(a for a in result.attempts if a.kind == "gcp")
    assert "rung flexstart_a100_80" in (first_gcp.detail or "")
    assert not any("spot" in (a.detail or "") for a in result.attempts if a.kind == "gcp")


def test_flexstart_rung_threads_flex_provisioning():
    """#680: the flex rung's spec carries ``provisioning_model == 'FLEX_START'``
    and the label ``flexstart_a100_80`` (direct unit test of the ladder
    builder, no routing)."""
    rungs = _ladder_specs(_short_lora_spec())
    flex = [(s, label) for s, label in rungs if label.startswith("flexstart_")]
    assert len(flex) == 1, f"expected exactly one flex rung, got labels {[lbl for _, lbl in rungs]}"
    spec, label = flex[0]
    assert label == "flexstart_a100_80"
    assert spec.extra["provisioning_model"] == "FLEX_START"
    assert spec.extra["machine_kind_tag"] == "A100-80"


def test_max_gcp_attempts_per_day_is_sixteen():
    """#1121 (replaces the #680 ``..._is_eight`` pin): the width-aware
    short-job walk is up to 14 rungs (3 wide widths x {spot, flex, ondemand}
    + the 5-rung base tail); 16 = 14 + margin, the same sizing logic as
    #680's 5+margin -> 8. Still an attempt COUNT, never a dollar cap
    (``tests/test_no_dollar_budget_caps.py``). This pin makes a silent revert
    a hard FAIL; a deliberate change updates the pin + its rationale."""
    assert MAX_GCP_ATTEMPTS_PER_DAY == 16


# ---------------------------------------------------------------------------
# #1121: width-aware wide-rung prefix (a2-ultragpu-{8,4,2}g before the base
# ladder for a --gpus-declaring dispatch; width-1 dispatches byte-identical)
# ---------------------------------------------------------------------------


def _width_spec(
    issue: int = 137,
    intent: str = "capture-7b",
    gpus: int | None = 8,
    **kwargs: Any,
) -> RunSpec:
    """A width-declaring auto-routing spec (#1121 tests)."""
    return RunSpec(issue=issue, intent=intent, backend="auto", gpus=gpus, **kwargs)


def _gcp_rung_labels(result) -> list[str]:
    """EXACT ordered rung-label list from the GCP attempts trail.

    Parses the token after ``rung `` in each GCP attempt's detail and strips
    a trailing colon. Deliberately NOT the file's substring ``_idx`` idiom:
    base-width labels are proper PREFIXES of wide labels
    (``flexstart_a100_80`` is a substring of ``flexstart_a100_80x8``), so a
    substring index scan can pass VACUOUSLY with the base tail missing —
    exact full-list equality is the only non-vacuous order assert here.
    """
    labels: list[str] = []
    for a in result.attempts:
        if a.kind != "gcp":
            continue
        m = re.search(r"\brung (\S+)", a.detail or "")
        assert m is not None, f"unparseable gcp attempt detail: {a.detail!r}"
        labels.append(m.group(1).rstrip(":"))
    return labels


_W8_LONG_ORDER = [
    "flexstart_a100_80x8",
    "ondemand_a100_80x8",
    "flexstart_a100_80x4",
    "ondemand_a100_80x4",
    "flexstart_a100_80x2",
    "ondemand_a100_80x2",
    "flexstart_a100_80",
    "ondemand_a100_80",
    "ondemand_a100_40",
]

_W8_SHORT_ORDER = [
    "spot_a100_80x8",
    "flexstart_a100_80x8",
    "ondemand_a100_80x8",
    "spot_a100_80x4",
    "flexstart_a100_80x4",
    "ondemand_a100_80x4",
    "spot_a100_80x2",
    "flexstart_a100_80x2",
    "ondemand_a100_80x2",
    "spot_a100_80",
    "spot_a100_40",
    "flexstart_a100_80",
    "ondemand_a100_80",
    "ondemand_a100_40",
]


def test_width8_long_job_walks_wide_rungs_width_major(lease_store, marker_poster, captured_markers):
    """#1121 AC1: a width-8 LONG/unknown-length capture-7b with EVERY rung
    capacity-missing walks EXACTLY the width-major long-job order (all
    provisioning models at width 8 before width 4, then 2, then the
    byte-identical base tail), and RunPod is last."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _width_spec(),  # no time budget -> LONG branch
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert _gcp_rung_labels(result) == _W8_LONG_ORDER
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert runpod_idxs and runpod_idxs[-1] == len(outcomes) - 1


def test_width8_short_job_full_rung_order(lease_store, marker_poster, captured_markers):
    """#1121: a spot_tolerant width-8 capture-7b with EVERY rung
    capacity-missing walks EXACTLY the 14-rung width-major SHORT order
    (spot -> flex -> ondemand within each width, then the base short tail)."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _width_spec(extra={"spot_tolerant": True}),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    labels = _gcp_rung_labels(result)
    assert labels == _W8_SHORT_ORDER
    assert len(labels) == 14


def test_width8_first_wide_rung_launches_a2_ultragpu_8g(
    lease_store, marker_poster, captured_markers
):
    """#1121 AC1 (launch shape) + the §4c RouteResult.extra lift: the first
    wide rung admits -> the launched spec carries the a2-ultragpu-8g machine
    override, the handle records requested/realized width, and BOTH ride
    ``RouteResult.extra`` (the ``epm:backend-selected`` marker surface the
    workload's re-shard contract reads)."""
    gcp = _GcpBackendDouble()
    result = route(
        _width_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "flexstart_a100_80x8"
    launched = gcp.launches[0]
    assert launched.extra["machine_spec_override"]["machine_type"] == "a2-ultragpu-8g"
    assert result.handle.extra["requested_gpus"] == 8
    assert result.handle.extra["realized_gpu_count"] == 8
    assert result.extra["requested_gpus"] == 8
    assert result.extra["realized_gpu_count"] == 8


def test_width_degradation_on_capacity_miss_lands_4g(lease_store, marker_poster, captured_markers):
    """#1121 AC1 (degradation): both width-8 rungs capacity-miss -> the walk
    degrades to the first width-4 rung; realized width 4, requested width 8."""
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80x8/FLEX_START": GcpProvisioningError("flex8 OUT", evidence={}),
            "A100-80x8/STANDARD": GcpProvisioningError("ondemand8 OUT", evidence={}),
        }
    )
    result = route(
        _width_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "flexstart_a100_80x4"
    assert gcp.launches[0].extra["machine_spec_override"]["machine_type"] == "a2-ultragpu-4g"
    assert result.extra["realized_gpu_count"] == 4
    assert result.extra["requested_gpus"] == 8


@pytest.mark.parametrize("gpus", [None, 1])
@pytest.mark.parametrize("length", ["short", "long"])
def test_width1_ladder_byte_identical_explicit_gpus_none_and_matching(length, gpus):
    """#1121 AC2: for ``gpus=None`` AND ``gpus == base.gpu_count`` the ladder
    output (labels + machine overrides + provisioning models) equals the
    PRE-CHANGE list. PROVENANCE: the frozen expected lists were generated
    from the PRE-#1121 ``_gcp_ladder_specs`` at merge-base commit
    ``88d6feb9d7a5c08599e66b0dcf501d7f53e2f818`` (main, 2026-07-08), BEFORE
    the width-aware refactor touched the function — not from the refactored
    code (that would be circular)."""
    frozen = {
        "short": [
            ("spot_a100_80", "a2-ultragpu-1g", 1, "SPOT"),
            ("spot_a100_40", "a2-highgpu-1g", 1, "SPOT"),
            ("flexstart_a100_80", "a2-ultragpu-1g", 1, "FLEX_START"),
            ("ondemand_a100_80", None, None, None),
            ("ondemand_a100_40", "a2-highgpu-1g", 1, "STANDARD"),
        ],
        "long": [
            ("flexstart_a100_80", "a2-ultragpu-1g", 1, "FLEX_START"),
            ("ondemand_a100_80", None, None, None),
            ("ondemand_a100_40", "a2-highgpu-1g", 1, "STANDARD"),
        ],
    }[length]
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="auto",
        gpus=gpus,
        time_budget_hours=1.0 if length == "short" else None,
    )
    got = []
    for s, label in _ladder_specs(spec):
        ex = s.extra or {}
        ov = ex.get("machine_spec_override")
        got.append(
            (
                label,
                ov["machine_type"] if ov else None,
                ov["gpu_count"] if ov else None,
                ex.get("provisioning_model"),
            )
        )
    assert got == frozen


def test_width_ladder_never_emits_h100_machine():
    """#1121: for every width-eligible intent x gpus in {2,4,8}, no rung's
    RESOLVED machine type is in the a3- (H100) family — H100 stays out of
    the width walk (quota exactly 8, no on-demand pool, headroom-blind)."""
    from explore_persona_space.backends.gcp import WIDTH_ELIGIBLE_INTENTS, machine_for_intent

    for intent in sorted(WIDTH_ELIGIBLE_INTENTS):
        for gpus in (2, 4, 8):
            for rung_spec, label in _ladder_specs(
                RunSpec(issue=137, intent=intent, backend="auto", gpus=gpus)
            ):
                machine = machine_for_intent(rung_spec)
                assert not machine.machine_type.startswith("a3-"), (intent, gpus, label)


def test_wide_rung_headroom_skip_is_free_and_advances(lease_store, marker_poster, captured_markers):
    """#1121: an insufficient-headroom reading at needed=8 on the first wide
    rung SKIPS it without burning a daily attempt (outcome
    ``quota_headroom_insufficient``) and the walk advances to the next wide
    rung — the launched result reports exactly ONE counted create."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80x8/FLEX_START": QuotaHeadroom(
                metric="PREEMPTIBLE_NVIDIA_A100_80GB_GPUS",
                region="us-central1",
                limit=16.0,
                usage=16.0,
                needed=8,
            )
        }
    )
    result = route(
        _width_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "ondemand_a100_80x8"
    outcomes = [(a.outcome) for a in result.attempts if a.kind == "gcp"]
    assert outcomes[0] == "quota_headroom_insufficient"
    # The skip was FREE: only the launched rung's create bumped the counter.
    assert result.extra["gcp_attempts_today"] == 1


def test_width8_cap_hit_mid_walk_falls_through_to_slurm_then_runpod(
    lease_store, marker_poster, captured_markers
):
    """#1121: the per-day attempt cap hit mid-wide-walk keeps today's
    behavior byte-for-byte — creates STOP at the cap, the chain proceeds to
    the SLURM lane and the RunPod terminal rung last."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _width_spec(),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=3,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    creates = [o for k, o in outcomes if k == "gcp" and o == "provisioning_failure"]
    assert len(creates) == 3  # creates stopped AT the injected cap
    assert ("gcp", "attempt_cap_exceeded") in outcomes
    nibi_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "nibi"]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert nibi_idxs and runpod_idxs and runpod_idxs[-1] == len(outcomes) - 1


def test_runpod_terminal_rung_receives_requested_gpus_8(
    lease_store, marker_poster, captured_markers
):
    """#1121 §4f: with every GCP rung + the SLURM lane exhausted, the RunPod
    terminal rung receives the REQUESTED width verbatim (``spec.gpus == 8``)
    on the translated intent (``capture-7b`` -> ``eval`` per the untouched
    ``RUNPOD_INTENT_FOR_GCP_INTENT`` map)."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _width_spec(),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=99,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    assert rp.launches[0].gpus == 8
    assert rp.launches[0].intent == "eval"  # capture-7b -> eval (#940 map, untouched)


def test_ft7b_width8_degrades_to_base_four_never_two(lease_store, marker_poster):
    """#1121 §4b design point 6: ft-7b (base 4x) with gpus=8 walks width [8]
    then its base 4x ladder — NEVER a 2-wide rung (its ZeRO-3 world size
    scales to 8 but not down to 2), and no x4 wide label either (width 4 IS
    the base)."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _width_spec(intent="ft-7b"),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    labels = _gcp_rung_labels(result)
    assert labels == [
        "flexstart_a100_80x8",
        "ondemand_a100_80x8",
        "flexstart_a100_80",
        "ondemand_a100_80",
    ]
    assert not any("x2" in label or "x4" in label for label in labels)


def test_width_ineligible_intent_gpus_ignored_with_warning(caplog):
    """#1121: a non-width-eligible intent (lora-7b-h100) with gpus=8 gets
    ``_requested_wide_widths == []`` + a logged warning, and its ladder is
    identical to the no-gpus ladder (today's ignore-spec.gpus semantics)."""
    from explore_persona_space.backends.gcp import machine_for_intent
    from explore_persona_space.backends.router import _requested_wide_widths

    spec = RunSpec(
        issue=137,
        intent="lora-7b-h100",
        backend="auto",
        gpus=8,
        extra={"provisioning_model": "SPOT"},  # H100 needs a non-STANDARD pin
    )
    base = machine_for_intent(RunSpec(issue=137, intent="lora-7b-h100", backend="auto"))
    with caplog.at_level("WARNING"):
        assert _requested_wide_widths(spec, base) == []
    assert any("non-width-eligible" in r.message for r in caplog.records)
    no_gpus = RunSpec(
        issue=137,
        intent="lora-7b-h100",
        backend="auto",
        extra={"provisioning_model": "SPOT"},
    )
    assert [label for _s, label in _ladder_specs(spec)] == [
        label for _s, label in _ladder_specs(no_gpus)
    ]


def test_unsupported_width_library_seam_no_snap_down(lease_store, marker_poster, caplog):
    """#1121 §4b design point 6 (library seam): ``gpus=6`` fed DIRECTLY to
    ``_requested_wide_widths`` / ``route()`` — bypassing the CLI guard —
    returns [] + a warning and uses the BASE-width ladder; NEVER a silent
    snap-down to [4, 2]."""
    from explore_persona_space.backends.gcp import machine_for_intent
    from explore_persona_space.backends.router import _requested_wide_widths

    spec = _width_spec(gpus=6)
    base = machine_for_intent(RunSpec(issue=137, intent="capture-7b", backend="auto"))
    with caplog.at_level("WARNING"):
        assert _requested_wide_widths(spec, base) == []
    assert any("no snap-down" in r.message for r in caplog.records)
    gcp = _GcpBackendDouble()
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    # Base-width rung, NO x-suffix: the unsupported width was ignored loudly,
    # never snapped down to a narrower wide rung.
    assert result.extra["gcp_ladder_rung"] == "flexstart_a100_80"


def test_classification_at_widest_not_base():
    """#1121 §4b design point 2 (kills the classify-at-base mutant): a
    width-8 capture-7b with time_budget_hours=1 reads 1 GPU-h at base
    (SHORT) but 8 GPU-h at width 8 (LONG) — the walk MUST be the flex-lead
    LONG sequence with NO spot rung anywhere."""
    labels = [label for _s, label in _ladder_specs(_width_spec(time_budget_hours=1.0))]
    assert labels == _W8_LONG_ORDER
    assert not any(label.startswith("spot") for label in labels)


def test_eval_intent_width8_upgrades_to_a100_family_then_l4_tail():
    """#1121 (cross-family upgrade + degradation): eval (base 1x L4) with
    gpus=8 walks the wide a2-ultragpu rungs first, and its BASE tail is
    today's L4 ladder byte-identically (labels + overrides + provisioning;
    frozen from the pre-change code at merge-base ``88d6feb9d7``)."""
    got = []
    for s, label in _ladder_specs(_width_spec(intent="eval")):
        ex = s.extra or {}
        ov = ex.get("machine_spec_override")
        got.append(
            (
                label,
                ov["machine_type"] if ov else None,
                ov["gpu_count"] if ov else None,
                ex.get("provisioning_model"),
            )
        )
    assert [g[0] for g in got[:6]] == _W8_LONG_ORDER[:6]  # wide prefix
    assert got[-3:] == [
        ("flexstart_l4", "g2-standard-4", 1, "FLEX_START"),
        ("ondemand_l4", None, None, None),
        ("ondemand_a100_40", "a2-highgpu-1g", 1, "STANDARD"),
    ]


def test_pinned_provisioning_spot_walks_spot_at_every_width():
    """#1121 §4b design point 4: a caller SPOT pin + gpus=8 walks spot at
    EVERY width then the pinned base tail — never silently un-pins. The
    2-label base tail (``spot_a100_80, spot_a100_40``) was confirmed against
    the PRE-CHANGE pinned-branch output at merge-base ``88d6feb9d7``."""
    labels = [
        label for _s, label in _ladder_specs(_width_spec(extra={"provisioning_model": "SPOT"}))
    ]
    assert labels == [
        "spot_a100_80x8",
        "spot_a100_80x4",
        "spot_a100_80x2",
        "spot_a100_80",
        "spot_a100_40",
    ]


def test_workload_error_on_wide_rung_fails_over_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """#1121 (width variant of the #680 MF2 test): a GcpWorkloadError on a
    WIDE rung short-circuits STRAIGHT to RunPod — no later GCP rung, no
    SLURM lane."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80x8/FLEX_START": GcpWorkloadError(
                "workload crashed", evidence={"phase": "train"}
            ),
        }
    )
    result = route(
        _width_spec(),
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
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    assert ("gcp", "workload_failure") in outcomes
    # STRAIGHT to RunPod: no GCP rung after the workload error, no SLURM lane.
    gcp_after_failure = [
        o for k, o in outcomes[outcomes.index(("gcp", "workload_failure")) + 1 :] if k == "gcp"
    ]
    assert gcp_after_failure == []
    assert not any(k == "nibi" for k, _o in outcomes)


# ---------------------------------------------------------------------------
# #1379: explicit wide-intent width degradation (sweep-8g-a100 appends
# degraded a2-ultragpu-{4,2}g rungs AFTER its 8g base rungs on capacity
# miss; --width-required pins; sweep-8g-h100 never degrades)
# ---------------------------------------------------------------------------


def _sweep_spec(intent: str = "sweep-8g-a100", **kwargs: Any) -> RunSpec:
    """An explicit-wide-intent auto-routing spec (#1379 tests)."""
    return RunSpec(issue=137, intent=intent, backend="auto", **kwargs)


def _ladder_tuples(spec: RunSpec) -> list[tuple[str, str | None, int | None, str | None]]:
    """(label, override machine_type, override gpu_count, provisioning) per rung.

    The same exact-full-list idiom as
    ``test_width1_ladder_byte_identical_explicit_gpus_none_and_matching`` —
    base labels are proper PREFIXES of wide labels, so substring scans are
    vacuous; only full-tuple equality is a non-vacuous order assert.
    """
    got: list[tuple[str, str | None, int | None, str | None]] = []
    for s, label in _ladder_specs(spec):
        ex = s.extra or {}
        ov = ex.get("machine_spec_override")
        got.append(
            (
                label,
                ov["machine_type"] if ov else None,
                ov["gpu_count"] if ov else None,
                ex.get("provisioning_model"),
            )
        )
    return got


def test_explicit_sweep8g_a100_long_ladder_order_with_degraded_rungs():
    """#1379 AC1 (T1): an explicit ``sweep-8g-a100`` LONG/unknown-length
    dispatch (the #825 shape — no time budget) gains degraded x4 then x2
    rungs AFTER its 8g base rungs, width-major, with the base tail
    byte-identical (the as-is on-demand rung stays the caller spec)."""
    assert _ladder_tuples(_sweep_spec()) == [
        ("flexstart_a100_80", "a2-ultragpu-8g", 8, "FLEX_START"),
        ("ondemand_a100_80", None, None, None),
        ("flexstart_a100_80x4", "a2-ultragpu-4g", 4, "FLEX_START"),
        ("ondemand_a100_80x4", "a2-ultragpu-4g", 4, "STANDARD"),
        ("flexstart_a100_80x2", "a2-ultragpu-2g", 2, "FLEX_START"),
        ("ondemand_a100_80x2", "a2-ultragpu-2g", 2, "STANDARD"),
    ]


def test_explicit_sweep8g_a100_short_ladder_order_with_degraded_rungs():
    """#1379 AC1 (T2): the SHORT (spot_tolerant) explicit sweep walk is the
    9-rung width-major order — spot -> flex -> ondemand within width 8
    (the base rungs ARE the width-8 rungs), then x4, then x2."""
    assert _ladder_tuples(_sweep_spec(extra={"spot_tolerant": True})) == [
        ("spot_a100_80", "a2-ultragpu-8g", 8, "SPOT"),
        ("flexstart_a100_80", "a2-ultragpu-8g", 8, "FLEX_START"),
        ("ondemand_a100_80", None, None, None),
        ("spot_a100_80x4", "a2-ultragpu-4g", 4, "SPOT"),
        ("flexstart_a100_80x4", "a2-ultragpu-4g", 4, "FLEX_START"),
        ("ondemand_a100_80x4", "a2-ultragpu-4g", 4, "STANDARD"),
        ("spot_a100_80x2", "a2-ultragpu-2g", 2, "SPOT"),
        ("flexstart_a100_80x2", "a2-ultragpu-2g", 2, "FLEX_START"),
        ("ondemand_a100_80x2", "a2-ultragpu-2g", 2, "STANDARD"),
    ]


def test_explicit_sweep8g_a100_degrades_to_4g_on_8wide_capacity_miss(
    lease_store, marker_poster, captured_markers
):
    """#1379 AC1+AC5 (T3, route() end-to-end): both 8-wide rungs
    capacity-miss -> the walk degrades to the first x4 rung; the launched
    override is a2-ultragpu-4g and the marker/handle width fields read
    requested=8 (via ``_declared_width`` — pre-#1379 this was null for an
    explicit intent) / realized=4."""
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80x8/FLEX_START": GcpProvisioningError("flex8 OUT", evidence={}),
            "A100-80x8/STANDARD": GcpProvisioningError("ondemand8 OUT", evidence={}),
        }
    )
    result = route(
        _sweep_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "flexstart_a100_80x4"
    assert gcp.launches[0].extra["machine_spec_override"]["machine_type"] == "a2-ultragpu-4g"
    assert result.extra["requested_gpus"] == 8
    assert result.extra["realized_gpu_count"] == 4
    # AC5 folded in: the handle sidecar carries the same two width fields.
    assert result.handle.extra["requested_gpus"] == 8
    assert result.handle.extra["realized_gpu_count"] == 4


def test_explicit_sweep8g_a100_width_required_pins_full_width(
    lease_store, marker_poster, captured_markers
):
    """#1379 AC2 (T4): ``width_required`` pins the intent at its full width —
    the ladder is byte-identical to the pre-#1379 one (no degraded rungs on
    the long, short, OR pinned branch), and an exhausted walk falls through
    to RunPod exactly as today with no x4/x2 label in the attempts trail."""
    # (a) ladder-level: the pre-change 2-rung long base list, no suffix.
    assert _ladder_tuples(_sweep_spec(extra={"width_required": True})) == [
        ("flexstart_a100_80", "a2-ultragpu-8g", 8, "FLEX_START"),
        ("ondemand_a100_80", None, None, None),
    ]
    # Encouraged companions (plan §13): the short + pinned variants pin the
    # pre-change base lists too (labels only — shapes covered by T1/T2/T7).
    assert [
        t[0]
        for t in _ladder_tuples(_sweep_spec(extra={"width_required": True, "spot_tolerant": True}))
    ] == ["spot_a100_80", "flexstart_a100_80", "ondemand_a100_80"]
    assert [
        t[0]
        for t in _ladder_tuples(
            _sweep_spec(extra={"width_required": True, "provisioning_model": "SPOT"})
        )
    ] == ["spot_a100_80"]
    # (b) route()-level: every GCP rung missing -> RunPod last, no degraded
    # rung ever attempted.
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _sweep_spec(extra={"width_required": True}),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    labels = _gcp_rung_labels(result)
    assert labels == ["flexstart_a100_80", "ondemand_a100_80"]
    assert not any("x4" in lbl or "x2" in lbl for lbl in labels)


def test_explicit_wide_degrade_never_emits_h100_machine():
    """#1379 AC3 (T5): the explicit-path sibling of the untouched #1121
    ``test_width_ladder_never_emits_h100_machine`` invariant — every rung of
    every ``sweep-8g-a100`` ladder variant (long, short, pinned) RESOLVES to
    an a2- (A100) machine, never a3- (H100)."""
    from explore_persona_space.backends.gcp import machine_for_intent

    for extra in (None, {"spot_tolerant": True}, {"provisioning_model": "SPOT"}):
        for rung_spec, label in _ladder_specs(_sweep_spec(extra=extra)):
            machine = machine_for_intent(rung_spec)
            assert machine.machine_type.startswith("a2-"), (extra, label)
            assert not machine.machine_type.startswith("a3-"), (extra, label)


def test_explicit_sweep8g_h100_ladder_unchanged_no_degradation():
    """#1379 AC3 (T6, fork decision pinned): ``sweep-8g-h100`` is EXCLUDED
    from explicit-wide degradation — its pinned-SPOT ladder (the realistic
    H100 dispatch shape: no on-demand pool) is exactly the 1-rung base list,
    no label carries a degraded x4/x2 suffix, and no rung's override names
    an a2-ultragpu-{4,2}g machine (cross-type degradation would silently
    change silicon). Label-level asserts ONLY — an h100 ondemand rung is
    never rendered here (H100+STANDARD raises at render by design, A12)."""
    pinned = _ladder_tuples(
        _sweep_spec(intent="sweep-8g-h100", extra={"provisioning_model": "SPOT"})
    )
    assert pinned == [("spot_h100_80", "a3-highgpu-8g", 8, "SPOT")]
    # Encouraged companion (plan §13): the long-branch label list is the
    # pre-change 2-rung base list (labels only — never rendered).
    long_labels = [t[0] for t in _ladder_tuples(_sweep_spec(intent="sweep-8g-h100"))]
    assert long_labels == ["flexstart_h100_80", "ondemand_h100_80"]
    for tuples in (pinned, _ladder_tuples(_sweep_spec(intent="sweep-8g-h100"))):
        for label, machine_type, _gpu_count, _prov in tuples:
            assert "x4" not in label and "x2" not in label, label
            assert machine_type not in {"a2-ultragpu-4g", "a2-ultragpu-2g"}, label


def test_explicit_sweep8g_a100_pinned_spot_walks_pinned_degraded_rungs():
    """#1379 (T7): a caller ``provisioning_model`` pin is honored at every
    degraded width (the #537/#680 pin contract extended, mirroring the
    #1121 per-width pin behavior) — the pinned-SPOT walk is spot-only at
    widths 8, 4, 2."""
    assert _ladder_tuples(_sweep_spec(extra={"provisioning_model": "SPOT"})) == [
        ("spot_a100_80", "a2-ultragpu-8g", 8, "SPOT"),
        ("spot_a100_80x4", "a2-ultragpu-4g", 4, "SPOT"),
        ("spot_a100_80x2", "a2-ultragpu-2g", 2, "SPOT"),
    ]


def test_explicit_sweep8g_a100_runpod_still_last_after_full_degraded_walk(
    lease_store, marker_poster, captured_markers
):
    """#1379 AC6 (T8): width degradation stays WITHIN the GCP ladder — with
    every GCP rung (incl. the degraded ones) and the SLURM lane missing,
    RunPod launches exactly once and its attempt is strictly LAST."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _sweep_spec(),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=99,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    assert _gcp_rung_labels(result) == [
        "flexstart_a100_80",
        "ondemand_a100_80",
        "flexstart_a100_80x4",
        "ondemand_a100_80x4",
        "flexstart_a100_80x2",
        "ondemand_a100_80x2",
    ]
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    gcp_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "gcp"]
    nibi_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "nibi"]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert len(gcp_idxs) == 6  # the full 6-rung degraded long walk attempted
    assert nibi_idxs, "the free SLURM lane must have been attempted"
    assert runpod_idxs and runpod_idxs[-1] == len(outcomes) - 1  # runpod LAST
    assert max(gcp_idxs) < runpod_idxs[-1]
    assert max(nibi_idxs) < runpod_idxs[-1]


def test_workload_error_on_degraded_rung_fails_over_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """#1379 AC6 (encouraged companion, plan §13 — the degraded-suffix
    parametrization of the #1121 wide-rung failover pin): a GcpWorkloadError
    on a DEGRADED x4 rung short-circuits STRAIGHT to RunPod — no later GCP
    rung, no SLURM lane."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80x8/FLEX_START": GcpProvisioningError("flex8 OUT", evidence={}),
            "A100-80x8/STANDARD": GcpProvisioningError("ondemand8 OUT", evidence={}),
            "A100-80x4/FLEX_START": GcpWorkloadError(
                "workload crashed", evidence={"phase": "train"}
            ),
        }
    )
    result = route(
        _sweep_spec(),
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
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD
    assert len(rp.launches) == 1
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    assert ("gcp", "workload_failure") in outcomes
    gcp_after_failure = [
        o for k, o in outcomes[outcomes.index(("gcp", "workload_failure")) + 1 :] if k == "gcp"
    ]
    assert gcp_after_failure == []
    assert not any(k == "nibi" for k, _o in outcomes)


def test_declared_width_none_for_non_sweep_intents():
    """#1379 (T9): ``_declared_width`` returns ``spec.gpus`` VERBATIM off the
    explicit-wide path (byte-identity of the marker width fields for every
    non-degradable intent), the intent's own base width (8) for a
    ``sweep-8g-a100`` dispatch with no --gpus, and None for
    ``sweep-8g-h100`` (not in the degrade set)."""
    from explore_persona_space.backends.router import _declared_width

    assert _declared_width(RunSpec(issue=137, intent="lora-7b", backend="auto")) is None
    assert _declared_width(RunSpec(issue=137, intent="lora-7b", backend="auto", gpus=8)) == 8
    assert _declared_width(RunSpec(issue=137, intent="sweep-8g-a100", backend="auto")) == 8
    assert _declared_width(RunSpec(issue=137, intent="sweep-8g-h100", backend="auto")) is None


def test_workload_error_on_later_rung_fails_over_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """#680 MF2: a GcpWorkloadError on a LATER rung (the flex rung, after both
    spot rungs capacity-miss on a short lora-7b) fails over STRAIGHT to RunPod —
    the failover is rung-position-independent, not just rung-1. No on-demand GCP
    rung is attempted after the workload error, and no SLURM lane is attempted."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # would start eventually
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80/SPOT": GcpProvisioningError("spot OUT", evidence={}),
            "A100-40/SPOT": GcpProvisioningError("spot40 OUT", evidence={}),
            "A100-80/FLEX_START": GcpWorkloadError("workload crashed", evidence={"phase": "train"}),
        }
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD
    assert len(rp.launches) == 1
    # Exactly one workload_failure (the flex rung); the two spot rungs
    # capacity-missed (provisioning_failure) BEFORE it.
    workload = [a for a in result.attempts if a.kind == "gcp" and a.outcome == "workload_failure"]
    assert len(workload) == 1
    # No on-demand GCP rung attempted after the workload error short-circuit.
    assert not any("ondemand" in (a.detail or "") for a in result.attempts if a.kind == "gcp")
    # The failover routed STRAIGHT to RunPod — no SLURM lane attempted.
    assert not any(a.kind == "nibi" for a in result.attempts)


def test_happy_path_a100_80_ondemand_launches_after_flex_miss(
    lease_store, marker_poster, captured_markers
):
    """T6 (#680: re-scoped to a LONG lora-7b — flex-80 -> ondemand-80 ->
    ondemand-40, no spot). flex-80 insufficient + on-demand A100-80 sufficient
    → on-demand A100-80 launches and the ladder never advances past it
    (regression: the on-demand A100-80 common case still works)."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/FLEX_START": _A100_80_STD_SHORT,
            "A100-80/STANDARD": QuotaHeadroom(
                metric="NVIDIA_A100_80GB_GPUS", region="us-central1", limit=8.0, usage=0.0, needed=1
            ),
        }
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", time_budget_hours=10.0)  # long
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert len(gcp.launches) == 1
    assert result.extra["gcp_ladder_rung"] == "ondemand_a100_80"
    # Exactly one launched attempt, no A100-40 rung touched, no spot rung.
    launched = [a for a in result.attempts if a.kind == "gcp" and a.outcome == "launched"]
    assert len(launched) == 1
    assert "A100-40" not in (gcp.launches[0].extra.get("machine_kind_tag") or "A100-80")
    assert not any("spot" in (a.detail or "") for a in result.attempts if a.kind == "gcp")


def test_no_budget_job_is_not_short_no_spot_rung(lease_store, marker_poster, captured_markers):
    """T7: a spec with no time_budget_hours and no estimated_gpu_hours is
    treated as unknown-length (NOT short) → the long branch has NO spot rung,
    and an all-full job routes to RunPod. #680: the long branch is
    flex-80 -> ondemand-80 -> ondemand-40."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "QUOTA_EXCEEDED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto")  # no budget
    result = route(
        spec,
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    gcp_fails = [
        a for a in result.attempts if a.kind == "gcp" and a.outcome == "provisioning_failure"
    ]
    # flex-80 + ondemand-80 + ondemand-40 — no spot rung (unknown length).
    assert len(gcp_fails) == 3
    assert not any("spot" in (a.detail or "") for a in gcp_fails)


def test_debug_intent_rung_label_reflects_l4_not_a100(lease_store, marker_poster, captured_markers):
    """#672: the rung-1 label is COMPOSED from the resolved machine's
    accelerator kind, so a ``debug`` intent (mapped to g2-standard-4 = L4)
    is labeled ``ondemand_l4`` — NOT the historically-hardcoded
    ``ondemand_a100_80`` (which lied about the machine being attempted).

    Force a capacity miss on the create so the rung label lands in the
    attempt trail's ``detail``, where a post-hoc debugger reads it.
    """
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    spec = RunSpec(issue=137, intent="debug", backend="auto", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=_PassiveRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    # The rung-1 attempt detail names the L4 machine actually attempted.
    rung1_details = [
        a.detail
        for a in result.attempts
        if a.kind == "gcp" and a.detail and a.detail.startswith("rung ondemand_")
    ]
    assert rung1_details, "no on-demand GCP rung attempt recorded"
    assert any("rung ondemand_l4:" in d for d in rung1_details)
    # The historical hardcoded A100 label must NOT appear for an L4 intent.
    assert not any("ondemand_a100_80" in (a.detail or "") for a in result.attempts)
    # The pre-escalation intermediate breadcrumb carries the L4 rung too.
    intermediates = [
        body
        for body in _by_reason(captured_markers, ROUTE_REASON_AUTO_FALLBACK_GCP)
        if body.get("extra", {}).get("intermediate") is True
    ]
    assert any(body["extra"].get("rung") == "ondemand_l4" for body in intermediates)


def test_intermediate_marker_records_requested_kind_on_explicit_gcp_pin(
    lease_store, marker_poster, captured_markers
):
    """#672: the pre-escalation intermediate breadcrumb records the user's
    ORIGINAL ``--backend`` ask in ``requested_kind`` — ``"gcp"`` for an
    explicit ``backend: gcp`` pin — instead of the hardcoded ``None`` that
    made an explicit override indistinguishable from an auto-chain
    escalation post-hoc.
    """
    gcp = _GcpBackendDouble()  # the explicit pin launches on GCP
    spec = RunSpec(issue=137, intent="lora-7b", backend="gcp", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),  # RunPod must NOT be reached
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.requested_kind == "gcp"
    intermediates = [
        body
        for body in _by_reason(captured_markers, ROUTE_REASON_AUTO_FALLBACK_GCP)
        if body.get("extra", {}).get("intermediate") is True
    ]
    assert intermediates, "pre-escalation intermediate breadcrumb missing"
    assert all(body["requested_kind"] == "gcp" for body in intermediates)


def test_intermediate_marker_requested_kind_none_on_auto_chain(
    lease_store, marker_poster, captured_markers
):
    """#672 companion: on the AUTO chain (no ``--backend``) the intermediate
    breadcrumb keeps ``requested_kind: None`` — the threading must not
    accidentally stamp ``"auto"`` / ``"gcp"`` on a router-chosen escalation
    (consistency with the final/terminal markers' auto-path convention).
    """
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts → escalate to GCP
    gcp = _GcpBackendDouble()
    result = route(
        _spec(backend=None),  # AUTO
        runpod_backend=_ExplodingRunpod(),
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
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
    intermediates = [
        body
        for body in _by_reason(captured_markers, ROUTE_REASON_AUTO_FALLBACK_GCP)
        if body.get("extra", {}).get("intermediate") is True
    ]
    assert intermediates, "pre-escalation intermediate breadcrumb missing"
    assert all(body["requested_kind"] is None for body in intermediates)


def test_workload_error_on_a_rung_fails_over_to_runpod_no_rung_advance(
    lease_store, marker_poster, captured_markers
):
    """T8 (REVERSED for task #658): a GcpWorkloadError on the FIRST rung
    STOPS the ladder (does NOT advance to the cheaper rungs — re-running
    broken code on A100-40 / spot burns credit) and FAILS OVER to RunPod.
    Exactly one GCP create is attempted; RunPod launches once."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpWorkloadError("workload crashed", evidence={"phase": "train"})
    )
    result = route(
        _short_lora_spec(),  # has A100-40 + spot rungs available
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD
    assert len(rp.launches) == 1
    # Ladder did NOT advance — exactly ONE GCP rung recorded a
    # workload_failure before failing over (the workload error
    # short-circuited the remaining cheaper rungs). The double raises before
    # appending, so probe via the attempts trail, not gcp.launches.
    gcp_workload = [
        a for a in result.attempts if a.kind == "gcp" and a.outcome == "workload_failure"
    ]
    assert len(gcp_workload) == 1
    # No advanced-rung provisioning_failure attempts were recorded.
    assert not [
        a for a in result.attempts if a.kind == "gcp" and a.outcome == "provisioning_failure"
    ]


def test_per_day_attempt_cap_stops_gcp_creates_falls_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """T9: with the lease pre-seeded at the per-day attempt cap, the ladder
    issues NO new GCP create and falls through to RunPod (cap stays
    count-based, never a dollar cap)."""
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble()  # would succeed if it ever launched
    # Pre-seed the lease at the cap for today.
    lease = Lease(
        issue=137,
        spec_hash="x",
        attempt_id="att-x",
        gcp_attempts_today=2,
        gcp_attempts_date=datetime.now(tz=UTC).date().isoformat(),
    )
    lease_store.write(lease)
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=2),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert gcp.launches == []  # NO new GCP create issued
    assert any(a.outcome == "attempt_cap_exceeded" for a in result.attempts)


def test_per_day_attempt_cap_hit_mid_ladder_stops_remaining_rungs(
    lease_store, marker_poster, captured_markers
):
    """T9b: the cap is RE-READ each rung. Pre-seed the lease one below the
    cap; rung 1 capacity-misses and bumps to the cap; rung 2 must NOT even call
    create — the cap stops it mid-ladder and the route falls to RunPod. #680:
    for a short job rung 1 is the spot A100-80 rung."""
    rp = _PassiveRunpod()
    # The spot A100-80 (rung 1) create capacity-misses (advances + bumps the
    # attempt counter); rung 2 would succeed if it were ever reached.
    gcp = _GcpBackendDouble(
        launch_raises_by_rung={
            "A100-80/SPOT": GcpProvisioningError(
                "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
            )
        }
    )
    lease = Lease(
        issue=137,
        spec_hash="x",
        attempt_id="att-x",
        gcp_attempts_today=1,  # one below the cap of 2
        gcp_attempts_date=datetime.now(tz=UTC).date().isoformat(),
    )
    lease_store.write(lease)
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=2),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    # The A100-40 rung never launched (cap hit after the A100-80 bump).
    assert gcp.launches == []
    outcomes = [a.outcome for a in result.attempts]
    assert "provisioning_failure" in outcomes  # rung-1 A100-80 create failed
    assert "attempt_cap_exceeded" in outcomes  # rung-2 stopped by the cap


def test_spot_tolerant_forces_spot_rung_past_threshold(
    lease_store, marker_poster, captured_markers
):
    """spot_tolerant is retained as a FORCE-spot override: a LONG lora-7b
    (10 GPU-h > threshold) tagged spot_tolerant still gets the spot rungs
    (the caller explicitly opted into preemption)."""
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/STANDARD": _A100_80_STD_SHORT,
            "A100-40/STANDARD": _A100_40_STD_SHORT,
            "A100-80/SPOT": _A100_80_SPOT_AMPLE,
        }
    )
    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="auto",
        time_budget_hours=10.0,  # over the 2 GPU-h threshold
        extra={"spot_tolerant": True},  # but force-spot
    )
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "spot_a100_80"


def test_spot_max_gpu_hours_env_override(lease_store, marker_poster, monkeypatch):
    """The short-job threshold is env-overridable: raising it to 20 GPU-h
    makes a 10-GPU-h lora-7b 'short' so the spot rung fires."""
    monkeypatch.setenv(ENV_SPOT_MAX_GPU_HOURS, "20")
    gcp = _GcpBackendDouble(
        quota_headroom_by_rung={
            "A100-80/STANDARD": _A100_80_STD_SHORT,
            "A100-40/STANDARD": _A100_40_STD_SHORT,
            "A100-80/SPOT": _A100_80_SPOT_AMPLE,
        }
    )
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", time_budget_hours=10.0)
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "spot_a100_80"


def test_gcp_route_marker_carries_provisioning_and_quota_pool(
    lease_store, marker_poster, captured_markers
):
    """#631 D4: the SUCCESS-path GCP epm:backend-selected marker carries the
    additive provisioning_model + quota_pool fields. #680: the default lora-7b
    spec is unknown-length => the long branch, whose FIRST rung is the
    flex-start A100-80 launch, so the resolved values are FLEX_START +
    PREEMPTIBLE_NVIDIA_A100_80GB_GPUS (the preemptible A100 pool)."""
    gcp = _GcpBackendDouble(
        quota_headroom=QuotaHeadroom(
            metric="PREEMPTIBLE_NVIDIA_A100_80GB_GPUS",
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
    assert result.extra["provisioning_model"] == "FLEX_START"
    assert result.extra["quota_pool"] == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"
    # The SUCCESS-path marker body (reason auto_fallback_gcp) carries them too.
    bodies = _by_reason(captured_markers, "auto_fallback_gcp")
    assert bodies, "no success-path GCP epm:backend-selected marker was posted"
    final = bodies[-1]
    assert final["extra"]["provisioning_model"] == "FLEX_START"
    assert final["extra"]["quota_pool"] == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"
    # Pre-existing fields untouched (additive change).
    assert "gcp_attempts_today" in final["extra"]


def test_gcp_explicit_override_marker_carries_provisioning_and_quota_pool(
    lease_store, marker_poster, captured_markers
):
    """#631 round-3: an explicit ``backend: gcp`` fresh launch (the
    ``_override_free_or_gcp`` terminal path, reason ``override``) carries the
    additive provisioning_model + quota_pool fields on BOTH the result object
    and its epm:backend-selected marker — the GCP analogue of an explicit
    ``backend: runpod`` override, which round 1 left without the new fields.
    #680: the default lora-7b spec is unknown-length => the long branch whose
    first rung is flex-start, so the resolved values are FLEX_START +
    PREEMPTIBLE_NVIDIA_A100_80GB_GPUS."""
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
    assert result.extra["provisioning_model"] == "FLEX_START"
    assert result.extra["quota_pool"] == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"
    # The override-path marker body carries them too.
    bodies = _by_reason(captured_markers, ROUTE_REASON_OVERRIDE)
    assert bodies, "no explicit-override GCP epm:backend-selected marker was posted"
    final = bodies[-1]
    assert final["extra"]["provisioning_model"] == "FLEX_START"
    assert final["extra"]["quota_pool"] == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"


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
    # #680: the nobudget lora-7b is unknown-length => long branch (flex-80,
    # ondemand-80, ondemand-40, NO spot), all three prepare-fail; ``launch``
    # still NEVER runs after a failed prepare on any rung.
    assert gcp.calls == ["prepare", "prepare", "prepare"], (
        "launch must NOT run after a failed prepare"
    )
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


def test_runpod_terminal_rung_under_gcp_first_default(lease_store):
    """#656 reversed-invariant under the GCP-first default: GCP capacity-fails
    on every rung, the free lane fails, and RunPod IS reached as the
    documented TERMINAL rung (the chain no longer raises
    NoComputeAvailableError — RunPod is the new tail).

    The real-money safety property is preserved by ORDERING (the GCP rungs +
    nibi are all recorded failed BEFORE the RunPod launch), pinned by
    `test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted`."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi")
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError("QUOTA_EXCEEDED", evidence={"matched_pattern": "Q"})
    )
    result = route(
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
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_RUNPOD_FALLBACK
    assert len(rp.launches) == 1
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    assert ("gcp", "provisioning_failure") in outcomes
    assert any(kind == "nibi" for kind, _o in outcomes)
    # RunPod is the LAST attempt.
    assert outcomes[-1] == ("runpod", "launched")


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


# ---------------------------------------------------------------------------
# #656 — CLAUDE.md doc-drift guard (AC5: the prose matches the reversed
# contract; the retired no-auto-RunPod phrasing is gone).
# ---------------------------------------------------------------------------


def _claude_md_path() -> Path:
    """Resolve the repo-root CLAUDE.md from this test file's location."""
    return Path(__file__).resolve().parent.parent / "CLAUDE.md"


def test_claude_md_compute_backends_section_matches_656_contract() -> None:
    """AC5 doc-drift guard: the CLAUDE.md `Compute backends` section reflects
    the #656 reversed contract (RunPod is the documented terminal fallback)
    and does NOT carry the retired no-auto-RunPod phrasing. Fail-loud if a
    future edit reintroduces the stale wording or drops the new contract."""
    text = _claude_md_path().read_text(encoding="utf-8")
    # NEW contract wording MUST be present.
    assert "auto_fallback_runpod" in text, "CLAUDE.md missing the new RunPod-fallback reason code"
    assert "RunPod terminal rung" in text, "CLAUDE.md missing the 'RunPod terminal rung' contract"
    # RETIRED phrasing MUST be gone (the reversed invariant).
    assert "The auto chain NEVER calls RunPod" not in text, (
        "CLAUDE.md still carries the retired no-auto-RunPod phrasing (#656 reversed it)"
    )
    assert "test_no_auto_runpod_path_under_any_failure" not in text, (
        "CLAUDE.md still references the replaced negative test by name"
    )


# ---------------------------------------------------------------------------
# #659 — ASYNC GCP-workload-failure → RunPod failover (poller / dispatch path)
#
# The synchronous route()-time failover (#658) cannot reach a GCP VM that was
# already up and crashed its WORKLOAD minutes in — there is no live route()
# call to raise GcpWorkloadError from. #659 adds a PUBLIC router helper the
# poller (scripts/backend_poll.py) calls to re-dispatch that dead GCP workload
# onto the SAME RunPod terminal rung the sync path uses, exactly once, labeled
# with a DISTINCT async reason so the marker trail tells the two detection
# paths apart. These tests fail TODAY (the helper + the _ASYNC reason constant
# do not exist yet -> ImportError) and pass after the router helper lands.
# ---------------------------------------------------------------------------


def test_async_gcp_workload_crash_fails_over_to_runpod_exactly_once(
    lease_store, marker_poster, captured_markers
):
    """#659 (HEADLINE acceptance gate): a poller-detected dead GCP workload
    re-dispatches on RunPod once, via the SAME terminal rung as the sync path,
    labeled with the async reason."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=_spec(backend="gcp"),
        runpod_backend=rp,
        evidence={"source": "async_poller", "current_phase": "terminal_workload_failed"},
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC
    assert len(rp.launches) == 1  # exactly once
    finals = _by_reason(captured_markers, ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC)
    assert finals
    assert finals[-1]["extra"].get("gcp_workload_evidence", {}).get("source") == "async_poller"


def test_failover_runpod_unavailable_emits_infra_no_compute(
    lease_store, marker_poster, captured_markers
):
    """#659 sibling #4: when the RunPod failover launch raises
    ``NoComputeAvailableError`` (RunPod truly unavailable), the helper
    propagates it so the poller can emit a terminal infra JSON with
    ``failure_class: "infra"`` + ``reason: "no_compute_available"`` — the
    watcher's capacity-retry pass re-drives that reason once a lane frees.

    The Statistics-reconciler standing rec is to pin BOTH the class AND the
    reason: a bare ``infra`` without ``no_compute_available`` would not be
    re-driven (``TRANSIENT_CAPACITY_REASONS`` keys on the reason)."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _ExplodingRunpodNoCompute()
    with pytest.raises(NoComputeAvailableError):
        failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=lease_store,
            now_fn=_clock(),
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        )
    # The poller-side mapping (scripts/backend_poll.py) turns that raise into
    # the infra JSON; the contract the poller relies on is asserted in
    # tests/test_backend_poll.py::
    #   test_failover_runpod_unavailable_emits_infra_no_compute_poll_json
    # (both failure_class == "infra" AND reason == "no_compute_available").


def test_async_failover_reason_distinct_from_sync():
    """#659 sibling #5: the async reason VALUE differs from the sync one so the
    ``epm:backend-selected`` marker trail tells the route()-time failover apart
    from the poller-detected one (same RunPod target, different detection
    path)."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    )

    assert (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC != ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD
    )
    # The async reason carries an "async" discriminator for grep-ability.
    assert "async" in ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC


# ---------------------------------------------------------------------------
# #783 — GCP FLEX_START queue-timeout → RunPod failover (poller / router seam)
#
# The queue-timeout failover reuses the SAME router seam as the #659 async
# workload-crash failover (failover_to_runpod_after_async_workload_crash → the
# RunPod terminal rung), passing the distinct queue-timeout reason. These tests
# pin the router-level contract: the seam carries the queue-timeout reason onto
# the launched RouteResult + the epm:backend-selected marker, and the two
# preserved invariant tests (RunPod-is-terminal-rung, workload-error-no-cascade)
# are unaffected (they live above, run in the same suite).
# ---------------------------------------------------------------------------


def test_queue_timeout_failover_seam_carries_queue_timeout_reason(
    lease_store, marker_poster, captured_markers
):
    """#783 (router seam): calling the failover seam with the queue-timeout
    reason launches RunPod once, labels the RouteResult + the
    ``epm:backend-selected`` marker with ``gcp_queue_timeout_failover_runpod``,
    and carries the evidence onto the marker ``extra``."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=_spec(backend="gcp"),
        runpod_backend=rp,
        evidence={
            "source": "async_poller_queue_timeout",
            "current_phase": "terminal_queue_timeout",
        },
        reason=ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD
    assert len(rp.launches) == 1  # exactly once
    finals = _by_reason(captured_markers, ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD)
    assert finals
    assert (
        finals[-1]["extra"].get("gcp_workload_evidence", {}).get("source")
        == "async_poller_queue_timeout"
    )


def test_failover_seam_default_reason_unchanged_byte_for_byte(
    lease_store, marker_poster, captured_markers
):
    """#783 regression guard: adding the ``reason=`` param must NOT change the
    default #659 behavior — a call that OMITS ``reason`` still labels the result
    with the async workload-crash reason (byte-identical to pre-#783)."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=_spec(backend="gcp"),
        runpod_backend=rp,
        evidence={"source": "async_poller"},
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert result.reason == ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC


def test_queue_timeout_reason_distinct_from_crash_and_capacity_reasons():
    """#783: the queue-timeout reason VALUE is distinct from BOTH workload-crash
    reasons AND the capacity-exhaustion fallback reason, so the marker trail
    tells a stuck FLEX_START queue apart from a crash and a capacity miss."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        ROUTE_REASON_RUNPOD_FALLBACK,
    )

    reasons = {
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        ROUTE_REASON_RUNPOD_FALLBACK,
    }
    assert len(reasons) == 4  # all four are distinct strings
    assert ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD == "gcp_queue_timeout_failover_runpod"


class _ExplodingRunpodNoCompute(_BaseBackend):
    """RunPod double whose ``launch`` raises ``NoComputeAvailableError`` — the
    "RunPod also unavailable" failover branch (#659 sibling #4)."""

    @property
    def name(self) -> BackendKind:
        return "runpod"

    def launch(self, spec: RunSpec) -> RunHandle:
        raise NoComputeAvailableError("RunPod has no capacity for the failover re-launch")


# ---------------------------------------------------------------------------
# issue #669 — M3b: the GCP->RunPod failover is exactly-once under N CONCURRENT
# triggerers (the wedge classifier introduces a 2nd one), via the in-flock
# _lease_records_failover_of re-check + in-flock gcp_failover_of stamp.
# ---------------------------------------------------------------------------


def test_concurrent_failover_triggers_single_runpod_launch(
    lease_store, marker_poster, captured_markers
):
    """M2.6 (#669, the load-bearing atomicity test): two triggerers of the SAME
    GCP-crash failover (the poller-detected terminal_workload_wedged AND the
    watchdog-driven terminal_wedged_terminated, on the same handle) reach
    ``failover_to_runpod_after_async_workload_crash`` from overlapping
    processes. With the M3b in-flock re-check + stamp, exactly ONE launches
    RunPod; the second sees the stamp and short-circuits to the existing
    handle.

    Sequential calls on a SHARED LeaseStore model the serialized-by-flock
    outcome: the first stamps gcp_failover_of in-flock, the second's in-flock
    re-check matches and returns the existing lease with NO second launch."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    identity = {"pod_name": "eps-issue-137", "job_id": "instance-fake-1"}
    cfg = RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0)

    def _trigger():
        return failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=lease_store,
            now_fn=_clock(),
            config=cfg,
            gcp_failover_of_identity=identity,
        )

    first = _trigger()
    second = _trigger()  # the concurrent triggerer, serialized by the per-issue flock

    assert len(rp.launches) == 1  # EXACTLY ONCE — no double paid launch
    assert first.chosen_kind == "runpod"
    assert second.chosen_kind == "runpod"
    # The second returned the existing failover (no new launch).
    assert second.extra.get("failover_already_launched") is True
    # The lease records the GCP-crash identity this failover is OF.
    lease = lease_store.read(137)
    assert lease is not None
    assert lease.gcp_failover_of == identity


def test_distinct_gcp_crash_identity_gets_its_own_failover(
    lease_store, marker_poster, captured_markers
):
    """M3b keying: a GENUINELY-NEW GCP crash on the same issue (a fresh dispatch
    → a different pod_name/job_id identity) does NOT match the prior stamp, so
    it still gets its own single failover launch — the in-flock re-check is
    keyed to the GCP-crash identity, NOT to the issue."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    cfg = RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0)

    def _trigger(identity):
        return failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=lease_store,
            now_fn=_clock(),
            config=cfg,
            gcp_failover_of_identity=identity,
        )

    _trigger({"pod_name": "eps-issue-137", "job_id": "instance-crash-1"})
    _trigger({"pod_name": "eps-issue-137", "job_id": "instance-crash-2"})  # NEW crash

    assert len(rp.launches) == 2  # each distinct crash gets its own failover


def test_single_triggerer_failover_unchanged_when_identity_none(
    lease_store, marker_poster, captured_markers
):
    """#659 regression guard: with gcp_failover_of_identity=None (the legacy
    single-triggerer path) the failover behaves byte-for-byte as #659 — one
    launch, no in-flock short-circuit, no gcp_failover_of stamp."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=_spec(backend="gcp"),
        runpod_backend=rp,
        evidence={"source": "async_poller"},
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert len(rp.launches) == 1
    assert result.chosen_kind == "runpod"
    assert result.extra.get("failover_already_launched") is None
    lease = lease_store.read(137)
    assert lease is not None
    assert lease.gcp_failover_of is None  # NOT stamped on the legacy path


# ---------------------------------------------------------------------------
# CPU-only intent routing (#677)
# ---------------------------------------------------------------------------


def test_gcp_ladder_cpu_intent_single_ondemand_rung():
    """#677: the GCP ladder for a SHORT cpu-bigmem job yields exactly ONE
    on-demand CPU rung — no A100-40, no spot.

    The 1.0h budget is deliberately "short" (1.0h * max(1, gpu_count=0)=1 = 1.0
    GPU-h <= the 2 GPU-h EPS_GCP_SPOT_MAX_GPU_HOURS default), so a GPU intent at
    the same budget WOULD get a spot rung. The §4.2 short-circuit is the only
    thing suppressing it here — removing it turns this test RED.
    """
    from explore_persona_space.backends.router import _gcp_ladder_specs

    spec = RunSpec(issue=137, intent="cpu-bigmem", backend="gcp", time_budget_hours=1.0)
    rungs = _gcp_ladder_specs(spec)
    assert len(rungs) == 1
    _rung_spec, label = rungs[0]
    assert label == "ondemand_cpu"
    # No rung anywhere carries spot (label OR provisioning override).
    for rung_spec, lbl in rungs:
        assert "spot" not in lbl.lower()
        prov = (rung_spec.extra or {}).get("provisioning_model")
        assert prov is None or "spot" not in str(prov).lower()
    # And no A100-40 fallback rung.
    assert not any("a100_40" in lbl for _s, lbl in rungs)


def test_gcp_ladder_short_gpu_intent_DOES_get_spot_rung():
    """#677 positive control: the SAME 1.0h budget on a lora-7b intent DOES
    produce a spot rung — proving the budget really is 'short' and the CPU
    test's no-spot result is the short-circuit's doing, not an accidentally
    long job."""
    from explore_persona_space.backends.router import _gcp_ladder_specs

    spec = RunSpec(issue=137, intent="lora-7b", backend="gcp", time_budget_hours=1.0)
    rungs = _gcp_ladder_specs(spec)
    assert any("spot" in lbl.lower() for _s, lbl in rungs), [lbl for _s, lbl in rungs]


def test_ladder_cpu_intent_length_aware_still_yields_one_on_demand_rung():
    """#680 regression guard: the LENGTH-AWARE ladder must NOT promote a
    cpu-bigmem job (gpu_count == 0) onto spot / flex / A100-40 rungs via
    _is_short_job's gpu_count-floor-to-1, on EITHER length branch.

    #677 added the ``base.gpu_count == 0`` short-circuit; the #680 length-aware
    rewrite of _gcp_ladder_specs (spot-first short / flex-first long, plus the
    caller provisioning-model pin branch) re-authored the function around it.
    The existing #677 test exercises only the SHORT branch (1.0h budget). This
    pins BOTH branches under the post-#680 structure:

    - SHORT (1.0h budget, is_short=True): the short-circuit must precede the
      spot-first short branch.
    - UNKNOWN length (no budget, is_short=False): the short-circuit must precede
      the flex-first long branch.

    For each: exactly one rung, labelled ``ondemand_cpu``, with NO
    provisioning_model threaded (a pin would be a spot/flex/standard override
    the CPU rung must never carry). Re-dropping the short-circuit during a future
    ladder edit turns this RED.
    """
    from explore_persona_space.backends.router import _gcp_ladder_specs

    short_cpu = RunSpec(issue=137, intent="cpu-bigmem", backend="gcp", time_budget_hours=1.0)
    unknown_cpu = RunSpec(issue=137, intent="cpu-bigmem", backend="gcp")  # no budget -> long branch
    for spec in (short_cpu, unknown_cpu):
        rungs = _gcp_ladder_specs(spec)
        assert len(rungs) == 1, [lbl for _s, lbl in rungs]
        rung_spec, label = rungs[0]
        assert label == "ondemand_cpu", label
        # The single CPU rung must NOT thread any provisioning model (spot / flex
        # / standard) — it is the spec as-is, the on-demand-only short-circuit.
        assert (rung_spec.extra or {}).get("provisioning_model") is None
        # No spot / flex / A100-40 rung anywhere.
        assert not any("spot" in lbl.lower() for _s, lbl in rungs)
        assert not any("flexstart" in lbl.lower() for _s, lbl in rungs)
        assert not any("a100_40" in lbl for _s, lbl in rungs)


def test_router_cpu_intent_capacity_miss_no_runpod_fallback(
    lease_store, marker_poster, captured_markers
):
    """#677: a cpu-bigmem auto route whose GCP CPU rung capacity-misses raises
    CpuExhaustedNoRunpodLaneError and NEVER launches RunPod (RunPod is GPU-only,
    has no CPU lane). The terminal epm:backend-selected marker carries the
    distinct reason cpu_exhausted_no_runpod_lane.

    Removing the §4.2c _runpod_terminal_rung CPU guard turns this RED — it would
    instead launch RunPod / crash in the RunPod intent resolver.
    """
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD,
        CpuExhaustedNoRunpodLaneError,
    )

    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED",
            evidence={"matched_pattern": "RESOURCE_EXHAUSTED"},
        )
    )
    spec = RunSpec(issue=137, intent="cpu-bigmem", backend="auto", time_budget_hours=1.0)
    with pytest.raises(CpuExhaustedNoRunpodLaneError):
        route(
            spec,
            runpod_backend=rp,
            free_backends={"nibi": _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("x"))},
            gcp_backend=gcp,
            lease_store=lease_store,
            marker_poster=marker_poster,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    # RunPod NEVER attempted.
    assert len(rp.launches) == 0
    # The terminal marker carries the CPU-specific reason (NOT
    # auto_fallback_runpod / no_compute_available).
    cpu_terminals = _by_reason(captured_markers, ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD)
    assert cpu_terminals, captured_markers
    assert not _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)


# ---------------------------------------------------------------------------
# Cheap CPU-only intents: cpu-small / cpu-mid + GCP→RunPod CPU failover (#747)
# ---------------------------------------------------------------------------
#
# NOTE: the cpu-bigmem "still raises the typed terminal, never reaches RunPod"
# regression guard is the EXISTING test above —
# test_router_cpu_intent_capacity_miss_no_runpod_fallback — which already uses
# intent="cpu-bigmem" (absent from RUNPOD_CPU_INSTANCE_FOR_INTENT). We do NOT
# duplicate it under a cpu-bigmem-named alias (single canonical name).


def _short_cpu_small_spec(issue: int = 747) -> RunSpec:
    """A SHORT (1 CPU-h) auto-routing cpu-small spec.

    _is_short_job floors gpu_count to 1 via max(1, gpu_count) in
    _estimated_gpu_hours, so 1.0h * 1 = 1.0 CPU-h <= the 2 GPU-h
    EPS_GCP_SPOT_MAX_GPU_HOURS default -> short -> the cheap CPU intent earns a
    GCP spot rung. (time_budget_hours MUST be threaded or the spot rung never
    fires — Methodology concern #1.)
    """
    return RunSpec(issue=issue, intent="cpu-small", backend="gcp", time_budget_hours=1.0)


def test_gcp_ladder_cpu_small_short_yields_spot_then_ondemand():
    """#747: a SHORT cpu-small job yields [spot_cpu, ondemand_cpu] — the cheap
    CPU intent IS spot-eligible on a short job (unlike cpu-bigmem). NO flex,
    NO A100-40 rung."""
    from explore_persona_space.backends.router import _gcp_ladder_specs

    rungs = _gcp_ladder_specs(_short_cpu_small_spec())
    labels = [lbl for _s, lbl in rungs]
    assert labels == ["spot_cpu", "ondemand_cpu"], labels
    # The spot rung threads SPOT provisioning; the on-demand rung threads none.
    spot_spec, _spot_lbl = rungs[0]
    assert str((spot_spec.extra or {}).get("provisioning_model")).upper() == "SPOT"
    ondemand_spec, _od_lbl = rungs[1]
    assert (ondemand_spec.extra or {}).get("provisioning_model") is None
    # No flex / A100-40 rung anywhere.
    assert not any("flexstart" in lbl for lbl in labels)
    assert not any("a100_40" in lbl for lbl in labels)


def test_gcp_ladder_cpu_small_unknown_length_ondemand_only():
    """#747: an UNKNOWN-length cpu-small job (no time budget) yields a single
    on-demand rung — NOT short, so no spot (preemption too costly). This is the
    correct fail-safe: a CPU caller that does NOT thread time_budget_hours gets
    reliable on-demand, never a spot rung."""
    from explore_persona_space.backends.router import _gcp_ladder_specs

    spec = RunSpec(issue=747, intent="cpu-small", backend="gcp")  # no budget
    rungs = _gcp_ladder_specs(spec)
    labels = [lbl for _s, lbl in rungs]
    assert labels == ["ondemand_cpu"], labels
    assert not any("spot" in lbl for lbl in labels)


def test_gcp_ladder_cpu_bigmem_still_single_ondemand_rung():
    """#747 regression guard on #677: cpu-bigmem (NOT in the RunPod-CPU map)
    STILL yields exactly one on-demand rung even on a SHORT job — no spot, no
    flex, no A100-40. The #747 spot-rung branch must NOT leak into cpu-bigmem."""
    from explore_persona_space.backends.router import _gcp_ladder_specs

    short = RunSpec(issue=747, intent="cpu-bigmem", backend="gcp", time_budget_hours=1.0)
    rungs = _gcp_ladder_specs(short)
    labels = [lbl for _s, lbl in rungs]
    assert labels == ["ondemand_cpu"], labels
    assert not any("spot" in lbl for lbl in labels)
    assert (rungs[0][0].extra or {}).get("provisioning_model") is None


def test_router_cpu_small_capacity_miss_falls_over_to_runpod(
    lease_store, marker_poster, captured_markers
):
    """#747: a cpu-small auto route whose GCP CPU lane is exhausted FALLS OVER
    to RunPod CPU (does NOT raise CpuExhaustedNoRunpodLaneError). GCP-first is
    preserved: the FIRST attempt is GCP, and RunPod is reached only AFTER it.
    RunPod is launched EXACTLY once, carrying --intent cpu-small."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD,
        CpuExhaustedNoRunpodLaneError,
    )

    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED",
            evidence={"matched_pattern": "RESOURCE_EXHAUSTED"},
        )
    )
    spec = RunSpec(issue=747, intent="cpu-small", backend="auto", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=rp,
        free_backends={"nibi": _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("x"))},
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    # Fell over to RunPod (NOT the CPU terminal).
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    # ORDER-SENSITIVE: a GCP attempt precedes the RunPod launch (GCP-first;
    # Statistics concern #1 — assert ordering, not just len(rp.launches) == 1).
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    gcp_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "gcp"]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert gcp_idxs, outcomes
    assert runpod_idxs, outcomes
    assert max(gcp_idxs) < runpod_idxs[-1], outcomes
    # The RunPod launch carries --intent cpu-small (resolved by Surface 5 to the
    # RunPod CPU instance_id), and NO CPU-terminal marker was posted.
    assert rp.launches[0].intent == "cpu-small"
    assert not _by_reason(captured_markers, ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD)
    # The typed terminal was NOT raised (sanity — route returned a result).
    assert not isinstance(result, CpuExhaustedNoRunpodLaneError)


def test_runpod_cpu_instance_map_is_single_source_of_truth():
    """#747: gpu_heuristics resolves CPU intents from the SAME router map
    (single source of truth, NOT a duplicated copy). Asserts every router-map
    key+value round-trips through gpu_heuristics.resolve_cpu_intent, and a
    non-CPU intent resolves to None."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import gpu_heuristics

    from explore_persona_space.backends.router import RUNPOD_CPU_INSTANCE_FOR_INTENT

    for intent, instance_id in RUNPOD_CPU_INSTANCE_FOR_INTENT.items():
        assert gpu_heuristics.resolve_cpu_intent(intent) == instance_id
    # A GPU intent (and any unmapped string) is NOT a CPU intent.
    assert gpu_heuristics.resolve_cpu_intent("lora-7b") is None
    assert gpu_heuristics.resolve_cpu_intent("cpu-bigmem") is None


# ---------------------------------------------------------------------------
# #1010 — RunPod CPU-fallback footprint feasibility gate (incident #958)
# ---------------------------------------------------------------------------


def _cpu_caps_for_intent(intent: str):
    """The RunPodCpuInstanceCaps row the gate reads for a mapped CPU intent."""
    from explore_persona_space.backends.router import (
        RUNPOD_CPU_INSTANCE_CAPS,
        RUNPOD_CPU_INSTANCE_FOR_INTENT,
    )

    return RUNPOD_CPU_INSTANCE_CAPS[RUNPOD_CPU_INSTANCE_FOR_INTENT[intent]]


def _exhausted_gcp_double() -> _GcpBackendDouble:
    """A GCP double whose every create capacity-misses (the #747 fallover shape)."""
    return _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED",
            evidence={"matched_pattern": "RESOURCE_EXHAUSTED"},
        )
    )


@pytest.mark.parametrize("intent", ["cpu-small", "cpu-mid"])
def test_runpod_cpu_fallback_refuses_oversized_disk_requirement(
    lease_store, marker_poster, captured_markers, intent
):
    """#1010: a mapped CPU intent whose plan-stated boot_disk_gb exceeds the
    instance's container-disk cap raises the TYPED CpuFallbackInfeasibleError
    at the terminal rung, BEFORE any RunPod launch — no pod is provisioned,
    and the terminal marker carries the DISTINCT reason
    cpu_fallback_infeasible_for_plan (never auto_fallback_runpod).
    Parametrized over BOTH mapped intents (per-instance caps differ)."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
        CpuFallbackInfeasibleError,
    )

    caps = _cpu_caps_for_intent(intent)
    rp = _PassiveRunpod()
    spec = RunSpec(
        issue=1010,
        intent=intent,
        backend="auto",
        time_budget_hours=1.0,
        extra={"boot_disk_gb": caps.max_container_disk_gb + 20},
    )
    with pytest.raises(CpuFallbackInfeasibleError):
        route(
            spec,
            runpod_backend=rp,
            free_backends={"nibi": _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("x"))},
            gcp_backend=_exhausted_gcp_double(),
            lease_store=lease_store,
            marker_poster=marker_poster,
            config=RouterConfig(
                free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99
            ),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    # RunPod NEVER attempted — the refusal is pre-API, pre-provision.
    assert len(rp.launches) == 0
    assert _by_reason(captured_markers, ROUTE_REASON_CPU_FALLBACK_INFEASIBLE), captured_markers
    assert not _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)


def test_runpod_cpu_fallback_refuses_oversized_ram_requirement(
    lease_store, marker_poster, captured_markers
):
    """#1010: min_ram_gb above the instance's FIXED RAM (cpu3c-8-16 = 16 GB)
    refuses with the same typed terminal — the #958 shape (a 32 GB-RAM plan
    on a 16 GB pod) now refuses at $0 instead of after a provision cycle."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
        CpuFallbackInfeasibleError,
    )

    rp = _PassiveRunpod()
    spec = RunSpec(
        issue=1010,
        intent="cpu-mid",
        backend="auto",
        time_budget_hours=1.0,
        extra={"min_ram_gb": 32},
    )
    with pytest.raises(CpuFallbackInfeasibleError):
        route(
            spec,
            runpod_backend=rp,
            free_backends={"nibi": _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("x"))},
            gcp_backend=_exhausted_gcp_double(),
            lease_store=lease_store,
            marker_poster=marker_poster,
            config=RouterConfig(
                free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99
            ),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert len(rp.launches) == 0
    assert _by_reason(captured_markers, ROUTE_REASON_CPU_FALLBACK_INFEASIBLE), captured_markers


def test_runpod_cpu_fallback_feasible_requirement_launches_and_preserves_extra(
    lease_store, marker_poster, captured_markers
):
    """#1010 control: a feasible stated footprint (below the cap) still falls
    over to RunPod exactly as the no-requirement #747 path does, with
    boot_disk_gb PRESERVED on the launched spec.extra (RunPodBackend.launch
    threads it into --container-disk-gb) and NO infeasible marker."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
    )

    caps = _cpu_caps_for_intent("cpu-mid")
    rp = _PassiveRunpod()
    spec = RunSpec(
        issue=1010,
        intent="cpu-mid",
        backend="auto",
        time_budget_hours=1.0,
        extra={"boot_disk_gb": caps.max_container_disk_gb - 20},
    )
    result = route(
        spec,
        runpod_backend=rp,
        free_backends={"nibi": _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("x"))},
        gcp_backend=_exhausted_gcp_double(),
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, max_gcp_attempts_per_day=99),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    assert rp.launches[0].extra["boot_disk_gb"] == caps.max_container_disk_gb - 20
    assert not _by_reason(captured_markers, ROUTE_REASON_CPU_FALLBACK_INFEASIBLE)


def test_runpod_cpu_instance_caps_cover_every_mapped_instance():
    """#1010 completeness (the #841/#940 pattern): every instance_id the
    intent map can route MUST have a caps row — a future intent/instance
    cannot land without deciding its caps (the gate's direct [] lookup would
    otherwise KeyError at route time instead of failing CI at the adding PR)."""
    from explore_persona_space.backends.router import (
        RUNPOD_CPU_INSTANCE_CAPS,
        RUNPOD_CPU_INSTANCE_FOR_INTENT,
        RunPodCpuInstanceCaps,
    )

    assert set(RUNPOD_CPU_INSTANCE_CAPS) == set(RUNPOD_CPU_INSTANCE_FOR_INTENT.values())
    for caps in RUNPOD_CPU_INSTANCE_CAPS.values():
        assert isinstance(caps, RunPodCpuInstanceCaps)
        assert caps.vcpu > 0 and caps.ram_gb > 0 and caps.max_container_disk_gb > 0


def test_async_failover_seam_cpu_infeasible_disk_raises_typed(lease_store, marker_poster):
    """#1010: the ASYNC failover seam (poller-detected GCP crash / queue
    timeout) inherits the feasibility gate via _runpod_terminal_rung — an
    oversized cpu-mid spec raises the typed terminal instead of provisioning
    an undersized fallback pod (the #958 shape on the async paths)."""
    from explore_persona_space.backends.router import (
        CpuFallbackInfeasibleError,
        failover_to_runpod_after_async_workload_crash,
    )

    caps = _cpu_caps_for_intent("cpu-mid")
    rp = _PassiveRunpod()
    spec = RunSpec(
        issue=1010,
        intent="cpu-mid",
        backend="runpod",
        extra={"boot_disk_gb": caps.max_container_disk_gb + 30},
    )
    with pytest.raises(CpuFallbackInfeasibleError):
        failover_to_runpod_after_async_workload_crash(
            spec=spec,
            runpod_backend=rp,
            lease_store=lease_store,
            marker_poster=marker_poster,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
            now_fn=_clock(),
        )
    assert len(rp.launches) == 0


# ---------------------------------------------------------------------------
# #774 round 2 — RouteAttempt.evidence carries the GCP per-zone fan-out to the
# epm:backend-selected marker. A GcpProvisioningError whose evidence holds
# per_zone_attempts must round-trip through the catch site -> _attempt_to_dict
# -> marker note attempts[].evidence intact; an attempt with NO evidence (the
# common shape) must serialize byte-identically to the pre-#774 7-field dict.
# ---------------------------------------------------------------------------


def test_route_attempt_evidence_field_round_trips_per_zone_attempts(
    lease_store, marker_poster, captured_markers
):
    """An explicit GCP override that capacity-fails with a per-zone fan-out on
    its GcpProvisioningError.evidence surfaces that fan-out on the terminal
    epm:backend-selected marker's attempts[0].evidence — both zones, the 5
    per-zone keys preserved, and the summary string intact."""
    from explore_persona_space.backends.router import ROUTE_REASON_NO_COMPUTE

    per_zone = [
        {
            "zone": "us-central1-a",
            "returncode": 1,
            "matched_pattern": "ZONE_RESOURCE_POOL_EXHAUSTED",
            "elapsed_s": 1.5,
            "stderr_tail": "us-central1-a tail",
        },
        {
            "zone": "us-central1-c",
            "returncode": 1,
            "matched_pattern": "ZONE_RESOURCE_POOL_EXHAUSTED",
            "elapsed_s": 1.7,
            "stderr_tail": "us-central1-c tail",
        },
    ]
    summary = "all 2 zone(s) [a, c] exhausted"
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "capacity exhausted",
            evidence={
                "matched_pattern": "ZONE_RESOURCE_POOL_EXHAUSTED",
                "per_zone_attempts": per_zone,
                "zones_attempted_summary": summary,
            },
        )
    )
    with pytest.raises(NoComputeAvailableError):
        route(
            _spec(backend="gcp"),
            runpod_backend=_ExplodingRunpod(),
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
    attempts = terminal[-1]["attempts"]
    gcp_attempt = next(a for a in attempts if a["kind"] == "gcp")
    assert gcp_attempt["evidence"]["per_zone_attempts"] == per_zone
    assert gcp_attempt["evidence"]["zones_attempted_summary"] == summary
    # The 5 per-zone keys survived the JSON round-trip on every record.
    for entry in gcp_attempt["evidence"]["per_zone_attempts"]:
        assert set(entry) == {
            "zone",
            "returncode",
            "matched_pattern",
            "elapsed_s",
            "stderr_tail",
        }


def test_route_attempt_dict_omits_evidence_when_empty():
    """A default-constructed RouteAttempt (no evidence) serializes to the
    pre-#774 7-field dict — no 'evidence' key — so every existing marker reader
    is unaffected."""
    attempt = RouteAttempt(
        kind="gcp",
        cluster=None,
        est_start_seconds_raw=0.0,
        est_start_seconds_clamped=0.0,
        outcome="provisioning_failure",
        detail="rung A100-80/SPOT: capacity miss",
        elapsed_seconds=1.234,
    )
    d = _attempt_to_dict(attempt)
    assert "evidence" not in d
    assert set(d) == {
        "kind",
        "cluster",
        "est_start_seconds_raw",
        "est_start_seconds_clamped",
        "outcome",
        "detail",
        "elapsed_seconds",
    }


# ---------------------------------------------------------------------------
# #909 — the async failover seam opts custom-workload specs into the RunPod
# execution leg (spec.extra["execute_workload"] = True); hydra-args specs
# arrive WITHOUT it (+ a loud named-residual log line). These pin AC4.
# ---------------------------------------------------------------------------


def test_async_failover_sets_execute_workload_on_custom_workload_spec(lease_store, marker_poster):
    """#909 AC4: a custom-workload spec dispatched through
    ``failover_to_runpod_after_async_workload_crash`` arrives at
    ``backend.launch`` with ``extra["execute_workload"] is True`` — the
    automated no-experimenter failover paths are the execution leg's
    primary consumer."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=RunSpec(
            issue=909,
            intent="lora-7b",
            backend="gcp",
            workload_cmd="bash scripts/issue909_dispatch.sh",
            extra={"repo_branch": "issue-909"},
        ),
        runpod_backend=rp,
        evidence={"source": "async_poller"},
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    launched = rp.launches[0]
    assert launched.extra.get("execute_workload") is True
    assert launched.workload_cmd == "bash scripts/issue909_dispatch.sh"
    # Pre-existing extra keys survive the opt-in replace.
    assert launched.extra.get("repo_branch") == "issue-909"


def test_async_failover_hydra_spec_arrives_without_execute_workload(
    lease_store, marker_poster, caplog
):
    """#909 AC4 (negative): a hydra-args spec is NOT opted in (the RunPod
    execution leg cannot execute a hydra run) and the seam logs the named
    residual LOUD."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    with caplog.at_level("WARNING", logger="explore_persona_space.backends.router"):
        failover_to_runpod_after_async_workload_crash(
            spec=RunSpec(
                issue=909,
                intent="lora-7b",
                backend="gcp",
                hydra_args=("condition=c1", "seed=42"),
            ),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=lease_store,
            now_fn=_clock(),
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        )
    assert len(rp.launches) == 1
    assert "execute_workload" not in (rp.launches[0].extra or {})
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "hydra" in warning and "no backend-side executor" in warning


# ---------------------------------------------------------------------------
# #934: lane-suffixed attempt ids
# ---------------------------------------------------------------------------


def test_threaded_attempt_id_carries_lane_suffix():
    """A spec whose extra carries lane_suffix mints att-<ts>-<suffix>, so
    two lanes launched in the SAME second keep disjoint HF crash-persist
    (issue<N>_partial/<attempt>/) + sentinel namespaces; without a suffix
    the shape is byte-identical to the pre-#934 mint."""
    writes: list[Any] = []
    spec = RunSpec(issue=934, intent="lora-7b", backend="nibi", extra={"lane_suffix": "cpu"})
    new_spec, lease = _thread_attempt_id_into(spec, None, writes.append)
    assert re.fullmatch(r"att-\d{8}-\d{6}-cpu", new_spec.extra["attempt_id"])
    assert lease.attempt_id == new_spec.extra["attempt_id"]

    unsuffixed = RunSpec(issue=934, intent="lora-7b", backend="nibi")
    new_unsuffixed, _lease2 = _thread_attempt_id_into(unsuffixed, None, writes.append)
    assert re.fullmatch(r"att-\d{8}-\d{6}", new_unsuffixed.extra["attempt_id"])


def test_make_attempt_id_rejects_malformed_lane_suffix():
    """Fail loud, never strip: a malformed suffix raises out of the mint."""
    from explore_persona_space.backends.router import _make_attempt_id

    assert re.fullmatch(r"att-\d{8}-\d{6}", _make_attempt_id())
    assert re.fullmatch(r"att-\d{8}-\d{6}-cpu", _make_attempt_id("cpu"))
    with pytest.raises(ValueError):
        _make_attempt_id("Not_Valid")


def test_caller_pinned_attempt_id_wins_over_lane_suffix_mint():
    """A caller-pinned extra['attempt_id'] still takes precedence (the
    explicit re-attach tooling contract) — the suffix mint never clobbers
    it."""
    writes: list[Any] = []
    spec = RunSpec(
        issue=934,
        intent="lora-7b",
        backend="nibi",
        extra={"lane_suffix": "cpu", "attempt_id": "att-pinned-1"},
    )
    new_spec, lease = _thread_attempt_id_into(spec, None, writes.append)
    assert new_spec.extra["attempt_id"] == "att-pinned-1"
    assert lease.attempt_id == "att-pinned-1"


# ---------------------------------------------------------------------------
# #940 — GCP-only GPU intent translation at the RunPod launch paths
#
# The RunPod terminal rung (and the explicit `backend: runpod` override)
# translate GCP-only GPU intents (capture-7b / lora / lora-7b-h100) to the
# nearest same-or-narrower RunPod-provisionable intent via the router-owned
# RUNPOD_INTENT_FOR_GCP_INTENT map, so the sanctioned last-rung fallback
# actually fires instead of dying in gpu_heuristics.resolve_intent's KeyError
# (the #841 incident: `provision --issue 841 --intent capture-7b` exit 1 →
# NoComputeAvailableError despite live RunPod 1-GPU capacity). An unmapped
# GCP GPU intent (eval-h100, in RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS)
# fails loud PRE-launch naming the missing map row.
# ---------------------------------------------------------------------------


def test_gcp_only_intent_exhausted_ladder_fires_runpod_with_translated_intent(
    lease_store, marker_poster, captured_markers
):
    """T1 / the #841 regression test: a capture-7b auto route with every GCP
    rung + SLURM exhausted launches RunPod EXACTLY ONCE with the translated
    `--intent eval`, RunPod LAST in the attempt trail, and the marker extra
    carries the translation record."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)  # never starts
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        RunSpec(issue=940, intent="capture-7b", backend="auto", time_budget_hours=1.0),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=99,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_RUNPOD_FALLBACK
    assert len(rp.launches) == 1
    # The launched spec carries the TRANSLATED RunPod-provisionable intent.
    assert rp.launches[0].intent == "eval"
    # RunPod is the FINAL attempt, behind every GCP rung + the nibi park-fail
    # (capture-7b is in INTENT_A100_40_FALLBACK, so the short 5-rung ladder
    # applies exactly as lora-7b's).
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    gcp_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "gcp"]
    nibi_idxs = [i for i, (k, _o) in enumerate(outcomes) if k == "nibi"]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert len(gcp_idxs) == 5, outcomes  # all five short-ladder rungs attempted + failed
    assert nibi_idxs, "the free SLURM lane must have been attempted"
    assert runpod_idxs and runpod_idxs[-1] == len(outcomes) - 1
    assert max(gcp_idxs) < runpod_idxs[-1]
    assert max(nibi_idxs) < runpod_idxs[-1]
    finals = _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)
    assert finals
    assert finals[-1]["extra"]["runpod_intent_translation"] == {
        "from": "capture-7b",
        "to": "eval",
    }


def test_runpod_native_intent_terminal_rung_untranslated_no_marker_key(
    lease_store, marker_poster, captured_markers
):
    """T2: an identity-row intent (lora-7b) passes through the terminal rung
    verbatim — no translation record on the marker (byte-identical to the
    pre-#940 marker shape for existing intents)."""
    rp = _PassiveRunpod()
    nibi = _FreeLaneBackend(kind="nibi", starts_when=10**9)
    gcp = _GcpBackendDouble(
        launch_raises=GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        free_backends={"nibi": nibi},
        gcp_backend=gcp,
        lease_store=lease_store,
        is_started=lambda _b, _h: False,
        is_live_after_cancel=lambda _b, _h: False,
        marker_poster=marker_poster,
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=99,
        ),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert len(rp.launches) == 1
    assert rp.launches[0].intent == "lora-7b"
    finals = _by_reason(captured_markers, ROUTE_REASON_RUNPOD_FALLBACK)
    assert finals
    assert "runpod_intent_translation" not in finals[-1]["extra"]


@pytest.mark.parametrize("reason_name", ["async_crash", "queue_timeout"])
def test_async_failover_translates_gcp_only_intent(
    lease_store, marker_poster, captured_markers, reason_name
):
    """T3: the async poller failover seam (#659) AND the queue-timeout
    failover (#783) — both via failover_to_runpod_after_async_workload_crash —
    inherit the translation from the shared terminal rung: a capture-7b
    failover launches RunPod with `--intent eval`, and the marker extra
    carries BOTH the translation record and the crash evidence."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        failover_to_runpod_after_async_workload_crash,
    )

    reason = {
        "async_crash": ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        "queue_timeout": ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
    }[reason_name]
    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=RunSpec(issue=940, intent="capture-7b", backend="auto", workload_cmd="bash x.sh"),
        runpod_backend=rp,
        evidence={"source": "async_poller"},
        reason=reason,
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == reason
    assert len(rp.launches) == 1
    assert rp.launches[0].intent == "eval"
    finals = _by_reason(captured_markers, reason)
    assert finals
    assert finals[-1]["extra"]["runpod_intent_translation"] == {
        "from": "capture-7b",
        "to": "eval",
    }
    assert finals[-1]["extra"]["gcp_workload_evidence"]["source"] == "async_poller"


def test_unmapped_gcp_gpu_intent_fails_loud_naming_map_row(
    lease_store, marker_poster, captured_markers
):
    """T4: a GCP-mapped GPU intent with NO translation row (eval-h100) fails
    loud BEFORE any provision attempt, inside the existing
    NoComputeAvailableError / no_compute_available terminal shape, with a
    message naming the missing RUNPOD_INTENT_FOR_GCP_INTENT row."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    with pytest.raises(NoComputeAvailableError) as excinfo:
        failover_to_runpod_after_async_workload_crash(
            spec=RunSpec(issue=940, intent="eval-h100", backend="auto"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=lease_store,
            now_fn=_clock(),
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        )
    assert "eval-h100" in str(excinfo.value)
    assert "RUNPOD_INTENT_FOR_GCP_INTENT" in str(excinfo.value)
    # NO provision was attempted (the failure is pre-launch).
    assert rp.launches == []
    # A no_compute_available terminal marker was posted.
    from explore_persona_space.backends.router import ROUTE_REASON_NO_COMPUTE

    assert _by_reason(captured_markers, ROUTE_REASON_NO_COMPUTE)


def test_translated_runpod_intent_helper_unit():
    """T4 companion: the helper itself raises ValueError naming the missing
    row for eval-h100, and returns the record shape for a real translation."""
    from explore_persona_space.backends.router import _translated_runpod_intent

    with pytest.raises(ValueError, match="RUNPOD_INTENT_FOR_GCP_INTENT"):
        _translated_runpod_intent(RunSpec(issue=1, intent="eval-h100", backend="auto"))
    assert _translated_runpod_intent(RunSpec(issue=1, intent="capture-7b", backend="auto")) == (
        "eval",
        {"from": "capture-7b", "to": "eval"},
    )
    assert _translated_runpod_intent(RunSpec(issue=1, intent="lora-7b", backend="auto")) == (
        "lora-7b",
        None,
    )


def test_translation_map_total_over_gcp_gpu_intents():
    """T5 completeness/drift pin: every gpu_count>0 key of gcp.INTENT_TO_MACHINE
    is EITHER in the translation map OR in the deliberate-gap set (disjointly) —
    a future GCP intent added without deciding its RunPod fate fails HERE, at
    the adding PR, instead of crashing at the terminal rung months later (the
    #841 failure mode, recurrence-proofed)."""
    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE
    from explore_persona_space.backends.router import (
        RUNPOD_INTENT_FOR_GCP_INTENT,
        RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS,
    )

    gpu_keys = {k for k, m in INTENT_TO_MACHINE.items() if m.gpu_count > 0}
    covered = set(RUNPOD_INTENT_FOR_GCP_INTENT) | set(RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS)
    assert gpu_keys == covered, (
        f"gcp.INTENT_TO_MACHINE GPU intents not reconciled with the RunPod "
        f"translation map: missing={sorted(gpu_keys - covered)} "
        f"stale={sorted(covered - gpu_keys)}"
    )
    assert not set(RUNPOD_INTENT_FOR_GCP_INTENT) & RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS


def test_translation_never_widens_gpu_count_and_targets_provisionable():
    """T6 property test: every translation target is RunPod-provisionable
    (a gpu_heuristics.INTENTS key) at a same-or-narrower GPU width than the
    GCP machine it translates from (never widen — constraint 2)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import gpu_heuristics

    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE
    from explore_persona_space.backends.router import RUNPOD_INTENT_FOR_GCP_INTENT

    for gcp_intent, runpod_intent in RUNPOD_INTENT_FOR_GCP_INTENT.items():
        assert runpod_intent in gpu_heuristics.INTENTS, (
            f"{gcp_intent!r} -> {runpod_intent!r}: target is not a RunPod-provisionable "
            f"intent (gpu_heuristics.INTENTS)"
        )
        assert (
            gpu_heuristics.INTENTS[runpod_intent].gpu_count
            <= INTENT_TO_MACHINE[gcp_intent].gpu_count
        ), f"{gcp_intent!r} -> {runpod_intent!r} WIDENS the GPU count"


def test_explicit_runpod_override_translates_gcp_only_intent(
    lease_store, marker_poster, captured_markers
):
    """T7: the explicit `backend: runpod` override (the documented manual
    recovery for the #667 hung-GCP-VM gap) translates a GCP-only intent too —
    previously it could only crash (uncaught KeyError class), so no working
    behavior is altered."""
    rp = _PassiveRunpod()
    result = route(
        RunSpec(issue=940, intent="capture-7b", backend="runpod"),
        runpod_backend=rp,
        lease_store=lease_store,
        marker_poster=marker_poster,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(rp.launches) == 1
    assert rp.launches[0].intent == "eval"
    finals = _by_reason(captured_markers, ROUTE_REASON_OVERRIDE)
    assert finals
    assert finals[-1]["extra"]["runpod_intent_translation"] == {
        "from": "capture-7b",
        "to": "eval",
    }


def test_explicit_runpod_override_runpod_only_intent_verbatim(
    lease_store, marker_poster, captured_markers
):
    """T7 companion: a RunPod-only intent (ft-70b — NOT GCP-mapped) passes
    through the override verbatim, no raise, no translation record
    (byte-identical to pre-#940). NOTE: ft-70b is reachable ONLY via the
    override path — the terminal rung's pre-existing CPU guard
    (machine_for_intent) raises gcp.py's own ValueError on non-GCP intents
    before the translation helper would run there."""
    rp = _PassiveRunpod()
    result = route(
        RunSpec(issue=940, intent="ft-70b", backend="runpod"),
        runpod_backend=rp,
        lease_store=lease_store,
        marker_poster=marker_poster,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(rp.launches) == 1
    assert rp.launches[0].intent == "ft-70b"
    finals = _by_reason(captured_markers, ROUTE_REASON_OVERRIDE)
    assert finals
    assert "runpod_intent_translation" not in finals[-1]["extra"]


def test_explicit_runpod_override_gpu_intent_preserves_boot_disk_gb(
    lease_store, marker_poster, captured_markers
):
    """#1118 route-side leg (the exact #1112 manual-pivot shape): a GPU spec
    with a stated boot_disk_gb reaches RunPodBackend.launch through
    _override_runpod with the extra intact — the GPU sibling of the CPU-shaped
    test_runpod_cpu_fallback_feasible_requirement_launches_and_preserves_extra
    (launch then threads it into --volume-gb, pinned in
    tests/test_runpod_workload_exec.py)."""
    rp = _PassiveRunpod()
    result = route(
        RunSpec(issue=1118, intent="lora-7b", backend="runpod", extra={"boot_disk_gb": 575}),
        runpod_backend=rp,
        lease_store=lease_store,
        marker_poster=marker_poster,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_OVERRIDE
    assert len(rp.launches) == 1
    assert rp.launches[0].intent == "lora-7b"
    assert rp.launches[0].extra["boot_disk_gb"] == 575


def test_explicit_runpod_override_unmapped_gcp_intent_raises_valueerror(
    lease_store, marker_poster, captured_markers
):
    """Ensemble-review addition: the OVERRIDE x eval-h100 cell fails loud with
    the helper's RAW ValueError (a config error, per §4d — NOT wrapped into
    NoComputeAvailableError), pre-launch, naming the missing map row. Pins the
    exception shape so a future "wrap it silently" regression fails here."""
    rp = _PassiveRunpod()
    with pytest.raises(ValueError, match="RUNPOD_INTENT_FOR_GCP_INTENT"):
        route(
            RunSpec(issue=940, intent="eval-h100", backend="runpod"),
            runpod_backend=rp,
            lease_store=lease_store,
            marker_poster=marker_poster,
        )
    assert rp.launches == []


def test_translation_helper_cpu_intents_pass_through_verbatim():
    """T8: CPU intents (gpu_count == 0 in INTENT_TO_MACHINE) pass through the
    helper verbatim — the #677/#747 CPU semantics are untouched at the helper
    level (the end-to-end pins are the existing CPU tests above)."""
    from explore_persona_space.backends.router import _translated_runpod_intent

    for cpu_intent in ("cpu-small", "cpu-mid", "cpu-bigmem"):
        assert _translated_runpod_intent(RunSpec(issue=1, intent=cpu_intent, backend="auto")) == (
            cpu_intent,
            None,
        )


# ---------------------------------------------------------------------------
# #954 — PARTIAL RunPod terminal-rung failure: pod PROVISIONED, workload start
# FAILED (RunPodWorkloadStartError carrying the partial handle). The rung must
# persist the SAME launch records the success path writes (on_launched sidecar
# hook + in-flock lease incl. the M3b gcp_failover_of stamp), record a DISTINCT
# RouteAttempt + terminal marker (runpod_workload_start_failed), and re-raise
# TYPED — never collapse into NoComputeAvailableError (a pod exists and BILLS;
# the #931 incident's mislabel invited a second paid dispatch).
# ---------------------------------------------------------------------------


class _WorkloadStartFailedRunpod(_BaseBackend):
    """RunPod double modeling the #954 PARTIAL failure: ``launch`` provisions
    (records the spec) then raises the typed workload-start error CARRYING the
    partial handle — the exact shape ``RunPodBackend.launch`` produces when the
    #909 execution leg fails after the pod exists."""

    def __init__(self) -> None:
        self.launches: list[RunSpec] = []

    def launch(self, spec: RunSpec) -> RunHandle:
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
                "workload_executed": False,
                "workload_start_error": "branch sync timed out (ssh TimeoutExpired)",
            },
        )
        raise RunPodWorkloadStartError("branch sync timed out (ssh TimeoutExpired)", handle=partial)


class _WorkloadStartFailedNoHandleRunpod(_BaseBackend):
    """The HANDLE-LESS typed error (the pre-provision guard shape): nothing was
    provisioned, nothing bills — the rung's blanket NoCompute branch applies."""

    def launch(self, spec: RunSpec) -> RunHandle:
        from explore_persona_space.backends.runpod import RunPodWorkloadStartError

        raise RunPodWorkloadStartError(
            "execute_workload requested with empty workload_cmd — refusing before provisioning"
        )


def test_runpod_terminal_rung_workload_start_failed_persists_launch_records_and_reraises_typed(
    lease_store, marker_poster, captured_markers
):
    """#954 AC1: a typed-with-handle workload-start failure at the terminal rung
    (i) invokes on_launched with the PARTIAL handle, (ii) writes the in-flock
    lease incl. the M3b gcp_failover_of stamp, (iii) records the DISTINCT
    RouteAttempt + terminal marker, and (iv) re-raises the TYPED error — never
    NoComputeAvailableError."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_NO_COMPUTE,
        ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED,
        failover_to_runpod_after_async_workload_crash,
    )
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    rp = _WorkloadStartFailedRunpod()
    hooked: list[RunHandle] = []
    identity = {"pod_name": "eps-issue-137", "job_id": "instance-fake-1"}
    with pytest.raises(RunPodWorkloadStartError) as ei:
        failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            on_launched=hooked.append,
            lease_store=lease_store,
            now_fn=_clock(),
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            gcp_failover_of_identity=identity,
        )
    # (iv) The TYPED class, not the NoCompute collapse.
    assert not isinstance(ei.value, NoComputeAvailableError)
    assert ei.value.handle is not None
    assert ei.value.handle.pod_name == "pod-137"
    # (i) on_launched fired exactly once, with the PARTIAL handle.
    assert len(hooked) == 1
    assert hooked[0] is ei.value.handle
    # (ii) The in-flock lease records the submit + the M3b identity stamp.
    lease = lease_store.read(137)
    assert lease is not None
    assert lease.backend == "runpod"
    assert lease.gcp_failover_of == identity
    # (iii) Terminal marker with the DISTINCT reason + the RouteAttempt outcome.
    finals = _by_reason(captured_markers, ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED)
    assert finals
    attempt = finals[-1]["attempts"][-1]
    assert attempt["outcome"] == "runpod_workload_start_failed"
    assert "pod-137" in attempt["detail"]
    assert "RUNNING for" in attempt["detail"]
    # No NoCompute marker was posted for this launch.
    assert not _by_reason(captured_markers, ROUTE_REASON_NO_COMPUTE)


def test_runpod_terminal_rung_workload_start_failed_without_handle_keeps_no_compute(
    lease_store, marker_poster, captured_markers
):
    """#954 AC2: a HANDLE-LESS typed error (the pre-provision guard — nothing
    provisioned, nothing bills) keeps the pre-#954 blanket branch byte-for-byte:
    NoComputeAvailableError raised, ROUTE_REASON_NO_COMPUTE marker, NO lease
    write, on_launched NOT called."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_NO_COMPUTE,
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _WorkloadStartFailedNoHandleRunpod()
    hooked: list[RunHandle] = []
    with pytest.raises(NoComputeAvailableError):
        failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            on_launched=hooked.append,
            lease_store=lease_store,
            now_fn=_clock(),
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
        )
    assert hooked == []  # on_launched never fired
    assert lease_store.read(137) is None  # no lease write
    finals = _by_reason(captured_markers, ROUTE_REASON_NO_COMPUTE)
    assert finals
    assert finals[-1]["attempts"][-1]["outcome"] == "runpod_fallback_failed"


def test_concurrent_triggerer_after_partial_workload_start_failure_no_second_launch(
    lease_store, marker_poster, captured_markers
):
    """#954 AC4-ii: triggerer 1 partial-fails (typed raise; lease stamped
    IN-FLOCK); an M3b concurrent second triggerer with the SAME GCP identity
    short-circuits in-flock — total launch count stays 1 (the load-bearing
    invariant; sequential calls on a shared LeaseStore model the
    serialized-by-flock outcome, exactly as M2.6 does)."""
    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    rp = _WorkloadStartFailedRunpod()
    identity = {"pod_name": "eps-issue-137", "job_id": "instance-fake-1"}
    cfg = RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0)

    def _trigger():
        return failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=lease_store,
            now_fn=_clock(),
            config=cfg,
            gcp_failover_of_identity=identity,
        )

    with pytest.raises(RunPodWorkloadStartError):
        _trigger()  # triggerer 1: partial failure, lease stamped in-flock
    second = _trigger()  # triggerer 2: in-flock re-check short-circuits

    assert len(rp.launches) == 1  # EXACTLY ONCE — no second paid launch
    assert second.extra.get("failover_already_launched") is True


def test_runpod_workload_start_failed_reason_cross_module_parity():
    """#954 AC5 (SR1 pattern): the router constant equals the established
    ``dispatch_issue.py`` literal, and the reason is NOT in the watcher's
    TRANSIENT_CAPACITY_REASONS allowlist — the capacity-retry pass never auto
    re-drives a run whose pod exists and bills."""
    import inspect

    import scripts.dispatch_issue as cli
    from explore_persona_space.backends.router import (
        ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED,
    )
    from scripts.autonomous_session_watch import TRANSIENT_CAPACITY_REASONS

    assert ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED == "runpod_workload_start_failed"
    # The dispatch CLI's typed arm uses the SAME literal (one reason per
    # failure class across paths, #909/#954).
    assert '"runpod_workload_start_failed"' in inspect.getsource(cli)
    assert ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED not in TRANSIENT_CAPACITY_REASONS


def test_runpod_terminal_rung_partial_lease_write_failure_preserves_typed_error(
    tmp_path, marker_poster, captured_markers
):
    """#954 AC1 guard (round-1 critique, alternatives MF2): a lease-write
    failure on the partial branch must NEVER replace the typed error — the
    failover legs' ``except RunPodWorkloadStartError`` is the rescue path, and
    an OSError escaping here would blind it exactly when rescue is needed. The
    marker still posts with the DISTINCT reason, and the RouteAttempt detail
    records the lease_write_failed note."""
    from contextlib import contextmanager

    from explore_persona_space.backends.router import (
        failover_to_runpod_after_async_workload_crash,
    )
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    class _RaisingWriteLeaseStore(LeaseStore):
        @contextmanager
        def transaction(self, issue):
            with super().transaction(issue) as (lease, _write):

                def _raising_write(new_lease):
                    raise OSError("Disk quota exceeded (EDQUOT) writing the lease")

                yield lease, _raising_write

    store = _RaisingWriteLeaseStore(lease_dir=tmp_path / ".eps-routing")
    rp = _WorkloadStartFailedRunpod()
    with pytest.raises(RunPodWorkloadStartError):  # STILL the typed class
        failover_to_runpod_after_async_workload_crash(
            spec=_spec(backend="gcp"),
            runpod_backend=rp,
            evidence={"source": "async_poller"},
            marker_poster=marker_poster,
            lease_store=store,
            now_fn=_clock(),
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            gcp_failover_of_identity={"pod_name": "eps-issue-137", "job_id": "i-1"},
        )
    finals = _by_reason(captured_markers, "runpod_workload_start_failed")
    assert finals  # the terminal marker still posted
    assert "lease_write_failed" in finals[-1]["attempts"][-1]["detail"]


# ---------------------------------------------------------------------------
# issue #1029 — GCP pre-workload boot-loop breaker: the durable streak record
# (Lease field + helpers) and the route()-side rung skip
# ---------------------------------------------------------------------------


def test_lease_round_trips_gcp_boot_death_streaks():
    """#1029: the ``gcp_boot_death_streaks`` Lease field round-trips through
    to_json/from_json, and a MALFORMED payload (non-dict) tolerantly reads as
    ``{}`` — fail-open toward today's behavior, mirroring ``raw_failover``."""
    from explore_persona_space.backends.router import Lease

    rec = {
        "flexstart_l4": {
            "count": 2,
            "date": "2026-07-05",
            "last_ts": 1234.5,
            "last_incarnation": "5583329098377891015",
        }
    }
    lease = Lease(issue=1029, spec_hash="h", attempt_id="a", gcp_boot_death_streaks=dict(rec))
    round_tripped = Lease.from_json(lease.to_json())
    assert round_tripped.gcp_boot_death_streaks == rec

    # Malformed payload (a string where the dict belongs) -> {}.
    payload = lease.to_json()
    payload["gcp_boot_death_streaks"] = "garbage"
    assert Lease.from_json(payload).gcp_boot_death_streaks == {}
    # Absent key (pre-#1029 lease on disk) -> {}.
    payload.pop("gcp_boot_death_streaks")
    assert Lease.from_json(payload).gcp_boot_death_streaks == {}


def test_record_gcp_boot_death_day_rollover_resets_count(lease_store):
    """#1029: a streak recorded on a PRIOR UTC day rolls over — the next death
    starts a fresh count of 1 (mirrors the gcp_attempts_date probe: a stale
    streak must not poison a rung after the cause is gone), and the read
    helper reads a prior-day record as 0."""
    from explore_persona_space.backends.router import (
        Lease,
        gcp_boot_death_streak,
        record_gcp_boot_death,
    )

    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_boot_death_streaks={
                "spot_a100_80": {
                    "count": 5,
                    "date": "2020-01-01",  # a prior UTC day
                    "last_ts": 1.0,
                    "last_incarnation": "old-instance",
                }
            },
        )
    )
    # The read helper treats the stale record as 0 (same-day scoping).
    assert gcp_boot_death_streak(137, "spot_a100_80", lease_store=lease_store) == 0
    # A fresh death starts a NEW streak of 1, not 6.
    count = record_gcp_boot_death(
        137, "spot_a100_80", incarnation="new-instance", lease_store=lease_store
    )
    assert count == 1
    assert gcp_boot_death_streak(137, "spot_a100_80", lease_store=lease_store) == 1


def test_ladder_skips_rung_with_boot_loop_streak_and_advances(
    lease_store, marker_poster, captured_markers
):
    """#1029 AC-3: a rung whose same-UTC-day streak is >= N is SKIPPED on the
    auto chain (RouteAttempt outcome ``boot_loop_rung_skipped``) and the ladder
    proceeds to the next rung, which launches."""
    from explore_persona_space.backends.router import Lease, _today_utc_iso

    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_boot_death_streaks={
                "spot_a100_80": {
                    "count": 2,
                    "date": _today_utc_iso(),
                    "last_ts": 1.0,
                    "last_incarnation": "inst-2",
                }
            },
        )
    )
    gcp = _GcpBackendDouble()
    result = route(
        _short_lora_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    # The boot-looped spot-80 rung was skipped; the NEXT rung launched.
    assert result.extra["gcp_ladder_rung"] == "spot_a100_40"
    skips = [a for a in result.attempts if a.outcome == "boot_loop_rung_skipped"]
    assert len(skips) == 1
    assert "rung spot_a100_80" in (skips[0].detail or "")
    assert len(gcp.launches) == 1  # exactly one create — on the next rung


def test_record_gcp_boot_death_then_route_skips_rung_end_to_end(
    lease_store, marker_poster, captured_markers
):
    """#1029 Must-Fix (writer->reader integration): NO direct lease seeding —
    two REAL ``record_gcp_boot_death`` calls (the poller's writer, distinct
    incarnations) against one LeaseStore, then the ladder walk against the
    SAME store SKIPS the rung. This is the only test that catches a key-shape
    drift between the poller's writer and route()'s reader (stub-seeded suites
    stay green under a drift)."""
    from explore_persona_space.backends.router import record_gcp_boot_death

    first = record_gcp_boot_death(137, "spot_a100_80", incarnation="i1", lease_store=lease_store)
    second = record_gcp_boot_death(137, "spot_a100_80", incarnation="i2", lease_store=lease_store)
    assert (first, second) == (1, 2)

    gcp = _GcpBackendDouble()
    result = route(
        _short_lora_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    assert result.extra["gcp_ladder_rung"] == "spot_a100_40"  # skipped past spot-80
    assert any(a.outcome == "boot_loop_rung_skipped" for a in result.attempts)
    assert len(gcp.launches) == 1  # ZERO creates on the boot-looped rung
    # The skip consumed ZERO daily attempts: the counter reads exactly the ONE
    # create the next rung spent.
    lease = lease_store.read(137)
    assert lease is not None and lease.gcp_attempts_today == 1


def test_record_gcp_boot_death_then_route_skips_rung_cpu_bigmem_variant(
    lease_store, marker_poster, captured_markers
):
    """#1029 Must-Fix companion (cpu-bigmem): the route()-side skip is
    cpu-bigmem's ONLY breaker (the poller records but never rewrites — no
    RunPod lane, #677). Real recorder writes on its single ``ondemand_cpu``
    rung make the ladder yield nothing -> the #677 typed terminal fires with
    ZERO GCP creates and NO RunPod launch."""
    from explore_persona_space.backends.router import (
        CpuExhaustedNoRunpodLaneError,
        record_gcp_boot_death,
    )

    record_gcp_boot_death(137, "ondemand_cpu", incarnation="c1", lease_store=lease_store)
    record_gcp_boot_death(137, "ondemand_cpu", incarnation="c2", lease_store=lease_store)

    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble()
    spec = RunSpec(issue=137, intent="cpu-bigmem", backend="auto", time_budget_hours=1.0)
    with pytest.raises(CpuExhaustedNoRunpodLaneError):
        route(
            spec,
            runpod_backend=rp,
            free_backends={"nibi": _FreeLaneBackend(kind="nibi", launch_raises=RuntimeError("x"))},
            gcp_backend=gcp,
            lease_store=lease_store,
            marker_poster=marker_poster,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
            now_fn=_clock(),
            sleep_fn=lambda _s: None,
        )
    assert len(gcp.launches) == 0  # the boot-looped rung was never re-created
    assert len(rp.launches) == 0  # cpu-bigmem never falls over to RunPod


def test_boot_loop_rung_skip_does_not_bump_gcp_attempts_today(
    lease_store, marker_poster, captured_markers
):
    """#1029 AC-3 / success criterion 3: skips consume ZERO daily attempts —
    with EVERY rung boot-loop-skipped, the per-day counter stays 0 (the cap
    counts CREATES; a skip avoids the create)."""
    from explore_persona_space.backends.router import Lease, _today_utc_iso

    today = _today_utc_iso()
    all_rungs = {
        label: {"count": 2, "date": today, "last_ts": 1.0, "last_incarnation": "i2"}
        for label in (
            "spot_a100_80",
            "spot_a100_40",
            "flexstart_a100_80",
            "ondemand_a100_80",
            "ondemand_a100_40",
        )
    }
    lease_store.write(
        Lease(issue=137, spec_hash="h", attempt_id="a", gcp_boot_death_streaks=all_rungs)
    )
    result = route(
        _short_lora_spec(),
        runpod_backend=_PassiveRunpod(),
        gcp_backend=_GcpBackendDouble(),
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    lease = lease_store.read(137)
    assert lease is not None and lease.gcp_attempts_today == 0


def test_all_rungs_boot_loop_skipped_falls_through_lane_order(
    lease_store, marker_poster, captured_markers
):
    """#1029 (the Goal's "next rung / RunPod naturally" half): with EVERY GCP
    rung boot-loop-skipped, the auto chain proceeds down the existing lane
    order to the RunPod terminal rung — five ``boot_loop_rung_skipped``
    attempts, zero GCP creates, RunPod launched."""
    from explore_persona_space.backends.router import Lease, _today_utc_iso

    today = _today_utc_iso()
    all_rungs = {
        label: {"count": 3, "date": today, "last_ts": 1.0, "last_incarnation": "i3"}
        for label in (
            "spot_a100_80",
            "spot_a100_40",
            "flexstart_a100_80",
            "ondemand_a100_80",
            "ondemand_a100_40",
        )
    }
    lease_store.write(
        Lease(issue=137, spec_hash="h", attempt_id="a", gcp_boot_death_streaks=all_rungs)
    )
    rp = _PassiveRunpod()
    gcp = _GcpBackendDouble()
    result = route(
        _short_lora_spec(),
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    skips = [a for a in result.attempts if a.outcome == "boot_loop_rung_skipped"]
    assert len(skips) == 5
    assert len(gcp.launches) == 0
    assert len(rp.launches) == 1


def test_explicit_gcp_pin_ignores_boot_loop_skip(lease_store, marker_poster, captured_markers):
    """#1029: an explicit ``backend: gcp`` pin (count_attempt_cap=False) is
    EXEMPT from the boot-loop skip — an explicit user ask attempts the rung
    anyway, mirroring the attempt-cap exemption."""
    from explore_persona_space.backends.router import Lease, _today_utc_iso

    lease_store.write(
        Lease(
            issue=137,
            spec_hash="h",
            attempt_id="a",
            gcp_boot_death_streaks={
                "spot_a100_80": {
                    "count": 4,
                    "date": _today_utc_iso(),
                    "last_ts": 1.0,
                    "last_incarnation": "i4",
                }
            },
        )
    )
    gcp = _GcpBackendDouble()
    spec = RunSpec(issue=137, intent="lora-7b", backend="gcp", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    # The boot-looped FIRST rung was attempted (and launched) despite the streak.
    assert result.extra["gcp_ladder_rung"] == "spot_a100_80"
    assert not any(a.outcome == "boot_loop_rung_skipped" for a in result.attempts)


def test_boot_loop_reason_distinct_from_crash_capacity_queue_reasons():
    """#1029: the boot-loop reason VALUE is distinct from BOTH workload-crash
    reasons, the queue-timeout reason, AND the capacity-exhaustion fallback
    reason — the marker trail tells a boot loop apart from every sibling."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    )

    reasons = {
        ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        ROUTE_REASON_RUNPOD_FALLBACK,
    }
    assert len(reasons) == 5  # all five are distinct strings
    assert ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD == "gcp_boot_loop_failover_runpod"


def test_launched_gcp_handle_extra_carries_rung_label_and_launched_ts(
    lease_store, marker_poster, captured_markers
):
    """#1029 §4.3: a launched GCP handle's ``extra`` carries the ladder-rung
    label (the poller's streak key) and a WALL-CLOCK launch ts (the poller ages
    it against time.time(); route()'s now_fn default is time.monotonic — a
    different epoch — so the stamp must be epoch seconds)."""
    import time as _t

    gcp = _GcpBackendDouble()
    result = route(
        _short_lora_spec(),
        runpod_backend=_ExplodingRunpod(),
        gcp_backend=gcp,
        lease_store=lease_store,
        marker_poster=marker_poster,
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0),
        now_fn=_clock(),
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "gcp"
    extra = result.handle.extra
    assert extra.get("gcp_ladder_rung") == result.extra["gcp_ladder_rung"]
    ts = extra.get("gcp_launched_ts")
    assert isinstance(ts, float)
    # Wall-clock epoch seconds (NOT the fake monotonic _clock() counter).
    assert abs(ts - _t.time()) < 300


# ---------------------------------------------------------------------------
# #1116 — GCP FLEX_START queue-VANISH → RunPod failover (poller / router seam)
#
# The queue-vanish failover reuses the SAME router seam as the #659/#783/#1029
# siblings (failover_to_runpod_after_async_workload_crash → the RunPod terminal
# rung), passing the distinct queue-vanish reason. These tests pin the
# router-level contract; the poller-side end-to-end tests live in
# tests/test_backend_poll.py (the #1116 block).
# ---------------------------------------------------------------------------


def test_queue_vanish_failover_seam_carries_queue_vanish_reason(
    lease_store, marker_poster, captured_markers
):
    """#1116 (router seam): calling the failover seam with the queue-vanish
    reason launches RunPod once, labels the RouteResult + the
    ``epm:backend-selected`` marker with ``gcp_queue_vanish_failover_runpod``,
    and carries the evidence (incl. the clock discriminator) onto the marker
    ``extra``."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
        failover_to_runpod_after_async_workload_crash,
    )

    rp = _PassiveRunpod()
    result = failover_to_runpod_after_async_workload_crash(
        spec=_spec(backend="gcp"),
        runpod_backend=rp,
        evidence={
            "source": "async_poller_queue_vanish",
            "current_phase": "terminal_queue_vanish",
            "last_observed_phase": "pending",
        },
        reason=ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
        marker_poster=marker_poster,
        lease_store=lease_store,
        now_fn=_clock(),
        config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD
    assert len(rp.launches) == 1  # exactly once
    finals = _by_reason(captured_markers, ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD)
    assert finals
    evidence = finals[-1]["extra"].get("gcp_workload_evidence", {})
    assert evidence.get("source") == "async_poller_queue_vanish"
    assert evidence.get("last_observed_phase") == "pending"


def test_queue_vanish_reason_distinct_from_crash_capacity_queue_boot_reasons():
    """#1116: the queue-vanish reason VALUE is pairwise-distinct from BOTH
    workload-crash reasons, the queue-timeout reason, the boot-loop reason,
    AND the capacity-exhaustion fallback reason (auto_fallback_runpod) — the
    marker trail tells a server-side queue drop apart from every sibling."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    )

    reasons = {
        ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
        ROUTE_REASON_RUNPOD_FALLBACK,
    }
    assert len(reasons) == 6  # all six are distinct strings
    assert ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD == "gcp_queue_vanish_failover_runpod"
