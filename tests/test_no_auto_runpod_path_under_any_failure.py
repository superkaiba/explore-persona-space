"""Fixed pointer for the historical ``test_no_auto_runpod_path_under_any_failure``.

#656 REVERSED the no-auto-RunPod invariant: the auto chain now reaches
RunPod as the documented TERMINAL rung, after every cheaper GCP ladder rung
(on-demand A100-80 → A100-40 → spot) AND the free SLURM lanes are exhausted.
The old negative test (`tests/test_router.py::test_no_auto_runpod_path_under_any_failure`)
was replaced in place by
`test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted`, which pins
the NEW contract.

This module is a self-contained pointer so the literal acceptance command in
the task body — `uv run pytest tests/test_no_auto_runpod_path_under_any_failure.py`
— still resolves to a runnable test (rather than erroring on a missing path).
It re-asserts the NEW ordering contract end-to-end without depending on the
`test_router.py` fixtures (which do not travel across modules on a bare
re-import).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from explore_persona_space.backends.base import (
    BackendKind,
    PollResult,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.gcp import GcpProvisioningError
from explore_persona_space.backends.router import (
    ROUTE_REASON_RUNPOD_FALLBACK,
    LeaseStore,
    RouterConfig,
    route,
)


def test_no_auto_runpod_invariant_was_reversed_in_656() -> None:
    """Tripwire: the new RunPod-terminal-rung reason code exists.

    The historical invariant ("auto NEVER calls RunPod") was deliberately
    reversed in #656; the new reason code distinguishes the auto fallback
    from a user-pinned RunPod override.
    """
    assert ROUTE_REASON_RUNPOD_FALLBACK == "auto_fallback_runpod"


class _PassiveRunpodPointer:
    """Minimal RunPod double that records launches (no infrastructure)."""

    def __init__(self) -> None:
        self.launches: list[RunSpec] = []

    @property
    def name(self) -> BackendKind:
        return "runpod"

    def prepare(self, spec: RunSpec) -> None:
        return None

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

    def estimate_start(self, spec: RunSpec) -> datetime:
        return datetime.now(tz=UTC)

    def poll(self, handle: RunHandle) -> PollResult:
        return PollResult(
            status="running",
            current_phase="running",
            new_milestone=False,
            last_log_mtime_sec_ago=0,
            pid_alive=True,
            log_tail_excerpt="",
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        return ""

    def fetch_results(self, handle: RunHandle) -> None:
        return None

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        return True

    def teardown(self, handle: RunHandle) -> None:
        return None


class _GcpAllRungsExhausted(_PassiveRunpodPointer):
    """GCP double whose every create capacity-misses (ladder exhausts)."""

    @property
    def name(self) -> BackendKind:
        return "gcp"

    def launch(self, spec: RunSpec) -> RunHandle:
        raise GcpProvisioningError(
            "ZONE_RESOURCE_POOL_EXHAUSTED", evidence={"matched_pattern": "RESOURCE_EXHAUSTED"}
        )

    def estimate_start_seconds(self, spec: RunSpec) -> float:
        return 0.0


def test_runpod_reached_only_after_full_gcp_ladder_exhausted(tmp_path: Any) -> None:
    """The #656 ordering contract end-to-end: a short lora-7b whose every GCP
    rung capacity-misses falls through to the RunPod TERMINAL rung — and the
    RunPod launch is the LAST attempt in the trail (never first, never
    skipping a cheaper GCP rung)."""
    rp = _PassiveRunpodPointer()
    gcp = _GcpAllRungsExhausted()
    spec = RunSpec(issue=137, intent="lora-7b", backend="auto", time_budget_hours=1.0)
    result = route(
        spec,
        runpod_backend=rp,
        gcp_backend=gcp,
        lease_store=LeaseStore(lease_dir=tmp_path / ".eps-routing"),
        config=RouterConfig(
            free_wait_seconds=1,
            poll_interval=0.0,
            cancel_grace_seconds=0,
            max_gcp_attempts_per_day=99,
        ),
        now_fn=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )
    assert result.chosen_kind == "runpod"
    assert result.reason == ROUTE_REASON_RUNPOD_FALLBACK
    assert len(rp.launches) == 1
    outcomes = [(a.kind, a.outcome) for a in result.attempts]
    gcp_fail_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "gcp"]
    runpod_idxs = [i for i, (k, o) in enumerate(outcomes) if k == "runpod" and o == "launched"]
    assert gcp_fail_idxs, "the GCP ladder must have been attempted first"
    assert runpod_idxs and runpod_idxs[-1] == len(outcomes) - 1  # RunPod is LAST
    assert max(gcp_fail_idxs) < runpod_idxs[-1]
