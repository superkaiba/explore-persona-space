"""Tests for the slice-6 `/issue` dispatch helper (`backends.issue_dispatch`).

Slice-6 surface coverage (the four areas the implementer brief
enumerates):

1. **RunPod characterization** — the RunPod backend's
   ``poll`` / ``fetch_logs`` / ``fetch_results`` / ``confirm_artifacts``
   produce behaviour equivalent to today's flow (poll → delegates to
   ``scripts.poll_pipeline.poll_once``; fetch_logs → ssh tail;
   fetch_results → ``pod.py sync results --all`` argv;
   confirm_artifacts → already implemented, asserts FAIL on missing
   declaration).
2. **GCP scp-back** — ``GcpBackend.fetch_results`` issues a sentinel
   scp call (mandatory) + best-effort artifact-dir scp calls (logs
   on failure but does not raise). Mocks the gcloud runner.
3. **Dispatch helper** — empty frontmatter → ``RunSpec.backend ==
   "auto"``; ``"cluster"`` legacy → ``"nibi"``; terminal exceptions
   translate to the right ``epm:failure v1`` / status pair.
4. **Bg-Bash poll contract preservation** — the orchestrator's
   bg-Bash poll re-invocation model is preserved: poll stays a JSON-
   line emitter (``scripts/backend_poll.py``), the handle is
   round-tripped via the sidecar JSON, and the JSON line shape
   matches the legacy ``poll_pipeline.py`` output keys.

Nothing in this file requires RunPod / SLURM / GCP / SSH to be live —
every external call is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.gcp import (
    GcloudRunResult,
    GcpBackend,
    GcpConfig,
)
from explore_persona_space.backends.issue_dispatch import (
    DispatchOutcome,
    build_run_spec,
    classify_terminal_exception,
    default_handle_sidecar_path,
    deserialize_handle,
    dispatch_for_issue,
    normalize_backend_value,
    read_handle_sidecar,
    serialize_handle,
    write_handle_sidecar,
)
from explore_persona_space.backends.router import (
    BackendPrepareError,
    GcpAttemptCapExceededError,
    LeaseStore,
    ManualAttentionRequiredError,
    NoComputeAvailableError,
    RouterConfig,
    WorkloadSurfacedError,
)
from explore_persona_space.backends.runpod import RunPodBackend, _runpod_pid_file_path


@pytest.fixture(autouse=True)
def _gcp_rollback_build_for_legacy_suite(request, monkeypatch):
    """Run the legacy dispatch suite under the #2028 rollback build (flag OFF).

    GCP provisioning is disabled by policy (#2028) but the gated dispatch
    paths stay test-covered — the single-constant rollback lever. Flag-ON
    production pins carry ``@pytest.mark.gcp_policy_default``.
    """
    if request.node.get_closest_marker("gcp_policy_default"):
        return
    from explore_persona_space.backends import router as router_module

    monkeypatch.setattr(router_module, "GCP_PROVISIONING_DISABLED", False)
    # #2054 put runpod FIRST in _default_auto_lane_order(); the legacy
    # ladder/failover suite exercises the pre-#2054 fellows/gcp/SLURM
    # machinery, so pin the pre-#2054 rollback order (no runpod head) here.
    # Runpod-first default-contract tests carry
    # @pytest.mark.gcp_policy_default (module defaults: flag ON,
    # runpod-first order).
    monkeypatch.setattr(
        router_module, "DEFAULT_AUTO_LANE_ORDER", ("fellows", "gcp", "nibi", "fir", "mila")
    )


# ---------------------------------------------------------------------------
# Section 1 — RunPod backend wiring (characterization tests)
# ---------------------------------------------------------------------------


def _runpod_handle(issue: int = 137) -> RunHandle:
    """Build a RunPod handle as ``RunPodBackend.launch`` would shape it."""
    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name=f"pod-{issue}",
        scratch_dir="/workspace",
        log_path=f"/workspace/logs/issue-{issue}.log",
        extra={
            "intent": "lora-7b",
            "issue": issue,
            "pid_file": _runpod_pid_file_path(issue),
        },
    )


def test_runpod_launch_stuffs_issue_and_pid_file_onto_handle_extra(monkeypatch) -> None:
    """The slice-6 launch path must populate ``extra['issue']`` AND
    ``extra['pid_file']`` so the unified ``poll`` / ``fetch_results``
    paths don't need to re-derive them.
    """
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    # #1465: the provision leg now routes through the Popen-based
    # pod_lifecycle relay — the blanket subprocess.run patch no longer
    # intercepts it, so no-op the helper too (else a REAL provision runs).
    monkeypatch.setattr(
        "explore_persona_space.backends.runpod._run_pod_lifecycle_relay",
        lambda cmd, **k: None,
    )
    backend = RunPodBackend()
    spec = RunSpec(issue=99, intent="lora-7b", backend="runpod")
    handle = backend.launch(spec)
    assert handle.extra["issue"] == 99
    assert handle.extra["pid_file"] == "/workspace/logs/issue-99.pid"
    assert handle.pod_name == "pod-99"


def test_runpod_poll_delegates_to_poll_pipeline_and_returns_typed_pollresult(
    monkeypatch,
) -> None:
    """``RunPodBackend.poll`` must call into ``scripts.poll_pipeline.poll_once``
    AND return a ``backends.base.PollResult`` (not the script's PollResult
    class — so cross-backend ``isinstance`` checks work).

    The legacy ``poll_pipeline.py`` is the battle-tested probe; slice 6
    keeps it as the implementation while routing every backend through
    the same ``backend.poll(handle)`` surface.
    """
    captured: dict[str, Any] = {}

    def fake_poll_once(*, issue, pod, log_path, pid_file, state_file):
        captured.update(
            issue=issue,
            pod=pod,
            log_path=log_path,
            pid_file=pid_file,
            state_file=state_file,
        )
        # Mimic ``scripts.poll_pipeline.PollResult`` shape (frozen
        # dataclass with the same fields as backends.base.PollResult).
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _PR:
            status: str = "running"
            current_phase: str = "phase-foo"
            new_milestone: bool = False
            last_log_mtime_sec_ago: int = 12
            pid_alive: bool = True
            log_tail_excerpt: str = "..."
            gate: str | None = None
            sentinels_processed: int = 0
            phase_log_mtime_sec_ago: int = 100
            shard_log_mtime_sec_ago: int = 200
            gpu_util: str = "95"
            # Non-default sentinel (base.PollResult defaults to 540) so the
            # pass-through assertion below discriminates a real thread-through
            # from a silent fall-back to the default.
            next_interval: int = 1800
            # #664 added stall_reason to backends.base.PollResult; RunPodBackend.poll
            # reads raw.stall_reason directly (not getattr), so the stub must carry it.
            stall_reason: str | None = None

        return _PR()

    monkeypatch.setattr("scripts.poll_pipeline.poll_once", fake_poll_once)
    backend = RunPodBackend()
    handle = _runpod_handle(issue=137)
    result = backend.poll(handle)
    assert isinstance(result, PollResult)
    assert result.status == "running"
    assert result.current_phase == "phase-foo"
    # Adaptive bg-poll interval (anti-stall redesign §7) must thread through
    # the typed-PollResult rebuild, not fall back to the 540 default.
    assert result.next_interval == 1800
    assert captured["issue"] == 137
    assert captured["pod"] == "pod-137"
    assert captured["log_path"] == "/workspace/logs/issue-137.log"
    assert captured["pid_file"] == "/workspace/logs/issue-137.pid"


def test_runpod_fetch_logs_runs_ssh_tail_and_returns_stdout(monkeypatch) -> None:
    """``fetch_logs`` is a one-shot ssh tail; the exact argv pattern
    matters because it's what an operator would inspect on a flake."""
    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "log line 1\nlog line 2\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured.append(list(argv))
        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
    backend = RunPodBackend()
    handle = _runpod_handle(issue=137)
    out = backend.fetch_logs(handle)
    assert out == "log line 1\nlog line 2\n"
    assert len(captured) == 1
    argv = captured[0]
    assert argv[0] == "ssh"
    assert argv[1] == "pod-137"
    # Verify the tail command includes the log path with a tail limit.
    assert "tail" in argv[2]
    assert "/workspace/logs/issue-137.log" in argv[2]


def test_runpod_fetch_logs_is_best_effort_on_ssh_failure(monkeypatch) -> None:
    """A non-zero ssh exit returns ``""`` (NEVER raises) — the legacy
    orchestrator's progress notes shouldn't crash on a transient SSH
    blip."""

    class _Proc:
        returncode = 255  # ssh connection refused
        stdout = ""
        stderr = "ssh: connect to host failed"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    backend = RunPodBackend()
    out = backend.fetch_logs(_runpod_handle())
    assert out == ""


def test_runpod_fetch_results_invokes_pod_py_sync_results_all(monkeypatch) -> None:
    """The fetch_results path must call ``pod.py sync results --all`` —
    the same orchestrator-driven path Step 8 invokes today, preserved
    behavior."""
    captured: list[list[str]] = []

    def fake_run(argv, **kwargs):
        captured.append(list(argv))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)
    backend = RunPodBackend()
    backend.fetch_results(_runpod_handle(issue=137))
    assert len(captured) == 1
    argv = captured[0]
    # argv[0] is sys.executable (uv-run python); argv[1] is pod.py
    assert any("pod.py" in str(a) for a in argv)
    assert "sync" in argv
    assert "results" in argv
    assert "--all" in argv


def test_runpod_confirm_artifacts_already_wired_fails_on_missing_decl() -> None:
    """Slice 2's verifier delegated path: a handle without
    ``expected_artifacts`` MUST FAIL (silent-loss safeguard)."""
    backend = RunPodBackend()
    handle = _runpod_handle(issue=137)
    # No EXPECTED_ARTIFACTS_HANDLE_KEY in handle.extra → FAIL.
    assert backend.confirm_artifacts(handle) is False


def test_runpod_issue_recovery_handles_legacy_pod_name() -> None:
    """A handle with the legacy ``epm-issue-<N>`` pod name (pre-canonical
    rename) must still resolve to its issue number — round-tripped
    handles from older sessions should keep working."""
    backend = RunPodBackend()
    handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name="epm-issue-200",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-200.log",
        extra={},
    )
    assert backend._issue_from_handle(handle) == 200


# ---------------------------------------------------------------------------
# Section 2 — GCP fetch_results pull-back (ssh sudo cat sentinel + scp dirs)
# ---------------------------------------------------------------------------


def _gcp_config(*, vm_scratch_dir: str = "/workspace") -> GcpConfig:
    return GcpConfig(
        project="eps-test-project",
        gcloud_config="eps-test-config",
        primary_zone="us-central1-a",
        fallback_zones=(),
        image_family="img",
        image_project="img-project",
        repo_url="https://example/repo.git",
        vm_scratch_dir=vm_scratch_dir,
    )


def _gcp_handle(
    issue: int = 137,
    attempt_id: str = "att-001",
    *,
    vm_scratch_dir: str = "/workspace",
) -> RunHandle:
    cfg = _gcp_config(vm_scratch_dir=vm_scratch_dir)
    workload = f"{vm_scratch_dir}/eps-issue-{issue}"
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="111",
        pod_name=f"eps-issue-{issue}",
        scratch_dir=workload,
        log_path=f"{vm_scratch_dir}/logs/issue-{issue}.log",
        extra={
            "issue": issue,
            "zone": cfg.primary_zone,
            "attempt_id": attempt_id,
            "intent": "lora-7b",
            "project": cfg.project,
            "gcloud_config": cfg.gcloud_config,
        },
    )


class _RecordingGcloudRunner:
    """Records every argv; returns a scripted result per call (FIFO)."""

    def __init__(self, results: list[GcloudRunResult] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results or [])

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if self._results:
            return self._results.pop(0)
        return GcloudRunResult(returncode=0, stdout="", stderr="")


def test_gcp_fetch_results_issues_sentinel_pull_as_first_ssh_call(tmp_path) -> None:
    """The sentinel pull is MANDATORY (the verifier reads it locally;
    a missing local sentinel = silent-loss). It is the FIRST ssh call so
    its failure surfaces before the best-effort dir pulls — preceded only
    by the #1454 `instances describe` reachability probe (which classifies
    the transport BEFORE any ssh is attempted; the rig default rc=0/empty
    stdout is indeterminate, failing open to the legacy transport). The
    pull is `gcloud compute ssh ... sudo -n cat`, NOT scp — the
    startup-script runs as root, so the workload tree is root-owned and
    the OS-Login scp user gets Permission denied (#588
    att-20260611-064703)."""
    runner = _RecordingGcloudRunner()
    backend = GcpBackend(
        config=_gcp_config(vm_scratch_dir=str(tmp_path)),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    backend.fetch_results(_gcp_handle(vm_scratch_dir=str(tmp_path)))
    assert runner.calls, "fetch_results made no gcloud call"
    # Call 0 is the #1454 reachability probe (instances describe, JSON).
    probe_argv = runner.calls[0]
    assert "describe" in probe_argv
    assert "instances" in probe_argv
    assert "eps-issue-137" in probe_argv
    # The FIRST ssh call is `gcloud compute ssh <name> --command='sudo -n cat <sentinel>'`.
    first_ssh = next(a for a in runner.calls if "ssh" in a)
    assert "compute" in first_ssh
    assert "scp" not in first_ssh
    assert "eps-issue-137" in first_ssh
    command_arg = next(a for a in first_ssh if a.startswith("--command="))
    assert command_arg.startswith("--command=sudo -n cat ")
    assert ".completion-sentinel.json" in command_arg


def test_gcp_fetch_results_falls_back_best_effort_on_artifact_dir_failure(tmp_path) -> None:
    """A best-effort dir tar-pull failure logs + continues (eval_results/figures
    are authoritative on HF/WandB/git already). Post-#790 the best-effort pulls
    are `ssh ... sudo -n bash -o pipefail -c 'tar | base64'`, NOT scp — the
    workload tree is root-owned (#588), so plain scp Permission-denies."""
    runner = _RecordingGcloudRunner(
        results=[
            # #1454 reachability probe (indeterminate {} -> legacy fail-open)
            GcloudRunResult(returncode=0, stdout="{}", stderr=""),
            # sentinel pull (ssh sudo cat) PASS
            GcloudRunResult(returncode=0, stdout='{"phase": "done", "issue": 137}\n', stderr=""),
            GcloudRunResult(returncode=1, stdout="", stderr="not found"),  # eval_results FAIL
            GcloudRunResult(returncode=1, stdout="", stderr="not found"),  # figures FAIL
        ],
    )
    backend = GcpBackend(
        config=_gcp_config(vm_scratch_dir=str(tmp_path)),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    # No raise — best-effort.
    backend.fetch_results(_gcp_handle(vm_scratch_dir=str(tmp_path)))
    # One ssh sentinel pull + two ssh tar dir pulls; no scp (#790).
    ssh_calls = [a for a in runner.calls if "ssh" in a]
    scp_calls = [a for a in runner.calls if "scp" in a]
    assert len(ssh_calls) == 3
    assert len(scp_calls) == 0


def test_gcp_fetch_results_skips_when_handle_missing_issue(tmp_path) -> None:
    """A handle without ``extra['issue']`` cannot resolve a sentinel path;
    log + return rather than mis-scp (the alternative is a default
    issue=0 which would silently land in the wrong directory)."""
    runner = _RecordingGcloudRunner()
    backend = GcpBackend(
        config=_gcp_config(vm_scratch_dir=str(tmp_path)),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="111",
        pod_name="eps-issue-0",
        scratch_dir=str(tmp_path / "eps-issue-0"),
        log_path=str(tmp_path / "logs/issue-0.log"),
        extra={"zone": "us-central1-a"},  # no 'issue' field
    )
    backend.fetch_results(handle)
    # No scp issued.
    assert not runner.calls


def test_gcp_fetch_results_skips_when_handle_missing_attempt_id(tmp_path) -> None:
    """An attempt_id is REQUIRED to resolve the sentinel sub-directory
    (the GCP sentinel namespace is per-attempt)."""
    runner = _RecordingGcloudRunner()
    backend = GcpBackend(
        config=_gcp_config(vm_scratch_dir=str(tmp_path)),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="111",
        pod_name="eps-issue-137",
        scratch_dir=str(tmp_path / "eps-issue-137"),
        log_path=str(tmp_path / "logs/issue-137.log"),
        extra={"issue": 137, "zone": "us-central1-a"},  # no attempt_id
    )
    backend.fetch_results(handle)
    assert not runner.calls


# ---------------------------------------------------------------------------
# Section 3 — Dispatch helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("runpod", "runpod"),
        ("nibi", "nibi"),
        ("fir", "fir"),
        ("gcp", "gcp"),
        ("mila", "mila"),
        ("fellows", "fellows"),  # #1609 charmander lane
        ("  FELLOWS  ", "fellows"),  # case + whitespace tolerant
        ("cluster", "nibi"),  # legacy alias normalization
        ("  CLUSTER  ", "nibi"),  # case + whitespace tolerant
    ],
)
def test_normalize_backend_value_table(raw: Any, expected: BackendKind) -> None:
    """The dispatch helper's normalizer MUST map every legal frontmatter
    value to a router-acceptable backend kind."""
    assert normalize_backend_value(raw) == expected


def test_normalize_backend_value_rejects_typo() -> None:
    """A typo (``runpd``) must NOT silently auto-route — that would mask
    a frontmatter bug and burn GPU on the wrong lane."""
    with pytest.raises(ValueError, match="unknown backend"):
        normalize_backend_value("runpd")


def test_build_run_spec_empty_frontmatter_routes_auto() -> None:
    """A task with no ``backend:`` frontmatter must construct a RunSpec
    whose ``backend == "auto"`` (the router's cost-safe sentinel)."""
    spec = build_run_spec(
        issue=137,
        intent="lora-7b",
        backend_value=None,
        hydra_args=("condition=c1", "seed=42"),
    )
    assert spec.backend == "auto"
    assert spec.issue == 137
    assert spec.intent == "lora-7b"
    assert spec.hydra_args == ("condition=c1", "seed=42")


def test_build_run_spec_legacy_cluster_maps_to_nibi() -> None:
    """``backend: cluster`` in legacy frontmatter must NOT reach the
    router (which rejects the bare literal); the helper maps it."""
    spec = build_run_spec(issue=137, intent="lora-7b", backend_value="cluster")
    assert spec.backend == "nibi"


@pytest.mark.parametrize(
    "exc_factory, expected_failure_class, expected_status, expected_reason_substr",
    [
        (
            lambda: NoComputeAvailableError("everything failed", attempts=[]),
            "infra",
            "blocked",
            "no_compute_available",
        ),
        (
            lambda: BackendPrepareError(
                "backend.prepare failed for nibi/nibi (CalledProcessError: rsync rc=255)",
                kind="nibi",
                cluster="nibi",
            ),
            "infra",
            "blocked",
            "backend_prepare_failed",
        ),
        (
            lambda: WorkloadSurfacedError("workload crashed", chosen_kind="gcp"),
            "code",
            "blocked",
            "workload_failure",
        ),
        (
            lambda: GcpAttemptCapExceededError(issue=137, attempts_today=5, cap=5),
            "infra",
            "blocked",
            "gcp_attempt_cap_exceeded",
        ),
        (
            lambda: ManualAttentionRequiredError(
                kind="nibi",
                cluster="nibi",
                orphaned_job_id="42",
                attempts=[],
            ),
            "infra",
            "blocked",
            "manual_attention_required",
        ),
    ],
)
def test_classify_terminal_exception_translates_to_epm_failure_pair(
    exc_factory,
    expected_failure_class: str,
    expected_status: str,
    expected_reason_substr: str,
) -> None:
    """Every router terminal must translate to an ``epm:failure v1`` body
    + status the orchestrator (SKILL.md Step 7) already routes on."""
    translation = classify_terminal_exception(exc_factory())
    assert translation.failure_class == expected_failure_class
    assert translation.status == expected_status
    # The body's first line must carry the failure_class= prefix so
    # SKILL.md Step 7's classification table short-circuits.
    first_line = translation.note.splitlines()[0]
    assert first_line == f"failure_class: {expected_failure_class}"
    assert expected_reason_substr in translation.note


def test_classify_terminal_exception_manual_attention_carries_orphaned_id() -> None:
    """The ManualAttention failure note must carry the orphaned job_id
    (the operator needs it to manually scancel)."""
    exc = ManualAttentionRequiredError(
        kind="nibi",
        cluster="nibi",
        orphaned_job_id="JOB-XYZ-7777",
        attempts=[],
    )
    note = classify_terminal_exception(exc).note
    assert "JOB-XYZ-7777" in note
    assert "scancel" in note.lower()


def test_classify_terminal_cpu_exhausted_emits_distinct_reason() -> None:
    """#677: a CpuExhaustedNoRunpodLaneError maps to the DISTINCT reason
    cpu_exhausted_no_runpod_lane (NOT no_compute_available), so the watcher's
    capacity-retry pass does not hot-retry a structurally-unservable run.

    Removing the isinstance branch (or placing it AFTER the base
    NoComputeAvailableError branch) turns this RED — the CPU exception would be
    caught by the generic branch and emit reason: no_compute_available.
    """
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD,
        CpuExhaustedNoRunpodLaneError,
    )

    exc = CpuExhaustedNoRunpodLaneError(
        "CPU intent 'cpu-bigmem': GCP exhausted and RunPod is GPU-only",
        attempts=[{"kind": "gcp", "outcome": "capacity_miss"}],
    )
    t = classify_terminal_exception(exc)
    assert t.failure_class == "infra"
    assert t.status == "blocked"
    # The machine-greppable reason token is the CPU-specific one ...
    assert f"reason: {ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD}" in t.note
    assert "reason: no_compute_available" not in t.note
    # ... and the human message survives as detail (inherited .reason).
    assert "detail: CPU intent 'cpu-bigmem'" in t.note


def test_classify_stopped_pod_collision() -> None:
    """#1997 (b6): RunPodStoppedPodCollisionError maps to infra/blocked with
    the DISTINCT reason runpod_stopped_pod_collision (NOT no_compute_available)
    — the token is NOT in the watcher's TRANSIENT_CAPACITY_REASONS, so the
    capacity-retry pass never hot-retries a structural refusal only a human
    can clear (resume / approved terminate / --name-suffix)."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_RUNPOD_STOPPED_POD_COLLISION,
        RunPodStoppedPodCollisionError,
    )
    from scripts.autonomous_session_watch import TRANSIENT_CAPACITY_REASONS

    exc = RunPodStoppedPodCollisionError(
        "RunPod terminal rung refused: a STOPPED pod-137 already exists",
        attempts=[{"kind": "runpod", "outcome": "runpod_stopped_pod_collision"}],
    )
    t = classify_terminal_exception(exc)
    assert t.failure_class == "infra"
    assert t.status == "blocked"
    assert f"reason: {ROUTE_REASON_RUNPOD_STOPPED_POD_COLLISION}" in t.note
    assert "reason: no_compute_available" not in t.note
    assert "detail: RunPod terminal rung refused" in t.note
    # The recovery paths for the human who must clear the refusal.
    assert "pod.py resume" in t.note
    assert "--yes --approve" in t.note
    assert "--name-suffix" in t.note
    # The route-attempt trail survives into the note (epm:failure evidence).
    assert "runpod_stopped_pod_collision" in t.note
    assert ROUTE_REASON_RUNPOD_STOPPED_POD_COLLISION not in TRANSIENT_CAPACITY_REASONS


def test_classify_terminal_generic_no_compute_still_no_compute() -> None:
    """#677 control: the subclass branch did NOT shadow the generic branch —
    a plain NoComputeAvailableError still maps to reason: no_compute_available."""
    from explore_persona_space.backends.router import ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD

    exc = NoComputeAvailableError(
        "every free lane park-failed AND GCP quota has no headroom",
        attempts=[],
    )
    t = classify_terminal_exception(exc)
    assert "reason: no_compute_available" in t.note
    assert ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD not in t.note


# ---------------------------------------------------------------------------
# Dispatch helper: end-to-end with mocked router
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_lease_store(tmp_path) -> LeaseStore:
    """LeaseStore rooted in a per-test tmp dir.

    Tests MUST pass this into ``dispatch_for_issue`` rather than
    letting it default to ``~/.eps-routing/`` — a hung pytest under
    real ``time.sleep`` (e.g. an unmocked park-watchdog) would
    otherwise leak a flock onto the user's home and serialize every
    follow-on test on the same issue id.
    """
    return LeaseStore(lease_dir=tmp_path / ".eps-routing")


@pytest.fixture
def fast_clock():
    """Deterministic monotonic clock (1.0s per call) for park-watchdog tests."""
    counter = {"t": 0.0}

    def _now():
        counter["t"] += 1.0
        return counter["t"]

    return _now


class _MockBackend(ComputeBackend):
    """Minimal ABC fill-in for dispatch tests."""

    def __init__(self, kind: BackendKind = "nibi") -> None:
        self._kind = kind
        self.launches: list[RunSpec] = []

    @property
    def name(self) -> BackendKind:
        return self._kind

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        self.launches.append(spec)
        return RunHandle(
            backend=self._kind,
            cluster=self._kind if self._kind != "runpod" else None,
            job_id="job-1",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/scratch",
            log_path="/log",
            extra={"issue": spec.issue},
        )

    def estimate_start(self, spec: RunSpec):
        from datetime import UTC, datetime

        return datetime.now(tz=UTC)

    def estimate_start_seconds(self, spec: RunSpec) -> float | None:
        return 0.0

    def poll(self, handle: RunHandle) -> PollResult:
        return PollResult(
            status="running",
            current_phase="x",
            new_milestone=False,
            last_log_mtime_sec_ago=1,
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


def test_dispatch_for_issue_writes_handle_sidecar(tmp_path, tmp_lease_store) -> None:
    """The orchestrator's bg-Bash poller reads the per-issue handle from
    a sidecar JSON; the dispatch helper MUST write it on every successful
    route."""
    nibi = _MockBackend(kind="nibi")
    spec = RunSpec(issue=200, intent="lora-7b", backend="nibi")
    sidecar = tmp_path / "issue-200-handle.json"
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
    )
    assert isinstance(outcome, DispatchOutcome)
    assert outcome.handle_sidecar_path == sidecar
    assert sidecar.exists()
    # Sidecar round-trips back to a RunHandle matching what route() returned.
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "nibi"
    assert recovered.pod_name == "pod-200"


def test_sidecar_written_before_backend_selected_marker(tmp_path, tmp_lease_store) -> None:
    """C1 ordering regression: the handle sidecar must exist ON DISK
    BEFORE the ``epm:backend-selected`` marker post fires.

    Pre-fix order was launch -> marker post -> sidecar write; a
    marker-post crash (or any crash in between) stranded a live job
    with NO sidecar, so ``dispatch_issue.py finalize`` had nothing to
    tear down. The router's ``on_launched`` hook now persists the
    handle immediately after launch, ahead of every marker."""
    nibi = _MockBackend(kind="nibi")
    spec = RunSpec(issue=204, intent="lora-7b", backend="nibi")
    sidecar = tmp_path / "issue-204-handle.json"
    marker_calls: list[tuple[str, bool]] = []

    def recording_poster(**kwargs):
        # Record whether the sidecar existed at the moment of the post.
        marker_calls.append((kwargs.get("marker", "?"), sidecar.exists()))

    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
        marker_poster=recording_poster,
    )
    assert outcome.handle_sidecar_path == sidecar
    assert marker_calls, "no marker was posted -- the ordering claim was not exercised"
    assert all(existed for _marker, existed in marker_calls), (
        f"marker post(s) fired BEFORE the sidecar landed on disk: {marker_calls!r} -- "
        "a crash at the marker would strand an unrecoverable live job"
    )


def test_dispatch_for_issue_sidecar_oserror_carries_error_not_crash(
    tmp_path, tmp_lease_store, monkeypatch
) -> None:
    """C1: an ``OSError`` on the sidecar write after a SUCCESSFUL launch
    must not escape ``dispatch_for_issue`` (the pre-fix path converted
    it to dispatch-CLI rc=4 with a live job and no handle on stdout).
    The outcome carries ``sidecar_write_error`` so the CLI prints the
    handle JSON + the error loudly instead."""
    import explore_persona_space.backends.issue_dispatch as idp

    def exploding_write(_handle, _path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(idp, "write_handle_sidecar", exploding_write)

    nibi = _MockBackend(kind="nibi")
    spec = RunSpec(issue=205, intent="lora-7b", backend="nibi")
    sidecar = tmp_path / "issue-205-handle.json"
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
    )
    # Launch happened; the failure is carried, not raised.
    assert len(nibi.launches) == 1
    assert outcome.sidecar_write_error is not None
    assert "No space left on device" in outcome.sidecar_write_error
    assert outcome.handle_sidecar_path is None  # nothing landed on disk
    assert outcome.result.handle.pod_name == "pod-205"


def test_dispatch_for_issue_skip_sidecar_when_caller_asks(tmp_lease_store) -> None:
    """``write_sidecar=False`` is for test callers that don't want FS
    writes; the helper must honor it."""
    nibi = _MockBackend(kind="nibi")
    spec = RunSpec(issue=201, intent="lora-7b", backend="nibi")
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        write_sidecar=False,
    )
    assert outcome.handle_sidecar_path is None


def test_dispatch_for_issue_threads_expected_artifacts_when_handle_missing_it(
    tmp_path,
    tmp_lease_store,
) -> None:
    """When the launch path didn't populate ``expected_artifacts``, the
    dispatch helper threads the caller-provided declaration onto
    ``handle.extra`` so ``confirm_artifacts`` has a declaration to
    verify against."""
    nibi = _MockBackend(kind="nibi")  # this mock leaves extra alone
    spec = RunSpec(issue=202, intent="lora-7b", backend="nibi")
    expected = {
        "issue": 202,
        "sentinel_path": "/tmp/.completion-sentinel.json",
        "git_paths": ["eval_results/issue_202/"],
    }
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=tmp_path / "h.json",
        expected_artifacts=expected,
    )
    decl = outcome.result.handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    assert decl["issue"] == 202
    assert decl["sentinel_path"] == "/tmp/.completion-sentinel.json"


def test_dispatch_for_issue_does_not_overwrite_backend_populated_declaration(
    tmp_path,
    tmp_lease_store,
) -> None:
    """A backend that populates ``expected_artifacts`` itself (e.g. GCP)
    must NOT have its declaration overwritten by the dispatch helper's
    caller-provided fallback."""

    class _DeclPopulatingBackend(_MockBackend):
        def launch(self, spec: RunSpec) -> RunHandle:
            return RunHandle(
                backend="nibi",
                cluster="nibi",
                job_id="job-1",
                pod_name=f"pod-{spec.issue}",
                scratch_dir="/scratch",
                log_path="/log",
                extra={
                    "issue": spec.issue,
                    EXPECTED_ARTIFACTS_HANDLE_KEY: {
                        "issue": spec.issue,
                        "sentinel_path": "/backend/sentinel.json",
                    },
                },
            )

    nibi = _DeclPopulatingBackend(kind="nibi")
    spec = RunSpec(issue=203, intent="lora-7b", backend="nibi")
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=tmp_path / "h.json",
        expected_artifacts={"issue": 203, "sentinel_path": "/caller/different.json"},
    )
    decl = outcome.result.handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    # Backend's declaration WINS.
    assert decl["sentinel_path"] == "/backend/sentinel.json"


def _real_slurm_backend(tmp_path, *, job_id: str = "7777"):
    """Real :class:`SlurmBackend` with every external seam faked (no network)."""
    from explore_persona_space.backends.slurm import SlurmBackend

    (tmp_path / "pyproject.toml").write_text("")
    return SlurmBackend(
        src_root=tmp_path,
        submitter=lambda *, robot_alias, sbatch_script: job_id,
        rsyncer=lambda **_kw: None,
        # #1913: prepare now materializes a snapshot via git_cloner and verifies
        # the sync — fake both (returning src_root keeps the launch-side
        # sentinel-path assertions on tmp_path unchanged).
        rsync_verifier=lambda **_kw: None,
        git_cloner=lambda *, src_root, branch, issue: src_root,
        marker_poster=lambda **_kw: None,
        secrets_pusher=lambda **_kw: None,
        runtime_clearer=lambda **_kw: None,
    )


def test_slurm_backend_declaration_not_overwritten_by_caller(tmp_path, tmp_lease_store) -> None:
    """#598 SLURM variant of the key-absent caller-threading guard: the
    REAL ``SlurmBackend.launch`` now populates the declaration, so a
    caller-passed ``expected_artifacts`` dict must NOT overwrite it."""
    nibi = _real_slurm_backend(tmp_path, job_id="7777")
    spec = RunSpec(
        issue=206,
        intent="lora-7b",
        backend="nibi",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
    )
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        write_sidecar=False,
        expected_artifacts={"issue": 206, "sentinel_path": "/caller/should-lose.json"},
    )
    decl = outcome.result.handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    # The launch-built declaration wins: local post-rsync sentinel path
    # under src_root, attempt-namespaced by the SLURM job id.
    assert decl["sentinel_path"] == str(
        tmp_path / "eval_results/issue_206/slurm-7777/.completion-sentinel.json"
    )
    assert decl["hf_data_paths"] == ["issue206_slurm-7777/raw_completions/"]


def test_declaration_survives_sidecar_roundtrip(tmp_path, tmp_lease_store) -> None:
    """#598: the launch-time declaration round-trips through the sidecar
    JSON (``serialize_handle`` → ``write_handle_sidecar`` →
    ``read_handle_sidecar``) and reconstructs to an identical
    :class:`ExpectedArtifacts` (lists tuple-coerced on read) — the
    finalize CLI consumes exactly this recovered form."""
    from explore_persona_space.backends.artifacts import expected_artifacts_from_handle

    nibi = _real_slurm_backend(tmp_path, job_id="7777")
    spec = RunSpec(
        issue=207,
        intent="lora-7b",
        backend="nibi",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em",),
    )
    sidecar = tmp_path / "issue-207-handle.json"
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
    )
    handle = outcome.result.handle
    assert sidecar.exists()
    # Round-trip the exact bytes the bg poller / finalize CLI will read.
    payload = serialize_handle(handle)
    assert (
        payload["extra"][EXPECTED_ARTIFACTS_HANDLE_KEY]
        == handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]
    )
    recovered = read_handle_sidecar(sidecar)
    expected = expected_artifacts_from_handle(recovered)
    assert expected is not None
    assert expected == expected_artifacts_from_handle(handle)
    assert expected.issue == 207
    assert expected.sentinel_path == str(
        tmp_path / "eval_results/issue_207/slurm-7777/.completion-sentinel.json"
    )
    assert expected.hf_data_paths == ("issue207_slurm-7777/raw_completions/",)
    # #790: pure-hydra (no workload_cmd) declares NEITHER default git path —
    # train.py runs with skip_eval=True and writes no figures, so both were
    # guaranteed false-FAILs. Only extra_git_paths (none here) would remain.
    assert expected.git_paths == ()


def test_dispatch_for_issue_raises_router_terminal_for_caller_translation(
    tmp_lease_store, fast_clock
) -> None:
    """``dispatch_for_issue`` is a thin wrapper — it must RAISE router
    terminals verbatim so the orchestrator can translate via
    :func:`classify_terminal_exception` (the split keeps the helper pure).

    #656: the auto chain no longer raises ``NoComputeAvailableError`` when a
    free lane dies + no GCP is wired — it falls through to the RunPod
    terminal rung. The only TRULY-no-compute terminal is now when the RunPod
    rung ITSELF fails. So this wires a RunPod whose ``launch`` raises (no
    compute anywhere); the router re-raises that as ``NoComputeAvailableError``
    and ``dispatch_for_issue`` must propagate it verbatim.
    """

    class _ImmediatelyDeadBackend(_MockBackend):
        def poll(self, handle: RunHandle) -> PollResult:
            return PollResult(
                status="dead",
                current_phase="dead",
                new_milestone=False,
                last_log_mtime_sec_ago=10**9,
                pid_alive=False,
                log_tail_excerpt="",
            )

    class _ExplodingRunpodLaunch(_MockBackend):
        def launch(self, spec: RunSpec) -> RunHandle:
            raise RuntimeError("runpod provisioning failed — no compute anywhere")

    nibi = _ImmediatelyDeadBackend(kind="nibi")
    spec = RunSpec(issue=204, intent="lora-7b", backend="auto")

    with pytest.raises(NoComputeAvailableError):
        dispatch_for_issue(
            spec,
            runpod_backend=_ExplodingRunpodLaunch(kind="runpod"),
            free_backends={"nibi": nibi},
            gcp_backend=None,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            lease_store=tmp_lease_store,
            # Fast clock + 0-cap config so even an unmocked sleep
            # never blocks long enough to leak a flock onto a stuck
            # process — the surface we exposed in
            # ``dispatch_for_issue`` exists to keep tests honest.
            config=RouterConfig(free_wait_seconds=0, poll_interval=0.0, cancel_grace_seconds=0),
            now_fn=fast_clock,
            sleep_fn=lambda _s: None,
            write_sidecar=False,
        )


def test_dispatch_for_issue_threads_started_evidence_probe(tmp_lease_store, fast_clock) -> None:
    """The dispatch helper must thread ``started_evidence_probe`` to the
    router: a fast-failing job (terminal before observed RUNNING) whose
    scratch dir holds runtime artifacts classifies as a WORKLOAD
    failure (surface, NO GCP fallback) — not ``no_compute_available``
    (which would escalate a doomed workload to GCP on the auto lane)."""

    class _ImmediatelyDeadBackend(_MockBackend):
        def poll(self, handle: RunHandle) -> PollResult:
            return PollResult(
                status="dead",
                current_phase="preflight-failed",
                new_milestone=False,
                last_log_mtime_sec_ago=10**9,
                pid_alive=False,
                log_tail_excerpt="",
            )

    nibi = _ImmediatelyDeadBackend(kind="nibi")
    gcp = _MockBackend(kind="gcp")
    spec = RunSpec(issue=205, intent="lora-7b", backend="auto")

    with pytest.raises(WorkloadSurfacedError) as excinfo:
        dispatch_for_issue(
            spec,
            runpod_backend=_MockBackend(kind="runpod"),
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            started_evidence_probe=lambda _b, _h: {
                "phase": "preflight-failed",
                "job_out_tail": "[FAIL] secrets file not found",
                "status_json": {},
            },
            lease_store=tmp_lease_store,
            # Legacy free-first order: the nibi fast-fail classification is
            # the behavior under test; the GCP-first standing default would
            # resolve at GCP before nibi ever launches.
            config=RouterConfig(
                free_wait_seconds=0,
                poll_interval=0.0,
                cancel_grace_seconds=0,
                lane_order=("nibi", "fir", "mila", "gcp"),
            ),
            now_fn=fast_clock,
            sleep_fn=lambda _s: None,
            write_sidecar=False,
        )
    assert excinfo.value.chosen_kind == "nibi"
    assert excinfo.value.evidence.get("phase") == "preflight-failed"
    assert gcp.launches == [], "workload failure must NOT escalate to GCP"


# ---------------------------------------------------------------------------
# Section 4 — Bg-Bash poll contract preservation
# ---------------------------------------------------------------------------


def test_handle_sidecar_round_trips_via_json(tmp_path) -> None:
    """The bg-Bash poller deserializes the handle from the sidecar
    JSON the dispatch helper wrote; round-trip must preserve every
    field the backend uses."""
    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="gce-1234",
        pod_name="eps-issue-300",
        scratch_dir="/workspace/eps-issue-300",
        log_path="/workspace/logs/issue-300.log",
        extra={
            "issue": 300,
            "zone": "us-central1-a",
            "attempt_id": "att-001",
            EXPECTED_ARTIFACTS_HANDLE_KEY: {
                "issue": 300,
                "sentinel_path": "/x/sentinel.json",
                "hf_data_paths": ["foo/"],
                "git_paths": [],
            },
        },
    )
    sidecar = tmp_path / "h.json"
    write_handle_sidecar(handle, sidecar)
    recovered = read_handle_sidecar(sidecar)
    assert recovered == handle


def test_default_handle_sidecar_path_is_absolute_and_cwd_independent(monkeypatch, tmp_path) -> None:
    """The default sidecar is ABSOLUTE, anchored at the MAIN checkout's
    ``.claude/cache/`` regardless of cwd: a launch dispatched from an
    issue worktree and a poll tick run from the repo root must converge
    on the SAME file. The pre-fix cwd-relative form split the contract
    (incident #612: worktree-cwd launch wrote
    ``<worktree>/.claude/cache/``, repo-root poll probed
    ``<root>/.claude/cache/`` → false ``status=dead /
    missing_handle_sidecar`` on a healthy run)."""
    import subprocess

    import explore_persona_space.backends.issue_dispatch as idp

    module_dir = Path(idp.__file__).resolve().parent
    common_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(module_dir),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    expected = common_dir.parent / ".claude" / "cache" / "issue-137-handle.json"

    idp._main_checkout_root.cache_clear()
    path = default_handle_sidecar_path(137)
    assert path.is_absolute()
    assert path == expected

    # cwd-independence: re-resolve from an unrelated cwd with the cache
    # cleared so the git probe actually re-runs (the lru_cache would
    # otherwise mask a cwd-dependent implementation).
    monkeypatch.chdir(tmp_path)
    idp._main_checkout_root.cache_clear()
    assert default_handle_sidecar_path(137) == expected


def test_serialize_handle_round_trips_through_json_strings() -> None:
    """Defensive: the sidecar IS JSON so the serialized form must be
    json-dumps-loads stable."""
    handle = _runpod_handle(issue=400)
    payload = serialize_handle(handle)
    rebuilt = deserialize_handle(json.loads(json.dumps(payload)))
    assert rebuilt == handle


def test_deserialize_handle_rejects_missing_required_field() -> None:
    """A corrupted sidecar with a missing field must FAIL LOUD (not
    silently bind to the wrong handle)."""
    payload = {
        "backend": "runpod",
        "cluster": None,
        "job_id": "",
        # 'pod_name' missing
        "scratch_dir": "/workspace",
        "log_path": "/log",
        "extra": {},
    }
    with pytest.raises(KeyError, match="pod_name"):
        deserialize_handle(payload)


def test_backend_poll_script_produces_legacy_poll_pipeline_json_shape(
    tmp_path, monkeypatch
) -> None:
    """``scripts/backend_poll.py`` must print ONE JSON line whose keys
    match the legacy ``scripts/poll_pipeline.py`` JSON-line contract (the
    PollResult-field subset of ``poll_pipeline.py.main``'s output that
    ``backend_poll._serialize_poll_result`` emits) — that is the
    orchestrator's parser contract."""
    # Write a handle sidecar for a RunPod handle.
    handle = _runpod_handle(issue=500)
    sidecar = tmp_path / "issue-500-handle.json"
    write_handle_sidecar(handle, sidecar)

    # Mock the poll_pipeline so the script's downstream call returns a
    # known PollResult.
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _PR:
        status: str = "done"
        current_phase: str = "done"
        new_milestone: bool = True
        last_log_mtime_sec_ago: int = 0
        pid_alive: bool = True
        log_tail_excerpt: str = "tail"
        gate: str | None = None
        sentinels_processed: int = 3
        phase_log_mtime_sec_ago: int = 5
        shard_log_mtime_sec_ago: int = 6
        gpu_util: str = "95"
        # Non-default sentinel (the serializer's getattr fallback and
        # base.PollResult default are both 540) so the value assertion below
        # discriminates a real thread-through from a silent fall-back.
        next_interval: int = 1800
        # #664 added stall_reason to backends.base.PollResult; the field-set
        # assertion below requires "stall_reason" in the emitted JSON keys.
        stall_reason: str | None = None

    monkeypatch.setattr("scripts.poll_pipeline.poll_once", lambda **kw: _PR())
    # The handle is backend="runpod", so backend_poll's poll path runs
    # _maybe_escalate_runpod_wedge, which calls the LIVE runpod_api.get_pod_by_name
    # (team-scoped GraphQL; raises on unset RUNPOD_API_KEY). Stub the hook to a
    # pass-through so this unit stays hermetic — its pass/fail must depend on the
    # poll-pipeline JSON shape, NOT on ambient RunPod credentials / live team
    # state (#703). The stub intercepts the call before the hook's internal
    # ``from runpod_api import get_pod_by_name`` is reached, so it also avoids
    # the live-API dependency entirely.
    monkeypatch.setattr(
        "scripts.backend_poll._maybe_escalate_runpod_wedge",
        # Signature mirrors the real hook: (handle, result, sidecar, *, now=...).
        # Return the poll RESULT (2nd positional) unchanged — NOT the handle.
        lambda handle, result, *a, **k: result,
    )

    # Capture stdout via a redirect.
    import io
    import sys
    from contextlib import redirect_stdout

    # Ensure ``scripts/`` is on sys.path so ``from scripts.backend_poll import
    # main`` resolves its TRANSITIVE bare ``from runpod_api import ...`` in
    # single-file isolation (mirrors tests/test_runpod_wedge_detection.py:27-30).
    _scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))

    from scripts.backend_poll import main as backend_poll_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = backend_poll_main(["--issue", "500", "--handle-file", str(sidecar)])
    assert rc == 0

    # Output is exactly ONE JSON line with the legacy shape.
    line = buf.getvalue().strip()
    decoded = json.loads(line)
    # Field set must exactly match what poll_pipeline.py.main prints.
    assert set(decoded.keys()) == {
        "status",
        "current_phase",
        "new_milestone",
        "last_log_mtime_sec_ago",
        "pid_alive",
        "log_tail_excerpt",
        "gate",
        "sentinels_processed",
        "phase_log_mtime_sec_ago",
        "shard_log_mtime_sec_ago",
        "gpu_util",
        # Adaptive bg-poll interval (anti-stall redesign §7) — emitted by
        # both poll_pipeline.py.main and backend_poll._serialize_poll_result.
        "next_interval",
        # Machine-readable stall cause (#664) — zombie-GPU-allocation stall on
        # the RunPod lane; None on every other lane / verdict.
        "stall_reason",
        # #983 post-done phase-consistency guard — emitted by both
        # poll_pipeline.py.main and backend_poll._serialize_poll_result
        # (getattr-defended defaults False / [] on lanes that never set them).
        "post_done_phase_advisory_posted",
        "post_done_phase_lines",
        # GCP-lane GPU-idle advisory + escalation parity (#730 / #727 RunPod
        # analogue) — always emitted by backend_poll.main, default False on
        # every non-GCP / non-running tick.
        "gcp_gpu_idle_advisory_posted",
        "gcp_gpu_idle_escalation_posted",
        # GCP-lane GPU-WIDTH advisory (#873, the width mirror of the #730
        # idle reuse) — always emitted by backend_poll.main, default False.
        # (Pin update landed with #909's green-gate pass: #873 added the
        # emitter without updating this shape test — pre-existing on main.)
        "gcp_gpu_width_advisory_posted",
        # #1786 WARN-only handle-staleness flags — always emitted by
        # backend_poll.main on the NORMAL tick-JSON tail, default False;
        # never verdict-bearing (WARN-only observability).
        "handle_stale_vs_live",
        "handle_older_than_relaunch",
    }
    # Values were correctly threaded through.
    assert decoded["status"] == "done"
    assert decoded["new_milestone"] is True
    assert decoded["sentinels_processed"] == 3
    assert decoded["next_interval"] == 1800


def test_backend_poll_script_resolves_per_backend_class(tmp_path, monkeypatch) -> None:
    """The script's backend resolver MUST instantiate the right class
    per ``handle.backend`` (so a single script handles every backend)."""
    from scripts.backend_poll import _resolve_backend

    # All three live backends are constructible.
    runpod = _resolve_backend("runpod")
    assert runpod.__class__.__name__ == "RunPodBackend"
    slurm = _resolve_backend("nibi")
    assert slurm.__class__.__name__ == "SlurmBackend"
    gcp = _resolve_backend("gcp")
    assert gcp.__class__.__name__ == "GcpBackend"
    # Unknown backend raises (no silent default to RunPod that would
    # mis-route a GCP/SLURM poll).
    with pytest.raises(ValueError, match="unknown backend"):
        _resolve_backend("totally-bogus")


# ---------------------------------------------------------------------------
# issue #588 — build_run_spec threads workload_cmd
# ---------------------------------------------------------------------------


def test_build_run_spec_threads_workload_cmd() -> None:
    spec = build_run_spec(
        issue=588,
        intent="debug",
        backend_value=None,
        workload_cmd="bash scripts/issue588_smoke.sh",
    )
    assert spec.workload_cmd == "bash scripts/issue588_smoke.sh"
    assert spec.hydra_args == ()
    assert spec.backend == "auto"


def test_build_run_spec_workload_cmd_default_empty() -> None:
    """Builder stays permissive on neither (test factories +
    finalize-adjacent uses build bare specs)."""
    spec = build_run_spec(issue=588, intent="debug", backend_value=None)
    assert spec.workload_cmd == ""


def test_build_run_spec_both_workload_cmd_and_hydra_raises() -> None:
    """Both-set propagates the RunSpec.__post_init__ raise."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_run_spec(
            issue=588,
            intent="debug",
            backend_value=None,
            hydra_args=("seed=1",),
            workload_cmd="bash scripts/issue588_smoke.sh",
        )


# ---------------------------------------------------------------------------
# #934: per-lane handle sidecar + reconnect loudness
# ---------------------------------------------------------------------------


def test_default_handle_sidecar_path_lane_suffix(monkeypatch, tmp_path) -> None:
    """Suffixed sidecar stem (#934) + unsuffixed byte-identity + fail-loud
    validation of a malformed suffix."""
    import explore_persona_space.backends.issue_dispatch as idp

    monkeypatch.setattr(idp, "_main_checkout_root", lambda: tmp_path)
    assert default_handle_sidecar_path(137).name == "issue-137-handle.json"
    assert default_handle_sidecar_path(137, lane_suffix=None).name == "issue-137-handle.json"
    assert default_handle_sidecar_path(137, lane_suffix="cpu").name == "issue-137-cpu-handle.json"
    # Same parent dir either way (the .claude/cache anchor is unchanged).
    assert (
        default_handle_sidecar_path(137, lane_suffix="cpu").parent
        == default_handle_sidecar_path(137).parent
    )
    with pytest.raises(ValueError):
        default_handle_sidecar_path(137, lane_suffix="Bad_Suffix")


def test_dispatch_for_issue_writes_lane_suffixed_sidecar(
    tmp_path, tmp_lease_store, monkeypatch
) -> None:
    """A spec whose extra carries lane_suffix lands the sidecar at the
    SUFFIXED canonical path — two lanes keep independent handles (#934)."""
    import explore_persona_space.backends.issue_dispatch as idp

    monkeypatch.setattr(idp, "_main_checkout_root", lambda: tmp_path)
    nibi = _MockBackend(kind="nibi")
    spec = RunSpec(issue=201, intent="lora-7b", backend="nibi", extra={"lane_suffix": "cpu"})
    outcome = dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": nibi},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
    )
    expected = tmp_path / ".claude" / "cache" / "issue-201-cpu-handle.json"
    assert outcome.handle_sidecar_path == expected
    assert expected.exists()
    # The unsuffixed sidecar was NOT touched.
    assert not (tmp_path / ".claude" / "cache" / "issue-201-handle.json").exists()


class _ReconnectedExtraBackend(_MockBackend):
    """launch() returns a handle marked extra['reconnected']=True — the
    GCP-INTERNAL reconnect shape (route() thinks it launched fresh, the
    reason is NOT 'reconnect'; only the handle extra carries the flag)."""

    def launch(self, spec: RunSpec) -> RunHandle:
        from dataclasses import replace

        h = super().launch(spec)
        return replace(h, extra={**h.extra, "reconnected": True})


def _dispatch_reconnect_cell(
    *,
    issue: int,
    tmp_path,
    tmp_lease_store,
    reconnect_source: str,
    workload: str | None,
):
    """Drive dispatch_for_issue through one (reconnect-branch x workload) cell."""
    spec_kwargs: dict[str, Any] = {}
    if workload == "workload_cmd":
        spec_kwargs["workload_cmd"] = "bash scripts/x.sh"
    elif workload == "hydra":
        spec_kwargs["hydra_args"] = ("smoke=1",)
    spec = RunSpec(issue=issue, intent="lora-7b", backend="nibi", **spec_kwargs)
    dispatch_kwargs: dict[str, Any] = dict(
        runpod_backend=_MockBackend(kind="runpod"),
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=tmp_path / f"issue-{issue}-handle.json",
    )
    if reconnect_source == "reason":
        # Router-scan reconnect: reason == ROUTE_REASON_RECONNECT; the
        # recovered handle carries NO extra['reconnected'] flag.
        live = RunHandle(
            backend="nibi",
            cluster="nibi",
            job_id="live-1",
            pod_name=f"eps-issue-{issue}",
            scratch_dir="/s",
            log_path="/l",
            extra={"issue": issue},
        )
        dispatch_kwargs["free_backends"] = {"nibi": _MockBackend(kind="nibi")}
        dispatch_kwargs["reconnect_fn"] = lambda _b, k, _s: live if k == "nibi" else None
    elif reconnect_source == "extra":
        # Backend-internal reconnect: fresh-looking reason, only the
        # handle extra carries reconnected=True (the GCP shape).
        dispatch_kwargs["free_backends"] = {"nibi": _ReconnectedExtraBackend(kind="nibi")}
    else:  # no reconnect at all
        dispatch_kwargs["free_backends"] = {"nibi": _MockBackend(kind="nibi")}
    return dispatch_for_issue(spec, **dispatch_kwargs)


_RECONNECT_WARNING_SNIPPET = "dispatched NO workload"


@pytest.mark.parametrize("workload", ["workload_cmd", "hydra"])
@pytest.mark.parametrize("reconnect_source", ["reason", "extra"])
def test_dispatch_for_issue_reconnect_with_workload_warns(
    tmp_path, tmp_lease_store, caplog, reconnect_source: str, workload: str
) -> None:
    """Round-1 Must-Fix matrix: BOTH reconnect branches (router-scan
    reason-only x GCP-internal extra-only) x BOTH workload surfaces
    (workload_cmd x hydra_args) must emit the library-level warning — an
    implementation checking only `reason` (or only workload_cmd) FAILs a
    named cell (the exact #923 recurrence path)."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        _dispatch_reconnect_cell(
            issue=202,
            tmp_path=tmp_path,
            tmp_lease_store=tmp_lease_store,
            reconnect_source=reconnect_source,
            workload=workload,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any(_RECONNECT_WARNING_SNIPPET in m for m in messages), messages


@pytest.mark.parametrize("reconnect_source", ["reason", "extra"])
def test_dispatch_for_issue_reconnect_without_workload_no_warning(
    tmp_path, tmp_lease_store, caplog, reconnect_source: str
) -> None:
    """A workload-less spec (est-start probes, bare reconnect ticks) never
    warns — reconnect IS the desired outcome there."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        _dispatch_reconnect_cell(
            issue=203,
            tmp_path=tmp_path,
            tmp_lease_store=tmp_lease_store,
            reconnect_source=reconnect_source,
            workload=None,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert not any(_RECONNECT_WARNING_SNIPPET in m for m in messages), messages


@pytest.mark.parametrize("workload", ["workload_cmd", "hydra"])
def test_dispatch_for_issue_fresh_launch_with_workload_no_warning(
    tmp_path, tmp_lease_store, caplog, workload: str
) -> None:
    """A genuinely fresh launch (no reconnect signal on either layer)
    never warns."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        _dispatch_reconnect_cell(
            issue=205,
            tmp_path=tmp_path,
            tmp_lease_store=tmp_lease_store,
            reconnect_source="none",
            workload=workload,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert not any(_RECONNECT_WARNING_SNIPPET in m for m in messages), messages


def test_cpu_fallback_infeasible_maps_to_distinct_reason() -> None:
    """#1010: a CpuFallbackInfeasibleError maps to the DISTINCT reason
    cpu_fallback_infeasible_for_plan — NOT the parent's
    cpu_exhausted_no_runpod_lane and NOT no_compute_available — so the
    watcher's capacity-retry pass never hot-retries a structurally-infeasible
    launch, and the note carries the concrete cpu-bigmem recovery command.

    Removing the isinstance branch (or placing it AFTER the parent
    CpuExhaustedNoRunpodLaneError branch) turns this RED — the subclass would
    be shadowed and emit the parent's reason."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
        CpuFallbackInfeasibleError,
    )

    exc = CpuFallbackInfeasibleError(
        "CPU intent 'cpu-mid': RunPod CPU fallback (cpu3c-8-16) cannot satisfy "
        "the plan footprint — disk: plan requires 80 GB > cpu3c-8-16 max "
        "container disk 50 GB",
        attempts=[{"kind": "gcp", "outcome": "capacity_miss"}],
    )
    t = classify_terminal_exception(exc)
    assert t.failure_class == "infra"
    assert t.status == "blocked"
    assert f"reason: {ROUTE_REASON_CPU_FALLBACK_INFEASIBLE}" in t.note
    assert "reason: cpu_exhausted_no_runpod_lane" not in t.note
    assert "reason: no_compute_available" not in t.note
    # The recovery line names the big-footprint lane.
    assert "cpu-bigmem" in t.note
    assert "detail: CPU intent 'cpu-mid'" in t.note


def test_cpu_exhausted_parent_does_not_emit_infeasible_reason() -> None:
    """#1010 control: the PARENT CpuExhaustedNoRunpodLaneError keeps its own
    reason verbatim — the new subclass branch narrows, never widens."""
    from explore_persona_space.backends.router import (
        ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
        CpuExhaustedNoRunpodLaneError,
    )

    exc = CpuExhaustedNoRunpodLaneError(
        "CPU intent 'cpu-bigmem': GCP exhausted and RunPod has no CPU lane",
        attempts=[],
    )
    t = classify_terminal_exception(exc)
    assert "reason: cpu_exhausted_no_runpod_lane" in t.note
    assert ROUTE_REASON_CPU_FALLBACK_INFEASIBLE not in t.note


# ---------------------------------------------------------------------------
# #1122 — reconnect sidecar rewrites carry forward the prior workload extras
#
# Incident #1090: an exit-75 same-command rerun RECONNECTED (GCP-internal,
# handle extra reconnected=True) and dispatch_for_issue's on_launched hook
# OVERWROTE the complete launch-handle sidecar with the minimal reconnect
# handle — stranding the #783 queue-timeout failover. Edit 1 (#1122) fixes
# the workload-carrying rerun at the producer; these tests pin edit 2's
# residual-class safety net: a workload-LESS reconnect rewrite (manual
# provision-only re-invocation) merges the PRIOR sidecar's failover extras
# into both sidecar writes, bound on backend + pod_name + job_id identity.
# ---------------------------------------------------------------------------

_PRIOR_WORKLOAD_CMD_1122 = "REPO_ROOT=/workspace bash scripts/issue1122_dispatch.sh --full"

#: The COMPLETE prior sidecar extra (the launch-path failover keys, #659/
#: #909/#677/#1010) — repo_branch deliberately NON-empty (#1122 §11.5(4)).
_PRIOR_EXTRA_1122: dict[str, Any] = {
    "issue": 1122,
    "intent": "lora-7b",
    "zone": "us-central1-a",
    "workload_cmd": _PRIOR_WORKLOAD_CMD_1122,
    "hydra_args": [],
    "gpus": 1,
    "time_budget_hours": 4.0,
    "repo_branch": "issue-1122",
    "gpu_count": 1,
    "boot_disk_gb": 200,
}


def _prior_handle_1122(*, backend: str = "nibi", job_id: str = "gce-777") -> RunHandle:
    """The COMPLETE predecessor handle a prior launch wrote to the sidecar."""
    return RunHandle(
        backend=backend,
        cluster=backend if backend != "runpod" else None,
        job_id=job_id,
        pod_name="eps-issue-1122",
        scratch_dir="/s",
        log_path="/l",
        extra=dict(_PRIOR_EXTRA_1122),
    )


class _ReconnectShapedBackend(_MockBackend):
    """launch() returns a RECONNECT-shaped handle mirroring the GCP
    ``reconnect_or_none`` MINIMAL extra for a workload-LESS spec —
    ``reconnected=True`` + probe-derived keys only, NO workload extras —
    with a FIXED job_id so a pre-seeded prior sidecar can identity-bind."""

    def __init__(self, kind: BackendKind = "nibi", job_id: str = "gce-777") -> None:
        super().__init__(kind=kind)
        self._job_id = job_id

    def launch(self, spec: RunSpec) -> RunHandle:
        self.launches.append(spec)
        return RunHandle(
            backend=self._kind,
            cluster=self._kind if self._kind != "runpod" else None,
            job_id=self._job_id,
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir="/s",
            log_path="/l",
            extra={
                "issue": spec.issue,
                "reconnected": True,
                "status_at_reconnect": "RUNNING",
            },
        )


def _dispatch_1122(spec: RunSpec, *, sidecar: Path, tmp_lease_store) -> DispatchOutcome:
    """Drive dispatch_for_issue through the reconnect-shaped fake backend."""
    return dispatch_for_issue(
        spec,
        runpod_backend=_MockBackend(kind="runpod"),
        free_backends={"nibi": _ReconnectShapedBackend()},
        is_started=lambda _b, _h: True,
        lease_store=tmp_lease_store,
        handle_sidecar_path=sidecar,
    )


def test_dispatch_reconnect_rewrite_preserves_prior_sidecar_workload_extras(
    tmp_path, tmp_lease_store
) -> None:
    """T7 (#1122): a workload-LESS re-invocation whose backend reconnects
    (reconnected=True) no longer clobbers the sidecar's workload extras —
    the prior sidecar's failover keys (incl. the NON-empty repo_branch,
    §11.5(4)) survive onto BOTH the on-disk sidecar and the returned
    handle."""
    sidecar = tmp_path / "issue-1122-handle.json"
    write_handle_sidecar(_prior_handle_1122(), sidecar)

    spec = RunSpec(issue=1122, intent="lora-7b", backend="nibi")  # bare — no workload
    outcome = _dispatch_1122(spec, sidecar=sidecar, tmp_lease_store=tmp_lease_store)

    recovered = read_handle_sidecar(sidecar)
    assert recovered.extra["workload_cmd"] == _PRIOR_WORKLOAD_CMD_1122
    assert recovered.extra["hydra_args"] == []
    assert recovered.extra["repo_branch"] == "issue-1122"
    for key in ("gpus", "time_budget_hours", "gpu_count", "boot_disk_gb"):
        assert recovered.extra[key] == _PRIOR_EXTRA_1122[key], key
    # The reconnect probe keys from the NEW handle are retained.
    assert recovered.extra["reconnected"] is True
    assert recovered.extra["status_at_reconnect"] == "RUNNING"
    # The RETURNED handle (the authoritative-write input + the caller's
    # view) carries the same merge.
    assert outcome.result.handle.extra["workload_cmd"] == _PRIOR_WORKLOAD_CMD_1122
    assert outcome.result.handle.extra["repo_branch"] == "issue-1122"


def test_carry_forward_skips_on_job_id_mismatch() -> None:
    """T8 (#1122): a stale sidecar from a DEAD incarnation (different GCE
    instance id) never donates — and an EMPTY job_id on either side is
    treated as no-match (§11.5(1)), so a degenerate ("gcp", name, "")
    binding cannot match across incarnations."""
    from dataclasses import replace

    from explore_persona_space.backends.issue_dispatch import _carry_forward_reconnect_extras

    handle = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="gce-NEW",
        pod_name="eps-issue-1122",
        scratch_dir="/s",
        log_path="/l",
        extra={"issue": 1122, "reconnected": True},
    )
    prior = {
        "backend": "nibi",
        "pod_name": "eps-issue-1122",
        "job_id": "gce-OLD",
        "extra": {"workload_cmd": _PRIOR_WORKLOAD_CMD_1122, "hydra_args": []},
    }
    assert _carry_forward_reconnect_extras(handle, prior) is handle  # identity

    # Empty job_id on BOTH sides would tuple-compare equal — must no-match.
    handle_empty = replace(handle, job_id="")
    prior_empty = {**prior, "job_id": ""}
    assert _carry_forward_reconnect_extras(handle_empty, prior_empty) is handle_empty


def test_carry_forward_noop_on_fresh_launch_handle() -> None:
    """T9 (#1122): a FRESH-launch handle (no ``reconnected`` flag — every
    non-GCP-reconnect shape, incl. SLURM query_by_name recoveries) is
    returned unchanged (identity): the merge is reconnect-scoped by
    construction."""
    from explore_persona_space.backends.issue_dispatch import _carry_forward_reconnect_extras

    handle = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="gce-777",
        pod_name="eps-issue-1122",
        scratch_dir="/s",
        log_path="/l",
        extra={"issue": 1122},  # no 'reconnected'
    )
    prior = {
        "backend": "nibi",
        "pod_name": "eps-issue-1122",
        "job_id": "gce-777",
        "extra": {"workload_cmd": _PRIOR_WORKLOAD_CMD_1122, "hydra_args": []},
    }
    assert _carry_forward_reconnect_extras(handle, prior) is handle


def test_carry_forward_workload_pair_atomic() -> None:
    """T10 (#1122): the (workload_cmd, hydra_args) pair merges ATOMICALLY —
    a handle already carrying hydra_args is never given a second
    workload_cmd (RunSpec mutual exclusion) — while the per-key fills
    still run: edit 1's ``repo_branch: ""`` write is treated as absent and
    fills from the prior's non-empty value (§11.5(4))."""
    from explore_persona_space.backends.issue_dispatch import _carry_forward_reconnect_extras

    handle = RunHandle(
        backend="nibi",
        cluster="nibi",
        job_id="gce-777",
        pod_name="eps-issue-1122",
        scratch_dir="/s",
        log_path="/l",
        extra={
            "issue": 1122,
            "reconnected": True,
            # The handle already carries ONE side of the pair.
            "hydra_args": ["smoke=1"],
            # Edit 1 writes "" when the rerun spec has no repo_branch.
            "repo_branch": "",
        },
    )
    prior = {
        "backend": "nibi",
        "pod_name": "eps-issue-1122",
        "job_id": "gce-777",
        "extra": {
            "workload_cmd": _PRIOR_WORKLOAD_CMD_1122,
            "hydra_args": [],
            "repo_branch": "issue-1122",
            "gpus": 1,
        },
    }
    out = _carry_forward_reconnect_extras(handle, prior)
    assert out is not handle  # something carried
    # The pair did NOT merge — the handle carries hydra_args already.
    assert "workload_cmd" not in out.extra
    assert out.extra["hydra_args"] == ["smoke=1"]
    # Mutual exclusion holds on the merged shape.
    assert not (out.extra.get("workload_cmd") and out.extra.get("hydra_args"))
    # Per-key fills still ran: ""-repo_branch filled from the prior; gpus filled.
    assert out.extra["repo_branch"] == "issue-1122"
    assert out.extra["gpus"] == 1


def test_on_launched_early_write_carries_reconnect_merge(
    tmp_path, tmp_lease_store, monkeypatch
) -> None:
    """T11 (#1122, round-1 Statistics Must-Fix): the ``on_launched`` EARLY
    sidecar write — not just the authoritative post-route write — carries
    the carry-forward merge. Simulate a crash BETWEEN the two writes (a
    signature-conformant raiser patched over _warn_on_reconnected_workload,
    which sits after the early write and before the authoritative write):
    the ON-DISK sidecar left by the early write alone must already carry
    the prior workload extras, so a crash window can never strand an
    un-merged impoverished sidecar."""

    def _boom(spec: RunSpec, result, handle: RunHandle) -> None:
        raise RuntimeError("simulated crash between the two sidecar writes (#1122 T11)")

    monkeypatch.setattr(
        "explore_persona_space.backends.issue_dispatch._warn_on_reconnected_workload",
        _boom,
    )
    sidecar = tmp_path / "issue-1122-handle.json"
    write_handle_sidecar(_prior_handle_1122(), sidecar)

    spec = RunSpec(issue=1122, intent="lora-7b", backend="nibi")  # bare — no workload
    with pytest.raises(RuntimeError, match="simulated crash between the two sidecar writes"):
        _dispatch_1122(spec, sidecar=sidecar, tmp_lease_store=tmp_lease_store)

    # The authoritative write never ran; the on-disk sidecar IS the early
    # on_launched write — and it already carries the merged extras.
    recovered = read_handle_sidecar(sidecar)
    assert recovered.extra["reconnected"] is True
    assert recovered.extra["workload_cmd"] == _PRIOR_WORKLOAD_CMD_1122
    assert recovered.extra["hydra_args"] == []
    assert recovered.extra["repo_branch"] == "issue-1122"
    for key in ("gpus", "time_budget_hours", "gpu_count", "boot_disk_gb"):
        assert recovered.extra[key] == _PRIOR_EXTRA_1122[key], key


# ---------------------------------------------------------------------------
# #1669 — launch env pins: reconnect carry-forward + validator rules
# ---------------------------------------------------------------------------


def test_reconnect_carry_forward_includes_env_pins(tmp_path) -> None:
    """#1669: ``env_pins`` is a RECONNECT_CARRY_FORWARD_EXTRA_KEYS member
    and the sidecar snapshot keeps a non-empty pins dict (the
    ``v not in (None, "", [])`` filter keeps a dict — plan assumption 8 —
    and skips an absent key on a legacy sidecar)."""
    from explore_persona_space.backends.issue_dispatch import (
        RECONNECT_CARRY_FORWARD_EXTRA_KEYS,
        _prior_sidecar_failover_extras,
        write_handle_sidecar,
    )

    assert "env_pins" in RECONNECT_CARRY_FORWARD_EXTRA_KEYS

    pins = {"WANDB_PROJECT": "issue1586_methodgen"}
    sidecar = tmp_path / "issue-1669-handle.json"
    write_handle_sidecar(
        RunHandle(
            backend="gcp",
            cluster=None,
            job_id="gce-1",
            pod_name="eps-issue-1669",
            scratch_dir="/s",
            log_path="/l",
            extra={"workload_cmd": "bash scripts/x.sh", "hydra_args": [], "env_pins": pins},
        ),
        sidecar,
    )
    prior = _prior_sidecar_failover_extras(sidecar)
    assert prior is not None
    assert prior["extra"]["env_pins"] == pins

    # Legacy sidecar (no env_pins key): the snapshot omits it.
    sidecar2 = tmp_path / "issue-1669-legacy-handle.json"
    write_handle_sidecar(
        RunHandle(
            backend="gcp",
            cluster=None,
            job_id="gce-2",
            pod_name="eps-issue-1669",
            scratch_dir="/s",
            log_path="/l",
            extra={"workload_cmd": "bash scripts/x.sh", "hydra_args": []},
        ),
        sidecar2,
    )
    prior2 = _prior_sidecar_failover_extras(sidecar2)
    assert prior2 is not None
    assert "env_pins" not in prior2["extra"]


@pytest.mark.parametrize(
    ("pins", "ok"),
    [
        ({"WANDB_PROJECT": "issue1586_methodgen"}, True),
        ({"WANDB_TAGS": "a=b", "WANDB_RUN_GROUP": "g 1"}, True),
        ({"MALLOC_ARENA_MAX": "2", "OMP_NUM_THREADS": "8"}, True),  # #1803 runtime-tuning keys
        (None, True),  # None → {}
        ({}, True),  # empty → {}
        ({"WANDB_API_KEY": "x"}, False),  # non-allowlisted (secret) key
        ({"wandb_project": "x"}, False),  # case-sensitive allowlist
        ({"WANDB_PROJECT": ""}, False),  # empty value
        ({"WANDB_PROJECT": "a\nb"}, False),  # multi-line value
        ({"WANDB_PROJECT": 42}, False),  # non-str value
        ({"WANDB_PROJECT": "x" * 513}, False),  # over ENV_PIN_VALUE_MAX_LEN
        ("WANDB_PROJECT=x", False),  # non-mapping input
    ],
)
def test_validate_env_pins_allowlist_and_value_rules(pins, ok) -> None:
    """#1669: the strict validator's allowlist + value rules (the CLI +
    renderer defense); the secret-shaped-value case is covered separately
    below (runtime-constructed token, never a committed literal)."""
    from explore_persona_space.backends.base import validate_env_pins

    if ok:
        out = validate_env_pins(pins)
        assert out == (dict(pins) if pins else {})
    else:
        with pytest.raises(ValueError):
            validate_env_pins(pins)


def test_validate_env_pins_rejects_secret_shaped_value_and_sanitize_splits() -> None:
    """#1669: a secret-shaped VALUE is rejected by the strict validator,
    and ``sanitize_env_pins`` (the reconstructor-side per-key variant)
    keeps valid entries while reporting each dropped one."""
    from explore_persona_space.backends.base import sanitize_env_pins, validate_env_pins

    secret_shaped = "sk-" + "A" * 20  # constructed at runtime
    with pytest.raises(ValueError, match="secret-shaped"):
        validate_env_pins({"WANDB_PROJECT": secret_shaped})

    kept, dropped = sanitize_env_pins(
        {"WANDB_PROJECT": "ok", "WANDB_RUN_GROUP": secret_shaped, "HF_TOKEN": "x"}
    )
    assert kept == {"WANDB_PROJECT": "ok"}
    assert len(dropped) == 2
    # Non-mapping input drops wholesale with one reason, never raises.
    kept2, dropped2 = sanitize_env_pins(["WANDB_PROJECT=x"])
    assert kept2 == {} and len(dropped2) == 1


def test_env_pin_allowlist_keeps_runtime_tuning_keys() -> None:
    """#1803: the house runtime-tuning set (OOM / thread-cap remediation,
    incident #1739) stays in ``ENV_PIN_ALLOWED_KEYS`` — a silent drop in a
    future allowlist rewrite turns this membership pin red.
    #1852 adds the CUDA allocator knob (gotchas.md CUDA-OOM remedy #1)."""
    from explore_persona_space.backends.base import ENV_PIN_ALLOWED_KEYS, sanitize_env_pins

    runtime_tuning = {
        "MALLOC_ARENA_MAX",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    assert runtime_tuning <= ENV_PIN_ALLOWED_KEYS

    # #1852: the CUDA allocator knob round-trips ``sanitize_env_pins`` with
    # the gotchas.md hot-fix value (colon is a legal single-line char).
    pin = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    kept, dropped = sanitize_env_pins(pin)
    assert kept == pin
    assert dropped == []
    # Comma-bearing multi-option value survives too (free coverage).
    pin_multi = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True"}
    kept2, dropped2 = sanitize_env_pins(pin_multi)
    assert kept2 == pin_multi
    assert dropped2 == []


# ---------------------------------------------------------------------------
# #2161 — per-process free-lane park budget env wiring
# ---------------------------------------------------------------------------


def test_env_park_process_budget_resolution(monkeypatch):
    """#2161: EPS_LAUNCH_PARK_PROCESS_BUDGET_SECONDS resolves at call time —
    unset/malformed → the 420 s default; 0/negative → None (unlimited,
    the legacy park semantics); a positive int → itself."""
    from explore_persona_space.backends import issue_dispatch as idp

    monkeypatch.delenv(idp.PARK_PROCESS_BUDGET_ENV, raising=False)
    assert idp._env_park_process_budget() == idp.PARK_PROCESS_BUDGET_DEFAULT_SECONDS == 420
    monkeypatch.setenv(idp.PARK_PROCESS_BUDGET_ENV, "not-a-number")
    assert idp._env_park_process_budget() == 420
    monkeypatch.setenv(idp.PARK_PROCESS_BUDGET_ENV, "")
    assert idp._env_park_process_budget() == 420
    monkeypatch.setenv(idp.PARK_PROCESS_BUDGET_ENV, "0")
    assert idp._env_park_process_budget() is None
    monkeypatch.setenv(idp.PARK_PROCESS_BUDGET_ENV, "-5")
    assert idp._env_park_process_budget() is None
    monkeypatch.setenv(idp.PARK_PROCESS_BUDGET_ENV, "900")
    assert idp._env_park_process_budget() == 900
