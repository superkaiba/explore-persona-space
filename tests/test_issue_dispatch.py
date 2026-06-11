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

        return _PR()

    monkeypatch.setattr("scripts.poll_pipeline.poll_once", fake_poll_once)
    backend = RunPodBackend()
    handle = _runpod_handle(issue=137)
    result = backend.poll(handle)
    assert isinstance(result, PollResult)
    assert result.status == "running"
    assert result.current_phase == "phase-foo"
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
# Section 2 — GCP fetch_results scp-back
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
        log_path=f"{workload}/logs/issue-{issue}.log",
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


def test_gcp_fetch_results_issues_sentinel_scp_before_anything_else(tmp_path) -> None:
    """The sentinel pull is MANDATORY (the verifier reads it locally;
    a missing local sentinel = silent-loss). It is issued first so its
    failure surfaces before the best-effort dir pulls."""
    runner = _RecordingGcloudRunner()
    backend = GcpBackend(
        config=_gcp_config(vm_scratch_dir=str(tmp_path)),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    backend.fetch_results(_gcp_handle(vm_scratch_dir=str(tmp_path)))
    assert runner.calls, "fetch_results made no gcloud call"
    first_argv = runner.calls[0]
    # First call must be `gcloud compute scp <host>:<sentinel_remote> <local>`.
    assert "scp" in first_argv
    assert "compute" in first_argv
    sentinel_arg = next(a for a in first_argv if "completion-sentinel" in a)
    # The sentinel argv has the host: prefix.
    assert sentinel_arg.startswith("eps-issue-137:")
    assert ".completion-sentinel.json" in sentinel_arg


def test_gcp_fetch_results_falls_back_best_effort_on_artifact_dir_failure(tmp_path) -> None:
    """A best-effort dir scp failure logs + continues (eval_results/figures
    are authoritative on HF/WandB/git already)."""
    runner = _RecordingGcloudRunner(
        results=[
            GcloudRunResult(returncode=0, stdout="", stderr=""),  # sentinel PASS
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
    # Three scp calls were issued (sentinel + two dir pulls).
    scp_calls = [a for a in runner.calls if "scp" in a]
    assert len(scp_calls) == 3


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
        log_path=str(tmp_path / "eps-issue-0/logs/issue-0.log"),
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
        log_path=str(tmp_path / "eps-issue-137/logs/issue-137.log"),
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


def test_dispatch_for_issue_raises_router_terminal_for_caller_translation(
    tmp_lease_store, fast_clock
) -> None:
    """``dispatch_for_issue`` is a thin wrapper — it must RAISE router
    terminals verbatim so the orchestrator can translate via
    :func:`classify_terminal_exception` (the split keeps the helper pure).

    Uses a free-lane backend whose ``poll`` immediately reports a
    terminal status — so ``park_until_running_or_cap`` returns
    ``terminal_before_running`` without ever sleeping. With no GCP
    backend wired, the auto-route chain reaches
    :class:`NoComputeAvailableError`.
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

    nibi = _ImmediatelyDeadBackend(kind="nibi")
    spec = RunSpec(issue=204, intent="lora-7b", backend="auto")

    with pytest.raises(NoComputeAvailableError):
        dispatch_for_issue(
            spec,
            runpod_backend=_MockBackend(kind="runpod"),
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
        log_path="/workspace/eps-issue-300/logs/issue-300.log",
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


def test_default_handle_sidecar_path_is_under_cache_dir() -> None:
    """The default sidecar lives under ``.claude/cache/`` (worktree-
    relative) so a stale sidecar is reaped by the worktree audit cron
    alongside the rest of the cache."""
    path = default_handle_sidecar_path(137)
    assert str(path) == ".claude/cache/issue-137-handle.json"


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
    match ``scripts/poll_pipeline.py.main``'s output — that is the
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

    monkeypatch.setattr("scripts.poll_pipeline.poll_once", lambda **kw: _PR())

    # Capture stdout via a redirect.
    import io
    from contextlib import redirect_stdout

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
    }
    # Values were correctly threaded through.
    assert decoded["status"] == "done"
    assert decoded["new_milestone"] is True
    assert decoded["sentinels_processed"] == 3


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
