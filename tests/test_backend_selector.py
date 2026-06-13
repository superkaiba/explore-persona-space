"""Selector decision-table + RunPod-characterization tests.

Slice 1 of the SLURM cluster backend ships the selector control flow
(real) + a SLURM stub that raises ``NotImplementedError`` with a
sentinel message. The selector MUST:

1. Route a task with no ``backend:`` frontmatter to the RunPod backend
   with ZERO mention of the cluster code path (the "byte-for-byte
   RunPod preserved" guarantee from the plan).
2. Route an explicit ``backend: runpod`` to RunPod as well.
3. For ``backend: cluster|nibi|fir``, attempt the SLURM backend, catch
   the slice-1 ``NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE)``,
   and fall back to RunPod recording ``reason="slurm_not_implemented"``.
4. For a SLURM backend that submits successfully but stays PENDING past
   ``max_wait_seconds``, ``scancel`` and fall back to RunPod with
   ``reason="slurm_max_wait_exceeded"``.
5. For a SLURM backend that hard-fails the launch (e.g. auth error),
   fall back to RunPod with ``reason="slurm_hard_failure"``.

These tests mock RunPod and SLURM backends so no real provision happens.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest import mock

import pytest

from explore_persona_space.backends import (
    SLURM_NOT_IMPLEMENTED_MESSAGE,
    BackendDecision,
    BackendKind,
    BackendSelectionError,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
    select_backend,
)
from explore_persona_space.backends.selector import (
    _is_slurm_stub_unavailable,
    _parse_backend_kind,
    _resolve_cluster_name,
    _SlurmStubBackend,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRunPodBackend(ComputeBackend):
    """RunPod test double — records launches without provisioning."""

    def __init__(self) -> None:
        self.launches: list[RunSpec] = []
        self.teardowns: list[RunHandle] = []

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
            job_id="fake-pod-id",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/workspace",
            log_path=f"/workspace/logs/issue-{spec.issue}.log",
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        return datetime.now(tz=UTC)

    def poll(self, handle: RunHandle) -> PollResult:
        raise NotImplementedError

    def fetch_logs(self, handle: RunHandle) -> str:
        raise NotImplementedError

    def fetch_results(self, handle: RunHandle) -> None:
        raise NotImplementedError

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        raise NotImplementedError

    def teardown(self, handle: RunHandle) -> None:
        self.teardowns.append(handle)


class _PendingSlurmBackend(ComputeBackend):
    """SLURM test double — launch succeeds, poll stays PENDING forever.

    Exercises the submit-and-park watchdog: the selector should
    ``scancel`` (via teardown) and fall back to RunPod once the watcher
    exceeds ``max_wait_seconds``.
    """

    def __init__(self) -> None:
        self.launches: list[RunSpec] = []
        self.teardowns: list[RunHandle] = []
        self.poll_calls: int = 0

    @property
    def name(self) -> BackendKind:
        return "cluster"

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        self.launches.append(spec)
        return RunHandle(
            backend="cluster",
            cluster=spec.cluster,
            job_id="9999",
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir=f"/scratch/tjiral/eps/issue-{spec.issue}",
            log_path=f"/scratch/tjiral/eps/issue-{spec.issue}/job.out",
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        return None

    def poll(self, handle: RunHandle) -> PollResult:
        self.poll_calls += 1
        # The selector treats anything OTHER than {running, done,
        # stalled, dead, gate} as "still pending" and keeps polling.
        return PollResult(
            status="pending",
            current_phase="pending",
            new_milestone=False,
            last_log_mtime_sec_ago=10**9,
            pid_alive=False,
            log_tail_excerpt="",
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        return ""

    def fetch_results(self, handle: RunHandle) -> None:
        return None

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        return True

    def teardown(self, handle: RunHandle) -> None:
        self.teardowns.append(handle)


class _HardFailingSlurmBackend(ComputeBackend):
    """SLURM double that fails the launch itself (e.g. submit/auth error)."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def name(self) -> BackendKind:
        return "cluster"

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        raise self._exc

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        return None

    def poll(self, handle: RunHandle) -> PollResult:
        raise NotImplementedError

    def fetch_logs(self, handle: RunHandle) -> str:
        raise NotImplementedError

    def fetch_results(self, handle: RunHandle) -> None:
        raise NotImplementedError

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        raise NotImplementedError

    def teardown(self, handle: RunHandle) -> None:
        return None


def _spec(issue: int = 137, backend: BackendKind = "runpod") -> RunSpec:
    return RunSpec(issue=issue, intent="lora-7b", backend=backend)


@pytest.fixture(autouse=True)
def captured_markers(monkeypatch):
    """Capture every ``post-marker`` call so tests can assert against them
    AND so the selector / monitor never pollute real tasks/<N>/events.jsonl.

    Autouse: every test in this module gets a clean capture list AND a
    fake poster, so a test that forgets to thread ``marker_poster=``
    still doesn't shell out to the real ``task.py``. Tests can use the
    fixture directly to assert the captured calls.

    Patches the default ``post_marker_via_task_py`` symbol; tests that
    pass an explicit ``marker_poster=`` to ``select_backend`` bypass
    this fixture (intentional — the explicit injection wins).
    """
    captured: list[dict] = []

    def fake(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        fake,
    )
    return captured


# ---------------------------------------------------------------------------
# RunPod-unchanged characterization (no `backend:` => RunPod, zero SLURM)
# ---------------------------------------------------------------------------


def test_no_backend_frontmatter_routes_to_runpod_no_slurm_path() -> None:
    """RunPod path must be reached without instantiating the SLURM backend.

    We guard against the regression where the selector eagerly built
    BOTH backends regardless of frontmatter — by patching
    ``_build_slurm_backend`` to raise on call, we prove the RunPod
    branch never touches the SLURM factory.
    """
    rp = _FakeRunPodBackend()

    from explore_persona_space.backends import selector as selector_mod

    sentinel = AssertionError("SLURM factory must not be invoked on the RunPod path")
    with mock.patch.object(selector_mod, "_build_slurm_backend", side_effect=sentinel):
        decision = select_backend(
            task={},  # no `backend:` key
            spec=_spec(),
            runpod_backend=rp,
        )

    assert isinstance(decision, BackendDecision)
    assert decision.chosen_kind == "runpod"
    assert decision.requested_kind == "runpod"
    assert decision.reason == "frontmatter_default"
    assert decision.cluster is None
    assert decision.handle is not None
    assert decision.handle.backend == "runpod"
    assert len(rp.launches) == 1
    assert rp.launches[0].issue == 137


def test_explicit_runpod_backend_routes_to_runpod() -> None:
    rp = _FakeRunPodBackend()
    decision = select_backend(
        task={"backend": "runpod"},
        spec=_spec(backend="runpod"),
        runpod_backend=rp,
    )
    assert decision.chosen_kind == "runpod"
    assert decision.reason == "frontmatter_explicit"
    assert len(rp.launches) == 1


def test_no_task_at_all_defaults_to_runpod() -> None:
    rp = _FakeRunPodBackend()
    decision = select_backend(
        task=None,
        spec=_spec(),
        runpod_backend=rp,
    )
    assert decision.chosen_kind == "runpod"
    assert len(rp.launches) == 1


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "runpod"),
        ("", "runpod"),
        ("runpod", "runpod"),
        ("RunPod", "runpod"),
        ("  runpod  ", "runpod"),
        ("cluster", "cluster"),
        ("nibi", "nibi"),
        ("fir", "fir"),
        ("NIBI", "nibi"),
    ],
)
def test_parse_backend_kind_accepts_known_values(raw: Any, expected: BackendKind) -> None:
    assert _parse_backend_kind(raw) == expected


def test_parse_backend_kind_rejects_unknown() -> None:
    with pytest.raises(BackendSelectionError, match="unknown backend"):
        _parse_backend_kind("aws")


def test_parse_backend_kind_rejects_non_string() -> None:
    with pytest.raises(BackendSelectionError, match="must be a string"):
        _parse_backend_kind(42)


def test_resolve_cluster_name_explicit_wins() -> None:
    assert _resolve_cluster_name("cluster", "fir") == "fir"
    assert _resolve_cluster_name("nibi", "fir") == "fir"


def test_resolve_cluster_name_alias_to_self() -> None:
    assert _resolve_cluster_name("nibi", None) == "nibi"
    assert _resolve_cluster_name("fir", None) == "fir"


def test_resolve_cluster_name_generic_defaults_to_nibi() -> None:
    assert _resolve_cluster_name("cluster", None) == "nibi"


def test_resolve_cluster_name_runpod_is_none() -> None:
    assert _resolve_cluster_name("runpod", None) is None


# ---------------------------------------------------------------------------
# SLURM fall-back paths
# ---------------------------------------------------------------------------


def test_slurm_stub_falls_back_to_runpod_with_reason() -> None:
    """Slice-1 stub: SLURM launch raises with sentinel => RunPod runs."""
    rp = _FakeRunPodBackend()
    stub = _SlurmStubBackend()

    decision = select_backend(
        task={"backend": "cluster"},
        spec=_spec(backend="cluster"),
        runpod_backend=rp,
        slurm_backend=stub,
    )

    assert decision.chosen_kind == "runpod"
    assert decision.requested_kind == "cluster"
    assert decision.reason == "slurm_not_implemented"
    assert decision.cluster == "nibi"  # generic cluster -> nibi by default
    assert len(rp.launches) == 1
    # The spec passed to RunPod has its backend field normalized to "runpod".
    assert rp.launches[0].backend == "runpod"
    assert rp.launches[0].cluster is None


def test_nibi_backend_falls_back_to_runpod_via_stub() -> None:
    rp = _FakeRunPodBackend()
    stub = _SlurmStubBackend()
    decision = select_backend(
        task={"backend": "nibi"},
        spec=_spec(backend="nibi"),
        runpod_backend=rp,
        slurm_backend=stub,
    )
    assert decision.chosen_kind == "runpod"
    assert decision.requested_kind == "nibi"
    assert decision.cluster == "nibi"
    assert decision.reason == "slurm_not_implemented"


def test_slurm_max_wait_exceeded_falls_back_and_scancels() -> None:
    """Submit-and-park watchdog: stuck-pending => scancel + RunPod."""
    rp = _FakeRunPodBackend()
    slurm = _PendingSlurmBackend()

    # Deterministic monotonic clock that advances 10s per call. With a
    # max_wait of 60s the watchdog will exceed the cap after a few polls.
    # The clock is consulted at: started_at + each watchdog loop iter
    # (poll + elapsed-check) + final elapsed_seconds bookkeeping.
    counter = {"t": 0.0}

    def fake_now() -> float:
        counter["t"] += 10.0
        return counter["t"]

    sleeps: list[float] = []

    decision = select_backend(
        task={"backend": "cluster"},
        spec=_spec(backend="cluster"),
        runpod_backend=rp,
        slurm_backend=slurm,
        max_wait_seconds=30,
        poll_interval_seconds=1.0,
        now_fn=fake_now,
        sleep_fn=sleeps.append,
    )

    assert decision.chosen_kind == "runpod"
    assert decision.requested_kind == "cluster"
    assert decision.reason == "slurm_max_wait_exceeded"
    assert len(slurm.launches) == 1, "SLURM should be invoked first"
    assert len(slurm.teardowns) == 1, "selector must scancel the stuck job"
    assert slurm.teardowns[0].job_id == "9999"
    assert len(rp.launches) == 1, "RunPod should run after fall-back"


def test_slurm_hard_failure_falls_back() -> None:
    """A launch exception (e.g. ssh auth fail) => RunPod fall-back."""
    rp = _FakeRunPodBackend()
    slurm = _HardFailingSlurmBackend(RuntimeError("ssh: auth refused"))

    decision = select_backend(
        task={"backend": "cluster"},
        spec=_spec(backend="cluster"),
        runpod_backend=rp,
        slurm_backend=slurm,
    )

    assert decision.chosen_kind == "runpod"
    assert decision.reason == "slurm_hard_failure"
    assert decision.extra["slurm_error_class"] == "RuntimeError"
    assert "ssh: auth refused" in decision.extra["slurm_error_msg"]
    assert len(rp.launches) == 1


def test_slurm_stub_helper_sentinel_match() -> None:
    """The fall-back guard MUST only match the exact sentinel message."""
    exc = NotImplementedError(SLURM_NOT_IMPLEMENTED_MESSAGE + ": more detail")
    assert _is_slurm_stub_unavailable(exc) is True

    other = NotImplementedError("a different unimplemented method")
    assert _is_slurm_stub_unavailable(other) is False

    wrong_type = RuntimeError(SLURM_NOT_IMPLEMENTED_MESSAGE)
    assert _is_slurm_stub_unavailable(wrong_type) is False


def test_unrelated_notimplemented_propagates_not_silently_fallback() -> None:
    """A non-stub NotImplementedError must NOT be treated as a fall-back.

    Silently falling back on every NotImplementedError would mask real
    bugs in a future SLURM backend (e.g. forgot to implement a code
    path). The sentinel match keeps the fall-back narrow — and a
    DIFFERENT NotImplementedError propagates instead of being silently
    converted into a RunPod run. This is the deliberate design choice
    from `selector._is_slurm_stub_unavailable`.
    """

    class _BuggyBackend(_HardFailingSlurmBackend):
        @property
        def name(self) -> BackendKind:
            return "cluster"

    rp = _FakeRunPodBackend()
    buggy = _BuggyBackend(NotImplementedError("some real missing feature"))

    with pytest.raises(NotImplementedError, match="some real missing feature"):
        select_backend(
            task={"backend": "cluster"},
            spec=_spec(backend="cluster"),
            runpod_backend=rp,
            slurm_backend=buggy,
        )

    # And critically: RunPod was NOT touched.
    assert len(rp.launches) == 0


# ---------------------------------------------------------------------------
# launch=False (decision-only mode for tests / dry runs)
# ---------------------------------------------------------------------------


def test_launch_false_returns_decision_without_running() -> None:
    rp = _FakeRunPodBackend()
    decision = select_backend(
        task={},
        spec=_spec(),
        runpod_backend=rp,
        launch=False,
    )
    assert decision.handle is None
    assert decision.chosen_kind == "runpod"
    assert len(rp.launches) == 0


def test_launch_false_for_cluster_does_not_invoke_slurm() -> None:
    rp = _FakeRunPodBackend()
    stub = _SlurmStubBackend()
    decision = select_backend(
        task={"backend": "cluster"},
        spec=_spec(backend="cluster"),
        runpod_backend=rp,
        slurm_backend=stub,
        launch=False,
    )
    assert decision.chosen_kind == "cluster"
    assert decision.handle is None


def test_launch_true_without_spec_raises() -> None:
    rp = _FakeRunPodBackend()
    with pytest.raises(ValueError, match="requires a RunSpec"):
        select_backend(task={}, spec=None, runpod_backend=rp, launch=True)


# ---------------------------------------------------------------------------
# Blocker 2: selector posts epm:backend-selected v1 on every launch=True
# decision (RunPod default, explicit RunPod, SLURM fall-back paths).
# ---------------------------------------------------------------------------


def _markers_of_kind(captured: list[dict], kind: str) -> list[dict]:
    return [m for m in captured if m.get("marker") == kind]


def test_selector_posts_backend_selected_on_runpod_default(captured_markers) -> None:
    rp = _FakeRunPodBackend()
    select_backend(task={}, spec=_spec(), runpod_backend=rp)

    posts = _markers_of_kind(captured_markers, "epm:backend-selected")
    assert len(posts) == 1
    assert posts[0]["issue"] == 137
    assert posts[0]["version"] == 1
    body = json.loads(posts[0]["note"])
    assert body["requested_kind"] == "runpod"
    assert body["chosen_kind"] == "runpod"
    assert body["reason"] == "frontmatter_default"


def test_selector_posts_backend_selected_on_runpod_explicit(captured_markers) -> None:
    rp = _FakeRunPodBackend()
    select_backend(task={"backend": "runpod"}, spec=_spec(), runpod_backend=rp)

    posts = _markers_of_kind(captured_markers, "epm:backend-selected")
    assert len(posts) == 1
    body = json.loads(posts[0]["note"])
    assert body["reason"] == "frontmatter_explicit"


def test_selector_posts_backend_selected_on_slurm_fallback(captured_markers) -> None:
    rp = _FakeRunPodBackend()
    stub = _SlurmStubBackend()
    select_backend(
        task={"backend": "cluster"},
        spec=_spec(backend="cluster"),
        runpod_backend=rp,
        slurm_backend=stub,
    )
    posts = _markers_of_kind(captured_markers, "epm:backend-selected")
    assert len(posts) == 1
    body = json.loads(posts[0]["note"])
    assert body["requested_kind"] == "cluster"
    assert body["chosen_kind"] == "runpod"
    assert body["reason"] == "slurm_not_implemented"
    assert body["cluster"] == "nibi"


def test_selector_posts_backend_selected_on_slurm_hard_failure(captured_markers) -> None:
    rp = _FakeRunPodBackend()
    slurm = _HardFailingSlurmBackend(RuntimeError("ssh: auth refused"))
    select_backend(
        task={"backend": "cluster"},
        spec=_spec(backend="cluster"),
        runpod_backend=rp,
        slurm_backend=slurm,
    )
    posts = _markers_of_kind(captured_markers, "epm:backend-selected")
    assert len(posts) == 1
    body = json.loads(posts[0]["note"])
    assert body["reason"] == "slurm_hard_failure"
    assert body["extra"]["slurm_error_class"] == "RuntimeError"


def test_selector_decision_only_skips_marker(captured_markers) -> None:
    """``launch=False`` is a dry run — MUST NOT touch events.jsonl."""
    rp = _FakeRunPodBackend()
    select_backend(task={}, spec=_spec(), runpod_backend=rp, launch=False)
    assert _markers_of_kind(captured_markers, "epm:backend-selected") == []


def test_selector_marker_poster_injection_overrides_default(captured_markers) -> None:
    """An explicit ``marker_poster=`` MUST win over the autouse fixture
    so a caller-supplied poster reaches every decision path."""
    rp = _FakeRunPodBackend()
    explicit: list[dict] = []
    select_backend(
        task={},
        spec=_spec(),
        runpod_backend=rp,
        marker_poster=lambda **kw: explicit.append(kw),
    )
    assert len(explicit) == 1
    assert explicit[0]["marker"] == "epm:backend-selected"
    # The autouse capture stays empty because the override won.
    assert _markers_of_kind(captured_markers, "epm:backend-selected") == []
