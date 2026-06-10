"""Tests for ``scripts/dispatch_issue.py`` (the operational `/issue` CLI).

The slice-6 router + ``backends.issue_dispatch`` helper are fully
unit-tested elsewhere; this file pins the THIN operational CLI that
SKILL.md Step 6b / 6d / 8 actually shells:

1. ``launch`` action: empty frontmatter → auto chain (mock free
   backend wins; RunPod NEVER launched; sidecar written).
2. ``launch`` action with ``--backend runpod`` → RunPod launched +
   sidecar written.
3. ``launch`` action with ``--backend cluster`` (legacy) → mapped to
   nibi.
4. ``launch`` action on a router terminal → ``failure_class:`` JSON
   line + nonzero exit code.
5. ``finalize`` action: sidecar present → confirm_artifacts PASS →
   teardown called.
6. ``finalize`` action: confirm_artifacts FAIL → teardown SKIPPED +
   nonzero exit code.
7. ``finalize`` action: missing sidecar → infra failure JSON + nonzero
   exit code (CLI never crashes the orchestrator).
8. backend_poll.py: missing sidecar → terminal infra JSON (not
   FileNotFoundError) — the BLOCKER 3 regression test.

Nothing here requires RunPod / SLURM / GCP / SSH to be live; every
external call is mocked via the ``backends_factory`` seam on the CLI.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
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
from explore_persona_space.backends.issue_dispatch import (
    default_handle_sidecar_path,
    read_handle_sidecar,
    write_handle_sidecar,
)

# ---------------------------------------------------------------------------
# Mock backend + dependency factory
# ---------------------------------------------------------------------------


class _MockBackend(ComputeBackend):
    """Records every launch / poll / teardown call for assertions."""

    def __init__(
        self,
        kind: BackendKind = "nibi",
        *,
        launch_should_raise: Exception | None = None,
        confirm_passes: bool = True,
    ) -> None:
        self._kind = kind
        self.launches: list[RunSpec] = []
        self.teardowns: list[RunHandle] = []
        self.confirms: list[RunHandle] = []
        self._launch_should_raise = launch_should_raise
        self._confirm_passes = confirm_passes

    @property
    def name(self) -> BackendKind:
        return self._kind

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        if self._launch_should_raise is not None:
            raise self._launch_should_raise
        self.launches.append(spec)
        return RunHandle(
            backend=self._kind,
            cluster=self._kind if self._kind in {"nibi", "fir"} else None,
            job_id="job-MOCK",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/scratch",
            log_path="/log",
            extra={"issue": spec.issue, "intent": spec.intent},
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
        self.confirms.append(handle)
        return self._confirm_passes

    def teardown(self, handle: RunHandle) -> None:
        self.teardowns.append(handle)


def _build_mock_factory(
    *,
    runpod: _MockBackend | None = None,
    nibi: _MockBackend | None = None,
    fir: _MockBackend | None = None,
    gcp: _MockBackend | None = None,
    mila_alive: bool = False,
) -> Any:
    """Return a backends_factory closure suitable for ``main(backends_factory=...)``."""

    def _factory() -> dict[str, Any]:
        free = {}
        if nibi is not None:
            free["nibi"] = nibi
        if fir is not None:
            free["fir"] = fir
        if mila_alive and "mila" not in free:
            # mila is rare in tests; absent default is fine.
            pass
        return {
            "runpod_backend": runpod or _MockBackend(kind="runpod"),
            "free_backends": free,
            "gcp_backend": gcp,
            "marker_poster": lambda **_kw: None,
            "is_started": lambda _b, _h: True,
            "is_live_after_cancel": lambda _b, _h: False,
            "reconnect_fn": lambda _b, _k, _s: None,
            "mila_socket_alive": lambda: mila_alive,
        }

    return _factory


# ---------------------------------------------------------------------------
# launch action
# ---------------------------------------------------------------------------


def _cd_to_tmp(monkeypatch, tmp_path):
    """Change cwd into ``tmp_path`` so the default sidecar path
    ``.claude/cache/issue-<N>-handle.json`` lands under the tmp dir
    (test isolation; never write under the real worktree's cache)."""
    monkeypatch.chdir(tmp_path)


def test_launch_empty_frontmatter_auto_routes_to_free_and_never_runpod(
    monkeypatch, tmp_path
) -> None:
    """No ``--backend`` ⇒ auto. With nibi wired, the free lane wins;
    RunPod's ``launch`` must NEVER be called.
    """
    _cd_to_tmp(monkeypatch, tmp_path)
    # RunPod backend whose launch raises — if the auto path ever reaches
    # it, the exception bubbles + the assertion below fires.
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=AssertionError("RunPod.launch must not be called on auto"),
    )
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["launch", "--issue", "300", "--intent", "lora-7b"], backends_factory=factory)
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["chosen_kind"] == "nibi"
    assert body["requested_kind"] is None  # auto
    # Sidecar landed at the default per-issue path.
    sidecar = default_handle_sidecar_path(300)
    assert sidecar.exists()
    # Round-trip: the persisted handle is the one the bg-Bash poller
    # will read tick-after-tick.
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "nibi"
    assert recovered.pod_name == "pod-300"
    # Nibi got the launch; RunPod did not.
    assert len(nibi.launches) == 1
    assert len(runpod.launches) == 0


def test_launch_backend_runpod_explicit_provisions_runpod_and_writes_sidecar(
    monkeypatch, tmp_path
) -> None:
    """``--backend runpod`` is the only path that spends real money;
    the launch path must reach RunPod AND write the sidecar uniformly
    (so Step 6d.2's bg-Bash poller has a handle to read, same as the
    SLURM/GCP paths)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "301", "--intent", "lora-7b", "--backend", "runpod"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["chosen_kind"] == "runpod"
    assert body["requested_kind"] == "runpod"
    # Sidecar was written for RunPod too — Step 8 finalize will read it.
    sidecar = default_handle_sidecar_path(301)
    assert sidecar.exists()
    recovered = read_handle_sidecar(sidecar)
    assert recovered.backend == "runpod"
    # RunPod got the launch; nibi did not.
    assert len(runpod.launches) == 1
    assert len(nibi.launches) == 0


def test_launch_sidecar_write_error_still_prints_handle_json(monkeypatch, tmp_path) -> None:
    """C1: a sidecar-write ``OSError`` after a SUCCESSFUL launch must not
    become rc=4 (the pre-fix path stranded a live job with no handle on
    stdout). The CLI prints the handle JSON line — the only recovery
    record — plus ``sidecar_write_error``, and exits 0."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import explore_persona_space.backends.issue_dispatch as idp

    def exploding_write(_handle, _path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(idp, "write_handle_sidecar", exploding_write)

    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "302", "--intent", "lora-7b", "--backend", "nibi"],
            backends_factory=factory,
        )
    assert rc == 0, "sidecar-write failure must not convert a successful launch to a crash rc"
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    # The handle JSON IS the recovery record — every field present.
    assert body["ok"] is True
    assert body["chosen_kind"] == "nibi"
    assert body["pod_name"] == "pod-302"
    assert body["job_id"] == "job-MOCK"
    assert body["handle_sidecar_path"] is None
    assert "No space left on device" in body["sidecar_write_error"]
    # The launch really happened.
    assert len(nibi.launches) == 1


def test_launch_sidecar_write_error_body_round_trips_deserialize_handle(
    monkeypatch, tmp_path
) -> None:
    """M4.1: the JSON printed on ``sidecar_write_error`` must carry the
    FULL serialized handle — ``deserialize_handle`` requires
    backend/scratch_dir/log_path beyond the summary fields, so an
    operator must be able to hand-write a ``--handle-file`` sidecar
    straight from the printed body and run finalize."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import explore_persona_space.backends.issue_dispatch as idp
    from explore_persona_space.backends.issue_dispatch import deserialize_handle

    def exploding_write(_handle, _path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(idp, "write_handle_sidecar", exploding_write)

    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "305", "--intent", "lora-7b", "--backend", "nibi"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip().splitlines()[-1])
    # The full handle dict round-trips through deserialize_handle (no
    # KeyError on a required field) and reconstructs the launch handle.
    recovered = deserialize_handle(body["handle"])
    assert recovered.backend == "nibi"
    assert recovered.job_id == "job-MOCK"
    assert recovered.pod_name == "pod-305"
    assert recovered.scratch_dir == "/scratch"
    assert recovered.log_path == "/log"
    # And a hand-written sidecar from that dict satisfies finalize: the
    # recovered handle is the same shape ``read_handle_sidecar`` yields.
    sidecar = tmp_path / "issue-305-recovered.json"
    sidecar.write_text(json.dumps(body["handle"]))
    assert read_handle_sidecar(sidecar) == recovered


def test_launch_backend_cluster_legacy_maps_to_nibi(monkeypatch, tmp_path) -> None:
    """``backend: cluster`` is the legacy selector alias; the dispatch
    helper maps it to ``nibi`` BEFORE building the spec (the slice-5
    router rejects the bare ``"cluster"`` literal)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=runpod, nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "302", "--intent", "lora-7b", "--backend", "cluster"],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["chosen_kind"] == "nibi"
    assert body["requested_kind"] == "nibi"  # router sees the normalized value
    assert len(nibi.launches) == 1


def test_launch_router_terminal_prints_failure_class_and_nonzero_exits(
    monkeypatch, tmp_path
) -> None:
    """A router terminal (``NoComputeAvailableError``) must print a
    ``failure_class``-tagged JSON line + exit nonzero so the
    orchestrator can post ``epm:failure v1`` and ``set-status blocked``."""
    _cd_to_tmp(monkeypatch, tmp_path)
    # No free backends + no GCP wired → auto chain immediately raises
    # NoComputeAvailableError (router stage 3 has nowhere to escalate).
    runpod = _MockBackend(
        kind="runpod",
        launch_should_raise=AssertionError("RunPod must not be called on auto"),
    )
    factory = _build_mock_factory(runpod=runpod, nibi=None, gcp=None)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["launch", "--issue", "303", "--intent", "lora-7b"], backends_factory=factory)
    # Exit code 2 = router terminal (per the CLI docstring).
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["status"] == "blocked"
    assert body["exception"] == "NoComputeAvailableError"
    # The note's first line carries the failure_class= prefix so the
    # orchestrator's Step 7 classifier short-circuits.
    assert body["note"].splitlines()[0] == "failure_class: infra"
    assert "no_compute_available" in body["note"]
    # Sidecar NOT written on terminal exception (the router raises
    # BEFORE the sidecar write).
    assert not default_handle_sidecar_path(303).exists()


def test_launch_hydra_args_threaded_into_spec(monkeypatch, tmp_path) -> None:
    """``--hydra k=v`` (repeatable) must land on the spec verbatim so
    the SLURM render / RunPod launch script picks them up."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--hydra",
                "condition=c1",
                "--hydra",
                "seed=42",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert nibi.launches[0].hydra_args == ("condition=c1", "seed=42")


# ---------------------------------------------------------------------------
# repo-branch default (fix19 production mirror — round-2, task #535)
# ---------------------------------------------------------------------------


def test_launch_repo_branch_defaults_to_current_branch_for_gcp_lane(monkeypatch, tmp_path) -> None:
    """Without ``--repo-branch``, a gcp/auto dispatch from a feature-branch
    checkout must thread the CURRENT branch into spec.extra — the GCE
    startup script clones from origin and would otherwise silently run
    stale main (the exact fix19 bug re-created on the production path)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-535-feature")
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            ["launch", "--issue", "304", "--intent", "lora-7b", "--backend", "gcp"],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "issue-535-feature"


def test_launch_repo_branch_explicit_flag_wins_over_current_branch(monkeypatch, tmp_path) -> None:
    """An explicit ``--repo-branch`` always wins; the current-branch
    default never overrides operator intent."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-535-feature")
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            [
                "launch",
                "--issue",
                "304",
                "--intent",
                "lora-7b",
                "--backend",
                "gcp",
                "--repo-branch",
                "release-x",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert gcp.launches[0].extra.get("repo_branch") == "release-x"


def test_launch_repo_branch_not_defaulted_on_explicit_slurm_lane(monkeypatch, tmp_path) -> None:
    """An explicit SLURM lane never escalates to GCP, so the gcp-only
    repo_branch knob is not threaded (SLURM rsyncs the local worktree)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "issue-535-feature")
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(nibi=nibi)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            ["launch", "--issue", "304", "--intent", "lora-7b", "--backend", "nibi"],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in nibi.launches[0].extra


def test_launch_repo_branch_not_defaulted_when_on_main(monkeypatch, tmp_path) -> None:
    """A main-branch checkout keeps the GCE clone default ("main") — no
    spurious extra key, no log noise."""
    _cd_to_tmp(monkeypatch, tmp_path)
    import scripts.dispatch_issue as di

    monkeypatch.setattr(di, "_current_git_branch", lambda: "main")
    gcp = _MockBackend(kind="gcp")
    factory = _build_mock_factory(gcp=gcp)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = di.main(
            ["launch", "--issue", "304", "--intent", "lora-7b", "--backend", "gcp"],
            backends_factory=factory,
        )
    assert rc == 0
    assert "repo_branch" not in gcp.launches[0].extra


# ---------------------------------------------------------------------------
# finalize action
# ---------------------------------------------------------------------------


def _seed_sidecar(tmp_path, issue: int, kind: BackendKind = "nibi") -> RunHandle:
    """Write a sidecar for ``finalize`` tests; return the handle."""
    handle = RunHandle(
        backend=kind,
        cluster=kind if kind in {"nibi", "fir"} else None,
        job_id="job-fin",
        pod_name=f"pod-{issue}",
        scratch_dir="/scratch",
        log_path="/log",
        extra={
            "issue": issue,
            "intent": "lora-7b",
            EXPECTED_ARTIFACTS_HANDLE_KEY: {
                "issue": issue,
                "sentinel_path": "/tmp/sentinel.json",
            },
        },
    )
    sidecar = tmp_path / f"issue-{issue}-handle.json"
    write_handle_sidecar(handle, sidecar)
    return handle


def test_finalize_confirm_artifacts_pass_runs_teardown(monkeypatch, tmp_path) -> None:
    """The happy path: sidecar present + confirm PASS → teardown called.
    Exit 0; JSON line carries phase=teardown."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 400, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "400",
                "--handle-file",
                str(tmp_path / "issue-400-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is True
    assert body["phase"] == "teardown"
    assert body["chosen_kind"] == "nibi"
    assert len(nibi.confirms) == 1
    assert len(nibi.teardowns) == 1


def test_finalize_confirm_artifacts_fail_skips_teardown_and_exits_nonzero(
    monkeypatch, tmp_path
) -> None:
    """A FAIL on confirm_artifacts MUST skip teardown (preserve evidence)
    + exit code 3 so the orchestrator escalates instead of silently
    losing the live backend handle."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 401, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "401",
                "--handle-file",
                str(tmp_path / "issue-401-handle.json"),
            ],
            backends_factory=factory,
        )
    # Exit 3 = confirm_artifacts FAIL (per the CLI docstring).
    assert rc == 3
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["phase"] == "confirm_artifacts"
    assert body["reason"] == "confirm_artifacts_failed"
    # confirm was called; teardown was NOT.
    assert len(nibi.confirms) == 1
    assert len(nibi.teardowns) == 0


def test_finalize_skip_confirm_artifacts_forces_teardown(monkeypatch, tmp_path) -> None:
    """``--skip-confirm-artifacts`` matches ``pod.py terminate
    --skip-upload-verify`` — escape hatch for crashes that left no
    artifacts to verify."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 402, kind="nibi")
    nibi = _MockBackend(kind="nibi", confirm_passes=False)
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "402",
                "--handle-file",
                str(tmp_path / "issue-402-handle.json"),
                "--skip-confirm-artifacts",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(nibi.confirms) == 0  # skipped
    assert len(nibi.teardowns) == 1


def test_finalize_renames_sidecar_after_successful_teardown(monkeypatch, tmp_path) -> None:
    """Mn4.3: a successful teardown retires the sidecar by renaming it
    to ``<name>.finalized`` (kept for audit) so a LATER finalize for
    the same issue cannot tear down a fresh run through the stale
    handle — the duplicate tick no-ops with the benign rc=2
    missing-sidecar shape, and the backend sees exactly ONE teardown."""
    _cd_to_tmp(monkeypatch, tmp_path)
    _seed_sidecar(tmp_path, 404, kind="nibi")
    sidecar = tmp_path / "issue-404-handle.json"
    original_payload = sidecar.read_text()
    nibi = _MockBackend(kind="nibi", confirm_passes=True)
    factory = _build_mock_factory(nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["finalize", "--issue", "404", "--handle-file", str(sidecar)],
            backends_factory=factory,
        )
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["phase"] == "teardown"
    # The sidecar was renamed, not deleted: audit copy intact.
    finalized = tmp_path / "issue-404-handle.json.finalized"
    assert not sidecar.exists()
    assert finalized.exists()
    assert finalized.read_text() == original_payload
    assert body["sidecar_finalized"] == str(finalized)

    # Second finalize for the same issue: benign rc=2 no-op, NO second
    # teardown against the retired handle.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = main(
            ["finalize", "--issue", "404", "--handle-file", str(sidecar)],
            backends_factory=factory,
        )
    assert rc2 == 2
    body2 = json.loads(buf2.getvalue().strip())
    assert body2["ok"] is False
    assert body2["reason"] == "missing_handle_sidecar"
    assert len(nibi.teardowns) == 1


def test_finalize_missing_sidecar_returns_infra_failure_not_crash(monkeypatch, tmp_path) -> None:
    """A missing sidecar must produce a clean JSON line + nonzero exit
    code (NEVER a FileNotFoundError / traceback that crashes the
    orchestrator's bg-Bash parser)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    factory = _build_mock_factory(nibi=_MockBackend(kind="nibi"))

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "finalize",
                "--issue",
                "403",
                "--handle-file",
                str(tmp_path / "issue-403-handle.json"),
            ],
            backends_factory=factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"


# ---------------------------------------------------------------------------
# backend_poll.py missing-sidecar regression (BLOCKER 3)
# ---------------------------------------------------------------------------


def test_backend_poll_missing_sidecar_emits_terminal_infra_json(tmp_path) -> None:
    """The BLOCKER 3 regression test: ``scripts/backend_poll.py`` MUST
    emit a single ``status: "dead"`` JSON line with
    ``failure_class: "infra"`` when the sidecar is missing. Previously
    it raised FileNotFoundError → empty stdout → the orchestrator's
    bg-Bash JSON-line parser had nothing to parse → loop spun forever
    on "stalled"."""
    from scripts.backend_poll import main as backend_poll_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = backend_poll_main(
            ["--issue", "500", "--handle-file", str(tmp_path / "nonexistent.json")]
        )
    # Exit 0 (the script always emits valid JSON; the failure is
    # encoded IN the JSON via failure_class / status, NOT via exit code).
    assert rc == 0
    line = buf.getvalue().strip()
    assert line, "backend_poll must emit a JSON line, never empty stdout"
    body = json.loads(line)
    # Legacy poll_pipeline shape preserved so the orchestrator's
    # existing parser handles it without a per-backend branch.
    assert body["status"] == "dead"
    assert body["pid_alive"] is False
    # Failure-classifier hint keys (the orchestrator reads these
    # alongside status: dead to post epm:failure v1 with the matching
    # failure_class).
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"


def test_backend_poll_unreadable_sidecar_also_emits_infra_json(tmp_path) -> None:
    """A corrupted JSON sidecar should hit the same failure shape — the
    orchestrator can't poll either way, so a malformed sidecar reads
    operationally as 'missing' from its perspective."""
    bad = tmp_path / "issue-501-handle.json"
    bad.write_text("{not valid json}")

    from scripts.backend_poll import main as backend_poll_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = backend_poll_main(["--issue", "501", "--handle-file", str(bad)])
    assert rc == 0
    body = json.loads(buf.getvalue().strip())
    assert body["status"] == "dead"
    assert body["failure_class"] == "infra"
    assert body["reason"] == "missing_handle_sidecar"


# ---------------------------------------------------------------------------
# Backend-for-handle resolver
# ---------------------------------------------------------------------------


def test_resolve_backend_for_handle_routes_runpod_slurm_gcp() -> None:
    """The finalize path's resolver must dispatch the right backend per
    ``handle.backend`` — silent mis-routing would terminate the WRONG
    live backend on a multi-tenant orchestrator."""
    from scripts.dispatch_issue import _resolve_backend_for_handle

    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    gcp = _MockBackend(kind="gcp")
    deps = {
        "runpod_backend": runpod,
        "free_backends": {"nibi": nibi},
        "gcp_backend": gcp,
    }
    assert _resolve_backend_for_handle(_handle_for("runpod"), deps) is runpod, (
        "runpod handle → runpod backend"
    )
    assert _resolve_backend_for_handle(_handle_for("nibi"), deps) is nibi, (
        "nibi handle → nibi backend"
    )
    assert _resolve_backend_for_handle(_handle_for("gcp"), deps) is gcp, "gcp handle → gcp backend"
    # Legacy 'cluster' kind falls back to ANY available SLURM backend.
    assert _resolve_backend_for_handle(_handle_for("cluster"), deps) is nibi


def test_resolve_backend_for_handle_rejects_unknown_kind() -> None:
    """An unknown backend kind on the handle MUST raise rather than
    silently default to RunPod (which would terminate the wrong live
    backend)."""
    from scripts.dispatch_issue import _resolve_backend_for_handle

    deps = {
        "runpod_backend": _MockBackend(kind="runpod"),
        "free_backends": {},
        "gcp_backend": None,
    }
    with pytest.raises(ValueError, match=r"unknown handle\.backend"):
        _resolve_backend_for_handle(_handle_for("totally-bogus"), deps)


def _handle_for(kind: str) -> RunHandle:
    return RunHandle(
        backend=kind,  # type: ignore[arg-type]
        cluster=kind if kind in {"nibi", "fir", "cluster"} else None,
        job_id="j",
        pod_name="p",
        scratch_dir="/s",
        log_path="/l",
        extra={},
    )


# ---------------------------------------------------------------------------
# Production-backends factory smoke test (M3: regression guard for C1)
# ---------------------------------------------------------------------------


def test_build_production_backends_wires_all_keys_and_smokes_closures(monkeypatch) -> None:
    """Call the REAL :func:`scripts.dispatch_issue._build_production_backends`
    (not the ``_build_mock_factory`` the other tests inject) and smoke
    every closure on a benign :class:`RunSpec`.

    This is the regression guard for the C1 bug fixed in router slice 6
    fix2: ``_reconnect(kind="gcp")`` previously reached
    ``gcp_backend._runner`` but :class:`backends.gcp.GcpBackend` stores
    its runner as ``self._run``. The pre-fix code path AttributeError'd
    on every explicit ``backend: gcp`` lane AND every auto-chain GCP
    escalation that hit the reconnect path. The fix is to expose the
    injection seam through a public ``runner`` property on the backend
    AND have the dispatch ``_reconnect`` closure read that property
    rather than reaching into the underscored name. The mock-factory
    tests above did NOT catch this because they injected a deps dict
    that skipped the real factory entirely — this test closes that gap.

    To stay infra-free the test patches the source modules the
    closure's lazy imports resolve against:

    * ``explore_persona_space.backends.gcp.reconnect_or_none`` —
      captures the ``config=`` / ``runner=`` kwargs the closure passed
      it; the assertion is that BOTH resolved to non-None values
      pulled off ``gcp_backend.config`` / ``gcp_backend.runner`` (the
      public property reads that would have raised pre-fix).
    * ``explore_persona_space.backends.slurm_monitor.query_by_name`` —
      short-circuits the ``ssh robot-nibi squeue ...`` call so the
      SLURM closures don't require gcloud / DRAC SSH / a robot alias
      to be live.

    Both patches target the SOURCE module symbol (NOT a re-bound name
    on ``scripts.dispatch_issue``) because the factory's closures
    lazy-import their helpers from the source modules on each
    invocation — patching only ``scripts.dispatch_issue`` would miss
    the lazy import.

    The smoke is bounded: it exercises factory wiring + closure call
    sites, not real cloud / cluster contact. Failures look like
    ``AttributeError: 'GcpBackend' object has no attribute 'runner'``
    (C1 pre-fix) or a KeyError on the deps dict (a future refactor
    drops a key) — both are exactly the regression class this test
    pins.
    """
    from explore_persona_space.backends import gcp as gcp_module
    from explore_persona_space.backends import slurm_monitor as slurm_monitor_module
    from scripts import dispatch_issue as di

    # Patch the helpers BEFORE building the factory — the closures
    # lazy-import them at factory-call time and close over the result,
    # so a post-build patch would miss the rebind.
    captured_gcp_kwargs: dict[str, Any] = {}

    def _fake_gcp_reconnect(*, spec, config, runner):  # type: ignore[no-untyped-def]
        captured_gcp_kwargs["spec"] = spec
        captured_gcp_kwargs["config"] = config
        captured_gcp_kwargs["runner"] = runner
        return None  # "no live instance" — same shape the real fn returns

    def _fake_query_by_name(*, robot_alias, job_name, timeout=30):  # type: ignore[no-untyped-def]
        return None  # "no live job"

    monkeypatch.setattr(gcp_module, "reconnect_or_none", _fake_gcp_reconnect)
    monkeypatch.setattr(slurm_monitor_module, "query_by_name", _fake_query_by_name)
    # Slice-7: the production factory wires the real
    # ``backends.slurm.mila_socket_alive`` probe (``ssh mila true``);
    # in CI we have no real Mila socket, so patch it to a deterministic
    # ``False`` so the dependency-smoke check below doesn't reach out
    # over SSH. The router-skip-Mila path is exercised by the slice-7
    # ``test_router_skips_mila_when_socket_down`` test.
    from explore_persona_space.backends import slurm as slurm_module

    monkeypatch.setattr(slurm_module, "mila_socket_alive", lambda: False)

    expected_keys = {
        "runpod_backend",
        "free_backends",
        "gcp_backend",
        "marker_poster",
        "is_started",
        "is_live_after_cancel",
        "started_evidence_probe",
        "reconnect_fn",
        "mila_socket_alive",
    }

    deps = di._build_production_backends()
    assert set(deps) == expected_keys, (
        f"factory dropped or added keys: expected {expected_keys}, got {set(deps)}"
    )

    # Sanity: the public injection-seam reads (the C1 fix) actually
    # resolve. Pre-fix `gcp_backend._runner` would AttributeError;
    # the property promotion makes `.config` / `.runner` the public
    # reads.
    gcp_backend = deps["gcp_backend"]
    assert gcp_backend.config is not None, "GcpBackend.config public property must resolve"
    assert gcp_backend.runner is not None, "GcpBackend.runner public property must resolve"

    spec = RunSpec(
        issue=999,
        intent="lora-7b",
        backend="auto",
        extra={},
    )

    reconnect_fn = deps["reconnect_fn"]

    # GCP reconnect — this is the C1 site. Pre-fix this AttributeError'd
    # on ``gcp_backend._runner``. Post-fix it routes to the patched
    # ``_fake_gcp_reconnect`` with the public ``.config`` / ``.runner``
    # property values.
    out_gcp = reconnect_fn(deps["gcp_backend"], "gcp", spec)
    assert out_gcp is None, "patched _fake_gcp_reconnect returns None"
    assert captured_gcp_kwargs["config"] is gcp_backend.config, (
        "GCP reconnect must pass the backend's public ``config`` property — "
        "pre-fix this read raised AttributeError on the underscored name."
    )
    assert captured_gcp_kwargs["runner"] is gcp_backend.runner, (
        "GCP reconnect must pass the backend's public ``runner`` property — "
        "pre-fix the code path read ``gcp_backend._runner`` which doesn't "
        "exist (GcpBackend stores the runner as ``self._run``)."
    )

    # SLURM reconnect — patched query_by_name returns None so the
    # closure exits the no-live-job branch cleanly. The smoke validates
    # the M2 fix (public ``scratch_dir_for`` import) compiles + executes.
    nibi_backend = deps["free_backends"].get("nibi")
    assert nibi_backend is not None, "production factory must wire nibi"
    out_nibi = reconnect_fn(nibi_backend, "nibi", spec)
    assert out_nibi is None, "patched query_by_name returns None → reconnect returns None"

    # RunPod / unknown kinds: per the closure's docstring, both return
    # None (RunPod's existing pod_lifecycle.py is idempotent on its own).
    assert reconnect_fn(deps["runpod_backend"], "runpod", spec) is None
    assert reconnect_fn(deps["runpod_backend"], "wibble", spec) is None

    # is_started / is_live_after_cancel — for handles with
    # ``cluster is None`` (RunPod / GCP), they fall through to a poll
    # on the backend. Smoke that path with a stub backend; the closure
    # itself is the unit under test, not the backend.
    poll_calls: dict[str, int] = {"is_started": 0, "is_live_after_cancel": 0}

    class _StubPollBackend:
        def poll(self, _handle):  # type: ignore[no-untyped-def]
            from explore_persona_space.backends.base import PollResult

            poll_calls["is_started"] += 1
            return PollResult(
                status="running",
                current_phase="x",
                new_milestone=False,
                last_log_mtime_sec_ago=1,
                pid_alive=True,
                log_tail_excerpt="",
            )

    handle_gcp_like = _handle_for("gcp")  # cluster=None
    assert deps["is_started"](_StubPollBackend(), handle_gcp_like) is True
    # PollResult.status=="running" → is_live_after_cancel returns True
    # (the closure's "still-live" check rejects only {"done", "dead"}).
    assert deps["is_live_after_cancel"](_StubPollBackend(), handle_gcp_like) is True

    # started_evidence_probe — non-SLURM handles (cluster=None) return
    # None WITHOUT any SSH/rsync (the probe is SLURM-scratch-specific).
    assert deps["started_evidence_probe"](_StubPollBackend(), handle_gcp_like) is None

    # marker_poster + mila_socket_alive — exist and are callable, no
    # network needed to smoke.
    assert callable(deps["marker_poster"])
    # Slice-7: factory wires the REAL ``backends.slurm.mila_socket_alive``
    # probe. We monkeypatched it above to a deterministic ``False`` so
    # the smoke does not reach out over SSH; the wiring is what matters
    # here (router-skip-Mila behaviour is covered by the dedicated
    # ``test_router_skips_mila_when_socket_down`` test).
    assert deps["mila_socket_alive"]() is False
