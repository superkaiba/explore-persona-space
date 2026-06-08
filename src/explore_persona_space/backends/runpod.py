"""RunPod backend adapter.

Thin :class:`~base.ComputeBackend` wrapper around the existing
``scripts/pod_lifecycle.py`` + ``scripts/poll_pipeline.py`` flow. The
contract for slice 1: **zero behavior change** when a task has no
``backend:`` frontmatter — every call delegates to a script the
orchestrator already invokes today.

What this slice ships:

* ``name`` = ``"runpod"`` (matches the selector default).
* ``prepare`` — currently a no-op (provision triggers bootstrap inline).
* ``launch`` — delegates to ``scripts/pod_lifecycle.py provision`` via
  the existing subprocess entrypoint and returns a :class:`RunHandle`
  built from the resulting ``pods_ephemeral.json`` row.
* ``estimate_start`` — returns "now" (UTC); RunPod pods come up within
  a few minutes, so a precise estimate would be noise.
* ``poll`` — delegates to ``scripts/poll_pipeline.py`` (orchestrator
  re-uses this in its bg-Bash loop; the adapter is offered so SLURM and
  RunPod present a uniform interface to future call sites).
* ``fetch_logs`` — reads the existing ``/workspace/logs/issue-<N>.log``
  via SSH (one-shot tail, mirrors the orchestrator's pull pattern).
* ``fetch_results`` — pulls ``eval_results/`` + ``figures/`` back via
  the existing ``scripts/pod.py sync results`` flow.
* ``confirm_artifacts`` — delegates to ``scripts/verify_uploads.py``.
* ``teardown`` — delegates to ``scripts/pod_lifecycle.py terminate``.

What slice 1 explicitly does NOT do: re-implement any of the orchestrator-
side polling / sentinel-drain / rsync code in Python. Those scripts are
already battle-tested (see #260, #405, #468, #488 incidents) and any
re-implementation would silently lose those guards. The adapter is a
thin shim, not a re-write.

Several methods raise ``NotImplementedError`` with a clear message when
the existing CLI entrypoint is what the orchestrator actually invokes
today (e.g. ``fetch_results`` is done via ``pod.py sync results`` from
the orchestrator's bg-Bash loop, not the experimenter agent's foreground
context). The selector's RunPod path does NOT call those methods today;
they exist as the seat for a future cleanup that pulls all RunPod
invocations through the backend.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)

if TYPE_CHECKING:
    pass


# Repository root resolved relative to this file (src/explore_persona_space/
# backends/runpod.py -> ../../../). Used to locate scripts/ for subprocess
# delegation. Falls back to cwd if the layout has been mangled.
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (*here.parents,):
        if (parent / "scripts" / "pod_lifecycle.py").exists():
            return parent
    return Path.cwd()


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _runpod_log_path(issue: int) -> str:
    """Canonical RunPod-side log path for a `/issue` run.

    Matches the convention ``scripts/poll_pipeline.py`` parses (see its
    module docstring; ``/workspace/logs/issue-<N>.log``).
    """
    return f"/workspace/logs/issue-{issue}.log"


def _runpod_pod_name(issue: int) -> str:
    """Canonical pod name (April 2026 rename: ``pod-<N>``).

    Mirrors ``scripts/pod_lifecycle.py::_canonical_pod_name``. The legacy
    ``epm-issue-<N>`` prefix is recognized by readers but never used for
    fresh provisions.
    """
    return f"pod-{issue}"


class RunPodBackend(ComputeBackend):
    """Backend adapter over the existing RunPod tooling.

    Methods that the orchestrator already drives directly (poll, fetch,
    terminate via bg-Bash) are provided here as thin shims so future
    call sites can dispatch polymorphically. The slice-1 selector exercises
    only ``name`` + ``launch`` + ``teardown`` — those routes are fully
    wired; the others are seats for future refactoring.
    """

    @property
    def name(self) -> BackendKind:
        return "runpod"

    # ----- launch ----------------------------------------------------------

    def prepare(self, spec: RunSpec) -> None:
        """RunPod provisioning is one shot — no separate prepare step.

        Pod creation, SSH wait, and ``bootstrap_pod.sh`` all happen inside
        ``scripts/pod_lifecycle.py provision``. There is no useful action
        for ``prepare`` to take on the RunPod path, so this is a no-op.
        """
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        """Provision a pod for ``spec.issue``; return a :class:`RunHandle`.

        Delegates to ``scripts/pod_lifecycle.py provision`` (same path
        ``pod.py provision`` invokes). Honors ``--intent`` from the spec;
        an explicit ``spec.gpus`` would map to ``--gpu-count`` but the
        slice-1 selector does not set that field (the intent default
        suffices for every RunPod workload today).
        """
        cmd = [
            sys.executable,
            str(_scripts_dir() / "pod_lifecycle.py"),
            "provision",
            "--issue",
            str(spec.issue),
            "--intent",
            spec.intent,
        ]
        if spec.gpus is not None:
            cmd += ["--gpu-count", str(spec.gpus)]
        # subprocess.run raises CalledProcessError on non-zero exit; that
        # propagates to the selector, which logs + lets the orchestrator
        # surface the failure as `epm:failure` (slice 1 does NOT add a
        # provision retry — the existing `--wait-for-capacity` retry inside
        # `pod_lifecycle.py` already handles SUPPLY_CONSTRAINT).
        subprocess.run(cmd, check=True)
        pod_name = _runpod_pod_name(spec.issue)
        return RunHandle(
            backend="runpod",
            cluster=None,
            # The RunPod pod_id is set inside pod_lifecycle.py and persisted
            # to pods_ephemeral.json; we read it back from there rather than
            # parsing stdout. For slice 1 the orchestrator does not need the
            # raw pod_id (it routes by name through SSH config) — empty
            # string is the truthful "we did not capture this here" marker;
            # a future revision should round-trip pods_ephemeral.json.
            job_id="",
            pod_name=pod_name,
            scratch_dir="/workspace",
            log_path=_runpod_log_path(spec.issue),
            extra={"intent": spec.intent},
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        """RunPod pods come up in minutes — informational "now"."""
        del spec  # parameter is part of the ABC contract; unused here.
        return datetime.now(tz=UTC)

    # ----- monitor ---------------------------------------------------------

    def poll(self, handle: RunHandle) -> PollResult:
        """One-tick poll via ``scripts/poll_pipeline.py``.

        Slice 1 does not call this from the selector — the orchestrator's
        bg-Bash loop drives ``poll_pipeline.py`` directly. The adapter is
        provided so future call sites can dispatch polymorphically across
        backends. Raises :class:`NotImplementedError` to make the seat
        explicit; remove the raise when a caller actually needs it.
        """
        del handle
        raise NotImplementedError(
            "RunPodBackend.poll: the `/issue` orchestrator invokes "
            "scripts/poll_pipeline.py directly via bg-Bash today. Wire "
            "this method when a foreground caller (e.g. future PM-session "
            "dashboard) needs a single in-process poll."
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        """Tail the remote log via SSH.

        Slice 1: not wired into the selector. The orchestrator's bg-Bash
        loop tails logs directly inside ``poll_pipeline.py`` (the JSON-line
        output carries ``log_tail_excerpt``). Raise so a future
        foreground caller fills this in deliberately rather than getting
        a silent empty string.
        """
        del handle
        raise NotImplementedError(
            "RunPodBackend.fetch_logs: orchestrator tails inside "
            "poll_pipeline.py today. Wire when a foreground caller needs "
            "a one-shot tail."
        )

    # ----- teardown --------------------------------------------------------

    def fetch_results(self, handle: RunHandle) -> None:
        """Pull eval_results/ + figures/ back to the VM.

        Slice 1: delegated to the orchestrator's existing
        ``scripts/pod.py sync results`` flow (driven from /issue Step 8).
        The adapter raises so we don't silently double-rsync.
        """
        del handle
        raise NotImplementedError(
            "RunPodBackend.fetch_results: orchestrator already runs "
            "`scripts/pod.py sync results --all` in /issue Step 8. Wire "
            "this when the backend abstraction subsumes Step 8."
        )

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        """Run the upload-verifier check.

        Slice 1: orchestrator dispatches ``upload-verifier`` as an Agent
        with the typed handle; this method is the seat where the agent's
        subprocess call could land if/when we collapse it into Python.
        """
        del handle
        raise NotImplementedError(
            "RunPodBackend.confirm_artifacts: orchestrator dispatches "
            "the `upload-verifier` agent today (it shells out to "
            "scripts/verify_uploads.py + Hub list_repo_files). Wire when "
            "the agent's checks are folded into a Python helper."
        )

    def teardown(self, handle: RunHandle) -> None:
        """Terminate the pod (volume gone).

        Delegates to ``scripts/pod_lifecycle.py terminate``. Idempotent
        on the RunPod side — ``cmd_terminate`` no-ops when no live pod
        matches the issue. The ``--skip-upload-verify`` guard is NOT
        passed: the orchestrator runs ``confirm_artifacts`` before
        teardown, so the verifier guard inside ``cmd_terminate`` should
        always see a PASS marker.
        """
        # The pod name carries the issue; parse it back (canonical
        # ``pod-<N>``) so we don't need extra state on the handle.
        issue: int | None = None
        if handle.pod_name.startswith("pod-"):
            try:
                issue = int(handle.pod_name[len("pod-") :])
            except ValueError:
                issue = None
        if issue is None and handle.pod_name.startswith("epm-issue-"):
            try:
                issue = int(handle.pod_name[len("epm-issue-") :])
            except ValueError:
                issue = None
        if issue is None:
            raise ValueError(f"cannot parse issue from RunPod handle pod_name={handle.pod_name!r}")
        cmd = [
            sys.executable,
            str(_scripts_dir() / "pod_lifecycle.py"),
            "terminate",
            "--issue",
            str(issue),
            "--yes",
        ]
        # Inherit current env so RUNPOD_API_KEY etc. propagate. ``check=True``
        # lets the selector see a non-zero terminate exit (e.g. survivors
        # detected by the post-terminate live-API re-query inside
        # ``cmd_terminate``).
        subprocess.run(cmd, check=True, env=os.environ.copy())
