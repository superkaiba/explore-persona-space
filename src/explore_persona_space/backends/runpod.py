"""RunPod backend adapter.

Thin :class:`~base.ComputeBackend` wrapper around the existing
``scripts/pod_lifecycle.py`` + ``scripts/poll_pipeline.py`` flow. The
foundation slice's contract was **zero behavior change** when a task has
no ``backend:`` frontmatter; the slice-6 unification keeps that property
but routes every call (RunPod included) through ``backend.<method>(handle)``
so the dispatch / poll / teardown surface is shared with SLURM + GCP.

What this module ships (post slice 6):

* ``name`` = ``"runpod"`` (matches the selector default).
* ``prepare`` — currently a no-op (provision triggers bootstrap inline).
* ``launch`` — delegates to ``scripts/pod_lifecycle.py provision`` via
  the existing subprocess entrypoint and returns a :class:`RunHandle`
  built from the resulting ``pods_ephemeral.json`` row.
* ``estimate_start`` — returns "now" (UTC); RunPod pods come up within
  a few minutes, so a precise estimate would be noise.
* ``poll`` — delegates to :func:`scripts.poll_pipeline.poll_once` so
  the bg-Bash poll loop the orchestrator already runs (Step 6d.2) keeps
  the SAME JSON-line shape, and a foreground caller (the unified
  ``scripts/backend_poll.py`` helper) can also dispatch through the
  backend without re-implementing the probe logic.
* ``fetch_logs`` — pulls the last ~200 lines of
  ``/workspace/logs/issue-<N>.log`` via SSH for orchestrator progress
  notes.
* ``fetch_results`` — drives ``scripts/pod.py sync results --all``
  (which calls ``scripts/pull_results.py``) so ``eval_results/`` +
  ``figures/`` are pulled back to the VM. Mirrors the SLURM
  ``rsync_pull`` path so Step 8 can run uniformly across backends.
* ``confirm_artifacts`` — delegates to
  :func:`backends.artifacts.confirm_artifacts_from_handle` (the
  mechanical gate SLURM + GCP also use); the upload-verifier agent
  stays the canonical exploratory pass.
* ``teardown`` — delegates to ``scripts/pod_lifecycle.py terminate``.

The slice-6 wiring keeps the JSON-line poll contract verbatim: the
PollResult fields ``poll`` returns match
``scripts/poll_pipeline.PollResult`` byte-for-byte, so the
orchestrator's existing JSON parser keeps working unchanged across
backends.

Implementation note: ``poll`` / ``fetch_logs`` / ``fetch_results``
require the handle to carry the production fields the orchestrator
populates at launch time (``pid_file`` in ``extra`` for ``poll``;
``log_path`` for ``fetch_logs``; ``extra["issue"]`` for
``fetch_results``). The ``launch`` path stuffs all of these onto the
handle so a caller never has to re-derive them.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
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

logger = logging.getLogger(__name__)

#: How many lines of the remote log :meth:`RunPodBackend.fetch_logs`
#: pulls. Matches the magnitude orchestrator progress notes need
#: (Step 6d.2's bg-Bash poller emits a ~5-line tail in the JSON-line
#: output; a one-shot foreground tail gets a bit more headroom).
LOG_TAIL_LINES = 200


def _shell_quote(s: str) -> str:
    """Single-quote ``s`` for a remote bash command (poor-man's shlex.quote).

    Sufficient for fixed log paths (``/workspace/logs/issue-<N>.log``)
    that the launch path controls; we accept the small risk that a
    handle with a manually-edited path containing a single quote would
    mis-tail rather than pulling in ``shlex`` for one call.
    """
    return "'" + s.replace("'", "'\\''") + "'"


def _runpod_pid_file_path(issue: int) -> str:
    """Canonical RunPod-side pid file path the experimenter launcher writes.

    Mirrors the ``epm:run-launched`` ``pid_file=`` convention in
    ``.claude/agents/experimenter.md`` § "During Execution". The
    launcher's ``echo $$ > /workspace/logs/issue-<N>.pid`` writes here
    so ``poll_pipeline.poll_once`` can probe it as the liveness signal.
    """
    return f"/workspace/logs/issue-{issue}.pid"


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


def mint_runpod_attempt_id() -> str:
    """Launch-scoped attempt id, GCP-style (minted pre-provision; #598).

    RunPod has no scheduler job id, so launch mints
    ``rp-<UTCstamp>-<4hex>``. The id namespaces the completion
    sentinel: a prior attempt's sentinel can never satisfy this
    launch's declaration (``_check_sentinel`` validates phase+issue
    only, so the PATH is the staleness defense — same reasoning as the
    SLURM ``slurm-<jobid>`` namespacing). Attempt-binding is REQUIRED
    here: ``/workspace`` is the persistent volume, nothing clears it
    across same-pod relaunches, and same-pod retries are the routine
    ``/issue`` recovery path — a flat per-issue path would let attempt
    N-1's sentinel turn a crashed retry into a green finalize +
    teardown on unuploaded state.
    """
    import secrets

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"rp-{stamp}-{secrets.token_hex(2)}"


def runpod_sentinel_path(issue: int, attempt_id: str) -> str:
    """Pod-side completion-sentinel path, attempt-namespaced (#598).

    Under ``/workspace`` (the persistent volume), NOT under the repo
    clone, so the path is stable regardless of where the workload
    checked out the repo. Attempt-namespaced because ``/workspace``
    survives same-pod relaunches and no hygiene step clears a flat
    sentinel (see :func:`mint_runpod_attempt_id`). The workload-side
    writer convention lives in ``.claude/agents/experimenter.md`` —
    the path is read from the launch sidecar's
    ``extra.expected_artifacts.sentinel_path``, the write is chained
    on the workload's exit status, and stale sentinels are cleared
    pre-(re)launch.
    """
    from explore_persona_space.backends.artifacts import SENTINEL_FILENAME

    return f"/workspace/eval_results/issue_{issue}/{attempt_id}/{SENTINEL_FILENAME}"


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
        # Expected-artifacts declaration (#598): attempt id minted at
        # launch (GCP-style) and embedded in the pod-side sentinel path
        # so a prior attempt's sentinel on the persistent /workspace
        # volume can never satisfy this launch's declaration. ALL RunPod
        # workloads are experimenter-driven custom dispatches, so the
        # declaration carries NO launch-time HF prefix guess (the #601
        # false-negative-teardown trap, a fortiori on this lane).
        from explore_persona_space.backends.artifacts import (
            EXPECTED_ARTIFACTS_HANDLE_KEY,
            build_expected_artifacts_declaration,
        )

        attempt_id = mint_runpod_attempt_id()
        # ``extra`` carries the production fields the orchestrator + the
        # unified ``poll`` / ``fetch_results`` paths need without having
        # to re-derive them from the issue id:
        # * ``issue`` — round-tripped so ``confirm_artifacts`` /
        #   ``fetch_results`` / cross-backend reconnect can index by it.
        # * ``intent`` — preserved for marker bodies + downstream
        #   re-provision intent re-use.
        # * ``pid_file`` — absolute path the experimenter launcher
        #   writes; ``poll`` forwards it to
        #   ``poll_pipeline.poll_once(pid_file=...)``.
        # * ``runpod_attempt_id`` — plain field so the orchestrator /
        #   experimenter can read the attempt id without parsing the
        #   declaration.
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
            extra={
                "intent": spec.intent,
                "issue": int(spec.issue),
                "pid_file": _runpod_pid_file_path(spec.issue),
                "runpod_attempt_id": attempt_id,
                EXPECTED_ARTIFACTS_HANDLE_KEY: build_expected_artifacts_declaration(
                    issue=spec.issue,
                    sentinel_path=runpod_sentinel_path(spec.issue, attempt_id),
                    custom_workload=True,
                    attempt_id=attempt_id,
                    wandb_run_path=spec.extra.get("wandb_run_path"),
                ),
            },
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        """RunPod pods come up in minutes — informational "now"."""
        del spec  # parameter is part of the ABC contract; unused here.
        return datetime.now(tz=UTC)

    # ----- monitor ---------------------------------------------------------

    def poll(self, handle: RunHandle) -> PollResult:
        """One-tick poll via :func:`scripts.poll_pipeline.poll_once`.

        Delegates to the existing battle-tested poll path (see
        ``scripts/poll_pipeline.py`` module docstring + the
        ``#260 / #405 / #468 / #488`` incidents that hardened it). The
        returned :class:`PollResult` shape matches
        ``poll_pipeline.PollResult`` byte-for-byte, so the orchestrator's
        bg-Bash JSON-line parser is interchangeable across backends.

        Reads the pid-file path from ``handle.extra['pid_file']`` (the
        ``launch`` path populates it); falls back to the canonical
        ``/workspace/logs/issue-<N>.pid`` if absent (defensive — a
        handle round-tripped from an older serializer might not carry
        the field).

        Lazy-imports the poller so this module stays importable in a
        context that does not have ``scripts/`` on ``sys.path``.
        """
        # Lazy import — the poller module pulls in subprocess + ssh
        # helpers that are pointless when the caller only needs
        # ``launch`` / ``teardown``. ``scripts`` is a package with an
        # ``__init__.py`` so the import works under the project's
        # canonical ``uv run`` sys.path.
        from scripts.poll_pipeline import DEFAULT_STATE_DIR, poll_once

        issue = self._issue_from_handle(handle)
        pid_file = handle.extra.get("pid_file") or _runpod_pid_file_path(issue)
        # The poll-pipeline state file mirrors the orchestrator's
        # default (``.claude/cache/poll-pipeline-<N>.json``) so a poll
        # call from inside this backend shares its phase-cache with the
        # orchestrator's bg-Bash loop (avoids spurious ``new_milestone``
        # double-posts on the next tick).
        state_file = DEFAULT_STATE_DIR / f"poll-pipeline-{issue}.json"
        raw = poll_once(
            issue=issue,
            pod=handle.pod_name,
            log_path=handle.log_path,
            pid_file=pid_file,
            state_file=state_file,
        )
        # ``poll_once`` returns ``scripts.poll_pipeline.PollResult`` whose
        # fields match ``backends.base.PollResult`` byte-for-byte; we
        # rebuild as the backend-typed class so cross-backend callers see
        # ONE PollResult class (otherwise an ``isinstance(...,
        # backends.base.PollResult)`` check would fail on the RunPod
        # return). The field set is held in sync by the docstring
        # contract in ``base.py``.
        return PollResult(
            status=raw.status,
            current_phase=raw.current_phase,
            new_milestone=raw.new_milestone,
            last_log_mtime_sec_ago=raw.last_log_mtime_sec_ago,
            pid_alive=raw.pid_alive,
            log_tail_excerpt=raw.log_tail_excerpt,
            gate=raw.gate,
            sentinels_processed=raw.sentinels_processed,
            phase_log_mtime_sec_ago=raw.phase_log_mtime_sec_ago,
            shard_log_mtime_sec_ago=raw.shard_log_mtime_sec_ago,
            gpu_util=raw.gpu_util,
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        """One-shot tail of the remote log via SSH.

        Pulls the last ``LOG_TAIL_LINES`` lines of
        ``handle.log_path`` for the orchestrator's progress notes /
        failure-classifier excerpts. Best-effort — a missing log file or
        a flaky SSH returns ``""`` (never raises) so a caller using this
        for a progress note isn't crashed by a transient SSH blip. A
        load-bearing fetch (e.g. confirming a crash) should still go
        through ``ssh_execute`` directly with its own error handling.
        """
        argv = [
            "ssh",
            handle.pod_name,
            f"tail -{LOG_TAIL_LINES} {_shell_quote(handle.log_path)} 2>/dev/null || true",
        ]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(
                "RunPodBackend.fetch_logs: ssh tail failed for %s (%s: %s); returning empty.",
                handle.pod_name,
                type(exc).__name__,
                exc,
            )
            return ""
        if proc.returncode != 0:
            logger.warning(
                "RunPodBackend.fetch_logs: ssh tail returned %d for %s; returning empty.",
                proc.returncode,
                handle.pod_name,
            )
            return ""
        return proc.stdout or ""

    # ----- teardown --------------------------------------------------------

    def fetch_results(self, handle: RunHandle) -> None:
        """Pull eval_results/ + figures/ back to the VM.

        Drives ``scripts/pod.py sync results --all`` (which calls
        ``scripts/pull_results.py`` for the WandB-side pull). The
        existing path Step 8 invokes today — keeping the implementation
        as a wrapped subprocess preserves all its battle-tested
        behaviour (incident-hardened pull order, partial-resume
        semantics) rather than re-implementing it in Python.

        Non-fatal on failure: the call logs + returns. A guaranteed
        rsync would block teardown of a healthy run that uploaded
        everything during training (the authoritative path); a failed
        ``sync results`` is a missing local mirror, not missing
        artifacts.
        """
        issue = self._issue_from_handle(handle)
        cmd = [
            sys.executable,
            str(_scripts_dir() / "pod.py"),
            "sync",
            "results",
            "--all",
        ]
        logger.info(
            "RunPodBackend.fetch_results: invoking pod.py sync results --all for issue=%d",
            issue,
        )
        try:
            subprocess.run(cmd, check=False, timeout=600, env=os.environ.copy())
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "RunPodBackend.fetch_results: timed out (%s); continuing without local mirror.",
                exc,
            )

    def _ssh_read_sentinel(self, handle: RunHandle) -> Callable[[str], str | None]:
        """Build a remote sentinel reader bound to ``handle``'s pod (#598).

        The verifier's default ``read_sentinel`` is a local-FS read; the
        RunPod sentinel lives on the pod (``/workspace/eval_results/
        issue_<N>/<attempt>/...``). The pod is guaranteed alive at
        confirm time (teardown is gated on the PASS), so a remote read
        is reliable. Semantics:

        * rc=0 → return stdout (the sentinel content).
        * non-zero with "no such file" in stderr → ``None`` (the
          verifier reads this as FAIL "sentinel missing at <path>").
        * any other non-zero (transport / auth / DNS) → raise. A
          transport failure must NOT read as "missing" — the raise
          surfaces through ``_check_sentinel``'s catch as FAIL with the
          REAL reason (fail-loud per the artifacts.py contract).
        """

        def read(path: str) -> str | None:
            proc = subprocess.run(
                ["ssh", handle.pod_name, f"cat {_shell_quote(path)}"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout
            stderr = (proc.stderr or "").lower()
            if "no such file" in stderr:
                return None
            raise RuntimeError(
                f"ssh sentinel read from {handle.pod_name} failed "
                f"rc={proc.returncode}: {(proc.stderr or '')[:300]}"
            )

        return read

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        """Backend-agnostic artifact verification.

        Delegates to :func:`backends.artifacts.confirm_artifacts_from_handle`
        — the same mechanical gate SLURM (and slice-3 GCP) use. The
        ``upload-verifier`` agent still drives the exploratory pass
        (SSHing the pod for unuploaded files); this gate is the
        complementary mechanical check that won't be soft-passed by an
        optimistic agent run.

        Reads the :class:`~backends.artifacts.ExpectedArtifacts`
        declaration the launch path stuffed onto ``handle.extra`` under
        :data:`~backends.artifacts.EXPECTED_ARTIFACTS_HANDLE_KEY`. A
        missing declaration is itself a FAIL (the launch path is
        responsible for populating it; silently passing a handle that
        forgot is the silent-loss hole the verifier closes). The
        sentinel check reads the pod-side file over SSH via
        :meth:`_ssh_read_sentinel` (#598); HF / WandB / git checks keep
        their default wires.
        """
        # Lazy import to keep the runpod module importable without the
        # artifacts module's optional deps loaded yet.
        from explore_persona_space.backends.artifacts import (
            VerifierIO,
            confirm_artifacts_from_handle,
        )

        verdict = confirm_artifacts_from_handle(
            handle, io=VerifierIO(read_sentinel=self._ssh_read_sentinel(handle))
        )
        if not verdict.passed:
            # Use print rather than a module logger here so the failure
            # surfaces in the bg-Bash captured output the orchestrator
            # already reads (the runpod path otherwise has no logger
            # wired up); keep the line stable so /issue Step 8 marker-
            # extraction can grep for it on resume.
            print(
                f"[RunPodBackend.confirm_artifacts] FAIL for handle={handle.pod_name}: "
                f"{'; '.join(verdict.reasons)}",
                file=sys.stderr,
            )
        return verdict.passed

    def _issue_from_handle(self, handle: RunHandle) -> int:
        """Recover the issue number from a handle (``extra`` first, then pod name).

        Prefers ``handle.extra['issue']`` (the canonical field
        ``launch`` populates); falls back to parsing the pod name
        (canonical ``pod-<N>`` or legacy ``epm-issue-<N>``) so a handle
        round-tripped from an older serializer (pre-slice-6) still
        works. Raises ``ValueError`` on a handle we cannot index — a
        silent default would mis-route ``fetch_results`` / ``poll`` to
        the wrong issue.
        """
        from_extra = handle.extra.get("issue")
        if from_extra is not None:
            return int(from_extra)
        name = handle.pod_name
        for prefix in ("pod-", "epm-issue-"):
            if name.startswith(prefix):
                try:
                    return int(name[len(prefix) :])
                except ValueError:
                    continue
        raise ValueError(
            f"RunPodBackend: cannot recover issue from handle "
            f"(extra={handle.extra!r}, pod_name={handle.pod_name!r})"
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
