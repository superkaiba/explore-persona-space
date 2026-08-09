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
  built from the resulting ``pods_ephemeral.json`` row. On the
  backend-executed leg (#909) the rendered launcher chains the
  success-gated completion-sentinel write and (#977) waits on fresh
  detached ``/workspace/logs/*.pid`` workloads before the sentinel
  write (GCP #601 parity).
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

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
    validate_env_pins,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

#: How many lines of the remote log :meth:`RunPodBackend.fetch_logs`
#: pulls. Matches the magnitude orchestrator progress notes need
#: (Step 6d.2's bg-Bash poller emits a ~5-line tail in the JSON-line
#: output; a one-shot foreground tail gets a bit more headroom).
LOG_TAIL_LINES = 200

#: Floor for the #1010 CPU-fallback container-disk threading in
#: :meth:`RunPodBackend.launch` — mirrors ``runpod_api.DEFAULT_CONTAINER_DISK_GB``
#: (50), the value an un-threaded provision gets, so threading a plan's
#: ``boot_disk_gb`` can only ever GROW the container disk relative to
#: today's default, never shrink it. (Not imported from ``scripts/runpod_api``
#: — this module's imports stay ``base``-only by documented convention.)
_CPU_CONTAINER_DISK_FLOOR_GB = 50

#: Floor for the #1118 GPU-lane volume threading in :meth:`RunPodBackend.launch`
#: — mirrors pod_lifecycle.py's ``provision --volume-gb`` argparse default (200),
#: the /workspace volume an un-threaded GPU provision gets, so threading a
#: plan's ``boot_disk_gb`` can only ever GROW the volume relative to today's
#: default, never shrink it. (Not imported from ``scripts/pod_lifecycle`` —
#: this module's imports stay ``base``-only by documented convention.)
_GPU_VOLUME_FLOOR_GB = 200


# Bounded stderr tail carried on a failed pod_lifecycle subprocess (#1465).
# 60 lines covers the 20-50-line traceback class (#775 B2: a 5-line tail
# truncated vLLM tracebacks; the GCE EXIT trap tails 40); 400 chars/line
# bounds a pathological single line, worst case ~24 KB << the 50k marker cap.
_POD_LIFECYCLE_TAIL_MAX_LINES = 60
_POD_LIFECYCLE_TAIL_MAX_LINE_CHARS = 400


#: pod_lifecycle.py's structured still-waiting exit (EX_TEMPFAIL, #603): the
#: bounded wait-for-capacity loop reached its per-process wall-clock budget
#: with NOTHING provisioned and NOTHING billing — the caller RE-RUNS the same
#: command to continue waiting (the wait loop is state-free). Mirrored (not
#: imported) from ``scripts/pod_lifecycle.py::EXIT_STILL_WAITING`` — this
#: module's imports stay ``base``-only by documented convention (see the
#: _GPU_VOLUME_FLOOR_GB note above). Parity pinned by
#: tests/test_dispatch_issue_cli.py::test_exit_still_waiting_matches_pod_lifecycle.
EXIT_STILL_WAITING = 75

#: pod_lifecycle.py's stopped-pod same-name collision refusal (#1997): a
#: same-named STOPPED (EXITED) pod exists and the provision REFUSED to mint a
#: duplicate-named pod (whose name-keyed pods.conf / pods_ephemeral.json rows
#: would hijack the stopped pod's — the #1739 4-duplicate incident). NOTHING
#: was provisioned, NOTHING bills; recovery is a HUMAN action (resume /
#: approved terminate / --name-suffix / --allow-stopped-duplicate), so the
#: router terminal rung raises the typed, NON-watcher-re-drivable
#: ``RunPodStoppedPodCollisionError`` on this code instead of the re-drivable
#: ``no_compute_available`` terminal. Mirrored (not imported) from
#: ``scripts/pod_lifecycle.py::EXIT_STOPPED_POD_COLLISION`` — this module's
#: imports stay ``base``-only by documented convention (see EXIT_STILL_WAITING
#: above). Parity pinned by
#: tests/test_dispatch_issue_cli.py::test_exit_stopped_pod_collision_matches_pod_lifecycle.
EXIT_STOPPED_POD_COLLISION = 76


class PodLifecycleProcessError(subprocess.CalledProcessError):
    """``CalledProcessError`` whose ``str()`` carries the child's stderr tail.

    Deliberately a SUBCLASS so every existing contract holds verbatim:
    ``except subprocess.CalledProcessError`` catches it, and
    ``dispatch_issue._provision_still_waiting`` reads the SAME
    ``returncode`` / ``cmd`` fields (the exit-75 still-waiting contract).
    ``self.stderr`` holds the BOUNDED tail (last
    ``_POD_LIFECYCLE_TAIL_MAX_LINES`` lines), not the full stream.
    """

    def __str__(self) -> str:
        base = super().__str__()
        if not self.stderr:
            return base
        return (
            f"{base}\n--- pod_lifecycle stderr tail "
            f"(last {_POD_LIFECYCLE_TAIL_MAX_LINES} lines max) ---\n{self.stderr}"
        )


def _run_pod_lifecycle_relay(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    relay: Any | None = None,
) -> None:
    """Run a ``pod_lifecycle.py`` subprocess, TEEING its stderr live.

    stdout stays INHERITED (untouched — provision progress prints pass
    through exactly as before). stderr is piped and relayed line-by-line
    to ``relay`` (default ``sys.stderr``) with an immediate flush, so the
    ``[wait-for-capacity]`` heartbeat lines the orchestrator scans live
    (SKILL.md Step 6b) keep streaming in real time across a multi-hour
    wait, while a bounded deque retains the tail. On non-zero exit raises
    :class:`PodLifecycleProcessError` with ``returncode`` + ``cmd``
    verbatim and the tail as ``stderr`` (#1465; incident #1336: an opaque
    ``exit status 1`` with zero diagnostics).

    EOF assumption: no pod_lifecycle local child detaches while inheriting
    stderr (true today — all its grandchildren are foreground
    ``subprocess.call``); a future backgrounded grandchild holding fd 2
    open would convert child-exit into a pipe-EOF wait here.
    """
    out = relay if relay is not None else sys.stderr
    tail: deque[str] = deque(maxlen=_POD_LIFECYCLE_TAIL_MAX_LINES)
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stderr is not None  # stderr=PIPE => stream exists
    with proc:
        try:
            # iter(readline, "") — unambiguous no-readahead line streaming.
            for line in iter(proc.stderr.readline, ""):
                out.write(line)
                out.flush()
                if len(line) > _POD_LIFECYCLE_TAIL_MAX_LINE_CHARS:
                    line = line[:_POD_LIFECYCLE_TAIL_MAX_LINE_CHARS] + "…[line truncated]\n"
                tail.append(line)
            rc = proc.wait()
        except BaseException:
            # subprocess.run parity (CPython 3.11 run(): `except: # Including
            # KeyboardInterrupt ... process.kill(); raise`): kill the child on
            # ANY interruption rather than hang in __exit__'s wait — preserves
            # today's Ctrl-C behavior exactly (#1465 plan §12 A10).
            proc.kill()
            raise
    if rc != 0:
        raise PodLifecycleProcessError(rc, cmd, output=None, stderr="".join(tail))


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


def _provisioned_pod_id(pod_name: str) -> str | None:
    """Exact RunPod pod id for a just-provisioned ``pod_name``, or ``None``.

    Reads the LIVE ``pods_ephemeral.json`` sidecar the provision subprocess
    just wrote (``pod_lifecycle.py`` persists ``pod_id`` there on every
    provision; path resolved via ``pod_config.resolve_live_pods_ephemeral``,
    which honors the documented ``pod_config.PODS_EPHEMERAL_JSON`` test
    seam). Best-effort BY DESIGN (#2038): a read failure logs a WARNING and
    returns ``None`` — the launch proceeds id-less exactly as pre-#2038, and
    the emergency-teardown arm degrades to a loud no-op (never a name-keyed
    terminate: multiple bare ``pod-<N>`` pods can coexist, #1739, and an
    ``--issue``-wide terminate destroys every pod resolving to the issue,
    #1485).
    """
    try:
        from scripts.pod_config import (
            resolve_live_pods_ephemeral,  # lazy: module top stays base-only
        )

        raw = json.loads(resolve_live_pods_ephemeral().read_text(encoding="utf-8"))
        row = (raw.get("pods") or {}).get(pod_name) or {}
        pod_id = str(row.get("pod_id") or "").strip()
        return pod_id or None
    except Exception as exc:
        logger.warning(
            "could not read the provisioned pod id for %s from pods_ephemeral.json "
            "(%r) — extra['pod_id'] omitted; emergency teardown falls back to the "
            "id-less loud no-op (#2038)",
            pod_name,
            exc,
        )
        return None


def _terminate_just_provisioned(
    *,
    pod_id: str | None,
    pod_name: str,
    issue: int,
    cause: str,
    terminate_fn: Callable[[str], Any] | None = None,
) -> bool:
    """Best-effort terminate of a pod THIS launch just provisioned (#2038).

    Fired ONLY from :meth:`RunPodBackend.launch`'s post-provision failure
    path, when a non-``RunPodWorkloadStartError`` exception leaves a pod
    RUNNING with no handle / sidecar / lease record — the invisible-billing
    shape (#1739). Terminates by EXACT pod id (never name-keyed, never
    ``--issue``-wide: #1739 duplicate names, #1485 issue-wide blast radius)
    under the sanctioned owner-driven :func:`kill_approval.verified_teardown`
    grant (upload verification is vacuous — the workload never started, so
    nothing was produced). Logs LOUDLY on every branch and NEVER raises —
    the caller re-raises the ORIGINAL launch exception, and a teardown
    failure must not mask it (returns ``True`` iff a terminate was issued
    and confirmed by the API).
    """
    if not pod_id:
        logger.error(
            "post-provision launch failure on issue %s: pod %s has NO captured pod id — "
            "cannot terminate by exact id; the pod may still be billing. Manual teardown: "
            "uv run python scripts/pod.py terminate --issue %s --yes (#2038, cause: %s)",
            issue,
            pod_name,
            issue,
            cause,
        )
        return False
    try:
        if terminate_fn is None:
            from scripts.runpod_api import terminate_pod  # lazy: module top stays base-only

            terminate_fn = terminate_pod
        from explore_persona_space.backends.kill_approval import verified_teardown

        with verified_teardown(
            target=f"{pod_name} ({pod_id})",
            reason=(
                "owner-driven teardown of a just-provisioned pod whose launch failed "
                "post-provision before any workload started — uploads vacuously "
                f"verified (#2038, cause: {cause})"
            ),
        ):
            terminate_fn(pod_id)
        logger.error(
            "post-provision launch failure on issue %s: terminated just-provisioned pod "
            "%s (pod_id=%s) to stop invisible billing (#2038, cause: %s)",
            issue,
            pod_name,
            pod_id,
            cause,
        )
        return True
    except Exception:
        logger.exception(
            "best-effort teardown of just-provisioned pod %s (pod_id=%s, issue %s) FAILED — "
            "the pod may still be billing; manual teardown required (#2038)",
            pod_name,
            pod_id,
            issue,
        )
        return False


def _dispose_post_provision_failure(
    *,
    exc: BaseException,
    workload_started: bool,
    pod_id: str | None,
    pod_name: str,
    issue: int,
) -> None:
    """Loud disposition for a post-provision non-typed launch failure (#2038).

    Called from :meth:`RunPodBackend.launch`'s outer ``except Exception``
    arm, immediately before it re-raises the ORIGINAL exception. Two arms:
    workload already started → the pod is doing live work, log LOUDLY and
    touch nothing; workload not started → best-effort exact-id terminate via
    :func:`_terminate_just_provisioned`. NEVER raises — a teardown failure
    must not mask the original launch exception (mask-guard pin:
    ``tests/test_issue2038_fallback_teardown.py::``
    ``test_mask_guard_original_exception_never_swallowed``).
    """
    if workload_started:
        # The workload is RUNNING — terminating would destroy live work.
        logger.error(
            "post-start launch failure on issue %s: pod %s (pod_id=%s) is "
            "left RUNNING because its workload already started; the launch "
            "records may be incomplete — inspect + finalize manually "
            "(#2038, cause: %s: %s)",
            issue,
            pod_name,
            pod_id,
            type(exc).__name__,
            exc,
        )
        return
    # No workload started: the pod bills invisibly. Best-effort terminate by
    # EXACT pod id; _terminate_just_provisioned never raises, and the extra
    # guard here keeps a monkeypatched / future-refactored teardown from ever
    # masking the ORIGINAL exception.
    try:
        _terminate_just_provisioned(
            pod_id=pod_id,
            pod_name=pod_name,
            issue=issue,
            cause=f"{type(exc).__name__}: {exc}",
        )
    except Exception:
        logger.exception(
            "emergency teardown wrapper itself raised for pod %s (issue %s) — "
            "pod may still be billing (#2038)",
            pod_name,
            issue,
        )


def _boot_disk_provision_args(spec: RunSpec) -> list[str]:
    """Disk-threading argv terms for ``pod_lifecycle.py provision`` (#1010/#1118).

    Extracted verbatim from :meth:`RunPodBackend.launch` (#2038 C901 relief;
    behavior byte-identical). CPU lane (#1010): the pod's only writable disk
    is the container overlay (/workspace rides it; incident #958) —
    ``--container-disk-gb``, floored at ``runpod_api.DEFAULT_CONTAINER_DISK_GB``
    (50, ``_CPU_CONTAINER_DISK_FLOOR_GB`` here) so threading can never REDUCE
    below today's behavior. The router's feasibility gate guarantees
    ``boot_disk_gb`` <= the instance cap on every AUTOMATED path; an explicit
    ``backend: runpod`` pin above the cap fails loud at pod_lifecycle's
    pre-API cap check / RunPod's own create-time validation. GPU lane
    (#1118): the big-data mount is the /workspace VOLUME (pod_lifecycle
    threads ``--volume-gb`` -> runpod_api volumeInGb), floored at the 200 GB
    argparse default (``_GPU_VOLUME_FLOOR_GB``) — thread-or-grow, never
    shrink. No deterministic pre-API cap exists for GPU volumeInGb (unlike
    the probe-verified CPU caps) — an unsatisfiable size surfaces LOUD at
    RunPod create time (RunPodError -> non-zero provision exit ->
    CalledProcessError) or as a capacity miss (wait-for-capacity budget),
    never as a silent downsize (the #1112 ENOSPC incident: a ~575 GB plan on
    the default 200 GB volume). Returns ``[]`` when the spec states no
    footprint; raises the named fail-loud ValueError on a malformed /
    fractional value BEFORE any pod is paid for.
    """
    raw_boot_disk = (spec.extra or {}).get("boot_disk_gb") or 0
    try:
        boot_disk_gb = int(raw_boot_disk)
        if float(raw_boot_disk) != boot_disk_gb:
            # A fractional value (e.g. 575.5) would silently TRUNCATE via
            # int() -- reject it instead of provisioning less disk than
            # the plan stated.
            raise ValueError("fractional GB value")
    except (TypeError, ValueError) as exc:
        # Named fail-loud parse mirroring router._footprint_int (#1118):
        # GPU intents bypass the router's CPU-only footprint gate, so a
        # malformed value would otherwise hit a bare int() traceback.
        # Raised BEFORE the provision subprocess -- no pod is paid for.
        raise ValueError(
            f"spec.extra['boot_disk_gb'] is not an integer: {raw_boot_disk!r} "
            f"(malformed disk requirement on issue {spec.issue})"
        ) from exc
    if not boot_disk_gb:
        return []
    from explore_persona_space.backends.router import (
        RUNPOD_CPU_INSTANCE_FOR_INTENT,  # lazy: module top stays base-only
    )

    if spec.intent in RUNPOD_CPU_INSTANCE_FOR_INTENT:
        return [
            "--container-disk-gb",
            str(max(_CPU_CONTAINER_DISK_FLOOR_GB, boot_disk_gb)),
        ]
    return [
        "--volume-gb",
        str(max(_GPU_VOLUME_FLOOR_GB, boot_disk_gb)),
    ]


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
    sentinel (see :func:`mint_runpod_attempt_id`). Two writer
    conventions, split by executor: on EXPERIMENTER-driven dispatches
    the convention lives in ``.claude/agents/experimenter.md`` — the
    path is read from the launch sidecar's
    ``extra.expected_artifacts.sentinel_path``, the write is chained
    on the workload's exit status, and stale sentinels are cleared
    pre-(re)launch; on the BACKEND-executed leg
    (``execute_workload``, #909) the rendered launcher chains the
    same success-gated write itself (see
    :func:`_render_launch_script`) — there is no experimenter on
    that path.
    """
    from explore_persona_space.backends.artifacts import SENTINEL_FILENAME

    return f"/workspace/eval_results/issue_{issue}/{attempt_id}/{SENTINEL_FILENAME}"


# ---------------------------------------------------------------------------
# Workload execution leg (#909)
# ---------------------------------------------------------------------------

#: Seconds between the detach call and the liveness-verification SSH call.
#: A same-invocation probe cannot catch SIGHUP-on-disconnect death, so the
#: verify runs from a SEPARATE SSH invocation a few seconds later (the
#: ``.claude/agents/experimenter.md`` § "During Execution" convention).
WORKLOAD_VERIFY_DELAY_SECONDS = 3.0

#: Branch names interpolated into the remote sync script must be plain git
#: ref characters — the branch comes from ``spec.extra["repo_branch"]``
#: (orchestrator-controlled), but a quoting slip would only surface pod-side,
#: so fail LOUD here instead.
_BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]+")

#: Verify-script success line: ``LAUNCH-OK pid=<int>`` (optionally
#: ``via=<fresh pidfile>`` for the GCP-parity self-daemonizing acceptance).
_LAUNCH_OK_RE = re.compile(r"LAUNCH-OK pid=(\d+)")

#: Double-fire guard line from the launch script (exit 5): the live PID.
_ALREADY_RUNNING_PID_RE = re.compile(r"ALREADY-RUNNING pid=(\d+)")

#: Local ssh bound (seconds) for each branch-sync attempt (#1858; caps
#: recalibrated #1981). The remote sync script self-bounds its three git
#: mutation ops with ``timeout -k 10`` at 120/90/90 — worst case 330 s
#: when every TERM needs the 10 s KILL grace — and the remote bounds MUST
#: fire before this local bound so a FUSE-hung git is killed REMOTELY (its
#: ``.git/index.lock`` hold dies with it) instead of surviving a local
#: ``TimeoutExpired`` that kills only the local ssh client (incident #1769
#: fu1: the orphaned remote git held the lock and wedged the launch
#: terminally). 360 keeps ~30 s of ssh-connect + script overhead over the
#: 330 s remote worst case. The 90 s checkout/reset caps are sized ~1.5x
#: the ~59.5 s ``git status`` latency measured on pod-1895's slow MooseFS
#: mount (2026-08-02, #1895), so a genuinely-divergent healthy-but-slow
#: mount finishes under the cap where the pre-#1981 20 s caps timed out at
#: rc=124 on both attempts of the parent incident. The already-at-tip
#: short-circuit in ``_render_branch_sync_script`` avoids the mutation
#: paths entirely on the common case.
SYNC_SSH_TIMEOUT_SECONDS = 360

#: Reap-script (#1858) terminal report line: ``REAP-OK killed=<n>
#: survivors=<m> lock_removed=yes``. ``survivors>0`` = git pids that
#: outlived SIGKILL (uninterruptible D-state — the MooseFS mount itself is
#: wedged, so a sync retry is futile).
_SYNC_REAP_OK_RE = re.compile(r"REAP-OK killed=(\d+) survivors=(\d+) lock_removed=yes")


class RunPodWorkloadStartError(RuntimeError):
    """A requested ``--workload-cmd`` execution did not start on the pod (#909).

    Raised by the RunPod execution leg on ANY start failure — missing
    pods.conf row, branch-sync mismatch, double-fire guard, dead PID with
    no fresh pidfile, missing log. ``scripts/dispatch_issue.py launch``
    surfaces it as a ``reason: runpod_workload_start_failed`` failure JSON
    + exit 2 — a requested execution that did not start NEVER returns ok.
    The pod is left RUNNING for SSH diagnosis (the RunPod-as-diagnosis-lane
    doctrine, ``.claude/rules/compute-backend-failover.md``). The window is
    BOUNDED (#1997): the autonomous-session watcher's diagnosis-window arm
    reversibly STOPS (never terminates) the pod after
    ``EPS_RUNPOD_DIAGNOSIS_TTL_HOURS`` (default 6h) once no ``keep-running``
    tag and no live owner remain — volume + ``/workspace`` logs preserved;
    ``pod.py resume --issue <N>`` re-opens a fresh window.
    """

    def __init__(self, message: str, *, handle: RunHandle | None = None) -> None:
        super().__init__(message)
        #: The PARTIAL RunHandle when a pod was provisioned before the start
        #: leg failed (#954); None when nothing was provisioned. The router's
        #: terminal rung + the backend_poll failover legs key their
        #: persist-then-surface behavior on this — a pod exists and BILLS, so
        #: collapsing this case into ``NoComputeAvailableError`` would be
        #: FALSE (the #931 incident: the mislabel invited a second paid
        #: dispatch while pod-931 billed invisibly).
        self.handle = handle


class RunPodProvisionBranchMismatchError(RunPodWorkloadStartError):
    """Post-bootstrap branch assertion failed — the pod is not on the requested branch (#1698).

    Raised by :func:`_assert_pod_on_branch` when the ``--repo-branch`` value
    threaded from the orchestrator (``spec.extra["repo_branch"]``) does NOT
    match the on-pod ``git rev-parse --abbrev-ref HEAD`` after
    ``bootstrap_pod.sh`` completes. This surfaces a PLUMBING regression LOUD
    instead of running stale code for 25+ minutes: the #1689 R8/R9 shape
    landed the pod on ``main`` twice while every plan-level probe read the
    branch value as correctly threaded.

    Subclasses :class:`RunPodWorkloadStartError` so the existing partial-handle
    machinery at :meth:`RunPodBackend.launch`'s ``#954`` catch site wraps it
    into a partial-handle failure — the pod exists and BILLS, so the
    failover / diagnostics contract is identical to any other workload-start
    failure. The pod is left RUNNING for SSH diagnosis (the
    RunPod-as-diagnosis-lane doctrine; the window is bounded by the
    watcher's #1997 diagnosis-window arm — see
    :class:`RunPodWorkloadStartError`).
    """


def _assert_pod_on_branch(pod_name: str, expected_branch: str) -> None:
    """Post-bootstrap fail-loud: pod HEAD branch MUST equal ``expected_branch`` (#1698).

    SSH-probes ``git rev-parse --abbrev-ref HEAD`` on the pod's
    ``/workspace/explore-persona-space`` clone and raises
    :class:`RunPodProvisionBranchMismatchError` on ANY mismatch — the pod
    landed on the wrong branch, the plumbing dropped the requested value
    (``bootstrap_pod.sh:52`` defaults ``BOOTSTRAP_BRANCH=main``), and the
    workload would silently run stale code.

    Called by :meth:`RunPodBackend.launch` immediately AFTER
    ``_run_pod_lifecycle_relay`` returns, and ONLY when the launch explicitly
    requested a non-``main`` branch via ``spec.extra["repo_branch"]``: the
    default-``main`` case is a no-op (the assertion binds only to launches
    that named a specific branch). See :meth:`RunPodBackend.launch` for the
    bind condition.
    """
    host, port = _resolve_pod_endpoint(pod_name)
    out = _ssh_pod_run(
        host,
        port,
        "cd /workspace/explore-persona-space && git rev-parse --abbrev-ref HEAD",
        timeout=30,
        context=f"post-bootstrap branch verify on {pod_name}",
    )
    # git rev-parse --abbrev-ref HEAD prints ONE line: the branch name (or
    # 'HEAD' on a detached checkout). Take the last non-empty line so a
    # bootstrap that appended a trailing progress line does not corrupt the
    # comparison — the branch line itself is the last real content.
    actual = ""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped:
            actual = stripped
    if actual != expected_branch:
        raise RunPodProvisionBranchMismatchError(
            f"pod {pod_name!r} bootstrapped onto branch {actual!r}, expected "
            f"{expected_branch!r} — the --repo-branch plumbing dropped the "
            f"value (bootstrap_pod.sh:52 defaults BOOTSTRAP_BRANCH=main when "
            f"the env var is unset); refusing to proceed on stale code"
        )


def _resolve_pod_endpoint(pod_name: str) -> tuple[str, int]:
    """Resolve ``(host, port)`` for ``pod_name`` from the live pods.conf.

    pods.conf is the SSH config source (refreshed from the live RunPod API
    by provision — ``pod_lifecycle._upsert_pods_conf``), so a freshly
    provisioned pod always has a row. Raises
    :class:`RunPodWorkloadStartError` on a missing/invalid row — the
    execution leg cannot SSH without it.
    """
    # Lazy import — same style as ``poll``'s ``from scripts.poll_pipeline
    # import ...``; keeps this module importable without scripts/ on
    # sys.path for launch/teardown-only callers.
    from scripts.pod_config import parse_pods_conf

    try:
        pods = parse_pods_conf()
    except Exception as exc:
        raise RunPodWorkloadStartError(
            f"cannot read pods.conf to resolve {pod_name!r} for workload execution "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    for pod in pods:
        if pod.name == pod_name:
            if not pod.host or not pod.port:
                raise RunPodWorkloadStartError(
                    f"pods.conf row for {pod_name!r} has no usable host/port "
                    f"(host={pod.host!r}, port={pod.port!r}); cannot SSH to execute "
                    "the workload — run `pod.py config --refresh-from-api` and retry"
                )
            return pod.host, int(pod.port)
    raise RunPodWorkloadStartError(
        f"pod {pod_name!r} has no pods.conf row — cannot SSH to execute the workload "
        "(provision normally upserts the row; run `pod.py config --refresh-from-api`)"
    )


def _ssh_pod_run(host: str, port: int, command: str, *, timeout: int, context: str) -> str:
    """Run ``command`` on the pod over SSH; return stdout, raise typed error on failure.

    Mirrors the battle-tested explicit transport of
    ``scripts/pod_lifecycle._restore_uv_on_pod`` / ``bootstrap_pod.sh``
    (host/port from pods.conf; NOT the ``~/.ssh/config``-by-name form —
    pods.conf is the config SOURCE here). Any non-zero exit / timeout /
    transport error raises :class:`RunPodWorkloadStartError` carrying the
    stdout + stderr tails, prefixed with ``context`` so the caller's error
    names the failing step.
    """
    argv = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "BatchMode=yes",
        "-i",
        str(Path.home() / ".ssh" / "id_ed25519"),
        "-p",
        str(port),
        f"root@{host}",
        command,
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RunPodWorkloadStartError(
            f"{context}: ssh to {host}:{port} failed ({type(exc).__name__}: {exc})"
        ) from exc
    if proc.returncode != 0:
        raise RunPodWorkloadStartError(
            f"{context}: remote command exited rc={proc.returncode} on {host}:{port}; "
            f"stdout tail: {(proc.stdout or '')[-1500:]!r}; "
            f"stderr tail: {(proc.stderr or '')[-800:]!r}"
        )
    return proc.stdout or ""


def _render_branch_sync_script(branch: str) -> str:
    """Remote script (a): sync the pod clone to ``branch`` (fetch + reset, never pull).

    ``git pull`` is banned here — it exits 0 on a stale ``.git/index.lock``
    without moving HEAD, and divergent ``.claude/**`` spec files block the
    ff-only form (both in ``.claude/rules/gotchas.md``). The fetch +
    ``checkout -f -B`` + ``reset --hard`` sequence defeats both, and the
    pod-side ``HEAD == FETCH_HEAD`` verification never trusts pull stdout.
    (Pod-side ``reset --hard`` is sanctioned — the VM-tree ban does not
    apply to the disposable pod clone.)

    #1858: each git MUTATION op self-bounds with ``timeout -k 10`` (120 s
    fetch / 90 s checkout / 90 s reset — worst case 330 s, under the
    ``SYNC_SSH_TIMEOUT_SECONDS`` local bound) so a MooseFS-FUSE-hung git
    is killed REMOTELY, releasing its ``.git/index.lock`` hold, instead of
    surviving the local ssh timeout as an orphaned lock-holder (incident
    #1769 fu1). The rev-parse verification lines stay bare — ref reads,
    not FUSE-heavy object-store ops.

    #1981: after the fetch, compare ``HEAD`` to ``FETCH_HEAD`` (both are
    cheap ref reads) and short-circuit ``SYNC-OK`` when the pod tree is
    ALREADY at the fetched tip on the requested branch — the mutation
    paths (checkout + reset) never execute, avoiding the FUSE-heavy git
    ops that timed out at 20 s on pod-1895's slow-but-healthy MooseFS
    mount (~59.5 s ``git status``, incident #1895). The genuinely-
    divergent path retains the ``checkout -f -B`` + ``reset --hard``
    sequence at 90 s per op (~1.5x the pod-1895 latency measurement), so
    a real HEAD change succeeds where the pre-#1981 caps failed loud.
    """
    if not _BRANCH_RE.fullmatch(branch):
        raise RunPodWorkloadStartError(
            f"refusing to sync suspicious branch name {branch!r} "
            "(expected plain git ref characters [A-Za-z0-9._/-])"
        )
    return "\n".join(
        [
            "set -eu",
            'export PATH="/root/.local/bin:$PATH"',
            "cd /workspace/explore-persona-space",
            "pgrep -x git >/dev/null 2>&1 || rm -f .git/index.lock",
            f'timeout -k 10 120 git fetch origin "refs/heads/{branch}"',
            # #1981: cheap ref reads decide whether the mutation paths need
            # to run at all. HEAD_SHA cannot be "none" on the matched path
            # (git rev-parse HEAD on a valid clone always emits a full sha
            # or fails set -eu), so the SYNC-OK regex in _attempt_sync
            # (r"SYNC-OK ([0-9a-f]+)") captures the real head.
            "HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo none)",
            "FETCH_SHA=$(git rev-parse FETCH_HEAD)",
            "CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)",
            f'if [ "$HEAD_SHA" = "$FETCH_SHA" ] && [ "$CUR_BRANCH" = "{branch}" ]; then',
            '  echo "SYNC-OK $HEAD_SHA (already-at-tip short-circuit)"',
            "  exit 0",
            "fi",
            f'timeout -k 10 90 git checkout -q -f -B "{branch}" FETCH_HEAD',
            "timeout -k 10 90 git reset --hard -q FETCH_HEAD",
            "HEAD_SHA=$(git rev-parse HEAD); FETCH_SHA=$(git rev-parse FETCH_HEAD)",
            '[ "$HEAD_SHA" = "$FETCH_SHA" ] || '
            '{ echo "SYNC-MISMATCH head=$HEAD_SHA fetch=$FETCH_SHA" >&2; exit 3; }',
            'echo "SYNC-OK $HEAD_SHA"',
        ]
    )


def _render_sync_reap_script(clone_dir: str = "/workspace/explore-persona-space") -> str:
    """Remote reap script (#1858): SIGKILL leftover git pids + clear the lock.

    Runs after a FAILED branch-sync attempt, before the single retry. The
    MODAL designed path is ZERO live git pids (the sync script's per-op
    remote ``timeout`` already killed the hung git), so the ``pgrep -x
    git`` enumerations are GUARDED (``|| true``) — an empty match yields
    ``killed=0 survivors=0`` and exit 0, never a ``set -eu`` death (the
    critic round-1 Must-Fix). Any live git pids are SIGKILLed with a
    bounded ~10 s death-wait poll, survivors re-enumerated (a pid that
    outlives SIGKILL is in uninterruptible D-state — the MooseFS mount
    itself is wedged), then ``.git/index.lock`` is removed UNCONDITIONALLY.
    Always emits ``REAP-OK killed=<n> survivors=<m> lock_removed=yes`` and
    exits 0 on every non-wedged path. ``clone_dir`` is an internal seam so
    the real-bash regression test can execute the rendered script against
    a tmp fake clone; production callers use the default. No ``task.py``
    shellout (pods run on ``issue-<N>`` branches — the CLAUDE.md hard rule).
    """
    return "\n".join(
        [
            "set -eu",
            f"cd {clone_dir}",
            "GIT_PIDS=$(pgrep -x git || true)",
            "KILLED=0",
            "for p in $GIT_PIDS; do",
            '  kill -9 "$p" 2>/dev/null || true',
            "  KILLED=$((KILLED + 1))",
            "done",
            'if [ "$KILLED" -gt 0 ]; then',
            "  i=0",
            '  while [ "$i" -lt 10 ]; do',
            "    sleep 1",
            "    pgrep -x git >/dev/null 2>&1 || break",
            "    i=$((i + 1))",
            "  done",
            "fi",
            "SURVIVORS=$(pgrep -x git || true)",
            "NSURV=0",
            "for p in $SURVIVORS; do",
            "  NSURV=$((NSURV + 1))",
            "done",
            "rm -f .git/index.lock",
            'echo "REAP-OK killed=$KILLED survivors=$NSURV lock_removed=yes"',
            "exit 0",
        ]
    )


def _launch_epoch_path(issue: int) -> str:
    """Pod-side launch-epoch stamp anchoring the verify script's fresh-pidfile check."""
    return f"/workspace/logs/issue-{issue}.launch_epoch"


def _launcher_path(issue: int) -> str:
    """Canonical pod-side launcher-script path (the experimenter.md convention)."""
    return f"/workspace/launch_issue_{issue}.sh"


def _render_launch_script(
    *,
    issue: int,
    workload_cmd: str,
    log_path: str,
    pid_file: str,
    sentinel_path: str,
    attempt_id: str,
    env_pins: Mapping[str, str] | None = None,
) -> str:
    """Remote script (b): write the launcher + detach it (setsid + nohup + pidfile).

    Reuses the canonical experimenter launcher pattern
    (``.claude/agents/experimenter.md`` § "During Execution") verbatim,
    with one deliberate delta: no ``exec`` before the workload line —
    ``workload_cmd`` is an arbitrary shell line (possibly env-var
    prefixed / compound), which ``exec`` cannot take; the pidfile then
    holds the launcher bash PID, whose liveness tracks the workload
    (``poll_once`` probes liveness, not process identity). The
    ``ALREADY-RUNNING`` live-PID guard converts a flag+experimenter
    double-launch into a loud no-op (exit 5). ``workload_cmd`` is embedded
    VERBATIM inside the quoted heredoc (the GCP trusted-single-line
    doctrine; ``RunSpec.__post_init__`` guarantees single-line).

    Completion sentinel (#909 r2, ``runpod-execute-missing-completion-
    sentinel``): on this backend-owned leg there is NO experimenter to
    chain ``write_completion_sentinel`` (``experimenter.md`` step 11), so
    the launcher itself captures the workload's exit status and, ON
    SUCCESS ONLY (rc 0), writes the attempt-namespaced completion
    sentinel — ``{"phase": "done", "issue": <N>, "attempt_id": ...}``,
    the exact shape ``artifacts._check_sentinel`` validates — at the SAME
    ``sentinel_path`` the launch handle declares. This mirrors the
    established backend-owned writer convention on the sibling lanes
    (``gcp.py``'s startup-script sentinel heredoc; ``slurm.py``'s
    terminal block). The outer portion clears, before detach (same guard
    family as the pidfile rm), EVERY stale sentinel the #685
    single-live-sibling fallback could resolve on a same-pod
    re-execution: the declared attempt path PLUS the flat legacy path
    ``issue_<N>/.completion-sentinel.json`` and the attempt-sibling
    wildcard ``issue_<N>/*/.completion-sentinel.json`` — the
    experimenter step-11.3 breadth, ported here by #976 (a prior
    attempt's success sentinel survives on the persistent ``/workspace``
    volume, and ``_check_sentinel`` validates phase+issue only, so a
    resolved stale sibling would finalize a crashed re-execution green).
    The widened clear also removes a COMPLETED prior attempt's sentinel
    before a pending finalize reads it — the fail-loud direction
    (finalize FAILs "sentinel missing", never a false green; the same
    supersession semantics as the experimenter step-11.3 clear). The
    launcher exits with the workload's own rc so the exit status is
    unchanged by the chain.

    Detached-workload wait before the sentinel write (#977, the GCP #601
    parity — ``gcp.py``'s find-newer wait): ``workload_cmd`` is expected
    to BLOCK until the workload completes; a SELF-DAEMONIZING command
    (one that setsid-forks the real driver and returns at daemonize
    time) would otherwise publish ``phase=done`` minutes into a
    multi-hour run. So the rc==0 branch, BEFORE the sentinel write,
    waits on every live pid found in a FRESH ``/workspace/logs/*.pid``
    (mtime at or after the in-launcher workload-start epoch, captured
    immediately before ``workload_cmd``). Contract (write-before-return):
    a detached workload MUST write its driver pid to such a fresh
    pidfile BEFORE ``workload_cmd`` returns — the ``launch_issue_<N>.sh``
    convention — for the wait to bind; a pidfile written only AFTER
    ``workload_cmd`` returns may be missed by the loop's scan and
    degrades to the pre-#977 behavior (premature sentinel) — the same
    residual GCP #601 carries, documented, not fixed here. Two deliberate
    mechanism deltas from the GCP reference: freshness is the INCLUSIVE
    ``stat -c %Y >= $WORKLOAD_START_EPOCH`` (the verify-script idiom;
    GCP's strictly-newer ``find -newer`` would miss a pidfile written in
    the same second as the start-mark at coarse MooseFS mtime
    granularity), and self-exclusion is by PID VALUE
    (``[ "$wpid" = "$$" ]``, race-free at any mtime granularity — the
    launcher writes its OWN pid to the canonical pidfile pre-workload,
    so a naive port would self-deadlock) rather than by pidfile PATH — a
    convention-following detached driver OVERWRITES the canonical
    pidfile with its own pid, so a path-based skip would miss exactly
    the driver that must be waited on, while ``$$`` cannot be reused
    while this launcher is alive. Blocking workloads write no fresh
    pidfile, so the wait is a no-op pass-through. No in-script timeout
    (faithful parity — an in-script timeout would re-create the
    premature-done class on a slow-but-healthy run); the wait is bounded
    externally by the pod TTL + the poller's stall escalation + the
    watcher's pod-safety pass (the GCP analogue is
    ``--max-run-duration``). The sentinel is written after the wait
    regardless of the DETACHED process's exit status (``kill -0``
    polling cannot recover a non-child's exit code on either lane; the
    detached driver's own results sentinel / failure classification is
    the poller's outcome channel).
    """
    launcher = _launcher_path(issue)
    epoch_file = _launch_epoch_path(issue)
    sentinel_dir = sentinel_path.rsplit("/", 1)[0]
    # #976: stale-clear breadth mirrors artifacts._resolve_live_sentinel's
    # sibling probe (_default_glob_sentinels: grandparent-of-declared +
    # */<name>) plus the experimenter step-11.3 flat legacy path — whatever
    # the #685 single-live-sibling fallback COULD resolve on a same-pod
    # re-execution, this clear removes first. Derived from sentinel_path
    # itself (no second copy of the /workspace/eval_results/issue_<N> root).
    issue_dir = sentinel_dir.rsplit("/", 1)[0]
    sentinel_name = sentinel_path.rsplit("/", 1)[1]
    # #1669: launch env pins (WANDB_PROJECT et al., incident #1586) —
    # re-validated here as defense in depth (the rendered launcher executes
    # as root, and the handle sidecar the failover reconstructors read is a
    # hand-editable JSON), then rendered as shlex-quoted `export K=V` lines
    # spliced immediately BEFORE the `WANDB_PROJECT:-issue<N>` default below,
    # so the `:-` default preserves the pin and a pin-less render is
    # byte-identical. Values are validated single-line, so the quoted EPSEOF
    # heredoc delimiter cannot be terminated by a pin. An inline
    # `WANDB_PROJECT=... cmd` prefix in workload_cmd still supersedes the
    # export for that command (bash per-command env semantics — the
    # documented zero-code override; see the REPO_ROOT comment doctrine).
    pin_lines = [
        f"export {k}={shlex.quote(str(v))}" for k, v in sorted(validate_env_pins(env_pins).items())
    ]
    sentinel_json = json.dumps({"phase": "done", "issue": int(issue), "attempt_id": attempt_id})
    if "'" in sentinel_json:
        # The JSON is embedded single-quoted inside the launcher; both
        # inputs are internally minted, so this can only fire on a caller
        # bug — fail LOUD rather than render a broken script.
        raise RunPodWorkloadStartError(
            f"refusing to embed sentinel JSON containing a single quote: {sentinel_json!r}"
        )
    return "\n".join(
        [
            "set -eu",
            "mkdir -p /workspace/logs",
            f"OLD_PID=$(cat {pid_file} 2>/dev/null || true)",
            'if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then',
            '  echo "ALREADY-RUNNING pid=$OLD_PID" >&2',
            "  exit 5",
            "fi",
            f"rm -f {pid_file}",
            f"rm -f {sentinel_path} {issue_dir}/{sentinel_name} {issue_dir}/*/{sentinel_name}",
            f"mkdir -p {sentinel_dir}",
            f"date +%s > {epoch_file}",
            f"cat > {launcher} << 'EPSEOF'",
            "#!/bin/bash",
            "set -uo pipefail",
            'export PATH="/root/.local/bin:$PATH"',
            "cd /workspace/explore-persona-space",
            "set -a; [ -f .env ] && source .env; set +a",
            'export REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"',
            *pin_lines,
            f'export WANDB_PROJECT="${{WANDB_PROJECT:-issue{issue}}}"',
            f"echo $$ > {pid_file}",
            "WORKLOAD_START_EPOCH=$(date +%s)",
            workload_cmd,
            "WORKLOAD_RC=$?",
            "# Completion sentinel: written ONLY when the workload exited 0 (the",
            "# backend-owned twin of the GCP/SLURM terminal sentinel write).",
            'if [ "$WORKLOAD_RC" -eq 0 ]; then',
            "  # Wait for detached workloads (#601 GCP parity, #977): a workload_cmd",
            "  # that self-daemonizes (forks the real driver) returns immediately —",
            "  # writing the sentinel here would publish done at daemonize time.",
            "  # (Comment says 'forks', not the s-word: the liveness tests token-scan",
            "  # the rendered script for the detach line's own tokens.) Contract: a",
            "  # detached workload writes its pid to a fresh /workspace/logs/*.pid",
            "  # (the launch_issue_<N>.sh convention). Freshness = stat mtime >=",
            "  # workload-start epoch (inclusive, the verify-script predicate — a",
            "  # strictly-newer find would miss a same-second pidfile at MooseFS",
            "  # mtime granularity). Self-exclusion is by PID VALUE, never path:",
            "  # a convention-following driver OVERWRITES the canonical pidfile,",
            "  # so a path skip would miss it, while $$ cannot be reused while",
            "  # this launcher is alive. Blocking workloads write no fresh pidfile",
            "  # -> no-op. Bounded externally by the pod TTL + the poller's stall",
            "  # escalation (the GCP analogue is --max-run-duration).",
            "  for pf in /workspace/logs/*.pid; do",
            '    [ -f "$pf" ] || continue',
            '    [ "$(stat -c %Y "$pf" 2>/dev/null || echo 0)"'
            ' -ge "$WORKLOAD_START_EPOCH" ] || continue',
            '    wpid=$(cat "$pf" 2>/dev/null) || continue',
            '    [ -n "$wpid" ] || continue',
            '    [ "$wpid" = "$$" ] && continue',
            '    echo "[launcher] waiting on detached workload pid=$wpid ($pf)"',
            '    while kill -0 "$wpid" 2>/dev/null; do sleep 30; done',
            '    echo "[launcher] detached workload pid=$wpid exited"',
            "  done",
            f"  mkdir -p {sentinel_dir}",
            f"  printf '%s\\n' '{sentinel_json}' > {sentinel_path}",
            "fi",
            'exit "$WORKLOAD_RC"',
            "EPSEOF",
            f"chmod +x {launcher}",
            f"setsid nohup bash {launcher} > {log_path} 2>&1 < /dev/null &",
            'echo "WRAPPER-STARTED $!"',
        ]
    )


def _render_verify_script(*, issue: int, log_path: str, pid_file: str) -> str:
    """Remote script (c): liveness verification, run from a SEPARATE SSH invocation.

    ``LAUNCH-OK`` iff the canonical pidfile PID is alive, OR a FRESH
    pod-side ``/workspace/logs/*.pid`` written at/after the launch epoch
    carries a live PID (the GCP self-daemonizing-driver parity — a driver
    that setsid-forks the real work and exits would otherwise read a false
    ``LAUNCH-DEAD``; see ``backends/gcp.py``'s fresh-``*.pid`` wait), AND
    the log exists. Anything else prints ``LAUNCH-DEAD`` + a log tail and
    exits 4.
    """
    epoch_file = _launch_epoch_path(issue)
    return "\n".join(
        [
            f"LAUNCH_EPOCH=$(cat {epoch_file} 2>/dev/null || echo 0)",
            f"PID=$(cat {pid_file} 2>/dev/null || true)",
            f'if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && [ -f "{log_path}" ]; then',
            '  echo "LAUNCH-OK pid=$PID"',
            "  exit 0",
            "fi",
            "# Launcher PID gone: accept any FRESH pod-side pidfile (mtime >=",
            "# launch epoch) whose PID is alive — the GCP self-daemonizing-driver",
            "# convention.",
            "for f in /workspace/logs/*.pid; do",
            '  [ -f "$f" ] || continue',
            '  [ "$(stat -c %Y "$f" 2>/dev/null || echo 0)" -ge "$LAUNCH_EPOCH" ] || continue',
            '  FPID=$(cat "$f" 2>/dev/null || true)',
            f'  if [ -n "$FPID" ] && kill -0 "$FPID" 2>/dev/null && [ -f "{log_path}" ]; then',
            '    echo "LAUNCH-OK pid=$FPID via=$f"',
            "    exit 0",
            "  fi",
            "done",
            'echo "LAUNCH-DEAD pid=${PID:-none}"',
            f'tail -20 "{log_path}" 2>/dev/null || true',
            "exit 4",
        ]
    )


def _execute_workload_on_pod(
    spec: RunSpec,
    *,
    pod_name: str,
    log_path: str,
    pid_file: str,
    sentinel_path: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Sync the pod clone, start ``spec.workload_cmd`` detached, verify liveness (#909).

    Sequences remote scripts (a) branch sync → (b) write launcher + detach
    → sleep → (c) verify (a SEPARATE SSH invocation, catching
    SIGHUP-on-disconnect death). Returns ``{"workload_pid": int,
    "launcher_path": str, "synced_sha": str}`` on success; raises
    :class:`RunPodWorkloadStartError` on ANY start failure (the pod stays
    RUNNING for diagnosis).

    ``sentinel_path`` / ``attempt_id`` MUST be the SAME pair ``launch()``
    bakes into the handle's expected-artifacts declaration — script (b)
    chains the completion-sentinel write to that exact path on workload
    success (#909 r2), so ``confirm_artifacts`` / ``_cmd_finalize`` can
    pass on a successful backend-executed run.
    """
    host, port = _resolve_pod_endpoint(pod_name)
    branch = str((spec.extra or {}).get("repo_branch") or "") or "main"
    issue = int(spec.issue)

    # (a) Branch sync — fetch + checkout -f -B + reset --hard; pod-side
    # HEAD == FETCH_HEAD verification (never trusting pull stdout). #1858:
    # the FIRST failure (local TimeoutExpired-driven RunPodWorkloadStartError,
    # non-zero rc, or missing SYNC-OK) runs a bounded git kill-and-reap on
    # the pod, then retries the sync EXACTLY ONCE — incident #1769 fu1: a
    # MooseFS-hung remote git held .git/index.lock past the local timeout
    # and a manual kill+reap+resync later succeeded ON THE SAME POD, so a
    # single bounded automatic retry recovers this class. Unkillable
    # survivors (D-state — the mount itself is wedged) skip the retry.
    sync_ctx = f"branch sync of {pod_name} to {branch!r}"

    def _attempt_sync() -> tuple[str | None, str]:
        """One sync attempt → ``(synced_sha, "")`` or ``(None, failure summary)``."""
        try:
            out = _ssh_pod_run(
                host,
                port,
                _render_branch_sync_script(branch),
                timeout=SYNC_SSH_TIMEOUT_SECONDS,
                context=sync_ctx,
            )
        except RunPodWorkloadStartError as exc:
            return None, str(exc)
        match = re.search(r"SYNC-OK ([0-9a-f]+)", out)
        if match:
            return match.group(1), ""
        return None, f"{sync_ctx} did not confirm SYNC-OK; output tail: {out[-1500:]!r}"

    synced_sha, first_failure = _attempt_sync()
    if synced_sha is None:
        logger.warning(
            "%s failed; running git kill-and-reap before one retry (#1858): %s",
            sync_ctx,
            first_failure[-500:],
        )
        try:
            reap_out = _ssh_pod_run(
                host,
                port,
                _render_sync_reap_script(),
                timeout=60,
                context=f"git kill-and-reap on {pod_name}",
            )
        except RunPodWorkloadStartError as exc:
            raise RunPodWorkloadStartError(
                f"{sync_ctx} failed and the git kill-and-reap probe ALSO failed — "
                f"pod-level wedge, no retry; pod left RUNNING for diagnosis. "
                f"Sync failure: {first_failure}; reap failure: {exc}"
            ) from exc
        reap_match = _SYNC_REAP_OK_RE.search(reap_out)
        if not reap_match:
            raise RunPodWorkloadStartError(
                f"{sync_ctx} failed and the git kill-and-reap did not confirm REAP-OK — "
                f"no retry; pod left RUNNING for diagnosis. Sync failure: {first_failure}; "
                f"reap output tail: {reap_out[-1500:]!r}"
            )
        killed, survivors = int(reap_match.group(1)), int(reap_match.group(2))
        logger.info("git kill-and-reap on %s: %s", pod_name, reap_match.group(0))
        if survivors > 0:
            raise RunPodWorkloadStartError(
                f"{sync_ctx} failed with unkillable git survivors (moosefs D-state "
                f"signature) — {survivors} git pid(s) outlived SIGKILL, the MooseFS "
                "mount itself is wedged and a sync retry is futile; pod left RUNNING "
                f"for diagnosis. Sync failure: {first_failure}; reap: {reap_match.group(0)}"
            )
        logger.info("%s: retrying once after clean reap (killed=%d survivors=0)", sync_ctx, killed)
        synced_sha, second_failure = _attempt_sync()
        if synced_sha is None:
            raise RunPodWorkloadStartError(
                f"sync retry after reap failed on {pod_name} (branch {branch!r}); pod "
                f"left RUNNING for diagnosis. First failure: {first_failure}; reap: "
                f"{reap_match.group(0)}; retry failure: {second_failure}"
            )
        logger.info("%s: retry succeeded (sha %s)", sync_ctx, synced_sha)

    # (b) Write the launcher + detach it.
    try:
        launch_out = _ssh_pod_run(
            host,
            port,
            _render_launch_script(
                issue=issue,
                workload_cmd=spec.workload_cmd,
                log_path=log_path,
                pid_file=pid_file,
                sentinel_path=sentinel_path,
                attempt_id=attempt_id,
                # #1669: thread the launch env pins into the rendered
                # launcher — this call-site kwarg is what makes a failover
                # re-execution (reconstructed spec.extra carries env_pins)
                # actually re-export them on the fresh pod.
                env_pins=(spec.extra or {}).get("env_pins"),
            ),
            timeout=120,
            context=f"workload detach on {pod_name}",
        )
    except RunPodWorkloadStartError as exc:
        if "ALREADY-RUNNING" in str(exc):
            pid_match = _ALREADY_RUNNING_PID_RE.search(str(exc))
            live_pid = pid_match.group(1) if pid_match else "unknown"
            raise RunPodWorkloadStartError(
                f"double-fire guard: a live workload (pid={live_pid}) already holds "
                f"{pid_file} on {pod_name} — the experimenter (SKILL.md Step 6d.1) or a "
                "prior --execute-workload launch already started it; refusing to "
                f"double-launch. Original: {exc}"
            ) from exc
        raise
    if "WRAPPER-STARTED" not in launch_out:
        raise RunPodWorkloadStartError(
            f"workload detach on {pod_name} did not confirm WRAPPER-STARTED "
            f"(log {log_path}); output tail: {launch_out[-1500:]!r}"
        )

    # (c) Verify from a SEPARATE SSH invocation — a same-session probe
    # cannot catch SIGHUP-on-disconnect death.
    time.sleep(WORKLOAD_VERIFY_DELAY_SECONDS)
    try:
        verify_out = _ssh_pod_run(
            host,
            port,
            _render_verify_script(issue=issue, log_path=log_path, pid_file=pid_file),
            timeout=120,
            context=f"workload liveness verify on {pod_name}",
        )
    except RunPodWorkloadStartError as exc:
        raise RunPodWorkloadStartError(
            f"workload did NOT verify alive on {pod_name} (log {log_path}): {exc} — "
            "if the workload self-daemonizes without writing a pod-side pidfile, check "
            "the pod before treating this as dead; the pod is left RUNNING for diagnosis"
        ) from exc
    ok = _LAUNCH_OK_RE.search(verify_out)
    if not ok:
        raise RunPodWorkloadStartError(
            f"workload did NOT verify alive on {pod_name} (log {log_path}); "
            f"verify output tail: {verify_out[-1500:]!r} — if the workload "
            "self-daemonizes without writing a pod-side pidfile, check the pod before "
            "treating this as dead; the pod is left RUNNING for diagnosis"
        )
    return {
        "workload_pid": int(ok.group(1)),
        "launcher_path": _launcher_path(issue),
        "synced_sha": synced_sha,
    }


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

        CPU-intent failover (#747): a cheap CPU intent (``cpu-small`` /
        ``cpu-mid``, the router-mapped ``RUNPOD_CPU_INSTANCE_FOR_INTENT`` keys
        that fall over GCP→RunPod) is passed through verbatim as ``--intent
        cpu-small`` — this backend stays thin. ``pod_lifecycle.py cmd_provision``
        resolves it to a RunPod CPU instance_id via
        ``gpu_heuristics.resolve_cpu_intent`` (checked BEFORE the GPU
        ``_resolve_spec``) and provisions via ``runpod_api.create_cpu_pod``
        (``deployCpuPod``); ``cpu-bigmem`` never reaches here on any AUTOMATED
        path (the #677 typed terminal at the router's terminal rung precedes
        launch — it is absent from the RunPod-CPU map). An explicit
        ``backend: runpod`` pin of ``cpu-bigmem`` DOES reach launch
        (``_override_runpod`` has no CPU guard) and then fails loud downstream
        (``resolve_cpu_intent`` -> None and the GPU ``_resolve_spec`` cannot
        resolve it -> non-zero provision exit).
        """
        execute_workload = bool((spec.extra or {}).get("execute_workload"))
        if execute_workload and not spec.workload_cmd:
            # Defensive in-backend guard behind the dispatch CLI's parse-time
            # rejection (#909 AC3a): a PROGRAMMATIC caller must not silently
            # recreate the flag+hydra false-green cell (the execution leg
            # cannot execute a hydra-args run). Raised BEFORE the provision
            # subprocess so no pod is paid for.
            raise RunPodWorkloadStartError(
                "execute_workload requested with empty workload_cmd — the RunPod "
                "execution leg cannot execute a hydra-args run (#909); refusing "
                "before provisioning"
            )
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
        # #1010/#1118: thread the plan's disk requirement into the provision
        # argv. CPU lane (#1010): the pod's only writable disk is the
        # container overlay (/workspace rides it; incident #958) --
        # --container-disk-gb, floored at runpod_api.DEFAULT_CONTAINER_DISK_GB
        # (50, _CPU_CONTAINER_DISK_FLOOR_GB here) so threading can never
        # REDUCE below today's behavior. The router's feasibility gate
        # guarantees boot_disk_gb <= the instance cap on every AUTOMATED
        # path; an explicit `backend: runpod` pin above the cap fails loud at
        # pod_lifecycle's pre-API cap check / RunPod's own create-time
        # validation. GPU lane (#1118): the big-data mount is the /workspace
        # VOLUME (pod_lifecycle threads --volume-gb -> runpod_api volumeInGb),
        # floored at the 200 GB argparse default (_GPU_VOLUME_FLOOR_GB) --
        # thread-or-grow, never shrink. No deterministic pre-API cap exists
        # for GPU volumeInGb (unlike the probe-verified CPU caps) -- an
        # unsatisfiable size surfaces LOUD at RunPod create time (RunPodError
        # -> non-zero provision exit -> CalledProcessError) or as a capacity
        # miss (wait-for-capacity budget), never as a silent downsize (the
        # #1112 ENOSPC incident: a ~575 GB plan on the default 200 GB volume).
        cmd += _boot_disk_provision_args(spec)
        # #1698: plumb `spec.extra["repo_branch"]` into the provision
        # subprocess env as BOOTSTRAP_BRANCH so `bootstrap_pod.sh:52` picks
        # it up (``BOOTSTRAP_BRANCH="${BOOTSTRAP_BRANCH:-main}"``). Without
        # this thread, the argv for `pod_lifecycle.py provision` carries no
        # `--repo-branch` term (`pod_lifecycle.py` has no such argparse arg
        # — verified 2026-07-26) and `_bootstrap()` at :744-759 only pins
        # `POD_INTENT`, so the env var is unset in the subprocess and the
        # bash default lands every launch on `main` even when the caller
        # requested a specific branch. The #1689 R8/R9 pods bootstrapped
        # onto `main` twice through exactly this drop.
        #
        # DESIGN CHOICE (concern #4). The #1669 `env_pins` plumbing threads
        # workload-env pins (WANDB_PROJECT et al., see
        # `backends/base.py::ENV_PIN_ALLOWED_KEYS`) into the RENDERED launcher
        # via `_render_launch_script` — a different subprocess boundary that
        # fires only when the execution-leg opts in (`execute_workload`) and
        # scoped to the workload's env. `BOOTSTRAP_BRANCH` gates the
        # PRE-workload bootstrap subprocess (`pod_lifecycle.py provision` ->
        # `bootstrap_pod.sh`), which runs UNCONDITIONALLY on every provision
        # regardless of the execution flag. Piggybacking on `env_pins` would
        # (a) require extending the allowlist to a key that is provisioning-
        # scoped (mixing two orthogonal env-pin scopes), and (b) tie the
        # pin's flow to the `--execute-workload` opt-in when it applies
        # unconditionally to every provision. Keep a distinct
        # `env_for_provision` copy: mirrors the `teardown` shape at :1545
        # (`env=os.environ.copy()`), one localized change, no allowlist
        # surface widening.
        repo_branch_env = str((spec.extra or {}).get("repo_branch") or "").strip()
        env_for_provision = os.environ.copy()
        if repo_branch_env:
            env_for_provision["BOOTSTRAP_BRANCH"] = repo_branch_env
        # _run_pod_lifecycle_relay raises PodLifecycleProcessError (a
        # CalledProcessError subclass carrying the child's stderr tail,
        # #1465) on non-zero exit; that propagates to the selector, which
        # logs + lets the orchestrator surface the failure as `epm:failure`
        # with the diagnostics inline. The exit-75 still-waiting contract
        # (`dispatch_issue._provision_still_waiting`) is unchanged —
        # returncode + cmd ride verbatim. (Slice 1 does NOT add a provision
        # retry — the existing `--wait-for-capacity` retry inside
        # `pod_lifecycle.py` already handles SUPPLY_CONSTRAINT.)
        _run_pod_lifecycle_relay(cmd, env=env_for_provision)
        # #1698 Item 1(b) — fail-loud post-bootstrap branch assertion. Bind
        # ONLY when a specific non-`main` branch was requested: the default
        # case (empty / explicit `main`) is a no-op so a launch that
        # legitimately wants `main` is unaffected. `RunPodProvisionBranchMismatchError`
        # subclasses `RunPodWorkloadStartError`, so the :1124 catch at
        # `_execute_workload_on_pod`'s call site would wrap it into a #954
        # partial-handle failure IF the execution leg is opted in. On the
        # non-execute path (the default `/issue` Step 6d.1 flow) the
        # exception propagates verbatim — the pod stays RUNNING for SSH
        # diagnosis per the RunPod-as-diagnosis-lane doctrine.
        pod_name = _runpod_pod_name(spec.issue)
        # #2038: round-trip the exact RunPod pod id the provision just
        # persisted to pods_ephemeral.json (closing the "a future revision
        # should round-trip pods_ephemeral.json" gap noted at the job_id
        # construction below). Captured BEFORE the post-provision try block
        # so the emergency-teardown arm can terminate by EXACT id; best-effort
        # (None on any read failure — the id-less pre-#2038 shape).
        provisioned_pod_id = _provisioned_pod_id(pod_name)
        # #2038: True once _execute_workload_on_pod returns — a post-start
        # failure must NEVER terminate a pod whose workload is running.
        workload_started = False
        # #2038: post-provision protective wrapper. From here to the handle
        # return, the pod EXISTS and BILLS but no handle / sidecar / lease
        # records it yet — an escaping exception in this window strands an
        # invisible billing pod (#1739: a failed fallback launch left pod-1739
        # running with no record). RunPodWorkloadStartError (incl. the
        # RunPodProvisionBranchMismatchError subclass) keeps its #954
        # diagnosis-lane contract BYTE-UNCHANGED: the pod stays RUNNING for
        # SSH diagnosis (the #1997 watcher bounds that window). Any OTHER
        # exception, when the workload has NOT started, best-effort terminates
        # the just-provisioned pod by EXACT id, then re-raises the ORIGINAL.
        try:
            if repo_branch_env and repo_branch_env != "main":
                _assert_pod_on_branch(
                    pod_name=pod_name,
                    expected_branch=repo_branch_env,
                )
            # Attempt id + sentinel path minted BEFORE the execution leg (#909 r2,
            # `runpod-execute-missing-completion-sentinel`): the handle's
            # expected-artifacts declaration and the launcher's chained
            # completion-sentinel write MUST share ONE attempt-namespaced path —
            # one mint, one path, both sides. (Round 1 minted the id AFTER
            # `_execute_workload_on_pod`, so the declared path could not be
            # threaded into the launcher, no writer existed on the
            # no-experimenter leg, and every successful backend-executed run
            # would FAIL finalize — `_check_sentinel` FAILs a missing sentinel
            # and `_cmd_finalize` exits 3 + skips teardown when a declaration
            # is present but unsatisfied.)
            attempt_id = mint_runpod_attempt_id()
            sentinel_path = runpod_sentinel_path(spec.issue, attempt_id)
            # Execution leg (#909): execute iff workload_cmd is non-empty AND the
            # caller opted in via spec.extra["execute_workload"] (set automatically
            # by router.failover_to_runpod_after_async_workload_crash — the
            # no-experimenter automated failover paths — or explicitly via
            # `dispatch_issue.py launch --execute-workload`). The interactive
            # /issue Step 6b/6d.1 flow passes no flag: the experimenter stays the
            # sole executor there, so this branch is behavior-unchanged for it.
            exec_requested = execute_workload and bool(spec.workload_cmd)

            # Handle construction shared by the SUCCESS path and the #954
            # PARTIAL-failure path (pod provisioned, workload start failed).
            # Every input — pod_name, attempt_id, sentinel_path, the spec fields —
            # is minted BEFORE the execution leg (#909 r2), so the handle is fully
            # constructible at the failure point. The import is hoisted above the
            # execution leg for the same reason (the failure path needs it).
            from explore_persona_space.backends.artifacts import (
                EXPECTED_ARTIFACTS_HANDLE_KEY,
                build_expected_artifacts_declaration,
            )

            def _build_handle(
                workload_info: dict[str, Any],
                *,
                workload_executed: bool,
                workload_start_error: str | None = None,
            ) -> RunHandle:
                """Build the launch :class:`RunHandle`.

                Success path: ``workload_executed=exec_requested`` + the execution
                leg's ``workload_info`` — the ``extra`` dict is byte-identical to
                the pre-#954 inline construction (``workload_start_error`` is added
                ONLY on the failure path, so no new keys appear on success). #1118
                adds the CONDITIONAL footprint keys (``boot_disk_gb`` /
                ``min_ram_gb``) on both paths, OMITTED when absent/falsy — a spec
                without a stated footprint keeps the pre-#1118 key set.
                Failure path (#954): ``workload_executed=False`` (truthful — the
                workload did not start) + a truncated ``workload_start_error`` so
                downstream consumers (poll / finalize / re-drive) can tell the
                partial launch apart from a healthy one.
                """
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
                # * ``workload_cmd`` / ``hydra_args`` / ``gpus`` /
                #   ``time_budget_hours`` — the relaunch-critical RunSpec fields
                #   (#689 blocker, mirroring the GCP handle contract). The RunPod
                #   RUNNING-but-no-port wedge failover (``backend_poll`` /
                #   ``.claude/rules/compute-backend-failover.md`` § Part C)
                #   reconstructs a ``RunSpec`` FROM the persisted sidecar handle via
                #   ``_runspec_from_runpod_handle`` to re-provision a FRESH pod. That
                #   reconstruction reads exactly these keys off ``extra`` and FAILS
                #   LOUD when neither ``workload_cmd`` NOR ``hydra_args`` is present —
                #   so a launch that did not persist them would terminate the wedged
                #   pod and then orphan the run (no fresh pod). Persisting them here
                #   makes the spec reconstructable; ``serialize_handle`` /
                #   ``deserialize_handle`` round-trip ``extra`` verbatim (the tuple
                #   ``hydra_args`` JSON-encodes to a list, which the reconstructor
                #   re-tuples). ``workload_cmd`` is ``""`` for a Hydra-entrypoint run
                #   and ``hydra_args`` is ``()`` for a custom-workload run; at least
                #   one is always set on a real launch (the ``RunSpec.__post_init__``
                #   mutual-exclusion contract).
                extra: dict[str, Any] = {
                    "intent": spec.intent,
                    "issue": int(spec.issue),
                    "pid_file": _runpod_pid_file_path(spec.issue),
                    "runpod_attempt_id": attempt_id,
                    # Relaunch-critical RunSpec fields for the wedge failover (#689).
                    "workload_cmd": spec.workload_cmd,
                    "hydra_args": list(spec.hydra_args),
                    "gpus": spec.gpus,
                    "time_budget_hours": spec.time_budget_hours,
                    # #1118: footprint fields persisted so the wedge / CUDA-IMA
                    # fresh-pod re-provision (backend_poll._runspec_from_runpod_handle)
                    # forwards them — mirroring the GCP handle (gcp.py, #1010).
                    # Keys OMITTED when absent/falsy — never a None value — so
                    # legacy handle shapes stay byte-identical (the
                    # _PRE_954_SUCCESS_EXTRA_KEYS exact-set tests pin this).
                    **{
                        k: v
                        for k, v in {
                            "boot_disk_gb": (spec.extra or {}).get("boot_disk_gb"),
                            "min_ram_gb": (spec.extra or {}).get("min_ram_gb"),
                            # #1669: launch env pins — persisted so the wedge /
                            # CUDA-IMA fresh-pod re-provision forwards them and
                            # the fresh launcher re-exports them (#1586).
                            "env_pins": (spec.extra or {}).get("env_pins"),
                            # #2038: the exact RunPod pod id round-tripped from
                            # pods_ephemeral.json post-provision — the
                            # superseded-fallback reap (issue_dispatch) keys its
                            # exact-id disposition on it. Omit-when-absent: a
                            # failed read keeps the legacy id-less shape.
                            "pod_id": provisioned_pod_id,
                        }.items()
                        if v
                    },
                    # #909: the branch the run's code lives on (round-trips through
                    # the sidecar + backend_poll reconstructors so a failover
                    # re-execution syncs the ISSUE branch, not `main`) + the
                    # execution-leg outcome (workload_executed / workload_pid /
                    # launcher_path / synced_sha via **workload_info). Additive
                    # keys — every existing reader uses .get(...).
                    "repo_branch": str((spec.extra or {}).get("repo_branch") or ""),
                    "workload_executed": workload_executed,
                    **workload_info,
                    EXPECTED_ARTIFACTS_HANDLE_KEY: build_expected_artifacts_declaration(
                        issue=spec.issue,
                        # The SAME path threaded into the execution leg —
                        # one mint, one path, both sides (#909 r2).
                        sentinel_path=sentinel_path,
                        custom_workload=True,
                        attempt_id=attempt_id,
                        wandb_run_path=spec.extra.get("wandb_run_path"),
                        # #685 / #661: thread the per-issue worktree git root +
                        # the phase-scope flag off spec.extra (the same channel
                        # as wandb_run_path; _launch_extra_from_args populates
                        # both). None / False (absent) = established behavior.
                        git_repo_root=spec.extra.get("git_repo_root"),
                        skip_default_git_paths=bool(
                            spec.extra.get("skip_default_git_paths", False)
                        ),
                    ),
                }
                if workload_start_error is not None:
                    # #954 failure-path-only key: the truncated start-leg error, so
                    # the sidecar records WHY the pod is workload-less.
                    extra["workload_start_error"] = workload_start_error
                return RunHandle(
                    backend="runpod",
                    cluster=None,
                    # The RunPod pod_id is set inside pod_lifecycle.py and persisted
                    # to pods_ephemeral.json; #2038 rounds it back into
                    # extra["pod_id"] (omit-when-absent — see the extra dict above).
                    # job_id stays "" DELIBERATELY: the #1122 carry-forward treats
                    # an empty job_id as no-match, and every RunPod reader routes by
                    # name through SSH config — flipping job_id to the pod id would
                    # silently change those identity-binding semantics.
                    job_id="",
                    pod_name=pod_name,
                    scratch_dir="/workspace",
                    log_path=_runpod_log_path(spec.issue),
                    extra=extra,
                )

            workload_info: dict[str, Any] = {}
            if exec_requested:
                try:
                    workload_info = _execute_workload_on_pod(
                        spec,
                        pod_name=pod_name,
                        log_path=_runpod_log_path(spec.issue),
                        pid_file=_runpod_pid_file_path(spec.issue),
                        sentinel_path=sentinel_path,
                        attempt_id=attempt_id,
                    )
                except RunPodWorkloadStartError as exc:
                    # #954: the pod IS provisioned (and bills) — attach the fully-
                    # built partial handle to the typed error so the router's
                    # terminal rung + the backend_poll failover legs can persist
                    # the launch records (sidecar + lease) before surfacing the
                    # failure. Re-raising the SAME exception preserves message +
                    # traceback. The pod-stays-RUNNING-for-diagnosis contract is
                    # UNCHANGED.
                    exc.handle = _build_handle(
                        {},
                        workload_executed=False,
                        workload_start_error=str(exc)[:2000],
                    )
                    raise
                else:
                    # #2038: the workload is now RUNNING on the pod — the outer
                    # emergency-teardown arm must never terminate past this point.
                    workload_started = True
            elif spec.workload_cmd:
                logger.warning(
                    "workload_cmd persisted but NOT executed — EXPECTED when the "
                    "experimenter (SKILL.md Step 6d.1) launches it on this pod; otherwise "
                    "dispatch the experimenter on THIS pod (preferred — a re-launch "
                    "provisions a SECOND pod), or re-launch with --execute-workload (#909)"
                )
            # Expected-artifacts declaration (#598): the attempt id minted above
            # (GCP-style, pre-execution) is embedded in the pod-side sentinel path
            # so a prior attempt's sentinel on the persistent /workspace
            # volume can never satisfy this launch's declaration. The sentinel
            # WRITER depends on the executor: experimenter-driven dispatches
            # chain `write_completion_sentinel` per experimenter.md step 11;
            # the #909 backend-executed leg chains the write inside its rendered
            # launcher (same path — see the mint comment above). The declaration
            # carries NO launch-time HF prefix guess (the #601
            # false-negative-teardown trap, a fortiori on this lane).
            return _build_handle(workload_info, workload_executed=exec_requested)
        except RunPodWorkloadStartError:
            # #954 diagnosis-lane path — behavior unchanged (plan §3/§5): the
            # partial handle (when the exec leg attached one) rides the typed
            # error to the router terminal rung; the pod stays RUNNING for SSH
            # diagnosis per the RunPod-as-diagnosis-lane doctrine.
            raise
        except Exception as exc:
            _dispose_post_provision_failure(
                exc=exc,
                workload_started=workload_started,
                pod_id=provisioned_pod_id,
                pod_name=pod_name,
                issue=spec.issue,
            )
            raise

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
        #
        # ``stall_reason`` (#664) MUST be copied through: ``poll_once`` runs
        # the zombie-GPU-allocation probe and sets
        # ``stall_reason="vllm_worker_dead_zombie_gpu"`` when it overrides a
        # masked ``running`` verdict to ``stalled`` (a dead CUDA-worker PID
        # still holding VRAM while the EngineCore main process keeps the
        # session-CPU-advancing override alive). RunPod is the ONLY lane that
        # produces this value, and this rewrap is the slice-6 path's only seam
        # to ``poll_once``; dropping it here silently strips the distinguishing
        # reason before ``backend_poll._serialize_poll_result`` can surface it
        # (that serializer already reads ``stall_reason`` via ``getattr``, so
        # the whole chain is wired EXCEPT this copy). The detection itself is
        # NOT re-implemented here — it lives in ``poll_once``; this line is the
        # missing passthrough, not a second probe.
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
            next_interval=raw.next_interval,
            stall_reason=raw.stall_reason,
            # #775: copy the WIDE-tail crash signature through (same passthrough
            # contract as stall_reason above). poll_once populates it only on a
            # dead poll; backend_poll._maybe_escalate_runpod_cuda_ima reads it.
            # getattr-guarded so a mixed-version worktree (a poll_once that
            # predates the field) degrades to None rather than crashing.
            crash_signature=getattr(raw, "crash_signature", None),
            # #983: copy the post-done phase-consistency advisory surfaces
            # through (same passthrough contract as stall_reason above —
            # dropping them at this rewrap would silently strip the advisory
            # from the backend_poll lane's JSON before
            # ``_serialize_poll_result`` can surface it, the #664
            # stall_reason lesson). getattr-guarded for a mixed-version
            # worktree; the marker + Telegram push already fired inside
            # ``poll_once`` regardless.
            post_done_phase_advisory_posted=getattr(raw, "post_done_phase_advisory_posted", False),
            post_done_phase_lines=tuple(getattr(raw, "post_done_phase_lines", ()) or ()),
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
                encoding="utf-8",
                errors="replace",
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
                encoding="utf-8",
                errors="replace",
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

    def _ssh_glob_sentinels(self, handle: RunHandle) -> Callable[[str, int], list[str]]:
        """Build a remote sibling-sentinel glob bound to ``handle``'s pod (#709).

        SSH sibling of :meth:`_ssh_read_sentinel` and of
        ``artifacts._default_glob_sentinels``: enumerates the attempt-dir
        sibling sentinels ``<issue_dir>/*/<name>`` on the POD filesystem so
        ``_resolve_live_sentinel`` can resolve the #685-secondary stale-baked-
        attempt case on the live-pod path. The resolution's soundness leans on
        the #976 pre-workload stale-clear contract (the same ``issue_dir/*/
        <name>`` operand): a launch path that baked attempt-namespaced
        declarations WITHOUT the pre-workload ``rm`` would widen the
        single-live-sibling window to a prior attempt's stale sentinel.
        Semantics mirror the reader:

        * rc=0            -> stdout lines, stripped, sorted (parity with the
                             FS default's ``sorted(...)``).
        * rc!=0 + "no such file" in stderr -> [] (bash passes an unmatched
                             glob literally; ``ls`` errors "No such file or
                             directory" -> zero siblings).
        * any other rc    -> raise. A transport failure must NOT read as
                             "no siblings"; the resolver's existing probe
                             try/except turns the raise into the honest
                             declared-missing FAIL with a note.
        """

        def glob(declared: str, issue: int) -> list[str]:
            del issue  # parity with _default_glob_sentinels: encoded in the path.
            parts = declared.rsplit("/", 2)
            if len(parts) != 3:
                return []  # non-canonical; resolver scope guard blocks this upstream.
            issue_dir, _attempt, name = parts
            remote_cmd = f"ls -1d {_shell_quote(issue_dir)}/*/{_shell_quote(name)}"
            proc = subprocess.run(
                ["ssh", handle.pod_name, remote_cmd],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            if proc.returncode == 0:
                return sorted(ln.strip() for ln in proc.stdout.split("\n") if ln.strip())
            stderr = (proc.stderr or "").lower()
            if "no such file" in stderr:
                return []
            raise RuntimeError(
                f"ssh sentinel glob on {handle.pod_name} failed "
                f"rc={proc.returncode}: {(proc.stderr or '')[:300]}"
            )

        return glob

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
        :meth:`_ssh_read_sentinel` (#598), and the stale-baked-attempt
        sibling probe now also runs pod-side via
        :meth:`_ssh_glob_sentinels` (#709); HF / WandB / git checks keep
        their default wires.
        """
        # Lazy import to keep the runpod module importable without the
        # artifacts module's optional deps loaded yet.
        from explore_persona_space.backends.artifacts import (
            VerifierIO,
            confirm_artifacts_from_handle,
        )

        verdict = confirm_artifacts_from_handle(
            handle,
            io=VerifierIO(
                read_sentinel=self._ssh_read_sentinel(handle),
                glob_sentinels=self._ssh_glob_sentinels(handle),
            ),
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
        # #1698 Item 2 — idempotent teardown for an already-terminated pod.
        # When no live pod matches the issue AND no local sidecar record is
        # left, `pod_lifecycle._terminate_clear_stale_sidecar` at
        # `scripts/pod_lifecycle.py:2911` raises
        # `SystemExit("No live pod found for issue <N> (and no local
        # record). Nothing to terminate.")` (verified 2026-07-26: exactly 1
        # emission site in `pod_lifecycle.py`). The subprocess exits rc=1
        # and `_run_pod_lifecycle_relay:200` wraps it as
        # `PodLifecycleProcessError`, which pre-#1698 propagated all the
        # way up to `dispatch_issue._cmd_finalize`, exited non-zero, and
        # left the handle sidecar UNRENAMED — the #1689 recovery required
        # hand-`mv`ing `.claude/cache/issue-1689-handle.json` to
        # `…finalized` twice.
        #
        # Catch that specific stderr signature and treat as idempotent
        # success: the goal ("this pod is gone") already holds by
        # construction. Every OTHER `PodLifecycleProcessError` re-raises
        # verbatim (auth error, RunPod API 5xx, pod-exists-but-terminate-
        # refused, post-terminate live-API survivors, #1485
        # `keep-running`-tag refusal) so the fail-loud contract for real
        # terminate failures is UNCHANGED. `_cmd_finalize`'s
        # `<name>.finalized` sidecar rename in `dispatch_issue.py` then
        # executes cleanly on the already-gone case.
        try:
            _run_pod_lifecycle_relay(cmd, env=os.environ.copy())
        except PodLifecycleProcessError as exc:
            stderr_tail = (exc.stderr or "").lower()
            if "nothing to terminate" not in stderr_tail:
                raise
            logger.info(
                "RunPodBackend.teardown: pod-%s already gone (matched "
                "'Nothing to terminate' in pod_lifecycle stderr); treating "
                "as idempotent success so finalize can retire the sidecar "
                "(#1698 Item 2).",
                issue,
            )
