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

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


class RunPodWorkloadStartError(RuntimeError):
    """A requested ``--workload-cmd`` execution did not start on the pod (#909).

    Raised by the RunPod execution leg on ANY start failure — missing
    pods.conf row, branch-sync mismatch, double-fire guard, dead PID with
    no fresh pidfile, missing log. ``scripts/dispatch_issue.py launch``
    surfaces it as a ``reason: runpod_workload_start_failed`` failure JSON
    + exit 2 — a requested execution that did not start NEVER returns ok.
    The pod is left RUNNING for SSH diagnosis (the RunPod-as-diagnosis-lane
    doctrine, ``.claude/rules/compute-backend-failover.md``).
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
            f'git fetch origin "refs/heads/{branch}"',
            f'git checkout -q -f -B "{branch}" FETCH_HEAD',
            "git reset --hard -q FETCH_HEAD",
            "HEAD_SHA=$(git rev-parse HEAD); FETCH_SHA=$(git rev-parse FETCH_HEAD)",
            '[ "$HEAD_SHA" = "$FETCH_SHA" ] || '
            '{ echo "SYNC-MISMATCH head=$HEAD_SHA fetch=$FETCH_SHA" >&2; exit 3; }',
            'echo "SYNC-OK $HEAD_SHA"',
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
    terminal block). The outer portion clears any stale sentinel at the
    declared path before detach (same guard family as the pidfile rm),
    and the launcher exits with the workload's own rc so the exit status
    is unchanged by the chain.
    """
    launcher = _launcher_path(issue)
    epoch_file = _launch_epoch_path(issue)
    sentinel_dir = sentinel_path.rsplit("/", 1)[0]
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
            f"rm -f {sentinel_path}",
            f"mkdir -p {sentinel_dir}",
            f"date +%s > {epoch_file}",
            f"cat > {launcher} << 'EPSEOF'",
            "#!/bin/bash",
            "set -uo pipefail",
            'export PATH="/root/.local/bin:$PATH"',
            "cd /workspace/explore-persona-space",
            "set -a; [ -f .env ] && source .env; set +a",
            'export REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"',
            f'export WANDB_PROJECT="${{WANDB_PROJECT:-issue{issue}}}"',
            f"echo $$ > {pid_file}",
            workload_cmd,
            "WORKLOAD_RC=$?",
            "# Completion sentinel: written ONLY when the workload exited 0 (the",
            "# backend-owned twin of the GCP/SLURM terminal sentinel write).",
            'if [ "$WORKLOAD_RC" -eq 0 ]; then',
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
    # HEAD == FETCH_HEAD verification (never trusting pull stdout).
    sync_out = _ssh_pod_run(
        host,
        port,
        _render_branch_sync_script(branch),
        timeout=180,
        context=f"branch sync of {pod_name} to {branch!r}",
    )
    sync_match = re.search(r"SYNC-OK ([0-9a-f]+)", sync_out)
    if not sync_match:
        raise RunPodWorkloadStartError(
            f"branch sync of {pod_name} to {branch!r} did not confirm SYNC-OK; "
            f"output tail: {sync_out[-1500:]!r}"
        )
    synced_sha = sync_match.group(1)

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
        (``deployCpuPod``); ``cpu-bigmem`` never reaches here (it keeps the #677
        typed terminal — it is absent from the RunPod-CPU map).
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
        # subprocess.run raises CalledProcessError on non-zero exit; that
        # propagates to the selector, which logs + lets the orchestrator
        # surface the failure as `epm:failure` (slice 1 does NOT add a
        # provision retry — the existing `--wait-for-capacity` retry inside
        # `pod_lifecycle.py` already handles SUPPLY_CONSTRAINT).
        subprocess.run(cmd, check=True)
        pod_name = _runpod_pod_name(spec.issue)
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
            ONLY on the failure path, so no new keys appear on success).
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
                    skip_default_git_paths=bool(spec.extra.get("skip_default_git_paths", False)),
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
                # to pods_ephemeral.json; we read it back from there rather than
                # parsing stdout. For slice 1 the orchestrator does not need the
                # raw pod_id (it routes by name through SSH config) — empty
                # string is the truthful "we did not capture this here" marker;
                # a future revision should round-trip pods_ephemeral.json.
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
