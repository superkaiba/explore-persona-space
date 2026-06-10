#!/usr/bin/env python3
"""``/issue`` operational dispatch CLI — the bridge SKILL.md Step 6b/8 invoke.

The slice-5 router and slice-6 dispatch helper are fully testable in
isolation; SKILL.md is PROSE the orchestrator executes, so it cannot
construct production backends inline. This script is the THIN
operational seam between the two: it builds the production backends +
injected dependencies, calls :func:`backends.issue_dispatch.dispatch_for_issue`
(``launch`` action) or :func:`backends.runpod.RunPodBackend.teardown`-equivalent
through the backend handle (``finalize`` action), and converts the
router's typed terminals into the ``epm:failure v1`` notes the
orchestrator's failure-classifier already routes on.

Why this script exists
----------------------

Before slice 6 the SKILL.md Step 6b operational block ran
``pod.py provision`` unconditionally — the slice-6 router code shipped
but the operational path NEVER invoked it. Same for Step 8's
``pod.py terminate``. So an explicit ``backend: nibi`` task silently
provisioned a RunPod pod; the sidecar JSON the bg-Bash poller reads
was never written (``backend_poll.py`` FileNotFoundError'd every
tick); GCP credits were never reachable from ``/issue``.

This CLI is the dispatcher SKILL.md actually calls. The orchestrator
shells:

    uv run python scripts/dispatch_issue.py launch \
        --issue <N> --intent <intent> [--backend <override>] [--hydra k=v]...

    uv run python scripts/dispatch_issue.py finalize --issue <N>

and parses the JSON line printed on stdout. Every backend launch /
poll / teardown flows through the same RunHandle the bg-Bash poller
recovers from the sidecar — RunPod included (its launch shells the
existing ``pod_lifecycle.py`` underneath, but the sidecar is written
uniformly so Step 6d.2 / Step 8 don't branch per backend).

Exit codes
----------

* ``0`` — launch/finalize succeeded. ``stdout`` carries one JSON line
  with the resolved outcome (``chosen_kind`` / ``handle_sidecar_path``
  / ``failure_class`` / ``status``).
* ``2`` — router terminal (``NoComputeAvailableError`` /
  ``WorkloadSurfacedError`` / ``GcpAttemptCapExceededError`` /
  ``ManualAttentionRequiredError``). ``stdout`` carries the
  ``failure_class`` + ``status`` + ``note`` from
  ``classify_terminal_exception`` so the orchestrator can post
  ``epm:failure v1`` + ``set-status blocked`` without re-deriving the
  classification.
* ``3`` — confirm_artifacts FAIL on the ``finalize`` path
  (artifacts not landed; teardown SKIPPED to preserve evidence).
  ``stdout`` carries the per-check reasons.
* ``4`` — unexpected exception. ``stderr`` carries the traceback.

Bg-Bash contract preservation
-----------------------------

This script does NOT poll. Polling stays the orchestrator's bg-Bash
``scripts/backend_poll.py`` job, which reads the per-issue sidecar
written here. The two scripts are paired: ``dispatch_issue.py launch``
writes the sidecar; ``scripts/backend_poll.py`` reads it tick after
tick.

References
----------

* :mod:`explore_persona_space.backends.issue_dispatch` — the production
  dispatch helper this CLI wraps.
* :mod:`explore_persona_space.backends.router` — the underlying
  decision engine + terminal exception classes.
* ``.claude/skills/issue/SKILL.md`` Steps 6b / 6d / 8 — the
  orchestrator steps that shell this CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _build_production_backends() -> dict[str, Any]:
    """Construct the production ComputeBackend instances + injected deps.

    Centralised so ``launch`` and ``finalize`` share the same wiring
    (a divergence would split the routing decision from the teardown
    decision — exactly the bug this CLI exists to close).

    Returns a dict with:

    * ``runpod_backend`` — :class:`RunPodBackend` (shells to
      ``pod_lifecycle.py`` for launch + terminate so the sidecar is
      written uniformly across backends).
    * ``free_backends`` — ``{"nibi": SlurmBackend(), "fir": ...}`` for
      every cluster whose ``ClusterConfig.available`` is True. Fir is
      flagged ``available=False`` until v1.1; the router silently
      skips an absent kind so dropping it here is harmless.
    * ``gcp_backend`` — :class:`GcpBackend` (the credit-backed
      escalation target).
    * ``marker_poster`` — :func:`backends.slurm.post_marker_via_task_py`
      (the same shell-out the SLURM monitor uses; calls
      ``task.py post-marker`` under the workflow flock).
    * ``is_started`` — SLURM-aware ``squeue -j``-status==RUNNING probe.
      Required because ``SLURM_STATE_TO_STATUS`` maps PENDING→running
      for the legacy poll-result-status enum; the router needs to
      DISTINGUISH PENDING from RUNNING for its park watchdog.
    * ``is_live_after_cancel`` — by-name squeue probe ("still in the
      live queue?" — non-empty = yes). DRAC robots have no ``sacct``
      so this is the only authoritative still-live signal.
    * ``started_evidence_probe`` — scratch-dir runtime-artifact probe
      (rsync read of ``status.json`` / ``job.out``) the router consults
      on a terminal-before-running park outcome to classify
      "started-then-FAILED" as a workload failure instead of
      ``no_compute_available`` (which would wrongly escalate a doomed
      workload to GCP on the auto lane).
    * ``reconnect_fn`` — per-kind reconnect dispatch (SLURM:
      ``query_by_name``; GCP: :func:`backends.gcp.reconnect_or_none`).
    * ``mila_socket_alive`` — :func:`backends.slurm.mila_socket_alive`,
      the real ``ssh -o BatchMode=yes mila true`` probe over the 12 h
      email-OTP ControlMaster socket. Returns ``False`` on socket-down
      (skip-the-lane, NOT an error) and never raises in production —
      the router treats that as "Mila not available right now."
    """
    # Lazy imports — keeps the --help path fast and avoids dragging in
    # SSH / gcloud helpers when the CLI is run for a non-launch action.
    from explore_persona_space.backends.gcp import (
        GcpBackend,
    )
    from explore_persona_space.backends.gcp import (
        reconnect_or_none as gcp_reconnect_or_none,
    )
    from explore_persona_space.backends.runpod import RunPodBackend
    from explore_persona_space.backends.slurm import (
        CLUSTER_CONFIGS,
        SlurmBackend,
        post_marker_via_task_py,
    )
    from explore_persona_space.backends.slurm_monitor import (
        query_by_name,
        query_slurm_state,
    )

    # Build the free-lane map from CLUSTER_CONFIGS, skipping clusters
    # whose ``available`` flag is False (Fir in v1). A single shared
    # SlurmBackend instance suffices: its per-call ``_cluster_for_spec``
    # resolves the cluster from ``handle.cluster`` / ``spec.cluster``.
    slurm = SlurmBackend()
    free_backends: dict[str, Any] = {}
    for name, cfg in CLUSTER_CONFIGS.items():
        if not cfg.available:
            continue
        free_backends[name] = slurm

    runpod_backend = RunPodBackend()
    gcp_backend = GcpBackend()

    def _slurm_is_started(backend: Any, handle: Any) -> bool:
        """``squeue -j <id>`` status RUNNING (else PENDING/other = not started).

        The router's ``default_is_started`` falls back to
        ``backend.poll().status == "running"`` which is wrong for SLURM:
        ``SLURM_STATE_TO_STATUS`` maps PENDING→"running" for the
        orchestrator's legacy enum, so the park watchdog would
        immediately think a PENDING job was RUNNING and skip the wait.
        """
        cluster = _resolve_cluster_cfg(handle.cluster)
        if cluster is None:
            # Non-SLURM backends fall back to PollResult-based detection
            # (GCP returns "running" only when provisioning is done).
            return backend.poll(handle).status == "running"
        state = query_slurm_state(robot_alias=cluster.ssh_host, job_id=handle.job_id)
        return state.get("status") == "RUNNING"

    def _slurm_is_live_after_cancel(backend: Any, handle: Any) -> bool:
        """``squeue --name eps-issue-<N>`` non-empty = still live.

        DRAC robots reject ``sacct`` (allowlist), so "no longer visible
        in squeue" is the most authoritative terminal signal the cancel
        state machine can get. A live entry (any state — PENDING /
        RUNNING / COMPLETING) counts as still-live.

        ``query_by_name`` RAISES :class:`slurm_monitor.SlurmProbeError`
        on rc != 0 (probe failed — state UNKNOWN, not absent). We let it
        propagate: ``cancel_and_wait`` treats a raising live-probe as
        still-live and keeps polling under its grace budget, resolving
        to ``manual_attention`` if the transport stays broken — the
        pre-fix behavior read the failure as "job gone" and returned
        "cancelled" on a LIVE job (round-6 B1).
        """
        cluster = _resolve_cluster_cfg(handle.cluster)
        if cluster is None:
            status = backend.poll(handle).status
            return status not in {"done", "dead"}
        # Use the same job name the launch path stamped onto the
        # sbatch (``eps-issue-<N>``). query_by_name returns the most
        # recent matching live job_id or None.

        # Reconstruct the job name from handle.pod_name (the launch path
        # set it to ``job_name(spec, plan_hash)`` — either
        # ``eps-issue-<N>`` or ``eps-issue-<N>-<plan_hash>``).
        # query_by_name accepts the full name verbatim.
        found = query_by_name(robot_alias=cluster.ssh_host, job_name=handle.pod_name)
        return found is not None

    def _slurm_started_evidence(backend: Any, handle: Any) -> dict[str, Any] | None:
        """Scratch-dir probe for the router's terminal-before-running classification.

        A SLURM job that fast-fails (e.g. the in-job preflight) can
        transition PD→R→exit between router polls and "vanish" before
        it is ever observed RUNNING. If the scratch dir holds runtime
        artifacts (``status.json`` / ``job.out``), the job DID start —
        a WORKLOAD failure the router must surface (NO GCP fallback),
        not ``no_compute_available``. Transport is rsync (allowlisted
        by the robot forced-command wrapper; ``ssh <alias> cat`` is
        NOT). Non-SLURM handles return None (GCP's provision IS the
        start, so terminal-before-running cannot mask a workload
        failure there; RunPod never parks).

        ``min_artifact_ts`` (the launch path's ``submitted_at`` stamp on
        ``handle.extra``) gates out PRIOR-attempt artifacts: the
        per-issue scratch dir is reused across attempts, so without it
        a re-run's terminal park reads attempt-1's status.json/job.out
        as proof THIS job started — a guaranteed false workload-failure
        (issue 535 attempt 2).
        """
        del backend
        cluster = _resolve_cluster_cfg(handle.cluster)
        if cluster is None:
            return None
        from explore_persona_space.backends.slurm_monitor import (
            fetch_started_evidence,
        )

        submitted_at = handle.extra.get("submitted_at")
        return fetch_started_evidence(
            robot_alias=cluster.ssh_host,
            scratch_dir=handle.scratch_dir,
            job_id=str(handle.job_id),
            min_artifact_ts=float(submitted_at) if submitted_at is not None else None,
        )

    def _reconnect(backend: Any, kind: str, spec: Any) -> Any:
        """Per-kind reconnect dispatch.

        SLURM: ``squeue --name eps-issue-<N>`` — if a matching live job
        exists, rebuild a RunHandle from its id + the cluster's known
        scratch path. GCP: :func:`backends.gcp.reconnect_or_none`. RunPod
        and unknown kinds return None (the existing ``pod_lifecycle.py``
        flow is idempotent on its own).

        ``query_by_name`` raises ``SlurmProbeError`` on rc != 0 (probe
        failed, NOT job-absent); the router's ``_try_reconnect``
        propagates it so the lane is skipped / the override raises a
        typed terminal instead of blind-submitting a duplicate
        (round-6 B1).
        """
        if kind in {"nibi", "fir", "mila"}:
            # _resolve_cluster_cfg raises on a typo'd / unavailable
            # cluster — that's a real misconfiguration, NOT something to
            # paper over with a silent None fallback.
            cluster = _resolve_cluster_cfg(kind)
            from explore_persona_space.backends.slurm import (
                job_name,
                scratch_dir_for,
            )

            name = job_name(spec, plan_hash=spec.extra.get("plan_hash"))
            found_id = query_by_name(robot_alias=cluster.ssh_host, job_name=name)
            if not found_id:
                return None
            scratch_dir = scratch_dir_for(spec, cluster)
            log_path = f"{scratch_dir}/job.out"
            # Rebuild a RunHandle that matches the launch-path shape.
            from explore_persona_space.backends.base import RunHandle

            return RunHandle(
                backend="cluster",
                cluster=kind,
                job_id=found_id,
                pod_name=name,
                scratch_dir=scratch_dir,
                log_path=log_path,
                extra={
                    "account": cluster.account,
                    "robot_alias": cluster.robot_alias,
                    "intent": spec.intent,
                    "issue": int(spec.issue),
                },
            )
        if kind == "gcp":
            # Use the public ``config`` / ``runner`` properties — the
            # backend stores these internally as ``self._config`` /
            # ``self._run``, so reaching for ``gcp_backend.config`` and
            # ``gcp_backend._runner`` (the pre-fix code path)
            # AttributeError'd on EVERY explicit ``backend: gcp`` lane
            # and every auto-chain GCP escalation that hit the
            # reconnect path.
            return gcp_reconnect_or_none(
                spec=spec,
                config=gcp_backend.config,
                runner=gcp_backend.runner,
            )
        return None

    # Slice-7 wire: the real ``ssh mila true`` probe over the
    # ControlMaster socket. Returns False on socket-down (treated as
    # skip-the-lane, NOT as an error — see
    # ``backends.slurm.mila_socket_alive`` for the graceful-False
    # contract). Late-imported per factory call so tests can
    # ``monkeypatch.setattr(slurm, "mila_socket_alive", ...)`` BEFORE
    # the factory build and have the closure pick up the patch.
    from explore_persona_space.backends.slurm import (
        mila_socket_alive as _mila_socket_alive,
    )

    return {
        "runpod_backend": runpod_backend,
        "free_backends": free_backends,
        "gcp_backend": gcp_backend,
        "marker_poster": post_marker_via_task_py,
        "is_started": _slurm_is_started,
        "is_live_after_cancel": _slurm_is_live_after_cancel,
        "started_evidence_probe": _slurm_started_evidence,
        "reconnect_fn": _reconnect,
        "mila_socket_alive": _mila_socket_alive,
    }


def _resolve_cluster_cfg(name: str | None) -> Any | None:
    """Look up a :class:`ClusterConfig` by name.

    Returns ``None`` only when ``name`` itself is ``None`` (the caller
    has a non-SLURM handle — e.g. a RunPod / GCP handle whose
    ``handle.cluster`` is ``None`` by construction). For any non-None
    name we delegate straight to :func:`backends.slurm.get_cluster_config`
    and let its ``ValueError`` (unknown name) / ``RuntimeError``
    (``available=False``) propagate verbatim — those signal real
    misconfiguration (a typo'd ``backend:`` / ``cluster:`` in the task
    frontmatter, or a cluster the production wiring is gated against)
    and MUST crash loudly. Silently returning ``None`` here would drop
    the SLURM-aware ``_slurm_is_started`` /
    ``_slurm_is_live_after_cancel`` closures to their PollResult-based
    fallback, which silently re-introduces the PENDING→"running" enum
    bug those probes exist to prevent.
    """
    if name is None:
        return None
    from explore_persona_space.backends.slurm import get_cluster_config

    return get_cluster_config(name)


def _cmd_launch(args: argparse.Namespace, *, backends_factory: Callable[[], dict[str, Any]]) -> int:
    """``launch`` action: build spec → dispatch → write sidecar → print outcome.

    Translates router terminals via
    :func:`backends.issue_dispatch.classify_terminal_exception` into a
    structured JSON line on stdout + a non-zero exit code. This CLI
    only EMITS the failure JSON (and the matching exit code); it does
    NOT mutate task state itself. The orchestrator (``/issue`` SKILL.md
    Step 6b) reads the JSON line, posts ``epm:failure v1`` with the
    carried ``failure_class`` + ``note``, and calls
    ``scripts/task.py set-status <N> blocked`` itself — keeping all
    task-workflow mutations on the single ``task.py`` flock owner.
    """
    from explore_persona_space.backends.issue_dispatch import (
        build_run_spec,
        classify_terminal_exception,
        dispatch_for_issue,
    )
    from explore_persona_space.backends.router import RouteError

    extra: dict[str, Any] = {}
    if getattr(args, "repo_branch", None):
        # GCP-only knob: the GCE startup script clones from origin, so a
        # feature-branch workload must name its branch (issue 535 r6).
        extra["repo_branch"] = args.repo_branch
    spec = build_run_spec(
        issue=args.issue,
        intent=args.intent,
        backend_value=args.backend,
        gpus=args.gpus,
        time_budget_hours=args.time_budget_hours,
        account=args.account,
        cluster=args.cluster,
        hydra_args=tuple(args.hydra or ()),
        extra=extra,
    )

    deps = backends_factory()
    try:
        outcome = dispatch_for_issue(
            spec,
            runpod_backend=deps["runpod_backend"],
            free_backends=deps["free_backends"],
            gcp_backend=deps["gcp_backend"],
            mila_socket_alive=deps["mila_socket_alive"],
            marker_poster=deps["marker_poster"],
            is_started=deps["is_started"],
            is_live_after_cancel=deps["is_live_after_cancel"],
            started_evidence_probe=deps.get("started_evidence_probe"),
            reconnect_fn=deps["reconnect_fn"],
        )
    except RouteError as exc:
        translation = classify_terminal_exception(exc)
        body = {
            "ok": False,
            "issue": int(args.issue),
            "exception": type(exc).__name__,
            "failure_class": translation.failure_class,
            "status": translation.status,
            "note": translation.note,
        }
        print(json.dumps(body, sort_keys=True))
        return 2

    result = outcome.result
    body = {
        "ok": True,
        "issue": int(args.issue),
        "chosen_kind": result.chosen_kind,
        "requested_kind": result.requested_kind,
        "reason": result.reason,
        "cluster": result.cluster,
        "handle_sidecar_path": (
            str(outcome.handle_sidecar_path) if outcome.handle_sidecar_path else None
        ),
        "pod_name": result.handle.pod_name,
        "job_id": result.handle.job_id,
    }
    if outcome.sidecar_write_error is not None:
        # The launch SUCCEEDED (live VM / job) but the sidecar write
        # failed — print the handle JSON anyway (it IS the recovery
        # record) plus the error, instead of the pre-fix rc=4 crash
        # that stranded live infra with no handle on stdout. The FULL
        # serialized handle rides along (M4.1): ``deserialize_handle``
        # requires backend/scratch_dir/log_path too, so the summary
        # fields alone were NOT sufficient to hand-write a
        # ``--handle-file`` sidecar and run finalize.
        from explore_persona_space.backends.issue_dispatch import serialize_handle

        body["sidecar_write_error"] = outcome.sidecar_write_error
        body["handle"] = serialize_handle(result.handle)
        logging.getLogger("dispatch_issue").error(
            "launch succeeded but the handle sidecar write FAILED (%s); "
            "the JSON line below is the only recovery record — keep it. "
            "job_id=%s pod_name=%s chosen_kind=%s",
            outcome.sidecar_write_error,
            result.handle.job_id,
            result.handle.pod_name,
            result.chosen_kind,
        )
    print(json.dumps(body, sort_keys=True))
    return 0


def _cmd_finalize(
    args: argparse.Namespace, *, backends_factory: Callable[[], dict[str, Any]]
) -> int:
    """``finalize`` action: read sidecar → confirm_artifacts → teardown.

    Gates teardown on the per-backend ``confirm_artifacts`` PASS. A
    FAIL on confirm_artifacts SKIPS teardown (preserves the live
    backend so an operator can inspect what didn't upload). The
    orchestrator's Step 8 ALSO runs the upload-verifier agent against
    the same handle; this CLI is the complementary MECHANICAL gate
    (HF Hub list_repo_files + WandB run + git-figure + completion
    sentinel — see ``backends.artifacts.confirm_artifacts_from_handle``).

    After a SUCCESSFUL teardown the sidecar is renamed to
    ``<name>.finalized`` (audit record, never deleted) so a later
    finalize for the same issue cannot tear down a fresh run through
    the stale handle; the duplicate tick then no-ops with the benign
    rc=2 ``missing_handle_sidecar`` shape (Mn4.3).
    """
    from explore_persona_space.backends.issue_dispatch import (
        default_handle_sidecar_path,
        read_handle_sidecar,
    )

    sidecar = args.handle_file or default_handle_sidecar_path(args.issue)
    if not Path(sidecar).exists():
        body = {
            "ok": False,
            "issue": int(args.issue),
            "failure_class": "infra",
            "reason": "missing_handle_sidecar",
            "detail": f"no sidecar at {sidecar}",
        }
        print(json.dumps(body, sort_keys=True))
        return 2

    handle = read_handle_sidecar(Path(sidecar))
    deps = backends_factory()
    backend = _resolve_backend_for_handle(handle, deps)

    if not args.skip_confirm_artifacts:
        passed = backend.confirm_artifacts(handle)
        if not passed:
            body = {
                "ok": False,
                "issue": int(args.issue),
                "phase": "confirm_artifacts",
                "chosen_kind": handle.backend,
                "pod_name": handle.pod_name,
                "reason": "confirm_artifacts_failed",
            }
            print(json.dumps(body, sort_keys=True))
            return 3

    backend.teardown(handle)

    # Mn4.3: retire the sidecar AFTER a successful teardown by renaming
    # it to ``<name>.finalized`` (kept for audit, never deleted). A
    # sidecar left in place outlives its VM / job, and a LATER cleanup
    # finalize for the same issue (e.g. the harness's launch-crash
    # best-effort path) would tear down whatever live run the STALE
    # sidecar points at — destructive when the issue number is shared
    # with a production run. After the rename a second finalize sees a
    # missing sidecar → the benign rc=2 ``missing_handle_sidecar``
    # no-op. A rename failure is logged LOUD but does NOT flip the exit
    # code: teardown DID run, and rc!=0 here would make the harness
    # raise "teardown may NOT have run", which would be false.
    sidecar_path = Path(sidecar)
    finalized_path: Path | None = None
    try:
        candidate = sidecar_path.with_name(sidecar_path.name + ".finalized")
        sidecar_path.rename(candidate)
        finalized_path = candidate
    except OSError as exc:
        logging.getLogger("dispatch_issue").error(
            "teardown succeeded but the sidecar rename to *.finalized FAILED (%s: %s); "
            "the stale sidecar at %s can mis-target a LATER finalize for issue %d — "
            "remove or rename it manually.",
            type(exc).__name__,
            exc,
            sidecar_path,
            int(args.issue),
        )

    body = {
        "ok": True,
        "issue": int(args.issue),
        "phase": "teardown",
        "chosen_kind": handle.backend,
        "pod_name": handle.pod_name,
        "sidecar_finalized": str(finalized_path) if finalized_path else None,
    }
    print(json.dumps(body, sort_keys=True))
    return 0


def _resolve_backend_for_handle(handle: Any, deps: dict[str, Any]) -> Any:
    """Pick the right ComputeBackend instance for a serialized handle.

    The handle's ``backend`` field names the kind; we look it up in the
    production deps dict. Unknown kinds raise ``ValueError`` — a silent
    default would mis-route teardown to the wrong backend.
    """
    kind = handle.backend
    if kind == "runpod":
        return deps["runpod_backend"]
    if kind in {"cluster", "nibi", "fir"}:
        # ``cluster`` (legacy) / ``nibi`` / ``fir`` all route to the same
        # SlurmBackend instance (``_cluster_for_spec`` reads
        # ``handle.cluster``). ``free_backends`` keys on the kind name;
        # fall back to ANY available SLURM backend.
        free = deps["free_backends"]
        if kind in free:
            return free[kind]
        if free:
            return next(iter(free.values()))
        raise ValueError(f"no SLURM backend wired for handle.backend={kind!r}")
    if kind == "gcp":
        return deps["gcp_backend"]
    raise ValueError(f"unknown handle.backend={kind!r}; cannot resolve a backend instance")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    launch = sub.add_parser("launch", help="Dispatch a fresh run through the router.")
    launch.add_argument("--issue", type=int, required=True, help="Task / issue number.")
    launch.add_argument(
        "--intent",
        type=str,
        required=True,
        help="Workload intent (lora-7b, ft-7b, eval, debug, inf-70b, ft-70b).",
    )
    launch.add_argument(
        "--backend",
        type=str,
        default=None,
        help=(
            "Frontmatter ``backend:`` value verbatim (empty / absent → auto). "
            "One of: runpod, nibi, fir, gcp, mila, cluster (legacy alias), auto."
        ),
    )
    launch.add_argument("--cluster", type=str, default=None, help="SLURM cluster name (nibi/fir).")
    launch.add_argument("--gpus", type=int, default=None, help="Override GPU count.")
    launch.add_argument(
        "--time-budget-hours",
        type=float,
        default=None,
        help="Override wall-clock budget (hours; SLURM ``--time``).",
    )
    launch.add_argument("--account", type=str, default=None, help="SLURM ``--account`` override.")
    launch.add_argument(
        "--repo-branch",
        type=str,
        default=None,
        help=(
            "Git branch the GCE startup script clones (GCP lane only; "
            "SLURM lanes rsync the local worktree instead). Required when "
            "the workload's code/configs live on a feature branch — the "
            "default clone of main silently runs stale code (issue 535 r6)."
        ),
    )
    launch.add_argument(
        "--hydra",
        action="append",
        default=None,
        help="Hydra override (e.g. ``condition=c1``). Repeatable.",
    )

    finalize = sub.add_parser(
        "finalize",
        help="Run confirm_artifacts + teardown on the sidecar handle.",
    )
    finalize.add_argument("--issue", type=int, required=True)
    finalize.add_argument(
        "--handle-file",
        type=Path,
        default=None,
        help="Path to the per-issue handle sidecar JSON "
        "(default: .claude/cache/issue-<N>-handle.json).",
    )
    finalize.add_argument(
        "--skip-confirm-artifacts",
        action="store_true",
        help=(
            "Skip the confirm_artifacts gate (matches "
            "``pod.py terminate --skip-upload-verify``; use only when the "
            "experiment crashed before artifacts could land)."
        ),
    )

    # ``--debug`` lives on each SUBPARSER (NOT the top-level parser).
    # argparse evaluates positionally — a flag attached only to the top-
    # level parser MUST appear before the subcommand or argparse errors
    # "unrecognized arguments: --debug". Production invocations
    # (SKILL.md Step 6b / Step 8) put the flag AFTER the subcommand:
    # ``dispatch_issue.py launch --debug --issue N ...``. Putting
    # ``--debug`` on the subparsers is the only attachment that lets
    # that production form parse.
    debug_kw = {"action": "store_true", "help": "Log to stderr at DEBUG level."}
    launch.add_argument("--debug", **debug_kw)
    finalize.add_argument("--debug", **debug_kw)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    backends_factory: Callable[[], dict[str, Any]] | None = None,
) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    factory = backends_factory or _build_production_backends
    try:
        if args.action == "launch":
            return _cmd_launch(args, backends_factory=factory)
        if args.action == "finalize":
            return _cmd_finalize(args, backends_factory=factory)
        # argparse's required=True on the subparsers prevents this branch
        # in normal use; defensive against a future refactor that adds a
        # third action without wiring it here. ``parser.error`` calls
        # ``sys.exit(2)`` and never returns, so the return below is
        # unreachable — kept only to satisfy mypy's
        # ``Callable[..., int]`` signature on ``main``.
        parser.error(f"unknown action {args.action!r}")
        return 4  # pragma: no cover — unreachable; parser.error → SystemExit(2)
    except SystemExit:
        # Re-raise argparse / parser.error exits verbatim.
        raise
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        body = {
            "ok": False,
            "issue": int(getattr(args, "issue", 0)),
            "exception": type(exc).__name__,
            "detail": str(exc),
        }
        print(json.dumps(body, sort_keys=True))
        return 4


if __name__ == "__main__":
    sys.exit(main())


# Re-exports for tests (avoids reaching into private names).
__all__ = [
    "_build_production_backends",
    "_cmd_finalize",
    "_cmd_launch",
    "_resolve_backend_for_handle",
    "main",
]
