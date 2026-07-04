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
  classification. The pre-route ``--gpus``/GCP machine-type mismatch
  guard (``reason: gpus_machine_mismatch``, incident #599) exits 2
  with the same JSON shape: the GCP lane sizes its VM from ``--intent``
  alone (``backends/gcp.INTENT_TO_MACHINE``) and silently ignores
  ``--gpus``, so a gcp-reachable launch with a mismatched override is
  refused BEFORE any backend is built instead of provisioning a
  wrong-sized VM that crashes the workload at startup.
* ``3`` — confirm_artifacts FAIL on the ``finalize`` path
  (artifacts not landed; teardown SKIPPED to preserve evidence).
  ``stdout`` carries the per-check reasons. Special case: when the
  handle carries NO ``expected_artifacts`` declaration (every launch
  path — GCP, SLURM, RunPod — populates it as of #598, so this is
  pre-#598 in-flight handles only) the mechanical
  gate is structurally unsatisfiable; finalize then accepts
  agent-level upload-verification PASS evidence from the task's
  ``events.jsonl`` and proceeds to teardown with a LOUD log +
  ``"confirm_artifacts": "skipped_no_declaration_agent_pass"`` in the
  JSON (incident #585: every explicit ``--backend runpod`` finalize
  exited 3 on a fully verified run, forcing a raw ``pod.py
  terminate`` bypass that skipped the Mn4.3 sidecar retirement). With
  neither a declaration nor agent PASS evidence the exit stays 3 with
  ``reason: confirm_artifacts_no_declaration``.
* ``4`` — unexpected exception. ``stderr`` carries the traceback.
* ``75`` — still-waiting (EX_TEMPFAIL; mirrors
  ``pod_lifecycle.EXIT_STILL_WAITING``). TWO producers, same contract:
  (1) the RunPod lane's ``pod_lifecycle.py provision`` exited 75 because
  its bounded wait-for-capacity loop reached the per-process wall-clock
  budget while capacity / the fleet burn cap kept the provision queued
  (``reason: wait_for_capacity_budget_reached``); (2) the GCP lane's
  ``gcloud compute instances create`` exceeded the 300s subprocess cap on
  a FLEX_START rung but a post-timeout ``instances list`` probe found the
  instance live server-side — a FLEX_START preemptible-queueing state
  (``reason: gcloud_create_timeout_still_provisioning``, with additive
  ``instance_name`` / ``instance_status`` keys; #736). NEITHER is a
  failure: ``stdout`` carries ``still_waiting: true`` + ``rerun: true``
  and the caller RE-RUNS the same launch command to continue waiting
  (the RunPod wait loop is state-free; the GCP re-run reconnects to the
  live instance via ``reconnect_or_none`` with NO double-create, so both
  resume exactly). Do NOT post ``epm:failure v1`` / ``set-status
  blocked`` on this exit. (Incident #603, 2026-06-11: this exit
  previously fell through to the generic handler and crashed as an rc-4
  ``CalledProcessError``. Incident #658/#736, 2026-06-29: the GCP
  create-timeout case crashed as the undocumented rc-4 traceback below.)

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
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Repo-root sys.path bootstrap. Invoking this file as a script puts only
# scripts/ (the script's own dir) on sys.path — NOT the repo root — so any
# lazy `from scripts.X import ...` inside the backends this CLI wires
# (e.g. `backends/runpod.py` does `from scripts.poll_pipeline import ...`
# on its poll path) fails with ``ModuleNotFoundError: No module named
# 'scripts'`` unless PYTHONPATH is set manually. Insert the repo root so
# the documented invocation (``uv run python scripts/dispatch_issue.py
# launch --issue <N>``) works from any cwd (defensive parity with
# backend_poll.py, #571 — no launch/finalize-path scripts.* import exists
# today, but a backend refactor adding one would reproduce the incident).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _current_git_branch() -> str | None:
    """Current branch of the invoking checkout (None on detached HEAD / error).

    Mirrors ``router_acceptance.py:_current_git_branch`` — the production
    twin of the harness's r19 current-branch default (round-2 Claude
    Major, task #535).
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            env={**os.environ},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = proc.stdout.strip()
    return branch if branch and branch != "HEAD" else None


def _git_branch_of(root: str) -> str | None:
    """Checked-out branch of the git checkout at ``root``; None on detached/error.

    Uses ``git -C <root> branch --show-current``, which prints an empty line on a
    detached HEAD (mapped to None — no "HEAD" sentinel needed, unlike
    ``rev-parse --abbrev-ref``). Any subprocess failure also maps to None: the
    caller treats an unresolvable worktree branch as "no default". (task #824)
    """
    try:
        proc = subprocess.run(
            ["git", "-C", root, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            env={**os.environ},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = proc.stdout.strip()
    return branch or None


def _warn_if_branch_not_pushed(branch: str, git_root: str) -> None:
    """WARN loudly (never fail) when ``branch`` is not visible on origin.

    The GCE startup script clones from origin, so an unpushed defaulted branch
    guarantees a downstream clone/checkout failure (#812 failure class). A
    ls-remote failure here must not block the launch: WARN and continue.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", git_root, "ls-remote", "--exit-code", "origin", branch],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
        )
        visible = proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        visible = False
    if not visible:
        logging.getLogger("dispatch_issue").warning(
            "repo-branch %r is not visible on origin (git ls-remote failed or "
            "found no match) — the GCE startup clone will fail unless the "
            "branch is pushed first",
            branch,
        )


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
            from explore_persona_space.backends.artifacts import (
                EXPECTED_ARTIFACTS_HANDLE_KEY,
            )
            from explore_persona_space.backends.slurm import (
                expected_artifacts_declaration,
                job_name,
                scratch_dir_for,
            )

            name = job_name(spec, plan_hash=spec.extra.get("plan_hash"))
            found_id = query_by_name(robot_alias=cluster.ssh_host, job_name=name)
            if not found_id:
                return None
            scratch_dir = scratch_dir_for(spec, cluster)
            log_path = f"{scratch_dir}/job.out"
            # Rebuild a RunHandle that matches the launch-path shape —
            # INCLUDING the expected-artifacts declaration (#598, GCP
            # parity with gcp.reconnect_or_none): a reconnected handle
            # is exactly the handle finalize later consumes, and leaving
            # it bare would silently re-create the #588 "missing
            # declaration" FAIL on the recovery path. The attempt id is
            # derivable (slurm-<found_id>), so the rebuilt declaration
            # matches what launch() attached.
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
                    EXPECTED_ARTIFACTS_HANDLE_KEY: expected_artifacts_declaration(
                        spec=spec, job_id=found_id
                    ),
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


def _frontmatter_backend_value(issue: int) -> str | None:
    """The task's frontmatter ``backend:`` value, normalized for the override check.

    Returns ``""`` when the key is absent or the value is empty (the task
    itself says auto), the stripped + lowercased value otherwise (an
    explicit ``backend: auto`` returns ``"auto"`` — the caller treats it
    the same as absent/empty, since both state auto routing), and
    ``None`` when the frontmatter could not be read at all (missing task,
    unreadable body.md) — the caller then SKIPS the
    override-without-frontmatter check rather than guessing.

    Reads via ``task_workflow.get_task``, which resolves against the MAIN
    checkout's ``tasks/`` tree regardless of the invoking worktree (the
    resolver branch-guards to ``main``) — same pattern as
    :func:`_agent_upload_verification_passed`. Library import, not a
    ``task.py`` shell-out, and this CLI is VM-side only.
    """
    try:
        from explore_persona_space.task_workflow import get_task

        fm = get_task(int(issue)).get("frontmatter") or {}
    except Exception as exc:
        logging.getLogger("dispatch_issue").warning(
            "could not read frontmatter for issue=%d (%s: %s)",
            int(issue),
            type(exc).__name__,
            exc,
        )
        return None
    raw = fm.get("backend")
    if raw is None:
        return ""
    return str(raw).strip().lower()


def _recognized_frontmatter_backends() -> frozenset[str]:
    """Backend values the router (or the legacy selector surface) recognizes.

    Sourced from the router's OWN definition so the override-conflict
    guard can never drift from the routable set — never a duplicated
    hardcoded list. ``_VALID_BACKEND_VALUES`` is router-private; this
    import is a deliberate coupling (a router rename surfaces here as an
    ImportError in this CLI's tests rather than silently degrading the
    guard to "everything unrecognized"). The legacy ``"cluster"``
    literal (selector-surface alias, normalized to nibi by
    ``selector._resolve_cluster_name``) is added on top: a frontmatter
    ``backend: cluster`` names a real SLURM lane, so a runpod override
    against it is a CONFLICT, not a typo.
    """
    from explore_persona_space.backends.router import _VALID_BACKEND_VALUES

    return _VALID_BACKEND_VALUES | {"cluster"}


def _wrap_marker_poster_with_override_flag(
    poster: Callable[..., None],
    flags: dict[str, Any],
) -> Callable[..., None]:
    """Stamp CLI-side override-visibility ``flags`` onto backend-selected posts.

    The router builds the ``epm:backend-selected`` body itself
    (``router._post_backend_selected``) and only ``result.extra`` reaches
    the marker — ``spec.extra`` does not — so CLI-level facts about the
    explicit ``--backend runpod`` override (no frontmatter backing /
    conflicting frontmatter lane / unrecognized frontmatter value) are
    threaded by decorating the injected ``marker_poster`` instead of
    touching router internals. ``flags`` is merged into the body's
    ``extra`` dict (e.g. ``{"override_without_frontmatter": True}`` or
    ``{"override_conflicts_frontmatter": True, "frontmatter_backend":
    "gcp"}``). Non-backend-selected markers and unparseable notes pass
    through untouched. Observability only: never alters routing control
    flow, never fails the post.
    """

    def _wrapped(**kwargs: Any) -> None:
        if kwargs.get("marker") == "epm:backend-selected":
            try:
                body = json.loads(kwargs.get("note") or "")
            except (TypeError, json.JSONDecodeError):
                body = None
            if isinstance(body, dict) and isinstance(body.get("extra"), dict):
                body["extra"].update(flags)
                kwargs["note"] = json.dumps(body, sort_keys=True)
        poster(**kwargs)

    return _wrapped


def _gpus_gcp_lane_conflict(spec: Any) -> dict[str, Any] | None:
    """Pre-route ``--gpus`` vs GCP machine-type mismatch guard (incident #599).

    The GCP lane sizes its VM from ``spec.intent`` alone
    (``backends/gcp.INTENT_TO_MACHINE``) and silently IGNORES
    ``spec.gpus`` — unlike RunPod (maps it to ``pod_lifecycle.py
    --gpu-count``) and SLURM (maps it to the ``--gres`` render), which
    both honor the override. A gcp-reachable launch whose ``--gpus``
    mismatches the intent's machine therefore provisions a wrong-sized
    VM whose workload crashes at startup with no fallback (#599:
    ``--intent lora-7b --gpus 4`` → a2-ultragpu-1g, 1x A100-80, for a
    driver requiring N_GPUS=4). The mapping is static, so the mismatch
    is knowable BEFORE any backend is built — validate up front and
    fail LOUD.

    Returns the exit-2 failure body (same ``failure_class`` / ``status``
    / ``note`` shape as the router-terminal translation, so SKILL.md
    Step 6b and the failure classifier handle it unchanged) when the
    launch must be refused; ``None`` when the launch may proceed:

    * no ``--gpus`` override (intent defaults apply on every lane);
    * an explicit non-GCP backend (those lanes honor the override, and
      an explicit override never escalates to GCP);
    * ``backend: auto`` whose resolved lane order excludes ``gcp``
      (``EPM_AUTO_LANE_ORDER``) — GCP is unreachable;
    * a defective ``EPM_AUTO_LANE_ORDER`` (``auto_lane_order`` raises
      ``RouteError``) — skip the guard; ``route()`` surfaces the SAME
      defect through the existing terminal classification, which a
      gpus-mismatch message must not preempt;
    * an intent with no GCP machine mapping (``inf-70b`` / ``ft-70b``)
      — ``machine_for_intent`` already fails loud inside the GCP lane;
    * a matching GPU count.
    """
    if spec.gpus is None:
        return None
    from explore_persona_space.backends.router import RouteError, auto_lane_order

    if spec.backend == "gcp":
        gcp_reachable = True
    elif spec.backend == "auto":
        try:
            gcp_reachable = "gcp" in auto_lane_order()
        except RouteError:
            return None
    else:
        return None
    if not gcp_reachable:
        return None
    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE

    machine = INTENT_TO_MACHINE.get(spec.intent)
    requested = int(spec.gpus)
    if machine is None or machine.gpu_count == requested:
        return None
    matching = sorted(intent for intent, m in INTENT_TO_MACHINE.items() if m.gpu_count == requested)
    if matching:
        remedy = (
            f"use an intent whose GCP machine carries {requested} GPU(s): {', '.join(matching)}"
        )
    else:
        remedy = (
            f"no GCP intent maps to a {requested}-GPU machine — pick a backend that "
            "honors the override"
        )
    note = (
        "failure_class: infra\n"
        "reason: gpus_machine_mismatch\n"
        f"detail: --gpus {requested} is not honored by the GCP lane — intent "
        f"{spec.intent!r} maps to machine type {machine.machine_type!r} "
        f"({machine.gpu_count}x {machine.gpu_kind}) regardless of the override, so the "
        "VM would start wrong-sized and crash the workload (incident #599). "
        f"Fix: {remedy}; or drop --gpus (the intent default applies); or pin a backend "
        "that honors the override (--backend runpod maps it to pod_lifecycle "
        "--gpu-count; SLURM lanes map it to --gres)."
    )
    return {
        "ok": False,
        "issue": int(spec.issue),
        "failure_class": "infra",
        "status": "blocked",
        "reason": "gpus_machine_mismatch",
        "note": note,
    }


def _ft_intent_gcp_default_boot_disk(spec: Any) -> bool:
    """True when an ft-* intent is gcp-reachable with no ``--boot-disk-gb`` (incident #606).

    The GCP lane provisions its boot disk at the
    ``backends/gcp.GcpConfig.default_boot_disk_gb`` default (300 GB
    pd-ssd) unless ``spec.extra["boot_disk_gb"]`` overrides it. A ZeRO-3
    full fine-tune (``ft-7b``) fills 300 GB with optimizer-state
    checkpoints in ~1h; the #606 instance kernel-panicked on the full
    disk, cloud-init ENOSPC'd, the guest agent could not write
    ``authorized_keys`` (SSH publickey lockout), and the wedged VM idled
    on 4x A100 until deleted. WARNING only — NEVER a refusal:
    eval/lora intents on the default are fine, and even ft intents may
    legitimately run small-disk smokes.

    Mirrors the gcp-reachability logic of :func:`_gpus_gcp_lane_conflict`:
    stand down when the boot disk is explicitly sized, the intent is not
    an ft-* intent with a GCP machine mapping (``ft-70b`` has none —
    ``machine_for_intent`` fails loud inside the lane before disk
    matters), the backend is an explicit non-GCP lane, or ``auto``'s
    resolved lane order excludes ``gcp`` (a defective
    ``EPM_AUTO_LANE_ORDER`` also stands down — ``route()`` surfaces that
    defect through the existing terminal classification).
    """
    if (spec.extra or {}).get("boot_disk_gb"):
        return False
    from explore_persona_space.backends.gcp import INTENT_TO_MACHINE

    if not (str(spec.intent).startswith("ft-") and spec.intent in INTENT_TO_MACHINE):
        return False
    if spec.backend == "gcp":
        return True
    if spec.backend != "auto":
        return False
    from explore_persona_space.backends.router import RouteError, auto_lane_order

    try:
        return "gcp" in auto_lane_order()
    except RouteError:
        return False


def _warn_default_boot_disk_ft_intent(
    spec: Any, issue: int, marker_poster: Callable[..., None]
) -> Callable[..., None]:
    """Emit the #606 default-boot-disk warning + marker flag when applicable.

    Default-boot-disk visibility for gcp-reachable ft intents (incident
    #606): the relaunch dropped the plan's explicit "500 GB pd-ssd"
    Reproducibility spec, the 300 GB default filled in ~1h of ZeRO-3
    full-FT checkpoints, and the instance kernel-panicked into an SSH
    lockout while 4x A100 idled. Returns ``marker_poster`` unchanged when
    :func:`_ft_intent_gcp_default_boot_disk` stands down, or wrapped with
    ``extra.boot_disk_default_with_ft_intent=true`` after the LOUD stderr
    warning fires. ADDITIVE only — never blocks the launch.
    """
    if not _ft_intent_gcp_default_boot_disk(spec):
        return marker_poster
    logging.getLogger("dispatch_issue").warning(
        "gcp-reachable launch for issue=%d with --intent %s and no --boot-disk-gb — "
        "the GCP lane defaults the boot disk to 300 GB pd-ssd "
        "(backends/gcp.GcpConfig.default_boot_disk_gb), which a ZeRO-3 full-FT "
        "fills with optimizer-state checkpoints in ~1h (incident #606: kernel "
        "panic on the full disk, cloud-init ENOSPC, SSH key-provisioning lockout, "
        "4x A100 idling until deletion). Thread the plan's Reproducibility "
        "pod-row disk size via --boot-disk-gb on EVERY launch, relaunches "
        "included; for ft-* intents whose plan names no size, >=500 GB is the "
        "working default. Launch continues; the epm:backend-selected marker "
        "carries extra.boot_disk_default_with_ft_intent=true so the default-disk "
        "launch is visible on the events trail.",
        issue,
        spec.intent,
    )
    return _wrap_marker_poster_with_override_flag(
        marker_poster, {"boot_disk_default_with_ft_intent": True}
    )


# Still-waiting exit code (EX_TEMPFAIL). Mirrors
# ``scripts/pod_lifecycle.py::EXIT_STILL_WAITING`` — mirrored rather than
# imported so this CLI stays import-light at module load; the equality is
# pinned by ``tests/test_dispatch_issue_cli.py::
# test_exit_still_waiting_matches_pod_lifecycle``.
EXIT_STILL_WAITING = 75


def _provision_still_waiting(exc: subprocess.CalledProcessError) -> bool:
    """True iff ``exc`` is ``pod_lifecycle.py provision``'s still-waiting exit.

    ``pod_lifecycle.py provision`` exits :data:`EXIT_STILL_WAITING` (75,
    EX_TEMPFAIL) when its bounded wait-for-capacity loop reaches the
    per-process wall-clock budget — a NORMAL outcome of any capacity /
    fleet-burn-cap wait, documented in ``pod_lifecycle.py`` as "re-run
    the same command to continue waiting". The RunPod backend shells
    provision with ``check=True``, so that exit surfaces here as a
    ``CalledProcessError``. Matching on BOTH the returncode AND the
    command shape keeps an unrelated rc-75 subprocess from another lane
    (gcloud / ssh / sbatch) out of the still-waiting branch — only
    ``pod_lifecycle.py provision`` carries this contract.
    """
    if exc.returncode != EXIT_STILL_WAITING:
        return False
    cmd = exc.cmd if isinstance(exc.cmd, (list, tuple)) else [exc.cmd]
    parts = [str(p) for p in cmd]
    return any("pod_lifecycle.py" in p for p in parts) and "provision" in parts


def _issue_worktree_git_root(issue: int) -> str | None:
    """Absolute path to the per-issue worktree IF it exists, else ``None``.

    The committed eval/figure artifacts of a `/issue` run live on the
    unmerged ``issue-<N>`` branch checked out in the canonical worktree
    ``<repo_root>/.claude/worktrees/issue-<N>`` (auto-merge to ``main`` is
    at /issue Step 10d, AFTER finalize), so the artifact verifier's git
    check must run THERE, not the MAIN checkout on ``main`` (the #685
    structural ``not tracked by git`` FAIL). This returns that worktree
    path when it exists so the launch path can bake it into the
    declaration as ``git_repo_root``.

    Crucially this is derived from the ISSUE + repo_root ALONE — NOT
    gated on ``--repo-branch`` / the ``repo_branch`` spec field. An
    explicit ``backend: runpod`` launch with no ``--repo-branch`` (the
    #685 shape) still has its artifacts on the worktree branch, so
    gating the worktree resolution on ``repo_branch`` would leave the
    fix inert for exactly that launch (concern
    ``worktree-fix-inert-when-repo-branch-absent``). Returns ``None``
    when the canonical worktree does not exist (a non-worktree launch,
    or one whose artifacts are committed directly on ``main``), so the
    declaration falls back to the established pyproject-walk root.
    """
    from explore_persona_space.task_workflow import repo_root

    wt = repo_root() / ".claude" / "worktrees" / f"issue-{issue}"
    return str(wt) if wt.exists() else None


def _launch_extra_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build ``spec.extra`` from the launch CLI's lane-specific knobs.

    Returns the dict :func:`backends.issue_dispatch.build_run_spec`
    threads through to the lane renderers. Most keys are GCP-only and
    inert on SLURM / RunPod lanes, but NOT all: ``execute_workload``
    (#909) is RunPod-honored (the RunPod execution leg), and
    ``repo_branch`` is honored by GCP (GCE clone) AND RunPod (the #909
    execution leg's branch sync). Extracted from :func:`_cmd_launch` so
    each new knob doesn't push the dispatcher over the complexity cap.
    """
    extra: dict[str, Any] = {}
    if getattr(args, "lane_suffix", None):
        # Per-lane instance-name suffix (#934): honored by the GCP lane's
        # naming helpers (eps-issue-<N>-<suffix>) + the router's attempt-id
        # mint + the handle-sidecar composer. Parse-time validated
        # (_lane_suffix_arg); absent → key ABSENT (never None-valued — a
        # None-valued key would flip canonicalize_spec output and every
        # live unsuffixed lease spec-hash).
        extra["lane_suffix"] = args.lane_suffix
    if getattr(args, "execute_workload", False):
        # RunPod-honored knob (#909): opts the launch into the RunPod
        # execution leg (RunPodBackend.launch SSHes the fresh pod, syncs
        # the clone to repo_branch, and starts workload_cmd detached).
        # Without it an explicit-runpod --workload-cmd launch is
        # provision-only (the experimenter is the executor there).
        # main()'s parse-time guard rejects the flag without a non-empty
        # --workload-cmd, so this key never rides a hydra launch.
        extra["execute_workload"] = True
    if getattr(args, "boot_disk_gb", None):
        # GCP boot disk (backends/gcp.py reads spec.extra["boot_disk_gb"]);
        # ALSO read by the RunPod CPU fallback as of #1010 — container-disk
        # threading in RunPodBackend.launch + the feasibility gate in
        # router._runpod_terminal_rung. Inert on SLURM lanes.
        extra["boot_disk_gb"] = int(args.boot_disk_gb)
    if getattr(args, "min_ram_gb", None):
        # RunPod-CPU-fallback knob (#1010): read by the feasibility gate in
        # router._runpod_terminal_rung — RunPod CPU instances have FIXED RAM,
        # so an unsatisfiable requirement refuses the fallback typed
        # (reason: cpu_fallback_infeasible_for_plan) instead of provisioning
        # an undersized pod. GCP machine selection is unchanged (by intent);
        # inert on SLURM lanes.
        extra["min_ram_gb"] = int(args.min_ram_gb)
    if getattr(args, "max_run_duration", None):
        # GCP-only knob: the instance-create renderer reads
        # spec.extra["max_run_duration"], falling back to the 7d
        # GcpConfig.default_max_run_duration (#741). Before this flag a plan's
        # declared auto-delete fence (#628: 30h for a worst-case 20h
        # wall) had no CLI path from the /issue Step 6b launch. Inert
        # on SLURM / RunPod lanes.
        extra["max_run_duration"] = args.max_run_duration
    if getattr(args, "provisioning_model", None):
        # GCP-only knob: the instance-create renderer reads
        # spec.extra["provisioning_model"] (backends/gcp.resolve_provisioning_model);
        # SPOT / FLEX_START draw the PREEMPTIBLE accelerator quota pool
        # instead of the on-demand STANDARD pool. Reaches the idle
        # preemptible capacity that is unreachable from a STANDARD launch
        # when on-demand quota is short-by-one (#537). Inert on SLURM /
        # RunPod lanes. When PRESENT, this key HARD-PINS the GCP ladder to
        # the single named provisioning model (router._gcp_ladder_specs's
        # pinned branch, #680) — so a passed STANDARD pins to on-demand-only
        # while still recording the explicit choice. When ABSENT, the ladder
        # chooses provisioning by job length (spot-first short / flex-first
        # long, #680), NOT a STANDARD default.
        extra["provisioning_model"] = args.provisioning_model
    if getattr(args, "spot_tolerant", False):
        # GCP-only knob (#656): marks the workload preemption-recoverable.
        # The #656 GCP fallback ladder fires a SPOT rung by DEFAULT for any
        # "short" job (<= EPS_GCP_SPOT_MAX_GPU_HOURS); this flag is now a
        # FORCE-spot override that makes a job "short enough" for the spot
        # rungs even past the GPU-hour threshold (an explicit opt-into-
        # preemption). The retired EPS_GCP_SPOT_FALLBACK env gate (#537) is a
        # no-op back-compat shim. Inert on SLURM / RunPod lanes.
        extra["spot_tolerant"] = True
    if getattr(args, "repo_branch", None):
        # The GCE startup script clones from origin, so a feature-branch
        # workload must name its branch (issue 535 r6); the RunPod #909
        # execution leg syncs the pod clone to the same key.
        extra["repo_branch"] = args.repo_branch
    elif (args.backend or "auto") in {"auto", "gcp"} or (
        # #909 (AC6): the auto-default ALSO fires for an explicit
        # `--backend runpod --execute-workload` launch — the execution
        # leg's branch sync would otherwise target `main`, where per-issue
        # dispatch scripts do not exist (the #763-shaped manual command).
        # Explicit --repo-branch (above) always wins.
        (args.backend or "").strip().lower() == "runpod"
        and getattr(args, "execute_workload", False)
    ):
        # fix19's production mirror (round-2 Claude Major, task #535):
        # without this, the GCE clone defaults to "main" even when the
        # invoking checkout — the /issue worktree on an issue-<N> branch
        # — carries the code under test, silently re-creating the exact
        # stale-main bug the acceptance harness already guards against
        # (router_acceptance.py r19). Same policy as the harness: default
        # to the CURRENT branch with a logged INFO. Gated to the lanes
        # that can reach GCP (explicit "gcp", or "auto"/absent — absent
        # includes frontmatter-driven backends, and an explicit SLURM /
        # RunPod lane never escalates to GCP). The extra key is no longer
        # inert on any lane: GCP honors it by git-cloning the requested
        # branch in the GCE startup script, and RunPod's lifecycle layer
        # also checks out the branch. SLURM has no honoring mechanism and
        # REFUSES to submit when repo_branch names a non-"main" branch the
        # rsync source cannot be proven to carry (its source resolves to
        # "main", not the invoking worktree) — backends/slurm.py
        # _assert_repo_branch_synced() raises, the router wraps it as a
        # BackendPrepareError, and the auto chain advances to the next lane
        # rather than silently rsyncing stale "main" code (#653 round-8).
        branch = _current_git_branch()
        if branch and branch != "main":
            logging.getLogger("dispatch_issue").info(
                "repo-branch defaulted to current branch %r for the gcp/auto/runpod-execute lane — "
                "ensure it is pushed (the GCE startup script clones from origin)",
                branch,
            )
            extra["repo_branch"] = branch
        else:
            # Invoking checkout is on main (or unresolvable) — the common
            # orchestrator topology (repo root pinned to main). Fall back to the
            # issue worktree's checked-out branch so the GCE clone carries the
            # issue's code (task #824; incident #812: a repo-root dispatch
            # silently cloned main and the issue branch's scripts were absent).
            worktree_root = _issue_worktree_git_root(args.issue)
            wt_branch = _git_branch_of(worktree_root) if worktree_root else None
            if wt_branch and wt_branch != "main":
                logging.getLogger("dispatch_issue").info(
                    "repo-branch defaulted to issue worktree branch %r "
                    "(worktree %s) for the gcp/auto/runpod-execute lane — "
                    "invoking checkout "
                    "is on main/unresolvable",
                    wt_branch,
                    worktree_root,
                )
                _warn_if_branch_not_pushed(wt_branch, worktree_root)
                extra["repo_branch"] = wt_branch
    # #685: bake the per-issue worktree git root into the declaration so
    # confirm_artifacts' git check resolves against the worktree branch
    # (where the run committed eval_results/ + figures/), not the MAIN
    # checkout on `main`. Derived from issue + repo_root ALONE (NOT gated
    # on repo_branch), so an explicit `backend: runpod` launch without
    # --repo-branch (the #685 shape) is covered too. Absent (None) when
    # there is no worktree → established pyproject-walk root.
    worktree_root = _issue_worktree_git_root(args.issue)
    if worktree_root is not None:
        extra["git_repo_root"] = worktree_root
    # #604/#661: a phase-scoped launch declares NO full-task git paths
    # (the off-pod next phase produces them); the git check then SKIPs
    # rather than FAILing on artifacts this phase never produced. Inert
    # (False) by default — every other launch keeps the full-task paths.
    if getattr(args, "skip_default_git_paths", False):
        extra["skip_default_git_paths"] = True
    return extra


def _annotate_launch_body_reconnect_and_lane(
    body: dict[str, Any], *, args: argparse.Namespace, result: Any
) -> None:
    """Additive launch-JSON keys: reconnect loudness + lane-suffix visibility (#934/#923).

    A reconnect-resolved launch dispatched NO workload this invocation,
    but must stay ``ok: true`` (the exit-75 still-waiting contract
    instructs re-running the SAME command and relying on reconnect —
    flipping ``ok`` would break that rerun loop). The additive
    ``reconnected`` / ``workload_dispatched`` keys make the non-dispatch
    machine-detectable instead. BOTH reconnect layers trip it: the
    router scan (``reason == "reconnect"``) and the GCP-internal
    ``reconnect_or_none`` (which only marks
    ``handle.extra["reconnected"]``). Mutates ``body`` in place.
    """
    from explore_persona_space.backends.router import ROUTE_REASON_RECONNECT

    handle_extra = result.handle.extra or {}
    workload_requested = bool((args.workload_cmd or "").strip()) or bool(args.hydra)
    reconnected = bool(handle_extra.get("reconnected")) or result.reason == ROUTE_REASON_RECONNECT
    if reconnected:
        workload_dispatched = False
    elif handle_extra.get("workload_executed") is False:
        # #909 provision-only RunPod launch: the pod booted but nothing ran.
        workload_dispatched = False
    else:
        workload_dispatched = workload_requested
    body["reconnected"] = reconnected
    body["workload_dispatched"] = workload_dispatched
    lane_suffix = getattr(args, "lane_suffix", None)
    if lane_suffix:
        body["lane_suffix"] = lane_suffix
    if reconnected and workload_requested:
        body["reconnect_note"] = (
            "route() RECONNECTED to an existing live instance — this invocation dispatched "
            "NO workload (workload_dispatched=false). Benign iff an earlier run of the SAME "
            "command created it (exit-75 rerun); a concurrent second lane must relaunch with "
            "--lane-suffix (#934/#923)."
        )
        logging.getLogger("dispatch_issue").warning(
            "%s pod_name=%s reason=%s",
            body["reconnect_note"],
            result.handle.pod_name,
            result.reason,
        )
    if lane_suffix and result.chosen_kind != "gcp":
        body["lane_suffix_unhonored_by_lane"] = result.chosen_kind
        logging.getLogger("dispatch_issue").warning(
            "--lane-suffix=%s: instance/job-name isolation is GCP-only; chosen_kind=%s "
            "keeps per-issue naming (SLURM eps-issue-<N>, RunPod pod-<N>) — concurrent "
            "lanes are NOT isolated on this lane.",
            lane_suffix,
            result.chosen_kind,
        )


def _persist_partial_handle_sidecar(issue: int, spec: Any, partial: Any) -> tuple[bool, str]:
    """Best-effort #954 sidecar write for a PARTIAL RunPod launch.

    A ``RunPodWorkloadStartError`` carrying a handle means a pod was
    provisioned (it BILLS, left RUNNING for diagnosis) before the
    workload-start leg failed. Persist the handle sidecar so the pod is
    visible to the handle machinery (poll / finalize / re-drive stay
    chained) — the SAME path ``dispatch_for_issue`` would have written on
    success. Returns ``(sidecar_written, note_suffix)``; an ``OSError`` is
    recorded in the note suffix, NEVER raised (the typed failure the caller
    is surfacing must not be masked).
    """
    from explore_persona_space.backends.issue_dispatch import (
        default_handle_sidecar_path,
        write_handle_sidecar,
    )

    sidecar_path = default_handle_sidecar_path(issue, lane_suffix=spec.extra.get("lane_suffix"))
    try:
        write_handle_sidecar(partial, sidecar_path)
        return True, ""
    except OSError as write_exc:
        return False, f" [handle sidecar write FAILED: {type(write_exc).__name__}: {write_exc}]"


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
    from explore_persona_space.backends.gcp import GcpCreateTimedOutStillProvisioning
    from explore_persona_space.backends.issue_dispatch import (
        build_run_spec,
        classify_terminal_exception,
        dispatch_for_issue,
    )
    from explore_persona_space.backends.router import RouteError
    from explore_persona_space.backends.runpod import RunPodWorkloadStartError

    extra = _launch_extra_from_args(args)
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
        # Exactly-one-of was already enforced at the parser surface in
        # main() (#588); normalize None → "" and strip shell-quoting
        # slop (the presence check in main() strips identically, so an
        # unstripped value can never silently flip the gate).
        workload_cmd=(args.workload_cmd or "").strip(),
    )

    # Pre-route --gpus / GCP machine-type mismatch guard (#599): the GCP
    # lane ignores the override, so fail LOUD before any backend is
    # built instead of provisioning a wrong-sized VM.
    mismatch = _gpus_gcp_lane_conflict(spec)
    if mismatch is not None:
        print(json.dumps(mismatch, sort_keys=True))
        return 2

    deps = backends_factory()
    marker_poster = deps["marker_poster"]
    if (args.backend or "").strip().lower() == "runpod":
        # GCP-first bypass visibility (incident lineage #571 → 2026-06-11:
        # three launches passed explicit ``--backend runpod`` on tasks whose
        # frontmatter was ABSENT, on the stale pre-#588 justification "the
        # GCP lane is train.py-only"). The CLI cross-checks the task's
        # ACTUAL frontmatter and classifies it 3-ways, each with a
        # DISTINCT marker flag so the dashboard can tell "bypassed auto"
        # / "contradicted a named lane" / "task hygiene problem" apart:
        #   * absent/empty/``auto`` → no frontmatter backing → LOUD
        #     warning + ``override_without_frontmatter``;
        #   * a recognized NON-runpod lane (gcp/nibi/fir/mila, or the
        #     legacy ``cluster`` alias for nibi) → the task explicitly
        #     names a DIFFERENT lane, contradicting the override even
        #     more strongly than absence → LOUD warning +
        #     ``override_conflicts_frontmatter`` (+ the value);
        #   * anything else (typo'd / non-string YAML value, e.g.
        #     ``gpc`` or ``true``) → hygiene noise masquerading as
        #     backing → LOUD warning +
        #     ``frontmatter_backend_unrecognized`` (+ the value).
        # ``backend: runpod`` is the one legitimate backing — silent.
        # ADDITIVE only — the launch is never blocked and the CLI
        # argument contract is unchanged.
        fm_backend = _frontmatter_backend_value(args.issue)
        if fm_backend in ("", "auto"):
            logging.getLogger("dispatch_issue").warning(
                "explicit --backend runpod for issue=%d but the task's frontmatter does "
                "not name a backend (absent/empty, or an explicit 'auto') — the task "
                "itself says auto, and the standing default is "
                "GCP FIRST (credits before real money). 'the GCP lane is train.py-only' "
                "is STALE justification as of #588: every lane runs custom dispatch "
                "scripts via --workload-cmd. Name a residual gap in the launch note — "
                "70B intents (no GCP machine-type mapping) / interactive SSH-MCP "
                "experimenter orchestration / runs longer than GCP --max-run-duration "
                "(default 7d) / SLURM venv-extras mismatch — or drop the override and "
                "let auto route. Launch continues; the epm:backend-selected marker "
                "carries extra.override_without_frontmatter=true so the bypass is "
                "visible on the events trail.",
                int(args.issue),
            )
            marker_poster = _wrap_marker_poster_with_override_flag(
                marker_poster, {"override_without_frontmatter": True}
            )
        elif fm_backend is None:
            logging.getLogger("dispatch_issue").warning(
                "explicit --backend runpod for issue=%d but the task frontmatter could "
                "not be read — skipping the override-without-frontmatter check "
                "(launch continues).",
                int(args.issue),
            )
        elif fm_backend == "runpod":
            # Legitimate frontmatter-backed override — silent by design.
            pass
        elif fm_backend in _recognized_frontmatter_backends():
            fm_display = (
                "cluster (legacy alias, normalizes to nibi)"
                if fm_backend == "cluster"
                else fm_backend
            )
            logging.getLogger("dispatch_issue").warning(
                "explicit --backend runpod for issue=%d CONFLICTS with the task's own "
                "frontmatter 'backend: %s' — the task explicitly names a DIFFERENT "
                "lane, which contradicts the override even more strongly than absent "
                "frontmatter would. Name the residual gap that forces RunPod in the "
                "launch note, or fix the frontmatter to match the intended lane. "
                "Launch continues; the epm:backend-selected marker carries "
                "extra.override_conflicts_frontmatter=true plus the frontmatter value "
                "so the contradiction is visible on the events trail.",
                int(args.issue),
                fm_display,
            )
            marker_poster = _wrap_marker_poster_with_override_flag(
                marker_poster,
                {"override_conflicts_frontmatter": True, "frontmatter_backend": fm_backend},
            )
        else:
            logging.getLogger("dispatch_issue").warning(
                "explicit --backend runpod for issue=%d but the task's frontmatter "
                "'backend: %s' is not a recognized backend value (router accepts %s; "
                "the legacy 'cluster' alias also counts) — likely a typo or a "
                "non-string YAML value. This is task hygiene noise masquerading as "
                "frontmatter backing, NOT a legitimate override: fix the task's "
                "backend: frontmatter. Launch continues; the epm:backend-selected "
                "marker carries extra.frontmatter_backend_unrecognized=true plus the "
                "value so the hygiene problem is visible on the events trail.",
                int(args.issue),
                fm_backend,
                sorted(_recognized_frontmatter_backends() - {"cluster"}),
            )
            marker_poster = _wrap_marker_poster_with_override_flag(
                marker_poster,
                {"frontmatter_backend_unrecognized": True, "frontmatter_backend": fm_backend},
            )
    marker_poster = _warn_default_boot_disk_ft_intent(spec, int(args.issue), marker_poster)
    try:
        outcome = dispatch_for_issue(
            spec,
            runpod_backend=deps["runpod_backend"],
            free_backends=deps["free_backends"],
            gcp_backend=deps["gcp_backend"],
            mila_socket_alive=deps["mila_socket_alive"],
            marker_poster=marker_poster,
            is_started=deps["is_started"],
            is_live_after_cancel=deps["is_live_after_cancel"],
            started_evidence_probe=deps.get("started_evidence_probe"),
            reconnect_fn=deps["reconnect_fn"],
        )
    except RunPodWorkloadStartError as exc:
        # #909 AC3: a requested --execute-workload execution that did not
        # start NEVER returns ok. RunPodBackend.launch raises the typed
        # error UNWRAPPED through route()'s explicit-runpod override
        # (_prepare_and_launch only wraps prepare() failures, and
        # dispatch_for_issue does not wrap route()), so this arm catches it
        # directly; were a future router change to wrap it in a RouteError,
        # the arm below already yields the same exit 2 + failure JSON.
        #
        # #954: when the error carries the PARTIAL handle (a pod was
        # provisioned before the start leg failed — it BILLS, left RUNNING
        # for diagnosis), persist the handle sidecar best-effort (never mask
        # the typed failure) and name the pod in the failure JSON. NO lease
        # on this path by design: a manual retry hits pod_lifecycle's
        # provision-idempotency refuse (exit 1 on a live pod-N), fail-loud.
        note = str(exc)
        partial_fields: dict[str, Any] = {}
        partial = getattr(exc, "handle", None)
        if partial is not None:
            sidecar_written, note_suffix = _persist_partial_handle_sidecar(
                int(args.issue), spec, partial
            )
            note += note_suffix
            partial_fields = {"pod_name": partial.pod_name, "sidecar_written": sidecar_written}
        body = {
            "ok": False,
            "issue": int(args.issue),
            "exception": type(exc).__name__,
            "failure_class": "infra",
            "reason": "runpod_workload_start_failed",
            "note": note,
            **partial_fields,
        }
        print(json.dumps(body, sort_keys=True))
        return 2
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
    except subprocess.CalledProcessError as exc:
        if not _provision_still_waiting(exc):
            raise
        # pod_lifecycle.py provision's bounded wait-for-capacity loop hit
        # its per-process wall-clock budget (exit 75, EX_TEMPFAIL) — a
        # still-waiting outcome, NOT a failure (incident #603). The wait
        # loop is state-free, so the caller re-runs the SAME launch
        # command to continue waiting. Deliberately NO ``failure_class``
        # / ``status`` keys: the orchestrator must not post
        # ``epm:failure v1`` or ``set-status blocked`` on this exit.
        body = {
            "ok": False,
            "issue": int(args.issue),
            "still_waiting": True,
            "rerun": True,
            "reason": "wait_for_capacity_budget_reached",
            "note": (
                "pod_lifecycle.py provision exited 75 (EX_TEMPFAIL): its bounded "
                "wait-for-capacity loop reached the per-process wall-clock budget "
                "while RunPod capacity / the fleet burn cap kept the provision "
                "queued. Still waiting, not a failure — the wait loop is "
                "state-free, so re-run the SAME dispatch_issue.py launch command "
                "to continue waiting. Do not post epm:failure or set-status "
                "blocked on this exit."
            ),
        }
        print(json.dumps(body, sort_keys=True))
        return EXIT_STILL_WAITING
    except GcpCreateTimedOutStillProvisioning as exc:
        # The GCP-lane second producer of exit 75 (#736): a
        # ``gcloud compute instances create`` on a FLEX_START rung exceeded
        # the 300s subprocess cap, but a post-timeout ``instances list``
        # probe found the instance live server-side — a FLEX_START
        # preemptible-queueing state, NOT a failure. Mirror the #603
        # still-waiting contract exactly: deliberately NO ``failure_class``
        # / ``status`` keys, so the orchestrator does NOT post
        # ``epm:failure v1`` / ``set-status blocked``. The caller re-runs
        # the SAME launch command; ``reconnect_or_none`` (the idempotent
        # re-entry at the top of ``GcpBackend.launch``) reconnects to the
        # live instance with NO double-create. ``GcpCreateTimedOutStillProvisioning``
        # is a ``GcpBackendError`` (NOT a ``RouteError`` / ``CalledProcessError``),
        # so it shares a base with neither existing arm and arm order is moot.
        body = {
            "ok": False,
            "issue": int(args.issue),
            "still_waiting": True,
            "rerun": True,
            "reason": "gcloud_create_timeout_still_provisioning",
            "instance_name": exc.instance_name,
            "instance_status": exc.status,
            "note": (
                "gcloud compute instances create exceeded the 300s subprocess cap, "
                "but a post-timeout instances-list probe found the instance "
                f"{exc.instance_name} live (status={exc.status}) — a FLEX_START "
                "preemptible-queueing state, NOT a failure. Re-run the SAME "
                "dispatch_issue.py launch command to continue waiting; "
                "reconnect_or_none reconnects to the live instance with no "
                "double-create. Do not post epm:failure or set-status blocked "
                "on this exit."
            ),
        }
        print(json.dumps(body, sort_keys=True))
        return EXIT_STILL_WAITING

    result = outcome.result
    handle_extra = result.handle.extra or {}
    # #909 belt-and-suspenders fail-fast: a handle that CLAIMS the workload
    # executed but carries no workload_pid is a backend regression returning
    # ok on a provision-only result — never print ok:true on it.
    if handle_extra.get("workload_executed") and not handle_extra.get("workload_pid"):
        body = {
            "ok": False,
            "issue": int(args.issue),
            "failure_class": "infra",
            "reason": "runpod_workload_start_failed",
            "note": (
                "handle claims workload_executed=true but carries no workload_pid — "
                "backend regression; treating the requested execution as NOT started "
                f"(#909). pod_name={result.handle.pod_name} log_path={result.handle.log_path}"
            ),
        }
        print(json.dumps(body, sort_keys=True))
        return 2
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
        # #909: the execution-leg outcome (RunPod lane; None / absent-key
        # semantics on lanes that never set them) + the log path the caller
        # tails. workload_executed False on a provision-only RunPod launch
        # is the loud "the experimenter must launch this" signal.
        "workload_executed": handle_extra.get("workload_executed"),
        "workload_pid": handle_extra.get("workload_pid"),
        "log_path": result.handle.log_path,
    }
    _annotate_launch_body_reconnect_and_lane(body, args=args, result=result)
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


# The upload-verifier agent's verdict line inside an
# ``epm:upload-verification`` marker note (shape: ``**Verdict: PASS**``;
# see workflow.yaml § markers). Case-sensitive on purpose — the schema
# emits uppercase PASS/FAIL, and prose mentions of "pass" must not match.
_UPLOAD_VERIFICATION_PASS_RE = re.compile(r"Verdict:\s*PASS\b")


def _agent_upload_verification_passed(issue: int) -> bool:
    """Agent-level upload-verification PASS evidence on the task's events.jsonl.

    The finalize degrade path (handle carries no ``expected_artifacts``
    declaration — see :func:`_cmd_finalize`) consults this instead of the
    structurally-unsatisfiable mechanical gate. Two acceptable forms of
    evidence, mirroring SKILL.md Step 8:

    * an ``epm:upload-verified`` marker (the sticky PASS the skill posts
      right before the auto-terminate path), or
    * the LATEST ``epm:upload-verification`` marker whose note carries
      ``Verdict: PASS`` (latest wins — a FAIL → upload-fix → re-verify
      loop posts a fresh marker each round).

    Reads via ``task_workflow.find_task_path``, which resolves against the
    MAIN checkout's ``tasks/`` tree regardless of the invoking worktree
    (the resolver branch-guards to ``main``), so a finalize run from an
    ``issue-<N>`` worktree still reads the canonical markers.

    ANY read failure (missing task, unreadable events.jsonl) returns
    ``False`` after a logged warning — the safe direction: no evidence ⇒
    the caller keeps the exit-3 teardown-skip; we never tear down on a
    guess.
    """
    log = logging.getLogger("dispatch_issue")
    try:
        from explore_persona_space.task_workflow import find_task_path

        events_path = find_task_path(int(issue)) / "events.jsonl"
        if not events_path.exists():
            return False
        saw_sticky_pass = False
        latest_verification_note: str | None = None
        with events_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                kind = str(event.get("kind", ""))
                if kind == "epm:upload-verified":
                    saw_sticky_pass = True
                elif kind == "epm:upload-verification":
                    latest_verification_note = str(event.get("note", ""))
    except Exception as exc:
        log.warning(
            "could not read upload-verification evidence for issue=%d (%s: %s); "
            "treating as NO evidence (teardown stays gated)",
            int(issue),
            type(exc).__name__,
            exc,
        )
        return False
    if saw_sticky_pass:
        return True
    if latest_verification_note is not None:
        return bool(_UPLOAD_VERIFICATION_PASS_RE.search(latest_verification_note))
    return False


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

    Degrade path: when the handle carries NO ``expected_artifacts``
    declaration the mechanical gate is structurally unsatisfiable.
    Every launch path (GCP, SLURM, RunPod) populates the declaration as
    of #598, so a declaration-less handle is a pre-#598 in-flight
    sidecar only; a confirm FAIL on one falls back to the agent-level
    upload-verification PASS evidence on the task's ``events.jsonl``
    (:func:`_agent_upload_verification_passed`). Evidence found →
    teardown proceeds with a LOUD log + a ``confirm_artifacts`` field in
    the output JSON; no evidence → exit 3 with
    ``reason: confirm_artifacts_no_declaration``. A handle WITH a
    declaration never degrades — a real mechanical FAIL always exits 3.

    After a SUCCESSFUL teardown the sidecar is renamed to
    ``<name>.finalized`` (audit record, never deleted) so a later
    finalize for the same issue cannot tear down a fresh run through
    the stale handle; the duplicate tick then no-ops with the benign
    rc=2 ``missing_handle_sidecar`` shape (Mn4.3).
    """
    from explore_persona_space.backends.issue_dispatch import (
        read_handle_sidecar,
        resolve_handle_sidecar_path,
    )

    # Canonical <main-checkout>/.claude/cache/ path first, then the
    # legacy <cwd>/.claude/cache/ location (back-compat with sidecars
    # written by the pre-#612 cwd-relative composer — a finalize that
    # false-misses a live handle would SKIP teardown and leak a paid
    # VM / pod, so the probe is cheap insurance during the transition).
    sidecar, probed = resolve_handle_sidecar_path(
        args.issue, args.handle_file, lane_suffix=getattr(args, "lane_suffix", None)
    )
    if not Path(sidecar).exists():
        body = {
            "ok": False,
            "issue": int(args.issue),
            "failure_class": "infra",
            "reason": "missing_handle_sidecar",
            "detail": f"no sidecar at any probed path: {', '.join(str(p) for p in probed)}",
        }
        print(json.dumps(body, sort_keys=True))
        return 2

    handle = read_handle_sidecar(Path(sidecar))
    deps = backends_factory()
    backend = _resolve_backend_for_handle(handle, deps)

    # ``fetch_results`` BEFORE the confirm gate (#588 / latent slice-6
    # gap): the GCP completion sentinel lives ON the VM — ``GcpBackend.
    # fetch_results`` is the scp pull that lands it locally, and the
    # slice-2 verifier reads the LOCAL filesystem. Without this call
    # every real GCP finalize FAILed confirm on the missing local
    # sentinel. Matches the base.py ABC ordering (fetch_results →
    # confirm_artifacts → teardown). fetch_results is fail-soft by its
    # own two-tier contract — but wrap defensively: a fetch CRASH must
    # surface as the confirm FAIL (right surfacing, evidence preserved),
    # not as a finalize traceback.
    try:
        backend.fetch_results(handle)
    except Exception as exc:
        logging.getLogger("dispatch_issue").error(
            "finalize: fetch_results FAILED for issue=%d (%s: %s); continuing to the "
            "confirm_artifacts gate — a missing local sentinel will FAIL confirm with "
            "the right surfacing (teardown skipped, evidence preserved).",
            int(args.issue),
            type(exc).__name__,
            exc,
        )

    confirm_degraded: str | None = None
    if not args.skip_confirm_artifacts:
        passed = backend.confirm_artifacts(handle)
        if not passed:
            from explore_persona_space.backends.artifacts import (
                EXPECTED_ARTIFACTS_HANDLE_KEY,
            )

            extra = getattr(handle, "extra", None) or {}
            declaration_missing = EXPECTED_ARTIFACTS_HANDLE_KEY not in extra
            if declaration_missing and _agent_upload_verification_passed(args.issue):
                # Graceful degrade (incident #585, 2026-06-11): a
                # declaration-less handle made the mechanical gate
                # structurally unsatisfiable, and a hard exit 3 forced
                # orchestrators to bypass finalize with a raw ``pod.py
                # terminate`` — losing the Mn4.3 sidecar retirement
                # below (a stale sidecar can mis-target a LATER
                # finalize). As of #598 every launch path (GCP, SLURM,
                # RunPod) populates the declaration, so this branch
                # serves pre-#598 in-flight handles only. Teardown
                # still requires POSITIVE verification evidence: the
                # agent-level upload-verifier PASS marker on the task.
                # This branch never fires when a declaration IS present
                # — a real mechanical FAIL keeps the exit-3
                # evidence-preserving behavior unconditionally.
                confirm_degraded = "skipped_no_declaration_agent_pass"
                logging.getLogger("dispatch_issue").warning(
                    "finalize: handle for issue=%d carries no 'expected_artifacts' "
                    "declaration (launch path did not populate it) — mechanical "
                    "confirm_artifacts gate is unsatisfiable. Agent-level "
                    "upload-verification PASS evidence found on the task; "
                    "proceeding to teardown on that evidence.",
                    int(args.issue),
                )
            elif declaration_missing:
                body = {
                    "ok": False,
                    "issue": int(args.issue),
                    "phase": "confirm_artifacts",
                    "chosen_kind": handle.backend,
                    "pod_name": handle.pod_name,
                    "reason": "confirm_artifacts_no_declaration",
                    "detail": (
                        "handle.extra carries no 'expected_artifacts' declaration "
                        "AND no agent-level upload-verification PASS marker was "
                        "found on the task — teardown SKIPPED. Recover by running "
                        "the upload-verifier to a PASS (epm:upload-verification, "
                        "Verdict: PASS) and re-running finalize, or re-run with "
                        "--skip-confirm-artifacts if the run crashed before "
                        "artifacts could land."
                    ),
                }
                print(json.dumps(body, sort_keys=True))
                return 3
            else:
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
    if confirm_degraded is not None:
        body["confirm_artifacts"] = confirm_degraded
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


# gcloud composed-duration shape: one or more integer+unit groups
# ("30h", "1d12h", "90m", "86400s"). Bare integers are REJECTED — gcloud
# would read them as seconds, which is never what a plan's "30h" fence
# means when the unit is dropped by accident.
_MAX_RUN_DURATION_RE = re.compile(r"(?:\d+[dhms])+")


def _max_run_duration_arg(value: str) -> str:
    """Validate a gcloud-shaped duration for ``--max-run-duration``.

    Accepts the composed integer+unit form gcloud's
    ``--max-run-duration`` parses (``30h``, ``1d12h``, ``90m``);
    anything else — bare integers (ambiguous unit), negatives,
    fractions, embedded spaces — raises ``argparse.ArgumentTypeError``
    at the parser surface so a typo'd fence fails the launch BEFORE a
    VM is provisioned with the wrong auto-delete bound.
    """
    v = value.strip()
    if not _MAX_RUN_DURATION_RE.fullmatch(v):
        raise argparse.ArgumentTypeError(
            f"--max-run-duration {value!r} does not match the gcloud duration "
            "shape (integer+unit groups, units d/h/m/s: '30h', '1d12h', '90m')"
        )
    return v


def _lane_suffix_arg(value: str) -> str:
    """Validate ``--lane-suffix`` at the parser surface (#934).

    Delegates to the shared ``base.validate_lane_suffix`` (lowercase
    ``[a-z0-9-]``, <=43 chars — the attempt-label budget) so a malformed
    suffix errors friendly at parse time, BEFORE any backend is built.
    """
    from explore_persona_space.backends.base import validate_lane_suffix

    try:
        return validate_lane_suffix(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


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
    launch.add_argument(
        "--gpus",
        type=int,
        default=None,
        help=(
            "Override GPU count. Honored by the RunPod (--gpu-count) and SLURM "
            "(--gres) lanes; the GCP lane sizes its VM from --intent alone "
            "(backends/gcp.INTENT_TO_MACHINE), so a gcp-reachable launch (explicit "
            "gcp, or auto with gcp in the lane order) whose override mismatches the "
            "intent's machine is refused up front (exit 2, "
            "reason: gpus_machine_mismatch) instead of provisioning a wrong-sized "
            "VM (incident #599)."
        ),
    )
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
        "--provisioning-model",
        type=str,
        choices=["STANDARD", "SPOT", "FLEX_START"],
        default=None,
        help=(
            "GCP provisioning model (GCP lane only; threads to "
            "spec.extra['provisioning_model'], read by the instance-create "
            "renderer's --provisioning-model flag). SPOT / FLEX_START draw "
            "the PREEMPTIBLE accelerator quota pool, reaching idle "
            "preemptible capacity unreachable from a STANDARD launch when "
            "on-demand quota is short-by-one (issue 537: 4-GPU ft-7b failed "
            "with 3 free on-demand A100 while 16 preemptible sat idle). "
            "Default (absent): the length-aware GCP ladder chooses the "
            "provisioning order (#680 — spot-first for short jobs, "
            "flex-first for long/unknown-length jobs, on-demand last); a "
            "passed value HARD-PINS the ladder to that single provisioning "
            "model. Inert on non-GCP lanes."
        ),
    )
    launch.add_argument(
        "--spot-tolerant",
        action="store_true",
        help=(
            "Mark the workload preemption-recoverable (GCP lane only; threads "
            "to spec.extra['spot_tolerant']). #656: the GCP fallback ladder "
            "fires a SPOT rung by DEFAULT for any short job "
            "(<= EPS_GCP_SPOT_MAX_GPU_HOURS, default 2 GPU-h); this flag is a "
            "FORCE-spot override that opts a LONGER job into the spot rungs "
            "past the threshold (explicit opt-into-preemption). The retired "
            "EPS_GCP_SPOT_FALLBACK env gate (#537) is a no-op shim. Set it for "
            "scoring / eval rounds that checkpoint per phase. Inert on non-GCP "
            "lanes."
        ),
    )
    launch.add_argument(
        "--boot-disk-gb",
        type=int,
        default=None,
        help=(
            "Boot/data disk size the plan's stage requires, in GB. GCP lane: "
            "boot-disk size override (threads to spec.extra['boot_disk_gb'], "
            "honored by backends/gcp.py; default 300 GB is too tight for "
            "full-FT checkpoint grids — issue 606 needed 500: 13 consolidated "
            "ZeRO-3 ckpts ~= 195 GB + model + cache). RunPod CPU fallback "
            "(#1010): threaded into the pod's containerDiskInGb "
            "(max(50, value)) and checked by the feasibility gate — an "
            "unsatisfiable disk requirement refuses the fallback typed "
            "(cpu_fallback_infeasible_for_plan) instead of provisioning an "
            "undersized pod. Inert on SLURM lanes."
        ),
    )
    launch.add_argument(
        "--min-ram-gb",
        type=int,
        default=None,
        help=(
            "Minimum RAM (GB) the plan's CPU stage requires. Read by the "
            "RunPod CPU fallback feasibility gate (#1010) — RunPod CPU "
            "instances have FIXED RAM, so an unsatisfiable requirement "
            "refuses the fallback with reason cpu_fallback_infeasible_for_plan "
            "instead of provisioning an undersized pod. GCP machine selection "
            "is unchanged (by intent). Inert on SLURM lanes."
        ),
    )
    launch.add_argument(
        "--max-run-duration",
        type=_max_run_duration_arg,
        default=None,
        help=(
            "GCP VM auto-delete fence override (GCP lane only; threads to "
            "spec.extra['max_run_duration'], read by the instance-create "
            "renderer next to --instance-termination-action=DELETE). "
            "gcloud duration shape — integer+unit groups, e.g. '30h', "
            "'1d12h', '90m'. The 7d default "
            "(GcpConfig.default_max_run_duration, #741) lets a multi-day "
            "workload run to the FLEX_START ceiling without being stranded "
            "mid-run (#697); pin a SHORTER fence here to bound an orphaned VM "
            "sooner. A workload genuinely needing >7d cannot run on the GCP "
            "FLEX_START lane at all (7d is the GCP ceiling) — pin "
            "backend: runpod. Inert on non-GCP lanes."
        ),
    )
    launch.add_argument(
        "--hydra",
        action="append",
        default=None,
        help=(
            "Hydra override (e.g. ``condition=c1``). Repeatable. "
            "Mutually exclusive with --workload-cmd; exactly one of the two is required."
        ),
    )
    launch.add_argument(
        "--workload-cmd",
        type=str,
        default=None,
        help=(
            'Custom repo-relative shell command (e.g. "bash scripts/issue<N>_dispatch.sh"). '
            "Executed verbatim by the lane renderers from the repo checkout root after env "
            "bootstrap. GCP lane: may be blocking or self-daemonizing — a detached "
            "(setsid-forked) workload MUST write its pid to a fresh file under "
            "/workspace/logs/*.pid; the GCP startup script waits on it before declaring "
            "done (#601). SLURM lanes (nibi/fir/mila): the command MUST BLOCK until the "
            "workload finishes — the sbatch terminal block + job COMPLETED fire on command "
            "return and the job-exit cgroup teardown kills detached children (no /workspace "
            "pid contract exists there; #601 follow-up). Mutually "
            "exclusive with --hydra; exactly one of the two is required (#588)."
        ),
    )
    launch.add_argument(
        "--execute-workload",
        action="store_true",
        help=(
            "RunPod lane only (#909): after provisioning, SSH the fresh pod, sync "
            "its clone to --repo-branch (auto-defaulted for this shape), and start "
            "--workload-cmd detached (setsid launcher + pidfile + log), verifying "
            "liveness before returning — a requested execution that did not start "
            "exits 2 with reason=runpod_workload_start_failed, never ok:true. "
            "REQUIRES a non-empty --workload-cmd (rejected at parse time with "
            "--hydra: the execution leg cannot execute a hydra run). Without this "
            "flag an explicit-runpod --workload-cmd launch is provision-only — "
            "EXPECTED when the experimenter (SKILL.md Step 6d.1) launches it on "
            "the pod. The automated GCP→RunPod failover paths opt in "
            "automatically (they have no experimenter). Inert on non-RunPod lanes."
        ),
    )
    launch.add_argument(
        "--skip-default-git-paths",
        action="store_true",
        help=(
            "Phase-scoped launch (#604/#661): omit the auto "
            "eval_results/issue_<N>/ + figures/issue_<N>/ from the "
            "expected-artifacts declaration. The git check then SKIPs (this "
            "phase produces no git artifacts — e.g. a P3 extraction whose "
            "deliverable is on the HF data repo under analysis_tensors/, with "
            "the off-pod P5 analysis phase producing the git files NEXT). The "
            "HF + completion-sentinel checks STILL run, so the gate is NOT "
            "relaxed. Inert when the launch DOES commit git artifacts. Lane-"
            "agnostic (threads into spec.extra['skip_default_git_paths'], "
            "honored by every lane's declaration builder)."
        ),
    )
    launch.add_argument(
        "--lane-suffix",
        type=_lane_suffix_arg,
        default=None,
        help=(
            "Per-lane instance-name suffix (#934): the GCP lane provisions "
            "eps-issue-<N>-<suffix> and the handle sidecar becomes "
            "issue-<N>-<suffix>-handle.json, so two concurrent lanes for one "
            "issue coexist. Lowercase [a-z0-9-], <=43 chars. Instance/job-name "
            "isolation is GCP-lane only; SLURM job names and RunPod pod names "
            "remain per-issue (concurrent lanes both failing over to RunPod "
            "still contend on pod-<N>). In a multi-lane plan, suffix BOTH "
            "lanes — an unsuffixed lane 1 plus a suffixed lane 2 leaves a "
            "forgotten-suffix poll/finalize silently resolving lane 1's "
            "sidecar. Rerunning a suffixed launch WITHOUT the flag creates a "
            "second unsuffixed instance (no reconnect). Multi-lane "
            "orchestrators should prefer passing --handle-file from the "
            "launch JSON's handle_sidecar_path to poll/finalize."
        ),
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
        "(default: <main-checkout>/.claude/cache/issue-<N>-handle.json, "
        "with a legacy <cwd>/.claude/cache/ fallback probe).",
    )
    finalize.add_argument(
        "--lane-suffix",
        type=_lane_suffix_arg,
        default=None,
        help=(
            "Resolve the per-lane handle sidecar issue-<N>-<suffix>-handle.json "
            "(#934). Ignored when --handle-file is given. Pass the SAME suffix "
            "the launch used — a forgotten suffix silently finalizes the "
            "unsuffixed lane's handle instead."
        ),
    )
    finalize.add_argument(
        "--skip-confirm-artifacts",
        action="store_true",
        help=(
            "Skip the confirm_artifacts gate (matches "
            "``pod.py terminate --skip-upload-verify``; use when the "
            "experiment crashed before artifacts could land, or when a "
            "phase-scoped launch's declaration names artifacts only LATER "
            "VM-local phases produce — verify the phase deliverable on "
            "permanent storage first; incident #604)."
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
    # Load credential env BEFORE any subprocess spawns: `uv run python`
    # does NOT auto-load .env, and env={**os.environ} propagates the
    # parent's emptiness otherwise (issue #397 round-10' launch burn;
    # same contract as router_acceptance.py main()).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.action == "launch":
        # Exactly one of --workload-cmd / --hydra (#588). An explicitly-
        # empty ``--workload-cmd ''`` counts as not-provided (an empty
        # command can never be a workload) and errors with the same
        # message. parser.error prints usage + exits 2 — a friendlier
        # surface than the RunSpec.__post_init__ traceback, and it fires
        # BEFORE any backend is built.
        has_workload_cmd = bool((args.workload_cmd or "").strip())
        has_hydra = bool(args.hydra)
        if has_workload_cmd == has_hydra:
            parser.error(
                "launch requires exactly one of --workload-cmd / --hydra "
                f"(got {'both' if has_hydra else 'neither'}; an empty --workload-cmd '' "
                "counts as not provided)"
            )
        # #909 AC3a (upheld Must-Fix): --execute-workload with nothing to
        # execute (a --hydra run, or an empty/absent --workload-cmd) would
        # recreate the #763 silent false-green through the fix's own flag
        # surface — the execution leg would no-op, no WARNING would fire,
        # and ok:true would print on a paid provision-only pod. Reject at
        # parse time, BEFORE backends_factory is built or any provision
        # attempted (mirrors the exactly-one-of guard above).
        if getattr(args, "execute_workload", False) and not has_workload_cmd:
            parser.error(
                "--execute-workload requires a non-empty --workload-cmd "
                "(it cannot execute a --hydra run)"
            )
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
    "EXIT_STILL_WAITING",
    "_agent_upload_verification_passed",
    "_build_production_backends",
    "_cmd_finalize",
    "_cmd_launch",
    "_frontmatter_backend_value",
    "_gpus_gcp_lane_conflict",
    "_provision_still_waiting",
    "_recognized_frontmatter_backends",
    "_resolve_backend_for_handle",
    "_wrap_marker_poster_with_override_flag",
    "main",
]
