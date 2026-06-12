"""GCP Compute Engine VM backend (single-VM ephemeral lifecycle).

The third concrete :class:`~base.ComputeBackend` after :class:`RunPodBackend`
and :class:`SlurmBackend`. Mirrors the RunPod lifecycle over GCE VMs so the
router's auto chain (free → GCP credits) can burn the ~$100k GFS credit
pool (expires Aug 2, 2026) without bolting a different orchestration shape
onto the same pipeline.

Plan ``2026-06-08_224537-multi-backend-compute-router`` § ``gcp.py``.

What this slice ships
---------------------

* :class:`GcpBackend` — implements every method on :class:`ComputeBackend`
  by shelling out to ``gcloud`` (per the plan: "start by shelling out;
  migrate to ``google-cloud-compute`` only if typed errors are wanted").
* Intent → machine-type table (:data:`INTENT_TO_MACHINE`): ``lora-7b`` /
  ``lora`` → ``a2-ultragpu-1g`` (1x A100-80); ``ft-7b`` → ``a2-ultragpu-4g``
  (4x A100-80); ``eval`` → ``g2-standard-4`` (1x L4); ``debug`` → ``g2-standard-4``.
* :class:`GcpConfig` — per-call knobs (project, gcloud config name, zone +
  fallback zones, DLVM image family + project, default provisioning model,
  scratch path on the VM). No hardcoding inline; tests construct test
  :class:`GcpConfig` instances.
* :func:`render_startup_script` — pure function returning the startup-script
  the VM runs. Mirrors :func:`scripts.bootstrap_pod.sh` (git clone/pull +
  ``uv sync`` + ``.env`` push + HF cache redirect + invokes the workload).
  Custom ``workload_cmd`` runs are assumed BLOCKING; a self-daemonizing
  driver must write its detached pid to a fresh ``/workspace/logs/*.pid``
  file, which the script waits on before writing the completion sentinel
  (#601 — otherwise the poll reads terminal-success mid-run).
* :func:`render_create_argv` — pure function returning the ``gcloud compute
  instances create`` argv for a given (spec, config). Golden-tested.
* :func:`reconnect_or_none` — pre-launch idempotent reconnect via ``gcloud
  compute instances list --filter=name=eps-issue-<N>``. If a live instance
  exists, return a handle for it without re-provisioning.
* :func:`audit_stale_gcp_vms` — analogue of ``scripts/pod.py audit-stale``;
  lists ``eps-issue-*`` instances older than a threshold and deletes them.
  Cron wiring is the orchestrator's responsibility — this exposes the
  callable that the cron / a ``scripts/`` entry can invoke.
* Typed failure classifications: :class:`GcpProvisioningError` (capacity /
  quota / SSH bring-up) → the router falls back to the next tier;
  :class:`GcpWorkloadError` (a real workload exception after the VM is up)
  → surfaced, not auto-fallback'd (the router's contract: a workload
  failure observed AFTER ``[phase=...]`` started is NEVER auto-fallback'd
  because the next-tier re-run would reproduce the bug).
* Spot preemption recovery: a preempt produces a fresh idempotent re-run
  (artifacts pushed off-VM during the run are already there; the new
  attempt-id namespaces the next run so prior outputs aren't overwritten).

What this slice DOES NOT do
---------------------------

* Run a real GCE VM from tests. Unit-only; the live acceptance is the
  per-lane acceptance run (plan step 8). Every ``gcloud`` call goes
  through an injected ``runner`` callable so tests run with no network.
* Implement the slice-5 router. ``GcpBackend`` is consumed by the router
  via the existing :class:`ComputeBackend` interface; the router itself
  is a separate slice.
* Probe live GCP capacity for an estimate. ``estimate_start_seconds``
  returns 0 (on-demand provisions immediately; Spot is ~0 when capacity
  exists). Live capacity probing is deferred to v1.1.
* Push artifacts. Artifacts are pushed BY THE WORKLOAD during the run
  (HF Hub / WandB, per the Upload Policy) — this backend does not
  re-implement that path. ``fetch_results`` is a best-effort scp BEFORE
  delete (in case the workload missed something the verifier needs);
  the authoritative artifacts are already off-VM by the time the backend
  reads them.

Hard-coded facts (verified 2026-06-08 in ``~/my-goat/reference/gcp-compute
-execution-2026-06.md``):

* Project: ``eps-persona-gpu-jun2026`` (proj # 796887979789), linked to
  the GFS billing account so all spend draws the credit pool.
* gcloud config: ``eps-gcp`` (logged in as ``emanuel@nuclearsoftware.com``);
  EVERY ``gcloud`` call carries ``--configuration=eps-gcp`` (per-command,
  not env var, per the plan's "no ambient state" rule).
* DLVM image: ``pytorch-2-9-cu129-ubuntu-2204-nvidia-580`` in project
  ``deeplearning-platform-release`` (the same family the $1 credit-draw
  test on 2026-06-08 used).
* Zone: ``us-central1-a`` (where the GFS A100-80 quota lives).
* Quota: A100-80 standard=8, Spot=8, A2-CPUs=96 in us-central1, global
  GPUS_ALL_REGIONS=360 — auto-approved org-pre-boosted.

References:
* ``src/explore_persona_space/backends/runpod.py`` for the lifecycle shape
  this module mirrors.
* ``src/explore_persona_space/backends/slurm.py`` for the per-call
  ``runner`` / marker-poster injection pattern.
* ``src/explore_persona_space/backends/artifacts.py`` (slice 2): the
  :func:`confirm_artifacts_from_handle` core ``confirm_artifacts``
  delegates to. The slice-2 verifier FAILs an all-SKIP declaration so the
  launch path MUST populate :data:`EXPECTED_ARTIFACTS_HANDLE_KEY` with at
  least the completion sentinel.
* ``scripts/bootstrap_pod.sh`` for the bootstrap recipe ``render_startup_script``
  mirrors.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.backends.artifacts import (
    DEFAULT_HF_DATA_REPO,
    DEFAULT_HF_MODEL_REPO,
    EXPECTED_ARTIFACTS_HANDLE_KEY,
    SENTINEL_FILENAME,
)
from explore_persona_space.backends.base import (
    BackendKind,
    BackendProbeError,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-call config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GcpConfig:
    """Per-call knobs for the GCP backend.

    Everything project / image / zone specific lives here so the renderer
    + lifecycle helpers stay generic. A future change to the credited
    project, the DLVM family, or the primary zone is a config-only edit;
    tests construct test :class:`GcpConfig` instances with mocked names.

    Fields:

    * ``project`` — GCP project id linked to the credited billing account.
      Defaults to :data:`DEFAULT_PROJECT` (the dedicated EPS project).
    * ``gcloud_config`` — gcloud configuration name carrying the
      ``emanuel@nuclearsoftware.com`` credentials. Defaults to
      :data:`DEFAULT_GCLOUD_CONFIG`. EVERY shelled ``gcloud`` call
      threads this via ``--configuration=<name>`` so the backend NEVER
      depends on the ambient ``CLOUDSDK_ACTIVE_CONFIG_NAME`` env var
      (which is shared with my-goat / personal use).
    * ``primary_zone`` — first-choice GCE zone. Defaults to
      :data:`DEFAULT_PRIMARY_ZONE`. The GFS A100-80 quota lives in
      ``us-central1``; ``us-central1-a`` is the default.
    * ``fallback_zones`` — additional zones (same region) to try on a
      ``ZONE_RESOURCE_POOL_EXHAUSTED`` provisioning failure. Tried in
      order. Defaults to ``us-central1-b``, ``us-central1-c`` (same
      region so the GPUS_ALL_REGIONS quota covers all of them).
    * ``image_family`` / ``image_project`` — DLVM image. Defaults to the
      pytorch-2-9 family in ``deeplearning-platform-release`` (the family
      the $1 credit-draw test used on 2026-06-08).
    * ``default_boot_disk_gb`` — boot-disk size. 300 GB is the Upload Policy
      working-set headroom (model + checkpoints + HF cache + venv).
    * ``default_boot_disk_type`` — ``pd-ssd`` (the ``pd-balanced`` default
      is markedly slower for the model-load + HF-cache write path).
    * ``default_max_run_duration`` — VM auto-delete fence. Defaults to
      ``24h`` — generous enough to never interrupt an upload but short
      enough that an orphaned VM caps the credit burn at one day's worth.
      Tunable per spec via ``RunSpec.time_budget_hours`` (the renderer
      converts to ``<H>h`` for gcloud).
    * ``vm_scratch_dir`` — workload scratch root on the VM (where the
      sentinel + rsync'd repo land). Mirrors the RunPod ``/workspace``
      convention so workloads share filesystem layout across backends.
    * ``repo_url`` — git URL the startup-script clones from. Public
      HTTPS is fine for the open repo; private slices would extend
      ``render_startup_script`` to push a deploy key.
    * ``hf_data_repo`` / ``hf_model_repo`` — overrides for the artifact-
      verifier declaration. Defaults to the canonical EPS repos.
    """

    project: str = ""
    gcloud_config: str = ""
    primary_zone: str = ""
    fallback_zones: tuple[str, ...] = ()
    image_family: str = ""
    image_project: str = ""
    default_boot_disk_gb: int = 300
    default_boot_disk_type: str = "pd-ssd"
    default_max_run_duration: str = "24h"
    vm_scratch_dir: str = "/workspace"
    repo_url: str = ""
    hf_data_repo: str = DEFAULT_HF_DATA_REPO
    hf_model_repo: str = DEFAULT_HF_MODEL_REPO


#: Canonical project id linked to the credited GFS billing account.
DEFAULT_PROJECT = "eps-persona-gpu-jun2026"

#: Canonical gcloud configuration name carrying the right account.
#: Verified live 2026-06-08; threaded as ``--configuration=<name>`` per call
#: so the ambient ``CLOUDSDK_ACTIVE_CONFIG_NAME`` (which my-goat manipulates)
#: never silently mis-routes a backend call to a personal project.
DEFAULT_GCLOUD_CONFIG = "eps-gcp"

#: First-choice zone. The GFS A100-80 quota lives in ``us-central1``.
DEFAULT_PRIMARY_ZONE = "us-central1-a"

#: Same-region fallbacks for a capacity miss. The GPUS_ALL_REGIONS quota is
#: regional so any zone in ``us-central1`` is in scope without a quota
#: re-request.
DEFAULT_FALLBACK_ZONES: tuple[str, ...] = ("us-central1-b", "us-central1-c")

#: DLVM image family verified working on 2026-06-08 ($1 credit-draw test
#: provisioned ``a2-ultragpu-1g`` Spot with this image and ran nvidia-smi).
DEFAULT_IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"

#: DLVM project for the image family above.
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"

#: Canonical public HTTPS clone URL. The repo is open; private branches
#: would extend the startup-script to push a deploy key.
DEFAULT_REPO_URL = "https://github.com/superkaiba/explore-persona-space.git"


def default_gcp_config() -> GcpConfig:
    """Build the production :class:`GcpConfig` from module defaults.

    Centralized so production callers (the selector / router) and tests
    that want the "real" config but with one override (e.g. the zone) can
    use the same source of truth. Tests that want a fully-controlled
    config construct :class:`GcpConfig` directly.
    """
    return GcpConfig(
        project=DEFAULT_PROJECT,
        gcloud_config=DEFAULT_GCLOUD_CONFIG,
        primary_zone=DEFAULT_PRIMARY_ZONE,
        fallback_zones=DEFAULT_FALLBACK_ZONES,
        image_family=DEFAULT_IMAGE_FAMILY,
        image_project=DEFAULT_IMAGE_PROJECT,
        repo_url=DEFAULT_REPO_URL,
    )


# ---------------------------------------------------------------------------
# Intent → machine-type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MachineSpec:
    """A GCE machine + accelerator pair selected by workload intent.

    Fields:

    * ``machine_type`` — gcloud machine-type id (e.g. ``a2-ultragpu-1g``).
    * ``gpu_count`` — number of GPUs the machine carries (1 / 4 / etc.).
    * ``gpu_kind`` — short kind tag for logging / failure-classification
      ("A100-80", "L4"). Not threaded into gcloud (the ``a2-ultragpu-*``
      family hardcodes the accelerator).
    """

    machine_type: str
    gpu_count: int
    gpu_kind: str


#: Workload intent → GCE machine-type map. Matches the plan's "gcp.py"
#: Approach paragraph: lora-7b → a2-ultragpu-1g, ft-7b → a2-ultragpu-4g,
#: eval → g2-standard-4. The ``lora`` alias inherits ``lora-7b`` (mirrors
#: the SLURM ``_DEFAULT_GPUS_FOR_INTENT`` aliasing). ``debug`` reuses the
#: L4 machine — the smallest GPU available is the right "debug pod"
#: analogue. ``inf-70b`` / ``ft-70b`` are NOT in this table (the GFS
#: credit pool is for the A100-80 / L4 line; 70B inference belongs on
#: RunPod's H200 in v1).
INTENT_TO_MACHINE: dict[str, MachineSpec] = {
    "lora-7b": MachineSpec(
        machine_type="a2-ultragpu-1g",
        gpu_count=1,
        gpu_kind="A100-80",
    ),
    "lora": MachineSpec(
        machine_type="a2-ultragpu-1g",
        gpu_count=1,
        gpu_kind="A100-80",
    ),
    "ft-7b": MachineSpec(
        machine_type="a2-ultragpu-4g",
        gpu_count=4,
        gpu_kind="A100-80",
    ),
    "eval": MachineSpec(
        machine_type="g2-standard-4",
        gpu_count=1,
        gpu_kind="L4",
    ),
    "debug": MachineSpec(
        machine_type="g2-standard-4",
        gpu_count=1,
        gpu_kind="L4",
    ),
}


def machine_for_intent(spec: RunSpec) -> MachineSpec:
    """Resolve ``spec.intent`` to a :class:`MachineSpec`.

    Fails LOUD on an unknown intent rather than silently picking a
    default — a typo should crash the launch, NOT spin up the wrong
    instance type and burn credit on it. Consistent with the SLURM
    backend's :func:`~slurm.stages_for_spec` / :func:`~slurm.time_budget_hours`
    fail-fast policy.
    """
    if spec.intent not in INTENT_TO_MACHINE:
        raise ValueError(
            f"no GCP machine-type for intent {spec.intent!r}. "
            f"Supported intents: {sorted(INTENT_TO_MACHINE)}. "
            "Add a MachineSpec row to backends/gcp.INTENT_TO_MACHINE "
            "or pick a different backend (RunPod covers H200 / 70B paths)."
        )
    return INTENT_TO_MACHINE[spec.intent]


# ---------------------------------------------------------------------------
# Provisioning model + attempt-id
# ---------------------------------------------------------------------------


#: GCE provisioning models accepted by ``--provisioning-model``.
ProvisioningModel = str  # "SPOT" | "STANDARD"

#: Default provisioning model: STANDARD (on-demand) for the acceptance run
#: per the plan ("on-demand for acceptance; steady-state Spot once
#: idempotency is proven"). Caller switches to "SPOT" via
#: ``spec.extra["provisioning_model"]`` once the idempotency proofs land.
DEFAULT_PROVISIONING_MODEL: ProvisioningModel = "STANDARD"


def resolve_provisioning_model(spec: RunSpec) -> ProvisioningModel:
    """Pick the provisioning model for ``spec`` (Spot vs on-demand).

    Reads ``spec.extra["provisioning_model"]`` if present and uppercases
    it; otherwise returns :data:`DEFAULT_PROVISIONING_MODEL`. Raises on
    an unrecognized value so a typo doesn't silently downgrade an
    on-demand workload to Spot (or vice versa).
    """
    raw = spec.extra.get("provisioning_model")
    if raw is None:
        return DEFAULT_PROVISIONING_MODEL
    val = str(raw).upper()
    if val not in {"SPOT", "STANDARD"}:
        raise ValueError(
            f"unknown provisioning_model={raw!r}; expected 'SPOT' or 'STANDARD' "
            "(case-insensitive). Set via RunSpec.extra['provisioning_model']."
        )
    return val


def attempt_id_for(spec: RunSpec) -> str:
    """Stable per-attempt namespace tag.

    Used as a sub-folder under HF data / model paths AND as a sentinel
    sub-directory on the VM scratch so a fresh idempotent re-run after
    Spot preemption never overwrites an earlier attempt's artifacts.
    Reads ``spec.extra["attempt_id"]`` if set (the router/orchestrator
    passes a deterministic per-attempt id so reconnect after orchestrator
    re-spawn picks up the same namespace); otherwise falls back to a
    timestamp-only tag (``att-YYYYMMDD-HHMMSS``).

    The tag is shell-safe (only ``[A-Za-z0-9_-]``); the renderer threads
    it verbatim into the startup-script + the HF-paths declaration.
    """
    raw = spec.extra.get("attempt_id")
    if raw:
        # Defense in depth: refuse a tag that would shell-inject. The
        # router should send a sanitized id; raise loud if not.
        tag = str(raw)
        if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", tag):
            raise ValueError(f"attempt_id must match [A-Za-z0-9_-.]+, got {tag!r}")
        return tag
    now = datetime.now(tz=UTC)
    return f"att-{now.strftime('%Y%m%d-%H%M%S')}"


# ---------------------------------------------------------------------------
# Naming + paths
# ---------------------------------------------------------------------------


def instance_name_for(issue: int) -> str:
    """Canonical GCE instance name for a `/issue` run.

    ``eps-issue-<N>`` matches the prefix the GCP stale-VM reaper greps
    for. Mirrors RunPod's ``pod-<N>`` shape (issue-keyed, one-instance-
    per-issue).
    """
    return f"eps-issue-{issue}"


def workload_dir_for(config: GcpConfig, issue: int) -> str:
    """Workload root on the VM: ``<vm_scratch_dir>/issue-<N>``.

    Mirrors the RunPod ``/workspace/<repo>`` convention so the workload
    sees the same in-VM layout regardless of backend. The sentinel +
    eval_results live under here.
    """
    return f"{config.vm_scratch_dir}/eps-issue-{issue}"


def sentinel_path_for(config: GcpConfig, issue: int, attempt_id: str) -> str:
    """Absolute path to the completion sentinel the workload writes.

    Folded under ``<workload>/eval_results/issue_<N>/<attempt>/`` so a
    re-run after Spot preemption (with a fresh ``attempt_id``) lands in
    a SEPARATE directory — prior attempts' sentinels (and their per-
    attempt outputs) are never overwritten.
    """
    root = workload_dir_for(config, issue)
    return f"{root}/eval_results/issue_{issue}/{attempt_id}/{SENTINEL_FILENAME}"


# ---------------------------------------------------------------------------
# Expected-artifact declaration (artifacts.py bridge)
# ---------------------------------------------------------------------------


def expected_artifacts_declaration(
    *,
    spec: RunSpec,
    config: GcpConfig,
    attempt_id: str,
    wandb_run_path: str | None = None,
    extra_hf_data_paths: Sequence[str] = (),
    extra_hf_model_paths: Sequence[str] = (),
    extra_git_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the :data:`EXPECTED_ARTIFACTS_HANDLE_KEY` payload for launch.

    The slice-2 verifier (``artifacts.confirm_artifacts_from_handle``)
    FAILs a missing declaration AND an all-SKIP one — the launch path
    MUST populate this so teardown is gated on real evidence the run
    actually produced its outputs. We derive the declaration here so
    every launch route (selector / router / direct ``GcpBackend.launch``)
    computes the same shape.

    Mandatory: the per-run completion ``sentinel_path`` (under
    :data:`SENTINEL_FILENAME`). The verifier treats a SKIPped sentinel as
    a FAIL (silent-loss hole closure).

    Default included paths (mirrors the Upload Policy table):

    * HF data repo ``issue<N>_<attempt>/raw_completions/`` — hydra-lane
      (``scripts/train.py``) launches only. A custom ``workload_cmd``
      launch declares NO default HF data path: the prefix above is a
      launch-time GUESS the workload never promised, and dispatch-script
      drivers use their own contract prefix (``issue<N>_<slug>/...``) —
      the guess produced a false-negative ``confirm_artifacts`` FAIL
      (exit 3, teardown skipped) on a perfectly-uploaded run (incident
      #601 follow-up r1, 2026-06-12). An undeclared ``hf_data_paths``
      SKIPs the hf_data check (SKIP is not FAIL); the completion
      sentinel + git paths keep gating teardown, and HF-data coverage on
      that lane comes from the agent-level upload-verifier (`/issue`
      Step 8). Callers that DO know the workload's real prefix declare
      it via ``extra_hf_data_paths``.
    * Git paths ``eval_results/issue_<N>/`` + ``figures/issue_<N>/`` —
      both committed by the workload + verified on the orchestrator side.

    The caller can add experiment-specific paths via ``extra_hf_data_paths``
    / ``extra_hf_model_paths`` / ``extra_git_paths`` (e.g. a sweep with a
    specific adapter subfolder).

    Returns a serialization-friendly ``dict`` (no tuples) so the launch
    path can drop it onto ``handle.extra`` and round-trip via
    :func:`artifacts.expected_artifacts_from_handle`.
    """
    issue = spec.issue
    if spec.workload_cmd:
        # Custom dispatch scripts own their HF prefix; declaring a guessed
        # one turns the mechanical gate into a false-negative teardown
        # block (#601 follow-up r1). Explicit knowledge rides
        # extra_hf_data_paths.
        base_hf_data: tuple[str, ...] = ()
    else:
        base_hf_data = (f"issue{issue}_{attempt_id}/raw_completions/",)
    base_git = (
        f"eval_results/issue_{issue}/",
        f"figures/issue_{issue}/",
    )
    return {
        "issue": int(issue),
        "hf_data_repo": config.hf_data_repo,
        "hf_model_repo": config.hf_model_repo,
        "hf_data_paths": list(base_hf_data) + list(extra_hf_data_paths),
        "hf_model_paths": list(extra_hf_model_paths),
        "wandb_run_path": wandb_run_path,
        "git_paths": list(base_git) + list(extra_git_paths),
        "sentinel_path": sentinel_path_for(config, issue, attempt_id),
    }


# ---------------------------------------------------------------------------
# Startup-script (mirrors bootstrap_pod.sh)
# ---------------------------------------------------------------------------


# The startup-script env keys the orchestrator MUST set (via gcloud
# --metadata) so the in-VM bootstrap can talk to HF / WandB / Anthropic.
# Mirrors ``SECRET_ENV_KEYS`` in slurm.py.
STARTUP_SECRET_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "WANDB_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

# Non-secret env keys passed through to the in-VM workload environment
# via the same instance-metadata mechanism. Mirrors
# ``slurm.PASSTHROUGH_ENV_KEYS``: these are the delete-after-eval
# adapter-persist targets ``trainer.py:_persist_adapter`` reads from
# ``os.environ`` ON THE VM (see ``.claude/rules/upload-policy.md``) —
# plain configuration, NOT secrets, so they live in a SEPARATE list to
# keep ``STARTUP_SECRET_ENV_KEYS`` semantically "secrets only". Without
# this passthrough, a value set on the dispatch process env (e.g. by
# ``scripts/router_acceptance.py --live``) never reaches the remote
# workload and the HF adapter upload silently no-ops.
STARTUP_PASSTHROUGH_ENV_KEYS: tuple[str, ...] = (
    "EPM_PERSIST_ADAPTER_HF_REPO",
    "EPM_PERSIST_ADAPTER_SUBFOLDER",
)

#: The subset of :data:`STARTUP_SECRET_ENV_KEYS` the GCE workload cannot
#: run without: ``train.py`` calls ``wandb.init`` (WANDB_API_KEY) and the
#: adapter-persist path pushes to HF Hub (HF_TOKEN). :func:`resolve_launch_secrets`
#: fails LOUD at launch time when either is unresolvable — silently
#: dropping them provisioned a doomed VM that burned the full boot +
#: uv-sync spend before crashing at ``wandb.init`` (live finding, issue
#: 535 GCP lane r7: the dispatch process had no dotenv loaded, so every
#: ``--metadata KEY=value`` pair was dropped and the workload saw empty
#: exports). The remaining keys (ANTHROPIC/OPENAI) are genuinely optional
#: for a training workload and keep the drop-when-absent contract.
REQUIRED_LAUNCH_SECRET_KEYS: tuple[str, ...] = ("HF_TOKEN", "WANDB_API_KEY")


def render_startup_script(
    *,
    spec: RunSpec,
    config: GcpConfig,
    attempt_id: str,
    repo_branch: str = "main",
    hydra_args: Sequence[str] | None = None,
) -> str:
    """Render the GCE startup-script the VM runs on boot.

    Pure function — no side effects. Tests can assert on the rendered
    text without spinning up a VM. The script:

    1. Sets strict mode + umask.
    2. Reads secrets from the VM metadata (set via gcloud
       ``--metadata KEY=value``) and exports them.
    3. Clones / pulls the repo into ``<vm_scratch_dir>/eps-issue-<N>``
       at the requested branch (defaults to ``main``).
    4. Installs ``uv`` if missing, runs ``uv sync --frozen``.
    5. Redirects ``HF_HOME`` to a fast local SSD path so model downloads
       cache for the run (the boot disk is pd-ssd).
    6. Writes a per-attempt scratch dir and runs the workload (currently
       ``scripts/train.py`` with the spec's Hydra args).
    7. On clean exit writes the completion sentinel under the per-attempt
       eval_results directory (the artifact verifier reads this).
    8. On any failure exits non-zero so the VM enters TERMINATED status
       (the orchestrator's ``poll`` reads this as ``dead``).

    The script intentionally does NOT write artifacts off-VM itself — the
    workload's existing HF/WandB upload paths run during the run as the
    authoritative artifact route. The sentinel is a small completion
    proof, not a primary artifact.

    ``hydra_args`` defaults to ``spec.hydra_args`` (so the caller can
    override for a custom dispatch); ``repo_branch`` defaults to ``main``.

    When ``spec.workload_cmd`` is set (#588) the workload block runs
    that command verbatim instead of ``scripts/train.py``; all other
    lifecycle machinery (secrets fetch, in-VM preflight, ``eps/phase``
    guest attributes, EXIT trap, completion sentinel) is identical, and
    the hydra branch is byte-for-byte the pre-#588 render (pinned by the
    snapshot test).

    Workload-cmd blocking contract (#601): the completion sentinel is
    only valid once the workload is actually finished, so the script
    assumes the command BLOCKS. A command that self-daemonizes
    (``setsid``-forks the real driver — the standard
    ``launch_issue_<N>.sh`` pattern) returns immediately; it MUST write
    the detached process's pid to a fresh file under
    ``/workspace/logs/*.pid``, and the rendered script polls any such
    pid file written after the workload started until the process exits
    BEFORE writing the sentinel + publishing ``_eps_phase done``.
    Without the wait, ``backend_poll.py`` reports terminal-success
    minutes into a multi-hour run (incident #601 follow-up r1,
    eps-issue-601, 2026-06-12). Blocking commands write no fresh pid
    file, so the wait is a no-op on that path.
    """
    args = tuple(hydra_args if hydra_args is not None else spec.hydra_args)
    if spec.workload_cmd and args:
        # Reachable via the ``hydra_args`` parameter override on a
        # workload_cmd spec — RunSpec.__post_init__ only guards the
        # spec's own fields.
        raise ValueError("render_startup_script: workload_cmd and hydra_args both set")
    if not spec.workload_cmd and not args:
        raise ValueError(
            "render_startup_script: neither workload_cmd nor hydra_args set — refusing "
            "to render a bare 'scripts/train.py' launch (incident #571: it crashes at "
            "startup and the EXIT trap powers the VM off)."
        )
    workload_root = workload_dir_for(config, spec.issue)
    sentinel_abs = sentinel_path_for(config, spec.issue, attempt_id)
    sentinel_dir = sentinel_abs.rsplit("/", 1)[0]

    # Build the secret-fetch stanza. Each KEY is pulled from
    # ``/computeMetadata/v1/instance/attributes/<KEY>``. The
    # ``Metadata-Flavor: Google`` header is the GCE-required guard; the
    # curl path 404s cleanly when a key was not set (so an absent
    # secret produces an empty export, not a hard crash — the in-VM
    # workload's own preflight surfaces the missing token loudly).
    # The non-secret STARTUP_PASSTHROUGH_ENV_KEYS (adapter-persist
    # targets) ride the same fetch stanza — metadata is the one
    # env-delivery surface the VM has.
    secrets_fetch_lines: list[str] = []
    for key in STARTUP_SECRET_ENV_KEYS + STARTUP_PASSTHROUGH_ENV_KEYS:
        secrets_fetch_lines.append(
            f'{key}=$(curl -fsS -H "Metadata-Flavor: Google" '
            f'"http://metadata.google.internal/computeMetadata/v1/'
            f'instance/attributes/{key}" 2>/dev/null || true); export {key}'
        )

    # Hydra args, shell-quoted. Empty tuple → empty string.
    hydra_str = " ".join(shlex.quote(a) for a in args)

    # Workload block (#588): a custom workload_cmd is embedded VERBATIM —
    # it IS a complete shell command line (shlex-quoting would collapse
    # it to a single token). Trusted input by design (same trust level as
    # the plan's Reproducibility Card launch command; it runs as root on
    # the VM). The RunSpec.__post_init__ single-line check keeps the
    # rendered script structure intact. The hydra branch is the
    # byte-identical pre-#588 lines, gated only by ``if spec.workload_cmd``.
    if spec.workload_cmd:
        workload_block = [
            "# === WandB project default (#601 follow-up r1) ===",
            "# HF-Trainer workloads that never set WANDB_PROJECT land in WandB's",
            "# global default project 'huggingface', violating the Upload Policy",
            "# (training metrics → project=<experiment_name>). Default to the",
            "# per-issue project; :- fills only unset/empty, so an inline",
            "# WANDB_PROJECT=... prefix on the workload command — or the workload",
            "# setting its own project internally — still wins.",
            f'export WANDB_PROJECT="${{WANDB_PROJECT:-issue{spec.issue}}}"',
            "# === Run the workload (custom workload_cmd) ===",
            "# A non-zero exit propagates (set -e) → the EXIT trap publishes",
            "# phase=failed + powers off → poll reads dead.",
            "_eps_phase workload",
            "touch /tmp/eps-workload-start",
            spec.workload_cmd,
            "# === Wait for detached workloads (self-daemonizing drivers) ===",
            "# A workload_cmd that setsid-forks the real driver returns",
            "# immediately; declaring done here would publish the completion",
            "# sentinel mid-run (incident #601 follow-up r1: the poll read",
            "# terminal-success at T+4min of a ~2h run). Contract: a detached",
            "# workload writes its pid to a fresh file under",
            "# /workspace/logs/*.pid (the launch_issue_<N>.sh convention).",
            "# Only pid files NEWER than the workload start are waited on, so",
            "# stale files from prior attempts are skipped; blocking workloads",
            "# write no fresh pid file → the loop is a no-op. Bounded by the",
            "# instance's --max-run-duration (termination action DELETE).",
            "# kill -0 sits in condition contexts, so set -e never fires here.",
            "for pf in $(find /workspace/logs -maxdepth 1 -name '*.pid'"
            " -newer /tmp/eps-workload-start 2>/dev/null || true); do",
            '  wpid=$(cat "$pf" 2>/dev/null) || continue',
            '  echo "[startup-script] waiting on detached workload pid=$wpid ($pf)"',
            '  while kill -0 "$wpid" 2>/dev/null; do sleep 30; done',
            '  echo "[startup-script] detached workload pid=$wpid exited"',
            "done",
        ]
    else:
        workload_block = [
            "# === Run the workload (Hydra args = the spec's hydra_args) ===",
            "# A non-zero exit propagates (set -e) → the EXIT trap publishes",
            "# phase=failed + powers off → poll reads dead.",
            "_eps_phase workload",
            f"uv run python scripts/train.py {hydra_str}".rstrip(),
        ]

    parts = [
        "#!/bin/bash",
        "set -euo pipefail",
        "umask 077",
        # Publish the workload phase to the GCE guest attribute
        # ``eps/phase`` — the ONLY poll-readable surface the VM has
        # while staying RUNNING (the success path keeps the VM alive so
        # the sentinel can be scp'd, so instance status alone cannot
        # signal completion; issue 535 r9 spun the poll for the full 4 h
        # timeout on a 9-min success). Best-effort (`|| true`): a probe
        # hiccup must never kill the workload.
        '_eps_phase() { curl -fsS -X PUT -H "Metadata-Flavor: Google"'
        ' --data "$1" "http://metadata.google.internal/computeMetadata/v1/'
        'instance/guest-attributes/eps/phase" >/dev/null 2>&1 || true; }',
        # A failed startup script does NOT stop the VM — GCE just logs
        # "Script failed with error" and leaves the instance RUNNING,
        # billing the GPU with no workload (live finding, issue 535 GCP
        # lane r7: the VM idled ~85 min after a workload crash because
        # the monitoring session had died). The EXIT trap bounds that:
        # any non-zero exit publishes phase=failed (so the poll
        # classifies dead even before the instance state flips) and
        # powers the VM off; disk preserved for debugging; the harness
        # teardown deletes it. The rc==0 guard keeps the success path
        # ALIVE — the artifact verifier scp-pulls the completion
        # sentinel off the VM after a clean run.
        'trap \'rc=$?; if [ "$rc" -ne 0 ]; then'
        ' echo "[startup-script] FAILED rc=$rc — powering off to bound billing";'
        " _eps_phase failed;"
        " shutdown -h now; fi' EXIT",
        "_eps_phase startup",
        # GCE's metadata script runner executes as root WITHOUT $HOME set;
        # under `set -u` the first $HOME reference (uv PATH export) kills
        # the script (live finding, issue 535 GCP lane: `line 32: HOME:
        # unbound variable` → workload never started, GPU idle).
        'export HOME="${HOME:-/root}"',
        "",
        f"# === GCE startup-script (eps-issue-{spec.issue}) ===",
        f"export EPS_ISSUE={spec.issue}",
        f"export EPS_ATTEMPT_ID={shlex.quote(attempt_id)}",
        f"export WORKLOAD_ROOT={shlex.quote(workload_root)}",
        f"export EPS_SENTINEL_PATH={shlex.quote(sentinel_abs)}",
        "",
        "# === Secrets from instance metadata ===",
        *secrets_fetch_lines,
        "",
        # In-VM preflight (defense in depth — launch() already fails loud
        # via resolve_launch_secrets): an empty required secret kills the
        # script HERE, ~seconds after boot, instead of after the full
        # repo-clone + uv-sync spend at the workload's first credentialed
        # call. The non-zero exit fires the EXIT trap above → power off.
        "# === In-VM preflight: required workload secrets ===",
        *[
            f'[ -n "${{{key}:-}}" ] || {{ echo "[FAIL] {key} missing from instance metadata"; '
            "exit 78; }"
            for key in REQUIRED_LAUNCH_SECRET_KEYS
        ],
        "",
        "# === Repo clone / pull (idempotent) ===",
        'mkdir -p "$WORKLOAD_ROOT"',
        'if [ ! -d "$WORKLOAD_ROOT/.git" ]; then',
        f"  git clone --depth 1 --branch {shlex.quote(repo_branch)} "
        f'{shlex.quote(config.repo_url)} "$WORKLOAD_ROOT"',
        "else",
        f'  git -C "$WORKLOAD_ROOT" fetch --depth 1 origin {shlex.quote(repo_branch)}',
        f'  git -C "$WORKLOAD_ROOT" checkout {shlex.quote(repo_branch)}',
        f'  git -C "$WORKLOAD_ROOT" reset --hard origin/{shlex.quote(repo_branch)}',
        "fi",
        "",
        "# === Install uv if missing + sync env ===",
        "if ! command -v uv >/dev/null 2>&1; then",
        "  curl -LsSf https://astral.sh/uv/install.sh | sh",
        '  export PATH="$HOME/.local/bin:$PATH"',
        "fi",
        'cd "$WORKLOAD_ROOT"',
        # Pin the interpreter: the DLVM's system python is 3.10 (below
        # requires-python >=3.11), so an unpinned `uv sync` fetches the
        # NEWEST allowed CPython — 3.14 as of Jun 2026 — and torch 2.8.0
        # ships no cp314 wheel (live finding, issue 535 GCP lane r5:
        # 'no source distribution or wheel for the current platform').
        # 3.11 matches the RunPod image python + ruff's py311 target.
        "uv sync --frozen --python 3.11",
        "",
        "# === HF cache + sentinel dir ===",
        'export HF_HOME="$WORKLOAD_ROOT/.cache/huggingface"',
        'mkdir -p "$HF_HOME"',
        f"mkdir -p {shlex.quote(sentinel_dir)}",
        "",
        *workload_block,
        "",
        "# === Completion sentinel (workload exited cleanly) ===",
        "# The artifact verifier reads this back via list_repo_files / scp.",
        "# Phase=done + issue=<N> is the schema artifacts.py validates.",
        "cat > " + shlex.quote(sentinel_abs) + " <<EOF\n"
        '{"phase":"done","issue":'
        + str(spec.issue)
        + ',"attempt_id":'
        + json.dumps(attempt_id)
        + "}\nEOF",
        # Publish done LAST — the poll treats it as terminal-success and
        # the harness immediately proceeds to fetch_results (scp of the
        # sentinel written above), so the sentinel must already exist.
        "_eps_phase done",
        "",
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# gcloud argv renderers
# ---------------------------------------------------------------------------


def _base_gcloud_argv(config: GcpConfig, *cmd: str) -> list[str]:
    """Prepend the ``--configuration`` + ``--project`` flags to a gcloud call.

    Threaded per-command (NOT via env var) so the backend is independent
    of the ambient ``CLOUDSDK_ACTIVE_CONFIG_NAME`` shared with my-goat.
    """
    return [
        "gcloud",
        *cmd,
        f"--configuration={config.gcloud_config}",
        f"--project={config.project}",
    ]


def render_create_argv(
    *,
    spec: RunSpec,
    config: GcpConfig,
    attempt_id: str,
    zone: str | None = None,
    startup_script: str,
    secret_files: Mapping[str, str] | None = None,
) -> list[str]:
    """Build the ``gcloud compute instances create`` argv.

    Pure function — golden-tested without touching the network.

    Mirrors the verified-working recipe in
    ``~/my-goat/reference/gcp-compute-execution-2026-06.md`` (the 2026-06-08
    $1 credit-draw test). Hard requirements baked in:

    * ``--configuration`` + ``--project`` — every call threads these
      explicitly so the backend ignores ambient config.
    * ``--machine-type`` from the intent map.
    * ``--provisioning-model`` from the spec (Spot vs on-demand).
    * ``--instance-termination-action=DELETE`` — the leak guard. Whether
      Spot preempts or the ``--max-run-duration`` fence trips, the VM
      auto-deletes; combined with the GCP stale-VM reaper this caps
      credit leakage at the audit window.
    * ``--maintenance-policy=TERMINATE`` — GPUs cannot live-migrate, so
      the maintenance policy MUST be terminate (gcloud rejects MIGRATE
      on accelerator VMs anyway; explicit is clearer than implicit).
    * ``--max-run-duration`` — generous so it can't interrupt an upload
      (default 24h per config).
    * ``--image-family`` / ``--image-project`` — the DLVM image with
      pre-installed CUDA/driver.
    * ``--boot-disk-size`` / ``--boot-disk-type`` — 300 GB pd-ssd default.
    * ``--scopes=cloud-platform`` — broad VM-scope so the in-VM workload
      can push to GCS / WandB / HF without per-API token wrangling.
    * ``--metadata-from-file startup-script=<path>,KEY=<path>`` — the
      startup-script bootstraps the workload; SECRET keys are delivered
      from caller-owned 0600 tempfiles (``secret_files``) so token
      values never appear on the gcloud argv / process list (round-2
      Codex Major, task #535). The resulting instance metadata is
      identical to the old per-secret ``--metadata KEY=value`` shape —
      the in-VM fetch stanza reads the same ``attributes/<KEY>`` paths.

    ``zone`` defaults to ``config.primary_zone``; the caller passes a
    fallback zone explicitly on a capacity retry.

    ``secret_files`` maps each resolvable :data:`STARTUP_SECRET_ENV_KEYS`
    key to the tempfile holding its value; :meth:`GcpBackend.launch` owns
    that tempfile lifecycle (0600 create before render, unlink in a
    ``finally``). A secret that resolves to a value WITHOUT a threaded
    file path raises ``ValueError`` — silently dropping it would
    provision a doomed VM (the issue-535 r7 class), and inlining it
    would put the token back on the argv.

    The argv is returned as a list (not a string) so the caller can pass
    it straight to ``subprocess.run`` without shell parsing — defense
    against shell injection through the startup-script body.
    """
    machine = machine_for_intent(spec)
    provisioning = resolve_provisioning_model(spec)
    max_run = spec.extra.get("max_run_duration") or config.default_max_run_duration
    boot_disk_gb = int(spec.extra.get("boot_disk_gb") or config.default_boot_disk_gb)
    boot_disk_type = spec.extra.get("boot_disk_type") or config.default_boot_disk_type
    target_zone = zone or config.primary_zone
    name = instance_name_for(spec.issue)

    argv = _base_gcloud_argv(config, "compute", "instances", "create", name)
    argv += [
        f"--zone={target_zone}",
        f"--machine-type={machine.machine_type}",
        f"--provisioning-model={provisioning}",
        "--instance-termination-action=DELETE",
        "--maintenance-policy=TERMINATE",
        f"--max-run-duration={max_run}",
        f"--image-family={config.image_family}",
        f"--image-project={config.image_project}",
        f"--boot-disk-size={boot_disk_gb}GB",
        f"--boot-disk-type={boot_disk_type}",
        "--scopes=cloud-platform",
        "--labels=" + _format_labels(spec, attempt_id),
        "--format=json",
    ]

    # Metadata: startup-script body + the keys the script will fetch
    # back out of metadata. Each key arrives via os.environ so the
    # caller's environment dictates which values are forwarded. An absent
    # env var is dropped (matches render_secrets_env in slurm.py). The
    # non-secret STARTUP_PASSTHROUGH_ENV_KEYS (adapter-persist targets)
    # use the same ``spec.extra["secret_<KEY>"]``-then-env lookup so a
    # caller can thread either class per-launch.
    # enable-guest-attributes lets the in-VM startup script publish its
    # workload phase to a poll-readable surface (guest attribute
    # ``eps/phase``) — without it a SUCCESSFUL workload is undetectable
    # (the VM deliberately stays RUNNING so the sentinel can be scp'd,
    # and the coarse describe-based poll reads "running" until the hard
    # timeout; live finding, issue 535 GCP lane r9: 20-step smoke
    # finished in ~9 min, poll spun for the full 4 h timeout, teardown
    # destroyed the lane evidence).
    metadata_pairs = [
        f"eps-issue={spec.issue}",
        f"eps-attempt-id={attempt_id}",
        "enable-guest-attributes=TRUE",
    ]
    # SECRETS never ride the inline ``--metadata`` flag: every inline
    # pair is argv-visible (process list, shell trace, captured harness
    # logs) for the lifetime of the gcloud call (round-2 Codex Major,
    # task #535). Secret keys are delivered via ``--metadata-from-file``
    # from caller-owned 0600 tempfiles instead — the resulting INSTANCE
    # METADATA is identical, so the in-VM fetch stanza
    # (render_startup_script) is unchanged. Residual security boundary:
    # custom instance metadata remains readable to any principal with
    # ``compute.instances.get`` on the project — acceptable for the
    # dedicated single-user project (eps-persona-gpu-jun2026); the full
    # Secret Manager pull-from-VM migration is tracked as concern
    # ``gcp-secrets-secret-manager-migration`` on task #535.
    secret_file_pairs: list[str] = []
    for key in STARTUP_SECRET_ENV_KEYS:
        val = spec.extra.get(f"secret_{key}") or _envget(key)
        if val is None or val == "":
            continue
        path = (secret_files or {}).get(key)
        if not path:
            raise ValueError(
                f"render_create_argv: secret {key} resolved to a value but no "
                "--metadata-from-file tempfile was threaded via secret_files. "
                "Refusing to place a token on the gcloud argv; launch() owns "
                "the 0600 tempfile lifecycle."
            )
        secret_file_pairs.append(f"{key}={path}")
    for key in STARTUP_PASSTHROUGH_ENV_KEYS:
        val = spec.extra.get(f"secret_{key}") or _envget(key)
        if val is None or val == "":
            continue
        # Non-secret passthrough config; inline metadata is fine here.
        metadata_pairs.append(f"{key}={val}")
    # gcloud splits ``--metadata`` on commas, so a forwarded value
    # containing a comma would silently truncate every later pair. Keep
    # the plain comma-join for the common comma-free case (argv stays
    # byte-stable), and switch to gcloud's alternate-delimiter syntax
    # (``--metadata=^<delim>^k1=v1<delim>k2=v2`` — see ``gcloud topic
    # escaping``) whenever any pair carries a comma.
    if any("," in pair for pair in metadata_pairs):
        delim = next(
            (d for d in (":", "|", "#", "~") if not any(d in pair for pair in metadata_pairs)),
            None,
        )
        if delim is None:
            keys = [pair.split("=", 1)[0] for pair in metadata_pairs]
            raise ValueError(
                "render_create_argv: no safe --metadata delimiter — every candidate "
                f"appears in some pair value; keys={keys}"
            )
        argv.append(f"--metadata=^{delim}^" + delim.join(metadata_pairs))
    else:
        argv.append("--metadata=" + ",".join(metadata_pairs))
    # Startup-script via --metadata-from-file is the right shape (avoid
    # the 256KB metadata-line cap when the body grows). The caller writes
    # the script to a tempfile; the renderer asserts the contract via
    # spec.extra["startup_script_path"] OR an inline body. We choose the
    # tempfile path here so secrets-bearing scripts never leak through
    # the gcloud argv stdout/stderr.
    #
    # ONE combined --metadata-from-file flag carries the startup-script
    # AND the secret keys: gcloud dict-type flags don't merge when
    # repeated (a second occurrence replaces the first), so splitting
    # them would silently drop whichever flag came first. mkstemp paths
    # carry no commas, so the plain comma-join is safe here.
    sentinel = spec.extra.get("startup_script_path")
    if sentinel:
        from_file_pairs = [*secret_file_pairs, f"startup-script={sentinel}"]
        argv.append("--metadata-from-file=" + ",".join(from_file_pairs))
    else:
        if secret_file_pairs:
            argv.append("--metadata-from-file=" + ",".join(secret_file_pairs))
        # Inline body (golden tests + small startup scripts). The wrapper
        # caller is responsible for cap-checking. Inlined verbatim into
        # the metadata pairs constructed above is the right form; this
        # branch keeps the renderer self-contained when no tempfile path
        # is threaded through spec.extra.
        # gcloud's --metadata accepts startup-script= as a value; chain it
        # in a separate flag so it lands as a discrete metadata key.
        argv.append(f"--metadata=startup-script={startup_script}")
    return argv


def render_list_argv(*, config: GcpConfig, name_filter: str | None = None) -> list[str]:
    """Build a ``gcloud compute instances list`` argv with JSON output.

    Used by :func:`reconnect_or_none` + :func:`audit_stale_gcp_vms`.
    Filter syntax: gcloud accepts ``name=<exact>`` for an exact match
    and ``name~^prefix`` for a regex prefix; we pick exact for the
    reconnect path (one instance per issue) and the prefix form for
    the audit path.
    """
    argv = _base_gcloud_argv(config, "compute", "instances", "list", "--format=json")
    if name_filter:
        argv.append(f"--filter={name_filter}")
    return argv


def render_describe_argv(*, config: GcpConfig, name: str, zone: str) -> list[str]:
    """Build a ``gcloud compute instances describe`` argv (JSON)."""
    argv = _base_gcloud_argv(config, "compute", "instances", "describe", name)
    argv += [f"--zone={zone}", "--format=json"]
    return argv


def region_for_zone(zone: str) -> str:
    """``us-central1-a`` → ``us-central1`` (GCE zones are ``<region>-<suffix>``)."""
    return zone.rsplit("-", 1)[0]


def render_region_describe_argv(*, config: GcpConfig, region: str) -> list[str]:
    """Build the ``gcloud compute regions describe`` argv for the quota probe (JSON).

    The probe shape was verified live on issue 608 (2026-06-12): the
    response's ``quotas[]`` rows carry ``metric`` / ``usage`` / ``limit``
    for the regional accelerator quotas (e.g. ``NVIDIA_A100_80GB_GPUS``).
    """
    argv = _base_gcloud_argv(config, "compute", "regions", "describe", region)
    argv.append("--format=json")
    return argv


def render_guest_attributes_argv(*, config: GcpConfig, name: str, zone: str) -> list[str]:
    """Build a ``gcloud compute instances get-guest-attributes`` argv.

    Queries the ``eps/phase`` guest attribute the startup script
    publishes (``_eps_phase``) — the poll-readable workload-phase
    surface a RUNNING VM exposes (issue 535 r9: without it a successful
    workload is undetectable and the poll spins to the hard timeout).
    ``--query-path`` scopes the read to our namespace; gcloud exits
    non-zero when the attribute was never written (a VM still booting),
    which the poll treats as phase-unknown, NOT an error.
    """
    argv = _base_gcloud_argv(config, "compute", "instances", "get-guest-attributes", name)
    argv += [f"--zone={zone}", "--query-path=eps/phase", "--format=json"]
    return argv


def render_delete_argv(*, config: GcpConfig, name: str, zone: str) -> list[str]:
    """Build a ``gcloud compute instances delete`` argv (``--quiet`` for non-interactive)."""
    argv = _base_gcloud_argv(config, "compute", "instances", "delete", name)
    argv += [f"--zone={zone}", "--quiet"]
    return argv


def _format_labels(spec: RunSpec, attempt_id: str) -> str:
    """Build the ``--labels=`` value for create/list filtering.

    GCP label keys must be lowercase, may contain ``[a-z0-9_-]``, and
    have a 63-char cap. We emit a small fixed set — the prefix ``eps-``
    is the audit key. The ``attempt_id`` label normalizes underscores
    + hyphens (no caps allowed); we lowercase + replace anything else
    with a hyphen so the GCP API accepts the value.
    """
    sanitized_attempt = re.sub(r"[^a-z0-9_-]", "-", attempt_id.lower())[:63]
    return ",".join(
        [
            "managed-by=eps",
            f"eps-issue={spec.issue}",
            f"eps-attempt={sanitized_attempt}",
            f"eps-intent={spec.intent}",
        ]
    )


def _envget(key: str) -> str | None:
    """Read an env var without crashing when ``os`` is monkey-patched."""

    return os.environ.get(key)


def _default_src_root_for_fetch() -> Path:
    """Locate the repo root for ``fetch_results`` scp landings.

    Walks up from this module until a directory with ``pyproject.toml``
    is found (the same convention the SLURM backend's ``_default_src_root``
    uses). Used as the destination root for the best-effort
    ``eval_results/`` + ``figures/`` scp pulls so the pulled tree lands
    at the canonical project-relative paths.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


class GcpBackendError(RuntimeError):
    """Base class for typed GCP backend errors."""


class GcpProbeError(BackendProbeError):
    """The GCP state probe FAILED — instance state is UNKNOWN.

    Raised by :func:`reconnect_or_none` when ``gcloud compute instances
    list`` exits non-zero or returns unparseable JSON. "Couldn't ask"
    must never read as "no live instance" (the SLURM round-6 B1
    contract, mirrored here): pre-fix, an expired-auth ``list`` was
    swallowed as "assuming no live instance" and the router proceeded
    toward a blind ``create`` on the CREDIT-SPENDING lane (live GCP
    lane attempt 1, issue 535). The router's reconnect seams catch
    :class:`~explore_persona_space.backends.base.BackendProbeError`
    typed-ly: explicit lane → refuse-to-submit-blind terminal; auto
    escalation → no-compute terminal (fail-closed, no spend on unknown
    state).
    """


class GcpProvisioningError(GcpBackendError):
    """The VM never came up (capacity / quota / SSH / image fetch).

    The router's fallback logic catches THIS type and proceeds to the
    next tier (per the plan: "PROVISION/capacity/SSH/quota failure →
    next tier"). The orchestrator never auto-fallbacks on a different
    error class — a workload bug should surface, not silently re-run.
    """

    def __init__(self, reason: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


class GcpWorkloadError(GcpBackendError):
    """The workload itself failed AFTER the VM was up.

    Distinct from provisioning failure — the router MUST NOT auto-
    fallback on this (a deterministic workload bug would just re-crash
    on the next tier). Per plan: "WORKLOAD failure observed AFTER
    ``[phase=...]`` training has started → surface (post ``epm:failure``,
    ``status:blocked``), NO auto-fallback (it would re-crash)."
    """

    def __init__(self, reason: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


class GcpLaunchSecretsMissing(GcpBackendError):
    """Required workload secrets are unresolvable at launch time.

    Raised by :func:`resolve_launch_secrets` BEFORE any ``gcloud
    instances create`` — a VM provisioned without
    :data:`REQUIRED_LAUNCH_SECRET_KEYS` always burns the full boot +
    uv-sync spend and then crashes inside the workload (issue 535 GCP
    lane r7: ``wandb.errors.UsageError: No API key configured`` after
    ~3 min of A100 time, VM left RUNNING idle). This is a CONFIG error,
    not capacity: the router must surface it, never fall back to
    another tier (the same empty env would doom every backend).
    """


def resolve_launch_secrets(spec: RunSpec, env: Mapping[str, str] | None = None) -> None:
    """Resolve workload secrets into ``spec.extra["secret_<KEY>"]``, failing loud on gaps.

    Mirrors ``slurm.render_secrets_env``: secrets live in the repo
    ``.env`` (dotenv), NOT the ambient shell, so a bare ``os.environ``
    read from a clean dispatch process silently forwards NOTHING. When
    ``env`` is None, loads the project dotenv first
    (``resolve_dotenv_path`` walks to the main git worktree, so a linked
    worktree without its own ``.env`` still resolves; ``override=False``
    keeps already-exported vars authoritative) and snapshots
    ``os.environ``. Every resolved value is threaded through the
    existing ``spec.extra["secret_<KEY>"]`` contract that
    :func:`render_create_argv` already prefers over its env fallback, so
    the metadata the VM fetches is exactly what this function resolved.

    Raises :class:`GcpLaunchSecretsMissing` naming every
    :data:`REQUIRED_LAUNCH_SECRET_KEYS` member that is still absent or
    empty. Optional keys (ANTHROPIC/OPENAI + the adapter-persist
    passthroughs) keep the drop-when-absent contract.
    """
    if env is None:
        from explore_persona_space.orchestrate.env import load_dotenv as _load_dotenv

        _load_dotenv()
        env = os.environ
    missing: list[str] = []
    for key in STARTUP_SECRET_ENV_KEYS + STARTUP_PASSTHROUGH_ENV_KEYS:
        val = spec.extra.get(f"secret_{key}") or env.get(key)
        if val:
            spec.extra[f"secret_{key}"] = val
        elif key in REQUIRED_LAUNCH_SECRET_KEYS:
            missing.append(key)
    if missing:
        raise GcpLaunchSecretsMissing(
            "required workload secret(s) unresolvable at launch: "
            + ", ".join(missing)
            + " — not in spec.extra['secret_<KEY>'], the process env, or the project .env. "
            "A VM provisioned without them boots, burns the uv-sync spend, and crashes "
            "inside the workload (issue 535 GCP lane r7). Load the repo .env (or export "
            "the keys) before dispatching."
        )


# Substrings in gcloud stderr that indicate a provisioning failure
# (capacity / quota / image fetch). The classifier matches case-insensitively;
# anything not on this list bubbles up as a generic GcpBackendError so the
# router knows NOT to fall back blindly.
_PROVISIONING_STDERR_PATTERNS: tuple[str, ...] = (
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "QUOTA_EXCEEDED",
    "QUOTA EXCEEDED",
    # gcloud's regional accelerator-quota error is PROSE, not the API enum:
    # ``Quota 'NVIDIA_A100_80GB_GPUS' exceeded.  Limit: 8.0 in region
    # us-central1.`` — the metric name sits between "Quota" and "exceeded"
    # so neither QUOTA_EXCEEDED form above matches it. Four such creates on
    # issue 608 were classified "no known provisioning pattern" (2026-06-12).
    "Quota '",
    "RESOURCE_EXHAUSTED",
    "INSUFFICIENT_RESOURCES",
    # gcloud sometimes surfaces capacity as "does not have enough resources"
    "does not have enough resources",
    # Authentication / config errors should also surface as provisioning
    # failures so the router can fall back rather than wedge.
    "PERMISSION_DENIED",
    "permission denied",
    "Invalid value for field",
)


def classify_create_failure(*, returncode: int, stderr: str) -> GcpProvisioningError:
    """Map a non-zero ``gcloud compute instances create`` exit to a typed error.

    Inspects ``stderr`` for the known capacity / quota / auth substrings
    and packages them into :class:`GcpProvisioningError`. The caller
    (``GcpBackend.launch``) catches this and either retries on the next
    fallback zone (capacity) OR raises out so the router falls back to
    RunPod / blocks.
    """
    matched = next(
        (p for p in _PROVISIONING_STDERR_PATTERNS if p.lower() in stderr.lower()),
        None,
    )
    reason = (
        f"gcloud create returned {returncode}; matched provisioning pattern {matched!r}"
        if matched
        else f"gcloud create returned {returncode}; no known provisioning pattern (stderr below)"
    )
    return GcpProvisioningError(
        reason,
        evidence={
            "returncode": returncode,
            "stderr_tail": stderr[-2000:],
            "matched_pattern": matched,
        },
    )


# ---------------------------------------------------------------------------
# Runner injection seam (test plumbing)
# ---------------------------------------------------------------------------


@dataclass
class GcloudRunResult:
    """Captured ``gcloud`` exit status + stdout + stderr.

    The injectable :func:`GcpBackend` runner returns one of these so
    tests can fabricate any combination of (returncode, stdout, stderr)
    without spawning a subprocess.
    """

    returncode: int
    stdout: str
    stderr: str


GcloudRunner = Callable[[Sequence[str]], GcloudRunResult]


def default_gcloud_runner(argv: Sequence[str], *, timeout: int = 300) -> GcloudRunResult:
    """Default runner: shell out to ``gcloud`` via :mod:`subprocess`.

    Raises NOTHING on non-zero — the caller inspects ``returncode``.
    Timeouts propagate as :class:`subprocess.TimeoutExpired` (the
    backend treats them as provisioning failures via the catch in
    :meth:`GcpBackend.launch`).
    """
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return GcloudRunResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


# ---------------------------------------------------------------------------
# Reconnect (idempotent existing-instance lookup)
# ---------------------------------------------------------------------------


def reconnect_or_none(
    *,
    spec: RunSpec,
    config: GcpConfig,
    runner: GcloudRunner,
) -> RunHandle | None:
    """Return a handle for an existing live ``eps-issue-<N>`` instance, or None.

    Idempotency hinge: before any ``instances create`` call, this looks
    up the canonical instance name via ``gcloud compute instances list
    --filter='name=eps-issue-<N>'``. A live instance (status RUNNING,
    PROVISIONING, STAGING, STOPPING) returns a handle. A TERMINATED
    instance is treated as "not live" (the backend will create a fresh
    one); no instance returns None.

    Matches the "Idempotent: a per-run attempt-id is the sole write
    namespace; route() reconnects to an existing eps-issue-<N> GCE
    instance before re-provisioning" success criterion. The
    fresh-attempt-id namespace covers the artifact-overwrite concern
    even when a reconnect catches a still-running instance.

    Raises :class:`GcpProbeError` when the probe ITSELF fails (gcloud
    rc != 0 — expired auth, transport — or unparseable JSON from an
    rc=0 call): instance state is UNKNOWN, and "couldn't ask" must
    never read as "no live instance" on the credit-spending lane
    (round-6 B1 mirrored from SLURM; the pre-fix warn-and-None here let
    an expired-auth list fall through toward a blind create — live GCP
    attempt 1, issue 535). The router's reconnect seams handle
    ``BackendProbeError`` typed-ly on every lane.
    """
    name = instance_name_for(spec.issue)
    argv = render_list_argv(config=config, name_filter=f"name={name}")
    result = runner(argv)
    if result.returncode != 0:
        raise GcpProbeError(
            f"GCP reconnect probe failed for {name}: gcloud list rc={result.returncode} "
            f"stderr={result.stderr[:500]!r} — instance state UNKNOWN, refusing to "
            "assume no live instance"
        )
    try:
        instances = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError as exc:
        raise GcpProbeError(
            f"GCP reconnect probe returned unparseable JSON for {name}: {exc} — "
            "instance state UNKNOWN"
        ) from exc
    if not isinstance(instances, list):
        return None
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("name") != name:
            continue
        status = inst.get("status") or ""
        if status.upper() in {"TERMINATED", "STOPPED", "SUSPENDED"}:
            continue
        zone_url = inst.get("zone") or ""
        # The zone field is a URL; take the last path segment.
        zone = zone_url.rsplit("/", 1)[-1] if zone_url else config.primary_zone
        instance_id = str(inst.get("id") or "")
        # Recover the original attempt_id from the instance's labels (set
        # by ``_format_labels`` at create time as ``eps-attempt=<id>``).
        # WITHOUT this, ``launch()`` on the reconnect path would derive
        # the ExpectedArtifacts declaration from a FRESH attempt_id, but
        # the VM writes its sentinel + per-attempt artifact dirs under
        # the ORIGINAL attempt_id — so ``confirm_artifacts`` would always
        # FAIL on reconnect (sentinel-path mismatch). Labels accept only
        # ``[a-z0-9_-]``, so a colon/dot-bearing attempt_id would have
        # been sanitized at create time; downstream code must therefore
        # treat the recovered label value as the canonical attempt_id
        # for this instance's lifetime (the VM-side paths match it).
        labels = inst.get("labels") or {}
        recovered_attempt_id: str | None = None
        if isinstance(labels, dict):
            raw = labels.get("eps-attempt")
            if raw:
                recovered_attempt_id = str(raw)
        extra: dict[str, Any] = {
            "intent": spec.intent,
            "issue": int(spec.issue),
            "project": config.project,
            "gcloud_config": config.gcloud_config,
            "zone": zone,
            "instance_name": name,
            "status_at_reconnect": status,
            "reconnected": True,
        }
        if recovered_attempt_id is not None:
            extra["attempt_id"] = recovered_attempt_id
        return RunHandle(
            backend="gcp",
            cluster=None,
            job_id=instance_id,
            pod_name=name,
            scratch_dir=workload_dir_for(config, spec.issue),
            log_path=f"{workload_dir_for(config, spec.issue)}/logs/issue-{spec.issue}.log",
            extra=extra,
        )
    return None


# ---------------------------------------------------------------------------
# Pre-create regional-quota headroom probe (#608)
# ---------------------------------------------------------------------------


#: ``MachineSpec.gpu_kind`` → the regional accelerator-quota metric reported
#: by ``gcloud compute regions describe`` ``quotas[]``. Verified live on
#: issue 608 (2026-06-12): ``NVIDIA_A100_80GB_GPUS`` read usage 8.0 / limit
#: 8.0 while the ``ft-7b`` intent needed 4 — every create was doomed.
_GPU_KIND_TO_QUOTA_METRIC: dict[str, str] = {
    "A100-80": "NVIDIA_A100_80GB_GPUS",
    "L4": "NVIDIA_L4_GPUS",
}


@dataclass(frozen=True)
class QuotaHeadroom:
    """One regional accelerator-quota reading for a planned launch.

    ``sufficient`` is the router's skip predicate: headroom
    (``limit - usage``) must cover the machine's GPU count.
    """

    metric: str
    region: str
    limit: float
    usage: float
    needed: int

    @property
    def available(self) -> float:
        """GPUs the regional quota still admits (``limit - usage``)."""
        return self.limit - self.usage

    @property
    def sufficient(self) -> bool:
        """True when the remaining headroom covers ``needed`` GPUs."""
        return self.available >= self.needed


def preflight_quota_headroom(
    *, spec: RunSpec, config: GcpConfig, runner: GcloudRunner
) -> QuotaHeadroom | None:
    """Read the regional accelerator-quota headroom for ``spec``; ``None`` = no opinion.

    Called by the router's GCP lane BEFORE the per-day attempt-counter
    bump so a create that CANNOT succeed (regional quota already at its
    limit) is skipped without burning an attempt. Issue 608 (2026-06-12):
    four quota-doomed creates consumed the cap while
    ``NVIDIA_A100_80GB_GPUS`` sat at 8/8 with 4 needed.

    FAIL-OPEN contract — returns ``None`` ("no opinion; proceed to launch
    exactly as before") whenever:

    * the intent has no machine / quota-metric mapping (the launch path
      fails loud on its own),
    * a live ``eps-issue-<N>`` instance already exists (the launch path
      reconnects, consuming no new quota — and our own instance may BE
      the usage the probe would read),
    * the reconnect probe or the ``regions describe`` call fails in ANY
      way (rc != 0, missing gcloud, timeout, unparseable JSON, metric
      absent from ``quotas[]``).

    Only a successfully parsed quota row produces a verdict. A swallowed
    probe failure here never enables a blind create: the launch path
    re-runs its own reconnect probe and raises typed-ly on failure.
    """
    try:
        machine = machine_for_intent(spec)
    except ValueError:
        return None
    metric = _GPU_KIND_TO_QUOTA_METRIC.get(machine.gpu_kind)
    if metric is None:
        return None
    try:
        if reconnect_or_none(spec=spec, config=config, runner=runner) is not None:
            return None
    except Exception as exc:  # GcpProbeError / transport — fail OPEN (launch re-probes)
        logger.warning(
            "GCP quota pre-check: reconnect probe failed OPEN (%s: %s); proceeding to launch.",
            type(exc).__name__,
            exc,
        )
        return None
    region = region_for_zone(config.primary_zone)
    try:
        result = runner(render_region_describe_argv(config=config, region=region))
    except Exception as exc:  # missing gcloud / TimeoutExpired — fail OPEN per #608
        logger.warning(
            "GCP quota pre-check: regions describe failed OPEN (%s: %s); proceeding to launch.",
            type(exc).__name__,
            exc,
        )
        return None
    if result.returncode != 0:
        logger.warning(
            "GCP quota pre-check: regions describe rc=%d (%s); failing OPEN.",
            result.returncode,
            result.stderr[:300],
        )
        return None
    try:
        payload = json.loads(result.stdout or "{}")
        quotas = payload.get("quotas") or []
        row = next(
            (q for q in quotas if isinstance(q, dict) and q.get("metric") == metric),
            None,
        )
        if row is None:
            return None
        limit = float(row["limit"])
        usage = float(row["usage"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "GCP quota pre-check: unparseable quotas payload (%s: %s); failing OPEN.",
            type(exc).__name__,
            exc,
        )
        return None
    return QuotaHeadroom(
        metric=metric, region=region, limit=limit, usage=usage, needed=machine.gpu_count
    )


# ---------------------------------------------------------------------------
# Stale-VM reaper (cron entrypoint)
# ---------------------------------------------------------------------------


def audit_stale_gcp_vms(
    *,
    config: GcpConfig | None = None,
    runner: GcloudRunner | None = None,
    max_age_seconds: int = 24 * 3600,
    now: datetime | None = None,
    delete: bool = False,
) -> list[dict[str, Any]]:
    """List (and optionally delete) ``eps-issue-*`` instances older than the threshold.

    Analogue of ``scripts/pod.py audit-stale`` for GCP. Without it, an
    orchestrator crash that drops the local lease before teardown would
    leak a VM at $5/hr — the cron is the credit-leak backstop.

    Returns a list of ``{name, zone, status, created_at, age_seconds,
    action}`` records (``action`` ∈ {``"would-delete"``, ``"deleted"``,
    ``"skipped"``}). When ``delete=True``, instances over the threshold
    are issued a ``gcloud compute instances delete --quiet`` (errors are
    logged + folded into the record as ``action="delete-failed"`` — never
    raised, so the cron continues across the rest of the inventory).

    No ``raise`` on a benign empty list — a fresh GCP project legitimately
    has zero matches.
    """
    cfg = config or default_gcp_config()
    run = runner or default_gcloud_runner
    reference = now or datetime.now(tz=UTC)
    argv = render_list_argv(config=cfg, name_filter="name~^eps-issue-")
    result = run(argv)
    if result.returncode != 0:
        logger.error(
            "audit_stale_gcp_vms: list returned %d; cannot audit. stderr=%s",
            result.returncode,
            result.stderr[:500],
        )
        return []
    try:
        instances = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError as exc:
        logger.error("audit_stale_gcp_vms: bad JSON from gcloud list: %s", exc)
        return []
    if not isinstance(instances, list):
        return []

    records: list[dict[str, Any]] = []
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        name = inst.get("name") or ""
        if not name.startswith("eps-issue-"):
            continue
        zone_url = inst.get("zone") or ""
        zone = zone_url.rsplit("/", 1)[-1] if zone_url else cfg.primary_zone
        status = inst.get("status") or "UNKNOWN"
        created_at_raw = inst.get("creationTimestamp")
        age_seconds = _age_seconds(created_at_raw, reference)
        action: str
        if age_seconds is None or age_seconds < max_age_seconds:
            action = "skipped"
        elif not delete:
            action = "would-delete"
        else:
            del_argv = render_delete_argv(config=cfg, name=name, zone=zone)
            del_result = run(del_argv)
            if del_result.returncode == 0:
                action = "deleted"
            else:
                logger.error(
                    "audit_stale_gcp_vms: delete %s failed (%d): %s",
                    name,
                    del_result.returncode,
                    del_result.stderr[:300],
                )
                action = "delete-failed"
        records.append(
            {
                "name": name,
                "zone": zone,
                "status": status,
                "created_at": created_at_raw,
                "age_seconds": age_seconds,
                "action": action,
            }
        )
    return records


def _age_seconds(created_at_raw: Any, reference: datetime) -> float | None:
    """Parse ``creationTimestamp`` (ISO-8601 with offset) and return age in seconds."""
    if not isinstance(created_at_raw, str) or not created_at_raw:
        return None
    try:
        parsed = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (reference - parsed).total_seconds()


# ---------------------------------------------------------------------------
# GcpBackend — the ComputeBackend
# ---------------------------------------------------------------------------


class GcpBackend(ComputeBackend):
    """GCE-VM backend (single VM per issue, ephemeral lifecycle).

    Mirrors the RunPod lifecycle shape:

    * ``prepare`` — no-op (provision triggers bootstrap inline via the
      startup-script, exactly like ``pod_lifecycle.py provision``).
    * ``launch`` — reconnect (idempotent) → render startup-script →
      render create argv → run ``gcloud compute instances create`` →
      populate :class:`ExpectedArtifacts` on the handle → post marker →
      return handle. On a typed :class:`GcpProvisioningError` (capacity)
      retry on each ``config.fallback_zones`` zone before raising.
    * ``estimate_start`` — UTC now (GCP provisions immediately when
      capacity exists; no test-only probe analogue today).
    * ``poll`` — ``gcloud compute instances describe`` for the status;
      decode to a :class:`PollResult`. Slice 3 does NOT walk the
      in-VM log; that lands when the orchestrator-side bg poll is
      wired (slice 6).
    * ``fetch_logs`` — best-effort serial-port-1 pull via
      ``gcloud compute instances get-serial-port-output``. Returns ``""``
      when the call fails (a fresh VM has no serial output yet).
    * ``fetch_results`` — no-op (authoritative artifacts already off-VM
      during the run; a slice-6 cleanup may add a best-effort scp).
    * ``confirm_artifacts`` — delegates to the slice-2 verifier exactly
      like :class:`SlurmBackend.confirm_artifacts`. The launch path
      populates :class:`ExpectedArtifacts` on the handle's ``extra`` so
      a missing declaration is itself a FAIL.
    * ``teardown`` — ``gcloud compute instances delete --quiet``; the
      ``--instance-termination-action=DELETE`` + ``--max-run-duration``
      double-belt means a no-op teardown on a missing instance is the
      common path.

    Constructor parameters are injection seams; tests provide a fake
    runner + marker poster so the unit suite never hits gcloud.
    """

    def __init__(
        self,
        *,
        config: GcpConfig | None = None,
        runner: GcloudRunner | None = None,
        marker_poster: Callable[..., None] | None = None,
        marker_reader: Callable[..., dict[str, Any] | None] | None = None,
        startup_script_renderer: Callable[..., str] | None = None,
    ) -> None:
        self._config = config or default_gcp_config()
        self._run = runner or default_gcloud_runner
        # Lazy import default poster (matches SlurmBackend's pattern) so
        # this module stays importable without a configured task.py.
        if marker_poster is None:
            from explore_persona_space.backends.slurm import post_marker_via_task_py

            marker_poster = post_marker_via_task_py
        self._post_marker = marker_poster
        # Marker READ seam (``(issue, prefix) -> latest event dict | None``).
        # Default is the branch-guarded library read — the same pure,
        # no-commit pattern ``poll_pipeline._marker_pid`` uses — so
        # ``poll`` can follow an SSH-relaunched workload via its fresh
        # ``epm:run-launched`` marker (incident #612). Tests inject a fake.
        if marker_reader is None:
            from explore_persona_space.task_workflow import latest_event

            marker_reader = latest_event
        self._read_marker = marker_reader
        self._render_startup = startup_script_renderer or render_startup_script

    # ----- identity --------------------------------------------------------

    @property
    def name(self) -> BackendKind:
        return "gcp"

    # ----- public read-only handles to injection-seam state ---------------
    #
    # The dispatch-issue ``_reconnect`` closure needs to call
    # ``gcp.reconnect_or_none(spec=..., config=..., runner=...)`` and so
    # MUST be able to read the same ``GcpConfig`` and ``GcloudRunner`` this
    # backend instance was built with. Previously it reached into the
    # underscored fields (``gcp_backend.config`` and
    # ``gcp_backend._runner``), but the constructor stores them as
    # ``self._config`` and ``self._run`` — every explicit
    # ``backend: gcp`` lane (and every auto-chain GCP escalation that hit
    # the reconnect path) AttributeError'd at production-wiring time. The
    # properties below are the public read-only view; tests AND
    # production callers must use them rather than reaching into the
    # underscored names (parity with the ``runpod`` / SLURM backends,
    # which expose their injection seams through public properties /
    # methods).
    @property
    def config(self) -> GcpConfig:
        """The :class:`GcpConfig` this backend was constructed with."""
        return self._config

    @property
    def runner(self) -> GcloudRunner:
        """The ``GcloudRunner`` callable this backend was constructed with."""
        return self._run

    # ----- launch ----------------------------------------------------------

    def prepare(self, spec: RunSpec) -> None:
        """No-op. GCP bootstrap happens inside the startup-script the
        VM runs on first boot (same one-shot model as RunPod's
        ``pod_lifecycle.py provision``)."""
        del spec
        return None

    def preflight_quota_headroom(self, spec: RunSpec) -> QuotaHeadroom | None:
        """Regional accelerator-quota headroom for ``spec``, or ``None`` (no opinion).

        Duck-typed seam the router's GCP lane probes BEFORE bumping the
        per-day attempt counter (#608: quota-doomed creates burned the
        cap). Delegates to :func:`preflight_quota_headroom` with this
        backend's config + runner; the FAIL-OPEN contract lives there.
        """
        return preflight_quota_headroom(spec=spec, config=self._config, runner=self._run)

    def launch(self, spec: RunSpec) -> RunHandle:
        """Provision (or reconnect to) the GCE VM for ``spec.issue``.

        See class docstring for the per-step flow. Raises
        :class:`GcpProvisioningError` when every zone (primary +
        fallbacks) returns a capacity / quota / auth failure — the router
        catches that and proceeds to the next tier.
        """
        config = self._config
        attempt_id = attempt_id_for(spec)

        # Reconnect: a live instance with the canonical name is the
        # idempotent re-entry path (orchestrator re-spawn, manual
        # ``/issue`` re-invocation). Skip provisioning entirely.
        existing = reconnect_or_none(spec=spec, config=config, runner=self._run)
        if existing is not None:
            # Reconnect: thread the ORIGINAL attempt_id (recovered from
            # the instance's ``eps-attempt`` label by ``reconnect_or_none``)
            # into the ExpectedArtifacts declaration. The VM was provisioned
            # under that attempt_id and writes its sentinel + per-attempt
            # artifact dirs under it; deriving the declaration from a
            # FRESH attempt_id would make ``confirm_artifacts`` look at the
            # wrong sentinel path and FAIL on every reconnect. Fall back
            # to the freshly-generated ``attempt_id`` only when the label
            # wasn't present (e.g. an instance created by an older code
            # path before the labels were added).
            logger.info(
                "GCP reconnect: handle existing instance %s in %s",
                existing.pod_name,
                existing.extra.get("zone"),
            )
            return self._with_artifacts_declaration(
                handle=existing,
                spec=spec,
                config=config,
                attempt_id=str(existing.extra.get("attempt_id") or attempt_id),
                wandb_run_path=spec.extra.get("wandb_run_path"),
            )

        # Resolve workload secrets BEFORE rendering anything — fails loud
        # (GcpLaunchSecretsMissing) when the required keys are absent from
        # spec.extra / the process env / the project .env, so a doomed VM
        # is never provisioned (issue 535 GCP lane r7: empty WANDB_API_KEY
        # crashed the workload after the full boot + uv-sync spend). The
        # resolved values land in spec.extra["secret_<KEY>"], which
        # render_create_argv prefers over its bare-env fallback.
        resolve_launch_secrets(spec)

        # Render the startup-script + persist it to a per-launch tempfile,
        # then thread the path so ``render_create_argv`` takes the
        # ``--metadata-from-file=startup-script=<path>`` branch. The inline
        # ``--metadata=startup-script=<body>`` shape is mangled by gcloud's
        # KEY=VALUE dict parser whenever the body contains commas — and the
        # rendered body's completion-sentinel JSON
        # (``{"phase":"done","issue":...,"attempt_id":"..."}``) always does.
        # The renderer's docstring already prefers the tempfile path "so
        # secrets-bearing scripts never leak through argv"; this matches the
        # control flow to the docstring. Verified on the 2026-06-08 $1 live
        # GCP test, which failed with ``Bad syntax for dict arg`` until the
        # call was rewritten to use ``--metadata-from-file``.
        startup = self._render_startup(
            spec=spec,
            config=config,
            attempt_id=attempt_id,
            hydra_args=spec.hydra_args,
            # The startup script CLONES from origin (unlike the SLURM
            # backend, which rsyncs the local worktree) — a workload
            # whose code/configs live on a feature branch MUST thread
            # that branch or the VM silently runs stale main (live
            # finding, issue 535 GCP lane r6: the smoke condition config
            # existed only on the local branch; Hydra died listing
            # available conditions).
            repo_branch=str(spec.extra.get("repo_branch") or "main"),
        )
        # Mode 0o600 so the script — which carries the curl stanza that
        # fetches secrets from instance metadata — is never world-readable
        # on the VM either (matches the slurm secrets-tempfile pattern).
        fd, startup_path = tempfile.mkstemp(
            prefix=f"eps-gcp-startup-{spec.issue}-",
            suffix=".sh",
        )
        try:
            os.write(fd, startup.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(startup_path, 0o600)
        # In-place mutation of the mutable ``extra`` dict is the cleanest
        # way to thread the path through to ``render_create_argv``'s
        # existing ``spec.extra["startup_script_path"]`` contract (RunSpec
        # is frozen, but its ``extra`` dict is mutable by design).
        spec.extra["startup_script_path"] = startup_path

        # Per-secret 0600 tempfiles for the --metadata-from-file channel:
        # token values never touch the gcloud argv / process list (round-2
        # Codex Major, task #535). Same resolution order as the renderer
        # (spec.extra["secret_<KEY>"] from resolve_launch_secrets, then
        # env); the files are deleted in the finally below the moment the
        # create loop is done with them.
        secret_files: dict[str, str] = {}
        for key in STARTUP_SECRET_ENV_KEYS:
            val = spec.extra.get(f"secret_{key}") or _envget(key)
            if val is None or val == "":
                continue
            sfd, secret_path = tempfile.mkstemp(
                prefix=f"eps-gcp-secret-{spec.issue}-{key.lower()}-",
            )
            try:
                os.write(sfd, str(val).encode("utf-8"))
            finally:
                os.close(sfd)
            os.chmod(secret_path, 0o600)
            secret_files[key] = secret_path

        zones_to_try: list[str] = [config.primary_zone]
        zones_to_try.extend(z for z in config.fallback_zones if z and z != config.primary_zone)
        last_error: GcpProvisioningError | None = None
        try:
            for zone in zones_to_try:
                argv = render_create_argv(
                    spec=spec,
                    config=config,
                    attempt_id=attempt_id,
                    zone=zone,
                    startup_script=startup,
                    secret_files=secret_files,
                )
                logger.info("GCP create issue=%d in zone=%s", spec.issue, zone)
                result = self._run(argv)
                if result.returncode == 0:
                    break
                last_error = classify_create_failure(
                    returncode=result.returncode,
                    stderr=result.stderr,
                )
                # Only retry on a capacity-shaped failure (not on auth/quota
                # which won't be fixed by trying a different zone). The
                # classifier tags the matched pattern in evidence; capacity
                # patterns match the substring "RESOURCE" / "EXHAUSTED" /
                # "does not have enough resources".
                matched = (last_error.evidence.get("matched_pattern") or "").lower()
                if not any(tag in matched for tag in ("exhaust", "resource", "enough resources")):
                    # Non-capacity failure → don't retry; surface immediately.
                    raise last_error
                logger.warning(
                    "GCP create capacity miss in zone=%s; trying next fallback. reason=%s",
                    zone,
                    last_error.reason,
                )
            else:
                # for-else: executed when the for loop completes without
                # `break` — every zone failed.
                assert last_error is not None
                raise last_error
        finally:
            # gcloud has read the secret files by the time create returns
            # (success or failure) — shred the on-disk token copies.
            for secret_path in secret_files.values():
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(secret_path)

        # Successful create. Build the handle + thread the artifact
        # declaration through handle.extra. The handle name matches
        # the gcloud name (idempotent reconnect uses it).
        instance_name = instance_name_for(spec.issue)
        # gcloud returns the instance object as a list with one entry.
        instance_id = _parse_instance_id(result.stdout, instance_name)
        handle = RunHandle(
            backend="gcp",
            cluster=None,
            job_id=instance_id,
            pod_name=instance_name,
            scratch_dir=workload_dir_for(config, spec.issue),
            log_path=f"{workload_dir_for(config, spec.issue)}/logs/issue-{spec.issue}.log",
            extra={
                "intent": spec.intent,
                "issue": int(spec.issue),
                "project": config.project,
                "gcloud_config": config.gcloud_config,
                "zone": zone,
                "instance_name": instance_name,
                "attempt_id": attempt_id,
                "provisioning_model": resolve_provisioning_model(spec),
                "machine_type": machine_for_intent(spec).machine_type,
                "reconnected": False,
            },
        )
        handle = self._with_artifacts_declaration(
            handle=handle,
            spec=spec,
            config=config,
            attempt_id=attempt_id,
            wandb_run_path=spec.extra.get("wandb_run_path"),
        )

        # Marker: ``epm:cluster-launched`` is the SLURM analogue; we
        # reuse the same marker name so the dashboard surfaces GCP runs
        # in the same lane (the body carries ``backend: gcp``). This
        # mirrors SlurmBackend.launch.
        marker_body = json.dumps(
            {
                "backend": "gcp",
                "instance_name": instance_name,
                "instance_id": instance_id,
                "project": config.project,
                "zone": zone,
                "machine_type": machine_for_intent(spec).machine_type,
                "provisioning_model": resolve_provisioning_model(spec),
                "attempt_id": attempt_id,
                # Additive field (#588): which workload shape the startup
                # script renders — "custom" (spec.workload_cmd verbatim)
                # vs "hydra" (scripts/train.py + hydra_args).
                "workload": "custom" if spec.workload_cmd else "hydra",
            },
            sort_keys=True,
        )
        try:
            self._post_marker(
                issue=spec.issue,
                marker="epm:cluster-launched",
                note=marker_body,
                version=1,
                by="backends.gcp",
            )
        except Exception as exc:
            # Marker post is best-effort: the VM already exists, and
            # surfacing a marker failure shouldn't tear it down. Log
            # loudly so the operator can backfill if needed.
            logger.error(
                "GCP launch: marker post failed for issue=%d: %s; continuing.",
                spec.issue,
                exc,
            )

        return handle

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        """GCE on-demand provisions immediately; informational "now"."""
        del spec
        return datetime.now(tz=UTC)

    def estimate_start_seconds(
        self,
        spec: RunSpec,
        *,
        now: datetime | None = None,
    ) -> float | None:
        """Seconds until ``spec`` would start. GCE is ~0.

        Returned as 0.0 for both on-demand and Spot (Spot is ~0 when
        capacity exists; we don't probe live capacity in slice 3, and
        the router's 10-min park is the source of truth for "did the
        job actually start" anyway).
        """
        del spec, now
        return 0.0

    # ----- monitor ---------------------------------------------------------

    def poll(self, handle: RunHandle) -> PollResult:
        """One-tick poll via ``gcloud compute instances describe``.

        Slice 3 returns a coarse PollResult derived from the VM status
        only (``RUNNING`` → ``running``; ``TERMINATED`` → ``dead`` etc.).
        Slice 6 will overlay the per-phase heartbeat once the in-VM
        ``[phase=...]`` writes land on a poll-readable surface (the
        existing :class:`PollResult` shape carries the per-phase fields).

        Terminal guest-attribute phases (``done`` / ``failed``) are
        OVERRIDDEN when a fresh ``epm:run-launched`` relaunch marker
        names a live process on this instance — the startup script's
        phase write freezes at the FIRST workload's exit, so an SSH
        hot-fix relaunch is otherwise invisible (incident #612). See
        :meth:`_relaunch_marker_or_none` / :meth:`_probe_relaunched_workload`.
        """
        config = self._config
        zone = handle.extra.get("zone") or config.primary_zone
        argv = render_describe_argv(config=config, name=handle.pod_name, zone=zone)
        result = self._run(argv)
        if result.returncode != 0:
            # 404 → instance gone → terminal "dead". gcloud returns a
            # non-zero exit + a "was not found" stderr in that case.
            stderr_low = (result.stderr or "").lower()
            if "was not found" in stderr_low or "404" in stderr_low:
                return _terminal_dead_poll(reason="instance not found")
            # Other failures: treat as transient "stalled" so the
            # orchestrator's bg poll keeps retrying rather than tearing
            # down a healthy VM.
            return _coarse_poll(status="stalled", current_phase="describe_failed")
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            return _coarse_poll(status="stalled", current_phase="describe_bad_json")
        status = (payload.get("status") or "UNKNOWN").upper()
        if status == "RUNNING":
            # Drain workload-written sentinel files FIRST (mirrors
            # ``poll_pipeline.poll_once``): pod-side dispatchers post
            # markers by writing ``/workspace/logs/issue-<N>-*.json``
            # sentinels. Pre-#608 the GCP lane had NO drain at all, so a
            # completed run's ``epm:results`` sentinel sat root-owned
            # (mode 600 — the GCE startup script runs as root) on the VM
            # and the carried marker never posted; ``backend_poll``
            # reported a silent ``sentinels_processed=0`` with an empty
            # log tail. The drain + log-tail reads below go through
            # ``sudo -n`` for that reason.
            drained, drain_gate, drain_alarm, drain_log_tail = self._drain_sentinels(handle, zone)

            def _with_drain(base: PollResult) -> PollResult:
                return _overlay_drain(
                    base,
                    processed=drained,
                    gate=drain_gate,
                    alarm=drain_alarm,
                    log_tail=drain_log_tail,
                )

            # A RUNNING VM is ambiguous: booting, mid-workload, or DONE
            # (the success path deliberately keeps the VM up so the
            # completion sentinel can be scp'd — instance state alone
            # can never signal success; issue 535 r9 spun the poll for
            # the full 4 h timeout on a 9-min success). Overlay the
            # workload phase from the eps/phase guest attribute.
            try:
                phase = self._guest_phase(handle, zone)
            except GcpProbeError as exc:
                # Typed probe failure (auth / API / parse — NOT the
                # expected attribute-not-written-yet case, which returns
                # ""). Surface as a typed stalled tick so the bg poll's
                # consecutive-failure budget sees it instead of an
                # indistinguishable "still running" that can spin a
                # finished workload to the outer timeout (round-2 Codex
                # Major, task #535).
                logger.warning(
                    "GCP poll: guest-attribute probe failed for %s (%s); "
                    "returning typed stalled tick.",
                    handle.pod_name,
                    exc,
                )
                return _with_drain(
                    _coarse_poll(status="stalled", current_phase="guest_attr_probe_failed")
                )
            if phase in ("done", "failed"):
                # Relaunch-follow (incident #612): the eps/phase guest
                # attribute is written by the STARTUP SCRIPT, so it
                # freezes at the FIRST workload's terminal state. A
                # sanctioned SSH hot-fix relaunch (CLAUDE.md "push
                # through bugs", the experimenter respawn path) posts a
                # fresh ``epm:run-launched`` marker with ``pid=`` +
                # ``log_abs=`` precisely so pollers can follow the new
                # process — without this branch a HEALTHY mid-training
                # relaunch read as ``done``/``dead`` and steered the
                # orchestrator to a premature transition.
                relaunch = self._relaunch_marker_or_none(handle)
                if relaunch is not None:
                    pid, log_abs = relaunch
                    return _with_drain(
                        self._probe_relaunched_workload(handle, zone, pid=pid, log_path=log_abs)
                    )
            if phase == "done":
                return _with_drain(
                    PollResult(
                        status="done",
                        current_phase="workload_done",
                        new_milestone=True,
                        last_log_mtime_sec_ago=0,
                        pid_alive=False,
                        log_tail_excerpt="",
                    )
                )
            if phase == "failed":
                return _with_drain(_terminal_dead_poll(reason="workload_failed"))
            if phase:
                return _with_drain(_coarse_poll(status="running", current_phase=phase))
            return _with_drain(_gcp_status_to_poll_result(status))
        return _gcp_status_to_poll_result(status)

    def _guest_phase(self, handle: RunHandle, zone: str) -> str:
        """Read the ``eps/phase`` guest attribute; "" when not yet written.

        Two failure classes are deliberately distinguished (round-2
        Codex Major, task #535 — pre-fix, EVERY nonzero rc / bad-JSON
        read returned "" and was indistinguishable from "phase not
        written yet", so an auth/API/parse failure could spin a finished
        workload to the outer poll timeout):

        * EXPECTED not-written-yet — gcloud exits nonzero with a
          404 / "not found" stderr (the guest attribute does not exist
          until the startup-script's first ``_eps_phase`` write).
          Returns ``""`` so the caller keeps the coarse instance-status
          classification and retries next tick.
        * Probe failure — any OTHER nonzero rc (expired auth, permission
          denied, transport) or unparseable JSON from an rc=0 call.
          Raises :class:`GcpProbeError` (the probe-typing discipline
          from ``reconnect_or_none`` / the SLURM round-6 B1 contract:
          "couldn't ask" must never read as "not done yet"); ``poll()``
          translates it into a typed stalled tick.
        """
        config = self._config
        argv = render_guest_attributes_argv(config=config, name=handle.pod_name, zone=zone)
        result = self._run(argv)
        if result.returncode != 0:
            stderr_low = (result.stderr or "").lower()
            if "not found" in stderr_low or "404" in stderr_low:
                return ""  # attribute not written yet — legitimate pre-phase state
            raise GcpProbeError(
                f"GCP guest-attribute probe failed for {handle.pod_name}: "
                f"rc={result.returncode} stderr={result.stderr[:500]!r} — workload "
                "phase UNKNOWN, refusing to read a probe failure as still-running"
            )
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError as exc:
            raise GcpProbeError(
                f"GCP guest-attribute probe returned unparseable JSON for "
                f"{handle.pod_name}: {exc} — workload phase UNKNOWN"
            ) from exc
        # gcloud returns a list of {namespace, key, value} dicts.
        for item in payload if isinstance(payload, list) else []:
            if item.get("key") == "phase":
                return str(item.get("value") or "").strip()
        return ""

    # Log-tail trailer delimiters for the combined drain+tail SSH command.
    # Namespaced (``EPS_``) so a stray ``LOGTAIL`` substring in workload
    # output can't truncate the sentinel section.
    _LOGTAIL_START = "EPS_LOGTAIL_START"
    _LOGTAIL_END = "EPS_LOGTAIL_END"

    def _drain_sentinels(self, handle: RunHandle, zone: str) -> tuple[int, str | None, str, str]:
        """Drain ``/workspace/logs`` sentinels + pull a log tail via ssh sudo.

        ONE ``gcloud compute ssh`` round-trip runs the shared drain loop
        (``poll_pipeline.sentinel_drain_shell``) plus a log-tail trailer,
        wrapped in ``sudo -n bash -c``: the GCE startup script runs as
        root, so the sentinel files and workload log are root-owned mode
        600 and a plain user-mode read comes back EMPTY (incident #608 —
        a completed run's ``epm:results`` marker never posted). ``sudo
        -n`` works because the OS-Login user is in ``google-sudoers``
        (same transport as ``fetch_results``' sentinel pull).

        Parsed sentinels are posted via the transport-agnostic
        ``poll_pipeline.drain_sentinels_via`` (idempotent: each posted
        sentinel is renamed ``.processed`` through
        :meth:`_mark_sentinel_processed`, also via sudo).

        Returns ``(processed, gate, alarm, log_tail)``. ``alarm`` is ""
        normally; on a transport failure OR a matched-but-unprocessable
        sentinel set it carries a loud one-line diagnosis the caller
        surfaces in ``log_tail_excerpt`` — never a silent
        ``sentinels_processed=0`` (fail-LOUD contract, #608).
        """
        issue = int(handle.extra.get("issue") or 0)
        if issue <= 0:
            alarm = "gcp sentinel drain SKIPPED: handle missing 'issue' extra"
            logger.warning("GCP poll: %s. handle=%r", alarm, handle)
            return 0, None, alarm, ""

        # Lazy import (mirrors RunPodBackend.poll): production entrypoints
        # put the repo root on sys.path (backend_poll.py bootstrap, #571);
        # fall back to a __file__-derived insert for direct library use.
        try:
            from scripts.poll_pipeline import (
                drain_sentinels_via,
                parse_sentinel_stream,
                sentinel_drain_shell,
            )
        except ModuleNotFoundError:
            import sys

            repo_root = str(Path(__file__).resolve().parents[3])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from scripts.poll_pipeline import (
                drain_sentinels_via,
                parse_sentinel_stream,
                sentinel_drain_shell,
            )

        log_path = handle.log_path or ""
        tail_stanza = (
            f'echo "{self._LOGTAIL_START}"; '
            + (f"tail -n 30 {shlex.quote(log_path)} 2>/dev/null || true; " if log_path else "")
            + f'echo "{self._LOGTAIL_END}"'
        )
        script = sentinel_drain_shell(issue) + "; " + tail_stanza
        argv = _base_gcloud_argv(
            self._config,
            "compute",
            "ssh",
            handle.pod_name,
            f"--command=sudo -n bash -c {shlex.quote(script)}",
        )
        argv += [f"--zone={zone}"]
        res = self._run(argv)
        if res.returncode != 0:
            alarm = (
                f"gcp sentinel drain FAILED (rc={res.returncode}): "
                f"{(res.stderr or '').strip()[:300]}"
            )
            logger.error("GCP poll: %s", alarm)
            return 0, None, alarm, ""

        stdout = res.stdout or ""
        drain_part, _, tail_part = stdout.partition(self._LOGTAIL_START)
        log_tail = tail_part.split(self._LOGTAIL_END)[0].strip()[:2000] if tail_part else ""
        sentinels = parse_sentinel_stream(drain_part)
        processed, gate = drain_sentinels_via(
            issue=issue,
            list_sentinels=lambda: sentinels,
            mark_processed=lambda remote_path: self._mark_sentinel_processed(
                handle, zone, remote_path
            ),
        )
        if sentinels and processed == 0:
            # The glob matched files but nothing was posted (empty or
            # unparseable bodies, or marker-post failures). Pre-#608 this
            # exact situation reported a silent ``sentinels_processed=0``;
            # surface it loudly instead.
            alarm = (
                f"gcp sentinel drain: {len(sentinels)} sentinel(s) matched but 0 "
                "processed (empty/unparseable body or marker-post failure) — "
                "inspect /workspace/logs on the VM + poller stderr"
            )
            logger.error("GCP poll: %s", alarm)
            return 0, gate, alarm, log_tail
        return processed, gate, "", log_tail

    def _mark_sentinel_processed(self, handle: RunHandle, zone: str, remote_path: str) -> bool:
        """Rename a drained sentinel to ``<path>.processed`` via ssh sudo.

        ``mv -n`` (no clobber) mirrors ``poll_pipeline._ssh_mark_processed``;
        ``sudo -n`` because the file is root-owned (#608). Returns False on
        failure (the caller leaves the sentinel for the next tick).
        """
        quoted = shlex.quote(remote_path)
        argv = _base_gcloud_argv(
            self._config,
            "compute",
            "ssh",
            handle.pod_name,
            f"--command=sudo -n mv -n {quoted} {quoted}.processed",
        )
        argv += [f"--zone={zone}"]
        res = self._run(argv)
        if res.returncode != 0:
            logger.error(
                "GCP poll: sentinel rename failed for %s (rc=%d): %s",
                remote_path,
                res.returncode,
                (res.stderr or "")[:300],
            )
            return False
        return True

    # ----- relaunch-follow (incident #612) ---------------------------------
    #
    # ``epm:run-launched`` note tokens, per the relaunch contract in
    # `.claude/skills/issue/SKILL.md` ("Any relaunch must re-post
    # epm:run-launched" — `pod=<name> pid=<pid> log_abs=<abs path>`).
    # ``pid=`` mirrors ``poll_pipeline.MARKER_PID_RE``; ``log=`` is the
    # legacy fallback accepted through the transition window.
    _RELAUNCH_PID_RE = re.compile(r"\bpid=(\d+)")
    _RELAUNCH_LOG_ABS_RE = re.compile(r"\blog_abs=(\S+)")
    _RELAUNCH_LOG_LEGACY_RE = re.compile(r"\blog=(\S+)")
    _RELAUNCH_POD_RE = re.compile(r"\bpod=(\S+)")
    # Probe-output delimiters (namespaced like the drain's EPS_LOGTAIL_*).
    _RELAUNCH_TAIL_START = "EPS_RELAUNCH_TAIL_START"
    _RELAUNCH_TAIL_END = "EPS_RELAUNCH_TAIL_END"

    def _relaunch_marker_or_none(self, handle: RunHandle) -> tuple[int, str] | None:
        """Return ``(pid, log_path)`` from a relaunch marker, or ``None``.

        A relaunch marker is the latest ``epm:run-launched`` event whose
        note carries ``pid=`` AND ``log_abs=`` (legacy ``log=``) and that
        provably targets THIS instance generation:

        * its ``pod=`` field equals ``handle.pod_name`` (an SSH relaunch
          on the GCE VM posts the instance name; a stale RunPod-era
          marker posts ``pod-<N>`` and is rejected), AND
        * when the launch-time ``epm:cluster-launched`` marker exists,
          the relaunch marker is STRICTLY NEWER than it — a marker from
          a previous instance generation (VM deleted + re-provisioned)
          must not hijack the fresh generation's poll. When the
          cluster-launched marker is absent (its post is best-effort),
          the ``pod=`` match alone is accepted.

        Returns ``None`` (→ caller keeps the existing terminal-phase
        behavior) when the issue is unresolvable, the marker read fails,
        or any predicate fails. Marker-read failures are logged loudly
        but never crash a poll tick.
        """
        issue = int(handle.extra.get("issue") or 0)
        if issue <= 0:
            return None
        try:
            ev = self._read_marker(issue, "epm:run-launched")
        except Exception as exc:
            logger.warning(
                "GCP poll: epm:run-launched read failed for issue=%d (%s); "
                "keeping startup-script terminal state.",
                issue,
                exc,
            )
            return None
        if not ev:
            return None
        note = str(ev.get("note") or "")
        pid_m = self._RELAUNCH_PID_RE.search(note)
        log_m = self._RELAUNCH_LOG_ABS_RE.search(note) or self._RELAUNCH_LOG_LEGACY_RE.search(note)
        if not pid_m or not log_m:
            return None
        pod_m = self._RELAUNCH_POD_RE.search(note)
        pod_matches = bool(pod_m) and pod_m.group(1) == handle.pod_name
        if pod_m and not pod_matches:
            return None  # marker targets a different host (e.g. a RunPod pod)
        try:
            cluster_ev = self._read_marker(issue, "epm:cluster-launched")
        except Exception as exc:
            logger.warning(
                "GCP poll: epm:cluster-launched read failed for issue=%d (%s); "
                "keeping startup-script terminal state.",
                issue,
                exc,
            )
            return None
        cluster_ts = _parse_event_ts((cluster_ev or {}).get("ts"))
        marker_ts = _parse_event_ts(ev.get("ts"))
        if cluster_ts is not None:
            if marker_ts is None or marker_ts <= cluster_ts:
                return None  # predates the current instance generation
        elif not pod_matches:
            return None  # no generation baseline AND no instance-name link
        return int(pid_m.group(1)), log_m.group(1)

    def _probe_relaunched_workload(
        self, handle: RunHandle, zone: str, *, pid: int, log_path: str
    ) -> PollResult:
        """Probe the relaunched workload's pid + log over ssh sudo.

        ONE ``gcloud compute ssh`` round-trip (``sudo -n`` — the workload
        tree is root-owned, #608) checks ``kill -0 <pid>``, stats the
        relaunch log's mtime, and tails it. Classification mirrors
        ``poll_pipeline.poll_once``'s pid-corroborated semantics:

        * pid alive → ``running`` (the relaunch is the live workload).
        * pid dead + the log's latest real phase line is ``done`` (via
          ``poll_pipeline.latest_phase`` — inherits the #545/#597
          quoted-token noise guards) → ``done``.
        * pid dead otherwise → ``dead`` (exited without a clean done).
        * probe transport failure → typed ``stalled`` tick (the
          "couldn't ask" ≠ "not running" discipline, #535) — never read
          a probe failure as a terminal verdict.
        """
        quoted_log = shlex.quote(log_path)
        script = (
            f"if kill -0 {int(pid)} 2>/dev/null; "
            f"then echo EPS_RELAUNCH_PID=alive; else echo EPS_RELAUNCH_PID=dead; fi; "
            f"echo EPS_RELAUNCH_MTIME=$(stat -c %Y {quoted_log} 2>/dev/null || echo -1); "
            f"echo EPS_RELAUNCH_NOW=$(date +%s); "
            f"echo {self._RELAUNCH_TAIL_START}; "
            f"tail -n 30 {quoted_log} 2>/dev/null || true; "
            f"echo {self._RELAUNCH_TAIL_END}"
        )
        argv = _base_gcloud_argv(
            self._config,
            "compute",
            "ssh",
            handle.pod_name,
            f"--command=sudo -n bash -c {shlex.quote(script)}",
        )
        argv += [f"--zone={zone}"]
        res = self._run(argv)
        if res.returncode != 0:
            logger.warning(
                "GCP poll: relaunch probe failed for %s pid=%d (rc=%d): %s",
                handle.pod_name,
                pid,
                res.returncode,
                (res.stderr or "")[:300],
            )
            return _coarse_poll(status="stalled", current_phase="relaunch_probe_failed")
        stdout = res.stdout or ""
        alive = "EPS_RELAUNCH_PID=alive" in stdout
        mtime_ago = 10**9
        mtime_m = re.search(r"EPS_RELAUNCH_MTIME=(-?\d+)", stdout)
        now_m = re.search(r"EPS_RELAUNCH_NOW=(\d+)", stdout)
        if mtime_m and now_m and int(mtime_m.group(1)) >= 0:
            mtime_ago = max(0, int(now_m.group(1)) - int(mtime_m.group(1)))
        _, _, tail_part = stdout.partition(self._RELAUNCH_TAIL_START)
        tail_full = tail_part.split(self._RELAUNCH_TAIL_END)[0].strip() if tail_part else ""
        # Excerpt keeps the LAST 2000 chars (unlike the drain's head-cut):
        # the terminal ``[phase=done]`` line lives at the END of the tail,
        # and the done-corroboration below scans the UNtruncated text so a
        # long tail can never push the terminal line out of the parse.
        tail = tail_full[-2000:]
        if alive:
            return PollResult(
                status="running",
                current_phase="relaunched_workload",
                new_milestone=False,
                last_log_mtime_sec_ago=mtime_ago,
                pid_alive=True,
                log_tail_excerpt=tail,
            )
        # pid dead: corroborate done from the relaunch log's phase lines,
        # reusing poll_pipeline's parser (same lazy-import pattern as
        # ``_drain_sentinels`` — production entrypoints put the repo root
        # on sys.path; fall back to a __file__-derived insert).
        try:
            from scripts.poll_pipeline import latest_phase
        except ModuleNotFoundError:
            import sys

            repo_root = str(Path(__file__).resolve().parents[3])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from scripts.poll_pipeline import latest_phase
        if latest_phase(tail_full) == "done":
            return PollResult(
                status="done",
                current_phase="relaunched_workload_done",
                new_milestone=True,
                last_log_mtime_sec_ago=mtime_ago,
                pid_alive=False,
                log_tail_excerpt=tail,
            )
        return PollResult(
            status="dead",
            current_phase="relaunched_workload_exited",
            new_milestone=True,
            last_log_mtime_sec_ago=mtime_ago,
            pid_alive=False,
            log_tail_excerpt=tail,
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        """Best-effort serial-port-1 pull.

        The startup-script writes its progress to the VM's serial-port
        console; ``gcloud compute instances get-serial-port-output``
        pulls the rolling buffer. Returns ``""`` on any failure so the
        orchestrator's "report logs" message degrades gracefully.
        """
        config = self._config
        zone = handle.extra.get("zone") or config.primary_zone
        argv = _base_gcloud_argv(
            config, "compute", "instances", "get-serial-port-output", handle.pod_name
        )
        argv += [f"--zone={zone}", "--port=1"]
        result = self._run(argv)
        if result.returncode != 0:
            logger.warning(
                "GCP fetch_logs: serial-port-1 returned %d for %s; returning empty.",
                result.returncode,
                handle.pod_name,
            )
            return ""
        return result.stdout or ""

    # ----- teardown --------------------------------------------------------

    def fetch_results(self, handle: RunHandle) -> None:
        """Pull the completion sentinel (+ best-effort artifact dirs) back from the VM.

        Slice 6: gates ``confirm_artifacts`` for every GCP lane. The
        sentinel lives on the VM (the startup-script's clean-exit
        ``cat > $EPS_SENTINEL_PATH`` write); the slice-2 verifier reads
        the LOCAL filesystem and would FAIL every real run without this
        pull. Mirrors the SLURM ``rsync_pull`` shape (separate calls per
        target so a single failure doesn't bury the others).

        Two tiers:

        * **MANDATORY: sentinel** — pulled via ``gcloud compute ssh ...
          --command 'sudo -n cat <sentinel>'``, NOT scp. The GCE
          startup-script runs as root, so the whole
          ``/workspace/eps-issue-<N>`` tree is root-owned and the
          OS-Login scp user cannot traverse/read it — a plain scp fails
          with ``Permission denied`` on every real run (live finding,
          issue #588 att-20260611-064703). ``sudo -n`` works because the
          OS-Login user is in ``google-sudoers``; the captured stdout is
          written to the same local path. If the pull fails we LOG
          loudly and continue (``confirm_artifacts`` will FAIL on the
          missing file, which is the right surfacing — a workload that
          didn't write its sentinel is precisely the silent-loss hole
          the verifier catches).
        * **Best-effort: eval_results/ + figures/.** Both are authoritatively
          uploaded by the workload during the run (HF Hub / WandB / git);
          the local mirror is convenience for analyzer-local figure
          regeneration. A failure here (including the same root-owned
          ``Permission denied``) logs + continues.

        Reconnect-safe: reads the recovered ``attempt_id`` off
        ``handle.extra`` (populated by ``reconnect_or_none``); the
        sentinel sub-directory is namespaced per attempt so a re-run
        after Spot preemption never overwrites an earlier attempt.
        """
        config = self._config
        zone = handle.extra.get("zone") or config.primary_zone
        issue = int(handle.extra.get("issue") or 0)
        if issue <= 0:
            logger.error(
                "GcpBackend.fetch_results: handle missing 'issue' extra; cannot pull. handle=%r",
                handle,
            )
            return
        attempt_id = str(handle.extra.get("attempt_id") or "")
        if not attempt_id:
            logger.error(
                "GcpBackend.fetch_results: handle missing 'attempt_id' extra; cannot "
                "locate sentinel. handle=%r",
                handle,
            )
            return

        # 1) MANDATORY — pull the completion sentinel back. The slice-2
        # verifier reads its expected sentinel path off
        # ``EXPECTED_ARTIFACTS_HANDLE_KEY``; we land the file at the
        # SAME absolute path the declaration claims so the verifier
        # reads from one location regardless of backend. The VM-side
        # ``EPS_SENTINEL_PATH`` is `sentinel_path_for(config, issue,
        # attempt_id)` — the same function the declaration uses — so
        # the two are guaranteed to agree. Pulled via `ssh ... sudo -n
        # cat`, NOT scp: the startup-script runs as root, so the
        # workload tree is root-owned and the OS-Login scp user gets
        # `Permission denied` (live finding, att-20260611-064703).
        sentinel_abs = sentinel_path_for(config, issue, attempt_id)
        local_sentinel = Path(sentinel_abs)
        local_sentinel.parent.mkdir(parents=True, exist_ok=True)
        ssh_sentinel = _base_gcloud_argv(
            config,
            "compute",
            "ssh",
            handle.pod_name,
            f"--command=sudo -n cat {shlex.quote(sentinel_abs)}",
        )
        ssh_sentinel += [f"--zone={zone}"]
        sentinel_res = self._run(ssh_sentinel)
        if sentinel_res.returncode != 0:
            logger.error(
                "GcpBackend.fetch_results: sentinel pull (ssh sudo -n cat) from %s "
                "failed (rc=%d); confirm_artifacts will FAIL on the missing sentinel. "
                "stderr=%s",
                handle.pod_name,
                sentinel_res.returncode,
                sentinel_res.stderr[:500],
            )
        else:
            local_sentinel.write_text(sentinel_res.stdout)
            logger.info(
                "GcpBackend.fetch_results: sentinel pull PASS for issue=%d attempt=%s (%d bytes)",
                issue,
                attempt_id,
                len(sentinel_res.stdout),
            )

        # 2) Best-effort — pull eval_results/issue_<N>/ and
        # figures/issue_<N>/ back to the local repo. These are
        # authoritative on HF / WandB / git already; the local mirror
        # is convenience. Each subdir is its own scp call so one
        # failure doesn't bury the other.
        repo_root = _default_src_root_for_fetch()
        workload_root = workload_dir_for(config, issue)
        for subdir in (f"eval_results/issue_{issue}", f"figures/issue_{issue}"):
            remote_path = f"{workload_root}/{subdir}"
            local_path = repo_root / subdir
            local_path.parent.mkdir(parents=True, exist_ok=True)
            scp_dir = _base_gcloud_argv(
                config,
                "compute",
                "scp",
                "--recurse",
                f"{handle.pod_name}:{remote_path}",
                str(local_path.parent),
            )
            scp_dir += [f"--zone={zone}"]
            dir_res = self._run(scp_dir)
            if dir_res.returncode != 0:
                logger.warning(
                    "GcpBackend.fetch_results: best-effort scp of %s failed (rc=%d); "
                    "authoritative copy is on HF/WandB/git. stderr=%s",
                    remote_path,
                    dir_res.returncode,
                    dir_res.stderr[:300],
                )

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        """Backend-agnostic artifact verification.

        Delegates to :func:`backends.artifacts.confirm_artifacts_from_handle`
        — the same mechanical gate SLURM + RunPod use. The launch path
        is responsible for populating
        :data:`~backends.artifacts.EXPECTED_ARTIFACTS_HANDLE_KEY` on
        ``handle.extra``; a missing declaration is itself a FAIL.
        """
        from explore_persona_space.backends.artifacts import confirm_artifacts_from_handle

        verdict = confirm_artifacts_from_handle(handle)
        if not verdict.passed:
            logger.warning(
                "GcpBackend.confirm_artifacts FAIL for instance %s: %s",
                handle.pod_name,
                "; ".join(verdict.reasons),
            )
        return verdict.passed

    def teardown(self, handle: RunHandle) -> None:
        """``gcloud compute instances delete --quiet``; idempotent on a missing VM.

        The ``--instance-termination-action=DELETE`` + ``--max-run-duration``
        belts mean an unattended VM auto-deletes; an orchestrator-driven
        teardown is the explicit early path. A "was not found" stderr is
        the common case (the VM already auto-deleted) and is NOT raised.
        """
        config = self._config
        zone = handle.extra.get("zone") or config.primary_zone
        argv = render_delete_argv(config=config, name=handle.pod_name, zone=zone)
        result = self._run(argv)
        if result.returncode == 0:
            return
        stderr_low = (result.stderr or "").lower()
        if "was not found" in stderr_low or "404" in stderr_low:
            logger.info(
                "GCP teardown: %s already gone (was not found); treating as success.",
                handle.pod_name,
            )
            return
        # Anything else is a real failure (auth blip, transient API
        # error). Raise so the orchestrator surfaces it rather than
        # silently leaving a VM up.
        raise GcpBackendError(
            f"gcloud delete {handle.pod_name} returned {result.returncode}: {result.stderr[:500]}"
        )

    # ----- internal helpers ------------------------------------------------

    def _with_artifacts_declaration(
        self,
        *,
        handle: RunHandle,
        spec: RunSpec,
        config: GcpConfig,
        attempt_id: str,
        wandb_run_path: str | None = None,
    ) -> RunHandle:
        """Return a copy of ``handle`` with the artifact declaration attached.

        RunHandle is frozen, so we copy ``extra`` and rebuild. The
        verifier's ``confirm_artifacts_from_handle`` will read this back
        and fail loudly if the launch path forgot to populate it.
        """
        from dataclasses import replace

        decl = expected_artifacts_declaration(
            spec=spec,
            config=config,
            attempt_id=attempt_id,
            wandb_run_path=wandb_run_path,
        )
        new_extra = dict(handle.extra)
        new_extra[EXPECTED_ARTIFACTS_HANDLE_KEY] = decl
        return replace(handle, extra=new_extra)


# ---------------------------------------------------------------------------
# Poll-result helpers
# ---------------------------------------------------------------------------


def _parse_event_ts(raw: Any) -> datetime | None:
    """Parse an events.jsonl ``ts`` (UTC ISO-8601, ``Z`` suffix) or ``None``.

    ``task_workflow._utcnow_iso`` writes ``YYYY-MM-DDTHH:MM:SSZ``;
    normalize the ``Z`` for ``fromisoformat`` and fail soft on anything
    malformed (the caller treats ``None`` as "no usable timestamp").
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _coarse_poll(*, status: str, current_phase: str) -> PollResult:
    """Build a PollResult with the minimal fields populated."""
    return PollResult(
        status=status,
        current_phase=current_phase,
        new_milestone=False,
        last_log_mtime_sec_ago=10**9,
        pid_alive=status == "running",
        log_tail_excerpt="",
    )


def _terminal_dead_poll(*, reason: str) -> PollResult:
    """The instance is gone → terminal dead."""
    return PollResult(
        status="dead",
        current_phase=f"terminal_{reason}",
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
    )


def _overlay_drain(
    base: PollResult,
    *,
    processed: int,
    gate: str | None,
    alarm: str,
    log_tail: str,
) -> PollResult:
    """Thread sentinel-drain results into a coarse :class:`PollResult`.

    Gate precedence mirrors ``poll_pipeline.poll_once``: a drained gate
    sentinel wins over every other status — the orchestrator must park at
    the user gate before advancing. ``log_tail_excerpt`` carries (in
    priority order) the drain ALARM (a transport / permission failure must
    surface loudly, never as a silent ``sentinels_processed=0`` — incident
    #608), else whatever the base carried, else the sudo-read workload log
    tail.
    """
    from dataclasses import replace

    merged_gate = base.gate or gate
    return replace(
        base,
        status="gate" if merged_gate else base.status,
        gate=merged_gate,
        sentinels_processed=processed,
        log_tail_excerpt=alarm or base.log_tail_excerpt or log_tail,
    )


def _gcp_status_to_poll_result(status: str) -> PollResult:
    """Map a GCE ``status`` to our coarse :class:`PollResult` shape.

    See https://cloud.google.com/compute/docs/instances/instance-life-cycle
    for the GCE status enum. We map:

    * ``RUNNING`` → ``running`` (pid_alive=True)
    * ``PROVISIONING`` / ``STAGING`` → ``running`` (VM is coming up; the
      orchestrator's bg loop will keep polling)
    * ``STOPPING`` / ``REPAIRING`` → ``stalled`` (transient; bg loop retries)
    * ``TERMINATED`` / ``STOPPED`` / ``SUSPENDED`` → ``dead``
    """
    up = status.upper()
    if up == "RUNNING":
        return _coarse_poll(status="running", current_phase="running")
    if up in {"PROVISIONING", "STAGING"}:
        return _coarse_poll(status="running", current_phase=up.lower())
    if up in {"STOPPING", "REPAIRING"}:
        return _coarse_poll(status="stalled", current_phase=up.lower())
    if up in {"TERMINATED", "STOPPED", "SUSPENDED"}:
        return _terminal_dead_poll(reason=up.lower())
    return _coarse_poll(status="stalled", current_phase=f"unknown_{up.lower()}")


# ---------------------------------------------------------------------------
# Instance-id parsing
# ---------------------------------------------------------------------------


def _parse_instance_id(stdout: str, expected_name: str) -> str:
    """Best-effort instance-id pull from ``gcloud ... create --format=json`` stdout.

    Returns the numeric id as a string, or "" when the JSON is malformed
    (an empty string is the truthful "we did not capture" marker; the
    instance_name field is the authoritative identity throughout the
    backend, the id is only logged into the marker body).
    """
    if not stdout.strip():
        return ""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return ""
    # gcloud returns either a list (the common form) or a dict; handle both.
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("name") == expected_name:
                return str(item.get("id") or "")
        return ""
    if isinstance(payload, dict):
        return str(payload.get("id") or "")
    return ""


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_FALLBACK_ZONES",
    "DEFAULT_GCLOUD_CONFIG",
    "DEFAULT_IMAGE_FAMILY",
    "DEFAULT_IMAGE_PROJECT",
    "DEFAULT_PRIMARY_ZONE",
    "DEFAULT_PROJECT",
    "DEFAULT_PROVISIONING_MODEL",
    "DEFAULT_REPO_URL",
    "INTENT_TO_MACHINE",
    "STARTUP_PASSTHROUGH_ENV_KEYS",
    "STARTUP_SECRET_ENV_KEYS",
    "GcloudRunResult",
    "GcloudRunner",
    "GcpBackend",
    "GcpBackendError",
    "GcpConfig",
    "GcpProvisioningError",
    "GcpWorkloadError",
    "MachineSpec",
    "QuotaHeadroom",
    "attempt_id_for",
    "audit_stale_gcp_vms",
    "classify_create_failure",
    "default_gcloud_runner",
    "default_gcp_config",
    "expected_artifacts_declaration",
    "instance_name_for",
    "machine_for_intent",
    "preflight_quota_headroom",
    "reconnect_or_none",
    "region_for_zone",
    "render_create_argv",
    "render_delete_argv",
    "render_describe_argv",
    "render_list_argv",
    "render_region_describe_argv",
    "render_startup_script",
    "resolve_provisioning_model",
    "sentinel_path_for",
    "workload_dir_for",
]
