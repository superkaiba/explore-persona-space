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
  ``lora`` → ``a2-ultragpu-1g`` (1x A100-80); ``capture-7b`` →
  ``a2-ultragpu-1g`` (1x A100-80, the activation-capture eval path, #752);
  ``ft-7b`` → ``a2-ultragpu-4g`` (4x A100-80); ``eval`` → ``g2-standard-4``
  (1x L4); ``debug`` → ``g2-standard-4``.
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
  exists, return a handle for it without re-provisioning; refuses a RUNNING
  instance whose ``eps/phase`` is already terminal (``done``/``failed``/
  ``finalize_failed_artifacts_ok``/``wedged``) — the #908 gate-park zombie —
  which the pre-launch stale reclaim then deletes so the create does not
  collide (#632).
* :func:`audit_stale_gcp_vms` — analogue of ``scripts/pod.py audit-stale``;
  reaps ``eps-issue-*`` instances on TWO bounded predicates: a per-instance-
  fence-aware age backstop (#741 — reaped once the VM exceeds its OWN
  ``--max-run-duration`` + a 1h grace, or a fixed fallback fence when that
  field is unreadable) AND a prompt terminal-phase reap (a RUNNING VM that
  published ``eps/phase=done``/``failed`` past a short floor is a wedged
  zombie idle-billing — reaped well under the age fence; incident #634
  family). Cron wiring is the orchestrator's responsibility — this exposes
  the callable that the cron / a ``scripts/`` entry can invoke.
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

import base64
import contextlib
import io
import json
import logging
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
    recommend_lane_next_interval,
    validate_lane_suffix,
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
      ``7d`` (#741) — the FLEX_START ceiling (the longest the GCP-first
      auto lane's long-job branch can run; ``_assert_max_run_within_flex_cap``
      rejects ``>7d``), so a long multi-cell sweep is no longer stranded
      mid-run by a self-imposed 24h fence (#697 lost 24/64 cells to the old
      24h default). The credit-leak backstop is now the per-instance-fence-
      aware janitor age reap (``gcp_audit.py`` + :func:`_janitor_stale_reason`,
      reaping a VM only once it exceeds its OWN ``--max-run-duration`` + a 1h
      grace), NOT this fence. Tunable per spec via ``RunSpec.time_budget_hours``
      / ``spec.extra['max_run_duration']`` (the renderer converts to gcloud
      duration form).
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
    default_max_run_duration: str = "7d"
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
#: re-request. Each GCP rung walks ``[primary_zone, *fallback_zones]``
#: filtered by :data:`MACHINE_TYPE_ZONE_AVAILABILITY` before the rung is
#: counted a capacity miss, so adding a zone here widens the per-rung
#: fan-out for every machine type the RESTRICT map says is offered there.
#: ``us-central1-f`` is included for the ``a2-highgpu-1g`` (A100-40, the
#: #656 fallback rung) which is offered in ``{a, b, c, f}``; the RESTRICT
#: filter strips ``-f`` back out for the ``a2-ultragpu-*`` (A100-80) and
#: ``a3-highgpu-*`` (H100) families, which GCP does NOT offer in ``-f``
#: (verified via ``gcloud compute machine-types list
#: --configuration=eps-gcp``, 2026-06-30 re-verification, #774). So a
#: broader DEFAULT never leaks a doomed zone to a restricted family.
DEFAULT_FALLBACK_ZONES: tuple[str, ...] = (
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
)

#: DLVM image family verified working on 2026-06-08 ($1 credit-draw test
#: provisioned ``a2-ultragpu-1g`` Spot with this image and ran nvidia-smi).
DEFAULT_IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"

#: DLVM project for the image family above.
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"

#: Minimum boot-disk size accepted by a create against the pinned DLVM
#: image — GCP rejects any smaller disk ("Requested disk size cannot be
#: smaller than the image size (100 GB)"; incident #1336: --boot-disk-gb 60
#: failed the rung and exhausted the auto chain). render_create_argv clamps
#: UP to this floor (never down — a plan footprint is a minimum requirement,
#: so a bigger disk always satisfies it), mirroring the RunPod floors
#: (runpod.py _CPU_CONTAINER_DISK_FLOOR_GB / _GPU_VOLUME_FLOOR_GB). This is
#: a property of the IMAGE, not the config — if DEFAULT_IMAGE_FAMILY is ever
#: re-pinned, re-verify (family is a POSITIONAL arg, not --family=):
#:   gcloud --configuration=eps-gcp compute images describe-from-family
#:     <FAMILY> --project=<PROJECT> --format='value(diskSizeGb)'
_GCP_IMAGE_MIN_BOOT_DISK_GB = 100

#: Canonical public HTTPS clone URL. The repo is open, so the CLONE is
#: tokenless; PUSH auth comes from the #1205 env-reading credential
#: helper the workload_cmd branch configures when GITHUB_TOKEN was
#: delivered via instance metadata (see the credential block in
#: ``render_startup_script`` — the token is never at rest in
#: ``.git/config`` and never in a remote URL a crash-persisted log
#: could leak).
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
#: the SLURM ``_DEFAULT_GPUS_FOR_INTENT`` aliasing). ``capture-7b`` (#752)
#: is the A100-80 activation-capture EVAL path — a forward-pass-only 7B run
#: that captures hidden states, sized to A100-80 like ``lora-7b`` but kept
#: DISTINCT from ``eval``'s L4 default so the router never under-provisions a
#: hidden-state-capturing forward (no coupling of the router to workload
#: semantics). ``debug`` reuses the L4 machine — the smallest GPU available
#: is the right "debug pod" analogue. ``inf-70b`` / ``ft-70b`` are NOT in
#: this table (the GFS credit pool is for the A100-80 / L4 line; 70B
#: inference belongs on RunPod's H200 in v1).
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
    # Activation-capture intent (#752) — a 7B forward that captures hidden
    # states (all-layer residual streams, Welford activation accumulation,
    # per-token activation dumps) must clear A100-80 HBM: 7B bf16 weights are
    # ~14 GB and the captured activations push past the L4's 16-GB-class HBM
    # (the binding fact is the ordering L4 << A100-40 << A100-80, not the
    # exact L4 figure). Same machine as
    # lora-7b (a2-ultragpu-1g, 1x A100-80) but a DISTINCT intent so a caller /
    # the planner can declare "I capture activations" and the router sizes for
    # it instead of falling to the g2-standard-4 (L4) eval default. #666 (L4
    # OOM) and #744 (g2-standard-4 OOM) were both 7B activation-capture
    # forwards routed L4 under the eval default.
    "capture-7b": MachineSpec(
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
    # H100 intents (#631) — a3-highgpu cannot be created on-demand
    # (Spot / flex-start only), so a caller MUST pass
    # spec.extra["provisioning_model"] = "SPOT" | "FLEX_START" for these
    # (render_create_argv raises loud on H100 + STANDARD). lora-7b-h100 =
    # the 1x H100 lora-scale path; eval-h100 = the 2x H100 (TP=2) path.
    "lora-7b-h100": MachineSpec(
        machine_type="a3-highgpu-1g",
        gpu_count=1,
        gpu_kind="H100-80",
    ),
    "eval-h100": MachineSpec(
        machine_type="a3-highgpu-2g",
        gpu_count=2,
        gpu_kind="H100-80",
    ),
    # CPU-only analysis intent (#677) — gpu_count=0. The size-gated
    # CPU-off-pod rule (CLAUDE.md "CPU-only phases don't hold GPU pods"
    # data-footprint carve-out) routes a CPU/analysis phase whose local
    # footprint exceeds VM_ANALYSIS_FOOTPRINT_GB_MAX=50 GB HERE instead of
    # the shared VM. n2-highmem-16 = 16 vCPU / 128 GB RAM — enough RAM
    # headroom to hold a ~100-150 GB working set's hot slice in memory while
    # the bulk lives on the 300 GB boot disk; n2 is a standard
    # live-migratable family (maintenance-policy MIGRATE). Larger footprints
    # size the boot disk up via --boot-disk-gb (no new flag) or pass a
    # machine_spec_override for a bigger n2-highmem-{32,...}.
    "cpu-bigmem": MachineSpec(
        machine_type="n2-highmem-16",
        gpu_count=0,
        gpu_kind="CPU",
    ),
    # Cheap CPU-only intents (#747) — gpu_count=0, spot-eligible on a short
    # job (the router's CPU spot rung, unlike cpu-bigmem above which is
    # reliable on-demand only). The standing CPU-routing policy prefers a
    # dedicated cheap CPU pod over the shared VM for non-trivial CPU phases
    # (parallelizable / longer than the trivial floor). e2-standard-2 = 2 vCPU
    # / 8 GB (~$0.035/hr spot us-central1) — the parallel-fan-out workhorse;
    # e2-standard-8 = 8 vCPU / 32 GB — a mid CPU job that fits between
    # cpu-small's fan-out and cpu-bigmem's >50 GB analyses. NEITHER is for
    # >50 GB footprints — that stays cpu-bigmem (n2-highmem-16, above). These
    # ALSO have a RunPod CPU fallback lane (router.RUNPOD_CPU_INSTANCE_FOR_INTENT),
    # so unlike cpu-bigmem they fall over GCP->RunPod when GCP is exhausted.
    "cpu-small": MachineSpec(machine_type="e2-standard-2", gpu_count=0, gpu_kind="CPU"),
    "cpu-mid": MachineSpec(machine_type="e2-standard-8", gpu_count=0, gpu_kind="CPU"),
    # 8-GPU sweep intents (#743) — the FREE GCP credit lane at 8x width for
    # wide embarrassingly-parallel sweeps (driving case #697's 64-cell patch
    # sweep, ~2x faster on 8 GPUs than 4). The A100 half of #743's
    # explicit-only exclusion is SUPERSEDED by #1121: the router auto ladder
    # now walks WIDE_A100_80_BY_WIDTH rungs (a2-ultragpu-{8,4,2}g) for a
    # width-declaring dispatch (``--gpus N`` on a width-eligible intent), so
    # sweep-8g-a100 remains only as a back-compat explicit intent (redundant
    # with ``--intent capture-7b --gpus 8`` on the auto lane) — and, as of
    # #1379, an EXPLICIT sweep-8g-a100 dispatch likewise WIDTH-DEGRADES on
    # capacity miss (EXPLICIT_WIDE_DEGRADE_INTENTS below): the router ladder
    # appends 4g/2g rungs after the 8g base rungs unless the caller pins the
    # width via --width-required (spec.extra["width_required"]). sweep-8g-h100
    # is EXCLUDED from that degradation too (an explicit H100 pick is a
    # GPU-TYPE choice; its fallback is the type-preserving RunPod 8xH100
    # terminal rung). The H100 half
    # STANDS: 8x H100 preemptible quota in us-central1 is exactly 8 (one 8x,
    # zero concurrency headroom), there is no H100 on-demand pool, and the
    # H100 quota metrics are absent from ``regions describe`` on this project
    # (the fail-open headroom pre-check is blind for H100) — so sweep-8g-h100
    # stays EXPLICIT --intent ONLY, never auto-scheduled. sweep-8g-h100 is
    # a3-highgpu (H100): SPOT / FLEX_START only (render_create_argv raises on
    # H100 + STANDARD via gpu_kind, inherited).
    "sweep-8g-a100": MachineSpec(
        machine_type="a2-ultragpu-8g",
        gpu_count=8,
        gpu_kind="A100-80",
    ),
    "sweep-8g-h100": MachineSpec(
        machine_type="a3-highgpu-8g",
        gpu_count=8,
        gpu_kind="H100-80",
    ),
}


#: Intents whose A100-80 workload ALSO fits in a single 40 GB A100 — the
#: cheaper-but-smaller GCP rung the router's fallback ladder (#656) tries
#: when the 80 GB pool is exhausted. SINGLE-GPU 7B-scale work only: a 7B
#: LoRA fine-tune / eval / generation fits comfortably in 40 GB. Multi-GPU
#: full-FT (``ft-7b`` = 4xA100-80) and the 70B intents do NOT fit, so they
#: are absent from this map and the ladder skips the A100-40 rung for them.
#: ``a2-highgpu-1g`` is the 1xA100-40GB machine type (sibling of the
#: ``a2-ultragpu-1g`` 1xA100-80GB row above). ``capture-7b`` (#752, the
#: activation-capture EVAL path) is a single-GPU 7B forward that fits 40 GB
#: (7B bf16 weights ~14 GB + room for a moderate-batch all-layer hidden-state
#: capture), so it is a valid A100-40 fallback — its A100-80 primary is the
#: canonical fit, A100-40 the cheaper-but-smaller rung. ``eval`` / ``debug``
#: default to L4 on-demand, NOT A100-80, so the A100-40 rung is only a
#: meaningful STEP UP for them when L4 is constrained; they are included to
#: keep the "single-GPU 7B fits 40 GB" rule uniform and the inclusion is
#: harmless (their L4 on-demand rarely exhausts). Decision recorded in
#: plan §11.
INTENT_A100_40_FALLBACK: dict[str, MachineSpec] = {
    "lora-7b": MachineSpec(machine_type="a2-highgpu-1g", gpu_count=1, gpu_kind="A100-40"),
    "lora": MachineSpec(machine_type="a2-highgpu-1g", gpu_count=1, gpu_kind="A100-40"),
    "capture-7b": MachineSpec(machine_type="a2-highgpu-1g", gpu_count=1, gpu_kind="A100-40"),
    "eval": MachineSpec(machine_type="a2-highgpu-1g", gpu_count=1, gpu_kind="A100-40"),
    "debug": MachineSpec(machine_type="a2-highgpu-1g", gpu_count=1, gpu_kind="A100-40"),
}


def a100_40_fallback_for_intent(spec: RunSpec) -> MachineSpec | None:
    """Return the A100-40 :class:`MachineSpec` for ``spec.intent`` if the
    workload fits in 40 GB, else ``None``.

    A ``None`` return means "A100-40 is NOT a valid fallback for this
    intent" — the router's GCP ladder skips the A100-40 rung and proceeds
    to SPOT / the next lane. ``ft-7b`` (4xA100-80, ZeRO-3) and the 70B
    intents (no GCP mapping at all) return ``None`` because a 40 GB card
    cannot hold them; an unknown intent also returns ``None`` (the
    ladder simply has no A100-40 rung to add — the on-demand A100-80 rung
    still fails loud on the unknown intent via :func:`machine_for_intent`).
    """
    return INTENT_A100_40_FALLBACK.get(spec.intent)


#: Width -> wide A100-80 machine for the width-aware auto ladder (#1121).
#: a2-ultragpu-{2,4,8}g all live-verified offered in us-central1-{a,c}
#: (gcloud machine-types list, 2026-07-08). H100 (a3-highgpu-*) is
#: DELIBERATELY absent: preemptible quota is exactly 8 (one 8x, zero
#: concurrency headroom, #743), there is no on-demand pool, and the H100
#: quota metrics are absent from ``regions describe`` on this project, so
#: the fail-open headroom pre-check cannot protect a doomed create.
#: sweep-8g-h100 stays explicit-``--intent``-only.
WIDE_A100_80_BY_WIDTH: dict[int, MachineSpec] = {
    2: MachineSpec(machine_type="a2-ultragpu-2g", gpu_count=2, gpu_kind="A100-80"),
    4: MachineSpec(machine_type="a2-ultragpu-4g", gpu_count=4, gpu_kind="A100-80"),
    8: MachineSpec(machine_type="a2-ultragpu-8g", gpu_count=8, gpu_kind="A100-80"),
}

#: Intents whose workload shards across GPUs at 1 GPU (or, for ft-7b, its
#: ZeRO-3 world size) per shard, so a wide A100-80 machine serves them:
#: the single-GPU 7B-class intents + ft-7b (whose base IS the same
#: a2-ultragpu family at 4x). eval/debug are included even though their
#: BASE machine is L4 — a width>1 request upgrades them onto the A100-80
#: family (there is no multi-GPU g2 mapping in v1; deliberate, see plan
#: #1121 §11). H100-family intents (lora-7b-h100 / eval-h100 /
#: sweep-8g-*) are EXCLUDED (see WIDE_A100_80_BY_WIDTH note).
WIDTH_ELIGIBLE_INTENTS: frozenset[str] = frozenset(
    {"lora-7b", "lora", "capture-7b", "eval", "debug", "ft-7b"}
)

#: Explicit wide intents whose GPU WIDTH auto-degrades on capacity failure
#: (#1379): when every rung at the intent's full width capacity-misses at
#: CREATE time, the router ladder appends WIDE_A100_80_BY_WIDTH rungs at the
#: narrower widths (8->4->2) before falling out of the GCP lane. Opt out per
#: dispatch with --width-required (spec.extra["width_required"]).
#: sweep-8g-h100 is DELIBERATELY absent: an explicit H100 pick is a GPU-TYPE
#: choice (cross-type A100 degradation would silently change silicon), the
#: H100-never-in-a-degradation-walk invariant stands (#743/#1121), and the
#: type-preserving fallback is the RunPod 8xH100 terminal rung
#: (router.RUNPOD_INTENT_FOR_GCP_INTENT identity row).
EXPLICIT_WIDE_DEGRADE_INTENTS: frozenset[str] = frozenset({"sweep-8g-a100"})


def wide_a100_80_for_width(width: int) -> MachineSpec | None:
    """The wide A100-80 MachineSpec for ``width``; ``None`` when unsupported."""
    return WIDE_A100_80_BY_WIDTH.get(int(width))


def machine_for_intent(spec: RunSpec) -> MachineSpec:
    """Resolve ``spec.intent`` to a :class:`MachineSpec`.

    Consults ``spec.extra["machine_spec_override"]`` FIRST: the router's
    GCP fallback ladder (#656) threads an A100-40 (or SPOT-A100-80, etc.)
    :class:`MachineSpec` here so every downstream chokepoint
    (:func:`render_create_argv`, :func:`quota_metric_for`, the zone
    filter) resolves the rung's TRUE machine without mutating the frozen
    :class:`RunSpec`'s semantic ``intent``. The override may be a
    :class:`MachineSpec` or a plain ``{"machine_type", "gpu_count",
    "gpu_kind"}`` dict (the JSON-safe sidecar-serializable shape the
    router threads). Absent override → the normal intent map.

    Fails LOUD on an unknown intent (with no override) rather than
    silently picking a default — a typo should crash the launch, NOT spin
    up the wrong instance type and burn credit on it. Consistent with the
    SLURM backend's :func:`~slurm.stages_for_spec` /
    :func:`~slurm.time_budget_hours` fail-fast policy.
    """
    override = (spec.extra or {}).get("machine_spec_override")
    if override is not None:
        if isinstance(override, MachineSpec):
            return override
        if isinstance(override, Mapping):
            return MachineSpec(
                machine_type=override["machine_type"],
                gpu_count=int(override["gpu_count"]),
                gpu_kind=override["gpu_kind"],
            )
        raise ValueError(
            "spec.extra['machine_spec_override'] must be a MachineSpec or a "
            f"{{machine_type, gpu_count, gpu_kind}} dict, got {type(override).__name__!r}."
        )
    if spec.intent not in INTENT_TO_MACHINE:
        raise ValueError(
            f"no GCP machine-type for intent {spec.intent!r}. "
            f"Supported intents: {sorted(INTENT_TO_MACHINE)}. "
            "Add a MachineSpec row to backends/gcp.INTENT_TO_MACHINE "
            "or pick a different backend (RunPod covers H200 / 70B paths)."
        )
    return INTENT_TO_MACHINE[spec.intent]


#: Per-machine-type zone availability within ``us-central1`` (#653). The
#: uniform :data:`DEFAULT_FALLBACK_ZONES` is NOT valid for every machine
#: type: the A2-ultragpu family (``a2-ultragpu-1g`` / ``a2-ultragpu-4g``,
#: i.e. the ``lora`` / ``lora-7b`` / ``ft-7b`` intents) is NOT offered in
#: ``us-central1-b`` — only ``-a`` and ``-c``. Without this filter the
#: zone-fallback ladder tries ``us-central1-b`` on a capacity miss, the
#: gcloud create 400s with "Invalid value for field ... machine type ...
#: not found" (a CONFIG error, NOT a capacity miss), and the doomed
#: attempt burns the per-day GCP attempt counter (#653 round-8: a ft-7b
#: auto launch hit ZONE_RESOURCE_POOL_EXHAUSTED on ``-a``, fell to ``-b``
#: where ``a2-ultragpu-4g`` does not exist, and the config error aborted
#: the GCP lane). A machine type ABSENT from this map is assumed available
#: in every configured zone (no filtering) — the map only RESTRICTS, so an
#: unlisted future machine type fails open rather than being silently
#: dropped from every zone.
#:
#: Verify / refresh per machine type with:
#:   gcloud compute machine-types list \
#:     --filter="name=<machine-type> AND zone~us-central1" \
#:     --configuration=eps-gcp --format="value(zone)"
#: (run 2026-06-16 for the rows below).
MACHINE_TYPE_ZONE_AVAILABILITY: dict[str, frozenset[str]] = {
    # A2-ultragpu (A100-80) — us-central1-b does NOT offer this family
    # (re-verified 2026-06-30: not offered in -b OR -f, #774).
    "a2-ultragpu-1g": frozenset({"us-central1-a", "us-central1-c"}),
    # a2-ultragpu-2g (2x A100-80, the #1121 width-2 auto-ladder rung) — same
    # A2-ultragpu family restriction as its 1g/4g/8g siblings (NOT in -b or
    # -f). Live-verified 2026-07-08 (gcloud compute machine-types list).
    "a2-ultragpu-2g": frozenset({"us-central1-a", "us-central1-c"}),
    "a2-ultragpu-4g": frozenset({"us-central1-a", "us-central1-c"}),
    # a2-ultragpu-8g (8x A100-80, #743) — same A2-ultragpu family as the 1g/4g
    # rows above; us-central1-b does NOT offer the family. Verified 2026-06-29
    # (re-verified 2026-06-30: not offered in -b OR -f, #774).
    "a2-ultragpu-8g": frozenset({"us-central1-a", "us-central1-c"}),
    # g2 (L4) + a3-highgpu (H100, BOTH the 1g lora-7b-h100 AND the 2g
    # eval-h100 sizes) ARE offered in all three us-central1 zones — listed
    # for completeness so a future zone change is verified against this
    # table, not blind, and so neither a3-highgpu size is left implicitly
    # "assumed available" while its sibling is explicit (#653 round-8 follow-up:
    # eval-h100 / a3-highgpu-2g was the implicitly-absent size that prompted a
    # false "not offered in us-central1" report; gcloud machine-types list for
    # both a3-highgpu-1g AND a3-highgpu-2g on 2026-06-16 returns
    # us-central1-{a,b,c} for each — fail-open was already correct here, this
    # row just makes the verified fact explicit).
    "g2-standard-4": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
    "a3-highgpu-1g": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
    "a3-highgpu-2g": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
    # a3-highgpu-8g (8x H100-80, #743) — same a3-highgpu family as 1g/2g; offered
    # in all three us-central1 zones. Verified 2026-06-29.
    "a3-highgpu-8g": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
    # A2-highgpu (A100-40) — the #656 cheaper-but-smaller fallback rung.
    # Verified offered in us-central1-{a,b,c,f} (gcloud compute machine-types
    # list --filter="name=a2-highgpu-1g AND zone~us-central1"; 2026-06-30
    # re-verification, #774). -f is now REACHED because DEFAULT_FALLBACK_ZONES
    # carries it (the ladder [primary, *fallback_zones] resolves to {a,b,c,f}
    # for this rung), giving the A100-40 fallback rung one more zone to walk
    # than the A2-ultragpu family (restricted to -a/-c). The map RESTRICTS
    # only, so the broader DEFAULT does not leak -f to the restricted A100-80 /
    # H100 families below.
    "a2-highgpu-1g": frozenset(
        {"us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"}
    ),
    # n2-highmem (CPU-only, #677) — n2 is offered in all three us-central1
    # zones. Listed for completeness so a future zone change is verified
    # against this table, not blind (same convention as the g2 / a3 rows
    # above; the map only RESTRICTS, so a fail-open default was already
    # correct — this row just makes the verified fact explicit).
    "n2-highmem-16": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
    # E2 (cheap CPU-only, #747) — the cpu-small / cpu-mid intents. Verified
    # offered in us-central1-{a,b,c} (gcloud compute machine-types list
    # --filter="name=e2-standard-2 AND zone~us-central1", 2026-06-29; also
    # offered in -f, listed here for the documented us-central1-{a,b,c} set
    # the map RESTRICTS to — the map only restricts, so omitting -f is safe).
    "e2-standard-2": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
    "e2-standard-8": frozenset({"us-central1-a", "us-central1-b", "us-central1-c"}),
}


def zones_for_machine_type(machine_type: str, zones: Sequence[str]) -> list[str]:
    """Filter ``zones`` to those where ``machine_type`` is actually offered.

    Preserves ``zones`` order (the fallback ladder is priority-ordered).
    Drops only zones a machine type is KNOWN absent from per
    :data:`MACHINE_TYPE_ZONE_AVAILABILITY`; a machine type unlisted in the
    map fails OPEN (every input zone kept) so a future machine type is
    never silently filtered to nothing. This stops a guaranteed-to-fail
    create — and the GCP-attempt-counter burn it costs — on a zone where
    the machine type does not exist (#653).
    """
    available = MACHINE_TYPE_ZONE_AVAILABILITY.get(machine_type)
    if available is None:
        return list(zones)
    return [z for z in zones if z in available]


# ---------------------------------------------------------------------------
# Provisioning model + attempt-id
# ---------------------------------------------------------------------------


#: GCE provisioning models accepted by ``--provisioning-model``.
ProvisioningModel = str  # "SPOT" | "STANDARD" | "FLEX_START"

#: The provisioning models the resolver accepts. FLEX_START (DWS flex-start)
#: was added in #631 to reach the preemptible H100 + idle preemptible A100
#: quota pools without booking on-demand capacity.
_VALID_PROVISIONING_MODELS: frozenset[str] = frozenset({"SPOT", "STANDARD", "FLEX_START"})

#: Default provisioning model: STANDARD (on-demand) for the acceptance run
#: per the plan ("on-demand for acceptance; steady-state Spot once
#: idempotency is proven"). Caller switches to "SPOT" / "FLEX_START" via
#: ``spec.extra["provisioning_model"]`` once the idempotency proofs land.
DEFAULT_PROVISIONING_MODEL: ProvisioningModel = "STANDARD"

#: Default ``--request-valid-for-duration`` for a FLEX_START create — the
#: doc maximum (2h), i.e. the longest the DWS request stays valid waiting
#: for capacity. gcloud accepts 90s..2h; a caller pins a shorter window via
#: ``spec.extra["request_valid_for_duration"]``.
DEFAULT_REQUEST_VALID_FOR_DURATION: str = "2h"

#: FLEX_START ``--max-run-duration`` ceiling (docs: a standalone flex-start
#: VM may run up to seven days). Parsed from the resolved max-run-duration
#: to fail loud at render rather than mid-provision.
_FLEX_START_MAX_RUN_SECONDS: int = 7 * 24 * 3600

#: Grace added to an instance's OWN ``--max-run-duration`` before the janitor
#: age-reaps it (Option B, #741). A VM that exceeds its configured fence by more
#: than this is genuinely wedged (the ``--instance-termination-action=DELETE``
#: never fired) and is reaped. Mirrors the short ``terminal_phase_max_age_seconds``
#: precedent (10 min) — a small post-fence finalize window, not a multi-day wall.
_JANITOR_FENCE_GRACE_SECONDS: int = 3600

#: gcloud duration suffix → seconds. Bare integers are seconds.
_DURATION_SUFFIX_SECONDS: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def resolve_provisioning_model(spec: RunSpec) -> ProvisioningModel:
    """Pick the provisioning model for ``spec`` (Spot / on-demand / flex-start).

    Reads ``spec.extra["provisioning_model"]`` if present and uppercases
    it; otherwise returns :data:`DEFAULT_PROVISIONING_MODEL`. Raises on
    an unrecognized value so a typo doesn't silently downgrade an
    on-demand workload to Spot (or vice versa).
    """
    raw = spec.extra.get("provisioning_model")
    if raw is None:
        return DEFAULT_PROVISIONING_MODEL
    val = str(raw).upper()
    if val not in _VALID_PROVISIONING_MODELS:
        raise ValueError(
            f"unknown provisioning_model={raw!r}; expected one of "
            f"{sorted(_VALID_PROVISIONING_MODELS)} (case-insensitive). "
            "Set via RunSpec.extra['provisioning_model']."
        )
    return val


def resolve_request_valid_for_duration(spec: RunSpec) -> str:
    """``--request-valid-for-duration`` value for a FLEX_START create.

    gcloud duration syntax; the flex-start request window is 90s..2h.
    Defaults to :data:`DEFAULT_REQUEST_VALID_FOR_DURATION` (the doc max,
    longest queue tolerance) unless the caller pins it via
    ``spec.extra["request_valid_for_duration"]``.
    """
    return str(spec.extra.get("request_valid_for_duration") or DEFAULT_REQUEST_VALID_FOR_DURATION)


def _parse_gcloud_duration_seconds(duration: str) -> int:
    """Parse a gcloud duration string (``90s`` / ``2h`` / ``1d12h``) to seconds.

    Accepts the COMPOSED integer+unit form gcloud's ``--max-run-duration``
    parses (``1d12h``, ``1d12h30m``) the same way the CLI validator
    ``scripts/dispatch_issue.py:_MAX_RUN_DURATION_RE`` does, plus single
    groups (``24h``, ``7d``) and a bare integer (interpreted as seconds).
    Each ``(\\d+)([smhd]?)`` group is summed. Raises ``ValueError`` on an
    unparseable value (``abc``, ``5e3``, a trailing unit-less group like
    ``1d12``) so a malformed ``max_run_duration`` fails loud at render
    rather than letting gcloud reject it mid-provision.
    """
    text = str(duration).strip()
    # Guard FIRST: accept EITHER a bare integer (``3600`` → seconds) OR one
    # or more UNIT-BEARING groups (``2h``, ``1d12h``, ``1d12h30m``). A
    # unit-less group is legal only when it is the whole string (the bare
    # integer); a unit-less FINAL group in a composed form (``1d12``) is
    # rejected, as are ``5e3`` / ``abc`` / ``1d12h7`` / the empty string. A
    # lone ``re.findall`` would silently drop a non-matching tail.
    if re.fullmatch(r"\d+|(?:\d+[smhd])+", text) is None:
        raise ValueError(
            f"unparseable gcloud duration {duration!r}; expected e.g. "
            "'90s', '2h', '7d', '1d12h' (composed integer+unit groups; "
            "a bare integer is seconds)."
        )
    return sum(
        int(value) * _DURATION_SUFFIX_SECONDS.get(suffix, 1)
        for value, suffix in re.findall(r"(\d+)([smhd]?)", text)
    )


def _assert_max_run_within_flex_cap(*, max_run: str, provisioning: str) -> None:
    """Raise when a FLEX_START create's ``--max-run-duration`` exceeds 7 days.

    No-op for non-FLEX_START provisioning. Fails loud at render rather
    than letting gcloud reject the doomed create mid-provision.
    """
    if provisioning != "FLEX_START":
        return
    if _parse_gcloud_duration_seconds(max_run) > _FLEX_START_MAX_RUN_SECONDS:
        raise ValueError(
            f"FLEX_START --max-run-duration={max_run!r} exceeds the 7-day flex-start "
            "ceiling; pin spec.extra['max_run_duration'] to <= 7d."
        )


def attempt_id_for(spec: RunSpec) -> str:
    """Stable per-attempt namespace tag.

    Used as a sub-folder under HF data / model paths AND as a sentinel
    sub-directory on the VM scratch so a fresh idempotent re-run after
    Spot preemption never overwrites an earlier attempt's artifacts.
    Reads ``spec.extra["attempt_id"]`` if set (the router threads a
    FRESH per-launch id, #927); otherwise falls back to a timestamp-only
    tag (``att-YYYYMMDD-HHMMSS``). Reconnect namespace stability comes
    from the ``eps-attempt`` instance-label recovery in
    :func:`reconnect_or_none` (``launch()`` prefers the recovered label
    id over this value on reconnect), NOT from cross-launch reuse of the
    threaded id.

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


def lane_suffix_for(spec: RunSpec) -> str | None:
    """Validated per-lane suffix from ``spec.extra['lane_suffix']`` (#934), or None.

    Raises ``ValueError`` on a malformed value (fail loud, never strip)
    so a bad suffix can never silently derive a divergent instance name.
    """
    raw = spec.extra.get("lane_suffix")
    if not raw:
        return None
    return validate_lane_suffix(str(raw))


def instance_name_for(issue: int, lane_suffix: str | None = None) -> str:
    """Canonical GCE instance name for a `/issue` run.

    ``eps-issue-<N>[-<lane_suffix>]`` (#934: the optional suffix lets two
    concurrent GCP lanes for one issue coexist). The unsuffixed form
    matches the prefix the GCP stale-VM reaper greps for and mirrors
    RunPod's ``pod-<N>`` shape (issue-keyed, one-instance-per-lane); a
    suffixed name keeps the ``eps-issue-`` prefix, so the janitor's
    prefix classification still covers it. Raises ``ValueError`` when
    the composed name exceeds the 63-char RFC1035 cap (belt-and-
    suspenders behind the tighter attempt-label budget in
    ``base.validate_lane_suffix``).
    """
    name = f"eps-issue-{issue}" + (f"-{lane_suffix}" if lane_suffix else "")
    if len(name) > 63:
        raise ValueError(f"GCE instance name {name!r} exceeds the 63-char RFC1035 cap")
    return name


def workload_dir_for(config: GcpConfig, issue: int) -> str:
    """Workload root on the VM: ``<vm_scratch_dir>/issue-<N>``.

    Mirrors the RunPod ``/workspace/<repo>`` convention so the workload
    sees the same in-VM layout regardless of backend. The sentinel +
    eval_results live under here.
    """
    return f"{config.vm_scratch_dir}/eps-issue-{issue}"


def log_path_for(config: GcpConfig, issue: int) -> str:
    """Canonical workload log on the VM: ``<vm_scratch_dir>/logs/issue-<N>.log``.

    Mirrors the RunPod top-level-log convention (runpod.py ``log_path_for``)
    and deliberately lives OUTSIDE workload_dir_for() — the startup script
    redirects its output here BEFORE the repo clone, and ``git clone`` into
    ``$WORKLOAD_ROOT`` requires an empty target (#607).
    """
    return f"{config.vm_scratch_dir}/logs/issue-{issue}.log"


def sentinel_path_for(config: GcpConfig, issue: int, attempt_id: str) -> str:
    """Absolute path to the completion sentinel the workload writes.

    Folded under ``<workload>/eval_results/issue_<N>/<attempt>/`` so a
    re-run after Spot preemption (with a fresh ``attempt_id``) lands in
    a SEPARATE directory — prior attempts' sentinels (and their per-
    attempt outputs) are never overwritten.
    """
    root = workload_dir_for(config, issue)
    return f"{root}/eval_results/issue_{issue}/{attempt_id}/{SENTINEL_FILENAME}"


#: Positive-evidence deliverables sentinel filename (#1055). Distinct from
#: :data:`SENTINEL_FILENAME` (the COMPLETION sentinel the success tail writes):
#: this one is written by the WORKLOAD itself, mid-run, the moment its final
#: upload+verify step confirms every declared deliverable is on HF — so the
#: EXIT trap can tell a post-deliverables finalize/tail crash from a
#: data-losing one.
DELIVERABLES_OK_FILENAME = "deliverables_ok.json"


def deliverables_ok_path_for(config: GcpConfig, issue: int, attempt_id: str) -> str:
    """Positive-evidence sentinel path: workload-written after its final verify PASS (#1055).

    The WORKLOAD writes this file ONLY after its final upload+verify step
    confirms every declared deliverable is on HF; its presence at EXIT-trap
    time proves a non-zero exit is a finalize/tail failure, not a data-losing
    crash. Attempt-scoped (mirrors :func:`sentinel_path_for`) so a fresh
    attempt never reads a prior attempt's evidence.
    """
    root = workload_dir_for(config, issue)
    return f"{root}/eval_results/issue_{issue}/{attempt_id}/{DELIVERABLES_OK_FILENAME}"


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
    git_repo_root: str | None = None,
    skip_default_git_paths: bool = False,
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
    * Git paths — split by ``custom_workload`` (#790). A
      ``--workload-cmd`` launch declares ``eval_results/issue_<N>/`` only
      (dispatch drivers commit eval JSONs during the run; ``figures/`` is
      analyzer-generated POST-gate). A pure-hydra launch declares NEITHER
      default (``scripts/train.py`` runs with ``skip_eval=True`` and writes
      no figures, so both are false-FAILs) — only ``extra_git_paths``.
      Verified on the orchestrator side.

    The caller can add experiment-specific paths via ``extra_hf_data_paths``
    / ``extra_hf_model_paths`` / ``extra_git_paths`` (e.g. a sweep with a
    specific adapter subfolder). A run whose plan references intermediate
    analysis tensors as downstream inputs (``issue<N>_<slug>/analysis_tensors/``,
    Upload Policy #521) MUST declare that prefix via ``extra_hf_data_paths`` —
    those ``.npz`` / ``.pt`` binaries are ``.gitignore``-excluded, so they
    never land via the git paths and would otherwise slip the mechanical gate
    (incident #545, see :func:`artifacts.build_expected_artifacts_declaration`).

    Returns a serialization-friendly ``dict`` (no tuples) so the launch
    path can drop it onto ``handle.extra`` and round-trip via
    :func:`artifacts.expected_artifacts_from_handle`.

    Thin GCP-lane wrapper around
    :func:`artifacts.build_expected_artifacts_declaration` (the shared
    SLURM + RunPod builder, #598) so all three lanes compute one dict
    shape. The GCP-specific bits stay here: the GcpConfig-sourced HF repo
    ids and the per-attempt :func:`sentinel_path_for` path. All shape
    rules above (the #601 custom-workload carve-out, the hydra-lane
    ``issue<N>_<attempt>/raw_completions/`` default, the standard git
    paths) are implemented identically in the shared builder.
    """
    # Local import mirrors the slurm.py delegation entrypoint (one line,
    # cheap on the call path, obvious-from-context).
    from explore_persona_space.backends.artifacts import (
        build_expected_artifacts_declaration,
    )

    return build_expected_artifacts_declaration(
        issue=spec.issue,
        sentinel_path=sentinel_path_for(config, spec.issue, attempt_id),
        custom_workload=bool(spec.workload_cmd),
        attempt_id=attempt_id,
        hf_data_repo=config.hf_data_repo,
        hf_model_repo=config.hf_model_repo,
        wandb_run_path=wandb_run_path,
        extra_hf_data_paths=extra_hf_data_paths,
        extra_hf_model_paths=extra_hf_model_paths,
        extra_git_paths=extra_git_paths,
        git_repo_root=git_repo_root,
        skip_default_git_paths=skip_default_git_paths,
    )


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
    # Git push credential (#1205; incident #825 r6-r8): the GCE clone is
    # tokenless (public-repo read), so a workload's `git push` of committed
    # eval-result JSONs failed DETERMINISTICALLY on auth. OPTIONAL —
    # drop-when-absent like ANTHROPIC/OPENAI (deliberately NOT in
    # REQUIRED_LAUNCH_SECRET_KEYS: plenty of GCE workloads never push, and
    # the push-verify leg fails loud when a pushing workload lacks it).
    "GITHUB_TOKEN",
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
    # HF public-storage headroom knobs (#564): the soft ceiling, the opt-in
    # overflow routing, the kill switch, and the cache TTL must reach the
    # VM workload or a dispatch-process opt-in silently no-ops remotely.
    # EPM_HF_STORAGE_CACHE_PATH is deliberately NOT threaded (a VM-local
    # path is wrong on the worker; workers use the default).
    "EPM_HF_STORAGE_SOFT_CEILING_TB",
    "EPM_HF_OVERFLOW_ROUTING",
    "EPM_HF_STORAGE_CHECK",
    "EPM_HF_STORAGE_CACHE_TTL_S",
    # Size-aware projected-headroom probe floor (#1034): same remote-relevance
    # as the #564 knobs above — a dispatch-process floor override must reach
    # the VM workload or it silently no-ops remotely.
    "EPM_HF_LARGE_UPLOAD_PROBE_GB",
    # Local-SSD scratch root for the per-cell .npz write-decoupling helper
    # (#674): forwarded so a dispatch-process EPS_SCRATCH_DIR reaches the GCE
    # workload subprocess. The startup script ALSO sets a default (below), so
    # this passthrough is the override channel, the default is the floor.
    "EPS_SCRATCH_DIR",
    # HF Hub upload accelerator OVERRIDE channel (#745): forwarded so a
    # dispatch-process =0 / HF_HUB_DISABLE_XET=1 (the #515/#931 xet workaround)
    # reaches the GCE workload. The DEFAULTS (=1) are STATIC preamble exports
    # in render_startup_script (below); this passthrough is the override
    # channel only. Drop-when-absent contract preserved (an unset dispatch-env
    # key is simply not forwarded, so the static default stands).
    "HF_XET_HIGH_PERFORMANCE",
    "HF_HUB_ENABLE_HF_TRANSFER",
    # The REAL xet kill switch (#1195): huggingface_hub (0.36.2 pin) reads
    # HF_HUB_DISABLE_XET in constants.py — forwarding it lets a dispatch-process
    # =1 reach the worker (rung 1 of the upload-wedge ladder,
    # .claude/rules/upload-policy.md).
    "HF_HUB_DISABLE_XET",
    # Legacy no-op alias (consumed by nothing on the pinned stack — verified
    # #1049); kept so existing launch commands forward it harmlessly. Do not
    # cite it in new recipes.
    "HF_XET_DISABLE",
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

#: Default done-grace window (#935): 5400 s = 3x the 1800 s max adaptive
#: poll interval (``POLL_INTERVAL_QUIET_SEC``, scripts/poll_pipeline.py +
#: backends/base.py), so a healthy orchestrator's sentinel drain always
#: lands well inside the window before the self-poweroff can fire.
DEFAULT_DONE_POWEROFF_GRACE_SECONDS = 5400


def _done_grace_seconds() -> int:
    """Render-time grace for the #935 done-grace self-poweroff (0 disables).

    Reads ``EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`` from the DISPATCHER env
    (the established lane tuning surface — the render-time sibling of the
    poller-side ``EPS_GCP_QUEUE_WAIT_SECONDS``); the resolved integer is
    baked into the rendered script as ``EPS_DONE_GRACE``. ``0`` disables
    the countdown (negatives clamp to 0); a non-numeric value falls back
    to the default WITH a logged warning — a bad knob must never block a
    launch.
    """
    raw = os.environ.get(
        "EPS_GCP_DONE_POWEROFF_GRACE_SECONDS", str(DEFAULT_DONE_POWEROFF_GRACE_SECONDS)
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "EPS_GCP_DONE_POWEROFF_GRACE_SECONDS=%r is not an integer; using %d.",
            raw,
            DEFAULT_DONE_POWEROFF_GRACE_SECONDS,
        )
        return DEFAULT_DONE_POWEROFF_GRACE_SECONDS


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

    1. Sets strict mode + umask, saves the metadata runner's pipe on
       fd 3, and installs a ``trap ':' PIPE`` HANDLER (#607).
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
    9. On a CRASH (rc != 0), the EXIT trap uploads the workload log +
       any partial artifacts to the HF data repo under
       ``issue<N>_partial/<attempt_id>/`` via ``_eps_persist_diagnostics``
       BEFORE the ``shutdown -h now`` that triggers the
       ``--instance-termination-action=DELETE`` boot-disk destruction
       (#658). Without this a GCP crash loses its own traceback + partial
       output forever (the disk is gone), so the bug must be diagnosed by
       inference and every retry produces nothing recoverable. As of
       #854 the partial sweep covers ``eval_results/issue_<N>/`` AND both
       ``data/issue_<N>/`` / ``data/issue<N>/`` working-dir conventions
       (re-downloadable ``hf_dl``/``g*_dl``/``store`` caches excluded,
       per-dir byte cap ``EPS_PERSIST_DIR_CAP_BYTES``), a per-crash
       timestamped ``workload_<ts>.log`` copy plus a
       ``crash_persist_transcript.log`` audit upload LAST, every
       upload/skip streams an eager ``[crash-persist]`` line to the
       serial console AS IT HAPPENS, and the trap kills the reachability
       watchdog at ENTRY so no other in-guest actor can power the VM off
       mid-persist. As of #885 the trap ALSO sweeps ``worker_logs/`` —
       every regular file under ``$WORKLOAD_ROOT/logs/`` (fan-out
       per-worker logs carrying the real traceback; the canonical
       workload.log ends at the fan-out line), newest-first by mtime,
       per-file tail cap ``EPS_PERSIST_LOG_FILE_CAP_BYTES`` (default
       5 MiB; an oversized file is TAILED at stage time, never skipped
       wholesale — the traceback is at the END of a log), file-count
       bound ``EPS_PERSIST_LOG_MAX_FILES`` (default 40), staged into
       ``/tmp/eps-worker-logs`` and uploaded as ONE ``upload_folder``
       commit (never a per-file ``upload_file`` loop — the #664
       504-storm gotcha). As of #1339 a partial dir whose post-exclude
       file count exceeds ``EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT``
       (default 1000; ``< 1`` disables chunking with a WARN) uploads as
       newest-first staged batches (staging root
       ``EPS_PERSIST_DIR_STAGE_DIR``, default ``/tmp/eps-dir-batch``),
       each ONE ``upload_folder`` commit with one bounded retry
       (``EPS_PERSIST_RETRY_BACKOFF_S`` backoff), abandoning the dir
       loudly after ``EPS_PERSIST_DIR_BATCH_ABORT_STREAK`` (default 2)
       consecutive fully-failed batches — repo paths byte-identical to
       the unchunked upload (incident #1090: a 29,024-file single
       commit landed server-side but the gateway timed out delivering
       the response, so the client logged FAILED on a success). The
       sweep covers these three named directories plus the ``logs/``
       worker-log tree — still NOT universal artifact discovery.
    10. On a CLEAN exit, AFTER the completion sentinel + the ``done``
        publish, the script's LAST action is the #935 done-grace
        self-poweroff: a bounded countdown (``EPS_DONE_GRACE`` seconds,
        render-time knob ``EPS_GCP_DONE_POWEROFF_GRACE_SECONDS``, default
        5400; ``0`` disables) that aborts on an operator keepalive file
        (``EPS_DONE_KEEPALIVE_PATH``) or an ``eps/phase`` change (a
        sanctioned same-VM relaunch re-published ``workload`` per #908),
        best-effort persists the UNDRAINED sentinel set to the HF data
        repo under ``issue<N>_done/<attempt_id>/`` at expiry, then powers
        off UNCONDITIONALLY — so a done VM whose orchestrator/poller died
        bills at most ~the grace window instead of until the next
        dispatch's #908 reclaim or the daily janitor (~19-24 h worst
        case, the #763 done-zombie).

    The workload's existing HF/WandB upload paths remain the AUTHORITATIVE
    artifact route during a normal run; the sentinel is a small completion
    proof, not a primary artifact. The #658 EXIT-trap upload is a
    crash-only SAFETY NET (the clean-exit path keeps the VM alive for the
    success-sentinel scp — bounded, as of #935, by the done-grace
    self-poweroff window — + the workload already uploaded), fully guarded
    + 300s-bounded so it can never delay the poweroff that bounds billing.

    ``hydra_args`` defaults to ``spec.hydra_args`` (so the caller can
    override for a custom dispatch); ``repo_branch`` defaults to ``main``.

    When ``spec.workload_cmd`` is set (#588) the workload block runs
    that command verbatim instead of ``scripts/train.py``; all other
    lifecycle machinery (secrets fetch, in-VM preflight, ``eps/phase``
    guest attributes, EXIT trap, completion sentinel) is identical, and
    the hydra branch is byte-for-byte the pre-#588 render (pinned by the
    snapshot test).

    Push-verify backstop (#1205, incident #825 r6-r8): the workload_cmd
    branch additionally (a) configures an env-reading git credential
    helper when ``GITHUB_TOKEN`` arrived via instance metadata (the GCE
    clone is tokenless, so workload pushes of committed eval-result
    JSONs failed deterministically on auth), and (b) after the
    detached-wait loop verifies ``git rev-list --count
    origin/<branch>..HEAD == 0`` — retrying the push twice, then
    bundling the unpushed range into ``data/issue_<N>/``
    (crash-persist-swept, #854 item 5) and ``exit 86`` so the EXIT trap
    publishes ``failed`` + crash-persist + poweroff instead of a false
    ``done`` with the only copy of the commit on a self-DELETEing
    instance. The hydra branch is leg-free (``train.py`` has no git
    usage).

    Output redirect (#607, incident #491): the script NEVER streams
    workload output through startup-script stdout — the GCE metadata
    runner reads that pipe with a bounded line scanner and a giant
    newline-free line (vLLM/tqdm ``\\r``-progress bars) kills the script
    on SIGPIPE while the VM zombies at RUNNING. After the env-export
    block (before the secrets fetch) the script
    ``exec >>"$EPS_LOG_PATH" 2>&1`` into ``log_path_for(config, issue)``
    (= the handle's ``log_path``, so the poll's drain tail carries real
    workload output), exports ``TQDM_DISABLE=1`` as defense in depth,
    and keeps only sparse guarded heartbeats on the saved runner pipe
    (fd 3). A ``trap ':' PIPE`` HANDLER (not ignore — children must keep
    default SIGPIPE) converts any residual pipe-write death into a
    normal rc=1 failure so the EXIT trap fires with a real rc.

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
    deliverables_ok_abs = deliverables_ok_path_for(config, spec.issue, attempt_id)
    sentinel_dir = sentinel_abs.rsplit("/", 1)[0]
    # Done-grace self-poweroff constants (#935), baked at render time like
    # the other lane constants (no runtime metadata fetch). The keepalive
    # path carries NO .json suffix ON PURPOSE: the poller's sentinel drain
    # glob is /workspace/logs/issue-<N>-*.json, and the operator escape
    # hatch must never be ingested as a sentinel.
    done_grace = _done_grace_seconds()
    keepalive_path = f"/workspace/logs/issue-{spec.issue}-keepalive"

    # Build the secret-fetch stanza. Each KEY is pulled from
    # ``/computeMetadata/v1/instance/attributes/<KEY>``. The
    # ``Metadata-Flavor: Google`` header is the GCE-required guard; the
    # curl path 404s cleanly when a key was not set (so an absent
    # secret produces an empty fetch, not a hard crash — the in-VM
    # workload's own preflight surfaces the missing token loudly).
    # The non-secret STARTUP_PASSTHROUGH_ENV_KEYS (adapter-persist
    # targets) ride the same fetch stanza — metadata is the one
    # env-delivery surface the VM has.
    #
    # DEFAULT-PRESERVING fetch (#745, round 2): an ABSENT metadata key
    # MUST NOT export an empty value. The accelerator keys
    # (HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER) carry a STATIC
    # default ``=1`` exported earlier in the env-export block; the OLD
    # unconditional ``KEY=$(curl ... || true); export KEY`` overwrote that
    # ``1`` with ``""`` for every default GCE workload (the common case —
    # the dispatcher does not forward the accelerator keys, so their
    # metadata attribute is absent), silently disabling the load-bearing
    # GCE lane acceleration (round-1 binding blocker). Fetch into a temp
    # ``_VAL`` and ONLY ``export KEY="$_VAL"`` when ``_VAL`` is non-empty,
    # so an absent key leaves the PRIOR export intact (the static ``1`` for
    # the accelerators; an unset var for an absent secret — identical
    # ``[ -n "${KEY:-}" ]``-failing behaviour to the old empty export, so
    # the REQUIRED_LAUNCH_SECRET_KEYS preflight below still fires). An
    # explicit dispatcher-set ``0`` arrives as metadata ``0`` →
    # ``_VAL=0`` (non-empty) → ``export KEY=0``, so the override channel
    # (the #515 xet-CDN ``=0`` / ``HF_HUB_DISABLE_XET=1`` workaround) is
    # preserved. ``_VAL`` is scratch — never exported itself.
    secrets_fetch_lines: list[str] = []
    for key in STARTUP_SECRET_ENV_KEYS + STARTUP_PASSTHROUGH_ENV_KEYS:
        secrets_fetch_lines.append(
            f'_VAL=$(curl -fsS -H "Metadata-Flavor: Google" '
            f'"http://metadata.google.internal/computeMetadata/v1/'
            f'instance/attributes/{key}" 2>/dev/null || true); '
            f'[ -n "$_VAL" ] && export {key}="$_VAL"; unset _VAL'
        )

    # Hydra args, shell-quoted. Empty tuple → empty string.
    hydra_str = " ".join(shlex.quote(a) for a in args)

    # Workload block (#588, rc-wrapper #1004): a custom workload_cmd is
    # rendered as the SINGLE argument of an inner
    # ``bash -eu -o pipefail -c <shlex.quote(cmd)>`` — the inner bash
    # re-parses it as a complete shell line (full shell syntax preserved,
    # the original #588 verbatim concern), while from THIS script's
    # perspective the workload is one SIMPLE command, so the outer
    # ``set -e`` fires on ANY non-zero exit — including a
    # ``cmd1 && cmd2`` chain whose FIRST command crashes, which the
    # pre-#1004 bare splice rc-masked into a false phase=done (bash
    # exempts non-final &&/|| list members from errexit; incident #952
    # run 1). Residuals NOT closed by the wrapper: (a) a ``a && b; c``
    # shape where ``a`` fails and ``c`` succeeds still masks (the same
    # errexit exemption applies inside the inner bash; the #750 OOM
    # guard remains the independent backstop); (b) the detached-driver
    # pid-file wait loop below is liveness-only (``kill -0``), so a
    # setsid-detached driver that CRASHES still reaches the success
    # tail — a structurally different class (#601/#977 contract).
    # Trusted input by design (same trust level as the plan's
    # Reproducibility Card launch command; it runs as root on the VM).
    # The RunSpec.__post_init__ single-line check keeps the rendered
    # script line-structured (shlex.quote of a single-line string is a
    # single line). The hydra branch is the byte-identical pre-#588
    # lines, gated only by ``if spec.workload_cmd``.
    if spec.workload_cmd:
        workload_block = [
            "# === REPO_ROOT export (#641; trap #599) ===",
            "# The GCE startup script clones the repo to $WORKLOAD_ROOT",
            "# (/workspace/eps-issue-<N>) and runs workloads from there, but",
            "# many driver scripts default REPO_ROOT to the RunPod path",
            '# (REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}") and',
            "# `cd` to it under set -e. That fallback path does NOT exist on the",
            "# GCP lane, so the driver dies at its first cd and the EXIT trap",
            "# powers the VM off — burning a fresh GPU instance + wall-clock",
            "# (incident #599, recurred #641 attempt-3, 2026-06-18). Exporting",
            "# REPO_ROOT here makes every `${REPO_ROOT:-...}` fallback resolve",
            "# transparently with NO orchestrator-side --workload-cmd prefix.",
            "# Backward-compatible: a workload_cmd that DOES prefix",
            '# `REPO_ROOT="X" bash ...` keeps its value — a bash per-command env',
            "# var supersedes the inherited export for that one command.",
            'export REPO_ROOT="$WORKLOAD_ROOT"',
            "# === Repo-root PYTHONPATH (#1172; trap #823/#853) ===",
            "# Script-mode python puts the SCRIPT's dir on sys.path[0] — not",
            "# cwd, not the repo root — so a deferred `from scripts.X import`",
            "# in a src-layout driver dies mid-run with ModuleNotFoundError",
            "# (incident #823 Phase-3: a full GCE launch lost at ~30 min).",
            "# Prepend the repo root; the :+ form appends any inherited",
            "# PYTHONPATH after ours AND avoids the empty-value trailing",
            "# colon (a leading/trailing colon silently adds cwd to sys.path",
            "# — cpython #107353). Safe under set -u (:+ is nounset-exempt).",
            "# Masks the trap for managed launches only; hand launches still",
            "# need the gotchas.md _ensure_repo_root_on_syspath() guidance.",
            'export PYTHONPATH="$WORKLOAD_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
            "# === WandB project default (#601 follow-up r1) ===",
            "# HF-Trainer workloads that never set WANDB_PROJECT land in WandB's",
            "# global default project 'huggingface', violating the Upload Policy",
            "# (training metrics → project=<experiment_name>). Default to the",
            "# per-issue project; :- fills only unset/empty, so an inline",
            "# WANDB_PROJECT=... prefix on the workload command — or the workload",
            "# setting its own project internally — still wins.",
            f'export WANDB_PROJECT="${{WANDB_PROJECT:-issue{spec.issue}}}"',
            "# === Local-SSD scratch root default (#674) ===",
            "# The per-cell .npz write-decoupling helper",
            "# (orchestrate/scratch_io.py) routes to scratch only when",
            "# EPS_SCRATCH_DIR is set in the workload env; default it to a",
            "# local-SSD path so the GCE lane gets the network-decoupling",
            "# without an explicit dispatch-process override. :- fills only",
            "# unset/empty, so a forwarded EPS_SCRATCH_DIR (passthrough key) or",
            "# an inline prefix still wins. /tmp on the GCP DLVM image sits on",
            "# the local boot/SSD disk (NOT the network PD), so the hot writes",
            "# land off the NIC-shared plane — the whole point.",
            'export EPS_SCRATCH_DIR="${EPS_SCRATCH_DIR:-/tmp/eps_scratch}"',
            "# === Git push credential (#1205; incident #825 r6-r8, the 2026-07-08T11:17/11:19Z",
            "# upload-verification reads) ===",
            "# The GCE clone is tokenless (public-repo read), so a workload's",
            "# `git push origin issue-<N>` of committed eval JSONs failed",
            "# DETERMINISTICALLY on auth, and the `|| echo WARNING` shape swallowed",
            "# it (#825 r8: commit 87c9c73168 / 73 eval JSONs existed only on the",
            "# self-DELETEing instance). Configure an env-reading credential helper:",
            "# the token is never at rest in .git/config and never in a URL that a",
            "# crash-persisted workload log could leak (unlike the tokenized-remote",
            "# pattern bootstrap_pod.sh uses pod-side). Repo-local to $WORKLOAD_ROOT:",
            "# a workload that creates a SECOND clone gets neither the helper nor",
            "# the push-verify backstop below. Gated: a launch without GITHUB_TOKEN",
            "# keeps today's behavior; the push-verify leg below then fails the",
            "# workload loud instead of silently losing the commit.",
            "export GIT_TERMINAL_PROMPT=0",
            'if [ -n "${GITHUB_TOKEN:-}" ]; then',
            '  git -C "$WORKLOAD_ROOT" config credential.helper '
            "'!f() { echo username=x-access-token; echo \"password=${GITHUB_TOKEN}\"; }; f'",
            "fi",
            "# === Sentinel drain dir (#610) ===",
            "# The pod-side signaling contract names /workspace/logs/",
            "# issue-<N>-*.json as the poll's drain glob, but nothing on the",
            "# GCP lane created the dir — the issue-610 dispatcher found it",
            "# missing, fell back to its out_root logs dir, and the poll",
            "# reported done with sentinels_processed=0. Pre-create it",
            "# world-writable (umask 077 is active above; chmod covers any",
            "# workload sub-process that drops root) so workloads can write",
            "# sentinels + detach pid files at the canonical path.",
            "mkdir -p /workspace/logs",
            "chmod 777 /workspace/logs",
            "# === Run the workload (custom workload_cmd) ===",
            "# rc-preserving wrapper (#1004, incident #952): the command is the",
            "# single argument of an inner bash -eu -o pipefail -c, so it is ONE",
            "# simple command here — a failure ANYWHERE inside (including the",
            "# first member of a && chain, which errexit exempts mid-list) exits",
            "# the inner bash non-zero, the outer set -e fires, and the EXIT trap",
            "# publishes phase=failed + crash-persist + poweroff → poll reads dead.",
            "# The inner -eu -o pipefail mirror the outer flags, so the command",
            "# text's own execution semantics are identical to the pre-#1004",
            "# in-script splice; only the propagated rc changes. Residuals: an",
            "# `a && b; c` shape still masks (errexit exempts mid-list members",
            "# inside the inner bash too; the #750 OOM guard is the backstop),",
            "# and the pid-file wait below is liveness-only (kill -0) — a",
            "# CRASHED setsid-detached driver still reaches the success tail.",
            "_eps_phase workload",
            "touch /tmp/eps-workload-start",
            f"bash -eu -o pipefail -c {shlex.quote(spec.workload_cmd)}",
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
            "# === Push-verify leg (#1205; incident #825 r6-r8, the 2026-07-08T11:17/11:19Z",
            "# upload-verification reads) ===",
            "# Any commit the workload made on the cloned branch is a result commit",
            "# (Upload Policy: eval JSONs commit to the issue branch). git push",
            "# updates the remote-tracking ref on success, so count==0 IS",
            "# push-landed proof; a swallowed in-workload push failure leaves",
            "# origin/<branch>..HEAD non-empty. Retry here (authenticated via the",
            "# credential helper above); if commits remain unpushed, bundle them",
            "# into data/issue_<N>/ (crash-persist-swept, #854 item 5; gitignored so",
            "# a later gap-fill can never commit the bundle) and FAIL LOUD so the",
            "# EXIT trap publishes failed + crash-persist + poweroff instead of a",
            "# false done. rev-list failures are NOT defaulted-to-0: under set -e a",
            "# failing command substitution kills the script -> trap (fail-fast).",
            "# Ordering: this leg runs BEFORE the #750 OOM guard (EPS_OOM_FINAL) —",
            "# a rc-survived-OOM run whose push ALSO fails records rc 86, not 137",
            "# (accepted: rare double-failure; both route through the same trap).",
            f"_EPS_PUSH_BRANCH={shlex.quote(repo_branch)}",
            '_EPS_UNPUSHED="$(git -C "$WORKLOAD_ROOT" rev-list --count'
            ' "origin/${_EPS_PUSH_BRANCH}..HEAD")"',
            'if [ "$_EPS_UNPUSHED" != "0" ]; then',
            '  echo "[push-verify] ${_EPS_UNPUSHED} unpushed commit(s) on'
            ' ${_EPS_PUSH_BRANCH} — retrying push (#1205)"',
            '  git -C "$WORKLOAD_ROOT" push origin "HEAD:${_EPS_PUSH_BRANCH}"'
            ' || { sleep 20; git -C "$WORKLOAD_ROOT" push origin "HEAD:${_EPS_PUSH_BRANCH}"; }'
            " || true",
            '  _EPS_UNPUSHED="$(git -C "$WORKLOAD_ROOT" rev-list --count'
            ' "origin/${_EPS_PUSH_BRANCH}..HEAD")"',
            '  if [ "$_EPS_UNPUSHED" != "0" ]; then',
            '    echo "[push-verify] FAIL: ${_EPS_UNPUSHED} commit(s) still unpushed on'
            ' ${_EPS_PUSH_BRANCH} after retry — failing the workload loud (#1205)"',
            f'    mkdir -p "$WORKLOAD_ROOT/data/issue_{spec.issue}"',
            f'    git -C "$WORKLOAD_ROOT" bundle create "$WORKLOAD_ROOT/data/issue_{spec.issue}'
            '/unpushed-$(git -C "$WORKLOAD_ROOT" rev-parse --short HEAD).bundle"'
            ' "origin/${_EPS_PUSH_BRANCH}..HEAD" || true',
            "    exit 86",
            "  fi",
            '  echo "[push-verify] retry landed: origin/${_EPS_PUSH_BRANCH} now includes HEAD"',
            "else",
            '  echo "[push-verify] OK: no unpushed commits on ${_EPS_PUSH_BRANCH}"',
            "fi",
            "# === /push-verify leg ===",
        ]
    else:
        workload_block = [
            "# === Run the workload (Hydra args = the spec's hydra_args) ===",
            "# A non-zero exit propagates (set -e) → the EXIT trap publishes",
            "# phase=failed + powers off → poll reads dead.",
            "_eps_phase workload",
            f"uv run python scripts/train.py {hydra_str}".rstrip(),
        ]

    log_path = log_path_for(config, spec.issue)
    log_dir = log_path.rsplit("/", 1)[0]

    parts = [
        "#!/bin/bash",
        "set -euo pipefail",
        "umask 077",
        # fd-3 save + PIPE handler (#607): the metadata runner reads the
        # script's stdout line-by-line with a BOUNDED Go bufio.Scanner; a
        # giant newline-free line (vLLM/tqdm \r-progress bars) overflows
        # it, the runner closes the pipe, and the next builtin write dies
        # on SIGPIPE with the EXIT trap reading rc=0 — the #491 zombie.
        # The comments below are rendered into the script so the on-VM
        # copy is self-describing.
        "# === SIGPIPE + serial-heartbeat setup (#607, incident #491) ===",
        "# fd 3 = the metadata runner's pipe. Only tiny newline-terminated",
        "# heartbeats ever go there; raw workload output never does (the",
        "# runner's bounded line scanner kills the script on giant lines).",
        "# fd 3 is inherited by workload children: nothing legitimately writes",
        "# to it, and a stray giant `>&3` write now fails LOUDLY (child SIGPIPE",
        "# -> rc!=0 -> EXIT trap -> phase=failed + shutdown), never a zombie.",
        "exec 3>&1",
        "# A HANDLER (not ignore): the parent shell survives a closed runner",
        "# pipe as a normal write error (rc=1 -> set -e -> EXIT trap fires with",
        "# rc!=0), while children keep DEFAULT SIGPIPE (SIG_IGN would be",
        "# inherited across exec and break producer|head pipelines under",
        "# pipefail in workload drivers).",
        "trap ':' PIPE",
        # Publish the workload phase to the GCE guest attribute
        # ``eps/phase`` — the ONLY poll-readable surface the VM has
        # while staying RUNNING (the success path keeps the VM alive so
        # the sentinel can be scp'd — bounded by the #935 done-grace
        # self-poweroff — so instance status alone cannot signal
        # completion; issue 535 r9 spun the poll for the full 4 h
        # timeout on a 9-min success). Best-effort (`|| true`): a probe
        # hiccup must never kill the workload. #607 additions: a
        # ``[phase=...]`` echo on CURRENT stdout (post-redirect: the log
        # file, parseable by poll_pipeline.latest_phase; pre-redirect:
        # one tiny line on the pipe — safe) and a guarded
        # ``[startup-script] phase=...`` heartbeat on fd 3 (sparse serial
        # trace; deliberately NOT matching the ``\\[phase=`` parser).
        # Every heartbeat write is ``{ ...; } 2>/dev/null || true``-
        # guarded: with the PIPE handler a closed pipe yields a swallowed
        # write error, never an abort.
        '_eps_phase() { curl -fsS -X PUT -H "Metadata-Flavor: Google"'
        ' --data "$1" "http://metadata.google.internal/computeMetadata/v1/'
        'instance/guest-attributes/eps/phase" >/dev/null 2>&1 || true;'
        # ASYNC RunPod-failover discriminator (#659, MF3): publish a SEPARATE
        # write-once guest attribute ``eps/workload_started`` the instant the
        # WORKLOAD phase is entered. ``eps/phase`` is single-valued and the EXIT
        # trap overwrites it to ``failed`` on ANY non-zero exit, so a poll-time
        # read of ``failed`` cannot tell a real workload crash from a
        # setup/secrets/clone/uv-sync failure. A DIFFERENT key is never
        # overwritten by the ``eps/phase=failed`` write, so its presence at poll
        # time PROVES the workload phase was reached — letting ``GcpBackend.poll``
        # map ``failed`` to ``terminal_workload_failed`` (sentinel present, fail
        # over to RunPod) vs ``terminal_setup_failed`` (sentinel absent, do NOT
        # fail over — re-running a broken boot/setup script on RunPod just
        # re-crashes; §7 kill-criterion #1). Best-effort (``|| true``): a probe
        # hiccup here must never kill the workload, and the poll side treats a
        # PROBE FAILURE conservatively as workload-started (never as setup).
        ' if [ "$1" = "workload" ]; then'
        ' curl -fsS -X PUT -H "Metadata-Flavor: Google" --data "true"'
        ' "http://metadata.google.internal/computeMetadata/v1/'
        'instance/guest-attributes/eps/workload_started" >/dev/null 2>&1 || true; fi;'
        ' { echo "[phase=$1] startup-script $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }'
        " 2>/dev/null || true;"
        ' { echo "[startup-script] phase=$1" >&3; } 2>/dev/null || true; }',
        # Crash-persist breadcrumb channel (#1151): a SEPARATE guest-attribute
        # key ``eps/persist`` — NEVER ``eps/phase`` (the poll classification +
        # #908 zombie predicates key on eps/phase and must not see new values;
        # the #935 ``eps/done_persist`` separate-key discipline). Link-local
        # metadata write — zero HF / org-quota dependency — readable by the
        # poller on the TERMINATED instance exactly when the terminal
        # diagnosis runs (#811: every HF-riding persist signal failed
        # together; this is the HF-independent channel). ``-m 5`` so a wedged
        # metadata server can never eat the persist's 300s budget (the
        # done-grace READ uses the same cap). Values: ``attempted`` (entry) ->
        # ``ok`` | ``failed_uploads`` | ``timeout`` | ``failed_rc<N>`` |
        # ``skipped_no_token``. #1343: ``ok`` requires the verify gate — the
        # transcript existence probe read True, or >=1 client-confirmed
        # ``upload_folder`` return; ``failed_uploads`` = rc 3, zero uploads
        # verifiably succeeded. A
        # STANDING ``attempted`` with no final value = the persist was KILLED
        # mid-flight — a TERMINATED-only reading (a RUNNING-window read may
        # catch a healthy persist in flight). Decision table:
        # .claude/rules/compute-backend-failover.md § Part A item 8.
        '_eps_persist_status() { curl -fsS -m 5 -X PUT -H "Metadata-Flavor: Google"'
        ' --data "$1" "http://metadata.google.internal/computeMetadata/v1/'
        'instance/guest-attributes/eps/persist" >/dev/null 2>&1 || true; }',
        # Crash-diagnostics + partial-artifact preservation (#658). The
        # instance is created with --instance-termination-action=DELETE, so
        # the EXIT trap's `shutdown -h now` on a crash DESTROYS the boot
        # disk — taking the workload log (traceback / stderr) AND any
        # partial artifacts (eval_results JSONs the workload wrote before
        # crashing) with it. Incident #658 (2026-06-24): a deterministic
        # code crash lost its own traceback + ~30 partial output JSONs on
        # every retry, so the bug had to be diagnosed by inference and each
        # retry burned a GPU-hour producing nothing recoverable. This helper
        # uploads BOTH to the HF data repo under
        # ``issue<N>_partial/<attempt_id>/`` BEFORE the shutdown line, so a
        # crash is debuggable and partial progress is recoverable. It is
        # called from the EXIT trap's rc!=0 branch (the clean-exit path
        # keeps the VM alive for the success-sentinel scp — bounded by the
        # #935 done-grace self-poweroff — + the workload's own upload
        # paths already ran, so partial-upload there is moot).
        # Fully guarded + time-bounded: a hung/failed upload must NEVER
        # delay the `shutdown` that bounds billing — every step is
        # ``|| true`` and the whole upload is wrapped in ``timeout`` so the
        # trap always reaches the poweroff.
        #
        # #854 hardening (incident #825): the run-2 partial-data loss was a
        # silent COVERAGE-GAP skip (the sweep only looked in
        # eval_results/issue_<N>/ while the workload wrote data/issue_825/),
        # misdiagnosed as a poweroff race because every skip was silent and
        # the old `| cut | tail` pipe buffered all output until EOF. Now:
        # the sweep also covers data/issue_<N>/ + data/issue<N>/ (caches
        # excluded), every upload/failure/skip prints an eager
        # [crash-persist] line, a per-crash timestamped workload log copy +
        # a crash_persist_transcript.log audit survive same-attempt
        # re-crashes, and the EXIT trap reaps the reachability watchdog at
        # entry so nothing else can power off mid-upload.
        #
        # #885: fan-out dispatchers redirect each worker's output to
        # per-worker logs under $WORKLOAD_ROOT/logs/ (e.g.
        # logs/issue_779/corpus_gpu0_all.log), so the canonical workload.log
        # ends at the fan-out line and the REAL traceback lives only in a
        # worker log the dir sweep never covered (two #779 crashes each
        # needed a ~30-min manual boot-disk detach). The # 1b. sweep stages
        # logs/** (newest-first, per-file TAIL cap at stage time — the
        # traceback is at the END of a log) into a temp tree and uploads it
        # as ONE upload_folder commit — never a per-file upload_file loop,
        # which 504-storms on this large repo (#664, ~160 s/file measured:
        # 40 files would blow the 300s budget and starve the #854 artifacts).
        "_eps_persist_diagnostics() {",
        '  _rc="${1:-1}";',
        # #1151: unconditional entry breadcrumb — proof the persist was
        # ENTERED, on a channel with zero HF dependency. A standing
        # ``attempted`` with no later final value is the killed-mid-persist
        # signal (TERMINATED-only reading; see the decision table).
        '  _eps_persist_status "attempted";',
        # Nothing to do without a repo target or HF token (early-boot crash
        # before the env exports / secret fetch — let the trap power off).
        # #854: the skip is LOUD (fd 3 = serial console) — a silent early
        # return here is indistinguishable from a killed persist.
        '  if [ -z "${EPS_HF_DATA_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then',
        '    { echo "[crash-persist] SKIP-ALL: EPS_HF_DATA_REPO or HF_TOKEN unset'
        ' (early-boot crash)" >&3; } 2>/dev/null || true;',
        # #1151: the skip is also breadcrumbed off-VM — an early-boot crash
        # before the secrets fetch is otherwise indistinguishable from a
        # killed persist once the boot disk is DELETEd.
        '    _eps_persist_status "skipped_no_token";',
        "    return 0;",
        "  fi;",
        '  _dest="issue${EPS_ISSUE:-0}_partial/${EPS_ATTEMPT_ID:-unknown}";',
        '  _crash="/tmp/eps-crash-report.json";',
        # A compact crash report: exit code + timestamp + the log tail
        # (jq-free; the tail is JSON-escaped by python below at upload time).
        '  { printf \'{"issue":%s,"attempt_id":"%s","exit_code":%s,'
        '"ended_utc":"%s","kind":"gcp-exit-trap-crash-diagnostics"}\\n\''
        ' "${EPS_ISSUE:-0}" "${EPS_ATTEMPT_ID:-unknown}" "$_rc"'
        ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$_crash"; } 2>/dev/null || true;',
        '  { echo "[startup-script] uploading crash diagnostics + partial'
        ' artifacts to ${EPS_HF_DATA_REPO}/${_dest}"; } 2>/dev/null || true;',
        # Bounded HF upload via huggingface_hub (already installed by
        # `uv sync`). 300s ceiling so a stuck upload can't strand the VM
        # billing; cd into the repo so `uv run` resolves the synced env.
        # Prepend uv's install dir to PATH inside the subshell: the trap
        # can fire AFTER the secrets fetch (so the HF_TOKEN guard passes)
        # but BEFORE the later `export PATH="$HOME/.local/bin:$PATH"` in
        # the uv-install block — without this, `uv` is not found in that
        # narrow window and the diagnostics upload silently no-ops.
        # HF_HUB_DISABLE_PROGRESS_BARS: the fd-3 stream below is now EAGER
        # per-line (#854), so hub progress bars would spam the serial console.
        # #1151: rc-file capture — the persist subshell sits on the LEFT of
        # the streamer pipeline below, so its exit status is invisible to the
        # function body ($? after the pipeline is the STREAMER's rc); the
        # /tmp rc file is the only channel through the pipe (the #935
        # EPS_DONE_PERSIST_STATUS status-file pattern). Cleared here so a
        # same-boot re-crash never reads a PRIOR persist's rc.
        '  rm -f "${EPS_CRASH_PERSIST_RC:-/tmp/eps-crash-persist.rc}" 2>/dev/null || true;',
        # #1151: ``uv run --no-sync`` — at trap time the env was already
        # synced by the boot; --no-sync removes uv's lock-check / re-sync
        # network exposure (the H-B bootstrap-failure hypothesis) and its
        # latency from the 300s budget. Scope: the CRASH helper only (the
        # #935 done helper keeps its own record).
        '  ( export PATH="${HOME:-/root}/.local/bin:$PATH" HF_HUB_DISABLE_PROGRESS_BARS=1;'
        ' cd "${WORKLOAD_ROOT:-/}" 2>/dev/null'
        ' && timeout 300 uv run --no-sync python - "$_dest" "$_crash" <<\'EPS_PERSIST_PY\'',
        "import datetime, os, shutil, subprocess, sys, time",
        "from pathlib import Path",
        "from huggingface_hub import HfApi",
        "dest, crash = sys.argv[1], sys.argv[2]",
        'repo = os.environ["EPS_HF_DATA_REPO"]',
        'issue = os.environ.get("EPS_ISSUE", "0")',
        'log_path = os.environ.get("EPS_LOG_PATH", "")',
        'root = Path(os.environ.get("WORKLOAD_ROOT", ""))',
        'transcript = os.environ.get("EPS_PERSIST_TRANSCRIPT", "/tmp/eps-crash-persist.log")',
        "def _say(msg):",
        "    # every line is printed flush=True (eager fd-3 streaming) AND teed into the",
        "    # transcript file uploaded LAST — a poweroff-independent skip-vs-kill audit on",
        "    # HF (#854). The append is best-effort: stdout already carries the line, and a",
        "    # transcript write failure must never break the persist itself.",
        "    print(msg, flush=True)",
        "    try:",
        '        with open(transcript, "a") as fh:',
        '            fh.write(msg + "\\n")',
        "    except OSError:",
        "        pass",
        "_say(f\"[crash-persist] BEGIN repo={repo} dest={dest} log_path={log_path or 'UNSET'}\")",
        "api = HfApi()",
        "# #1343: client-confirmed upload successes — an upload_folder RETURN means",
        "# the commit response was delivered, i.e. the commit is durable (#1339's",
        "# reliable direction). Feeds the exit-3 honesty gate at the end.",
        'OK_UPLOADS = {"n": 0}',
        "# #1151: staged BUNDLES replace the per-file upload_file calls — on this",
        "# ~1M-file repo a per-file upload triggers a server-side recursive",
        "# tree-listing pre-check that 504s ~half the time at ~160 s/file (#664);",
        "# one or two stalled pre-checks would eat the entire 300s budget before",
        "# the first commit lands. upload_folder composes ONE commit per bundle.",
        "def _stage_into(bundle_dir, name, src):",
        "    # Copy src into the staged bundle dir under its REPO name (the bundle",
        "    # uploads to {dest}, so staged names land at the byte-identical repo",
        "    # paths). Failures logged, never raised — the persist must always",
        "    # reach the trap's shutdown.",
        "    try:",
        "        bundle_dir.mkdir(parents=True, exist_ok=True)",
        "        shutil.copyfile(src, bundle_dir / name)",
        '        _say(f"[crash-persist] staged {name} ({(bundle_dir / name).stat().st_size}'
        ' bytes)")',
        "    except Exception as exc:",
        '        _say(f"[crash-persist] FAILED staging {name}: {exc}")',
        "def _up_bundle(bundle_dir, label, retry):",
        "    # ONE upload_folder commit of the staged bundle -> {dest}/<staged names>.",
        "    # retry=True -> ONE retry after an env-tunable backoff (the #935",
        "    # EPS_DONE_PERSIST_RETRY_BACKOFF_S sibling) — the FIRST bundle only:",
        "    # the crash traceback is the highest-value artifact; further retries",
        "    # would risk the 300s budget.",
        "    files = ([p for p in bundle_dir.rglob('*') if p.is_file()]",
        "             if bundle_dir.is_dir() else [])",
        "    if not files:",
        '        _say(f"[crash-persist] SKIP bundle {label}: nothing staged")',
        "        return",
        "    attempts = 2 if retry else 1",
        "    for i in range(1, attempts + 1):",
        "        try:",
        '            _say(f"[crash-persist] uploading bundle {label} ({len(files)} files,'
        ' attempt {i}/{attempts}, one commit)")',
        "            api.upload_folder(folder_path=str(bundle_dir), path_in_repo=dest,",
        '                              repo_id=repo, repo_type="dataset")',
        '            _say(f"[crash-persist] uploaded bundle {label}")',
        '            OK_UPLOADS["n"] += 1',
        "            return",
        "        except Exception as exc:",
        '            _say(f"[crash-persist] FAILED bundle {label} attempt {i}/{attempts}: {exc}")',
        "            if i < attempts:",
        "                try:",
        '                    _b = int(os.environ.get("EPS_PERSIST_RETRY_BACKOFF_S", "10"))',
        "                except ValueError:",
        "                    _b = 10",
        "                time.sleep(max(0, _b))",
        "# Shared cache-exclude constants — used by BOTH the # 1b. worker-logs sweep and",
        "# the # 2. partial-dirs sweep below (hoisted above their first caller, #885).",
        'IGNORE = ["hf_dl/**", "g*_dl/**", "store/**", ".cache/**", "__pycache__/**",',
        '          "**/hf_dl/**", "**/g*_dl/**", "**/store/**", "**/.cache/**",',
        '          "**/__pycache__/**"]',
        'PRUNE = {"hf_dl", "store", ".cache", "__pycache__"}',
        'CAP = int(os.environ.get("EPS_PERSIST_DIR_CAP_BYTES", 2 * 1024**3))',
        "# 1. crash report + workload log, small-first (the traceback is the highest-value",
        "#    artifact; a worst-case timeout still lands it). A same-attempt re-crash",
        "#    overwrites the canonical names (#854: run-3 overwrote run-2's log on HF), so",
        "#    a per-crash timestamped log copy ALSO uploads — LAST, in the final bundle",
        "#    (small-first; the canonical log already carries the traceback early).",
        "#    #1151: both files ride ONE staged upload_folder commit with ONE retry",
        "#    (never per-file upload_file — the #664 pre-check stall class); repo paths",
        "#    stay byte-identical ({dest}/crash_report.json, {dest}/workload.log).",
        'stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")',
        'first_stage = Path(os.environ.get("EPS_PERSIST_FIRST_STAGE_DIR", "/tmp/eps-crash-first"))',
        "shutil.rmtree(first_stage, ignore_errors=True)",
        "if Path(crash).is_file():",
        '    _stage_into(first_stage, "crash_report.json", crash)',
        "else:",
        '    _say(f"[crash-persist] SKIP crash_report.json: no such file ({crash})")',
        "if log_path and Path(log_path).is_file():",
        '    _stage_into(first_stage, "workload.log", log_path)',
        "else:",
        '    _say(f"[crash-persist] SKIP workload.log: EPS_LOG_PATH unset or file missing'
        ' ({log_path!r})")',
        '_up_bundle(first_stage, "first", retry=True)',
        "# 1b. worker logs (#885) — fan-out dispatchers redirect the real traceback to",
        "#     per-worker logs under $WORKLOAD_ROOT/logs/ (e.g.",
        "#     logs/issue_779/corpus_gpu0_all.log); the canonical workload.log ends at the",
        "#     fan-out line. Newest-first, per-file TAIL cap at STAGE time (the traceback",
        "#     is at the END of a log — an oversized file is tailed, never skipped",
        "#     wholesale), file-count bound, then ONE upload_folder commit (NEVER a",
        "#     per-file upload_file loop — the #664 504-storm gotcha). Runs BEFORE the",
        "#     partial dirs so a worst-case timeout still lands the tracebacks.",
        "#     #1351: git-TRACKED files under logs/ (the repo is CLONED at",
        "#     $WORKLOAD_ROOT, so committed logs/daily+weekly retrospectives are on",
        "#     disk) are EXCLUDED at WALK time — already durable in git, they",
        "#     cluttered the crash prefix and consumed the LOG_MAX_FILES budget",
        "#     (#1345, 48 tracked files vs the 40-file bound). ANY git failure —",
        '#     nonzero rc (incl. a safe.directory "dubious ownership" refusal),',
        "#     timeout, or a missing git binary — FAILS OPEN to sweeping",
        "#     everything: crash forensics beat cleanliness.",
        "def _env_int(name, default):",
        "    try:",
        "        return int(os.environ.get(name, default))",
        "    except (TypeError, ValueError):",
        '        _say(f"[crash-persist] WARN {name} malformed; using default {default}")',
        "        return default",
        'LOG_FILE_CAP = _env_int("EPS_PERSIST_LOG_FILE_CAP_BYTES", 5 * 1024**2)',
        'LOG_MAX_FILES = _env_int("EPS_PERSIST_LOG_MAX_FILES", 40)',
        "def _up_logs():",
        '    logs_root = root / "logs"',
        "    if not logs_root.is_dir():",
        '        _say(f"[crash-persist] SKIP worker_logs: no such dir ({logs_root})")',
        "        return",
        "    if LOG_MAX_FILES < 1:",
        '        _say(f"[crash-persist] SKIP worker_logs:'
        ' EPS_PERSIST_LOG_MAX_FILES={LOG_MAX_FILES} < 1")',
        "        return",
        "    tracked = set()",
        "    try:",
        "        _git = subprocess.run(",
        '            ["git", "-C", str(root), "ls-files", "-z", "--", "logs"],',
        "            capture_output=True, timeout=10,",
        "            env={k: v for k, v in os.environ.items()",
        '                 if not k.startswith("GIT_")})',
        "        if _git.returncode == 0:",
        '            tracked = {t for t in _git.stdout.decode("utf-8", "replace")',
        '                       .split("\\0") if t}',
        "        else:",
        '            _say(f"[crash-persist] WARN worker_logs git-tracked exclude"',
        '                 f" unavailable (git rc={_git.returncode}); sweeping all")',
        "    except Exception as exc:",
        '        _say(f"[crash-persist] WARN worker_logs git-tracked exclude"',
        '             f" unavailable ({exc}); sweeping all")',
        "    n_tracked = 0",
        "    entries = []",
        "    for dirpath, dirnames, filenames in os.walk(logs_root):",
        "        dirnames[:] = [d for d in dirnames",
        '                       if d not in PRUNE and not (d.startswith("g") and'
        ' d.endswith("_dl"))]',
        "        for f in filenames:",
        "            p = Path(dirpath) / f",
        '            if tracked and "logs/" + p.relative_to(logs_root).as_posix() in tracked:',
        "                n_tracked += 1",
        "                continue",
        "            try:",
        "                st = p.stat()",
        "            except OSError:",
        "                continue",
        "            entries.append((st.st_mtime, st.st_size, p))",
        "    if n_tracked:",
        '        _say(f"[crash-persist] EXCLUDED {n_tracked} git-tracked file(s) from"',
        '             " worker_logs sweep (committed repo content, durable in git)")',
        "    if not entries:",
        '        _say("[crash-persist] SKIP worker_logs: empty after cache/git-tracked excludes")',
        "        return",
        "    entries.sort(reverse=True)  # newest first: the crashing worker wrote last",
        "    dropped = len(entries) - LOG_MAX_FILES",
        "    if dropped > 0:",
        '        _say(f"[crash-persist] SKIP {dropped} older worker log(s) beyond"',
        '             f" EPS_PERSIST_LOG_MAX_FILES={LOG_MAX_FILES}")',
        '    staged_root = Path(os.environ.get("EPS_PERSIST_LOG_STAGE_DIR",',
        '                                      "/tmp/eps-worker-logs"))',
        "    # a same-boot re-crash must not accumulate a PRIOR crash's staged files past",
        "    # the count bound; best-effort — staging below recreates what it needs.",
        "    shutil.rmtree(staged_root, ignore_errors=True)",
        "    n_staged = 0",
        "    for _, _, p in entries[:LOG_MAX_FILES]:",
        "        try:",
        "            if log_path and p.resolve() == Path(log_path).resolve():",
        '                _say(f"[crash-persist] SKIP worker_logs/{p.relative_to(logs_root)}:"',
        '                     " is the canonical workload.log")',
        "                continue",
        "            rel = p.relative_to(logs_root)",
        "            tmp = staged_root / rel",
        "            tmp.parent.mkdir(parents=True, exist_ok=True)",
        "            size = p.stat().st_size  # re-stat: may have grown/shrunk since the walk",
        '            with open(p, "rb") as fin, open(tmp, "wb") as fout:',
        "                if size > LOG_FILE_CAP:",
        "                    fin.seek(size - LOG_FILE_CAP)",
        '                    _say(f"[crash-persist] TAILED worker_logs/{rel}:"',
        '                         f" kept last {LOG_FILE_CAP} of {size} bytes")',
        "                fout.write(fin.read(LOG_FILE_CAP if size > LOG_FILE_CAP else size))",
        "            n_staged += 1",
        "        except Exception as exc:",
        '            _say(f"[crash-persist] FAILED staging worker log {p}: {exc}")',
        "    if n_staged == 0:",
        '        _say("[crash-persist] SKIP worker_logs: nothing staged")',
        "        return",
        "    try:",
        '        _say(f"[crash-persist] uploading dir worker_logs ({n_staged} files, one commit)")',
        "        api.upload_folder(folder_path=str(staged_root),",
        '                          path_in_repo=f"{dest}/worker_logs",',
        '                          repo_id=repo, repo_type="dataset")',
        '        _say("[crash-persist] uploaded dir worker_logs")',
        '        OK_UPLOADS["n"] += 1',
        "    except Exception as exc:",
        '        _say(f"[crash-persist] FAILED dir worker_logs: {exc}")',
        "_up_logs()",
        "# 2. partial artifacts — BOTH output conventions (#854: issue825 wrote its partials",
        "#    under data/issue_825/, structurally OUTSIDE the old eval_results-only sweep ->",
        "#    silent skip -> boot-disk surgery). The partial-DIRS sweep is these three named",
        "#    dirs (the # 1b. worker-logs tree is swept separately above) — not universal",
        "#    artifact discovery. Re-downloadable caches / stores are excluded at BOTH the",
        "#    top level and nested depths — under fnmatch the '**/'-prefixed forms do NOT",
        "#    match top-level paths, so both forms are listed; every skip is printed.",
        "def _dir_entries(local):",
        "    # (mtime, size, Path) per non-cache file — same PRUNE predicate as the old",
        "    # _dir_stats; the richer per-file return feeds the #1339 newest-first chunking.",
        "    entries = []",
        "    for dirpath, dirnames, filenames in os.walk(local):",
        "        dirnames[:] = [d for d in dirnames",
        '                       if d not in PRUNE and not (d.startswith("g") and'
        ' d.endswith("_dl"))]',
        "        for f in filenames:",
        "            p = Path(dirpath) / f",
        "            try:",
        "                st = p.stat()",
        "            except OSError:",
        "                continue",
        "            entries.append((st.st_mtime, st.st_size, p))",
        "    return entries",
        "# #1339: chunked partial-dir uploads. Incident #1090 fu5: eval_results_issue_1090",
        "# held 29,024 files / 14.4 MB and its ONE upload_folder commit processed ~31 s",
        "# server-side; the gateway timed out delivering the response and the client",
        "# logged FAILED on a commit that had LANDED. Dirs over the bound now upload as",
        "# newest-first staged batches of <= EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT files",
        "# (default 1000 — HF's own COMMIT_SIZE_SCALE per-commit ceiling), each ONE",
        "# upload_folder commit (never per-file upload_file — #664) with one bounded",
        "# retry, abandoning the dir loudly after EPS_PERSIST_DIR_BATCH_ABORT_STREAK",
        "# consecutive fully-failed batches. Repo paths stay byte-identical",
        "# ({dest}/{name}/<relpath>); a retried batch whose prior attempt actually",
        "# landed re-commits the same content at the same paths (content-identical",
        "# commit — idempotent effect). Staged copy volume is bounded by CAP (2 GiB)",
        "# on local disk; if CAP is ever raised far above 2 GiB, re-size this design.",
        'BATCH_MAX = _env_int("EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT", 1000)',
        'BATCH_ABORT_STREAK = _env_int("EPS_PERSIST_DIR_BATCH_ABORT_STREAK", 2)',
        "def _up_dir(local, name):",
        "    if not local.is_dir():",
        '        _say(f"[crash-persist] SKIP {name}: no such dir ({local})")',
        "        return",
        "    entries = _dir_entries(local)",
        "    n = len(entries)",
        "    size = sum(s for _, s, _ in entries)",
        "    if n == 0:",
        '        _say(f"[crash-persist] SKIP {name}: empty after cache excludes")',
        "        return",
        "    if size > CAP:",
        '        _say(f"[crash-persist] SKIP {name}: {size} bytes > cap {CAP}'
        ' (oversized; regenerate or reduce EPS_PERSIST_DIR_CAP_BYTES)")',
        "        return",
        "    if BATCH_MAX < 1 or n <= BATCH_MAX:",
        "        if BATCH_MAX < 1:",
        '            _say(f"[crash-persist] WARN EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT='
        '{BATCH_MAX} < 1; chunking disabled")',
        "        # at/under the bound: the pre-#1339 single-commit path — log lines +",
        "        # call shape byte-identical (pinned by the small-dir behavioral tests).",
        "        try:",
        '            _say(f"[crash-persist] uploading dir {name} ({n} files, {size} bytes)")',
        '            api.upload_folder(folder_path=str(local), path_in_repo=f"{dest}/{name}",',
        '                              repo_id=repo, repo_type="dataset", ignore_patterns=IGNORE)',
        '            _say(f"[crash-persist] uploaded dir {name}")',
        '            OK_UPLOADS["n"] += 1',
        "        except Exception as exc:",
        '            _say(f"[crash-persist] FAILED dir {name}: {exc}")',
        "        return",
        "    # over the bound: newest-first fixed-size staged batches (the #885 _up_logs",
        "    # sort idiom — a budget timeout / abort loses only the OLDEST files).",
        "    entries.sort(reverse=True)",
        "    nb = -(-n // BATCH_MAX)",
        '    _say(f"[crash-persist] chunking dir {name}: {n} files >"',
        '         f" EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT={BATCH_MAX};"',
        '         f" {nb} batches, newest-first")',
        '    stage_root = Path(os.environ.get("EPS_PERSIST_DIR_STAGE_DIR", "/tmp/eps-dir-batch"))',
        "    fail_streak = ok_batches = ok_files = 0",
        "    for bi in range(nb):",
        "        batch = entries[bi * BATCH_MAX:(bi + 1) * BATCH_MAX]",
        "        # a prior batch (or a same-boot re-crash) never leaks staged files in.",
        "        shutil.rmtree(stage_root, ignore_errors=True)",
        "        staged = 0",
        "        for _, _, p in batch:",
        "            try:",
        "                rel = p.relative_to(local)",
        "                tmp = stage_root / rel",
        "                tmp.parent.mkdir(parents=True, exist_ok=True)",
        "                shutil.copyfile(p, tmp)  # relpath preserved -> repo-path identity",
        "                staged += 1",
        "            except Exception as exc:",
        '                _say(f"[crash-persist] FAILED staging {name} file {p}: {exc}")',
        "        if staged == 0:",
        '            _say(f"[crash-persist] SKIP dir {name} batch {bi + 1}/{nb}: nothing staged")',
        "            continue",
        "        uploaded = False",
        "        for att in (1, 2):  # the #1151 _up_bundle bounded-retry mirror",
        "            try:",
        '                _say(f"[crash-persist] uploading dir {name} batch {bi + 1}/{nb}"',
        '                     f" ({staged} files, attempt {att}/2, one commit)")',
        "                api.upload_folder(folder_path=str(stage_root),",
        '                                  path_in_repo=f"{dest}/{name}",',
        '                                  repo_id=repo, repo_type="dataset",',
        "                                  ignore_patterns=IGNORE)",
        '                _say(f"[crash-persist] uploaded dir {name} batch {bi + 1}/{nb}")',
        "                uploaded = True",
        "                break",
        "            except Exception as exc:",
        '                _say(f"[crash-persist] FAILED dir {name} batch {bi + 1}/{nb}"',
        '                     f" attempt {att}/2: {exc}")',
        "                if att == 1:",
        '                    time.sleep(max(0, _env_int("EPS_PERSIST_RETRY_BACKOFF_S", 10)))',
        "        if uploaded:",
        "            fail_streak = 0",
        "            ok_batches += 1",
        '            OK_UPLOADS["n"] += 1',
        "            ok_files += staged",
        "        else:",
        "            fail_streak += 1",
        "            if fail_streak >= BATCH_ABORT_STREAK:",
        '                _say(f"[crash-persist] ABORT dir {name}: {fail_streak} consecutive"',
        '                     f" batch failures; {nb - bi - 1} batch(es) unsent")',
        "                break",
        "    shutil.rmtree(stage_root, ignore_errors=True)",
        '    _say(f"[crash-persist] dir {name}: {ok_batches}/{nb} batches uploaded"',
        '         f" ({ok_files}/{n} files)")',
        "for local, name in (",
        '    (root / "eval_results" / f"issue_{issue}", f"eval_results_issue_{issue}"),',
        '    (root / "data" / f"issue_{issue}", f"data_issue_{issue}"),',
        '    (root / "data" / f"issue{issue}", f"data_issue{issue}"),',
        "):",
        "    _up_dir(local, name)",
        "# per-crash timestamped log copy — LAST among the artifacts (see the note above),",
        "# staged into the FINAL bundle (#1151: one upload_folder commit, no retry).",
        'final_stage = Path(os.environ.get("EPS_PERSIST_FINAL_STAGE_DIR", "/tmp/eps-crash-final"))',
        "shutil.rmtree(final_stage, ignore_errors=True)",
        "if log_path and Path(log_path).is_file():",
        '    _stage_into(final_stage, f"workload_{stamp}.log", log_path)',
        '_say("[crash-persist] DONE")',
        "# 3. the audit transcript FINAL — staged into the final bundle AFTER the DONE",
        "#    line, so its uploaded copy records every earlier upload/skip line. Its",
        "#    presence on HF proves the persist ran to completion; its ABSENCE proves a",
        "#    killed persist (now co-signaled by a standing `attempted` eps/persist",
        "#    breadcrumb, #1151). The serial console is unreadable post-DELETE (#640),",
        "#    so this is the durable audit.",
        "if Path(transcript).is_file():",
        '    _stage_into(final_stage, "crash_persist_transcript.log", transcript)',
        '_up_bundle(final_stage, "final", retry=False)',
        "# #1343: eps/persist=ok honesty gate. Incident #1315: every upload FAILED",
        "# under the logged-never-raised guards, the python exited 0, and the",
        "# breadcrumb read ok while issue1315_partial/ 404'd. ONE positive existence",
        "# probe on the transcript (the #1339 lesson: a client-side FAILED can mask",
        "# a LANDED commit, so a positive probe beats trusting client returns);",
        "# a client-confirmed upload_folder RETURN is the OR-side fallback evidence",
        "# (#1339's reliable direction — also covers probe lag/transport). Guarded:",
        "# the probe never raises past the persist; bounded: one HEAD at hub's 10s",
        "# default request timeout, inside the surrounding `timeout 300`. These",
        "# VERIFY lines reach serial + the transcript FILE but post-date the",
        "# transcript UPLOAD by construction — the breadcrumb value is the durable",
        "# verify record.",
        "_verified = False",
        "try:",
        "    _verified = bool(api.file_exists(",
        '        repo, f"{dest}/crash_persist_transcript.log", repo_type="dataset"))',
        '    _say(f"[crash-persist] VERIFY transcript on hub: {_verified}")',
        "except Exception as exc:",
        '    _say(f"[crash-persist] VERIFY probe FAILED (treated as unverified): {exc}")',
        'if not _verified and OK_UPLOADS["n"] == 0:',
        '    _say("[crash-persist] VERIFY-FAIL: zero uploads verifiably succeeded'
        ' -> rc 3 (failed_uploads)")',
        "    sys.exit(3)",
        '_n_ok = OK_UPLOADS["n"]',
        '_say(f"[crash-persist] VERIFY-OK: probe={_verified} client_confirmed={_n_ok}")',
        "EPS_PERSIST_PY",
        # #1151: capture the `cd && timeout uv run python` compound's rc INSIDE
        # the subshell (set +e is global from the trap's first action, so a
        # failing compound does not abort; its rc is capturable): 0 = the
        # persist python exited clean AND the #1343 verify gate passed (the
        # transcript existence probe read True, or >=1 client-confirmed
        # upload_folder return — NOT "all artifacts landed": per-upload
        # failures are still logged, never raised; the gate verifies
        # at-least-one, and the transcript stays the per-upload audit),
        # 3 = the verify gate FAILED (zero uploads verifiably succeeded ->
        # "failed_uploads", #1315), 124 = the 300s timeout killed it, 127 = uv
        # missing, 1 = cd short-circuit OR a python top-level failure.
        '  _uprc=$?; { echo "$_uprc" >"${EPS_CRASH_PERSIST_RC:-/tmp/eps-crash-persist.rc}"; }'
        " 2>/dev/null || true;",
        # #854 eager bounded streamer, replacing `| cut -c1-2000 | tail -n 20`:
        # `tail` buffered everything to EOF, so a killed/skipped persist left
        # ZERO serial evidence — the diagnosability gap that let a coverage-gap
        # skip be misdiagnosed as a poweroff race on #825. The reader forwards
        # each line to fd 3 AS IT ARRIVES, caps line length at 2000 chars,
        # stops PRINTING after 120 lines but keeps READING to EOF — an early
        # pipe close would SIGPIPE-kill the python mid-upload, the exact loss
        # this path exists to prevent. That read-to-EOF property is pinned by
        # the string assert in test_render_startup_script_persist_streams_eagerly
        # (the behavioral heredoc test runs the python WITHOUT this bash
        # streamer, so it does not exercise SIGPIPE protection). The
        # `|| [ -n "$_l" ]` keeps a trailing unterminated line. Print-cap
        # sizing (#885, resized #1339): worst realistic chunked case ~= 16
        # base persist lines + ~43 worker-log lines (the #885 worst case) +
        # a chunk header + 30 batches x 2 lines + a summary ~= 122 — just
        # over the previous 120 cap (itself doubled from 60 at #885), so
        # raised to 200 (400 KB max at 2000 chars/line, well inside the
        # ~1 MB GCE serial buffer); the durable transcript is unaffected
        # either way (it has no line cap). A pathological
        # all-three-dirs-chunked crash may still truncate the serial view —
        # acceptable, the transcript is the audit of record (#854).
        '  ) 2>&1 | { _n=0; while IFS= read -r _l || [ -n "$_l" ]; do _n=$((_n + 1));',
        '    if [ "$_n" -le 200 ]; then'
        " { printf '%s\\n' \"${_l:0:2000}\" >&3; } 2>/dev/null || true; fi;",
        "  done; } 2>/dev/null || true;",
        # #1151: final-status breadcrumb from the rc-file readback (the
        # pipeline's own $? is the STREAMER's, so the file is the only rc
        # channel). A MISSING rc file deliberately writes NOTHING — the
        # standing `attempted` IS the killed-mid-persist signal; writing a
        # guessed value here would destroy that discriminator. <=3 metadata
        # writes per crash total, inside the 3/s burst + 10/min
        # guest-attribute caps.
        '  _prc="$(cat "${EPS_CRASH_PERSIST_RC:-/tmp/eps-crash-persist.rc}" 2>/dev/null || true)";',
        '  if [ -n "$_prc" ]; then case "$_prc" in',
        '    (0)   _eps_persist_status "ok" ;;',
        '    (3)   _eps_persist_status "failed_uploads" ;;',
        '    (124) _eps_persist_status "timeout" ;;',
        '    (*)   _eps_persist_status "failed_rc${_prc}" ;;',
        "  esac; fi;",
        "}",
        # In-VM REACHABILITY watchdog (#669) — NOT a liveness watchdog
        # (systemd #21083: a liveness /dev/watchdog keeps getting fed on a
        # wedged-but-RUNNING VM whose guest networking died, so it never
        # catches this class). The #667 failure mode: systemd-networkd loses
        # its DHCPv4 lease under load and does NOT retry the renew, so the NIC
        # loses connectivity PERMANENTLY while the VM stays RUNNING — the
        # poller reads eps/phase frozen at a non-terminal value forever and
        # neither GCP->RunPod failover path fires. This daemon probes ACTUAL
        # external reachability (the GCE metadata server AND an internet
        # endpoint) on a 30s cadence; on 10 consecutive failures of BOTH
        # (~5 min sustained loss) it writes eps/phase=wedged (best-effort) so
        # the poller maps the resulting TERMINATED to terminal_wedged_terminated
        # (the conservative #669 failover phase), then forces a clean
        # TERMINATED via the shutdown -> poweroff -> halt ladder (a VM wedged
        # enough that systemd-shutdown itself hangs still hard-stops). When the
        # phase write cannot land (fully network-dead), the poller-side
        # frozen-phase wedge detector is the independent backstop. Launched
        # AFTER the exec >> redirect so its [eps-watchdog] lines land in the
        # workload log (not the metadata-runner pipe — #607/#491 discipline)
        # and reaped in the clean-exit tail, or KILLED AT EXIT-TRAP ENTRY on a
        # crash (#854) so it cannot race the crash persist.
        "_eps_reachability_watchdog() {",
        "  local interval=30 threshold=10 fails=0 meta_ok ext_ok",
        "  while true; do",
        '    sleep "$interval";',
        # -m 5: 5s connect+xfer cap so a hung NIC can't block the loop. The two
        # endpoints are probed SEPARATELY and the fail counter resets if EITHER
        # one answers — it increments ONLY when BOTH fail (#669 code-review r1
        # blocker: the prior ``curl A && curl B`` conjunction incremented on a
        # single-endpoint failure, so a transient HF outage with healthy
        # metadata could drive a HEALTHY VM to ``wedged`` and trigger a spurious
        # RunPod failover — the exact false positive this watchdog exists to
        # avoid). The metadata server (169.254.169.254) is link-local — reachable
        # even with a dead DHCP lease IF routing survives — and the external HF
        # probe catches the full-network-loss case; only the BOTH-fail
        # conjunction is true network loss.
        "    meta_ok=0; ext_ok=0;",
        "    if curl -sf -m 5 -H 'Metadata-Flavor: Google'"
        " http://169.254.169.254/computeMetadata/v1/instance/id >/dev/null 2>&1; then"
        " meta_ok=1; fi;",
        "    if curl -sf -m 5 https://huggingface.co/ -o /dev/null 2>&1; then ext_ok=1; fi;",
        '    if [ "$meta_ok" -eq 1 ] || [ "$ext_ok" -eq 1 ]; then',
        "      fails=0;",
        "    else",
        "      fails=$((fails + 1));",
        '      { echo "[eps-watchdog] BOTH reachability probes FAILED ($fails/$threshold)"; }'
        " 2>/dev/null || true;",
        '      if [ "$fails" -ge "$threshold" ]; then',
        '        { echo "[eps-watchdog] sustained network loss -> wedged;'
        ' forcing clean TERMINATED (#669)"; } 2>/dev/null || true;',
        # Write eps/phase=wedged FIRST (best-effort): when the network is
        # healthy enough to land it, the poller maps TERMINATED+wedged ->
        # terminal_wedged_terminated; when it CANNOT land (fully dead), the
        # poller-side wedge detector is the backstop.
        "        _eps_phase wedged 2>/dev/null || true;",
        "        shutdown -h now 2>/dev/null || poweroff -f 2>/dev/null || halt -f;",
        "        return 0;",
        "      fi;",
        "    fi;",
        "  done",
        "}",
        # === OOM-detection helper (#750, incident #744) ===
        # The bash chain's rc-propagation has known weakness modes under
        # ``set -e`` + ``&&`` + SIGKILL: on eps-issue-744 a workload python
        # OOM-killed by the kernel (systemd: "Failed with result 'oom-kill'")
        # still returned rc=0 to the parent startup-script, so the success
        # path published ``_eps_phase done`` 2s after the kill and the run
        # advanced to verifying with zero usable artifacts. The kernel's
        # cgroup-v2 ``memory.events.local`` ``oom_kill`` counter is incremented
        # whenever an OOM-killer hit THIS cgroup (``.local`` = this cgroup only,
        # NOT a hierarchical aggregation of descendants), REGARDLESS of what rc
        # the bash chain reported — an authoritative cross-check the chain
        # cannot mask.
        #
        # PATH DERIVATION (the #750 v2 defect this fixes): ``memory.events`` /
        # ``memory.events.local`` exist on NON-ROOT cgroups only (kernel
        # cgroup-v2 docs, https://docs.kernel.org/admin-guide/cgroup-v2.html);
        # the unified-hierarchy ROOT ``/sys/fs/cgroup`` does NOT expose them
        # (empirically: ``cat /sys/fs/cgroup/memory.events`` -> ENOENT). The
        # workload's OWN cgroup dir is the ``0::`` unified-hierarchy line of
        # ``/proc/self/cgroup`` (format ``0::/system.slice/...``), prefixed by
        # the mount root: CGROUP_DIR=/sys/fs/cgroup$(that path). The #744
        # workload sat in ``/system.slice/google-startup-scripts.service`` — a
        # descendant — so only the derived path sees its oom_kill.
        #
        # COUNTER SEMANTICS: ``oom_kill`` is cumulative-since-cgroup-creation,
        # so an absolute "== 0" test can false-fire on a non-fresh cgroup; the
        # caller therefore reads a BASELINE before the workload and a FINAL
        # after, and fires only on an INCREASE (see the baseline + guard
        # entries below). This helper just emits the current integer.
        #
        # The helper runs under ``set -euo pipefail``; every line is
        # ``set -e``-safe — the ``awk`` runs only inside an ``if [ -r ... ]``
        # condition-guarded ``&&`` whose result is captured by an assignment,
        # and the function always reaches an explicit ``echo``/``return``.
        "_eps_oom_count() {",
        # 0:: line of /proc/self/cgroup = the cgroup-v2 unified-hierarchy path
        # for this process. ${x#0::} strips the literal prefix; a host with no
        # 0:: line yields an empty suffix -> _dir=/sys/fs/cgroup (root), whose
        # memory.events.local is absent -> the [ -r ] guard fails -> echo 0.
        "  local _rel _dir _n=0",
        '  _rel="$(awk -F: \'$1=="0"{print $3; exit}\' /proc/self/cgroup 2>/dev/null || true)"',
        '  _dir="/sys/fs/cgroup${_rel}"',
        # Read the LOCAL counter (this cgroup only). awk pulls just the integer
        # value of the oom_kill line; missing line / unreadable file -> _n stays
        # 0. The assignment consumes awk's exit status, so set -e never fires.
        '  if [ -r "$_dir/memory.events.local" ]; then',
        '    _n="$(awk \'$1=="oom_kill"{print $2; exit}\''
        ' "$_dir/memory.events.local" 2>/dev/null || true)"',
        "  fi",
        # Normalize a non-numeric / empty read (e.g. awk printed nothing) to 0
        # so the caller's numeric -gt comparison is always well-formed.
        '  case "$_n" in (*[!0-9]*|"") _n=0 ;; esac',
        '  echo "$_n"',
        "}",
        # === Done-grace self-poweroff helpers (#935) ===
        # The CLEAN-exit path deliberately leaves the VM RUNNING so the
        # poller can drain the completion/gate sentinels — but when the
        # orchestrator/poller is DEAD nothing tears the VM down until the
        # next dispatch's #908 reclaim or the 09:37 janitor (~19-24 h
        # worst case of idle GPU billing; the #763 done-zombie). The
        # success tail therefore calls _eps_done_grace_poweroff as the
        # LAST line of the script: a bounded countdown (EPS_DONE_GRACE
        # seconds; 0 disables) that aborts when (a) an operator touched
        # $EPS_DONE_KEEPALIVE_PATH (the #491 relaunch-runbook escape
        # hatch), or (b) eps/phase left "done" — a sanctioned same-VM
        # relaunch re-published eps/phase=workload (REQUIRED by the #908
        # zombie predicates), so the relaunch owns the VM now. An EMPTY
        # metadata read CONTINUES the countdown (fail toward bounding
        # billing: a down metadata server also means no relaunch could
        # have re-published). At expiry the slim persist helper below
        # best-effort uploads the UNDRAINED sentinel set to HF, then the
        # poweroff fires UNCONDITIONALLY — a persist failure never
        # re-opens the billing zombie. #607 discipline throughout: every
        # echo guarded, every step || true, the loop bounded by the
        # counter.
        #
        # _eps_persist_done_sentinels — slim clean-path sibling of
        # _eps_persist_diagnostics (deliberately NOT reused: the crash
        # helper targets issue<N>_partial/ — rc=0 entries there would
        # pollute crash forensics — and its $WORKLOAD_ROOT-relative sweep
        # structurally misses the /workspace/logs sentinel dir this
        # helper must protect). Stages the completion sentinel + any
        # undrained /workspace/logs/issue-<N>-*.json sentinels + a
        # <=5 MiB workload-log tail + a small done_report.json, uploads
        # them in ONE upload_folder commit to issue<N>_done/<attempt_id>/
        # (never a per-file upload_file loop — the #664 504-storm gotcha)
        # with ONE in-heredoc retry after a short backoff, then uploads
        # the persist transcript LAST via a single upload_file (#854
        # pattern — its presence proves the persist completed; exactly 2
        # upload calls, not a loop). Afterwards it best-effort PUTs
        # eps/done_persist=ok|failed on a SEPARATE guest-attribute key
        # (NEVER eps/phase — the poll's TERMINATED+done classification
        # and the #908 zombie predicates key on eps/phase) so a
        # STOP-outcome instance durably records whether the sentinel set
        # reached HF. Bounded (timeout 120) + fully guarded: a hung or
        # failed upload can never delay the poweroff that bounds billing.
        "_eps_persist_done_sentinels() {",
        "  local _ddest _dps;",
        '  if [ -z "${EPS_HF_DATA_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then',
        '    { echo "[done-grace] SKIP persist: EPS_HF_DATA_REPO or HF_TOKEN unset"; }'
        " 2>/dev/null || true;",
        "    return 0;",
        "  fi;",
        '  _ddest="issue${EPS_ISSUE:-0}_done/${EPS_ATTEMPT_ID:-unknown}";',
        '  { echo "[done-grace] persisting undrained sentinels to'
        ' ${EPS_HF_DATA_REPO}/${_ddest} (#935)"; } 2>/dev/null || true;',
        '  rm -f "${EPS_DONE_PERSIST_STATUS:-/tmp/eps-done-persist.status}" 2>/dev/null || true;',
        '  ( export PATH="${HOME:-/root}/.local/bin:$PATH" HF_HUB_DISABLE_PROGRESS_BARS=1;'
        ' cd "${WORKLOAD_ROOT:-/}" 2>/dev/null'
        " && timeout 120 uv run python - \"$_ddest\" <<'EPS_DONE_PERSIST_PY'",
        "import datetime, glob, json, os, shutil, sys, time",
        "from pathlib import Path",
        "from huggingface_hub import HfApi",
        "dest = sys.argv[1]",
        'repo = os.environ["EPS_HF_DATA_REPO"]',
        'issue = os.environ.get("EPS_ISSUE", "0")',
        'attempt = os.environ.get("EPS_ATTEMPT_ID", "unknown")',
        'grace = os.environ.get("EPS_DONE_GRACE", "0")',
        'log_path = os.environ.get("EPS_LOG_PATH", "")',
        'sentinel = os.environ.get("EPS_SENTINEL_PATH", "")',
        "# Env-overridable roots/paths: test isolation on the shared VM (the #885",
        "# EPS_PERSIST_LOG_STAGE_DIR precedent); production uses the defaults.",
        'logs_dir = os.environ.get("EPS_DONE_LOGS_DIR", "/workspace/logs")',
        'stage = Path(os.environ.get("EPS_DONE_PERSIST_STAGE_DIR", "/tmp/eps-done-persist"))',
        'status_path = os.environ.get("EPS_DONE_PERSIST_STATUS", "/tmp/eps-done-persist.status")',
        'transcript = os.environ.get("EPS_DONE_PERSIST_TRANSCRIPT", "/tmp/eps-done-persist.log")',
        "def _say(msg):",
        "    # printed to the workload log (post-redirect stdout) AND teed into the",
        "    # transcript uploaded LAST — the durable skip-vs-kill audit (#854 pattern).",
        "    print(msg, flush=True)",
        "    try:",
        '        with open(transcript, "a") as fh:',
        '            fh.write(msg + "\\n")',
        "    except OSError:",
        "        pass",
        '_say(f"[done-persist] BEGIN repo={repo} dest={dest}")',
        "shutil.rmtree(stage, ignore_errors=True)",
        "stage.mkdir(parents=True, exist_ok=True)",
        "def _stage(src, rel):",
        "    try:",
        "        p = Path(src)",
        "        if not p.is_file():",
        '            _say(f"[done-persist] SKIP {rel}: no such file ({src})")',
        "            return",
        "        out = stage / rel",
        "        out.parent.mkdir(parents=True, exist_ok=True)",
        "        shutil.copyfile(p, out)",
        '        _say(f"[done-persist] staged {rel} ({out.stat().st_size} bytes)")',
        "    except Exception as exc:",
        '        _say(f"[done-persist] FAILED staging {rel}: {exc}")',
        "if sentinel:",
        '    _stage(sentinel, "sentinel.json")',
        "else:",
        '    _say("[done-persist] SKIP sentinel.json: EPS_SENTINEL_PATH unset")',
        "# Any UNDRAINED pod-contract sentinels (epm:results payloads, gate sentinels)",
        "# — the exact loss the grace-expiry persist exists to prevent. The keepalive",
        "# escape-hatch file carries NO .json suffix, so this glob can never stage it.",
        'for p in sorted(glob.glob(f"{logs_dir}/issue-{issue}-*.json")):',
        '    _stage(p, f"logs_sentinels/{Path(p).name}")',
        "TAIL_CAP = 5 * 1024**2  # text rides the non-LFS Hub path; the tail keeps the END",
        "try:",
        "    if log_path and Path(log_path).is_file():",
        "        size = Path(log_path).stat().st_size",
        "        kept = min(size, TAIL_CAP)",
        '        with open(log_path, "rb") as fin, open(stage / "workload_tail.log", "wb") as fo:',
        "            if size > TAIL_CAP:",
        "                fin.seek(size - TAIL_CAP)",
        "            fo.write(fin.read(TAIL_CAP))",
        '        _say(f"[done-persist] staged workload_tail.log (last {kept} of {size} bytes)")',
        "    else:",
        '        _say(f"[done-persist] SKIP workload_tail.log: EPS_LOG_PATH unset or missing'
        ' ({log_path!r})")',
        "except Exception as exc:",
        '    _say(f"[done-persist] FAILED staging workload_tail.log: {exc}")',
        "try:",
        '    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")',
        "    report = {",
        '        "issue": issue,',
        '        "attempt_id": attempt,',
        '        "grace_s": grace,',
        '        "ts": ts,',
        '        "kind": "gcp-done-grace-sentinel-persist",',
        "    }",
        '    (stage / "done_report.json").write_text(json.dumps(report) + "\\n")',
        "except Exception as exc:",
        '    _say(f"[done-persist] FAILED writing done_report.json: {exc}")',
        "api = HfApi()",
        "ok = False",
        "# ONE upload_folder commit (never a per-file loop — #664) + ONE retry after a",
        "# short backoff; both attempts sit inside the same bash-side timeout budget.",
        "for i in (1, 2):",
        "    try:",
        '        _say(f"[done-persist] uploading dir (attempt {i}/2, one commit) -> {dest}")',
        "        api.upload_folder(folder_path=str(stage), path_in_repo=dest,",
        '                          repo_id=repo, repo_type="dataset")',
        '        _say("[done-persist] uploaded")',
        "        ok = True",
        "        break",
        "    except Exception as exc:",
        '        _say(f"[done-persist] FAILED upload attempt {i}/2: {exc}")',
        "        if i == 1:",
        "            try:",
        '                _b = int(os.environ.get("EPS_DONE_PERSIST_RETRY_BACKOFF_S", "10"))',
        "            except ValueError:",
        "                _b = 10",
        "            time.sleep(max(0, _b))",
        "_say(f\"[done-persist] DONE status={'ok' if ok else 'failed'}\")",
        "try:",
        '    Path(status_path).write_text("ok" if ok else "failed")',
        "except OSError:",
        "    pass",
        "# The transcript LAST via a single upload_file — its presence on HF proves",
        "# the persist ran to completion (#854 pattern); exactly 2 upload calls total.",
        "try:",
        "    if Path(transcript).is_file():",
        "        api.upload_file(path_or_fileobj=transcript,",
        '                        path_in_repo=f"{dest}/persist_transcript.log",',
        '                        repo_id=repo, repo_type="dataset")',
        "except Exception as exc:",
        '    print(f"[done-persist] FAILED transcript upload: {exc}", flush=True)',
        "EPS_DONE_PERSIST_PY",
        "  ) || true;",
        # The ok|failed breadcrumb: read back the status file the heredoc wrote
        # (missing/killed persist reads as failed) and best-effort PUT it on
        # the SEPARATE eps/done_persist guest-attribute key. Durable on the
        # STOP outcome; lost with the instance on DELETE (accepted, #935).
        '  _dps="$(cat "${EPS_DONE_PERSIST_STATUS:-/tmp/eps-done-persist.status}" 2>/dev/null'
        ' || true)";',
        '  case "$_dps" in (ok) ;; (*) _dps="failed" ;; esac;',
        '  curl -fsS -X PUT -H "Metadata-Flavor: Google" --data "$_dps"'
        ' "http://metadata.google.internal/computeMetadata/v1/'
        'instance/guest-attributes/eps/done_persist" >/dev/null 2>&1 || true;',
        '  { echo "[done-grace] persist status=$_dps (eps/done_persist breadcrumb'
        ' best-effort)"; } 2>/dev/null || true;',
        "  return 0;",
        "}",
        # _eps_done_grace_poweroff — the bounded countdown. Foreground (not a
        # bg daemon): nothing consumes startup-script completion (#491 proved
        # scripts may run indefinitely), the reachability watchdog is already
        # reaped on this path, and foreground keeps the ordering trivially
        # auditable. Tick = 60 s FIXED: guest-attribute queries are
        # rate-limited to 10/min per VM (GCP manage-guest-attributes docs),
        # so 1 GET/min sits 6x under the cap — the tick must never shrink
        # below ~6 s. Each iteration costs up to 65 s wall (sleep 60 +
        # curl -m 5) while `waited` advances 60, so the true wall bound is
        # ~grace*(65/60) + persist(<=120 s) + shutdown. Once aborted (the
        # relaunch case) the countdown does NOT re-arm for a second
        # workload's done — the relaunch operator owns teardown per the #491
        # runbook (documented residual).
        "_eps_done_grace_poweroff() {",
        '  local waited=0 tick=60 ph="" _grace="${EPS_DONE_GRACE:-0}";',
        '  case "$_grace" in (*[!0-9]*|""|0)',
        '    { echo "[done-grace] disabled (EPS_DONE_GRACE=${EPS_DONE_GRACE:-})"; }'
        " 2>/dev/null || true;",
        "    return 0 ;;",
        "  esac;",
        '  { echo "[done-grace] armed: self-poweroff in ${_grace}s unless drained/aborted'
        ' (#935)"; } 2>/dev/null || true;',
        '  while [ "$waited" -lt "$_grace" ]; do',
        '    sleep "$tick" 2>/dev/null || true;',
        "    waited=$((waited + tick));",
        "    # Escape hatch 1: operator keepalive (the #491 relaunch runbook).",
        '    if [ -e "${EPS_DONE_KEEPALIVE_PATH:-/nonexistent}" ]; then',
        '      { echo "[done-grace] keepalive present - aborting self-poweroff"; }'
        " 2>/dev/null || true;",
        "      return 0;",
        "    fi;",
        "    # Escape hatch 2: a sanctioned same-VM relaunch re-published",
        "    # eps/phase=workload (REQUIRED by the #908 zombie predicates) -",
        "    # the relaunch owns the VM now. An EMPTY read CONTINUES the",
        "    # countdown: it must never abort toward keep-billing (a down",
        "    # metadata server also means no relaunch could have re-published).",
        "    ph=\"$(curl -fsS -m 5 -H 'Metadata-Flavor: Google'"
        " 'http://metadata.google.internal/computeMetadata/v1/"
        "instance/guest-attributes/eps/phase' 2>/dev/null || true)\";",
        '    if [ -n "$ph" ] && [ "$ph" != "done" ]; then',
        '      { echo "[done-grace] eps/phase=$ph - a relaunch owns the VM; aborting'
        ' self-poweroff"; } 2>/dev/null || true;',
        "      return 0;",
        "    fi;",
        "  done;",
        "  # Best-effort persist FIRST (ONE in-heredoc retry; writes the",
        "  # eps/done_persist breadcrumb), then the UNCONDITIONAL ladder -",
        "  # a persist failure never re-opens the billing zombie (#935).",
        "  _eps_persist_done_sentinels || true;",
        '  { echo "[done-grace] grace expired (${_grace}s) - powering off to bound billing'
        ' (#935)"; } 2>/dev/null || true;',
        "  shutdown -h now 2>/dev/null || poweroff -f 2>/dev/null || halt -f;",
        "}",
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
        #
        # #607 hardening — the trap body must be NON-ABORTING: with the
        # PIPE handler active, a write to a closed runner pipe inside the
        # trap is an ordinary failing command; under ``set -e`` an
        # UNGUARDED echo would abort the trap body BEFORE ``_eps_phase
        # failed`` + ``shutdown -h now`` — re-opening the zombie exactly
        # on the path that must bound billing. Hence ``set +e`` as the
        # first action after capturing rc, plus every diagnostic write
        # guarded. ``${EPS_LOG_PATH:-}``: the trap is installed before
        # EPS_LOG_PATH exists; a bare reference in an early-failure trap
        # would error mid-trap under ``set -u`` and SKIP the shutdown.
        # ``cut -c1-2000`` bounds every serial line even if a giant line
        # somehow reached the log tail.
        "trap 'rc=$?; set +e;"
        ' if [ "$rc" -ne 0 ]; then'
        # #1055: POSITIVE-EVIDENCE classification — the workload writes
        # $EPS_DELIVERABLES_OK_PATH ONLY after its final upload+verify PASS,
        # so its presence at trap time proves the declared deliverables are
        # complete on HF and this non-zero exit is a finalize/tail failure,
        # not a data-losing crash. NEVER keyed on rc value or crash timing
        # (#1004 coherence: rc stays non-zero, diagnostics still run, the
        # poweroff that bounds billing is untouched, and literal `done` is
        # never published on this path). The -n guard is belt-and-suspenders
        # ([ -f "" ] is false); ${:-} keeps the trap safe under set -u when
        # it fires before the export.
        ' if [ -n "${EPS_DELIVERABLES_OK_PATH:-}" ] && [ -f "${EPS_DELIVERABLES_OK_PATH:-}" ];'
        " then"
        ' { echo "[startup-script] FINALIZE-FAILED rc=$rc — deliverables sentinel present;'
        ' artifacts complete on HF; powering off (#1055)"; } 2>/dev/null || true;'
        " _eps_phase finalize_failed_artifacts_ok;"
        " else"
        ' { echo "[startup-script] FAILED rc=$rc — powering off to bound billing"; }'
        " 2>/dev/null || true;"
        " _eps_phase failed;"
        " fi;"
        # #854: reap the reachability watchdog — the only OTHER in-guest
        # poweroff actor — at trap ENTRY, before the persist, so nothing can
        # power the VM off mid-upload; the trap itself guarantees the
        # billing-bounding shutdown below. Guarded: an unset PID (crash
        # before the watchdog launch) / already-dead daemon is a no-op.
        # (#1055: the watchdog reap + log tail + diagnostics + poweroff are
        # the SHARED tail — they run on BOTH sentinel arms, outside the
        # inner fi, ordering unchanged.)
        ' { kill "${EPS_WATCHDOG_PID:-}" 2>/dev/null; } || true;'
        ' if [ -n "${EPS_LOG_PATH:-}" ]; then'
        ' { tail -n 40 "$EPS_LOG_PATH" 2>/dev/null | cut -c1-2000 >&3; } 2>/dev/null || true; fi;'
        # #658: persist the workload log + partial artifacts to HF BEFORE
        # the DELETE-on-shutdown destroys the boot disk. Fully guarded +
        # time-bounded inside the helper, so it can never delay the
        # poweroff that bounds billing.
        ' _eps_persist_diagnostics "$rc";'
        " shutdown -h now; fi' EXIT",
        "_eps_phase startup",
        # #1151: boot-time clear of the eps/persist crash-persist breadcrumb —
        # guest attributes survive reboots of the SAME instance, so a
        # salvage-relaunch second boot would otherwise inherit a PRIOR
        # crash's final value and corrupt the standing-`attempted`
        # discriminator. Fail-soft DELETE (the guest-attributes metadata
        # endpoint accepts DELETE); -m 5 so a wedged metadata server never
        # delays boot.
        'curl -fsS -m 5 -X DELETE -H "Metadata-Flavor: Google"'
        ' "http://metadata.google.internal/computeMetadata/v1/'
        'instance/guest-attributes/eps/persist" >/dev/null 2>&1 || true',
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
        # #1055: positive-evidence deliverables sentinel — the workload
        # stamps this path ONLY after its final upload+verify PASS; the EXIT
        # trap then classifies a non-zero exit as finalize_failed_artifacts_ok
        # instead of failed. Fail-open: a workload that never writes it keeps
        # today's failed path byte-for-byte.
        f"export EPS_DELIVERABLES_OK_PATH={shlex.quote(deliverables_ok_abs)}",
        # #1055: stale-evidence hygiene — a re-booted instance (manual
        # `instances start` re-runs the startup script with the SAME
        # attempt_id + preserved disk) must not inherit a prior boot's
        # deliverables evidence. rm -f is a no-op pre-clone.
        'rm -f "$EPS_DELIVERABLES_OK_PATH" 2>/dev/null || true',
        # Crash-diagnostics target (#658): the EXIT trap uploads the
        # workload log + partial artifacts here BEFORE the
        # instance-termination-action=DELETE destroys the boot disk, so a
        # GCP crash is debuggable + partial progress is recoverable.
        f"export EPS_HF_DATA_REPO={shlex.quote(config.hf_data_repo)}",
        # Done-grace self-poweroff constants (#935): the render-time-resolved
        # countdown (env knob EPS_GCP_DONE_POWEROFF_GRACE_SECONDS; 0 disables)
        # + the operator keepalive escape hatch (NO .json suffix — see the
        # render-body comment).
        f"export EPS_DONE_GRACE={done_grace}",
        f"export EPS_DONE_KEEPALIVE_PATH={shlex.quote(keepalive_path)}",
        # Fast HF Hub uploads (#745) — STATIC DEFAULT export. Placed in the
        # env-export block BEFORE the output redirect + secrets fetch so that
        # (a) the workload (both the hydra and workload_cmd branches, which
        # follow `*workload_block` after `uv sync`) inherits it, and (b) the
        # crash-persist subshell (`_eps_persist_diagnostics`, the EXIT-trap
        # HfApi.upload_folder) — which runs with no `load_dotenv` and inherits
        # the parent shell env — gets it on any crash after this point.
        # HF_XET_HIGH_PERFORMANCE is the PRIMARY accelerator (the project repos
        # use the Xet backend); HF_HUB_ENABLE_HF_TRANSFER is the orthogonal LFS
        # accelerator (hf_transfer is a hard dep). The passthrough keys in
        # STARTUP_PASSTHROUGH_ENV_KEYS are the OVERRIDE channel — a forwarded
        # dispatch-process =0 / HF_HUB_DISABLE_XET=1 is fetched LATER by the
        # secrets-fetch stanza and supersedes these static defaults. That
        # stanza is DEFAULT-PRESERVING (#745, round 2): it only re-exports a
        # key when the metadata fetch is NON-empty, so an ABSENT override (the
        # common case) leaves THESE static defaults standing instead of
        # clobbering them to "" — see the secrets_fetch_lines comment above.
        'export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"',
        'export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"',
        "",
        # Output redirect (#607): everything from here down — secrets
        # fetch, preflight, clone, uv sync, the workload itself — writes
        # to the log file, never the runner pipe. ``exec >>`` (append,
        # whole-script) covers bootstrap output too and cannot be
        # bypassed by a future block edit; append preserves prior-boot /
        # relaunch content. The two ``# === ... ===`` marker comment
        # lines are LOAD-BEARING: the local integration tests slice the
        # runnable prelude at the end marker.
        "# === Output redirect (#607): never stream workload output through the",
        "# metadata runner (bufio.Scanner token-too-long kill, incident #491) ===",
        f"export EPS_LOG_PATH={shlex.quote(log_path)}",
        f"mkdir -p {shlex.quote(log_dir)}",
        '{ echo "[startup-script] redirecting all further output to $EPS_LOG_PATH"; }'
        " 2>/dev/null || true",
        'exec >>"$EPS_LOG_PATH" 2>&1',
        f'echo "=== startup-script begin issue={spec.issue} attempt=$EPS_ATTEMPT_ID'
        ' $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="',
        # tqdm: UNSET, never disable (#542). vLLM 0.11.0's batched
        # _run_engine (entrypoints/llm.py:1610) computes
        # ``in_spd = total_in_toks / pbar.format_dict["elapsed"]`` on the
        # first finished output. A DISABLED tqdm bar never starts its
        # timer, so ``elapsed`` stays 0.0 → ZeroDivisionError crashes
        # EVERY GCP workload that calls batched LLM.generate() (#542:
        # 4 dead eps-issue-542 VMs, identical traceback). The #491
        # giant-line zombie that originally motivated TQDM_DISABLE=1 is
        # already closed structurally by the ``exec >>"$EPS_LOG_PATH"``
        # redirect above (progress bars hit the unbounded log file, never
        # the metadata runner's bounded line scanner), so the disable was
        # pure log-cleanliness — and is now a crash. UNSET (not
        # ``=0``): tqdm 4.67.x reads TQDM_DISABLE via @envwrap, which
        # coerces ``bool("0") == True``, so ``export TQDM_DISABLE=0``
        # STILL disables the bar (timer dead → same crash; verified
        # empirically). Unsetting also clears any value inherited from
        # the DLVM image / instance metadata.
        "unset TQDM_DISABLE",
        "# === /output redirect (#607) ===",
        "",
        # Launch the reachability watchdog (#669) AFTER the redirect (so its
        # [eps-watchdog] output lands in the workload log, not the runner pipe)
        # and AFTER the redirect end marker (so the #607 prelude-slicing
        # integration tests do NOT execute the infinite daemon). ``< /dev/null``
        # detaches it from any inherited stdin so a backgrounded daemon never
        # holds a parent pipe open. Reaped on clean exit (the kill below before
        # the success sentinel) or at EXIT-trap ENTRY on a crash (#854 — before
        # the crash persist, so it cannot power off mid-upload).
        "_eps_reachability_watchdog < /dev/null &",
        "EPS_WATCHDOG_PID=$!",
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
        # === OOM baseline (#750): record this cgroup's oom_kill count BEFORE
        # the workload runs, so the post-workload guard can fire on an INCREASE
        # rather than an absolute count (the counter is cumulative-since-cgroup-
        # creation; the guest-startup cgroup is not guaranteed fresh).
        'EPS_OOM_BASELINE="$(_eps_oom_count)"',
        "",
        *workload_block,
        "",
        # === Refuse to publish done on a kernel OOM the chain rc-survived ===
        # (#750, incident #744). Re-read the cgroup-local oom_kill count and
        # compare to the pre-workload baseline; a STRICT INCREASE means an OOM
        # hit this cgroup during the workload. exit 137 (SIGKILL rc) routes
        # through the EXIT trap's rc!=0 branch -> _eps_phase failed +
        # _eps_persist_diagnostics + shutdown, so the poll reads
        # terminal_workload_failed and the async GCP->RunPod failover (#659) can
        # fire, instead of a false phase=done masking a complete failure. The
        # watchdog (still live here) is killed at EXIT-trap ENTRY (#854), not
        # the clean reaper below -- on the failure path the trap owns teardown.
        'EPS_OOM_FINAL="$(_eps_oom_count)"',
        'if [ "${EPS_OOM_FINAL:-0}" -gt "${EPS_OOM_BASELINE:-0}" ]; then',
        '  { echo "[startup-script] OOM-kill detected in cgroup'
        " (oom_kill ${EPS_OOM_BASELINE:-0} -> ${EPS_OOM_FINAL:-0}) despite rc=0"
        ' — routing to failed (#750)" >&3; } 2>/dev/null || true',
        "  exit 137",
        "fi",
        "",
        # Reap the reachability watchdog (#669) on the clean-exit path BEFORE
        # writing the success sentinel — a healthy teardown must not let the
        # daemon fire a spurious shutdown. Guarded: a missing PID / already-dead
        # daemon is a no-op. On a CRASH the EXIT-trap shutdown reaps it instead.
        '{ kill "${EPS_WATCHDOG_PID:-}" 2>/dev/null; } || true',
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
        # Publish done — the poll treats it as terminal-success and the
        # harness immediately proceeds to fetch_results (scp of the
        # sentinel written above), so the sentinel must already exist.
        "_eps_phase done",
        "",
        # The done-grace self-poweroff is the LAST line by design (#935):
        # strictly after the OOM guard, the clean-path watchdog reap, the
        # completion sentinel, and the done publish. On the abort paths
        # the script then exits rc=0, so the EXIT trap's rc!=0 branch
        # no-ops (the crash path is untouched).
        "# === Done-grace self-poweroff (#935): bound billing when the poller is dead ===",
        "_eps_done_grace_poweroff || true",
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
    * ``--maintenance-policy`` — CONDITIONAL on the resolved machine's GPU
      count (#677): ``TERMINATE`` for an accelerator VM (GPUs cannot
      live-migrate, so the policy MUST be terminate — gcloud rejects MIGRATE
      on accelerator VMs anyway), ``MIGRATE`` for a CPU-only machine
      (``gpu_count==0``), which lets a long CPU analysis phase survive a
      host-maintenance event instead of being killed mid-store-download.
    * ``--no-restart-on-failure`` (``automaticRestart=false``, #669) — a
      watchdog self-shutdown (or any crash) is FINAL; auto-restart would
      bring the VM back into the same wedged state. Independent scheduling
      field; composes freely with the two above.
    * ``--max-run-duration`` — the per-instance VM auto-delete fence, set
      to the FLEX_START ceiling (default 7d per config, #741) so a long
      multi-cell sweep is not stranded mid-run; the per-instance-fence-aware
      janitor age reap is the credit-leak backstop, not this fence.
    * ``--image-family`` / ``--image-project`` — the DLVM image with
      pre-installed CUDA/driver.
    * ``--boot-disk-size`` / ``--boot-disk-type`` — 300 GB pd-ssd default;
      a plan override below the DLVM image minimum is clamped UP to
      100 GB (#1336).
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
    # a3-highgpu (H100) cannot be created on-demand (docs: "you must create
    # instances by using Spot VMs or Flex-start VMs"). Fail loud at render
    # rather than issue a doomed STANDARD create (#631).
    if machine.gpu_kind == "H100-80" and provisioning == "STANDARD":
        raise ValueError(
            f"intent {spec.intent!r} ({machine.machine_type}, H100) cannot be created on-demand; "
            "pass spec.extra['provisioning_model'] = 'SPOT' or 'FLEX_START'."
        )
    max_run = spec.extra.get("max_run_duration") or config.default_max_run_duration
    _assert_max_run_within_flex_cap(max_run=max_run, provisioning=provisioning)
    requested_boot_disk_gb = int(spec.extra.get("boot_disk_gb") or config.default_boot_disk_gb)
    boot_disk_gb = max(_GCP_IMAGE_MIN_BOOT_DISK_GB, requested_boot_disk_gb)
    if boot_disk_gb != requested_boot_disk_gb:
        logger.warning(
            "boot-disk clamped UP to the DLVM image minimum: requested %d GB < %d GB "
            "(GCP rejects disks smaller than the image, #1336); provisioning %d GB.",
            requested_boot_disk_gb,
            _GCP_IMAGE_MIN_BOOT_DISK_GB,
            boot_disk_gb,
        )
    boot_disk_type = spec.extra.get("boot_disk_type") or config.default_boot_disk_type
    target_zone = zone or config.primary_zone
    name = instance_name_for(spec.issue, lane_suffix_for(spec))

    # GPUs cannot live-migrate (gcloud rejects MIGRATE on accelerator VMs), so
    # accelerator VMs MUST be TERMINATE. A CPU-only machine (gpu_count==0, #677)
    # CAN live-migrate, and MIGRATE is gcloud's default for non-accelerator VMs
    # — it lets a long CPU analysis phase survive a host-maintenance event
    # instead of being killed mid-store-download.
    maintenance_policy = "TERMINATE" if machine.gpu_count else "MIGRATE"

    argv = _base_gcloud_argv(config, "compute", "instances", "create", name)
    argv += [
        f"--zone={target_zone}",
        f"--machine-type={machine.machine_type}",
        f"--provisioning-model={provisioning}",
        "--instance-termination-action=DELETE",
        f"--maintenance-policy={maintenance_policy}",
        # automaticRestart=false (#669): a watchdog self-shutdown (or any
        # crash) must be FINAL — auto-restart would bring the VM back into the
        # same wedged state (the systemd-networkd DHCP-renewal bug recurs under
        # the same load). Composes freely with --instance-termination-action
        # and --max-run-duration (independent scheduling-block fields).
        "--no-restart-on-failure",
        f"--max-run-duration={max_run}",
        f"--image-family={config.image_family}",
        f"--image-project={config.image_project}",
        f"--boot-disk-size={boot_disk_gb}GB",
        f"--boot-disk-type={boot_disk_type}",
        "--scopes=cloud-platform",
        "--labels=" + _format_labels(spec, attempt_id),
        "--format=json",
    ]
    # FLEX_START (DWS flex-start) requires the request-validity window and
    # an explicit no-reservation affinity — verbatim in Google's canonical
    # flex-start create command (#631). STANDARD / SPOT keep their existing
    # argv (neither flag; a regression test pins their omission).
    if provisioning == "FLEX_START":
        argv.append(f"--request-valid-for-duration={resolve_request_valid_for_duration(spec)}")
        argv.append("--reservation-affinity=none")

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
    # (the VM deliberately stays RUNNING so the sentinel can be scp'd —
    # within the bounded #935 done-grace window — and the coarse
    # describe-based poll reads "running" until the hard
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


def render_guest_attributes_argv(
    *, config: GcpConfig, name: str, zone: str, query_path: str = "eps/phase"
) -> list[str]:
    """Build a ``gcloud compute instances get-guest-attributes`` argv.

    Queries the ``eps/phase`` guest attribute the startup script
    publishes (``_eps_phase``) — the poll-readable workload-phase
    surface a RUNNING VM exposes (issue 535 r9: without it a successful
    workload is undetectable and the poll spins to the hard timeout).
    ``--query-path`` scopes the read to our namespace; gcloud exits
    non-zero when the attribute was never written (a VM still booting),
    which the poll treats as phase-unknown, NOT an error.

    ``query_path`` is parameterized (#659): the workload-vs-setup
    discrimination issues a SECOND read scoped to ``eps/workload_started``
    (the write-once sentinel the workload-phase preamble publishes) to
    decide whether a ``phase==failed`` poll is a real workload crash or a
    pre-workload setup failure.
    """
    argv = _base_gcloud_argv(config, "compute", "instances", "get-guest-attributes", name)
    argv += [f"--zone={zone}", f"--query-path={query_path}", "--format=json"]
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


class GcpCreateTimedOutStillProvisioning(GcpBackendError):
    """The ``gcloud compute instances create`` subprocess exceeded the
    300s cap, BUT a post-timeout ``instances list`` probe found the
    instance live (RUNNING/PROVISIONING/STAGING/STOPPING) — the
    FLEX_START preemptible-queueing case (#658).

    This is NOT a failure: the launch is in flight server-side, the local
    ``subprocess.run`` just aborted at its wall-clock cap. The caller
    (``dispatch_issue.py::_cmd_launch``) converts this into the documented
    still-waiting exit (75 / ``EXIT_STILL_WAITING``, #603) and the
    orchestrator re-runs the SAME launch command; ``reconnect_or_none``
    (the idempotent re-entry at the top of :meth:`GcpBackend.launch`)
    reconnects to the now-observable live instance with NO double-create.

    Deliberately a :class:`GcpBackendError` subclass — the SAME base as
    :class:`GcpProvisioningError` / :class:`GcpWorkloadError`, NOT a
    ``router.RouteError`` (subclassing the latter would force ``gcp.py``
    to import ``router.py``, creating a router→gcp→router circular import).
    Because it is neither a ``RouteError`` nor a ``GcpProvisioningError``
    / ``GcpWorkloadError`` / ``BackendProbeError``, NONE of the router's
    typed ``except`` arms catch it, so it propagates clean to
    ``_cmd_launch``, where the explicit ``except`` arm is MANDATORY.

    Carries ``instance_name`` + observed ``status`` (+ ``issue``) for the
    still-waiting JSON the CLI emits.
    """

    def __init__(self, *, instance_name: str, status: str, issue: int) -> None:
        self.instance_name = instance_name
        self.status = status
        self.issue = issue
        super().__init__(
            f"GCP create for {instance_name} timed out at the "
            f"{GCLOUD_DEFAULT_TIMEOUT_SEC}s subprocess cap but the instance is live "
            f"(status={status}) — FLEX_START still-provisioning; re-run to continue waiting."
        )


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


#: Per-zone ``stderr_tail`` cap (chars) for the #774 fan-out record. Tighter
#: than ``classify_create_failure``'s 2000-char ``evidence['stderr_tail']`` so
#: the multi-zone marker stays compact; it reuses the SAME already-published
#: stderr text, introducing no new disclosure surface.
_PER_ZONE_STDERR_TAIL_CAP = 200


def _record_zone_outcome(
    outcomes: list[dict[str, Any]],
    zone: str,
    *,
    returncode: int,
    matched_pattern: str | None,
    started: float,
    stderr: str = "",
) -> None:
    """Append one per-zone GCP create outcome to ``outcomes`` (visibility for #774).

    One ``{zone, returncode, matched_pattern, elapsed_s, stderr_tail}`` record
    per zone the create for-loop tried. ``stderr`` is the SAME text
    ``classify_create_failure`` already truncates + publishes in
    ``evidence['stderr_tail']`` / the router ``detail`` field, so this
    introduces NO new disclosure surface; the per-zone copy is capped at
    ``_PER_ZONE_STDERR_TAIL_CAP`` chars to keep the marker compact.
    ``returncode=-1`` is the sentinel for a create-timeout (no real process
    exit code available). Module-level (not a nested closure) to keep
    :meth:`GcpBackend.launch` under the C901 complexity cap.
    """
    outcomes.append(
        {
            "zone": zone,
            "returncode": returncode,
            "matched_pattern": matched_pattern,
            "elapsed_s": round(time.monotonic() - started, 3),
            "stderr_tail": (stderr or "")[-_PER_ZONE_STDERR_TAIL_CAP:],
        }
    )


def _attach_fanout_evidence(
    error: GcpProvisioningError,
    outcomes: list[dict[str, Any]],
    *,
    machine_type: str,
    with_summary: bool,
) -> None:
    """Thread the #774 per-zone fan-out evidence onto a raised create error.

    Records the rich ``per_zone_attempts`` record list + the derived bare
    ``zones_attempted`` name-list (for back-compat) via ``setdefault`` so a
    re-raise never clobbers. When ``with_summary`` (the all-zones-exhausted
    for-else path) it also writes the human-readable
    ``zones_attempted_summary`` one-liner. Module-level (not inlined into
    :meth:`GcpBackend.launch`) to keep that method under the C901 cap.
    """
    zones_attempted = [o["zone"] for o in outcomes]
    error.evidence.setdefault("per_zone_attempts", list(outcomes))
    error.evidence.setdefault("zones_attempted", list(zones_attempted))
    if with_summary:
        error.evidence["zones_attempted_summary"] = (
            f"all {len(zones_attempted)} zone(s) "
            f"[{', '.join(zones_attempted)}] capacity-exhausted for "
            f"machine_type={machine_type}"
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


#: Wall-clock cap (seconds) for a single ``gcloud`` subprocess in
#: :func:`default_gcloud_runner`. Named (not a bare literal) so the
#: :meth:`GcpBackend.launch` create-timeout handler + the
#: :class:`GcpCreateTimedOutStillProvisioning` message reference the SAME
#: value the runner enforces and the two cannot drift (#736).
GCLOUD_DEFAULT_TIMEOUT_SEC = 300


def default_gcloud_runner(
    argv: Sequence[str], *, timeout: int = GCLOUD_DEFAULT_TIMEOUT_SEC
) -> GcloudRunResult:
    """Default runner: shell out to ``gcloud`` via :mod:`subprocess`.

    Raises NOTHING on non-zero — the caller inspects ``returncode``.
    Timeouts propagate as :class:`subprocess.TimeoutExpired`;
    :meth:`GcpBackend.launch` catches the create-call timeout, probes the
    canonical instance name via :func:`reconnect_or_none`, and either
    raises :class:`GcpCreateTimedOutStillProvisioning` (instance live
    server-side — the FLEX_START preemptible-queueing case) or
    :class:`GcpProvisioningError` (instance absent, or the probe itself
    failed), so the router routes the timeout as a provisioning outcome
    rather than an undocumented exit-4 traceback (#736).
    """
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return GcloudRunResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


#: Guest ``eps/phase`` values that mean the workload has FINISHED (terminal).
#: A RUNNING VM that has published one of these but never auto-deleted is a
#: wedged zombie — the workload is over, the VM is still billing — and the
#: stale-VM janitor (:func:`audit_stale_gcp_vms`) reaps it promptly past a
#: short terminal-phase floor rather than waiting out the per-fence age backstop.
#: ``finalize_failed_artifacts_ok`` (#1055 — deliverables verified on HF, then
#: a finalize/tail non-zero exit) is terminal too: the workload is over, so a
#: RUNNING VM stuck in it past the floor is a finished zombie the janitor MUST
#: reap (unlike ``wedged``, deliberately kept OUT of this set — see
#: :data:`_ZOMBIE_GUEST_PHASES`).
_TERMINAL_GUEST_PHASES: frozenset[str] = frozenset(
    {"done", "failed", "finalize_failed_artifacts_ok"}
)


# ---------------------------------------------------------------------------
# Stale-VM janitor scope + classification (#688)
# ---------------------------------------------------------------------------

#: Janitor inventory filter for the dedicated project. ``None`` lists the WHOLE
#: project (the broadened credit-leak backstop, #688) rather than only the
#: router-managed ``eps-issue-*`` names — a non-``eps-issue-*`` leftover (the
#: #680 ``eps-cap-probe2-1786331`` flex-start probe ran ~20h, ~$14) was
#: invisible to the old ``name~^eps-issue-`` filter. Exported so
#: ``scripts/gcp_audit.py``'s list-preflight stays byte-identical to the list
#: the reaper issues internally (the preflight imports this constant).
JANITOR_LIST_NAME_FILTER: str | None = None

#: Router-managed instance name prefix — the ONLY names the GCP backend
#: auto-creates and reconnects/reclaims by name (``eps-issue-<N>``). The
#: ROUTER seams (:func:`reconnect_or_none`, :func:`_stale_named_instance_or_none`)
#: keep their EXACT ``name=eps-issue-<N>`` list filters; only the JANITOR's
#: inventory query broadens. Janitor-managed names auto-reap on the existing
#: bounded fences exactly as before.
_MANAGED_NAME_PREFIX = "eps-issue-"

#: Non-managed name prefixes that are KNOWN-ephemeral throwaways — auto-reaped
#: by the janitor on the SAME bounded fences as managed VMs. A tuple constant
#: so the user can grow it as new probe-name patterns emerge. Default: the #680
#: flex-start capacity probes (``eps-cap-probe*``), the leak class that
#: motivated #688.
_EPHEMERAL_REAP_PREFIXES: tuple[str, ...] = ("eps-cap-probe",)

#: Name prefixes the janitor must NEVER reap OR escalate — a deliberate opt-out
#: for any long-lived instance that legitimately lives in the dedicated
#: project. Empty today (no such instance exists); the hook exists so the
#: design does not hard-assume "everything is disposable".
_JANITOR_KEEP_PREFIXES: tuple[str, ...] = ()

#: Janitor classification classes (the ``classification`` record field).
JANITOR_CLASS_MANAGED = "managed"
JANITOR_CLASS_ALLOWLISTED = "allowlisted-ephemeral"
JANITOR_CLASS_KEEP = "keep"
JANITOR_CLASS_UNMANAGED = "unmanaged"


def _classify_janitor_instance(name: str) -> str:
    """Classify a listed instance by name → reap/escalate routing class.

    Returns one of: ``"managed"`` (``eps-issue-*``; router-owned, auto-reap),
    ``"allowlisted-ephemeral"`` (a known-throwaway prefix, auto-reap),
    ``"keep"`` (opt-out prefix; never touched), or ``"unmanaged"``
    (everything else in the dedicated project; ESCALATE — never auto-delete).

    Precedence is deliberate: ``keep`` wins over ``managed``/``allowlisted``
    so an explicit opt-out is honored even for an ``eps-issue-*``/probe name;
    ``managed`` then ``allowlisted`` then the ``unmanaged`` fall-through.
    """
    if any(name.startswith(p) for p in _JANITOR_KEEP_PREFIXES):
        return JANITOR_CLASS_KEEP
    if name.startswith(_MANAGED_NAME_PREFIX):
        return JANITOR_CLASS_MANAGED
    if any(name.startswith(p) for p in _EPHEMERAL_REAP_PREFIXES):
        return JANITOR_CLASS_ALLOWLISTED
    return JANITOR_CLASS_UNMANAGED


def render_escalation_message(record: dict[str, Any]) -> str:
    """Human-facing one-liner for an UNMANAGED stale GCP VM the janitor will
    NOT auto-delete (mirrors ``vm_disk_guard``'s escalate-don't-delete posture).

    The VM is stale (an age / terminal-phase ``reason`` is already set) but is
    neither router-managed nor an allowlisted throwaway, so the safe default is
    to surface it for a human, never reap it blind.
    """
    age_seconds = record.get("age_seconds")
    age_str = f"{age_seconds / 3600:.1f}h" if isinstance(age_seconds, int | float) else "unknown"
    return (
        f"GCP janitor: UNMANAGED stale VM {record['name']} (zone={record['zone']}, "
        f"status={record['status']}, age={age_str}, reason={record['reason']}) in the "
        f"dedicated project — NOT auto-deleted (not router-managed, not an allowlisted "
        f"throwaway). Inspect + `gcloud compute instances delete {record['name']} "
        f"--zone={record['zone']}` if it is a leak."
    )


def _read_guest_phase(*, config: GcpConfig, name: str, zone: str, runner: GcloudRunner) -> str:
    """Read the ``eps/phase`` guest attribute for ``name``; "" if unwritten.

    Two failure classes are deliberately distinguished, mirroring the
    poll-side contract (round-2 Codex Major, task #535 — pre-fix, EVERY
    nonzero rc / bad-JSON read returned "" and was indistinguishable from
    "phase not written yet"):

    * EXPECTED not-written-yet — gcloud exits nonzero with a 404 / "not
      found" stderr (the attribute does not exist until the startup script's
      first ``_eps_phase`` write). Returns ``""``.
    * Probe failure — any OTHER nonzero rc (expired auth, permission denied,
      transport) or unparseable JSON from an rc=0 call. Raises
      :class:`GcpProbeError` ("couldn't ask" must never read as "not done
      yet"); the caller translates it into its own typed handling.
    """
    argv = render_guest_attributes_argv(config=config, name=name, zone=zone)
    result = runner(argv)
    if result.returncode != 0:
        stderr_low = (result.stderr or "").lower()
        if "not found" in stderr_low or "404" in stderr_low:
            return ""  # attribute not written yet — legitimate pre-phase state
        raise GcpProbeError(
            f"GCP guest-attribute probe failed for {name}: "
            f"rc={result.returncode} stderr={result.stderr[:500]!r} — workload "
            "phase UNKNOWN, refusing to read a probe failure as still-running"
        )
    try:
        payload = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError as exc:
        raise GcpProbeError(
            f"GCP guest-attribute probe returned unparseable JSON for "
            f"{name}: {exc} — workload phase UNKNOWN"
        ) from exc
    # gcloud returns a list of {namespace, key, value} dicts.
    for item in payload if isinstance(payload, list) else []:
        if item.get("key") == "phase":
            return str(item.get("value") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Reconnect (idempotent existing-instance lookup)
# ---------------------------------------------------------------------------

#: Instance statuses that ``reconnect_or_none`` treats as NOT-live (it
#: returns ``None`` rather than a handle). These are exactly the statuses
#: under which a record still OCCUPIES the canonical ``eps-issue-<N>`` name
#: — so a subsequent ``gcloud compute instances create`` collides with
#: ``resource ... already exists``. The pre-launch ``_stale_named_instance_or_none``
#: helper owns deleting them; the two sets MUST stay identical (a status
#: reconnect skips but the pre-launch check does NOT delete would re-create
#: the "already exists" wedge it exists to prevent). Incident #632
#: (2026-06-13): a workload-crash respawn hit "already exists" because the
#: prior TERMINATED record blocked re-provisioning and nothing deleted it.
#: The SAME identical-sets invariant extends to the phase-gated RUNNING
#: case below (:data:`_ZOMBIE_GUEST_PHASES`): a RUNNING record whose
#: ``eps/phase`` disqualifies it from reconnect MUST be deletable by the
#: pre-launch check, or the refusal dead-ends in the "already exists" wedge.
_NONLIVE_INSTANCE_STATUSES: frozenset[str] = frozenset({"TERMINATED", "STOPPED", "SUSPENDED"})

#: Guest ``eps/phase`` values that disqualify a RUNNING instance as a
#: reconnect target AND qualify it for pre-launch reclaim (#908/#763): the
#: janitor's terminal set (``done``/``failed``/``finalize_failed_artifacts_ok``,
#: :data:`_TERMINAL_GUEST_PHASES`)
#: plus the #669 reachability watchdog's pre-shutdown ``wedged`` write. A
#: RUNNING instance in any of these states is a finished-or-wedged zombie —
#: reconnecting to it silently no-ops the new dispatch (#763 leg 2), and
#: only deleting it frees the canonical name for the create (#632).
#: The two consumers (:func:`reconnect_or_none`,
#: :func:`_stale_named_instance_or_none`) MUST share this set — same
#: invariant as :data:`_NONLIVE_INSTANCE_STATUSES` above. Deliberately a
#: NEW constant, not an edit to :data:`_TERMINAL_GUEST_PHASES` (adding
#: ``wedged`` there would change janitor reap behavior — the #667
#: follow-up's business, out of scope here).
#: Relaunch contract (#908 leg 1b): the #491 same-VM SSH-relaunch recovery
#: recipe (`.claude/rules/gotchas.md`, GCE-metadata-runner entry) REQUIRES
#: re-publishing ``eps/phase=workload`` BEFORE resuming work, so an active
#: manual relaunch reads non-terminal here and is never classified a zombie.
_ZOMBIE_GUEST_PHASES: frozenset[str] = _TERMINAL_GUEST_PHASES | frozenset({"wedged"})


def reconnect_or_none(
    *,
    spec: RunSpec,
    config: GcpConfig,
    runner: GcloudRunner,
) -> RunHandle | None:
    """Return a handle for an existing live ``eps-issue-<N>[-<lane_suffix>]`` instance, or None.

    Idempotency hinge: before any ``instances create`` call, this looks
    up the canonical instance name (per-lane suffixed when
    ``spec.extra['lane_suffix']`` is set, #934) via ``gcloud compute
    instances list --filter='name=eps-issue-<N>[-<suffix>]'`` — so a
    suffixed lane never reconnects to a sibling lane's instance. A live
    instance (status RUNNING,
    PROVISIONING, STAGING, STOPPING) returns a handle. A TERMINATED
    instance is treated as "not live" (the backend will create a fresh
    one); no instance returns None.

    Zombie refusal (#908/#763): a RUNNING instance whose ``eps/phase``
    guest attribute is already in :data:`_ZOMBIE_GUEST_PHASES`
    (``done``/``failed``/``finalize_failed_artifacts_ok``/``wedged``) is
    NOT a live run to rejoin —
    reconnecting to it silently no-ops the new dispatch (#763: the
    phase-C launch "reconnected" to a gate-parked done VM and never
    ran). Such an instance returns ``None``; the pre-launch stale
    reclaim (:func:`_stale_named_instance_or_none`) then deletes it so
    the create does not collide (#632). An unwritten phase (``""`` —
    early boot, 404) reconnects normally; the probe fires for RUNNING
    status only (PROVISIONING/STAGING have no phase yet; STOPPING is a
    transient teardown state).

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
    ``BackendProbeError`` typed-ly on every lane. The #908 phase probe
    adds a second (guest-attribute) raise surface with the SAME
    semantics — a probe flake fails the launch typed and RETRIABLE
    (re-run the same command; idempotent by design, the #736 exit-75
    precedent), never a silent reconnect and never a delete.

    Failover-prerequisite extras (#1122): when the spec carries a
    workload (``workload_cmd`` or ``hydra_args``), the reconnect
    handle's ``extra`` mirrors the launch path's failover keys
    (``workload_cmd`` / ``hydra_args`` / ``gpus`` /
    ``time_budget_hours`` / ``repo_branch`` / ``gpu_count`` +
    ``boot_disk_gb`` / ``min_ram_gb`` when set). The #736 exit-75
    contract re-runs the SAME command, and
    ``issue_dispatch.dispatch_for_issue``'s ``on_launched`` hook
    OVERWRITES the handle sidecar with THIS handle — pre-#1122 the
    minimal probe-derived extra clobbered the launch handle's workload
    keys, so the #783 queue-timeout RunPod failover crashed at
    ``backend_poll._runspec_from_gcp_handle`` (incident #1090).
    """
    name = instance_name_for(spec.issue, lane_suffix_for(spec))
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
        if status.upper() in _NONLIVE_INSTANCE_STATUSES:
            continue
        zone_url = inst.get("zone") or ""
        # The zone field is a URL; take the last path segment.
        zone = zone_url.rsplit("/", 1)[-1] if zone_url else config.primary_zone
        # NEW (#908): a RUNNING instance whose workload already published a
        # terminal/wedged ``eps/phase`` is a zombie, NOT a live run to rejoin
        # — reconnecting to it silently no-ops the new dispatch (#763: the
        # phase-C launch "reconnected" to the gate-parked done VM and never
        # ran). ``""`` (unwritten — early boot, 404) reconnects normally. A
        # probe FAILURE raises GcpProbeError out of this function: state
        # UNKNOWN must never read as EITHER "live, reconnect" (would
        # resurrect the silent no-op) or "zombie, delete" (could reclaim a
        # healthy VM) — the #535 "couldn't ask" discipline, same as the
        # LIST probe above. RUNNING-only, matching the janitor's
        # ``should_probe_phase`` gate (PROVISIONING/STAGING have no phase
        # yet; STOPPING+done is the normal seconds-long teardown transition).
        if status.upper() == "RUNNING":
            phase = _read_guest_phase(config=config, name=name, zone=zone, runner=runner)
            if phase in _ZOMBIE_GUEST_PHASES:
                logger.warning(
                    "GCP reconnect: %s is RUNNING with terminal eps/phase=%r — refusing "
                    "reconnect (gate-park/finished zombie, #908); the pre-launch stale "
                    "reclaim deletes it before create.",
                    name,
                    phase,
                )
                continue
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
        # #1122: mirror the launch path's failover-prerequisite keys
        # (#659 MF1/MF2, #909 repo_branch, #677 gpu_count, #1010 footprint)
        # so an exit-75 same-command RERUN's reconnect handle — which
        # OVERWRITES the sidecar via issue_dispatch's on_launched hook —
        # stays failover-capable (backend_poll._runspec_from_gcp_handle).
        # Gated on the spec carrying a workload: a bare (provision-only /
        # probe) reconnect keeps the legacy extra shape, and the
        # issue_dispatch carry-forward (#1122 edit 2) preserves the prior
        # sidecar's values for that case instead.
        if spec.workload_cmd or spec.hydra_args:
            # MF1: workload_cmd is a str — preserve AS-IS, never list().
            # MF2: one of the pair is empty by RunSpec.__post_init__
            # mutual exclusion; write BOTH so the poller's verbatim
            # pass-through reconstruction holds.
            extra["workload_cmd"] = spec.workload_cmd or ""
            extra["hydra_args"] = list(spec.hydra_args or ())
            extra["gpus"] = spec.gpus
            extra["time_budget_hours"] = spec.time_budget_hours
            extra["repo_branch"] = str((spec.extra or {}).get("repo_branch") or "")
            # 0-vs-nonzero is all the poller's CPU-lane guards read
            # (_is_gcp_async_workload_failure / _is_gcp_queue_timeout),
            # and CPU-ness is intent-determined — override-invariant even
            # when the creating rung used a machine_spec_override.
            extra["gpu_count"] = machine_for_intent(spec).gpu_count
            extra.update(
                {
                    k: v
                    for k, v in {
                        "boot_disk_gb": (spec.extra or {}).get("boot_disk_gb"),
                        "min_ram_gb": (spec.extra or {}).get("min_ram_gb"),
                    }.items()
                    if v
                }
            )
        return RunHandle(
            backend="gcp",
            cluster=None,
            job_id=instance_id,
            pod_name=name,
            scratch_dir=workload_dir_for(config, spec.issue),
            log_path=log_path_for(config, spec.issue),
            extra=extra,
        )
    return None


# ---------------------------------------------------------------------------
# Pre-launch stale-name reclaim (#632)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaleNamedInstance:
    """A non-live ``eps-issue-<N>`` record still occupying the canonical name.

    Returned by :func:`_stale_named_instance_or_none` when a prior
    instance is in a :data:`_NONLIVE_INSTANCE_STATUSES` state (TERMINATED /
    STOPPED / SUSPENDED) — OR (#908) is RUNNING with a terminal/wedged
    ``eps/phase`` (:data:`_ZOMBIE_GUEST_PHASES`, a gate-park/finished
    zombie): the record blocks the next ``gcloud compute
    instances create`` with ``resource ... already exists`` even though the
    instance is doing nothing. ``zone`` is the parsed last-segment of the
    instance's zone URL so the launch path can delete it in the right zone.
    ``guest_phase`` is set ONLY on the #908 RUNNING+zombie case (the
    re-probed ``eps/phase`` value); ``None`` for the status-stale cases —
    defaulted so pre-#908 constructors stay valid.
    """

    name: str
    zone: str
    status: str
    guest_phase: str | None = None


def _stale_named_instance_or_none(
    *,
    spec: RunSpec,
    config: GcpConfig,
    runner: GcloudRunner,
) -> StaleNamedInstance | None:
    """Return a stale non-live record blocking the ``eps-issue-<N>[-<lane_suffix>]`` name, or None.

    Called by :meth:`GcpBackend.launch` ONLY after
    :func:`reconnect_or_none` has already returned ``None`` (no LIVE
    instance). A non-live record (TERMINATED / STOPPED / SUSPENDED — the
    same :data:`_NONLIVE_INSTANCE_STATUSES` reconnect skips) still owns the
    canonical name, so the upcoming ``instances create`` would collide with
    ``resource ... already exists`` (incident #632, 2026-06-13: a
    workload-crash respawn powered the VM off into TERMINATED, then the
    infra-class respawn's create was rejected until the operator manually
    ran ``gcloud compute instances delete``). The launch path deletes the
    returned record before re-provisioning.

    Returns:
        * ``StaleNamedInstance`` — a record in a non-live state exists,
          OR (#908) a RUNNING record whose re-probed ``eps/phase`` is in
          :data:`_ZOMBIE_GUEST_PHASES` (``done``/``failed``/
          ``finalize_failed_artifacts_ok``/``wedged`` —
          the gate-park/finished zombie ``reconnect_or_none`` now
          refuses; ``guest_phase`` carries the probed value). Both are
          safe to delete (no live workload); the matched skip/delete
          sets are the #632 invariant on
          :data:`_NONLIVE_INSTANCE_STATUSES`.
        * ``None`` — no record with the canonical name exists; ``create``
          will proceed clean.

    Raises:
        * :class:`GcpProbeError` — the ``list`` probe itself failed
          (rc != 0 / unparseable JSON), OR (#908) the ``eps/phase``
          guest-attribute probe on a RUNNING record failed. Instance /
          phase state is UNKNOWN, and "couldn't ask" must never read as
          "name is free" — and NEVER as "zombie, delete" — on the
          credit-spending lane (mirrors :func:`reconnect_or_none`).
        * :class:`GcpBackendError` — a record exists in a state that is
          NEITHER live-reconnectable NOR deletable: a non-RUNNING live
          status (PROVISIONING / STAGING / STOPPING / REPAIRING), or a
          RUNNING record whose re-probed phase is NON-terminal. This
          can only happen as a TOCTOU race against the reconnect probe
          (which would itself have returned a handle for a live status):
          refuse to auto-delete a possibly-live instance — deleting a
          RUNNING VM mid-provision is data loss. Surface loudly so the
          orchestrator retries the launch (whose reconnect will then catch
          the now-observable live instance).
    """
    name = instance_name_for(spec.issue, lane_suffix_for(spec))
    result = runner(render_list_argv(config=config, name_filter=f"name={name}"))
    if result.returncode != 0:
        raise GcpProbeError(
            f"GCP stale-name probe failed for {name}: gcloud list rc={result.returncode} "
            f"stderr={result.stderr[:500]!r} — instance state UNKNOWN, refusing to "
            "assume the name is free before create"
        )
    try:
        instances = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError as exc:
        raise GcpProbeError(
            f"GCP stale-name probe returned unparseable JSON for {name}: {exc} — "
            "instance state UNKNOWN"
        ) from exc
    if not isinstance(instances, list):
        return None
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("name") != name:
            continue
        status = (inst.get("status") or "").upper()
        zone_url = inst.get("zone") or ""
        zone = zone_url.rsplit("/", 1)[-1] if zone_url else config.primary_zone
        if status in _NONLIVE_INSTANCE_STATUSES:
            return StaleNamedInstance(name=name, zone=zone, status=status)
        # NEW (#908): reconnect now REFUSES a RUNNING instance with a
        # terminal/wedged eps/phase (see reconnect_or_none), so the SAME
        # extended set must be deletable here — a skip without a matching
        # delete re-creates the #632 "already exists" create collision
        # (the documented invariant on _NONLIVE_INSTANCE_STATUSES). The
        # phase is RE-PROBED locally (never carried over from the earlier
        # reconnect read); a probe failure propagates GcpProbeError — no
        # delete on unknown state. What a delete here costs on the
        # accepted same-workload re-entry race (R1): any UNDRAINED
        # VM-disk sentinel is destroyed AND the workload re-runs in full
        # (duplicate compute + duplicate non-idempotent side effects,
        # e.g. HF/WandB appends — the re-run regenerates the sentinel);
        # artifacts themselves were uploaded BEFORE ``[phase=done]`` per
        # the pod-side blocking contract, so no permanent data loss.
        if status == "RUNNING":
            phase = _read_guest_phase(config=config, name=name, zone=zone, runner=runner)
            if phase in _ZOMBIE_GUEST_PHASES:
                return StaleNamedInstance(name=name, zone=zone, status=status, guest_phase=phase)
        # A record exists in a state the non-live set does NOT cover. The
        # only way to reach here is a TOCTOU race vs the reconnect probe
        # (a live status would have reconnected). Never auto-delete a
        # possibly-live instance — surface loudly and let the orchestrator
        # re-launch (its reconnect catches the now-live instance).
        raise GcpBackendError(
            f"GCP pre-launch: instance {name} exists in non-deletable status "
            f"{status!r} (zone={zone}); refusing to auto-delete a possibly-live "
            "instance before create (a RUNNING record reaches this raise only "
            "with a checked, NON-terminal eps/phase — #908). Re-launch to "
            "reconnect, or delete manually if it is genuinely stale."
        )
    return None


# ---------------------------------------------------------------------------
# Pre-create regional-quota headroom probe (#608)
# ---------------------------------------------------------------------------


#: ``MachineSpec.gpu_kind`` → the ON-DEMAND regional accelerator-quota
#: metric reported by ``gcloud compute regions describe`` ``quotas[]``.
#: Verified live on issue 608 (2026-06-12): ``NVIDIA_A100_80GB_GPUS`` read
#: usage 8.0 / limit 8.0 while the ``ft-7b`` intent needed 4 — every create
#: was doomed. H100 has NO on-demand metric (a3-highgpu is Spot/flex-start
#: only), so it is absent here.
_GPU_KIND_TO_QUOTA_METRIC: dict[str, str] = {
    "A100-80": "NVIDIA_A100_80GB_GPUS",
    # A100-40 (a2-highgpu, the #656 fallback rung) draws the un-suffixed
    # NVIDIA_A100_GPUS pool — a SEPARATE regional quota from the 80GB pool,
    # so it can have headroom when A100-80 is full. Read ONLY by the
    # fail-OPEN pre-check, so a wrong name degrades to "no opinion; proceed
    # to create", never a false block (#656 §12 / same property as the H100
    # preemptible metric).
    "A100-40": "NVIDIA_A100_GPUS",
    "L4": "NVIDIA_L4_GPUS",
}

#: ``MachineSpec.gpu_kind`` → the PREEMPTIBLE regional accelerator-quota
#: metric. Spot AND flex-start both consume preemptible quota (docs: "you
#: must have sufficient preemptible quota for ... any attached GPUs"). The
#: H100 entry is the base a3-highgpu metric (NOT the ``_MEGA_`` variant);
#: it is only read by the fail-OPEN pre-check, so a wrong name degrades to
#: "no opinion; proceed", never a false block (#631 §8 / §12 assumption 9).
_GPU_KIND_TO_PREEMPTIBLE_QUOTA_METRIC: dict[str, str] = {
    "A100-80": "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS",
    # A100-40 preemptible pool (#656 spot-A100-40 rung) — the un-suffixed
    # PREEMPTIBLE_NVIDIA_A100_GPUS, sibling of the on-demand NVIDIA_A100_GPUS
    # above. Fail-OPEN pre-check only (a wrong name → "no opinion; proceed").
    "A100-40": "PREEMPTIBLE_NVIDIA_A100_GPUS",
    "L4": "PREEMPTIBLE_NVIDIA_L4_GPUS",
    "H100-80": "PREEMPTIBLE_NVIDIA_H100_GPUS",
}


def quota_metric_for(machine: MachineSpec, provisioning: str) -> str | None:
    """Regional accelerator-quota metric for ``machine`` under ``provisioning``.

    SPOT and FLEX_START draw the PREEMPTIBLE pool; STANDARD draws the
    on-demand pool. Returns ``None`` when the (gpu_kind, pool) pair has no
    mapping (e.g. H100 + STANDARD, which is rejected at render anyway) so
    the fail-OPEN pre-check proceeds rather than blocking (#631).
    """
    # CPU-only machines draw no accelerator quota (#677) — return None so the
    # fail-OPEN preflight ("no opinion; proceed") skips the accelerator-quota
    # probe entirely. (quota_metric_for already returns None for an unmapped
    # gpu_kind, so "CPU" was already None; this is the explicit, intent-
    # documenting guard that also survives a future map entry.)
    if machine.gpu_count == 0:
        return None
    if provisioning in {"SPOT", "FLEX_START"}:
        return _GPU_KIND_TO_PREEMPTIBLE_QUOTA_METRIC.get(machine.gpu_kind)
    return _GPU_KIND_TO_QUOTA_METRIC.get(machine.gpu_kind)


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
    * a RUNNING zombie (terminal/wedged ``eps/phase``, #908) occupies
      the canonical name — reconnect refuses it, but its GPUs still
      count in the usage read; launch's stale reclaim deletes it and
      frees the quota before create, so the headroom verdict would be
      stale by construction,
    * the reconnect probe, the stale-name probe, or the ``regions
      describe`` call fails in ANY way (rc != 0, missing gcloud,
      timeout, unparseable JSON, metric absent from ``quotas[]``).

    Only a successfully parsed quota row produces a verdict. A swallowed
    probe failure here never enables a blind create: the launch path
    re-runs its own reconnect probe and raises typed-ly on failure.
    """
    try:
        machine = machine_for_intent(spec)
    except ValueError:
        return None
    metric = quota_metric_for(machine, resolve_provisioning_model(spec))
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
    # NEW (#908): reconnect now REFUSES a RUNNING instance whose eps/phase is
    # terminal/wedged (a gate-park zombie), so it returns None above while
    # the zombie's allocated GPUs still COUNT in the regions-describe usage
    # read — in a tight-quota regime the headroom verdict would block the
    # GCP lane BEFORE launch's stale reclaim can delete the zombie and free
    # the quota. Restore the pre-#908 disposition for exactly that case: a
    # RUNNING record with a terminal guest phase is about to be reclaimed by
    # launch, so SKIP the headroom check ("no opinion"), same as the
    # reconnect-handle path above. Probe failures keep the broad fail-OPEN.
    try:
        stale = _stale_named_instance_or_none(spec=spec, config=config, runner=runner)
        if stale is not None and stale.guest_phase is not None:
            logger.info(
                "GCP quota pre-check: RUNNING zombie %s (eps/phase=%s) occupies the "
                "canonical name — skipping the headroom check; launch's stale reclaim "
                "frees its quota before create (#908).",
                stale.name,
                stale.guest_phase,
            )
            return None
    except Exception as exc:  # GcpProbeError / GcpBackendError / transport — fail OPEN
        logger.warning(
            "GCP quota pre-check: stale-name probe failed OPEN (%s: %s); proceeding to launch.",
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
    max_age_seconds: int = 192 * 3600,  # 8d fallback fence (#741): 7d default + grace
    terminal_phase_max_age_seconds: int = 600,
    now: datetime | None = None,
    delete: bool = False,
    escalate: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """List (and optionally delete / escalate) stale / wedged project VMs.

    Analogue of ``scripts/pod.py audit-stale`` for GCP. Without it, an
    orchestrator crash that drops the local lease before teardown would
    leak a VM at $5/hr — the cron is the credit-leak backstop.

    Scope (#688): the inventory query lists the WHOLE dedicated project
    (:data:`JANITOR_LIST_NAME_FILTER` is ``None``), not only the
    router-managed ``eps-issue-*`` names — a non-``eps-issue-*`` leftover
    was previously invisible to the name filter and could idle-bill
    indefinitely. Each listed instance is classified
    (:func:`_classify_janitor_instance`) and routed by the HYBRID posture:

    * ``managed`` (``eps-issue-*``) + ``allowlisted-ephemeral`` (a known
      throwaway prefix) → AUTO-DELETE on the bounded fences below, exactly
      as the managed names always did;
    * ``unmanaged`` (anything else in the project) → ESCALATE, never
      auto-delete — ``action="would-escalate"`` (report-only) /
      ``"escalated"`` (when an ``escalate`` callback is supplied and the
      VM is stale). A stale ``reason`` (age / terminal-phase) must be set
      before escalation fires; a not-stale unmanaged VM is ``"skipped"``;
    * ``keep`` (an opt-out prefix) → ``"skipped"`` (record-only),
      never reaped or escalated.

    ``escalate`` is a keyword-only callable invoked once per UNMANAGED
    stale VM with its record dict (the CLI injects the real Telegram +
    sidecar escalator under ``--delete``; ``None`` keeps the library
    I/O-pure for tests and report-only smokes). It is keyword-only so the
    existing CLI tests' ``functools.partial(audit_stale_gcp_vms, now=now)``
    binding stays correct.

    Two reap predicates, both bounded so the sweep never deletes a
    legitimately in-flight VM:

    * **Age backstop** (per-instance-fence-aware, #741) — an instance is
      age-stale, REGARDLESS of phase (sets ``reason="age"``; the reap-vs-
      escalate split is decided by classification below), when EITHER it has
      a readable ``scheduling.maxRunDuration`` fence and has exceeded that
      fence + a 1h grace (:data:`_JANITOR_FENCE_GRACE_SECONDS`), OR it has NO
      readable fence and has lived past the fixed ``max_age_seconds`` fallback
      (the legacy fence; the library default and the ``gcp_audit.py`` CLI
      default are BOTH 8d — ``192 * 3600`` — to cover the 7d
      ``default_max_run_duration`` + grace). The last-resort fence for a VM
      whose ``--max-run-duration`` DELETE somehow never fired — now tracking
      each instance's OWN fence rather than a single fixed wall-clock (which
      would re-create #697: a 7d job killed by the janitor's old 24h cap).
    * **Terminal-phase reap** (``terminal_phase_max_age_seconds``, default
      10 min) — a RUNNING instance that has published a TERMINAL
      ``eps/phase`` (``done`` / ``failed`` /
      ``finalize_failed_artifacts_ok``; see :data:`_TERMINAL_GUEST_PHASES`)
      but never auto-deleted is a wedged
      zombie: the workload finished, the VM is still billing, and waiting
      for the per-fence age backstop (up to the instance's own
      ``--max-run-duration`` + grace — now 7d by default) burns idle A100
      hours (incident #634
      family — the sibling :func:`reconnect_or_none` fix only reaps such
      a zombie at the NEXT relaunch against the same name, which may never
      come). This predicate reaps it PROMPTLY (recorded as
      ``reason="terminal-phase"``). The short age floor keeps the sweep
      from racing a legitimate post-completion ``finalize`` (scp +
      teardown is ~30-60s) — only a VM that has sat terminal-phase past
      the floor is reaped. A guest-attribute PROBE FAILURE (``couldn't
      ask`` ≠ ``done``) is caught per-instance, logged, and falls through
      to the age backstop — a probe blip never escalates a still-unknown
      VM to deletion, and never crashes the rest of the inventory sweep.

    Returns a list of ``{name, zone, status, created_at, age_seconds,
    phase, reason, classification, action}`` records (``classification`` ∈
    {``"managed"``, ``"allowlisted-ephemeral"``, ``"unmanaged"``,
    ``"keep"``}; ``action`` ∈ {``"would-delete"``, ``"deleted"``,
    ``"would-escalate"``, ``"escalated"``, ``"skipped"``,
    ``"delete-failed"``}; ``reason`` ∈ {``"age"``, ``"terminal-phase"``,
    ``None``}). When ``delete=True``, REAP-class (managed /
    allowlisted-ephemeral) stale instances are issued a ``gcloud compute
    instances delete --quiet`` (errors are logged + folded into the record
    as ``action="delete-failed"`` — never raised, so the cron continues
    across the rest of the inventory); unmanaged stale instances are
    ``escalate``d (or ``"would-escalate"`` in report-only) and never
    deleted.

    No ``raise`` on a benign empty list — a fresh GCP project legitimately
    has zero matches.
    """
    cfg = config or default_gcp_config()
    run = runner or default_gcloud_runner
    reference = now or datetime.now(tz=UTC)
    argv = render_list_argv(config=cfg, name_filter=JANITOR_LIST_NAME_FILTER)
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
        if not name:
            continue
        classification = _classify_janitor_instance(name)
        zone_url = inst.get("zone") or ""
        zone = zone_url.rsplit("/", 1)[-1] if zone_url else cfg.primary_zone
        status = inst.get("status") or "UNKNOWN"
        created_at_raw = inst.get("creationTimestamp")
        age_seconds = _age_seconds(created_at_raw, reference)

        # ----- KEEP-prefix opt-out: never reaped OR escalated -----
        # Emit a "skipped" record (no phase probe, no reason) so the operator
        # sees the janitor inspected it and deliberately left it alone.
        if classification == JANITOR_CLASS_KEEP:
            phase, reason = None, None
        else:
            phase, reason = _janitor_stale_reason(
                cfg=cfg,
                run=run,
                name=name,
                zone=zone,
                status=status,
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
                terminal_phase_max_age_seconds=terminal_phase_max_age_seconds,
                instance_max_run_seconds=_instance_max_run_seconds(inst),  # #741 Option B
            )

        record: dict[str, Any] = {
            "name": name,
            "zone": zone,
            "status": status,
            "created_at": created_at_raw,
            "age_seconds": age_seconds,
            "phase": phase,
            "reason": reason,
            "classification": classification,
            "action": "skipped",  # overwritten by the router below when stale
        }
        # keep → always skipped; otherwise route reap-vs-escalate on a stale reason.
        if classification != JANITOR_CLASS_KEEP:
            record["action"] = _janitor_route_action(
                cfg=cfg,
                run=run,
                record=record,
                classification=classification,
                reason=reason,
                delete=delete,
                escalate=escalate,
            )
        records.append(record)
    return records


def _instance_max_run_seconds(inst: dict[str, Any]) -> int | None:
    """Return the instance's configured ``--max-run-duration`` in seconds, or None.

    Reads ``scheduling.maxRunDuration.seconds`` from the instance JSON that
    ``gcloud compute instances list --format=json`` returns — GCP populates
    this natively whenever the create passed ``--max-run-duration`` (the v1
    REST instance schema carries ``scheduling.maxRunDuration: {seconds,
    nanos}``). gcloud emits ``seconds`` as either an int or a numeric string,
    so both are accepted; any other shape (absent block, junk value) returns
    None so the caller falls back to the fixed ``max_age_seconds`` fence — the
    backstop never silently disarms on a missing or malformed field.
    """
    sched = inst.get("scheduling")
    if not isinstance(sched, dict):
        return None
    mrd = sched.get("maxRunDuration")
    if not isinstance(mrd, dict):
        return None
    raw = mrd.get("seconds")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _janitor_stale_reason(
    *,
    cfg: GcpConfig,
    run: GcloudRunner,
    name: str,
    zone: str,
    status: str,
    age_seconds: float | None,
    max_age_seconds: int,
    terminal_phase_max_age_seconds: int,
    instance_max_run_seconds: int | None,
) -> tuple[str | None, str | None]:
    """Decide whether a janitor instance is stale, and why → ``(phase, reason)``.

    Age backstop (Option B, #741): an instance is age-stale when EITHER it has
    a readable ``--max-run-duration`` fence (``instance_max_run_seconds``, from
    :func:`_instance_max_run_seconds`) and has exceeded that fence + a 1h grace
    (:data:`_JANITOR_FENCE_GRACE_SECONDS`), OR it has NO readable fence and has
    lived past the fixed ``max_age_seconds`` fallback (the legacy fence, raised
    to 8d to cover the 7d default + grace). This makes the age backstop track
    each instance's OWN fence — a 24h job reaps at ~25h, a 7d job at ~7d+1h, a
    truly-wedged-never-terminal VM at its-fence+grace — instead of a single
    fixed wall-clock that would re-create the #697 bug (a 7d job killed by the
    janitor's fixed 24h cap).

    ``reason`` ∈ {``"age"``, ``"terminal-phase"``, ``None``}. The phase probe
    fires ONLY for a RUNNING VM that is NOT already age-reaped and has lived
    past the terminal-phase floor — a terminal-phase RUNNING zombie is reaped
    promptly, well under the age backstop. A guest-attribute probe FAILURE
    ("couldn't ask" ≠ "done") is caught, logged, and falls through (returns
    not-stale) — a probe blip never escalates a still-unknown VM and never
    crashes the inventory sweep.
    """
    phase: str | None = None
    if age_seconds is not None:
        if instance_max_run_seconds is not None:
            age_fence = instance_max_run_seconds + _JANITOR_FENCE_GRACE_SECONDS
        else:
            age_fence = max_age_seconds
        if age_seconds >= age_fence:
            return None, "age"
    should_probe_phase = (
        status.upper() == "RUNNING"
        and age_seconds is not None
        and age_seconds >= terminal_phase_max_age_seconds
    )
    if not should_probe_phase:
        return None, None
    try:
        phase = _read_guest_phase(config=cfg, name=name, zone=zone, runner=run)
    except GcpProbeError as exc:
        logger.warning(
            "audit_stale_gcp_vms: eps/phase probe failed for %s (%s); "
            "treating phase as UNKNOWN and falling through to the age "
            "backstop — never reaping on a probe failure.",
            name,
            exc,
        )
        return None, None
    reason = "terminal-phase" if phase in _TERMINAL_GUEST_PHASES else None
    return phase, reason


def _janitor_route_action(
    *,
    cfg: GcpConfig,
    run: GcloudRunner,
    record: dict[str, Any],
    classification: str,
    reason: str | None,
    delete: bool,
    escalate: Callable[[dict[str, Any]], None] | None,
) -> str:
    """Map a (classification, stale-reason) pair to the record's ``action``.

    HYBRID posture: not-stale → ``"skipped"`` for every class; an UNMANAGED
    stale VM is ESCALATEd (never auto-deleted); a managed / allowlisted-ephemeral
    stale VM is AUTO-DELETEd on the bounded fences.
    """
    if reason is None:
        return "skipped"
    if classification == JANITOR_CLASS_UNMANAGED:
        # ESCALATE, never auto-delete: an instance the janitor cannot
        # positively classify as throwaway is treated like active data.
        if not delete:
            return "would-escalate"
        if escalate is not None:
            escalate(record)
        return "escalated"
    # managed / allowlisted-ephemeral → AUTO-DELETE on the fences.
    if not delete:
        return "would-delete"
    del_result = run(render_delete_argv(config=cfg, name=record["name"], zone=record["zone"]))
    if del_result.returncode == 0:
        return "deleted"
    logger.error(
        "audit_stale_gcp_vms: delete %s failed (%d): %s",
        record["name"],
        del_result.returncode,
        del_result.stderr[:300],
    )
    return "delete-failed"


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

    def launch(self, spec: RunSpec) -> RunHandle:  # noqa: C901 — reconnect + zone fan-out + create-timeout + per-zone visibility (#774); the per-zone record/attach steps are already extracted to module helpers, the residual branches ARE the provisioning state machine.
        """Provision (or reconnect to) the GCE VM for ``spec.issue``.

        See class docstring for the per-step flow. Raises
        :class:`GcpProvisioningError` when every zone (primary +
        fallbacks) returns a capacity / quota / auth failure — the router
        catches that and proceeds to the next tier. The error raised after
        a full-fan-out capacity miss carries
        ``evidence["per_zone_attempts"]`` (one
        ``{zone, returncode, matched_pattern, elapsed_s, stderr_tail}`` record
        per zone tried, in fan-out order), the derived bare-name
        ``evidence["zones_attempted"]``, and a human-readable
        ``evidence["zones_attempted_summary"]`` so a post-mortem reads the
        complete fan-out with per-zone why/how-long, not just the last zone's
        stderr (#763/#774). On the success-after-miss path the preceding zone
        misses are threaded onto ``handle.extra["per_zone_attempts"]``.
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

        # Reclaim the canonical name from a STALE non-live record before
        # ``create``. ``reconnect_or_none`` returns None for a TERMINATED /
        # STOPPED / SUSPENDED instance (correctly — it is not a live run to
        # rejoin), but that record still OWNS the ``eps-issue-<N>`` name, so
        # the upcoming create would be rejected with ``resource ... already
        # exists`` (incident #632, 2026-06-13: a workload-crash respawn left
        # a TERMINATED VM, and the infra-class respawn's create was rejected
        # until the operator manually deleted it). Delete the stale record
        # here; the EXIT-trap teardown is per-INSTANCE, not name-scoped, so
        # nothing else reclaims the name. A possibly-live status raises out
        # of the probe (never auto-deleted — see _stale_named_instance_or_none).
        pre_launch_deleted_stale_instance = False
        stale = _stale_named_instance_or_none(spec=spec, config=config, runner=self._run)
        if stale is not None:
            logger.info(
                "GCP pre-launch: deleting stale %s instance %s (eps/phase=%s) in zone=%s "
                "to free the name before create (issue=%d).",
                stale.status,
                stale.name,
                stale.guest_phase,
                stale.zone,
                spec.issue,
            )
            del_result = self._run(
                render_delete_argv(config=config, name=stale.name, zone=stale.zone)
            )
            del_stderr_low = (del_result.stderr or "").lower()
            if del_result.returncode == 0 or (
                "was not found" in del_stderr_low or "404" in del_stderr_low
            ):
                # Deleted (or it vanished between the list and the delete —
                # either way the name is now free). "was not found" is the
                # benign race, same as teardown's handling.
                pre_launch_deleted_stale_instance = True
                logger.info(
                    "GCP pre-launch: stale instance %s reclaimed (rc=%d); proceeding to create.",
                    stale.name,
                    del_result.returncode,
                )
            else:
                # A real delete failure (auth blip, transient API error).
                # Raise rather than let create fail later with a more
                # confusing "already exists" — the name was NOT freed.
                raise GcpBackendError(
                    f"GCP pre-launch: failed to delete stale {stale.status} instance "
                    f"{stale.name} (rc={del_result.returncode}): {del_result.stderr[:500]} "
                    "— the canonical name was not freed; create would collide."
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
        # Drop zones where the resolved machine type is NOT offered (#653):
        # the uniform fallback list is not valid for every machine type
        # (the A2-ultragpu family is absent from us-central1-b), so a blind
        # fallback would issue a guaranteed-to-fail create (a CONFIG 400,
        # not a capacity miss) and burn the per-day GCP attempt counter.
        machine_type = machine_for_intent(spec).machine_type
        filtered = zones_for_machine_type(machine_type, zones_to_try)
        if filtered != zones_to_try:
            logger.info(
                "GCP issue=%d: machine-type %s unavailable in %s; zone ladder trimmed to %s",
                spec.issue,
                machine_type,
                [z for z in zones_to_try if z not in filtered],
                filtered,
            )
        zones_to_try = filtered
        if not zones_to_try:
            # No configured zone offers this machine type — fail LOUD
            # rather than fall through to a zero-zone for-loop that would
            # raise an opaque ``last_error is None`` assert.
            raise GcpProvisioningError(
                f"no configured us-central1 zone offers machine type {machine_type!r} "
                f"for intent {spec.intent!r} (primary={config.primary_zone}, "
                f"fallbacks={config.fallback_zones}). Update "
                f"backends/gcp.MACHINE_TYPE_ZONE_AVAILABILITY / GcpConfig zones, or "
                f"route to another backend.",
                evidence={"machine_type": machine_type, "intent": spec.intent},
            )
        last_error: GcpProvisioningError | None = None
        # Per-zone create outcome records, in fan-out order (#774). Each entry is
        # {zone, returncode, matched_pattern, elapsed_s, stderr_tail}; threaded
        # onto the GcpProvisioningError raised after the loop (and onto the
        # success-after-miss handle.extra) so a post-mortem reads the FULL zone
        # fan-out ("tried -a (exhausted), landed -c") with per-zone WHY/HOW-LONG
        # instead of only the last zone's stderr. Pre-fix (#763/#774) the marker
        # surfaced only `classify_create_failure`'s last-zone evidence, so a
        # reader could not tell whether the loop tried every available zone or
        # gave up after one (the #763 single-zone misdiagnosis). The bare
        # zone-name list (`zones_attempted` below) is derived from this and kept
        # for back-compat with existing readers/tests.
        per_zone_outcomes: list[dict[str, Any]] = []

        try:
            for zone in zones_to_try:
                _zone_started = time.monotonic()
                argv = render_create_argv(
                    spec=spec,
                    config=config,
                    attempt_id=attempt_id,
                    zone=zone,
                    startup_script=startup,
                    secret_files=secret_files,
                )
                logger.info("GCP create issue=%d in zone=%s", spec.issue, zone)
                try:
                    result = self._run(argv)
                except subprocess.TimeoutExpired as exc:
                    # FLEX_START rungs legitimately stay PENDING past the
                    # GCLOUD_DEFAULT_TIMEOUT_SEC subprocess cap, so the create
                    # OFTEN succeeds server-side even though the local
                    # subprocess.run aborted (#658 failure-lesson v2). Probe the
                    # canonical name BEFORE treating the timeout as a failure;
                    # pre-fix this TimeoutExpired propagated raw to main()'s
                    # exit-4 catch-all (#736).
                    live = self._reconnect_probe_after_create_timeout(spec, exc)
                    if live is not None:
                        # Instance live server-side → a still-waiting outcome,
                        # NOT a failure. Raise immediately (no continue into the
                        # next zone → no double-create); the CLI converts this to
                        # exit 75 and the re-run reconnects idempotently.
                        raise GcpCreateTimedOutStillProvisioning(
                            instance_name=live.pod_name,
                            status=str(live.extra.get("status_at_reconnect") or "PROVISIONING"),
                            issue=int(spec.issue),
                        ) from exc
                    # Instance truly absent → a provision-class failure. Build a
                    # capacity-shaped GcpProvisioningError (matched_pattern
                    # contains "resource" so the zone-retry predicate below
                    # matches) so the existing zone fallback / next-tier router
                    # path handles it exactly like any other create miss; never
                    # let the raw TimeoutExpired escape to main()'s exit-4
                    # catch-all. The slug is the POSITIONAL reason (the
                    # GcpProvisioningError constructor has no separate message
                    # arg); descriptive prose lives in evidence["detail"],
                    # mirroring classify_create_failure.
                    last_error = GcpProvisioningError(
                        "create_timeout_no_instance",
                        evidence={
                            "zone": zone,
                            "timeout_sec": GCLOUD_DEFAULT_TIMEOUT_SEC,
                            "issue": int(spec.issue),
                            "matched_pattern": "create timeout (resource not created)",
                            "detail": (
                                f"gcloud create for issue={spec.issue} in zone={zone} timed "
                                f"out at the {GCLOUD_DEFAULT_TIMEOUT_SEC}s subprocess cap and "
                                "no instance was found server-side after the timeout — "
                                "treating as a capacity/provisioning failure."
                            ),
                        },
                    )
                    _record_zone_outcome(
                        per_zone_outcomes,
                        zone,
                        returncode=-1,
                        matched_pattern="create timeout (resource not created)",
                        started=_zone_started,
                        stderr="create timeout, no stderr captured",
                    )
                    logger.warning(
                        "GCP create timed out in zone=%s with no server-side instance; "
                        "trying next fallback. issue=%d",
                        zone,
                        spec.issue,
                    )
                    continue
                if result.returncode == 0:
                    _record_zone_outcome(
                        per_zone_outcomes,
                        zone,
                        returncode=0,
                        matched_pattern=None,
                        started=_zone_started,
                    )
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
                _record_zone_outcome(
                    per_zone_outcomes,
                    zone,
                    returncode=result.returncode,
                    matched_pattern=last_error.evidence.get("matched_pattern"),
                    started=_zone_started,
                    stderr=result.stderr,
                )
                if not any(tag in matched for tag in ("exhaust", "resource", "enough resources")):
                    # Non-capacity failure → don't retry; surface immediately. Carry
                    # the partial per-zone trail (zones tried so far) onto the error
                    # so a non-capacity raise on a later zone still shows the earlier
                    # capacity misses (#774 — auth/quota path). No summary: not an
                    # all-zones-exhausted miss.
                    _attach_fanout_evidence(
                        last_error,
                        per_zone_outcomes,
                        machine_type=machine_type,
                        with_summary=False,
                    )
                    raise last_error
                logger.warning(
                    "GCP create capacity miss in zone=%s; trying next fallback. reason=%s",
                    zone,
                    last_error.reason,
                )
            else:
                # for-else: executed when the for loop completes without
                # `break` — every zone failed. Name the FULL zone fan-out on the
                # raised error (rich per_zone_attempts + derived zones_attempted +
                # the human-readable summary) so the marker / terminal JSON shows
                # every zone tried with per-zone WHY/HOW-LONG, not just the last
                # zone's stderr (#763/#774 observability gap).
                assert last_error is not None
                _attach_fanout_evidence(
                    last_error,
                    per_zone_outcomes,
                    machine_type=machine_type,
                    with_summary=True,
                )
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
        instance_name = instance_name_for(spec.issue, lane_suffix_for(spec))
        # gcloud returns the instance object as a list with one entry.
        instance_id = _parse_instance_id(result.stdout, instance_name)
        handle = RunHandle(
            backend="gcp",
            cluster=None,
            job_id=instance_id,
            pod_name=instance_name,
            scratch_dir=workload_dir_for(config, spec.issue),
            log_path=log_path_for(config, spec.issue),
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
                # ASYNC RunPod failover prerequisite (#659, MF1+MF2): thread the
                # spec fields the poller needs to reconstruct a RunSpec for the
                # RunPod re-launch when this VM crashes its workload minutes in
                # (the case the synchronous route()-time failover cannot reach).
                # The fact-checker confirmed (A7) the pre-#659 ``extra`` did NOT
                # carry the workload command, so ``backend_poll._runspec_from_gcp_handle``
                # had no way to rebuild it. ``serialize_handle``/``deserialize_handle``
                # round-trip ``extra`` faithfully, so these survive the sidecar
                # write/read with no serializer change.
                #
                # MF1: ``workload_cmd`` is a ``str`` (``base.py``: ``str = ""``), so
                # preserve it AS-IS — ``list(spec.workload_cmd or ())`` would explode a
                # real command STRING into a per-character list. ``hydra_args`` IS a
                # tuple, so coerce to ``list`` for JSON round-trip faithfulness; one of
                # the two is empty by construction (RunSpec.__post_init__ mutual
                # exclusion), which the poller's reconstruction relies on (MF2).
                "workload_cmd": spec.workload_cmd or "",
                "hydra_args": list(spec.hydra_args or ()),
                "gpus": spec.gpus,
                "time_budget_hours": spec.time_budget_hours,
                # #909: the branch the run's code lives on, so the async
                # GCP→RunPod failover reconstruction
                # (backend_poll._runspec_from_gcp_handle) re-executes against
                # the ISSUE branch, not `main` (per-issue dispatch scripts live
                # on issue branches). Additive key; "" when unset.
                "repo_branch": str(spec.extra.get("repo_branch") or ""),
                # CPU-lane async-failover guard prerequisite (#677). The async
                # poller's _is_gcp_async_workload_failure must EXCLUDE a CPU GCP
                # handle (gpu_count==0) from the GCP->RunPod failover (RunPod is
                # GPU-only — resolve_intent on a CPU intent KeyErrors,
                # runpod_api.py asserts gpu_count >= 1). The handle already
                # carries intent + machine_type but NOT the resolved gpu_count;
                # thread it so the predicate reads handle.extra["gpu_count"]
                # WITHOUT importing gcp into the poller. Resolved from the SAME
                # machine_for_intent(spec) the create used, so a
                # machine_spec_override rung reports its true gpu_count (an
                # A100-40 rung stays gpu_count>=1; only a cpu-bigmem rung is 0).
                # Additive key — extra is an opaque dict that round-trips via
                # serialize_handle's dict(handle.extra); every existing reader
                # uses .get(...), so no consumer breaks.
                "gpu_count": machine_for_intent(spec).gpu_count,
                # #1010: footprint fields for the RunPod CPU-fallback
                # feasibility gate + container-disk threading — forwarded by
                # backend_poll._runspec_from_gcp_handle on the async failover
                # paths (#659 crash / #783 queue-timeout). Keys OMITTED when
                # absent/falsy — never a None value — so legacy handle shapes
                # stay byte-identical.
                **{
                    k: v
                    for k, v in {
                        "boot_disk_gb": spec.extra.get("boot_disk_gb"),
                        "min_ram_gb": spec.extra.get("min_ram_gb"),
                    }.items()
                    if v
                },
            },
        )
        handle = self._with_artifacts_declaration(
            handle=handle,
            spec=spec,
            config=config,
            attempt_id=attempt_id,
            wandb_run_path=spec.extra.get("wandb_run_path"),
        )
        # #774: surface the per-zone create trail on the success path when an
        # earlier zone stocked out before THIS one landed (a "tried -a
        # (exhausted), landed -c" launch). `per_zone_outcomes[:-1]` is every
        # zone that MISSED before the landing zone — the last entry is the
        # successful zone, already recorded in handle.extra["zone"]. Emitted
        # only when ≥2 zones were tried so handle.extra stays byte-identical on
        # the no-miss happy path (first-zone-success → per_zone_outcomes ==
        # [landed-zone], len 1, no key added). `handle.extra` is a mutable dict
        # by design (base.RunHandle.extra: dict, field(default_factory=dict)),
        # so assign in place rather than dataclasses.replace.
        if len(per_zone_outcomes) >= 2:
            handle.extra["per_zone_attempts"] = per_zone_outcomes[:-1]

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
                # Additive field (#632): True when this launch first deleted
                # a STALE non-live (TERMINATED / STOPPED / SUSPENDED) record
                # occupying the canonical name before create — so the
                # name-reclaim is visible on the timeline.
                "pre_launch_deleted_stale_instance": pre_launch_deleted_stale_instance,
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

    def _reconnect_probe_after_create_timeout(
        self, spec: RunSpec, original_timeout: subprocess.TimeoutExpired
    ) -> RunHandle | None:
        """Post-create-timeout probe: is the ``eps-issue-<N>`` instance live?

        Returns the reconnect handle when a live instance (RUNNING /
        PROVISIONING / STAGING / STOPPING) is found, ``None`` when no
        instance exists. On a PROBE failure — :class:`GcpProbeError`
        (gcloud rc != 0 / bad JSON / expired auth) OR a hung
        ``instances list`` raising :class:`subprocess.TimeoutExpired` —
        RE-RAISES as a :class:`GcpProvisioningError` chained from the
        ORIGINAL create timeout: "couldn't ask" must NEVER read as "live"
        (would mask a real failure as still-waiting) NOR silently as
        "absent" (would lose the timeout signal). Provision-class so the
        router fallback handles it.

        The ``except`` catches BOTH ``GcpProbeError`` AND
        ``subprocess.TimeoutExpired``: :func:`reconnect_or_none` raises
        ``GcpProbeError`` on rc != 0 / bad JSON but does NOT wrap a hung
        list's ``TimeoutExpired`` — so narrowing this to
        ``except GcpProbeError`` would let a list-timeout escape raw to
        ``main()``'s exit-4 catch-all (the #736 bug on the probe-timeout
        branch). The chained ``from original_timeout`` (NOT the probe-side
        exception) makes ``__cause__`` trace back to the real cause — the
        create that timed out.
        """
        try:
            return reconnect_or_none(spec=spec, config=self._config, runner=self._run)
        except (GcpProbeError, subprocess.TimeoutExpired) as probe_exc:
            raise GcpProvisioningError(
                "create_timeout_probe_failed",
                evidence={
                    "issue": int(spec.issue),
                    "matched_pattern": "create timeout probe failed",
                    "detail": (
                        f"gcloud create for issue={spec.issue} timed out and the "
                        f"post-timeout instances-list probe ALSO failed ({probe_exc}); "
                        "instance state UNKNOWN — treating as a provisioning failure."
                    ),
                },
            ) from original_timeout

    # ----- monitor ---------------------------------------------------------

    def poll(self, handle: RunHandle) -> PollResult:
        """One-tick poll via ``gcloud compute instances describe``.

        Slice 3 returns a coarse PollResult derived from the VM status
        only (``RUNNING`` → ``running``; ``TERMINATED`` → ``dead`` etc.).
        Slice 6 will overlay the per-phase heartbeat once the in-VM
        ``[phase=...]`` writes land on a poll-readable surface (the
        existing :class:`PollResult` shape carries the per-phase fields).

        Terminal guest-attribute phases (``done`` / ``failed`` /
        ``finalize_failed_artifacts_ok``) are
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
            (
                drained,
                drain_gate,
                drain_alarm,
                drain_log_tail,
                drain_log_mtime_ago,
                drain_alarm_class,
            ) = self._drain_sentinels(handle, zone)

            def _with_drain(base: PollResult) -> PollResult:
                return _overlay_drain(
                    base,
                    processed=drained,
                    gate=drain_gate,
                    alarm=drain_alarm,
                    log_tail=drain_log_tail,
                    log_mtime_ago=drain_log_mtime_ago,
                )

            # A RUNNING VM is ambiguous: booting, mid-workload, or DONE
            # (the success path deliberately keeps the VM up so the
            # completion sentinel can be scp'd — bounded by the #935
            # done-grace self-poweroff — instance state alone
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
            if phase in ("done", "failed", "finalize_failed_artifacts_ok"):
                # Relaunch-follow (incident #612): the eps/phase guest
                # attribute is written by the STARTUP SCRIPT, so it
                # freezes at the FIRST workload's terminal state. A
                # sanctioned SSH hot-fix relaunch (CLAUDE.md "push
                # through bugs", the experimenter respawn path) posts a
                # fresh ``epm:run-launched`` marker with ``pid=`` +
                # ``log_abs=`` precisely so pollers can follow the new
                # process — without this branch a HEALTHY mid-training
                # relaunch read as ``done``/``dead`` and steered the
                # orchestrator to a premature transition. (#1055: the
                # new finalize_failed_artifacts_ok phase is a terminal
                # state of the FIRST workload too — a fresh relaunch
                # marker must win over the done-like classification
                # below, or a healthy relaunch reads done and the
                # orchestrator finalizes mid-run.)
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
            if phase == "finalize_failed_artifacts_ok":
                # #1055 — additive, mirrors the #935 stance: a run whose
                # deliverables verified complete on HF BEFORE a
                # finalize/tail non-zero exit (the trap's positive-evidence
                # branch). Classified as a SUCCESSFUL run whose finalize
                # hiccupped — status="done" fails BOTH async-failover
                # conjuncts by construction (backend_poll requires
                # status=="dead" + a terminal_* phase), so no RunPod
                # failover and no crash-fix routing; the distinct
                # current_phase keeps the finalize failure visible for
                # triage. Reachable in the brief RUNNING window between the
                # trap's phase write and the poweroff completing. #1151: the
                # trap ran _eps_persist_diagnostics on this arm too — append
                # the eps/persist breadcrumb (RUNNING qualifier: the persist
                # may still be in flight).
                return self._append_persist_breadcrumb(
                    _with_drain(
                        PollResult(
                            status="done",
                            current_phase="workload_done_finalize_failed",
                            new_milestone=True,
                            last_log_mtime_sec_ago=0,
                            pid_alive=False,
                            log_tail_excerpt="",
                        )
                    ),
                    handle,
                    zone,
                    instance_status="RUNNING",
                )
            if phase == "failed":
                # Workload-vs-setup discrimination (#659, MF3): ``eps/phase`` is
                # single-valued and the EXIT trap overwrites it to ``failed`` on
                # ANY non-zero exit, so ``failed`` alone cannot tell a real
                # workload crash from a pre-workload setup failure. The
                # write-once ``eps/workload_started`` sentinel (a DIFFERENT key,
                # so it survives the ``failed`` overwrite) resolves it: present →
                # ``terminal_workload_failed`` (the async predicate fails over to
                # RunPod); absent → ``terminal_setup_failed`` (NOT a workload
                # crash, so NO failover — re-running broken boot/setup on RunPod
                # just re-crashes). A probe failure on the sentinel read falls
                # back conservatively to workload-started (see
                # :meth:`_workload_started`).
                reason = (
                    "workload_failed" if self._workload_started(handle, zone) else "setup_failed"
                )
                # #1151: surface the eps/persist crash-persist breadcrumb on
                # every failed-classifying tick (diagnostic-only; the RUNNING
                # qualifier flags that the persist may still be in flight).
                return self._append_persist_breadcrumb(
                    _with_drain(_terminal_dead_poll(reason=reason)),
                    handle,
                    zone,
                    instance_status="RUNNING",
                )
            if phase:
                # Adaptive bg-poll interval (§7) — the GCP lane's quiet
                # heuristic applies ONLY to this known-mid-workload-phase
                # running branch. Run age comes from the describe payload's
                # ``creationTimestamp`` (zero extra round-trips; a missing /
                # unparseable timestamp reads as early-run → short). The
                # drain alarm (transport / permission failure, skipped
                # drain) is the lane anomaly — degraded observability never
                # goes quiet. Computed on the MERGED post-drain result so a
                # drained gate or sentinel activity forces the short
                # interval through the helper's own conditions. Every other
                # branch (booting with no phase yet, relaunched workload,
                # PROVISIONING/STAGING, non-running) keeps the short
                # default by construction.
                merged = _with_drain(_coarse_poll(status="running", current_phase=phase))
                # M1a (#669): surface a TYPED reachability_alarm set ONLY for
                # the transport class — the unreachable-VM signature the
                # poller's frozen-phase wedge gate reads. A
                # sentinel-processing-class alarm (healthy VM, noisy sentinel)
                # leaves it False so the wedge gate never fires on it.
                merged = replace(merged, reachability_alarm=(drain_alarm_class == "transport"))
                return _apply_lane_quiet_interval(
                    merged,
                    run_age_sec=_age_seconds(payload.get("creationTimestamp"), datetime.now(UTC)),
                    # lane_anomaly (poll INTERVAL) keeps BOTH alarm classes —
                    # degraded observability of any kind should keep the poll
                    # interval short. ONLY the wedge classifier reads the
                    # narrower reachability_alarm.
                    lane_anomaly=bool(drain_alarm),
                )
            return _with_drain(_gcp_status_to_poll_result(status))
        return self._non_running_poll_result(handle, zone, status)

    def _non_running_poll_result(self, handle: RunHandle, zone: str, status: str) -> PollResult:
        """Resolve a non-RUNNING VM status, with the #669/#935 phase discrimination.

        Watchdog self-terminate discrimination (#669, Consistency-checker
        Option 2): a TERMINATED VM whose in-VM reachability watchdog wrote
        ``eps/phase=wedged`` before ``shutdown -h now`` maps to the NEW
        ``terminal_wedged_terminated`` phase, which the poller's async failover
        accept-set recognizes.

        Done-grace discrimination (#935): a TERMINATED VM whose ``eps/phase``
        reads ``done`` is a SUCCESSFUL run whose done-grace self-poweroff (or
        a manual stop of a done VM) fired — classify ``done``
        (``workload_done_self_poweroff``), never dead, so a revived
        orchestrator does not spin crash-fix machinery on a run whose
        artifacts are all on HF. Only reachable when the guest shutdown ends
        in STOP rather than DELETE (or during the transient STOPPING window);
        a DELETEd instance 404s at describe and stays ``dead("instance not
        found")`` — the recovery breadcrumb there is the HF
        ``issue<N>_done/<attempt_id>/`` prefix. ``fetch_results``' sentinel
        scp will fail (VM off; fail-soft by contract) and finalize needs
        ``--skip-confirm-artifacts``.

        Setup-death discrimination (#1029): a TERMINATED VM whose ``eps/phase``
        reads ``failed`` runs the SAME §4.1.0b ``workload_started``
        discrimination the RUNNING path runs — sentinel ABSENT ⇒
        ``terminal_setup_failed`` (the classification is then
        timing-independent: the identical trap-written boot death no longer
        reads ``terminal_setup_failed`` in the brief RUNNING window but
        ``terminal_terminated`` after shutdown); sentinel present ⇒ a mid-run
        guest shutdown (e.g. a spot preemption whose EXIT trap completed) —
        KEEP ``terminal_terminated``, the #669 exclusion verbatim. A probe
        failure on the sentinel read falls back to workload-started (its
        existing contract) ⇒ keeps ``terminal_terminated`` (conservative:
        never manufactures a setup classification).

        Finalize-failed-but-artifacts-ok discrimination (#1055): a TERMINATED
        VM whose ``eps/phase`` reads ``finalize_failed_artifacts_ok`` — the
        EXIT trap's positive-evidence branch: the workload stamped
        ``$EPS_DELIVERABLES_OK_PATH`` after its final upload+verify PASS,
        then a finalize/tail step exited non-zero — classifies ``done``
        (``workload_done_finalize_failed``), never dead, mirroring the #935
        stance: the deliverables are complete on HF, so neither the #659
        async failover nor crash-fix routing should fire; finalize needs
        ``--skip-confirm-artifacts`` (the completion sentinel was never
        written) and Step 8 upload verification stays the independent gate.

        A TERMINATED VM with any other (or absent / unreadable) ``eps/phase``
        maps to ``terminal_terminated`` EXACTLY as today (spot preemption /
        max-run-duration / manual mid-run stop → straight to dead, NO
        failover — no spot regression). Mirrors the ``eps/phase=failed`` +
        ``workload_started`` discrimination on the RUNNING path. Every other
        non-RUNNING status falls through to the coarse mapping unchanged.
        """
        if status == "TERMINATED":
            try:
                phase = self._guest_phase(handle, zone)
            except GcpProbeError:
                # Guest attribute unreadable (instance record already gone /
                # auth-probe failure): the combo can't match → fall through to
                # the safe default ``terminal_terminated`` (no failover; the
                # poller-side wedge detector remains the backstop, §1.bis).
                phase = ""
            if phase == "wedged":
                return _terminal_dead_poll(reason="wedged_terminated")
            # #1029: the same §4.1.0b discrimination the RUNNING path runs
            # (see poll()'s ``phase == "failed"`` branch) — makes the
            # NOT-STARTED case's classification timing-independent (the
            # probe-failure and started=True cases remain on the
            # terminal_terminated default — both today's behavior,
            # deliberately preserved). _workload_started falls back to True on
            # a probe failure by its existing contract -> keeps
            # terminal_terminated (never manufactures a setup classification).
            # A "failed" phase WITH the workload started (a mid-run guest
            # shutdown, e.g. a spot preemption whose trap completed) falls
            # through to terminal_terminated — the #669 exclusion verbatim.
            if phase == "failed" and not self._workload_started(handle, zone):
                # #1151: append the eps/persist crash-persist breadcrumb on the
                # TERMINATED failed windows — the moment the terminal diagnosis
                # reads it (guest attributes survive TERMINATED; lost at DELETE).
                return self._append_persist_breadcrumb(
                    _terminal_dead_poll(reason="setup_failed"),
                    handle,
                    zone,
                    instance_status="TERMINATED",
                )
            if phase == "done":
                # #935 — purely ADDITIVE: do NOT refactor the existing
                # ``workload_done`` / ``relaunched_workload_done`` literals
                # (tests string-assert them). The new phase fails BOTH
                # async-failover conjuncts (status=="dead" + a terminal_*
                # phase), so no failover misfire is possible.
                return PollResult(
                    status="done",
                    current_phase="workload_done_self_poweroff",
                    new_milestone=True,
                    last_log_mtime_sec_ago=0,
                    pid_alive=False,
                    log_tail_excerpt="",
                )
            if phase == "finalize_failed_artifacts_ok":
                # #1055 — additive, same stance as the #935 block above: the
                # trap's positive-evidence branch proved the deliverables
                # complete on HF before the finalize/tail crash, so this is
                # a SUCCESSFUL run whose finalize hiccupped. status="done"
                # fails BOTH async-failover conjuncts by construction; the
                # distinct current_phase keeps the finalize failure visible
                # for triage. #1151: the trap ran _eps_persist_diagnostics on
                # this arm too — append the eps/persist breadcrumb.
                return self._append_persist_breadcrumb(
                    PollResult(
                        status="done",
                        current_phase="workload_done_finalize_failed",
                        new_milestone=True,
                        last_log_mtime_sec_ago=0,
                        pid_alive=False,
                        log_tail_excerpt="",
                    ),
                    handle,
                    zone,
                    instance_status="TERMINATED",
                )
            if phase == "failed":
                # failed + workload-started: the coarse terminal_terminated
                # mapping below (a mid-run guest shutdown whose EXIT trap
                # completed — the #669 exclusion verbatim, classification
                # unchanged), #1151-augmented with the eps/persist breadcrumb:
                # the trap ran _eps_persist_diagnostics on this path too.
                return self._append_persist_breadcrumb(
                    _gcp_status_to_poll_result(status),
                    handle,
                    zone,
                    instance_status="TERMINATED",
                )
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

    def _workload_started(self, handle: RunHandle, zone: str) -> bool:
        """Did the WORKLOAD phase get reached? Reads ``eps/workload_started`` (#659).

        The workload-phase preamble publishes the write-once
        ``eps/workload_started`` guest attribute (a DIFFERENT key from
        ``eps/phase``, so it survives the EXIT trap's ``eps/phase=failed``
        overwrite). Its presence PROVES the workload ran, so a
        ``phase==failed`` poll is a REAL workload crash (failover to RunPod)
        rather than a setup/secrets/clone/uv-sync failure (do NOT fail over —
        re-running a broken boot script on RunPod just re-crashes).

        Three outcomes, mapped exactly like ``_guest_phase`` types its probe:

        * sentinel present (``"true"``) → ``True`` (workload was reached).
        * sentinel ABSENT — gcloud exits non-zero with a 404 / "not found"
          stderr (the EXPECTED not-written-yet case) → ``False`` (setup
          failure: the workload never ran).
        * PROBE FAILURE — any OTHER non-zero rc (auth / permission /
          transport) or unparseable JSON. "Couldn't ask" must NEVER read as
          "setup failed" (that would suppress a legitimate workload failover),
          so this returns ``True`` — the CONSERVATIVE fallback to the existing
          ``terminal_workload_failed`` mapping (which routes to ``blocked``,
          the pre-#659 behavior). It NEVER raises.
        """
        config = self._config
        argv = render_guest_attributes_argv(
            config=config, name=handle.pod_name, zone=zone, query_path="eps/workload_started"
        )
        result = self._run(argv)
        if result.returncode != 0:
            stderr_low = (result.stderr or "").lower()
            if "not found" in stderr_low or "404" in stderr_low:
                # Attribute never written → the workload phase was not reached.
                return False
            # Probe FAILURE — conservative fallback: assume workload-started so a
            # legitimate workload crash still fails over (never misread an
            # unprovable read as "setup failed").
            logger.warning(
                "GCP poll: eps/workload_started probe failed for %s (rc=%s); "
                "conservatively assuming workload-started (failover-eligible).",
                handle.pod_name,
                result.returncode,
            )
            return True
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            logger.warning(
                "GCP poll: eps/workload_started returned unparseable JSON for %s; "
                "conservatively assuming workload-started (failover-eligible).",
                handle.pod_name,
            )
            return True
        for item in payload if isinstance(payload, list) else []:
            if item.get("key") == "workload_started":
                return str(item.get("value") or "").strip().lower() == "true"
        # rc=0 but the key is absent from the payload → not written → setup.
        return False

    def _guest_persist_breadcrumb(self, handle: RunHandle, zone: str) -> str:
        """Best-effort read of the ``eps/persist`` crash-persist breadcrumb (#1151).

        Diagnostic-only: ANY failure (nonzero rc, bad JSON, transport)
        returns ``""`` — it never raises and never gates classification
        (deliberately UNLIKE :meth:`_guest_phase`'s typed ``GcpProbeError``
        contract: the breadcrumb is a forensic annotation on an
        already-classified terminal tick, so "couldn't read" must never
        turn a valid classification into a stalled one).
        """
        try:
            argv = render_guest_attributes_argv(
                config=self._config, name=handle.pod_name, zone=zone, query_path="eps/persist"
            )
            result = self._run(argv)
            if result.returncode != 0 or not result.stdout.strip():
                return ""
            payload = json.loads(result.stdout)
            for item in payload if isinstance(payload, list) else []:
                if item.get("key") == "persist":
                    return str(item.get("value") or "").strip()
        except Exception:
            return ""
        return ""

    def _append_persist_breadcrumb(
        self, base: PollResult, handle: RunHandle, zone: str, *, instance_status: str
    ) -> PollResult:
        """Append the eps/persist breadcrumb line to a failed-classifying tick (#1151).

        The line rides ``log_tail_excerpt`` — which already reaches every
        terminal marker the orchestrator reads — so there is NO
        ``PollResult`` schema change and classification is never gated on
        the read (diagnostic channel only). The line self-discloses the
        instance status: on a RUNNING-window read a healthy persist may
        still be in flight, so a standing ``attempted`` is only meaningful
        once the instance is TERMINATED (decision table:
        ``.claude/rules/compute-backend-failover.md`` § Part A item 8).
        """
        crumb = self._guest_persist_breadcrumb(handle, zone) or "ABSENT"
        line = f"[crash-persist-breadcrumb] eps/persist={crumb} (instance {instance_status})"
        if instance_status == "RUNNING":
            line += " - persist may be in flight; standing-attempted reading valid once TERMINATED"
        excerpt = f"{base.log_tail_excerpt}\n{line}" if base.log_tail_excerpt else line
        return replace(base, log_tail_excerpt=excerpt)

    # Log-tail trailer delimiters for the combined drain+tail SSH command.
    # Namespaced (``EPS_``) so a stray ``LOGTAIL`` substring in workload
    # output can't truncate the sentinel section.
    _LOGTAIL_START = "EPS_LOGTAIL_START"
    _LOGTAIL_END = "EPS_LOGTAIL_END"

    def _drain_sentinels(
        self, handle: RunHandle, zone: str
    ) -> tuple[int, str | None, str, str, int | None, str]:
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

        Besides the canonical ``/workspace/logs`` glob, the drain also
        scans the workload root's out_root logs dir
        (``<workload_root>/eval_results/issue_<N>/logs/``) as a fallback
        (incident #610: ``/workspace/logs`` did not exist on the VM, the
        dispatcher wrote its results sentinel under its out_root logs
        dir, and the poll reported ``done`` with
        ``sentinels_processed=0``; the startup script now pre-creates
        ``/workspace/logs``, this glob is the read-side belt to that
        write-side brace).

        Returns ``(processed, gate, alarm, log_tail, log_mtime_ago,
        alarm_class)``. ``alarm`` is "" normally; on a transport failure OR a
        matched-but-unprocessable sentinel set it carries a loud one-line
        diagnosis the caller surfaces in ``log_tail_excerpt`` — never a
        silent ``sentinels_processed=0`` (fail-LOUD contract, #608).

        ``alarm_class`` (#669) makes the alarm CLASS explicit so the caller
        never has to re-derive it heuristically (``gate is None`` sniffing):

        * ``"transport"`` — the drain SSH itself returned non-zero (transport
          down / permission / timeout). This is the unreachable-VM signature
          that ``GcpBackend.poll`` maps to ``PollResult.reachability_alarm``
          (the poller's frozen-phase wedge gate, M1a).
        * ``"sentinel_processing"`` — the SSH SUCCEEDED (a HEALTHY VM
          answered) but a matched sentinel set produced 0 processed markers
          (empty / unparseable body or a transient marker-post failure). NOT a
          reachability problem — the wedge gate must NEVER fire on this class.
        * ``""`` — no alarm (clean drain), OR the pre-SSH config skip
          (``issue<=0`` — nothing was probed, so it carries no reachability
          signal).

        ``log_mtime_ago`` (#607) is the workload log's mtime age in
        seconds, piggy-backed on the SAME ssh round trip (``stat -c %Y``
        + ``date +%s`` echoes appended AFTER the ``EPS_LOGTAIL_END``
        delimiter, so the tail partition is unaffected). ``None`` when
        the keys are absent (old fixtures / transport failure) or the
        stat reported ``-1`` (missing file — e.g. a pre-#607 handle whose
        log never existed); the caller then keeps the legacy placeholder.
        """
        issue = int(handle.extra.get("issue") or 0)
        if issue <= 0:
            # Pre-SSH config skip: nothing was probed, so this carries NO
            # reachability signal (alarm_class="") — it must never be read as
            # a transport-down wedge signature (#669).
            alarm = "gcp sentinel drain SKIPPED: handle missing 'issue' extra"
            logger.warning("GCP poll: %s. handle=%r", alarm, handle)
            return 0, None, alarm, "", None, ""

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
        # ``| cut -c1-4000`` bounds each tailed line ON THE VM before it
        # transits gcloud ssh — ``tail -n 30`` is line-counted, not
        # byte-bounded, so a pathological newline-free chunk in the
        # (post-#607, now-real) log would otherwise ship in full every
        # poll tick before the Python-side ``[:2000]`` cap.
        tail_stanza = (
            f'echo "{self._LOGTAIL_START}"; '
            + (
                f"tail -n 30 {shlex.quote(log_path)} 2>/dev/null | cut -c1-4000 || true; "
                if log_path
                else ""
            )
            + f'echo "{self._LOGTAIL_END}"'
            + (
                f'; echo "EPS_LOG_MTIME=$(stat -c %Y {shlex.quote(log_path)}'
                ' 2>/dev/null || echo -1)"'
                '; echo "EPS_LOG_NOW=$(date +%s)"'
                if log_path
                else ""
            )
        )
        # Fallback glob (#610): also drain sentinels a dispatcher wrote
        # under its out_root logs dir when /workspace/logs was missing.
        # workload_dir_for is config-derived (no spaces/metacharacters),
        # matching sentinel_drain_shell's trusted-glob contract.
        workload_root = workload_dir_for(self._config, issue)
        fallback_glob = f"{workload_root}/eval_results/issue_{issue}/logs/issue-{issue}-*.json"
        script = sentinel_drain_shell(issue, extra_globs=(fallback_glob,)) + "; " + tail_stanza
        argv = _base_gcloud_argv(
            self._config,
            "compute",
            "ssh",
            handle.pod_name,
            f"--command=sudo -n bash -c {shlex.quote(script)}",
        )
        argv += [f"--zone={zone}"]
        try:
            res = self._run(argv)
        except subprocess.TimeoutExpired as exc:
            # TRANSPORT class (#1084): the drain-list SSH HUNG past the
            # runner's per-call cap (default 300s, ``default_gcloud_runner``)
            # instead of returning non-zero — the #952 gcloud-ssh
            # hostkey-drift wedge. Same alarm tuple shape + class as the
            # rc!=0 branch below, so the poller's wedge-gate semantics
            # (#669) are inherited unchanged. ``_mark_sentinel_processed``'s
            # TimeoutExpired needs no catch here — it propagates into
            # ``drain_sentinels_via``'s mark_processed rename-failure catch.
            alarm = (
                f"gcp sentinel drain SSH timed out after {exc.timeout}s "
                "(hung transport — #952 class)"
            )
            logger.error("GCP poll: %s", alarm)
            return 0, None, alarm, "", None, "transport"
        if res.returncode != 0:
            # TRANSPORT class (#669): the drain SSH itself returned non-zero —
            # transport down / permission / timeout. This is the unreachable-VM
            # signature the poller's frozen-phase wedge gate reads (alarm_class
            # "transport" -> PollResult.reachability_alarm=True).
            alarm = (
                f"gcp sentinel drain FAILED (rc={res.returncode}): "
                f"{(res.stderr or '').strip()[:300]}"
            )
            logger.error("GCP poll: %s", alarm)
            return 0, None, alarm, "", None, "transport"

        stdout = res.stdout or ""
        drain_part, _, tail_part = stdout.partition(self._LOGTAIL_START)
        log_tail = tail_part.split(self._LOGTAIL_END)[0].strip()[:2000] if tail_part else ""
        # Workload-log mtime age (#607) — same key convention as
        # ``_probe_relaunched_workload``; ``-1`` = missing file.
        log_mtime_ago: int | None = None
        mtime_m = re.search(r"EPS_LOG_MTIME=(-?\d+)", stdout)
        now_m = re.search(r"EPS_LOG_NOW=(\d+)", stdout)
        if mtime_m and now_m and int(mtime_m.group(1)) >= 0:
            log_mtime_ago = max(0, int(now_m.group(1)) - int(mtime_m.group(1)))
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
            # SENTINEL-PROCESSING class (#669): the SSH SUCCEEDED (a HEALTHY
            # VM answered) but a matched sentinel set produced 0 processed
            # markers. NOT a reachability problem — alarm_class
            # "sentinel_processing" leaves PollResult.reachability_alarm False
            # so the poller's wedge gate never fires on a merely noisy run.
            alarm = (
                f"gcp sentinel drain: {len(sentinels)} sentinel(s) matched but 0 "
                "processed (empty/unparseable body or marker-post failure) — "
                "inspect /workspace/logs on the VM + poller stderr"
            )
            logger.error("GCP poll: %s", alarm)
            return 0, gate, alarm, log_tail, log_mtime_ago, "sentinel_processing"
        return processed, gate, "", log_tail, log_mtime_ago, ""

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

    def _gcp_gpu_util_probe(self, handle: RunHandle, zone: str) -> str:
        """Best-effort per-GPU utilization (comma-joined) via gcloud compute ssh nvidia-smi.

        Returns e.g. ``"0,0,0,0"`` or the literal ``"unknown"`` on ANY failure
        (SSH down, permission, timeout, empty/garbled output, non-numeric
        token). FAIL-SOFT by construction: the consumer
        (``poll_pipeline._gpu_idle``) reads ``"unknown"`` as NOT idle, so a
        probe miss never accumulates toward a GPU-idle advisory and never
        crashes the poll. Mirrors the RunPod lane's ``nvidia-smi`` gpu_util
        probe (``poll_pipeline.poll_once``) and reuses the existing GCP drain
        SSH pattern (``_drain_sentinels``): ``sudo -n`` because the GCE startup
        script runs as root, and the bare ``self._run(argv)`` inherits the
        runner's default 300s timeout (``default_gcloud_runner``), so a hung VM
        (the #667 class) raises ``subprocess.TimeoutExpired`` -> caught here ->
        ``"unknown"`` rather than blocking the poll tick.
        """
        argv = _base_gcloud_argv(
            self._config,
            "compute",
            "ssh",
            handle.pod_name,
            "--command=sudo -n nvidia-smi --query-gpu=utilization.gpu "
            "--format=csv,noheader,nounits",
        )
        argv += [f"--zone={zone}"]
        try:
            res = self._run(argv)
        except Exception as exc:  # transport / subprocess / TimeoutExpired
            logger.warning(
                "GCP gpu-util probe failed for %s (%s); reporting 'unknown'",
                handle.pod_name,
                exc,
            )
            return "unknown"
        if res.returncode != 0:
            logger.warning(
                "GCP gpu-util probe rc=%d for %s; reporting 'unknown'",
                res.returncode,
                handle.pod_name,
            )
            return "unknown"
        toks = [t.strip() for t in (res.stdout or "").replace("\n", ",").split(",") if t.strip()]
        # Validate every token parses as an int (matches _gpu_idle's contract);
        # any non-numeric token -> "unknown" so a partial/garbled read never
        # masquerades as idle.
        if not toks or any(not t.lstrip("-").isdigit() for t in toks):
            return "unknown"
        return ",".join(toks)

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
        # is convenience. Each subdir is its own ssh call so one
        # failure doesn't bury the other.
        #
        # The workload tree is root-owned (#588: the GCE startup script
        # runs the workload as root), so a plain `gcloud compute scp
        # --recurse` from the OS-Login user Permission-denies and the
        # mirror silently stays empty. Pull each dir as a base64-encoded
        # tar stream via `sudo -n` instead — the SAME grant the mandatory
        # sentinel pull above uses (`sudo -n cat`), just `tar -c | base64`
        # inside a `bash -o pipefail -c` wrapper.
        #
        # The `bash -o pipefail` wrap is LOAD-BEARING: bash pipelines
        # return the LAST command's exit status unless pipefail is set
        # (OFF by default), so on a MISSING remote dir (the common path —
        # item-4 dropped figures/ from the gate precisely because that dir
        # is normally absent) `tar` exits non-zero but `base64 -w0` reads
        # empty stdin and exits 0, masking the tar failure. The pipefail
        # wrap makes the tar rc propagate so the `if returncode != 0:
        # continue` guard fires (otherwise `base64.b64decode("")` +
        # `tarfile.open` on empty bytes raises locally). Same wrapping-bash
        # idiom as `_drain_sentinels` (`--command=sudo -n bash -c ...`);
        # we add `-o pipefail` because this wraps a PIPELINE, not a
        # `;`-sequence.
        repo_root = _default_src_root_for_fetch()
        workload_root = workload_dir_for(config, issue)
        for subdir in (f"eval_results/issue_{issue}", f"figures/issue_{issue}"):
            remote_parent = shlex.quote(f"{workload_root}/{os.path.dirname(subdir)}")
            remote_leaf = shlex.quote(os.path.basename(subdir))
            local_parent = repo_root / os.path.dirname(subdir)
            local_parent.mkdir(parents=True, exist_ok=True)
            remote_cmd = f"tar -c -C {remote_parent} {remote_leaf} | base64 -w0"
            tar_argv = _base_gcloud_argv(
                config,
                "compute",
                "ssh",
                handle.pod_name,
                f"--command=sudo -n bash -o pipefail -c {shlex.quote(remote_cmd)}",
            )
            tar_argv += [f"--zone={zone}"]
            tar_res = self._run(tar_argv)
            if tar_res.returncode != 0:
                logger.warning(
                    "GcpBackend.fetch_results: best-effort sudo tar of %s/%s failed "
                    "(rc=%d); authoritative copy is on HF/WandB/git. stderr=%s",
                    workload_root,
                    subdir,
                    tar_res.returncode,
                    tar_res.stderr[:300],
                )
                continue
            # Decode + extract the captured base64 tar stream under
            # local_parent. Wrap the decode+extract in try/log/continue too:
            # a genuinely truncated/corrupt stream from a transport hiccup
            # mid-transfer must log-and-continue, NOT raise — the mirror is
            # best-effort and the dir is authoritative on HF/WandB/git.
            try:
                raw = base64.b64decode(tar_res.stdout)
                with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                    # filter="data" (PEP 706) blocks any path-traversal /
                    # unsafe member in the stream and silences the 3.14
                    # extractall-without-filter DeprecationWarning; it
                    # extracts <leaf>/ under local_parent.
                    tf.extractall(path=local_parent, filter="data")
            except (ValueError, tarfile.TarError, EOFError) as exc:
                logger.warning(
                    "GcpBackend.fetch_results: best-effort decode/extract of %s/%s "
                    "failed (%s: %s); authoritative copy is on HF/WandB/git.",
                    workload_root,
                    subdir,
                    type(exc).__name__,
                    str(exc)[:300],
                )
                continue

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

        Confirm-deleted guard (#683 defense-in-depth): a CLEAN
        ``eps/phase=done`` run leaves the VM RUNNING only within the bounded
        #935 done-grace window (the success path keeps it alive so the
        sentinel can be scp'd; a CRASH triggers the in-VM EXIT-trap
        ``shutdown``+DELETE / rc!=0 belt), so the orchestrator-driven
        ``teardown`` is the FAST reaper for a successful run that REACHES
        finalize — when finalize never runs (dead orchestrator), the in-VM
        done-grace self-poweroff is the bound. If teardown's own rc==0 delete silently
        no-ops (rc==0 but the instance lingers RUNNING), nothing else
        reclaims it until the per-instance ``--max-run-duration`` belt (7d
        by default, #741) OR the daily ``gcp_audit`` janitor. So after an
        rc==0 delete we ACTIVELY confirm
        the instance is gone via ``describe``; if it is still present AND
        ``RUNNING``, re-issue the delete ONCE. Idempotent + fails toward
        "gone" (404 / non-RUNNING / probe-failure never re-delete). NOTE:
        the original #683 leak occurred on a run where teardown was NOT
        reached before the leak was observed — that path is closed by the
        watcher-side complement (see plan §10), not by this guard.
        """
        config = self._config
        zone = handle.extra.get("zone") or config.primary_zone
        argv = render_delete_argv(config=config, name=handle.pod_name, zone=zone)
        result = self._run(argv)
        if result.returncode == 0:
            self._confirm_deleted(handle, zone)
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

    def _confirm_deleted(self, handle: RunHandle, zone: str) -> None:
        """Verify the post-delete instance is gone; re-issue the delete on a RUNNING zombie (#683).

        Called only after an rc==0 ``gcloud compute instances delete``.
        ``gcloud delete`` is normally synchronous, but a clean-terminal VM
        has been observed lingering RUNNING after a rc==0 delete (#683), so
        this is the orchestrator-side belt that catches that silent no-op
        BEFORE the once-daily janitor (the only other reaper of a clean
        run, up to ~24h of GPU billing later).

        Fail-toward-gone discipline (never a spurious re-delete):

        * describe 404 / "not found" → confirmed gone, the desired state;
        * describe rc != 0 (any other reason — auth blip / transient API) →
          do NOT re-delete blindly (the delete already returned rc==0; a
          probe failure is not evidence the VM survived); log + return;
        * describe rc==0 but the payload has no ``status`` / is unparseable
          / ``status != RUNNING`` → trust the delete, return;
        * describe rc==0 AND ``status == RUNNING`` → the #683 zombie
          signature → re-issue the delete ONCE (best-effort; a failure
          there is logged, not raised — the janitor remains the backstop).
        """
        config = self._config
        describe_argv = render_describe_argv(config=config, name=handle.pod_name, zone=zone)
        probe = self._run(describe_argv)
        if probe.returncode != 0:
            stderr_low = (probe.stderr or "").lower()
            if "was not found" in stderr_low or "404" in stderr_low:
                return  # confirmed gone — the desired post-delete state
            # Couldn't confirm either way; the delete returned rc==0, so do
            # not re-delete on an unrelated probe failure. The janitor backs us up.
            logger.warning(
                "GCP teardown: post-delete describe of %s failed (rc=%d); "
                "trusting the rc==0 delete (gcp_audit janitor is the backstop). stderr=%s",
                handle.pod_name,
                probe.returncode,
                (probe.stderr or "")[:300],
            )
            return
        try:
            payload = json.loads(probe.stdout) if probe.stdout.strip() else {}
        except json.JSONDecodeError:
            logger.warning(
                "GCP teardown: post-delete describe of %s returned unparseable JSON; "
                "trusting the rc==0 delete.",
                handle.pod_name,
            )
            return
        status = (payload.get("status") or "").upper()
        if status != "RUNNING":
            # Empty payload, STOPPING/TERMINATED (delete in progress), or any
            # non-RUNNING state — the delete is taking effect; nothing to re-do.
            return
        # The #683 zombie: rc==0 delete but the VM is still RUNNING. Re-issue
        # the delete ONCE so a successful run cannot bill an A100 until the
        # daily janitor reaps it. Best-effort — a failure here is logged, not
        # raised (the janitor + the per-instance --max-run-duration fence,
        # 7d by default (#741), remain the backstops).
        logger.warning(
            "GCP teardown: %s still RUNNING after an rc==0 delete (#683 zombie); "
            "re-issuing the delete once.",
            handle.pod_name,
        )
        redelete = self._run(render_delete_argv(config=config, name=handle.pod_name, zone=zone))
        if redelete.returncode != 0:
            redelete_low = (redelete.stderr or "").lower()
            if "was not found" in redelete_low or "404" in redelete_low:
                return  # the first delete landed between the probe and the retry
            logger.error(
                "GCP teardown: re-issued delete of the RUNNING zombie %s ALSO failed "
                "(rc=%d); gcp_audit janitor will reap it. stderr=%s",
                handle.pod_name,
                redelete.returncode,
                (redelete.stderr or "")[:300],
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

        ``git_repo_root`` (#685) and ``skip_default_git_paths`` (#661) are
        read off ``spec.extra`` — the SAME thread the launch CLI uses for
        ``wandb_run_path`` (``_launch_extra_from_args`` populates all
        three) — so BOTH the fresh-launch (3182) and reconnect (2916)
        call sites pick them up uniformly. ``None`` / ``False`` (absent)
        = the established behavior.
        """

        decl = expected_artifacts_declaration(
            spec=spec,
            config=config,
            attempt_id=attempt_id,
            wandb_run_path=wandb_run_path,
            git_repo_root=spec.extra.get("git_repo_root"),
            skip_default_git_paths=bool(spec.extra.get("skip_default_git_paths", False)),
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
    log_mtime_ago: int | None = None,
) -> PollResult:
    """Thread sentinel-drain results into a coarse :class:`PollResult`.

    Gate precedence mirrors ``poll_pipeline.poll_once``: a drained gate
    sentinel wins over every other status — the orchestrator must park at
    the user gate before advancing. ``log_tail_excerpt`` carries (in
    priority order) the drain ALARM (a transport / permission failure must
    surface loudly, never as a silent ``sentinels_processed=0`` — incident
    #608), else whatever the base carried, else the sudo-read workload log
    tail.

    ``log_mtime_ago`` (#607): when the drain read a real workload-log
    mtime AND the base classification is ``running``, overlay the
    truthful ``last_log_mtime_sec_ago`` (the coarse poll hardwires
    ``10**9``) so phase-stuck zombies become detectable. Terminal results
    (``done`` / ``dead`` / ``stalled``) keep their own values.
    """

    merged_gate = base.gate or gate
    merged = replace(
        base,
        status="gate" if merged_gate else base.status,
        gate=merged_gate,
        sentinels_processed=processed,
        log_tail_excerpt=alarm or base.log_tail_excerpt or log_tail,
    )
    if log_mtime_ago is not None and base.status == "running":
        merged = replace(merged, last_log_mtime_sec_ago=log_mtime_ago)
    return merged


def _apply_lane_quiet_interval(
    result: PollResult, *, run_age_sec: float | None, lane_anomaly: bool
) -> PollResult:
    """Thread the lane-shared §7 quiet interval into a running tick.

    Wraps :func:`~explore_persona_space.backends.base.recommend_lane_next_interval`
    over the MERGED (post-drain) result, so a drained gate (status flipped
    to ``gate``) or sentinel activity keeps the short interval through the
    helper's own conditions. Callers pass ``run_age_sec`` from the describe
    payload's ``creationTimestamp`` and ``lane_anomaly`` from the drain
    alarm.
    """

    return replace(
        result,
        next_interval=recommend_lane_next_interval(
            status=result.status,
            gate=result.gate,
            sentinels_processed=result.sentinels_processed,
            new_milestone=result.new_milestone,
            run_age_sec=run_age_sec,
            lane_anomaly=lane_anomaly,
        ),
    )


def _gcp_status_to_poll_result(status: str) -> PollResult:
    """Map a GCE ``status`` to our coarse :class:`PollResult` shape.

    See https://cloud.google.com/compute/docs/instances/instance-life-cycle
    for the GCE status enum. We map:

    * ``RUNNING`` → ``running`` (pid_alive=True)
    * ``PENDING`` → ``running`` (FLEX_START-queued for capacity; the
      orchestrator's bg loop keeps polling — mirrors ``reconnect_or_none``,
      which treats PENDING as live since it is not in
      ``_NONLIVE_INSTANCE_STATUSES``; #782/#778)
    * ``PROVISIONING`` / ``STAGING`` → ``running`` (VM is coming up; the
      orchestrator's bg loop will keep polling)
    * ``STOPPING`` / ``REPAIRING`` → ``stalled`` (transient; bg loop retries)
    * ``TERMINATED`` / ``STOPPED`` / ``SUSPENDED`` → ``dead``
    """
    up = status.upper()
    if up == "RUNNING":
        return _coarse_poll(status="running", current_phase="running")
    if up == "PENDING":
        # FLEX_START capacity-queue state — legitimately live, keep polling
        # (parity with ``reconnect_or_none`` / ``_NONLIVE_INSTANCE_STATUSES``;
        # #782). Distinct branch from PROVISIONING/STAGING (VM booting) so the
        # queued-vs-booting distinction is explicit at the call site.
        return _coarse_poll(status="running", current_phase="pending")
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
    "DELIVERABLES_OK_FILENAME",
    "EXPLICIT_WIDE_DEGRADE_INTENTS",
    "INTENT_TO_MACHINE",
    "MACHINE_TYPE_ZONE_AVAILABILITY",
    "STARTUP_PASSTHROUGH_ENV_KEYS",
    "STARTUP_SECRET_ENV_KEYS",
    "WIDE_A100_80_BY_WIDTH",
    "WIDTH_ELIGIBLE_INTENTS",
    "GcloudRunResult",
    "GcloudRunner",
    "GcpBackend",
    "GcpBackendError",
    "GcpConfig",
    "GcpProvisioningError",
    "GcpWorkloadError",
    "MachineSpec",
    "QuotaHeadroom",
    "StaleNamedInstance",
    "attempt_id_for",
    "audit_stale_gcp_vms",
    "classify_create_failure",
    "default_gcloud_runner",
    "default_gcp_config",
    "deliverables_ok_path_for",
    "expected_artifacts_declaration",
    "instance_name_for",
    "lane_suffix_for",
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
    "wide_a100_80_for_width",
    "workload_dir_for",
    "zones_for_machine_type",
]
