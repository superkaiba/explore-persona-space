"""Central multi-backend compute router (slice 5 of the compute-router plan).

This module is the canonical replacement for
:func:`backends.selector.select_backend`'s submit-and-park flow. Where the
selector dispatches on a single ``backend:`` frontmatter and falls back to
RunPod-on-error, ``route(spec)`` orchestrates the full multi-backend ladder:

1. **Explicit override** — ``spec.backend == "runpod" | "gcp" | "nibi" |
   "fir" | "mila"`` runs that lane directly. RunPod is ALSO reachable on
   the auto chain — as the COST-ORDERED TERMINAL FALLBACK, never first:
   when every cheaper GCP rung + free SLURM lane is exhausted on the
   CAPACITY path (item 8, ``reason: auto_fallback_runpod``, #656), when a
   GCP rung CRASHES THE WORKLOAD (item 5, ``reason:
   gcp_workload_failover_runpod``, #658), and for the mapped cheap CPU
   intents (item 8a, ``cpu-small`` / ``cpu-mid``, #747). Full failover
   policy: ``.claude/rules/compute-backend-failover.md`` (canonical).
2. **Auto** — walk the resolved auto lane order. The STANDING DEFAULT is
   **GCP first** (:data:`DEFAULT_AUTO_LANE_ORDER` =
   ``("gcp", "nibi", "fir", "mila")``): credits-backed GCP capacity is
   consumed BEFORE the free SLURM lanes. The order is overridable via the
   ``EPM_AUTO_LANE_ORDER`` env var (comma-separated lanes, validated —
   ``runpod`` and unknown names raise loudly) or per-call via
   :attr:`RouterConfig.lane_order`; there is deliberately NO date logic —
   flipping the order back is a human action (env override or a default
   edit), never a clock. Contiguous SLURM lanes in the order are ranked
   among themselves by tz-corrected ``estimate_start_seconds`` (a ranking
   HINT, never a gate), the best is submitted and parked up to
   ``FREE_WAIT`` (default 600 s) to reach RUNNING; PENDING-at-cap triggers
   cancel + the next lane. GCP has no synchronous ``route()``-time park —
   its "park" is the provision call itself — but a FLEX_START instance that
   stays PENDING (queued for capacity) is bounded by the ASYNC poller's
   queue-wait timeout (``EPS_GCP_QUEUE_WAIT_SECONDS``, task #783), which
   cancels the queued instance and fails it over to the RunPod terminal rung
   (``reason: gcp_queue_timeout_failover_runpod``) rather than waiting forever.
   And a GCP PRE-WORKLOAD BOOT LOOP — N (default 2,
   ``EPS_GCP_BOOT_DEATH_STREAK_N``) consecutive same-rung setup deaths,
   counted per (issue, rung) in the durable lease — is the FOURTH
   GCP→RunPod trigger (#1029, ``reason: gcp_boot_loop_failover_runpod``);
   the ladder walk additionally SKIPS a rung whose same-UTC-day streak is
   >= N on the auto chain (outcome ``boot_loop_rung_skipped``, no daily
   attempt burned; explicit ``backend: gcp`` pins are exempt).
3. **Cancel state machine** — request a cancel via the backend's
   ``teardown(handle)``, then poll via the injected ``is_live_after_cancel``
   callable until the job is no longer live in the cluster queue
   (DRAC robot allowlist has no ``sacct``; we cannot confirm terminal
   CANCELLED). A job that RACED to RUNNING during cancel is KEPT (it has
   started; tearing it down would burn the wait we already paid for). A
   timeout produces a ``manual-attention`` outcome rather than a silent
   leak.
4. **Fallback chain — within the resolved order, RunPod is the TERMINAL
   rung (never first, never skipping a cheaper rung).** A provision-class
   failure on any lane (free-lane PENDING-at-cap / provisioning failure;
   GCP provisioning / capacity / prepare / state-probe failure when lanes
   remain after it) continues DOWN the resolved order. Under the GCP-first
   default that means GCP capacity failures fall through to the SLURM
   lanes; under a free-first override the SLURM park-failures escalate to
   GCP exactly as before. Once the GCP ladder AND the free SLURM lanes are
   ALL exhausted on the capacity path, the auto chain falls through to the
   RunPod terminal rung (item 8, ``reason: auto_fallback_runpod``, #656);
   a GCP WORKLOAD crash short-circuits straight there (item 5, #658). The
   reversal of the historical "NEVER RunPod on auto" invariant — RunPod is
   the LAST capacity rung, not override-only.
5. **Failure classification** — :class:`gcp.GcpProvisioningError` (and
   any backend-marked ``provisioning_failure: True`` raise) routes to the
   next tier; a :class:`gcp.GcpWorkloadError` on a GCP rung FAILS OVER TO
   RUNPOD (task #658 — a GCP failure of ANY class routes the next attempt
   to RunPod; RunPod pods persist + are SSH-able for diagnosis where GCP
   DELETEs its boot disk on crash; ``reason:
   gcp_workload_failover_runpod``), STRAIGHT to the RunPod terminal rung
   without cascading across the remaining GCP rungs or the SLURM lanes
   (re-crashing broken code there burns queue time). The bound: RunPod
   runs the broken job at most ONCE more, then the poller surfaces
   ``failure_class: code`` → ``status:blocked`` (the watcher's
   capacity-retry pass never re-drives a code failure), so there is no
   infinite RunPod cascade. A WORKLOAD failure on a SLURM lane still
   surfaces :class:`WorkloadSurfacedError` (not GCP, no failover).
   "every free lane park-failed AND GCP capacity-failed" raises
   :class:`NoComputeAvailableError` for the orchestrator to translate
   into ``epm:failure (failure_class: infra) + status:blocked``. A
   ``backend.prepare()`` failure (rsync/secrets push) is provision-class
   too — :class:`BackendPrepareError`: next tier on auto, typed terminal
   on an explicit override. A ``terminal_before_running`` park outcome
   is probed via the injected ``started_evidence_probe`` (scratch-dir
   ``status.json`` / ``job.out`` read): runtime artifacts present means
   the job STARTED and fast-failed — a WORKLOAD failure
   (:class:`WorkloadSurfacedError`, NO GCP escalation), not no-compute.
   Every fresh launch goes through :func:`_prepare_and_launch`
   (``prepare`` → ``launch``); reconnect paths never re-``prepare`` (the
   SLURM prepare rsyncs with ``--delete`` under a possibly-RUNNING job).
6. **Durable lease + reconnect** — a flock'd JSON lease at
   ``~/.eps-routing/issue-<N>.json`` (outside the worktree — the 09:47
   cron reaps worktrees, so a lease there would silently disappear) is
   keyed by a canonicalized spec hash + attempt id. The flock is
   per-issue (``<lease_dir>/issue-<N>.lock``), NOT shared across the
   directory, so a 10-min park on issue 137 inside
   ``store.transaction(137)`` does NOT block a ``route()`` on issue
   200. Before any submit / provision, ``route()`` reconnects to an
   existing live job (SLURM ``squeue --name eps-issue-<N>``; GCE
   ``reconnect_or_none`` — which refuses a RUNNING instance whose
   ``eps/phase`` is already terminal, the #908 gate-park zombie the
   pre-launch stale reclaim then deletes) via the injected backend so
   a re-driving ``issue-tick`` cron does NOT double-submit. The external
   job/instance id is persisted IMMEDIATELY after submit so an
   orchestrator crash between submit and lease-write leaves an
   ``UNKNOWN_SUBMITTED`` recovery state.
7. **GCP attempt-count guard** — a per-issue/day attempt counter caps
   auto-chain GCP attempts at ``MAX_GCP_ATTEMPTS_PER_DAY`` (default 16 —
   #1121 raised 8 -> 16 to cover the up-to-14-rung width-aware short walk
   plus margin; the ladder is WIDTH-AWARE: a dispatch declaring a shardable
   axis via ``spec.gpus`` walks wide ``a2-ultragpu-{8,4,2}g`` rungs FIRST,
   width-major, degrading on capacity miss into the byte-identical base
   ladder — see :func:`_gcp_ladder_specs` / :func:`_requested_wide_widths`;
   and, as of #1379, an EXPLICIT ``sweep-8g-a100`` dispatch likewise
   degrades 8->4->2 via a wide-rung SUFFIX appended after its base rungs —
   opt out per dispatch with ``--width-required``; ``sweep-8g-h100`` never
   degrades — see :func:`_explicit_wide_degrade_widths`. The worst new walk
   is 9 creates < the 14-rung width-8 short walk the cap was sized for).
   It counts ACTUAL create attempts across the #656 fallback-ladder rungs
   (a headroom-skip does NOT consume one) and is RE-READ each rung. At the
   cap the ladder STOPS issuing GCP creates (zero credit spent) and the
   chain falls through — to the next lane and ultimately the RunPod
   terminal rung (#656). The cap NEVER raises ``GcpAttemptCapExceededError``
   from the ladder anymore (the class is kept only for
   ``classify_terminal_exception`` back-compat). This is NOT a dollar cap
   (``tests/test_no_dollar_budget_caps.py`` enforces "no SystemExit on
   budget"); it bounds the *number of provision attempts* so a broken
   classifier that loops can't burn the GFS credit unattended.
8. **RunPod terminal rung (#656)** — when the cost-ordered GCP ladder
   (on-demand A100-80 → A100-40 → spot) AND the free SLURM lanes are all
   exhausted, the auto chain (and an explicit ``backend: gcp`` pin) falls
   through to RunPod as the documented terminal fallback
   (``reason: auto_fallback_runpod``). The deliberate reversal of the
   historical no-auto-RunPod invariant: RunPod is reached ONLY here, after
   every cheaper rung, never skipping one. The SAME RunPod terminal rung is
   the failover target when a GCP rung CRASHES THE WORKLOAD (task #658,
   ``reason: gcp_workload_failover_runpod``) — that case short-circuits
   straight here, skipping the remaining GCP rungs + SLURM lanes. Only if
   the RunPod launch ITSELF fails does the chain raise
   :class:`NoComputeAvailableError`.
8a. **CPU intents — per-intent RunPod CPU fallover (#747).** The cheap CPU
   intents mapped in :data:`RUNPOD_CPU_INSTANCE_FOR_INTENT`
   (``cpu-small`` / ``cpu-mid``) fall over GCP cheap CPU (E2; spot on a
   short job, on-demand otherwise) → RunPod CPU (``deployCpuPod``) when
   GCP is exhausted (capacity) OR crashes its workload — the CPU analogue
   of item 8 / item 5, keyed on that POSITIVE map (the single source of
   truth, checked by both the synchronous terminal rung and the async
   poller). ``cpu-bigmem`` is ABSENT from the map, so it keeps the #677
   ``cpu_exhausted_no_runpod_lane`` typed terminal VERBATIM (the >50 GB
   analysis lane has no cheap RunPod equivalent) — it does NOT fall over
   to RunPod. RunPod CPU pods are on-demand only; a CPU no-capacity miss
   surfaces :class:`RunPodNoCapacityError` → terminal. As of #1010 the rung
   also gates a plan-STATED footprint (``spec.extra["boot_disk_gb"]`` /
   ``["min_ram_gb"]``) against :data:`RUNPOD_CPU_INSTANCE_CAPS`, raising the
   typed :class:`CpuFallbackInfeasibleError`
   (``reason: cpu_fallback_infeasible_for_plan``) BEFORE any RunPod API
   call when the mapped instance cannot hold it, and
   ``RunPodBackend.launch`` threads the disk requirement into the provision
   argv (``--container-disk-gb max(50, boot_disk_gb)``) for mapped CPU
   intents.
8b. **GCP-only GPU intent translation (#940).** The RunPod launch paths
   (terminal rung + explicit override) translate a GCP-only GPU intent to
   its nearest same-or-narrower RunPod intent via
   :data:`RUNPOD_INTENT_FOR_GCP_INTENT` (``capture-7b`` → ``eval``,
   ``lora`` / ``lora-7b-h100`` → ``lora-7b``); an unmapped GCP GPU intent
   (``eval-h100``, in :data:`RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS`)
   fails loud PRE-launch naming the missing map row, and a real translation
   rides the marker ``extra`` as ``runpod_intent_translation``.
9. **Markers** — extends the existing ``epm:backend-selected v1`` body
   (per-lane est-starts raw+clamped, chosen lane, fallback chain,
   canonical reason codes, ids). The orchestrator's marker poster is
   injected; tests pass a list-appender. NEVER hardcodes a
   ``task.py`` shell-out — slice 5 is router-only, slice 6 wires the
   real poster.

The wiring into ``/issue`` lives in slice 6. This module is fully
testable without RunPod / SLURM / GCP being live.

Authoritative companion docs:

* Plan: ``.claude/plans/2026-06-08_224537-multi-backend-compute-router.md``
* Markers: ``.claude/workflow.yaml § markers`` (``epm:backend-selected``,
  ``epm:cluster-launched``, ``epm:cluster-poll``, ``epm:cluster-terminal``)
* Halt criterion: ``CLAUDE.md § Halt-criterion contract`` — a no-compute
  outcome is the canonical "infrastructure exhaustion" block (#1).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.backends.base import (
    BackendKind,
    BackendProbeError,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
    validate_lane_suffix,
)
from explore_persona_space.backends.gcp import (
    EXPLICIT_WIDE_DEGRADE_INTENTS,
    INTENT_TO_MACHINE,
    WIDE_A100_80_BY_WIDTH,
    WIDTH_ELIGIBLE_INTENTS,
    GcpProvisioningError,
    GcpWorkloadError,
    MachineSpec,
    QuotaHeadroom,
    a100_40_fallback_for_intent,
    machine_for_intent,
    quota_metric_for,
    resolve_provisioning_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Always-on 10-minute park cap on every free-lane submit (per plan §
#: "Mid-review" — supersedes the 6 h ``DEFAULT_MAX_WAIT_SECONDS`` from
#: :mod:`selector` and the ``EPM_CLUSTER_MAX_WAIT_SECONDS`` env knob).
FREE_WAIT_SECONDS: int = 600

#: Default poll interval inside the park watchdog. The SLURM scheduler
#: state updates on multi-second cycles; faster polling burns ssh round
#: trips without speeding the result. Tests inject smaller values.
DEFAULT_POLL_INTERVAL: float = 5.0

#: Per-issue/day cap on auto-escalation to GCP. NOT a dollar cap (see
#: ``tests/test_no_dollar_budget_caps.py``); this counts ATTEMPTS so a
#: broken classifier cannot loop into credit burn. Tunable per call.
#: #680: bumped 5 -> 8 to cover the length-aware short-job ladder (up to 5
#: rungs: spot-80, spot-40, flex-80, ondemand-80, ondemand-40) plus a
#: same-day retry margin. #1121: bumped 8 -> 16 — the width-aware short-job
#: walk is up to 14 rungs (3 wide widths x {spot, flex, ondemand} + the
#: 5-rung base tail), and 16 = 14 + margin, the same sizing logic as #680's
#: 5+margin -> 8; the free quota-headroom + boot-loop skips absorb the
#: quota-doomed subset in practice. Still an attempt COUNT, never a dollar
#: cap.
MAX_GCP_ATTEMPTS_PER_DAY: int = 16

#: Cancel state-machine: how long to keep polling for the job to leave
#: the live queue after ``scancel``. SLURM robots have no ``sacct`` so
#: we cannot confirm terminal CANCELLED — only that the job is no
#: longer live. A long-running run that won't die after this cap drops
#: into ``manual-attention``.
CANCEL_LIVE_GRACE_SECONDS: int = 60

#: Lease store directory — OUTSIDE the worktree by deliberate design.
#: The 09:47 ``worktree_audit.py`` cron reaps idle worktrees under
#: ``.claude/worktrees/``; a lease there would silently disappear and
#: the next ``/issue`` invocation would double-submit. ``~/.eps-routing/``
#: lives in HOME and is owned by the orchestrator user.
LEASE_STORE_DIRNAME: str = ".eps-routing"

#: Canonical reason codes the router emits in the marker. The selector's
#: legacy codes (``frontmatter_default``, ``slurm_not_implemented``)
#: stay in :mod:`selector`; this set is router-specific.
ROUTE_REASON_OVERRIDE: str = "override"
ROUTE_REASON_RECONNECT: str = "reconnect"
ROUTE_REASON_AUTO_STARTED: str = "auto_started"
ROUTE_REASON_AUTO_FALLBACK_GCP: str = "auto_fallback_gcp"
ROUTE_REASON_NO_COMPUTE: str = "no_compute_available"
ROUTE_REASON_WORKLOAD_FAILURE: str = "workload_failure"
#: ``backend.prepare`` failed pre-launch (rsync / secrets push). Matches
#: the ``reason: backend_prepare_failed`` line the dispatch CLI's
#: ``classify_terminal_exception`` emits for :class:`BackendPrepareError`
#: — pre-fix the breadcrumb said ``no_compute_available`` while the
#: typed terminal said ``backend_prepare_failed`` (round-6 Mn1).
ROUTE_REASON_PREPARE_FAILED: str = "backend_prepare_failed"
#: The router fell back to RunPod as the TERMINAL rung after every cheaper
#: GCP rung (on-demand A100-80 → A100-40 → SPOT) AND the free SLURM lanes
#: were exhausted (#656). DISTINCT from :data:`ROUTE_REASON_OVERRIDE` so the
#: marker trail tells "user pinned RunPod" apart from "router fell back to
#: RunPod after exhausting cheaper compute". The ``extra`` carries
#: ``runpod_fallback_residual_gap`` naming which rungs ran dry. This is the
#: deliberate reversal of the historical no-auto-RunPod invariant
#: (user-directed 2026-06-17): RunPod is reached ONLY here, never first,
#: never skipping a cheaper rung.
ROUTE_REASON_RUNPOD_FALLBACK: str = "auto_fallback_runpod"
#: A CPU-only intent WITHOUT a RunPod-CPU lane (gpu_count==0 AND not in
#: :data:`RUNPOD_CPU_INSTANCE_FOR_INTENT` — i.e. ``cpu-bigmem``, #677) reached
#: the RunPod terminal rung — either the GCP lane was capacity-exhausted OR a
#: GCP CPU workload crashed (sync failover). ``cpu-bigmem`` has no cheap RunPod
#: equivalent (#747 added a RunPod CPU lane ONLY for the mapped cheap intents
#: cpu-small / cpu-mid), so surface a typed terminal instead of attempting an
#: unservable RunPod launch. DISTINCT from :data:`ROUTE_REASON_RUNPOD_FALLBACK` /
#: :data:`ROUTE_REASON_NO_COMPUTE` so the marker trail shows the
#: CPU-unservable cause, and DISTINCT from ``no_compute_available`` so the
#: watcher's capacity-retry pass (which keys on ``no_compute_available``) does
#: NOT auto-re-drive a structurally-unservable run. This is the token
#: ``classify_terminal_exception`` emits as the ``epm:failure`` note's
#: ``reason:`` field for a :class:`CpuExhaustedNoRunpodLaneError`.
ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD: str = "cpu_exhausted_no_runpod_lane"

#: CPU intents that HAVE a RunPod CPU fallback lane (#747) → their RunPod CPU
#: ``instanceId`` (``deployCpuPod``). A CPU intent in this map FALLS OVER
#: GCP→RunPod CPU when the GCP CPU lane is exhausted (capacity) OR crashes its
#: workload (the CPU analogue of the GPU GCP→RunPod failover, #656/#658),
#: keyed on this POSITIVE map — NOT on ``gpu_count == 0`` alone, which would
#: wrongly route ``cpu-bigmem`` to RunPod. A CPU intent NOT in this map
#: (``cpu-bigmem`` — the >50 GB large-footprint analysis lane, with no cheap
#: RunPod equivalent) keeps the #677 typed
#: :class:`CpuExhaustedNoRunpodLaneError` terminal verbatim. This is the SINGLE
#: SOURCE OF TRUTH for the CPU intent → RunPod instance_id mapping:
#: ``scripts/gpu_heuristics.resolve_cpu_intent`` and
#: ``scripts/backend_poll._is_gcp_async_workload_failure`` both import THIS
#: dict (no duplicated copy to drift). The intent → GCP machine-type mapping
#: lives separately in :data:`gcp.INTENT_TO_MACHINE`; this is intentionally the
#: RunPod-side companion (the two are distinct providers). RAM note: the GCP
#: ``cpu-mid`` (``e2-standard-8`` = 8 vCPU / 32 GB) and the RunPod ``cpu-mid``
#: (``cpu3c-8-16`` = 8 vCPU / 16 GB) differ in RAM by design — the RunPod lane
#: is a CAPACITY backstop, and a >16 GB CPU job should target ``cpu-bigmem``
#: anyway; the asymmetry is accepted, not a bug. As of #1010 a plan that
#: STATES its footprint (``spec.extra["boot_disk_gb"]`` / ``["min_ram_gb"]``)
#: is checked against the fixed capabilities in
#: :data:`RUNPOD_CPU_INSTANCE_CAPS` by the feasibility gate in
#: :func:`_runpod_terminal_rung` — an unsatisfiable footprint refuses the
#: fallback typed (:class:`CpuFallbackInfeasibleError`) instead of
#: provisioning an undersized pod (incident #958).
RUNPOD_CPU_INSTANCE_FOR_INTENT: dict[str, str] = {
    "cpu-small": "cpu3g-2-8",  # 2 vCPU / 8 GB, gen-3 general purpose
    "cpu-mid": "cpu3c-8-16",  # 8 vCPU / 16 GB, gen-3 compute-optimized
}


@dataclass(frozen=True)
class RunPodCpuInstanceCaps:
    """Fixed capabilities of one mapped RunPod CPU instance (#1010)."""

    vcpu: int
    ram_gb: int
    max_container_disk_gb: int


#: Fixed capabilities of the mapped RunPod CPU instances (#1010, incident
#: #958). RAM is fixed per instance_id (encoded in the id itself:
#: ``<flavor>-<vCPU>-<RAM_GB>``, not threadable). ``max_container_disk_gb``
#: bounds the EFFECTIVE ``deployCpuPod`` ``containerDiskInGb`` payload —
#: ``max(container_disk_gb, volume_gb)``, because
#: ``runpod_api._deploy_cpu_once`` folds the CPU volume request into the
#: container disk (``deployCpuPodInput`` has no volume field).
#: PROBE-VERIFIED 2026-07-04 (issue #1010, live ``deployCpuPod`` probes):
#:   * ``cpu3g-2-8`` -> 20; verified_by: accept-reject-only — API validation
#:     rejects effective 50 (today's untouched default payload) with
#:     "Container Disk must be less than or equal to 20" (flavor cpu3g).
#:     20 sits BELOW the 50 default, so this cap can only refuse — safe.
#:   * ``cpu3c-8-16`` -> 50; verified_by: accept-reject-only — the API
#:     ACCEPT bound is 80 (effective 80 accepted; effective 100 rejected
#:     with "Container Disk must be less than or equal to 80", flavor
#:     cpu3c), but no realized in-pod filesystem read was obtainable, so
#:     the recorded cap is the empirically-HONORED floor: pod-958's
#:     measured 50 GB overlay. An unverified value above the honored floor
#:     would be an unverified ALLOW cap re-creating the #958 shape inside
#:     the "accepted" band; a later df-verified probe may raise it to 80.
#: Keys MUST cover every value of :data:`RUNPOD_CPU_INSTANCE_FOR_INTENT`
#: (pinned by tests/test_router.py::
#: test_runpod_cpu_instance_caps_cover_every_mapped_instance); the
#: feasibility gate does a direct ``[]`` lookup so a missing row fails LOUD
#: (KeyError), never silently skips the check.
RUNPOD_CPU_INSTANCE_CAPS: dict[str, RunPodCpuInstanceCaps] = {
    "cpu3g-2-8": RunPodCpuInstanceCaps(vcpu=2, ram_gb=8, max_container_disk_gb=20),
    "cpu3c-8-16": RunPodCpuInstanceCaps(vcpu=8, ram_gb=16, max_container_disk_gb=50),
}

#: Reason token for a RunPod CPU fallback refused because the plan's STATED
#: footprint (``spec.extra["boot_disk_gb"]`` / ``spec.extra["min_ram_gb"]``)
#: exceeds the mapped instance's fixed capabilities (#1010, incident #958).
#: Distinct-reason-per-distinct-cause is the established router pattern
#: (:data:`ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD` #677,
#: :data:`ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD` #783): this intent
#: HAS a RunPod lane — it just cannot hold this plan — so reusing the
#: parent's "no runpod lane" token would mislead triage. Like the parent's
#: reason, NOT in the watcher's ``TRANSIENT_CAPACITY_REASONS``.
ROUTE_REASON_CPU_FALLBACK_INFEASIBLE: str = "cpu_fallback_infeasible_for_plan"

#: GCP GPU intent -> the RunPod intent the terminal rung provisions (#940).
#: TOTAL over the gpu_count>0 keys of gcp.INTENT_TO_MACHINE (identity rows
#: included) so the completeness test
#: (tests/test_router.py::test_translation_map_total_over_gcp_gpu_intents)
#: catches a future intent added without deciding its RunPod fate — the exact
#: #841 failure mode (a `capture-7b` RunPod terminal-rung launch died in
#: gpu_heuristics.resolve_intent's KeyError, voiding the sanctioned last rung
#: despite live RunPod capacity). Values MUST be same-or-narrower GPU width
#: and >= HBM (never widen; pinned by
#: test_translation_never_widens_gpu_count_and_targets_provisionable).
#: CPU intents are DELIBERATELY absent — RUNPOD_CPU_INSTANCE_FOR_INTENT
#: above owns them (#677/#747), byte-identical semantics.
RUNPOD_INTENT_FOR_GCP_INTENT: dict[str, str] = {
    # identity rows — intents in BOTH vocabularies (no-op, no marker record)
    "eval": "eval",  # GCP 1x L4      -> RunPod 1x H100
    "debug": "debug",  # GCP 1x L4      -> RunPod 1x H100
    "lora-7b": "lora-7b",  # GCP 1x A100-80 -> RunPod 1x H100
    "ft-7b": "ft-7b",  # GCP 4x A100-80 -> RunPod 4x H100
    "sweep-8g-a100": "sweep-8g-a100",  # 8x A100-80 -> 8x A100 (same width)
    "sweep-8g-h100": "sweep-8g-h100",  # 8x H100    -> 8x H100
    # GCP-only intents — nearest same-or-narrower RunPod intent
    "lora": "lora-7b",  # GCP alias of lora-7b (1x A100-80 -> 1x H100-80)
    "capture-7b": "eval",  # #752 activation-capture EVAL path: 1x A100-80
    # forward-pass -> 1x H100-80 (same width, HBM 80 >= 80)
    "lora-7b-h100": "lora-7b",  # 1x H100-80 -> 1x H100-80 (identical hardware)
}

#: GCP GPU intents DELIBERATELY not RunPod-servable via intent translation
#: (#940). eval-h100 (2x H100, TP=2): no same-width RunPod intent exists, and
#: narrowing 2->1 would silently break a 2-GPU-sharded --workload-cmd
#: mid-run on a paid pod — worse than failing loud at dispatch with a
#: message naming this row. Widening (e.g. to an 8x sweep intent) is
#: banned outright.
RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS: frozenset[str] = frozenset({"eval-h100"})


def _translated_runpod_intent(spec: RunSpec) -> tuple[str, dict[str, str] | None]:
    """Resolve spec.intent to a RunPod-provisionable intent (#940).

    Returns ``(runpod_intent, translation_record)`` where
    ``translation_record`` is ``{"from": ..., "to": ...}`` for a REAL
    translation, ``None`` for identity rows and pass-through intents.
    Raises :class:`ValueError` — naming the missing
    :data:`RUNPOD_INTENT_FOR_GCP_INTENT` row — for a GCP-mapped GPU intent
    with no row (the ``eval-h100`` / future-intent case). CPU intents
    (``gpu_count == 0`` in ``gcp.INTENT_TO_MACHINE``) and non-GCP intents
    (``ft-70b``, ``inf-70b``, custom) pass through verbatim: the #677/#747
    CPU semantics and the RunPod-native vocabulary are untouched.

    Uses a direct ``INTENT_TO_MACHINE.get(...)`` — NOT
    :func:`gcp.machine_for_intent` — so the helper (a) never raises gcp.py's
    unmapped-intent ``ValueError`` for RunPod-only intents like ``ft-70b``
    under explicit override, and (b) is immune to
    ``spec.extra["machine_spec_override"]`` rung threading.
    """
    intent = spec.intent
    target = RUNPOD_INTENT_FOR_GCP_INTENT.get(intent)
    if target is not None:
        if target == intent:
            return intent, None  # identity: byte-identical behavior
        return target, {"from": intent, "to": target}
    gcp_machine = INTENT_TO_MACHINE.get(intent)
    if gcp_machine is not None and gcp_machine.gpu_count > 0:
        raise ValueError(
            f"intent {intent!r} is GCP-mapped but has no RunPod translation: "
            f"add a same-or-narrower row to backends/router.py "
            f"RUNPOD_INTENT_FOR_GCP_INTENT (or list it in "
            f"RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS and keep it off RunPod). "
            f"Deliberate gaps: {sorted(RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS)}."
        )
    return intent, None  # CPU / RunPod-native / custom: verbatim


#: The router fell back to RunPod because a GCP attempt FAILED THE WORKLOAD
#: (a :class:`gcp.GcpWorkloadError`, not a capacity/headroom miss) — the
#: deliberate reversal of the historical "GCP workload failure surfaces
#: with NO fallback" invariant (user-directed 2026-06-24, task #658).
#: Rationale: when GCP is failing a run, re-running it on RunPod keeps the
#: science moving AND gives a persistent, SSH-able pod for diagnosis (GCP's
#: ``--instance-termination-action=DELETE`` destroys the boot disk on the
#: EXIT-trap teardown, so a GCP crash loses its own logs). DISTINCT from
#: :data:`ROUTE_REASON_RUNPOD_FALLBACK` (capacity exhaustion) so the marker
#: trail tells the two failover causes apart. The failover does NOT cascade
#: across the remaining GCP rungs or the free SLURM lanes (re-crashing
#: broken code there burns queue time, the original no-cascade concern) —
#: it routes the run STRAIGHT to the RunPod terminal rung. The bound on a
#: genuinely-broken job: RunPod runs it AT MOST ONCE more; if it crashes
#: again the poller surfaces ``failure_class: code`` → ``status:blocked``,
#: which the watcher's capacity-retry pass (``no_compute_available`` only)
#: never re-drives — so there is no infinite RunPod cascade.
ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD: str = "gcp_workload_failover_runpod"

#: The router failed over to RunPod because the ASYNC poller
#: (``scripts/backend_poll.py``) detected a GCP run that had ALREADY come up
#: and crashed its WORKLOAD minutes in (task #659). DISTINCT from
#: :data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD` (the SYNCHRONOUS
#: ``route()``-time ``GcpWorkloadError`` failover, #658) so the
#: ``epm:backend-selected`` marker trail tells the two DETECTION paths apart —
#: same RunPod target, same "GCP crashed the workload, run it on RunPod for a
#: persistent SSH-able pod" rationale, different point of detection. The async
#: path has no live ``route()`` call to raise ``_GcpWorkloadFailover`` from
#: (the VM is already launched), so the poller calls
#: :func:`failover_to_runpod_after_async_workload_crash` to reach the SAME
#: terminal rung. The "exactly once" bound is identical: the failover
#: re-points the handle sidecar to RunPod, so a RunPod re-crash polls a RunPod
#: handle (not GCP) and surfaces ``failure_class: code`` → ``status:blocked``,
#: which the watcher's capacity-retry pass never re-drives.
ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC: str = "gcp_workload_failover_runpod_async"

#: The router failed over to RunPod because the ASYNC poller
#: (``scripts/backend_poll.py``) detected a GCP instance that stayed in the
#: FLEX_START capacity QUEUE (``current_phase == "pending"``) past
#: ``EPS_GCP_QUEUE_WAIT_SECONDS`` without ever reaching RUNNING (task
#: #783/#778). DISTINCT from both WORKLOAD-crash reasons
#: (:data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD` /
#: ``_ASYNC``) — a queue that never advanced is NOT a crash — AND from
#: :data:`ROUTE_REASON_RUNPOD_FALLBACK` (capacity EXHAUSTION at create time;
#: this is a create that SUCCEEDED but the instance never dequeued). Same
#: RunPod target + terminal rung, distinct detection cause so the
#: ``epm:backend-selected`` marker trail tells a queue timeout apart from a
#: crash and a capacity miss. A queue-timeout cancel is a CLEAN advance: it
#: does NOT touch the per-day GCP attempt counter (that bumps only on a
#: create, inside ``_attempt_one_gcp_rung``, which the poller never re-enters).
ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD: str = "gcp_queue_timeout_failover_runpod"

#: The ASYNC poller detected a GCP PRE-WORKLOAD BOOT LOOP (#1029): N (default
#: :data:`GCP_BOOT_DEATH_STREAK_N_DEFAULT`) CONSECUTIVE pre-workload setup
#: deaths on the SAME ladder rung for the SAME issue, counted per-incarnation
#: in the durable lease's ``gcp_boot_death_streaks`` record, and failed the
#: run over to the RunPod terminal rung. DISTINCT from BOTH workload-crash
#: reasons (:data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD` / ``_ASYNC`` —
#: a boot death never reached the workload), from
#: :data:`ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD` (the instance there
#: never left the capacity queue; here it booted and DIED), and from
#: :data:`ROUTE_REASON_RUNPOD_FALLBACK` (capacity exhaustion at create time —
#: these creates all SUCCEEDED). Same RunPod target + terminal rung, distinct
#: detection cause so the ``epm:backend-selected`` marker trail tells a boot
#: loop apart from a crash, a queue timeout, and a capacity miss. The trigger
#: never touches the per-day GCP attempt counter (that bumps only on a create,
#: inside ``_attempt_one_gcp_rung``, which the poller never re-enters).
ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD: str = "gcp_boot_loop_failover_runpod"

#: The ASYNC poller detected a GCP FLEX_START instance that VANISHED while
#: PENDING (#1116/#1112): the create SUCCEEDED and the instance sat in the DWS
#: capacity queue (last observed ``current_phase == "pending"``, per the
#: sidecar phase clock), then disappeared from instances-list entirely (a dead
#: ``terminal_instance not found`` poll, NO delete operation) — the queue
#: dropped the request server-side, a pure CAPACITY event. DISTINCT from
#: :data:`ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD` (the instance there
#: dequeued nothing but still EXISTED server-side — the failover tears it
#: down; here the record is already DELETED, so there is nothing to tear
#: down), from :data:`ROUTE_REASON_GCP_BOOT_LOOP_FAILOVER_RUNPOD` (the
#: instance there BOOTED and died; here it never left the queue), from both
#: workload-crash reasons (nothing ever ran), and from
#: :data:`ROUTE_REASON_RUNPOD_FALLBACK` (capacity exhaustion at create time —
#: this create SUCCEEDED). Same RunPod target + terminal rung, distinct
#: detection cause so the ``epm:backend-selected`` marker trail tells a queue
#: vanish apart from every sibling. The trigger never touches the per-day GCP
#: attempt counter (that bumps only on a create, inside
#: ``_attempt_one_gcp_rung``, which the poller never re-enters).
ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD: str = "gcp_queue_vanish_failover_runpod"

#: Default N for the #1029 pre-workload boot-loop breaker: the Nth CONSECUTIVE
#: same-rung pre-workload boot death fails over to RunPod, and a rung whose
#: same-UTC-day streak is >= N is SKIPPED by the route()-side ladder walk
#: (outcome ``boot_loop_rung_skipped``, no daily attempt burned). N=2 gives a
#: lone transient setup death (a clone/uv-sync hiccup) exactly ONE free retry —
#: the documented manual protocol for the #640 guestTerminate class ("allow
#: exactly ONE blind GCP retry, then pivot to RunPod") and the #775 CUDA-IMA
#: second-same-signature precedent. Env-overridable at call time via
#: ``EPS_GCP_BOOT_DEATH_STREAK_N`` (see :func:`gcp_boot_death_streak_threshold`).
GCP_BOOT_DEATH_STREAK_N_DEFAULT: int = 2

#: The RunPod terminal rung PROVISIONED a pod but the #909 workload-start leg
#: FAILED (``RunPodWorkloadStartError`` carrying the partial handle, #954). A
#: pod EXISTS and BILLS (left RUNNING for diagnosis per the #909 contract), so
#: this is NOT :data:`ROUTE_REASON_NO_COMPUTE` — mislabeling it as
#: ``no_compute_available`` invites the watcher's capacity-retry pass to
#: re-drive the whole auto ladder (a fresh paid GCP attempt) while the pod
#: bills invisibly (the #931 incident). The string is deliberately IDENTICAL
#: to the ``dispatch_issue.py`` explicit-override arm's established
#: ``reason: runpod_workload_start_failed`` (#909) — one reason per failure
#: class across paths (cross-module parity pinned in ``tests/test_router.py``).
#: It is NOT in ``autonomous_session_watch.TRANSIENT_CAPACITY_REASONS``, so
#: the watcher never auto re-drives it.
ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED: str = "runpod_workload_start_failed"

#: Consecutive ``is_started`` probe failures tolerated inside the park
#: watchdog before it gives up with ``probe_failures_exceeded``.
#: Mirrors ``scripts/router_acceptance.py``'s
#: ``_POLL_MAX_CONSECUTIVE_FAILURES`` (same value, same reset-on-success
#: semantics). A probe failure means the job state is UNKNOWN — it must
#: NEVER read as "not started yet" indefinitely (round-6 B1): the
#: watchdog hands the lane to the cancel state machine, whose own
#: probe-failure handling (treat-as-still-live + grace) resolves to
#: ``manual_attention`` while the transport stays broken.
PARK_MAX_CONSECUTIVE_PROBE_FAILURES: int = 3

#: SLURM free-lane subset (DRAC + Mila), in legacy precedence order.
#: Kept as a public constant for callers that need "the free lanes";
#: the AUTO chain's order is :data:`DEFAULT_AUTO_LANE_ORDER` /
#: :func:`auto_lane_order`. RunPod is NEVER in either list — it's
#: override-only by deliberate design.
DEFAULT_FREE_LANE_ORDER: tuple[BackendKind, ...] = ("nibi", "fir", "mila")

#: Standing default auto lane order: **GCP first** (credits-backed GCP
#: capacity is consumed BEFORE the free SLURM lanes), then the SLURM
#: lanes in legacy precedence. This is an unconditional default — NO
#: date logic; flipping back to free-first later is a deliberate human
#: action (set :data:`ENV_AUTO_LANE_ORDER` or edit this default), never
#: a clock.
DEFAULT_AUTO_LANE_ORDER: tuple[BackendKind, ...] = ("gcp", *DEFAULT_FREE_LANE_ORDER)

#: Env override for the auto lane order — comma-separated lane names,
#: e.g. ``EPM_AUTO_LANE_ORDER=nibi,fir,mila,gcp`` to restore free-first.
#: Validated by :func:`auto_lane_order`: ``runpod`` raises loudly
#: (real-money safety — never silently dropped), as do unknown names,
#: ``auto``/``cluster`` literals, and duplicates.
ENV_AUTO_LANE_ORDER: str = "EPM_AUTO_LANE_ORDER"

#: DEPRECATED back-compat shim (#656). Was the env gate for the #537
#: STANDARD->SPOT auto-fallback. The #656 GCP fallback ladder SUBSUMES that
#: machinery — a SPOT rung now fires by DEFAULT for any "short" job
#: (:func:`_is_short_job`), so the env gate no longer gates anything. The
#: constant is kept defined (and exported) so a stale importer does not
#: break; setting it has NO effect. ``spec.extra["spot_tolerant"]`` survives
#: as a FORCE-spot override (declare a job preemption-recoverable past the
#: GPU-hour threshold). Remove this shim once no importer references it.
ENV_GCP_SPOT_FALLBACK: str = "EPS_GCP_SPOT_FALLBACK"

#: Default GPU-hour threshold below which a job is "short" enough to risk
#: SPOT preemption (#656). A short eval / LoRA absorbs a preemption restart
#: cheaply; a long training run would lose hours, so it skips the spot rungs
#: and falls to the next lane / RunPod instead. Env-overridable via
#: :data:`ENV_SPOT_MAX_GPU_HOURS` so an operator can tune it without a code
#: change. Source: task #656 originating prompt (threshold ~2 GPU-h).
DEFAULT_SPOT_MAX_GPU_HOURS: float = 2.0

#: Env override for :data:`DEFAULT_SPOT_MAX_GPU_HOURS` (a float; unparseable
#: → the default, logged loud). The short-job gate is threshold-sensitive:
#: a future debug-er tuning spot eligibility tunes THIS, and the resolved
#: gpu-h estimate + threshold ride the per-rung ``epm:backend-selected``
#: attempt detail for observability.
ENV_SPOT_MAX_GPU_HOURS: str = "EPS_GCP_SPOT_MAX_GPU_HOURS"

#: Every value the ROUTER accepts for ``spec.backend``. ``route()``
#: rejects anything outside this set at entry (closes the empty-string
#: / stringly-typed-miswire silent-auto-route hole). Narrower than
#: :data:`BackendKind` (``base.py``) by deliberate design: the legacy
#: ``"cluster"`` literal lives in the selector surface (``selector.py``)
#: and is NOT a routable backend at the slice-5 router level — a caller
#: that wants a free-cluster lane must name it (``"nibi"`` / ``"fir"``)
#: or leave ``backend`` unset to auto-route. Passing ``"cluster"`` here
#: is treated as a stringly-typed miswire.
_VALID_BACKEND_VALUES: frozenset[str] = frozenset({"runpod", "nibi", "fir", "gcp", "mila", "auto"})

#: Lanes the AUTO chain may contain — :data:`_VALID_BACKEND_VALUES`
#: minus ``runpod`` (override-only; real money) and ``auto`` (the
#: sentinel itself, not a lane).
_AUTO_LANE_VALUES: frozenset[str] = frozenset({"gcp", "nibi", "fir", "mila"})

#: Lanes whose kind IS a SLURM cluster name. The shared ``SlurmBackend``
#: resolves its target cluster from ``spec.cluster`` per call, so every
#: router site that touches one of these lanes MUST thread the lane kind
#: into ``spec.cluster`` via :func:`_spec_for_lane` first.
_PER_CLUSTER_LANES: frozenset[str] = frozenset({"nibi", "fir", "mila"})


# ---------------------------------------------------------------------------
# Public outcome / error types
# ---------------------------------------------------------------------------


class RouteError(RuntimeError):
    """Base class for router-terminal errors."""


class NoComputeAvailableError(RouteError):
    """Terminal: every free lane park-failed AND GCP capacity-failed.

    The orchestrator translates this into
    ``epm:failure (failure_class: infra) + status:blocked`` (the only
    autonomous-mode infra exhaustion block per CLAUDE.md § halt
    criterion #1's "fact only the user knows" — except this is "fact
    nobody knows: there is no compute right now").
    """

    def __init__(
        self,
        reason: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = list(attempts or [])


class CpuExhaustedNoRunpodLaneError(NoComputeAvailableError):
    """Terminal: a CPU-only intent WITHOUT a RunPod-CPU lane (gpu_count==0 AND
    not in :data:`RUNPOD_CPU_INSTANCE_FOR_INTENT` — i.e. ``cpu-bigmem``, #677)
    reached the RunPod terminal rung — the GCP lane was capacity-exhausted OR a
    GCP CPU workload crashed (sync failover) — and ``cpu-bigmem`` has no cheap
    RunPod equivalent. (The mapped cheap CPU intents cpu-small / cpu-mid DO fall
    over to RunPod CPU as of #747 and never raise this.)

    A :class:`NoComputeAvailableError` SUBCLASS so existing callers that catch
    the base class still catch it, but ``classify_terminal_exception``
    (``issue_dispatch.py``) can map it to a DISTINCT ``epm:failure`` note
    (``reason: cpu_exhausted_no_runpod_lane``) ahead of the generic
    ``no_compute_available`` note. The distinction matters because the
    watcher's capacity-retry pass (``autonomous_session_watch.py``'s
    ``TRANSIENT_CAPACITY_REASONS``) re-drives ONLY ``no_compute_available``; a
    structurally-unservable ``cpu-bigmem`` RunPod launch must NOT auto-retry (no
    lane will ever free up to make RunPod accept it). Inherits ``__init__``
    verbatim (reason message + attempts).
    """


class CpuFallbackInfeasibleError(CpuExhaustedNoRunpodLaneError):
    """Terminal: a mapped cheap CPU intent reached the RunPod terminal rung,
    but the plan's STATED footprint (``spec.extra["boot_disk_gb"]`` /
    ``spec.extra["min_ram_gb"]``) exceeds the mapped RunPod CPU instance's
    fixed capabilities in :data:`RUNPOD_CPU_INSTANCE_CAPS` (#1010, incident
    #958: an 80 GB-disk / 32 GB-RAM plan was dispatched onto a 50 GB / 16 GB
    ``cpu3c-8-16`` pod and refused only AFTER a full paid provision cycle).

    Subclass of :class:`CpuExhaustedNoRunpodLaneError` so every existing
    catch site keeps working unchanged; ``classify_terminal_exception``
    (``issue_dispatch.py``) maps it to the DISTINCT reason
    :data:`ROUTE_REASON_CPU_FALLBACK_INFEASIBLE`
    (``cpu_fallback_infeasible_for_plan``) — like the parent's reason, NOT in
    the watcher's ``TRANSIENT_CAPACITY_REASONS``: the RunPod instance can
    never grow to fit the plan, so auto-retry would loop a
    structurally-infeasible launch. (GCP capacity COULD free up later, but
    that sub-case is a deliberate manual re-dispatch decision — park-not-loop
    matches today's experimenter-refusal end-state.) Inherits ``__init__``
    verbatim (reason message + attempts).
    """


class BackendPrepareError(RouteError):
    """``backend.prepare(spec)`` failed BEFORE launch (provision-class).

    Nothing is live when this raises — ``prepare`` runs strictly before
    any submit/provision inside :func:`_prepare_and_launch`, so the
    failure carries normal provision-failure semantics: next tier on
    the auto chain, typed terminal on an explicit override (the
    dispatch CLI's ``classify_terminal_exception`` translates it to
    ``epm:failure (failure_class: infra)``). Wraps the underlying
    exception (rsync/scp non-zero exit, SSH refusal) via
    ``raise ... from exc``.
    """

    def __init__(
        self,
        reason: str,
        *,
        kind: BackendKind,
        cluster: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.kind = kind
        self.cluster = cluster


class WorkloadSurfacedError(RouteError):
    """A backend reported a WORKLOAD failure (not provisioning).

    The router does NOT auto-fallback on this — a deterministic
    workload bug would just re-crash on the next tier. The orchestrator
    posts ``epm:failure (failure_class: code)`` and parks.
    """

    def __init__(
        self,
        reason: str,
        *,
        chosen_kind: BackendKind,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.chosen_kind = chosen_kind
        self.evidence = dict(evidence or {})


class GcpAttemptCapExceededError(RouteError):
    """Per-issue/day GCP attempt-count guard tripped.

    The router refuses to escalate to GCP after
    :data:`MAX_GCP_ATTEMPTS_PER_DAY` attempts in the same UTC day for the
    same issue. The orchestrator surfaces this as an infra block (a
    looping classifier is in scope #1 of the halt criteria — "fact only
    the user knows: should I keep trying").
    """

    def __init__(self, *, issue: int, attempts_today: int, cap: int) -> None:
        super().__init__(
            f"GCP auto-escalation cap of {cap} attempts reached for issue {issue} "
            f"today (attempts_today={attempts_today}); refusing to escalate further. "
            "Lease counter resets at midnight UTC."
        )
        self.issue = issue
        self.attempts_today = attempts_today
        self.cap = cap


class ManualAttentionRequiredError(RouteError):
    """The cancel state machine timed out without confirming the job was dead.

    The router issued ``scancel``/``teardown`` but the job remained live in
    the cluster queue after :data:`CANCEL_LIVE_GRACE_SECONDS`. We CANNOT
    silently escalate to GCP: the free-lane job MAY still be alive, and a
    GCP escalation would launch a second copy under the same attempt-id
    namespace (artifact collision + double spend). The orchestrator surfaces
    this as an infra block with the orphaned job id so the operator can
    confirm + manually ``scancel``. The lease is left intact for the
    orchestrator to consult and the cluster job ``--time`` budget will
    eventually reap it on its own.
    """

    def __init__(
        self,
        *,
        kind: BackendKind,
        cluster: str | None,
        orphaned_job_id: str,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            f"cancel grace expired without confirming termination of "
            f"{kind}/{cluster or 'no-cluster'} job {orphaned_job_id!r}; "
            "refusing to escalate (would risk duplicate run). Operator: "
            f"verify job state, manually scancel if alive."
        )
        self.kind = kind
        self.cluster = cluster
        self.orphaned_job_id = orphaned_job_id
        self.attempts = list(attempts or [])


class _GcpWorkloadFailover(RouteError):
    """INTERNAL control-flow signal: a GCP rung failed the WORKLOAD; fail over to RunPod.

    NOT a public terminal — it never escapes :func:`route`. A
    :class:`gcp.GcpWorkloadError` on a GCP ladder rung raises this so the
    ladder STOPS (no cascade across the remaining GCP rungs) and the lane
    callers (:func:`_auto_route`, :func:`_override_gcp_with_ladder`) route
    the run STRAIGHT to the RunPod terminal rung — bypassing the free
    SLURM lanes too (broken workload code re-crashing there burns queue
    time, the original no-cascade concern). Carries the workload evidence +
    a residual-gap string for the RunPod marker.

    The reversed invariant (user-directed 2026-06-24, task #658): a GCP
    failure of ANY class — capacity miss OR workload failure — now routes
    the next attempt to RunPod, because RunPod pods persist + are SSH-able
    so they are strictly better for diagnosis than GCP's delete-on-crash
    boot disk. See :data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD` for the
    bound that prevents an infinite RunPod cascade.
    """

    def __init__(self, *, residual_gap: str, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(residual_gap)
        self.residual_gap = residual_gap
        self.evidence = dict(evidence or {})


# ---------------------------------------------------------------------------
# RouteResult — what the router returns on success
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteAttempt:
    """One per-tier attempt the router made.

    Recorded in :attr:`RouteResult.attempts` (and in the marker body) so
    the operator can see the full ladder: which lanes were tried, how
    each one resolved, and why the final lane was chosen.
    """

    kind: BackendKind
    cluster: str | None
    est_start_seconds_raw: float | None
    est_start_seconds_clamped: float | None
    outcome: str
    detail: str = ""
    elapsed_seconds: float = 0.0
    # Additive structured evidence for this attempt (#774). The GCP catch sites
    # populate it from the GcpProvisioningError's per-zone fan-out
    # (``per_zone_attempts`` + ``zones_attempted_summary``) so the
    # ``epm:backend-selected`` marker carries the full zone trail, not just the
    # last zone's stderr (the #763 single-zone misdiagnosis). Default empty so
    # every existing positional/keyword construction is unchanged AND
    # ``_attempt_to_dict`` omits the key for the common no-evidence attempt
    # (byte-identical serialized shape for every pre-#774 reader).
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteResult:
    """Outcome of a successful :func:`route` call.

    On terminal failure the router RAISES (:class:`NoComputeAvailableError`,
    :class:`WorkloadSurfacedError`, :class:`GcpAttemptCapExceededError`)
    rather than returning a result whose ``handle`` is None — the caller
    should never have to defensively check whether a result is "real".
    """

    backend: ComputeBackend
    handle: RunHandle
    requested_kind: BackendKind | None
    chosen_kind: BackendKind
    reason: str
    cluster: str | None
    attempts: list[RouteAttempt]
    elapsed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-lane spec threading
# ---------------------------------------------------------------------------


def _spec_for_lane(spec: RunSpec, kind: BackendKind) -> RunSpec:
    """Thread the lane's cluster name into ``spec.cluster`` for per-cluster lanes.

    The shared ``SlurmBackend`` instance serves every SLURM lane and
    resolves its target cluster from ``spec.cluster`` on each call.
    Nothing upstream set ``spec.cluster`` for explicit per-cluster lane
    overrides or auto-chain lane attempts, so the backend's defensive
    nibi default silently submitted EVERY lane to Nibi — live finding,
    issue 535: the 'mila' lane's sbatch landed on Nibi (job 15876369,
    account ``rrg-bengioy-ad_gpu``, ``/scratch/tjiral`` scratch) while
    every lane-level label (routing marker, HF subfolder, figure) said
    mila, and the lane PASSed its checklist vacuously. Non-cluster lanes
    (gcp / runpod) pass through unchanged; a contradicting explicit
    ``spec.cluster`` raises instead of guessing.
    """
    if kind not in _PER_CLUSTER_LANES:
        return spec
    if spec.cluster is None:
        return replace(spec, cluster=kind)
    if spec.cluster != kind:
        raise RouteError(
            f"spec.cluster={spec.cluster!r} contradicts the {kind!r} lane — refusing to launch"
        )
    return spec


# ---------------------------------------------------------------------------
# Spec canonicalization (stable hash for lease keying)
# ---------------------------------------------------------------------------


def canonicalize_spec(spec: RunSpec) -> dict[str, Any]:
    """Return a JSON-canonical dict representation of ``spec``.

    Two specs that produce the same workload (same issue, intent, gpus,
    hydra args, account, time budget, backend, cluster) MUST hash to the
    same key — harmless serialization diffs (dict ordering inside
    ``extra``, integer vs float wall-time) MUST NOT change the key.
    The lease reconnect path uses this hash to decide whether a stored
    lease applies to the current request; a flaky hash would silently
    miss live runs and double-submit.

    We canonicalize by:

    1. Sorting every dict (``extra``, nested dicts) at output time via
       ``json.dumps(..., sort_keys=True)`` — done at the call site that
       hashes the dict.
    2. Casting ``hydra_args`` to a tuple of strings (already frozen on
       :class:`RunSpec`, but defensively re-tuple in case a caller
       mutated).
    3. Coercing ``time_budget_hours`` to a normalized float string so
       ``6`` and ``6.0`` hash identically.
    4. Dropping ``extra`` keys the lease system itself sets
       (``attempt_id`` — recorded SEPARATELY in the lease, not in the
       spec-hash; ``startup_script_path`` — tempfile path that varies
       per launch; ``provisioning_model`` — included since it changes
       intent for re-attempts to be a different request shape; we keep
       it).

    Returns a dict; the caller hashes via
    ``hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()``.
    """
    extra_filtered = {
        k: v
        for k, v in (spec.extra or {}).items()
        if k not in {"attempt_id", "startup_script_path"}
    }
    # Stringify floats so 6 vs 6.0 collide.
    time_budget = (
        f"{float(spec.time_budget_hours):.6f}" if spec.time_budget_hours is not None else None
    )
    canonical: dict[str, Any] = {
        "issue": int(spec.issue),
        "intent": str(spec.intent),
        "gpus": None if spec.gpus is None else int(spec.gpus),
        "time_budget_hours": time_budget,
        "account": spec.account,
        "hydra_args": tuple(str(a) for a in (spec.hydra_args or ())),
        "backend": spec.backend,
        "cluster": spec.cluster,
        "extra": extra_filtered,
    }
    # ``workload_cmd`` (#588) is keyed ONLY when non-empty so every
    # existing hydra-only spec hashes byte-identically across the
    # upgrade (lease reconnect continuity), while a custom-cmd run for
    # the same issue is a distinct lease key.
    if spec.workload_cmd:
        canonical["workload_cmd"] = spec.workload_cmd
    return canonical


def spec_hash(spec: RunSpec) -> str:
    """SHA-256 of the canonicalized spec dict (first 16 hex chars).

    Used as the lease key's spec component; collisions in 16 hex chars
    are astronomically unlikely across a single issue's attempt history
    (issue 137 would need ~2^32 attempts to hit one).
    """
    blob = json.dumps(canonicalize_spec(spec), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Durable routing lease (~/.eps-routing/issue-<N>.json)
# ---------------------------------------------------------------------------


@dataclass
class Lease:
    """In-memory view of a per-issue routing lease.

    Persisted as JSON at ``<lease_dir>/issue-<N>.json``. Holds:

    * ``issue`` — task id (mirrors the filename for sanity).
    * ``spec_hash`` — :func:`spec_hash` of the canonicalized RunSpec the
      lease was opened for. A request whose hash matches reconnects;
      a mismatch implies a different workload shape and the lease is
      stale (the orchestrator's ``set-status approved`` flow should have
      cleared the old lease, but a fresh attempt for a different
      hyperparameter set is also OK — we replace the lease).
    * ``attempt_id`` — the attempt id of the LAST fresh submit (the GCP
      artifact namespace for that launch). Minted FRESH per launch by
      ``_thread_attempt_id_into`` and overwritten here at each fresh
      submit ("lease follows the launch", #927) — NOT stable across the
      lease lifetime. Reconnect id-stability comes from the router-level
      reconnect early-return + the GCP ``eps-attempt`` instance label
      recovery, never from this field.
    * ``backend`` — which backend was used last (``None`` if no submit
      has happened yet but the lease was opened to claim the attempt id).
    * ``cluster`` — cluster name for SLURM backends (``None`` for GCP).
    * ``job_id`` — external job id (SLURM job id; GCE instance id).
      Populated IMMEDIATELY after submit; absence + lease present =
      ``UNKNOWN_SUBMITTED`` recovery state.
    * ``submitted_at`` — Unix timestamp of the submit.
    * ``gcp_attempts_today`` — per-UTC-day GCP-escalation counter (for
      the attempt-count guard).
    * ``gcp_attempts_date`` — ISO date of the day the counter applies to
      (UTC). On a day-change the counter resets.
    """

    issue: int
    spec_hash: str
    attempt_id: str
    backend: BackendKind | None = None
    cluster: str | None = None
    job_id: str | None = None
    submitted_at: float | None = None
    gcp_attempts_today: int = 0
    gcp_attempts_date: str | None = None
    #: DURABLE idempotency record for the ASYNC GCP-workload->RunPod failover
    #: (#659). When this lease's RunPod launch was a failover OF a specific
    #: crashed GCP run, this holds that GCP run's stable identity
    #: (``{"pod_name": ..., "job_id": ...}``). It is the AUTHORITATIVE
    #: "exactly once per GCP crash" record: the poller
    #: (``scripts/backend_poll.py``) stamps it BEFORE the .claude/cache sidecar
    #: write, so even when EDQUOT / a read-only-fs / out-of-inodes fails BOTH
    #: the sidecar write AND the .claude/cache sentinel write, this record
    #: survives at ``~/.eps-routing/`` (a DIFFERENT directory, so independent
    #: of the sidecar's failure mode) and the next poll short-circuits instead
    #: of firing a paid second RunPod launch. ``None`` for every non-failover
    #: lease.
    gcp_failover_of: dict[str, Any] | None = None
    #: DURABLE idempotency record for the RunPod RUNNING-but-no-port host-wedge
    #: failover (#664/#689). The exact sibling of ``gcp_failover_of``: when this
    #: lease's fresh-pod re-provision was a failover OF a specific WEDGED RunPod
    #: pod, this holds that wedged pod's stable identity
    #: (``{"pod_name": ..., "job_id": ...}``). The poller
    #: (``scripts/backend_poll.py``) stamps it AFTER the fresh-pod relaunch
    #: succeeds, so even when EDQUOT / a read-only-fs / out-of-inodes fails BOTH
    #: the ``.claude/cache`` sidecar write AND the same-dir wedge sentinel write,
    #: this record survives at ``~/.eps-routing/`` (a DIFFERENT directory) and the
    #: next poll short-circuits instead of firing a paid second terminate +
    #: re-provision. ``None`` for every non-wedge-failover lease.
    runpod_wedge_failover_of: dict[str, Any] | None = None
    #: DURABLE idempotency record for the RunPod CUDA-IMA repeat host-wedge
    #: failover (#775). The exact sibling of ``runpod_wedge_failover_of``: when
    #: this lease's fresh-host re-provision was a failover OF a specific RunPod
    #: pod that crashed twice with the SAME CUDA-IMA crash signature, this holds
    #: that crashed pod's stable identity (``{"pod_name": ..., "job_id": ...}``).
    #: A SEPARATE field from ``runpod_wedge_failover_of`` so a no-port wedge
    #: failover and a CUDA-IMA repeat failover on the SAME issue do not
    #: cross-suppress (each gets its own one bounded pivot). The poller
    #: (``scripts/backend_poll.py``) stamps it AFTER the fresh-host relaunch
    #: succeeds (the durable safety net for the EDQUOT-on-``.claude/cache`` mode,
    #: same as its sibling) and reads it as the AUTHORITATIVE "this run already
    #: spent its one CUDA-IMA pivot" bound — a SECOND same-signature crash on the
    #: fresh host with this field already set routes to terminal
    #: ``failure_class: code``. ``None`` for every non-CUDA-IMA-failover lease.
    runpod_cuda_ima_failover_of: dict[str, Any] | None = None
    #: Per-rung consecutive pre-workload boot-death streaks (#1029). Keyed by
    #: the GCP ladder-rung label (e.g. ``"flexstart_l4"``; the poller's
    #: ``"unknown_rung"`` fallback pools pre-#1029 handles that lack the
    #: threaded label); each value is ``{"count": int, "date": "YYYY-MM-DD"
    #: (UTC), "last_ts": float, "last_incarnation": str}``. Same-UTC-day scoped
    #: (mirrors ``gcp_attempts_date``: a stale streak must not poison a rung
    #: after the cause — often a since-fixed commit — is gone). Written by the
    #: poller via :func:`record_gcp_boot_death` (incarnation-keyed idempotent
    #: increment), cleared per-rung by :func:`reset_gcp_boot_death_streak` on a
    #: POSITIVE workload signal, read by :func:`gcp_boot_death_streak` for the
    #: route()-side rung skip. Tolerant parse: a malformed payload reads as
    #: ``{}`` (fail-open toward today's behavior).
    gcp_boot_death_streaks: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "spec_hash": self.spec_hash,
            "attempt_id": self.attempt_id,
            "backend": self.backend,
            "cluster": self.cluster,
            "job_id": self.job_id,
            "submitted_at": self.submitted_at,
            "gcp_attempts_today": self.gcp_attempts_today,
            "gcp_attempts_date": self.gcp_attempts_date,
            "gcp_failover_of": self.gcp_failover_of,
            "runpod_wedge_failover_of": self.runpod_wedge_failover_of,
            "runpod_cuda_ima_failover_of": self.runpod_cuda_ima_failover_of,
            "gcp_boot_death_streaks": self.gcp_boot_death_streaks,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Lease:
        raw_failover = payload.get("gcp_failover_of")
        raw_wedge_failover = payload.get("runpod_wedge_failover_of")
        raw_cuda_ima_failover = payload.get("runpod_cuda_ima_failover_of")
        raw_boot_streaks = payload.get("gcp_boot_death_streaks")
        return cls(
            issue=int(payload["issue"]),
            spec_hash=str(payload["spec_hash"]),
            attempt_id=str(payload["attempt_id"]),
            backend=payload.get("backend"),
            cluster=payload.get("cluster"),
            job_id=payload.get("job_id"),
            submitted_at=payload.get("submitted_at"),
            gcp_attempts_today=int(payload.get("gcp_attempts_today", 0)),
            gcp_attempts_date=payload.get("gcp_attempts_date"),
            gcp_failover_of=raw_failover if isinstance(raw_failover, dict) else None,
            runpod_wedge_failover_of=(
                raw_wedge_failover if isinstance(raw_wedge_failover, dict) else None
            ),
            runpod_cuda_ima_failover_of=(
                raw_cuda_ima_failover if isinstance(raw_cuda_ima_failover, dict) else None
            ),
            gcp_boot_death_streaks=(raw_boot_streaks if isinstance(raw_boot_streaks, dict) else {}),
        )

    def is_unknown_submitted(self) -> bool:
        """True iff the lease has a backend but no job id (recovery state)."""
        return self.backend is not None and self.job_id is None


class LeaseStore:
    """flock'd JSON lease persistence at ``<lease_dir>/issue-<N>.json``.

    Every mutation holds an exclusive ``flock`` on the lease's
    PER-ISSUE lock file (``<lease_dir>/issue-<N>.lock``) — NOT on the
    lease JSON file itself, because the lease file is created/replaced
    atomically via a write-temp-then-rename and an flock on a file we
    are about to rename is fragile, AND NOT on a shared directory-level
    lock (which would serialize every issue against every other issue —
    a 600 s free-lane park on issue 137 inside ``store.transaction(137)``
    would block a ``route()`` on issue 200 for up to 10 min).

    The per-issue lock spans read+modify+write so a concurrent
    ``issue-tick`` cron and a manual ``/issue`` for the SAME issue
    can't both decide "no live job, submit fresh" and double-submit.
    Concurrent calls for DIFFERENT issues are not serialized — they
    take different locks and proceed in parallel.

    Defaults to ``~/.eps-routing/`` (override for tests via
    ``lease_dir=tmp_path``). The directory is created on first use with
    mode 0o700 (lease contents include job ids — not secrets, but the
    operator shouldn't need a world-readable record either).
    """

    def __init__(self, lease_dir: Path | None = None) -> None:
        self._lease_dir = lease_dir or (Path.home() / LEASE_STORE_DIRNAME)

    @property
    def lease_dir(self) -> Path:
        return self._lease_dir

    def _ensure_dir(self) -> None:
        self._lease_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # mkdir-with-mode does NOT chmod an existing dir; defensively
        # tighten if a prior run created it with a wider mode.
        os.chmod(self._lease_dir, 0o700)

    def _lease_path(self, issue: int) -> Path:
        return self._lease_dir / f"issue-{int(issue)}.json"

    def _lock_path(self, issue: int) -> Path:
        """Per-issue lock file (``<lease_dir>/issue-<N>.lock``).

        Per-issue (not directory-global) so a long-held lock on one
        issue cannot block routing on a different issue. Cross-issue
        contention is bounded to the concurrent invocations on the
        SAME issue that we are deliberately serializing.
        """
        return self._lease_dir / f"issue-{int(issue)}.lock"

    @contextmanager
    def _flock(self, issue: int) -> Iterator[None]:
        """Exclusive flock on the PER-ISSUE lock file for the block's duration.

        Read-modify-write on the lease MUST happen inside this context
        so a concurrent process on the SAME issue doesn't read a stale
        lease and overwrite a fresh one with stale data. Concurrent
        processes on DIFFERENT issues hold different locks and DO NOT
        contend.
        """
        self._ensure_dir()
        lock_path = self._lock_path(issue)
        # Open in append mode so the file is created if absent + no truncation.
        with open(lock_path, "ab+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def read(self, issue: int) -> Lease | None:
        """Read the lease for ``issue``. Returns ``None`` if absent / malformed."""
        path = self._lease_path(issue)
        with self._flock(issue):
            return self._read_locked(path)

    def _read_locked(self, path: Path) -> Lease | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("LeaseStore: could not read %s: %s; treating as absent.", path, exc)
            return None
        try:
            return Lease.from_json(payload)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("LeaseStore: malformed lease at %s: %s; treating as absent.", path, exc)
            return None

    def write(self, lease: Lease) -> None:
        """Atomic replace of the lease file (write-temp + rename)."""
        path = self._lease_path(lease.issue)
        with self._flock(lease.issue):
            self._write_locked(path, lease)

    def _write_locked(self, path: Path, lease: Lease) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(lease.to_json(), sort_keys=True, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def delete(self, issue: int) -> None:
        """Delete the lease file (idempotent on absent)."""
        path = self._lease_path(issue)
        with self._flock(issue):
            try:
                path.unlink()
            except FileNotFoundError:
                return

    @contextmanager
    def transaction(self, issue: int) -> Iterator[tuple[Lease | None, Callable[[Lease], None]]]:
        """Read-modify-write transaction under the per-issue flock.

        Yields ``(current_lease_or_None, write_fn)``. The caller computes
        the new lease state inside the ``with`` block and invokes
        ``write_fn(new_lease)`` to persist it. The per-issue flock is
        held until the block exits — concurrent ``transaction(other_issue)``
        calls do NOT block on it.

        Example::

            with store.transaction(issue=137) as (lease, write):
                if lease is None:
                    lease = Lease(issue=137, spec_hash=h, attempt_id=a)
                lease.job_id = "9999"
                write(lease)
        """
        self._ensure_dir()
        path = self._lease_path(issue)
        with self._flock(issue):
            current = self._read_locked(path)

            def write_fn(new_lease: Lease) -> None:
                self._write_locked(path, new_lease)

            yield current, write_fn


# ---------------------------------------------------------------------------
# Helpers (auto lane order, estimate ranking, GCP attempt counter)
# ---------------------------------------------------------------------------


def _validate_auto_lane_order(
    lanes: tuple[str, ...],
    *,
    source: str,
) -> tuple[BackendKind, ...]:
    """Validate an auto-chain lane order; raise :class:`RouteError` on any defect.

    Hard rules (all raise — a misconfigured order must NEVER be silently
    repaired by dropping entries):

    * ``runpod`` is FORBIDDEN — RunPod spends real money and stays
      override-only; an order that smuggles it in is a real-money safety
      violation, not a preference.
    * Unknown lane names (typos, the ``auto`` sentinel, the legacy
      ``cluster`` literal) raise.
    * Duplicates raise (a duplicated lane would be attempted twice).
    * An empty order raises.
    """
    if not lanes:
        raise RouteError(f"auto lane order from {source} is empty — refusing to route blind")
    for lane in lanes:
        if lane == "runpod":
            raise RouteError(
                f"auto lane order from {source} contains 'runpod' — RunPod spends "
                "real money and is reachable ONLY via an explicit backend override, "
                "never on the auto chain. Remove it from the order."
            )
        if lane not in _AUTO_LANE_VALUES:
            raise RouteError(
                f"auto lane order from {source} contains unknown lane {lane!r}; "
                f"valid lanes: {sorted(_AUTO_LANE_VALUES)}"
            )
    if len(set(lanes)) != len(lanes):
        raise RouteError(f"auto lane order from {source} contains duplicate lanes: {lanes!r}")
    return lanes  # type: ignore[return-value]


def auto_lane_order() -> tuple[BackendKind, ...]:
    """Resolve the auto-chain lane order: env override, else the standing default.

    * :data:`ENV_AUTO_LANE_ORDER` set (non-empty) → parse the
      comma-separated lane list and validate it (``runpod`` / unknown
      names / duplicates raise loudly — never silently dropped).
    * Otherwise → :data:`DEFAULT_AUTO_LANE_ORDER` (GCP first,
      unconditionally — no date gate of any kind).
    """
    raw = os.environ.get(ENV_AUTO_LANE_ORDER, "").strip()
    if not raw:
        return DEFAULT_AUTO_LANE_ORDER
    lanes = tuple(part.strip() for part in raw.split(",") if part.strip())
    return _validate_auto_lane_order(lanes, source=f"{ENV_AUTO_LANE_ORDER}={raw!r}")


def _split_lane_groups(kinds: list[BackendKind]) -> list[tuple[BackendKind, ...]]:
    """Split availability-filtered lane kinds into contiguous attempt groups.

    Each group is either ``("gcp",)`` or a maximal run of consecutive
    SLURM lanes. The auto chain walks groups in order; WITHIN a SLURM
    group the lanes keep the existing est-start ranking + park + cancel
    chain (ties preserve the configured order — ``rank_lanes`` is
    stable), while a GCP group is a single provision attempt.
    """
    groups: list[tuple[BackendKind, ...]] = []
    current: list[BackendKind] = []
    for kind in kinds:
        if kind == "gcp":
            if current:
                groups.append(tuple(current))
                current = []
            groups.append(("gcp",))
        else:
            current.append(kind)
    if current:
        groups.append(tuple(current))
    return groups


def rank_lanes(
    candidates: list[tuple[ComputeBackend, BackendKind, float | None]],
) -> list[tuple[ComputeBackend, BackendKind, float | None, float]]:
    """Sort candidates by clamped est-start (instant < soon < unknown).

    Input: list of ``(backend, kind, est_start_seconds_raw)``. ``None``
    raw means the lane returned no parseable estimate (still park-
    eligible, but ranks LAST). Negative est-starts clamp to ``0.0`` for
    ranking ("would start in the past" = instant, not "below zero / more
    instant than zero" — slice-4 review carry-forward).

    Returns: list of ``(backend, kind, raw, clamped)`` sorted by clamped
    ascending; unknowns (raw=None) sort to the end via ``float("inf")``
    sentinel. Stable across ties (preserves input order).
    """
    decorated: list[tuple[float, int, ComputeBackend, BackendKind, float | None, float]] = []
    for idx, (backend, kind, raw) in enumerate(candidates):
        if raw is None:
            clamped = float("inf")
        elif raw < 0:
            clamped = 0.0
        else:
            clamped = float(raw)
        decorated.append((clamped, idx, backend, kind, raw, clamped))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [(b, k, raw, clamped) for _c, _i, b, k, raw, clamped in decorated]


def _today_utc_iso() -> str:
    return datetime.now(tz=UTC).date().isoformat()


def _bump_gcp_attempt(lease: Lease) -> Lease:
    """Bump the per-day GCP attempt counter, rolling over on day change."""
    today = _today_utc_iso()
    if lease.gcp_attempts_date != today:
        lease.gcp_attempts_date = today
        lease.gcp_attempts_today = 0
    lease.gcp_attempts_today += 1
    return lease


# ---------------------------------------------------------------------------
# GCP pre-workload boot-death streaks (#1029)
# ---------------------------------------------------------------------------


def gcp_boot_death_streak_threshold() -> int:
    """The #1029 boot-loop breaker threshold N, defaulting to 2.

    Read at CALL time (not import time) from ``EPS_GCP_BOOT_DEATH_STREAK_N`` so
    ops can retune without restarting the poller (mirrors
    ``backend_poll._gcp_queue_wait_seconds``). A missing / non-integer / ``<1``
    value falls back to :data:`GCP_BOOT_DEATH_STREAK_N_DEFAULT` — the threshold
    can never be zero/negative (which would fail over on the FIRST setup death,
    killing legitimate transients) or crash a poll on a typo.
    """
    raw = os.environ.get("EPS_GCP_BOOT_DEATH_STREAK_N")
    if raw is None:
        return GCP_BOOT_DEATH_STREAK_N_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return GCP_BOOT_DEATH_STREAK_N_DEFAULT
    return val if val >= 1 else GCP_BOOT_DEATH_STREAK_N_DEFAULT


def record_gcp_boot_death(
    issue: int, rung: str, *, incarnation: str, lease_store: LeaseStore | None = None
) -> int:
    """flock'd increment of the (issue, rung) consecutive boot-death streak (#1029).

    Rollover-on-day-change: the streak resets when the record's ``date`` is not
    today (UTC), mirroring the ``gcp_attempts_date`` probe. IDEMPOTENT on the
    launch INCARNATION key: a re-poll of the SAME dead instance returns the
    current count unchanged; a DISTINCT incarnation increments and stamps
    ``last_ts`` / ``last_incarnation``. Returns the post-record count.

    The INCARNATION key identifies one VM CREATE, NOT one ``route()`` call: the
    poller builds it as ``str(handle.job_id)`` — the GCE instance id, distinct
    per create by construction and stable across re-polls of one sidecar —
    falling back to ``f"{attempt_id}:{gcp_launched_ts}"`` when ``job_id`` is
    absent. ``attempt_id`` ALONE is FORBIDDEN as the key: #763's five creates
    all shared ``att-20260630-141513`` with DISTINCT instance ids (verified
    from the ``epm:cluster-launched`` markers), so attempt_id-keying would
    dedupe REAL consecutive deaths and freeze the streak at 1 — the breaker
    would never fire on the motivating incident (#927's fresh-mint default
    landed post-incident, and a caller-pinned ``spec.extra`` attempt_id still
    takes precedence). A DEGENERATE key (job_id absent AND both fallback
    components empty — pre-fix handles) is the CALLER's to skip
    (``backend_poll._maybe_escalate_gcp_boot_loop`` skips the record entirely,
    logged, fail-open to today's behavior); this helper defensively no-ops on
    an empty incarnation too, returning the current count.

    A missing lease (no launch has written one — only reachable when the lease
    file was wiped, since every GCP launch writes a lease inside
    ``_attempt_one_gcp_rung``) is CREATED with placeholder ``spec_hash`` /
    ``attempt_id`` so the streak still records; the next real launch's
    ``_thread_attempt_id_into`` / ``_lease_after_submit`` overwrite the
    placeholders as usual.
    """
    store = lease_store or LeaseStore()
    rung = str(rung)
    incarnation = str(incarnation)
    with store.transaction(int(issue)) as (lease, write):
        if lease is None:
            lease = Lease(issue=int(issue), spec_hash="", attempt_id="")
        today = _today_utc_iso()
        streaks = lease.gcp_boot_death_streaks
        rec = streaks.get(rung)
        if not isinstance(rec, dict) or rec.get("date") != today:
            rec = {"count": 0, "date": today, "last_ts": 0.0, "last_incarnation": ""}
        current = _coerce_streak_count(rec.get("count"))
        if not incarnation:
            # Defensive no-op (the poller already skips degenerate keys):
            # keying on "" would dedupe every pre-fix handle's deaths together.
            logger.warning(
                "record_gcp_boot_death: empty incarnation key for issue %d rung %s; "
                "skipping the record (fail-open, #1029)",
                int(issue),
                rung,
            )
            return current
        if rec.get("last_incarnation") == incarnation and current > 0:
            # Re-poll of the SAME dead instance -> idempotent (no increment).
            return current
        count = current + 1
        streaks[rung] = {
            "count": count,
            "date": today,
            "last_ts": float(time.time()),  # wall-clock, not monotonic
            "last_incarnation": incarnation,
        }
        lease.gcp_boot_death_streaks = streaks
        write(lease)
        return count


def reset_gcp_boot_death_streak(
    issue: int, rung: str, *, lease_store: LeaseStore | None = None
) -> None:
    """flock'd drop of the (issue, rung) boot-death streak record (#1029).

    Called by the poller on a POSITIVE workload signal (boot demonstrably
    succeeded — see ``backend_poll._maybe_reset_gcp_boot_streak``). Read-before-
    write: a no-op write is avoided when no record exists for the rung, so the
    common healthy-poll tick never rewrites the lease.
    """
    store = lease_store or LeaseStore()
    rung = str(rung)
    with store.transaction(int(issue)) as (lease, write):
        if lease is None:
            return
        if rung in lease.gcp_boot_death_streaks:
            del lease.gcp_boot_death_streaks[rung]
            write(lease)


def gcp_boot_death_streak(issue: int, rung: str, *, lease_store: LeaseStore | None = None) -> int:
    """Plain read: the (issue, rung) boot-death count IF its date is today, else 0 (#1029).

    A missing / malformed lease or record — or a record from a PRIOR UTC day
    (the same-day scoping :func:`record_gcp_boot_death` writes) — reads 0:
    fail-open toward today's behavior (the rung is attempted), never a false
    skip.
    """
    store = lease_store or LeaseStore()
    try:
        lease = store.read(int(issue))
    except OSError as exc:
        logger.warning(
            "gcp_boot_death_streak: lease read failed for issue %s (%s: %s); reading 0",
            issue,
            type(exc).__name__,
            exc,
        )
        return 0
    if lease is None:
        return 0
    rec = lease.gcp_boot_death_streaks.get(str(rung))
    if not isinstance(rec, dict) or rec.get("date") != _today_utc_iso():
        return 0
    return _coerce_streak_count(rec.get("count"))


def _coerce_streak_count(raw: Any) -> int:
    """Coerce a streak-record ``count`` to a non-negative int (malformed -> 0)."""
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Cancel state machine
# ---------------------------------------------------------------------------


def cancel_and_wait(
    *,
    backend: ComputeBackend,
    handle: RunHandle,
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None = None,
    grace_seconds: int = CANCEL_LIVE_GRACE_SECONDS,
    poll_interval: float = 2.0,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Idempotent cancel: request, poll until job leaves the live queue.

    Returns one of:

    * ``"cancelled"`` — the job is no longer live in the queue (the
      DRAC robot's allowlist forbids ``sacct``, so "no longer live in
      ``squeue --name``" is the best terminal signal we can get).
    * ``"raced_to_running"`` — between cancel-requested and the next
      live-check the job transitioned to RUNNING. We KEEP the job —
      tearing it down would forfeit the wait we already paid for; the
      router uses this lane as the chosen outcome.
    * ``"manual_attention"`` — ``grace_seconds`` elapsed and the job is
      still live. Both the auto and explicit lanes raise
      :class:`ManualAttentionRequiredError` carrying the orphaned job
      id (no silent GCP escalation past a live job — fix6), so the
      operator can manually ``scancel``; the cluster job will
      eventually time out on its own ``--time`` budget regardless.

    ``is_live_after_cancel`` is the polled "is the job ID still
    visible?" probe. The SLURM backend's binding is
    ``squeue --name eps-issue-<N>`` (true while live, false on age-out
    / cancellation). GCP doesn't need this (the auto-fallback path
    never enters the cancel state machine — GCP's "park" is the
    provision call itself), but the abstraction stays uniform.

    ``is_running_after_cancel`` (optional): probe to distinguish
    "actually started running" from "still pending in queue" during the
    cancel grace. When provided, a "true" reply during the grace window
    KEEPS the job and returns ``"raced_to_running"``. When None, the
    function only polls is_live (and a job that flipped to RUNNING but
    is still live will eventually drop out of the live queue when its
    own ``scancel`` lands; we won't notice the RUNNING transition).
    """
    # Request cancel via the backend's teardown. Idempotent on a missing
    # job (the SLURM scancel wrapper logs but does not raise).
    try:
        backend.teardown(handle)
    except Exception as exc:
        logger.warning(
            "cancel_and_wait: teardown raised for %s/%s (%s: %s); continuing to live-poll.",
            handle.backend,
            handle.job_id,
            type(exc).__name__,
            exc,
        )

    start = now_fn()
    while True:
        # If the operator gave us a "did it start" probe and the job
        # is now actually RUNNING, KEEP it. The scancel we just issued
        # raced against the scheduler; the job won — let it finish.
        if is_running_after_cancel is not None:
            try:
                if is_running_after_cancel(backend, handle):
                    logger.info(
                        "cancel_and_wait: %s/%s raced to RUNNING during cancel; keeping.",
                        handle.backend,
                        handle.job_id,
                    )
                    return "raced_to_running"
            except Exception as exc:
                logger.warning(
                    "cancel_and_wait: is_running probe raised (%s: %s); continuing.",
                    type(exc).__name__,
                    exc,
                )
        # If the job is no longer live, we've cancelled successfully.
        try:
            live = is_live_after_cancel(backend, handle)
        except Exception as exc:
            logger.warning(
                "cancel_and_wait: is_live probe raised (%s: %s); treating as still-live.",
                type(exc).__name__,
                exc,
            )
            live = True
        if not live:
            return "cancelled"
        if now_fn() - start >= grace_seconds:
            logger.warning(
                "cancel_and_wait: %s/%s still live after %ds grace; manual_attention.",
                handle.backend,
                handle.job_id,
                grace_seconds,
            )
            return "manual_attention"
        sleep_fn(poll_interval)


# ---------------------------------------------------------------------------
# Park watchdog
# ---------------------------------------------------------------------------


def park_until_running_or_cap(
    *,
    backend: ComputeBackend,
    handle: RunHandle,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    cap_seconds: int = FREE_WAIT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[bool, str, str | None]:
    """Watch a launched handle for ``cap_seconds``; return (started, reason, terminal_status).

    ``is_started`` is the backend-aware probe — for SLURM it queries
    ``squeue -j <id>`` for state RUNNING (the production
    ``slurm_monitor.SLURM_STATE_TO_STATUS`` maps PENDING→running for
    historical reasons, so the router cannot use the PollResult.status
    field directly to distinguish PENDING from RUNNING). For GCP the
    binding is ``backend.poll(handle).status == "running"``. For tests
    the binding is whatever the test double exposes.

    Returns ``(started, reason, terminal_status)``:

    * ``(True, "running", None)`` — job reached RUNNING before the cap.
    * ``(False, "park_cap_exceeded", None)`` — still PENDING (or
      otherwise not-running) at the cap. Caller should run
      :func:`cancel_and_wait` and escalate to the next tier.
    * ``(False, "terminal_before_running", <poll.status>)`` — the
      probe-poll returned a terminal-ish status (done/dead/stalled/gate)
      before RUNNING. ``terminal_status`` is the triggering
      ``PollResult.status`` so the caller can distinguish genuinely-gone
      jobs (``done`` / ``dead`` — eligible for the started-evidence
      probe) from possibly-LIVE ones (``stalled`` covers RUNNING with a
      stale heartbeat and SUSPENDED; ``gate`` is a live wait) which MUST
      go through the cancel state machine first (round-6 M1 — the
      issue-535 live run skipped the cancel on a stalled-classified
      LIVE job and orphaned it).
    * ``(False, "probe_failures_exceeded", None)`` —
      :data:`PARK_MAX_CONSECUTIVE_PROBE_FAILURES` CONSECUTIVE
      ``is_started`` failures: the job state is UNKNOWN (transport
      down), which must never read as "still pending" forever NOR as
      terminal (round-6 B1). Caller routes to the cancel state machine;
      with the transport still broken that resolves to
      ``manual_attention``.
    """
    start = now_fn()
    consecutive_probe_failures = 0
    while True:
        try:
            started = is_started(backend, handle)
            consecutive_probe_failures = 0
        except Exception as exc:
            consecutive_probe_failures += 1
            logger.warning(
                "park: is_started probe raised (%s: %s); consecutive failure %d/%d — "
                "treating as still-pending.",
                type(exc).__name__,
                exc,
                consecutive_probe_failures,
                PARK_MAX_CONSECUTIVE_PROBE_FAILURES,
            )
            if consecutive_probe_failures >= PARK_MAX_CONSECUTIVE_PROBE_FAILURES:
                return False, "probe_failures_exceeded", None
            started = False
        if started:
            return True, "running", None
        # Check for terminal-before-running via the backend's poll.
        # Wrapped so a probe that ALSO raises here doesn't crash.
        try:
            poll = backend.poll(handle)
        except Exception as exc:
            logger.warning(
                "park: backend.poll raised (%s: %s); treating as still-pending.",
                type(exc).__name__,
                exc,
            )
            poll = None
        if poll is not None and _is_terminal_status(poll):
            return False, "terminal_before_running", poll.status
        if now_fn() - start >= cap_seconds:
            return False, "park_cap_exceeded", None
        sleep_fn(poll_interval)


def _is_terminal_status(poll: PollResult) -> bool:
    return poll.status in {"done", "dead", "stalled", "gate"}


def default_is_started(backend: ComputeBackend, handle: RunHandle) -> bool:
    """Default ``is_started`` probe: ``backend.poll(handle).status == "running"``.

    Production callers wiring the SLURM backend MUST override this with
    a ``squeue -j <id>``-based probe (slurm_monitor's state mapping
    treats SLURM PENDING as PollResult.status="running", which would
    short-circuit the park watchdog incorrectly). Tests use this default
    against backends whose ``poll`` is mocked to return "pending" /
    "running" as needed.
    """
    return backend.poll(handle).status == "running"


def default_is_live(backend: ComputeBackend, handle: RunHandle) -> bool:
    """Default ``is_live_after_cancel`` probe.

    Falls back to ``backend.poll(handle).status not in {done, dead}``
    which is a coarse proxy. Production SLURM callers should bind this
    to ``squeue --name eps-issue-<N>`` returning non-empty (the
    authoritative "still in the queue" signal even when ``scontrol``
    has aged out).
    """
    status = backend.poll(handle).status
    return status not in {"done", "dead"}


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Per-call knobs for :func:`route`.

    Defaults reproduce the plan's production policy. Tests override
    every callable + the lease store + the lane factories.
    """

    free_wait_seconds: int = FREE_WAIT_SECONDS
    poll_interval: float = DEFAULT_POLL_INTERVAL
    cancel_grace_seconds: int = CANCEL_LIVE_GRACE_SECONDS
    max_gcp_attempts_per_day: int = MAX_GCP_ATTEMPTS_PER_DAY
    #: Per-call auto lane order override. ``None`` (the default) resolves
    #: via :func:`auto_lane_order` (env override, else the GCP-first
    #: standing default). A non-None value is validated at ``route()``
    #: entry with the same rules as the env override (``runpod`` /
    #: unknown lanes / duplicates raise).
    lane_order: tuple[BackendKind, ...] | None = None


def route(
    spec: RunSpec,
    *,
    runpod_backend: ComputeBackend,
    free_backends: dict[BackendKind, ComputeBackend] | None = None,
    gcp_backend: ComputeBackend | None = None,
    lease_store: LeaseStore | None = None,
    mila_socket_alive: Callable[[], bool] | None = None,
    is_started: Callable[[ComputeBackend, RunHandle], bool] = default_is_started,
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool] = default_is_live,
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None = None,
    started_evidence_probe: (
        Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None
    ) = None,
    estimate_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None) = None,
    reconnect_fn: (
        Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None
    ) = None,
    marker_poster: Callable[..., None] | None = None,
    on_launched: Callable[[RunHandle], None] | None = None,
    config: RouterConfig | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], datetime] | None = None,
) -> RouteResult:
    """Dispatch ``spec`` to the right backend per the multi-backend ladder.

    See module docstring for the full decision flow.

    Required injections:

    * ``runpod_backend`` — the explicit ``backend: runpod`` override
      target AND (since #656) the auto chain's TERMINAL fallback rung,
      reached ONLY after the cost-ordered GCP ladder + the free SLURM
      lanes are all exhausted (``reason: auto_fallback_runpod``). The
      ordering invariant — RunPod never first, never skipping a cheaper
      rung — is pinned by
      ``test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted``.

    Optional injections:

    * ``free_backends`` — map of free-lane kind → backend instance
      (e.g. ``{"nibi": slurm, "fir": slurm, "mila": mila}``). Auto
      routing visits these at their position in the resolved lane order
      (:attr:`RouterConfig.lane_order`, else :func:`auto_lane_order` —
      env override, else the GCP-first standing default). A missing
      kind is skipped (e.g. ``mila`` absent → router skips Mila even
      when the socket is alive).
    * ``gcp_backend`` — the GCP fallback-ladder target. When ``None`` and
      the auto chain reaches GCP, the GCP lane is skipped and the chain
      falls through (to SLURM / the RunPod terminal rung).
    * ``lease_store`` — defaults to :class:`LeaseStore` at
      ``~/.eps-routing/``. Tests pass a store keyed on ``tmp_path``.
    * ``mila_socket_alive`` — predicate; when ``False``, Mila is
      skipped without a probe. Defaults to ``False`` (router behaves as
      if Mila is down unless wired).
    * ``is_started`` / ``is_live_after_cancel`` — backend-aware probes;
      see the helper docstrings. Defaults use ``backend.poll`` for both
      (sufficient for the test doubles; production wiring overrides for
      SLURM).
    * ``is_running_after_cancel`` — optional probe to detect the
      cancel-race; see :func:`cancel_and_wait`. Defaults to None (no
      race detection).
    * ``started_evidence_probe`` — ``(backend, handle) -> evidence dict
      | None``; consulted ONLY on a ``terminal_before_running`` park
      outcome to distinguish "never started" (no-compute) from "started
      and FAILED fast" (workload failure — surfaced via
      :class:`WorkloadSurfacedError`, NO auto-fallback). Production
      wiring scp/rsync-reads the SLURM scratch dir for ``status.json`` /
      ``job.out`` (``slurm_monitor.fetch_started_evidence``). Defaults
      to None (no probe — every terminal-before-running classifies as
      no-compute, the pre-fix behavior).
    * ``estimate_fn`` — ``(backend, kind, spec) -> seconds | None`` for
      free-lane ranking. Defaults to calling the backend's
      ``estimate_start_seconds(spec)`` method when available, else
      ``None`` (unranked but park-eligible).
    * ``reconnect_fn`` — ``(backend, kind, spec) -> RunHandle | None``;
      the router calls this BEFORE any submit/provision to find an
      existing live job. Defaults to None (no reconnect — fresh
      submit).
    * ``marker_poster`` — see ``epm:backend-selected`` in
      :data:`workflow.yaml`. Defaults to None (no marker posted; slice
      6 wires the real ``post_marker_via_task_py``).
    * ``on_launched`` — persistence hook invoked with the
      :class:`RunHandle` IMMEDIATELY after every successful launch /
      reconnect, BEFORE any marker post or further routing work. The
      dispatch helper wires the handle-sidecar write here so a launched
      handle is ALWAYS recoverable by ``dispatch_issue.py finalize``
      even if everything after the launch crashes. Guarded — a hook
      failure is logged loud and never kills a live launch.
    * ``config`` — see :class:`RouterConfig`. Defaults to a fresh
      instance with the module constants.

    Raises:

    * :class:`NoComputeAvailableError` — terminal no-compute outcome,
      now reached only when the RunPod terminal rung ITSELF fails (#656:
      the GCP ladder + cap no longer raise; they fall through to RunPod).
    * :class:`WorkloadSurfacedError` — a backend reported a
      :class:`GcpWorkloadError`; the router does NOT auto-fallback
      (broken workload code must not cascade across rungs/lanes).
    * :class:`ManualAttentionRequiredError` — a free-lane cancel could
      not confirm the job is dead (raised BEFORE the RunPod rung — an
      unconfirmed-dead orphan must not trigger a second submit).
    """
    cfg = config or RouterConfig()
    store = lease_store or LeaseStore()
    started_at = now_fn()
    attempts: list[RouteAttempt] = []

    # :class:`RunSpec.backend` defaults to ``"auto"`` so a direct
    # ``RunSpec(issue, intent)`` routes through the cost-safe auto chain
    # (free lanes → GCP) — a real-money RunPod launch ALWAYS requires
    # an explicit ``backend="runpod"``. Any other recognized value is
    # an explicit override; an unknown value would have been rejected at
    # :class:`BackendKind` parse time.
    #
    # Belt-and-suspenders: a stringly-typed miswire (``backend=""`` /
    # ``backend=None`` / a typo like ``"runpd"``) MUST NOT silently fall
    # through to the auto chain — that would mask a config bug in the
    # caller. ``BackendKind`` parse-time validation only covers spec
    # *construction*; a caller that bypasses the dataclass and mutates
    # ``spec.backend`` post hoc gets caught here.
    if spec.backend in (None, "") or spec.backend not in _VALID_BACKEND_VALUES:
        raise RouteError(
            f"route(): spec.backend must be one of "
            f"{sorted(_VALID_BACKEND_VALUES)!r}, got {spec.backend!r}. "
            "Empty / None / unknown backend strings are rejected to "
            "prevent silent auto-routing of a miswired override."
        )

    # ------------------------------ explicit override --------------------
    if spec.backend == "runpod":
        return _override_runpod(
            spec=spec,
            backend=runpod_backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
        )

    if spec.backend in {"nibi", "fir", "mila"}:
        free = (free_backends or {}).get(spec.backend)
        if free is None:
            raise RouteError(
                f"backend override {spec.backend!r} requested but no free backend wired for it"
            )
        spec = _spec_for_lane(spec, spec.backend)
        return _override_free_or_gcp(
            spec=spec,
            backend=free,
            kind=spec.backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            reconnect_fn=reconnect_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            started_evidence_probe=started_evidence_probe,
        )

    if spec.backend == "gcp":
        if gcp_backend is None:
            raise RouteError("backend override 'gcp' requested but no gcp_backend wired")
        return _override_gcp_with_ladder(
            spec=spec,
            gcp_backend=gcp_backend,
            runpod_backend=runpod_backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            reconnect_fn=reconnect_fn,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
        )

    # ----------------------------- auto chain ---------------------------
    # Resolve the lane order ONCE at entry (fail-fast on a malformed env
    # override / config order, before any reconnect or submit I/O).
    if cfg.lane_order is not None:
        lane_order = _validate_auto_lane_order(
            tuple(cfg.lane_order), source="RouterConfig.lane_order"
        )
        order_source = "RouterConfig.lane_order"
    else:
        lane_order = auto_lane_order()
        order_source = (
            f"{ENV_AUTO_LANE_ORDER} env override"
            if os.environ.get(ENV_AUTO_LANE_ORDER, "").strip()
            else "default (GCP-first standing order)"
        )
    logger.info(
        "route(): issue=%d auto lane order = %s (source: %s)",
        spec.issue,
        " -> ".join(lane_order),
        order_source,
    )
    return _auto_route(
        spec=spec,
        free_backends=free_backends or {},
        gcp_backend=gcp_backend,
        runpod_backend=runpod_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        cfg=cfg,
        lane_order=lane_order,
        is_started=is_started,
        is_live_after_cancel=is_live_after_cancel,
        is_running_after_cancel=is_running_after_cancel,
        started_evidence_probe=started_evidence_probe,
        mila_socket_alive=mila_socket_alive,
        estimate_fn=estimate_fn,
        reconnect_fn=reconnect_fn,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
        clock_fn=clock_fn,
    )


# ---------------------------------------------------------------------------
# Override paths
# ---------------------------------------------------------------------------


def _invoke_on_launched(
    on_launched: Callable[[RunHandle], None] | None,
    handle: RunHandle,
) -> None:
    """Run the post-launch persistence hook; NEVER let it kill a live launch.

    The hook fires IMMEDIATELY after a successful launch / reconnect,
    BEFORE any marker post, so the dispatch helper's handle-sidecar
    write lands while the only thing that has happened is the launch
    itself — a crash anywhere later still leaves a recoverable handle
    for ``dispatch_issue.py finalize``. A hook failure (e.g. disk
    error on the sidecar write) is logged LOUD and swallowed: the
    launch already succeeded, and the dispatch helper's authoritative
    final write is the second chance.
    """
    if on_launched is None:
        return
    try:
        on_launched(handle)
    except Exception:
        logger.exception(
            "route: on_launched hook FAILED for job_id=%s pod_name=%s — handle "
            "persistence may be missing; continuing (launch already succeeded).",
            handle.job_id,
            handle.pod_name,
        )


def _prepare_and_launch(
    backend: ComputeBackend,
    spec: RunSpec,
    *,
    kind: BackendKind,
    cluster: str | None = None,
) -> RunHandle:
    """FRESH-launch chokepoint: ``backend.prepare(spec)`` then ``backend.launch(spec)``.

    Every fresh launch site MUST go through this helper. Live-acceptance
    finding (issue 535): the router called ``launch`` directly at every
    site, so ``SlurmBackend.prepare`` — the rsync repo sync +
    ``render_secrets_env()`` + secrets push — had ZERO production
    callers, and the first live Nibi job died in in-job preflight with
    no rsynced repo and no ``secrets.env`` in its scratch dir.

    RECONNECT sites must NOT call this helper (and must not call
    ``prepare`` at all): ``SlurmBackend.prepare`` rsyncs the scratch dir
    with ``--delete``, so re-preparing the scratch of a RUNNING job
    could yank code out from under the live workload mid-run.

    A ``prepare`` failure raises :class:`BackendPrepareError` —
    provision-class, pre-launch, nothing live. The auto chain treats it
    like a launch failure (next tier); explicit overrides surface it as
    a typed terminal.
    """
    # ``prepare`` is an @abstractmethod on ComputeBackend, so every REAL backend
    # (SlurmBackend / GcpBackend / RunPodBackend) implements it — production
    # behavior is byte-identical. The getattr-guard only tolerates a minimal
    # duck-typed backend double that omits prepare entirely (e.g. a passive
    # RunPod stand-in modelling RunPodBackend.prepare's documented no-op); a
    # backend that DEFINES prepare and raises still surfaces as a
    # BackendPrepareError, so the #535 SLURM rsync/secrets guarantee is intact.
    prepare = getattr(backend, "prepare", None)
    if callable(prepare):
        try:
            prepare(spec)
        except Exception as exc:
            raise BackendPrepareError(
                f"backend.prepare failed for {kind}/{cluster or 'no-cluster'} "
                f"({type(exc).__name__}: {exc})",
                kind=kind,
                cluster=cluster,
            ) from exc
    return backend.launch(spec)


def _probe_started_evidence(
    probe: Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None,
    backend: ComputeBackend,
    handle: RunHandle,
) -> dict[str, Any] | None:
    """Run the started-evidence probe; fail OPEN (``None``) on probe failure.

    Used on a ``terminal_before_running`` park outcome to distinguish
    "never started" (genuine no-compute) from "started and FAILED fast"
    (PD→R→exit between polls — a WORKLOAD failure that must surface
    with NO auto-fallback; escalating it to GCP would burn paid credit
    re-running a doomed workload).

    ``None`` — whether because no probe is wired, the probe found no
    runtime artifacts, or the probe itself failed — preserves the
    legacy ``no_compute_available`` classification. A probe failure is
    logged LOUD but never becomes a new crash path between "job
    vanished" and the router terminal.
    """
    if probe is None:
        return None
    try:
        return probe(backend, handle)
    except Exception as exc:
        logger.warning(
            "route: started-evidence probe FAILED for %s/%s (%s: %s); "
            "falling back to no_compute classification.",
            handle.backend,
            handle.job_id,
            type(exc).__name__,
            exc,
        )
        return None


def _override_runpod(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult:
    """Explicit RunPod override — just submit. No park, no fallback.

    RunPod's "start time" is the few-minute provision; we don't gate it
    behind a park watchdog (the existing RunPod flow doesn't, and a 10
    min park would force a real user-meaningful timeout where today
    there is none). Reconnect via the lease's job_id is wired in slice
    6 (the RunPod backend doesn't yet expose a "find live pod by name"
    handle-reconstructor; today the existing pod_lifecycle.py path is
    idempotent itself).

    Lock discipline: holds the per-issue flock across the launch + lease
    write so a concurrent invocation cannot double-submit. RunPod
    provisioning is seconds-to-minutes; the lock is per-ISSUE (not
    cross-issue) so contention is bounded to the racing invocations we
    are deliberately serializing.
    """
    # #940: translate a GCP-only GPU intent to its RunPod-provisionable
    # equivalent BEFORE the launch (and BEFORE the per-issue flock, so a
    # translation ValueError never holds it). The helper's ValueError
    # propagates raw: an explicit `backend: runpod` pin of an unmapped
    # GCP-only intent (eval-h100) is a CONFIG error — fail loud pre-launch
    # with the map-row-naming message (same class as gcp.machine_for_intent's
    # ValueError on a gcp override). RunPod-native / RunPod-only intents
    # (lora-7b, ft-70b, custom) hit the verbatim branch — byte-identical.
    runpod_intent, intent_translation = _translated_runpod_intent(spec)
    if intent_translation is not None:
        logger.warning(
            "route: explicit runpod override translating GCP-only intent %r -> %r (issue %d).",
            spec.intent,
            runpod_intent,
            spec.issue,
        )
        spec = replace(
            spec,
            intent=runpod_intent,
            extra={**(spec.extra or {}), "runpod_intent_translation": intent_translation},
        )
    # Hold the per-issue flock across launch + persist so two concurrent
    # route() calls cannot both decide "no live job, submit fresh" and
    # provision twice.
    with store.transaction(spec.issue) as (lease, write):
        try:
            handle = _prepare_and_launch(backend, spec, kind="runpod")
        except BackendPrepareError as exc:
            attempts.append(
                RouteAttempt(
                    kind="runpod",
                    cluster=None,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind="runpod",
                attempts=attempts,
            )
            raise
        _invoke_on_launched(on_launched, handle)
        write(_lease_after_submit(lease, spec, "runpod", None, handle))
    attempt = RouteAttempt(
        kind="runpod",
        cluster=None,
        est_start_seconds_raw=None,
        est_start_seconds_clamped=None,
        outcome="launched",
        detail="explicit override",
        elapsed_seconds=now_fn() - started_at,
    )
    attempts.append(attempt)
    result = RouteResult(
        backend=backend,
        handle=handle,
        requested_kind="runpod",
        chosen_kind="runpod",
        reason=ROUTE_REASON_OVERRIDE,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        # #940: the GCP-only -> RunPod intent translation record, when one
        # applied, so the override marker records the intent swap too.
        extra=({"runpod_intent_translation": intent_translation} if intent_translation else {}),
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def _provisioning_detail(exc: GcpProvisioningError) -> str:
    """Attempt detail for a GCP provisioning failure — reason + captured stderr tail.

    ``classify_create_failure`` packages the gcloud stderr into
    ``exc.evidence["stderr_tail"]``, but the pre-#608 handlers recorded
    only ``exc.reason`` ("... (stderr below)" with nothing below): the
    four quota-doomed creates on issue 608 left no stderr anywhere
    (marker, failure JSON, logs) and root-causing took a manual gcloud
    reproduction. This detail flows into the ``epm:backend-selected``
    attempt rows AND the ``NoComputeAvailableError.attempts`` that
    ``classify_terminal_exception`` serializes into the terminal failure
    JSON, so the evidence survives in both surfaces.
    """
    tail = str(exc.evidence.get("stderr_tail") or "").strip()
    if not tail:
        return exc.reason
    return f"{exc.reason}; stderr_tail: {tail[-1024:]}"


def _provisioning_evidence(exc: GcpProvisioningError) -> dict[str, Any]:
    """Per-zone fan-out evidence for a GCP provisioning failure (#774).

    Lifts the rich per-zone trail (``per_zone_attempts`` — one
    ``{zone, returncode, matched_pattern, elapsed_s, stderr_tail}`` record per
    zone the GCP create loop tried) plus the human-readable
    ``zones_attempted_summary`` off the :class:`GcpProvisioningError`, so the
    ``RouteAttempt.evidence`` it populates carries the full zone coverage to the
    ``epm:backend-selected`` marker. The bare ``zones_attempted`` name-list is
    NOT lifted when ``per_zone_attempts`` is present — the richer record
    supersedes it (each per-zone record already carries its zone name).

    Returns an EMPTY dict when the error carries no fan-out evidence (e.g. a
    single-zone non-capacity raise that never entered the loop), so
    ``_attempt_to_dict`` omits the ``evidence`` key entirely and the attempt
    serializes byte-identically to the pre-#774 shape.
    """
    out: dict[str, Any] = {}
    per_zone = exc.evidence.get("per_zone_attempts")
    if per_zone:
        out["per_zone_attempts"] = per_zone
    summary = exc.evidence.get("zones_attempted_summary")
    if summary:
        out["zones_attempted_summary"] = summary
    return out


def _gcp_quota_headroom_or_none(backend: ComputeBackend, spec: RunSpec) -> QuotaHeadroom | None:
    """Run the GCP regional-quota headroom pre-check; fail OPEN (``None``) on any failure.

    Duck-typed via the backend's ``preflight_quota_headroom`` method so
    test doubles / backends without the probe skip the pre-check entirely
    (#608: four guaranteed-fail creates burned the daily attempt cap
    against an exhausted regional accelerator quota). ``None`` means "no
    opinion — proceed to launch exactly as before"; only a POSITIVE
    insufficient-headroom reading skips the lane.
    """
    probe = getattr(backend, "preflight_quota_headroom", None)
    if probe is None:
        return None
    try:
        return probe(spec)
    except Exception as exc:
        logger.warning(
            "route: GCP quota-headroom pre-check failed OPEN (%s: %s); "
            "proceeding to launch as before.",
            type(exc).__name__,
            exc,
        )
        return None


def _spot_max_gpu_hours() -> float:
    """The GPU-hour threshold below which a job is "short" enough for SPOT.

    Reads :data:`ENV_SPOT_MAX_GPU_HOURS`; falls back to
    :data:`DEFAULT_SPOT_MAX_GPU_HOURS` when unset or unparseable (logged
    LOUD on a bad value rather than silently mis-gating).
    """
    raw = os.environ.get(ENV_SPOT_MAX_GPU_HOURS, "").strip()
    if not raw:
        return DEFAULT_SPOT_MAX_GPU_HOURS
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is unparseable as a float; using the default %g GPU-hours.",
            ENV_SPOT_MAX_GPU_HOURS,
            raw,
            DEFAULT_SPOT_MAX_GPU_HOURS,
        )
        return DEFAULT_SPOT_MAX_GPU_HOURS


def _estimated_gpu_hours(spec: RunSpec, machine: MachineSpec) -> float | None:
    """Estimate GPU-hours = wall-budget x gpu_count; ``None`` when unknown.

    Reads ``spec.extra["estimated_gpu_hours"]`` first (an explicit override
    the plan / orchestrator may thread), else
    ``spec.time_budget_hours * gpu_count``. Returns ``None`` when neither is
    available — the CONSERVATIVE signal: an unknown-length job is NOT short
    (see :func:`_is_short_job`), so it never reaches a SPOT rung and is
    never silently preempted.
    """
    explicit = (spec.extra or {}).get("estimated_gpu_hours")
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            logger.warning(
                "route: spec.extra['estimated_gpu_hours']=%r is unparseable; "
                "treating the job as unknown-length (NOT short).",
                explicit,
            )
            return None
    if spec.time_budget_hours is None:
        return None
    return float(spec.time_budget_hours) * max(1, machine.gpu_count)


def _is_short_job(spec: RunSpec, machine: MachineSpec) -> bool:
    """True when the job is short enough to risk SPOT preemption.

    A caller may FORCE a job "short enough" past the threshold by setting
    ``spec.extra["spot_tolerant"]`` — the explicit opt-into-preemption
    override that survives the #537→#656 ladder migration (a workload that
    declares itself preemption-recoverable). Otherwise the job is short iff
    its estimated GPU-hours (:func:`_estimated_gpu_hours`) are known AND at
    or below :func:`_spot_max_gpu_hours`. UNKNOWN length ⇒ NOT short
    (conservative — see :func:`_estimated_gpu_hours`).
    """
    if bool((spec.extra or {}).get("spot_tolerant")):
        return True
    gh = _estimated_gpu_hours(spec, machine)
    return gh is not None and gh <= _spot_max_gpu_hours()


def _with_machine(spec: RunSpec, machine: MachineSpec, *, provisioning: str) -> RunSpec:
    """Return ``spec`` with the GCP machine override + provisioning model set.

    Threads the rung's :class:`MachineSpec` via
    ``spec.extra["machine_spec_override"]`` (a JSON-safe dict so it
    round-trips through the handle sidecar) so EVERY downstream chokepoint
    — :func:`gcp.machine_for_intent`, :func:`gcp.quota_metric_for`, the
    zone filter, :func:`gcp.render_create_argv` — resolves the rung's TRUE
    machine without mutating the frozen :class:`RunSpec`'s semantic
    ``intent``. ``machine_kind_tag`` rides alongside for marker readability.
    """
    return replace(
        spec,
        extra={
            **(spec.extra or {}),
            "machine_spec_override": {
                "machine_type": machine.machine_type,
                "gpu_count": machine.gpu_count,
                "gpu_kind": machine.gpu_kind,
            },
            "machine_kind_tag": machine.gpu_kind,
            "provisioning_model": provisioning,
        },
    )


def _machine_label(machine: MachineSpec) -> str:
    """Compose a rung-label suffix from the RESOLVED machine's accelerator kind.

    Derives the suffix from :attr:`MachineSpec.gpu_kind` (the field whose
    documented purpose is "short kind tag for logging") rather than hardcoding
    a position in the ladder: ``"A100-80" -> "a100_80"``, ``"L4" -> "l4"``,
    ``"A100-40" -> "a100_40"``, ``"H100-80" -> "h100_80"``. The A100 suffixes
    are byte-identical to the historical hardcoded labels, so existing rungs
    keep their labels; only a non-A100 intent (``debug`` / ``eval`` -> L4) now
    gets a label that matches the machine actually attempted (#672 — the rung-1
    label used to say ``ondemand_a100_80`` while the create attempted an L4).
    """
    return machine.gpu_kind.lower().replace("-", "_")


def _flex_start_rung(spec: RunSpec, machine: MachineSpec) -> tuple[RunSpec, str]:
    """A FLEX_START (DWS flex-start) ladder rung for ``machine``.

    Threads ``provisioning_model=FLEX_START`` via :func:`_with_machine`; the
    request-validity window (``DEFAULT_REQUEST_VALID_FOR_DURATION='2h'``) and
    the 7-day max-run-duration cap are resolved / asserted DOWNSTREAM at
    :func:`gcp.render_create_argv` (``resolve_request_valid_for_duration`` /
    ``_assert_max_run_within_flex_cap``) — the router does not set them. Label
    ``flexstart_<gpu_kind>`` (e.g. ``flexstart_a100_80``). #680.
    """
    return (
        _with_machine(spec, machine, provisioning="FLEX_START"),
        f"flexstart_{_machine_label(machine)}",
    )


def _requested_wide_widths(spec: RunSpec, base: MachineSpec) -> list[int]:
    """Descending wide widths for a width-declaring dispatch; ``[]`` otherwise.

    ``[]`` when ``spec.gpus`` is None, <= the base machine's width, the
    intent is not width-eligible, or the requested width is UNSUPPORTED
    (not a :data:`gcp.WIDE_A100_80_BY_WIDTH` key) — then today's
    ignore-``spec.gpus`` semantics hold. The ``dispatch_issue.py``
    pre-route guard is the loud front door; a library caller bypassing it
    gets a ``logger.warning``, not a silent nothing, AND never a silent
    snap-down: without the supported-width gate, a library-seam ``gpus=6``
    would walk ``[4, 2]`` — the snap-down plan #1121 §4b design point 6
    explicitly bans (an idle-width provision must be a deliberate choice).
    """
    if spec.gpus is None:
        return []
    g = int(spec.gpus)
    if g <= base.gpu_count:
        return []
    if spec.intent not in WIDTH_ELIGIBLE_INTENTS:
        logger.warning(
            "route: spec.gpus=%d ignored by the GCP lane for non-width-eligible intent %r (#1121).",
            g,
            spec.intent,
        )
        return []
    if g not in WIDE_A100_80_BY_WIDTH:
        logger.warning(
            "route: unsupported wide width spec.gpus=%d for intent %r — no snap-down; "
            "base-width ladder used. Supported: %s (#1121).",
            g,
            spec.intent,
            sorted(WIDE_A100_80_BY_WIDTH),
        )
        return []
    return [w for w in (8, 4, 2) if base.gpu_count < w <= g]


def _explicit_wide_degrade_widths(spec: RunSpec, base: MachineSpec) -> list[int]:
    """Descending DEGRADED widths for an explicit wide degradable intent; ``[]`` otherwise.

    The explicit-intent sibling of :func:`_requested_wide_widths` (#1379):
    an ``EXPLICIT_WIDE_DEGRADE_INTENTS`` dispatch (``sweep-8g-a100``) whose
    full-width rungs all capacity-miss degrades onto the narrower
    :data:`gcp.WIDE_A100_80_BY_WIDTH` machines (8->4->2) instead of
    starving on the single scarcest config (#825: 2.5h+ stuck on two empty
    8-GPU DWS pools while 4x/2x capacity was abundant). ``[]`` when:
    the intent is not degradable; ``spec.gpus`` names a DIFFERENT width
    (the --gpus path owns width then — pre-route guard refuses mismatches
    anyway); or the caller pinned the width via
    ``spec.extra["width_required"]`` (dispatch_issue.py --width-required —
    a shared-nothing 8-way memory/parallelism need that cannot re-shard).
    """
    if spec.intent not in EXPLICIT_WIDE_DEGRADE_INTENTS:
        return []
    if spec.gpus is not None and int(spec.gpus) != base.gpu_count:
        return []  # unreachable via dispatch_issue (guard refuses), safe for library callers
    if bool((spec.extra or {}).get("width_required")):
        logger.info(
            "route: width_required set — explicit wide intent %r pinned at %dx; "
            "no width degradation (#1379).",
            spec.intent,
            base.gpu_count,
        )
        return []
    return [w for w in sorted(WIDE_A100_80_BY_WIDTH, reverse=True) if w < base.gpu_count]


def _wide_rungs_for_widths(
    spec: RunSpec, widths: list[int], *, pinned: str | None, short: bool
) -> list[tuple[RunSpec, str]]:
    """(spec, label) rungs at each width in ``widths`` (width-major, intra-width
    per the length-aware order / caller pin) — shared by the #1121 --gpus wide
    PREFIX and the #1379 explicit-intent degraded SUFFIX."""
    rungs: list[tuple[RunSpec, str]] = []
    for w in widths:
        m = WIDE_A100_80_BY_WIDTH[w]
        if pinned is not None:
            # Caller pin honored at every width (#537/#680, extended per-width):
            # a pinned SPOT width-8 dispatch walks spot_*x8 -> spot_*x4 ->
            # spot_*x2 -> the pinned base tail, never silently un-pinning.
            wide_pinned_model = str(pinned).upper()
            wide_prefix = {
                "SPOT": "spot",
                "FLEX_START": "flexstart",
                "STANDARD": "ondemand",
            }.get(wide_pinned_model, wide_pinned_model.lower())
            rungs.append(
                (
                    _with_machine(spec, m, provisioning=wide_pinned_model),
                    f"{wide_prefix}_{_machine_label(m)}x{w}",
                )
            )
            continue
        if short:
            rungs.append(
                (_with_machine(spec, m, provisioning="SPOT"), f"spot_{_machine_label(m)}x{w}")
            )
        flex_spec, _flex_base_label = _flex_start_rung(spec, m)
        rungs.append((flex_spec, f"flexstart_{_machine_label(m)}x{w}"))
        rungs.append(
            (_with_machine(spec, m, provisioning="STANDARD"), f"ondemand_{_machine_label(m)}x{w}")
        )
    return rungs


def _declared_width(spec: RunSpec) -> int | None:
    """Caller-DECLARED GPU width for the handle/marker record (#1121/#1379):
    ``spec.gpus`` when set; else the explicit wide intent's own base width
    (so a degraded sweep-8g-a100 launch reads requested=8, realized=4|2
    instead of requested=null); else None. ``INTENT_TO_MACHINE.get`` (not
    ``machine_for_intent``) deliberately bypasses the rung's
    ``machine_spec_override`` — the declared width is the INTENT's, never
    the degraded rung's (the same pattern ``_translated_runpod_intent``
    uses)."""
    if spec.gpus is not None:
        return int(spec.gpus)
    if spec.intent in EXPLICIT_WIDE_DEGRADE_INTENTS:
        m = INTENT_TO_MACHINE.get(spec.intent)
        if m is not None:
            return int(m.gpu_count)
    return None


def _gcp_ladder_specs(spec: RunSpec) -> list[tuple[RunSpec, str]]:
    """Ordered ``(spec, rung_label)`` GCP provisioning attempts, length- and width-aware.

    The cost-ordered fallback ladder BOTH the auto-GCP path
    (:func:`_attempt_gcp_lane`) and the explicit ``backend: gcp`` path
    (:func:`_override_free_or_gcp`) walk, so the two get IDENTICAL fallback
    behavior (acceptance criterion 3 / the #654 fix).

    **Width-aware wide-rung prefix (#1121).** A dispatch that DECLARES a
    shardable multi-GPU axis (``spec.gpus`` > the intent's base machine
    width, width-eligible intent, supported width — see
    :func:`_requested_wide_widths`) walks WIDE ``a2-ultragpu-{8,4,2}g``
    rungs FIRST, width-major: ALL provisioning models at width ``w`` are
    exhausted before width ``w-1`` is accepted (wall-clock is the scarce
    resource; GCP credits are not — a spot-8g attempt is strictly
    preferable to an on-demand-4g one). Intra-width, the EXISTING
    length-aware order applies verbatim (spot -> flex -> on-demand on a
    short job; flex -> on-demand on a long/unknown one; a caller
    ``provisioning_model`` pin walks only the pinned model at every
    width). Wide rung labels carry an ``x<w>`` suffix
    (``spot_a100_80x8``); NO wide A100-40 rungs in v1 (on-demand
    a2-highgpu quota is 1 — a dead rung). Job LENGTH is classified ONCE
    at the WIDEST requested machine and threaded through the whole walk
    including the base tail: GPU-hours = wall x width is the honest
    total-work read, and a width-8-budgeted job that degrades to 1x
    genuinely runs ~8x the budgeted wall, so inheriting the (usually
    LONG) classification in the tail is conservative and correct. When no
    width is requested the classification is EXACTLY today's
    ``_is_short_job(spec, base)`` — width-1 ladders are byte-identical.
    The base-width tail below the wide prefix is the pre-#1121 ladder
    unchanged.

    **Explicit-wide degraded SUFFIX (#1379).** An EXPLICIT wide intent in
    :data:`gcp.EXPLICIT_WIDE_DEGRADE_INTENTS` (``sweep-8g-a100``) whose
    ``spec.gpus`` is None (or equals the base width) gains DEGRADED
    ``a2-ultragpu-{4,2}g`` rungs APPENDED after the base tail — the
    mirror-image of the #1121 wide prefix (the base rungs at the intent's
    own 8g machine ARE the width-8 rungs, so a suffix preserves
    width-major order with zero duplicate creates). Intra-width the same
    length-aware / caller-pinned order applies via the shared
    :func:`_wide_rungs_for_widths` builder; ``spec.extra["width_required"]``
    (dispatch_issue.py ``--width-required``) pins the full width — the
    ladder is then byte-identical to the pre-#1379 one. ``sweep-8g-h100``
    is deliberately NOT degradable (a GPU-TYPE choice; its fallback is the
    type-preserving RunPod 8xH100 terminal rung).

    The base order is keyed on job LENGTH (:func:`_is_short_job`) — #680:

    **SHORT jobs** (known length <= ``EPS_GCP_SPOT_MAX_GPU_HOURS`` OR
    ``spec.extra["spot_tolerant"]``) — spot leads, because a short job
    absorbs a spot preemption cheaply (#659 failover / checkpoint-resume),
    and spot is the cheapest live pool; flex sits between spot and on-demand
    as the "queue for capacity rather than fail" middle rung:

    1. SPOT (rung-1 machine) — ``spot_<gpu_kind>``. *Always present.*
    2. SPOT A100-40 (``a2-highgpu-1g``) — only when the intent fits in 40 GB
       (:func:`gcp.a100_40_fallback_for_intent`). ``spot_a100_40``.
    3. FLEX_START (rung-1 machine) — ``flexstart_<gpu_kind>``.
    4. on-demand (rung-1 machine) — the spec as-is. ``ondemand_<gpu_kind>``.
    5. on-demand A100-40 — only when fits 40 GB. ``ondemand_a100_40``.

    **LONG / UNKNOWN-length jobs** (known length > threshold, OR unknown
    length) — SPOT is barred (preemption too costly); flex —
    non-preemptible-once-running, queues for capacity — leads:

    1. FLEX_START (rung-1 machine) — ``flexstart_<gpu_kind>``.
    2. on-demand (rung-1 machine) — the spec as-is. ``ondemand_<gpu_kind>``.
    3. on-demand A100-40 — only when fits 40 GB. ``ondemand_a100_40``.

    **CPU-only intents (#677/#747)** (``base.gpu_count == 0``) short-circuit
    BEFORE any length / pin branching, splitting by whether the intent has a
    cheap RunPod-CPU lane (:data:`RUNPOD_CPU_INSTANCE_FOR_INTENT`):

    * ``cpu-bigmem`` (NOT in the map, #677) yields exactly one on-demand CPU
      rung (``ondemand_<gpu_kind>``) and NEVER picks up a spot / flex / A100-40
      rung — a reliable machine for a long HF-store download.
    * ``cpu-small`` / ``cpu-mid`` (mapped, #747) yield a GCP SPOT rung
      (``spot_<gpu_kind>``) THEN an on-demand rung WHEN the job is short
      (:func:`_is_short_job`), else on-demand only — the CPU analogue of the
      GPU length-aware axis; still NO flex / A100-40 rung.

    The short-circuit is load-bearing because :func:`_is_short_job` floors
    ``gpu_count`` to 1, so without it the GPU ladder would (mis)classify a CPU
    job.

    Each label is COMPOSED from the resolved machine's accelerator kind
    (:func:`_machine_label`) rather than hardcoded, so a rung's label always
    reflects the machine actually attempted (#672). Rungs that do not apply
    to this intent / length are simply absent, so a long ``ft-7b`` (no
    A100-40 fallback, not short) yields just (flex-80, ondemand-80) and the
    ladder falls through to the next lane / RunPod with NO spot rung
    (acceptance criterion 2). Each rung is a :func:`dataclasses.replace`
    carrying the right machine override + provisioning model so the create
    resolves the rung's true machine.

    **CLI provisioning-model pin honored (#680).** When the CALLER explicitly
    pinned ``spec.extra["provisioning_model"]`` (the ``dispatch_issue.py
    --provisioning-model SPOT|FLEX_START|STANDARD`` deliberate override, #537),
    the length-aware default order is NOT applied — the pin is a hard override.
    The ladder then walks ONLY the pinned provisioning (base machine, then the
    A100-40 fallback when the intent fits 40 GB), so a launch pinned to SPOT
    never silently launches on flex-start / on-demand. ``spec.extra`` is the
    ORIGINAL caller spec here (the per-rung machine override is applied INSIDE
    the returned tuples via :func:`_with_machine`), so this key reflects only a
    caller pin, never a ladder-set value.
    """
    base = machine_for_intent(spec)  # resolves the as-is intent
    # CPU-only intent (#677/#747): no GPU fallback ladder applies (the A100-40
    # rung is a "smaller GPU" fallback; the flex rung queues for GPU capacity —
    # neither makes sense for a CPU machine). The branch below splits by whether
    # the CPU intent has a cheap RunPod-CPU lane (#747): cpu-bigmem (no lane) is
    # reliable on-demand only, while cpu-small/cpu-mid (mapped) take a GCP SPOT
    # rung on a SHORT/restartable job. This MUST come before the length-aware
    # branching below: _is_short_job floors gpu_count to 1 via
    # max(1, machine.gpu_count) in _estimated_gpu_hours, so a CPU job WOULD
    # otherwise be (mis)classified by the GPU ladder — the short-circuit is
    # load-bearing, not redundant. It also precedes the caller provisioning-model
    # pin so a pinned CPU launch never silently picks up the A100-40 fallback rung.
    if base.gpu_count == 0:
        # cpu-bigmem (no RunPod-CPU lane, #677): reliable single on-demand rung —
        # it may download a big HF store, so a mid-download spot preemption is
        # costly. Yield exactly the on-demand CPU rung, NO spot/flex/A100-40.
        if spec.intent not in RUNPOD_CPU_INSTANCE_FOR_INTENT:
            return [(spec, f"ondemand_{_machine_label(base)}")]
        # Cheap CPU intent (cpu-small / cpu-mid, #747): spot-first on a SHORT /
        # restartable job (the CPU analogue of the GPU length-aware axis,
        # _is_short_job), else on-demand only. NO A100-40 rung (that is a GPU
        # fallback) and NO flex rung (flex queues for GPU capacity — pointless
        # for abundant CPU). _is_short_job floors gpu_count to 1 via
        # max(1, gpu_count) in _estimated_gpu_hours, so the spot rung fires iff
        # the caller threaded a time_budget_hours <= the spot threshold; an
        # UNKNOWN-length CPU job is NOT short -> on-demand only (the correct
        # fail-safe). The caller provisioning-model pin branch below is never
        # reached for a CPU intent (we return here), preserving the #677
        # invariant that a CPU launch never picks up an A100-40 fallback rung.
        cpu_rungs: list[tuple[RunSpec, str]] = []
        if _is_short_job(spec, base):
            cpu_rungs.append(
                (_with_machine(spec, base, provisioning="SPOT"), f"spot_{_machine_label(base)}")
            )
        cpu_rungs.append((spec, f"ondemand_{_machine_label(base)}"))
        return cpu_rungs
    a40 = a100_40_fallback_for_intent(spec)
    pinned = (spec.extra or {}).get("provisioning_model")

    # #1121 width-aware wide-rung PREFIX (width-major: every provisioning
    # model at width w before width w-1). Job length is classified ONCE at
    # the WIDEST requested machine and threaded through the whole walk incl.
    # the base tail (docstring above); when ``wide_widths == []`` this is
    # EXACTLY today's ``_is_short_job(spec, base)`` — width-1 byte-identity.
    # Fact-check note (plan §4b): an explicit
    # ``spec.extra["estimated_gpu_hours"]`` override is machine-independent,
    # so classify-at-widest is a no-op on that path — not an error.
    wide_widths = _requested_wide_widths(spec, base)
    degrade_widths = _explicit_wide_degrade_widths(spec, base)  # NEW (#1379)
    widest = WIDE_A100_80_BY_WIDTH[wide_widths[0]] if wide_widths else base
    # For an explicit wide intent, ``widest`` IS ``base`` (the intent's own
    # 8g machine), so _is_short_job already classifies at the 8-wide machine
    # (GPU-hours = wall x 8) and the degraded rungs inherit it — the same
    # conservative classify-at-widest logic #1121 documented.
    short = _is_short_job(spec, widest)
    # #1121 wide-rung PREFIX + #1379 explicit-intent degraded SUFFIX share
    # ONE per-width rung builder (single source of truth for rung shape /
    # labels). ``wide_widths`` and ``degrade_widths`` are mutually exclusive
    # by construction: _requested_wide_widths requires spec.gpus ABOVE the
    # base width on a WIDTH_ELIGIBLE_INTENTS member (sweep intents are not
    # members), while _explicit_wide_degrade_widths fires only for
    # EXPLICIT_WIDE_DEGRADE_INTENTS members at their own base width.
    rungs: list[tuple[RunSpec, str]] = _wide_rungs_for_widths(
        spec, wide_widths, pinned=pinned, short=short
    )
    degrade_rungs = _wide_rungs_for_widths(spec, degrade_widths, pinned=pinned, short=short)

    # BASE-width tail: the pre-#1121 construction, byte-identical labels AND
    # specs (the base on-demand rung stays the caller spec AS-IS), except
    # ``short`` comes from the single classification above. For an explicit
    # wide degradable intent the base rungs at the intent's own 8g machine
    # ARE the width-8 rungs, so appending ``degrade_rungs`` AFTER the tail
    # preserves width-major order (all provisioning models at width 8 before
    # width 4, before width 2) on every exit path (#1379).
    if pinned is not None:
        # CLI provisioning-model pin (#537/#680): walk ONLY the pinned model.
        pinned_model = str(pinned).upper()
        # Use the SAME label vocabulary the length-aware ladder emits, so a
        # pinned rung's label is consistent with an auto-selected one.
        prefix = {"SPOT": "spot", "FLEX_START": "flexstart", "STANDARD": "ondemand"}.get(
            pinned_model, pinned_model.lower()
        )
        rungs.append(
            (
                _with_machine(spec, base, provisioning=pinned_model),
                f"{prefix}_{_machine_label(base)}",
            )
        )
        if a40 is not None:
            rungs.append(
                (
                    _with_machine(spec, a40, provisioning=pinned_model),
                    f"{prefix}_{_machine_label(a40)}",
                )
            )
        return rungs + degrade_rungs  # #1379: degraded suffix on the pinned path too
    if short:
        # SHORT: spot-first -> flex -> on-demand (spot preemption is cheap here)
        rungs.append(
            (_with_machine(spec, base, provisioning="SPOT"), f"spot_{_machine_label(base)}")
        )
        if a40 is not None:
            rungs.append(
                (_with_machine(spec, a40, provisioning="SPOT"), f"spot_{_machine_label(a40)}")
            )
        rungs.append(_flex_start_rung(spec, base))
        rungs.append((spec, f"ondemand_{_machine_label(base)}"))
        if a40 is not None:
            rungs.append(
                (
                    _with_machine(spec, a40, provisioning="STANDARD"),
                    f"ondemand_{_machine_label(a40)}",
                )
            )
    else:
        # LONG / UNKNOWN: flex-first -> on-demand, NO spot (preemption too costly)
        rungs.append(_flex_start_rung(spec, base))
        rungs.append((spec, f"ondemand_{_machine_label(base)}"))
        if a40 is not None:
            rungs.append(
                (
                    _with_machine(spec, a40, provisioning="STANDARD"),
                    f"ondemand_{_machine_label(a40)}",
                )
            )
    rungs.extend(degrade_rungs)  # #1379: degraded suffix AFTER the base tail
    return rungs


def _skip_gcp_lane_no_headroom(
    *,
    spec: RunSpec,
    headroom: QuotaHeadroom,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    terminal: bool,
) -> None:
    """Skip the GCP lane on a POSITIVE insufficient-headroom reading.

    The original #608 behavior, extracted as a helper. Records a
    ``quota_headroom_insufficient`` attempt WITHOUT bumping the per-day
    attempt counter (the create cannot succeed; it should not consume a
    daily attempt). Non-terminal position: returns ``None`` to continue
    down the lane order. Terminal position: posts the no-compute marker and
    raises :class:`NoComputeAvailableError`. (#656: the per-rung skip inside
    the ladder is the analogous "skip this rung, advance" — this helper
    keeps the lane-level fallback for the LAST rung's terminal raise.)
    """
    detail = (
        f"regional accelerator quota {headroom.metric} in {headroom.region} has "
        f"usage {headroom.usage:g}/{headroom.limit:g} — headroom "
        f"{headroom.available:g} GPU(s) < needed {headroom.needed}; skipping the "
        "GCP lane without burning a daily attempt"
    )
    attempts.append(
        RouteAttempt(
            kind="gcp",
            cluster=None,
            est_start_seconds_raw=0.0,
            est_start_seconds_clamped=0.0,
            outcome="quota_headroom_insufficient",
            detail=detail,
            elapsed_seconds=now_fn() - started_at,
        )
    )
    if not terminal:
        logger.warning(
            "route: GCP quota headroom insufficient for issue %d (%s); "
            "continuing down the lane order.",
            spec.issue,
            detail,
        )
        return None
    _post_terminal_failure_marker(
        spec=spec,
        marker_poster=marker_poster,
        reason=ROUTE_REASON_NO_COMPUTE,
        chosen_kind="gcp",
        attempts=attempts,
    )
    raise NoComputeAvailableError(
        f"every free lane park-failed AND the GCP regional quota has no headroom: {detail}",
        attempts=[_attempt_to_dict(a) for a in attempts],
    )


def _refuse_infeasible_cpu_footprint(
    *,
    spec: RunSpec,
    machine_gpu_count: int,
    attempts: list[RouteAttempt],
    marker_poster: Callable[..., None] | None,
    residual_gap: str,
) -> None:
    """Feasibility gate for the RunPod CPU fallback (#1010, incident #958).

    The mapped RunPod CPU instance has FIXED RAM and a provider
    container-disk cap (:data:`RUNPOD_CPU_INSTANCE_CAPS`, probe-verified); a
    plan-stated footprint (``spec.extra["boot_disk_gb"]`` /
    ``["min_ram_gb"]``) that exceeds them is deterministically infeasible —
    refuse BEFORE the paid provision cycle by raising the typed
    :class:`CpuFallbackInfeasibleError` (after posting the terminal marker
    with :data:`ROUTE_REASON_CPU_FALLBACK_INFEASIBLE`). No stated
    requirement (the common case) => both ints are 0 => no-op,
    byte-identical to pre-#1010 routing. Called ONLY from
    :func:`_runpod_terminal_rung`'s CPU branch, AFTER the #677 not-in-map
    guard — the single placement that covers all four automated GCP→RunPod
    paths (capacity fallback #656, sync workload failover #658, async
    workload failover #659, queue-timeout failover #783). A GPU intent
    (``machine_gpu_count > 0``) is a no-op.
    """
    if machine_gpu_count != 0:
        return
    instance_id = RUNPOD_CPU_INSTANCE_FOR_INTENT[spec.intent]
    caps = RUNPOD_CPU_INSTANCE_CAPS[instance_id]  # missing row -> loud KeyError
    extra = spec.extra or {}

    def _footprint_int(key: str) -> int:
        """spec.extra[key] as a non-negative int; 0 when absent/falsy.

        A malformed (non-integral) value raises a ValueError NAMING the
        key -- fail-loud with a diagnosable message, never a bare int()
        traceback deep in the rung.
        """
        raw = extra.get(key) or 0
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"spec.extra[{key!r}] is not an integer: {raw!r} "
                f"(malformed footprint requirement on issue {spec.issue})"
            ) from exc

    required_disk_gb = _footprint_int("boot_disk_gb")
    required_ram_gb = _footprint_int("min_ram_gb")
    shortfalls: list[str] = []
    if required_disk_gb > caps.max_container_disk_gb:
        shortfalls.append(
            f"disk: plan requires {required_disk_gb} GB > "
            f"{instance_id} max container disk {caps.max_container_disk_gb} GB"
        )
    if required_ram_gb > caps.ram_gb:
        shortfalls.append(
            f"RAM: plan requires {required_ram_gb} GB > {instance_id} fixed {caps.ram_gb} GB"
        )
    if shortfalls:
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_CPU_FALLBACK_INFEASIBLE,
            chosen_kind="gcp",  # precedent: the #677 guard in the rung
            attempts=attempts,
        )
        raise CpuFallbackInfeasibleError(
            f"CPU intent {spec.intent!r}: RunPod CPU fallback ({instance_id}) "
            f"cannot satisfy the plan footprint — {'; '.join(shortfalls)}. "
            f"Route to cpu-bigmem (or shrink the footprint). "
            f"residual_gap: {residual_gap}",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )


def _runpod_terminal_rung(
    *,
    spec: RunSpec,
    runpod_backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None,
    residual_gap: str,
    reason: str = ROUTE_REASON_RUNPOD_FALLBACK,
    failover_evidence: dict[str, Any] | None = None,
    gcp_failover_of_identity: dict[str, Any] | None = None,
) -> RouteResult:
    """Final fallback rung: launch on RunPod after every cheaper rung failed.

    The deliberate reversal of the historical no-auto-RunPod invariant
    (user-directed 2026-06-17, task #656). RunPod is reached ONLY here —
    after the cost-ordered GCP ladder (on-demand A100-80 → A100-40 → SPOT)
    AND, on the auto chain, the free SLURM lanes have all failed. The
    launch is the same shape as the explicit ``backend: runpod`` override
    (:func:`_override_runpod`): per-issue flock across launch + lease-write,
    the ``on_launched`` sidecar hook BEFORE any marker. The residual gap
    (which rungs ran dry) is logged LOUD and rides the marker ``extra`` so
    a future debug-er sees exactly what was exhausted before money was
    spent.

    ``reason`` labels the launched :class:`RouteResult` /
    ``epm:backend-selected`` marker. The DEFAULT
    (:data:`ROUTE_REASON_RUNPOD_FALLBACK`) is the capacity-exhaustion
    fallback (#656). The GCP-workload-failover caller (task #658) passes
    :data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD` +
    ``failover_evidence`` (the GcpWorkloadError evidence) so the marker
    trail tells "fell back because GCP ran dry" apart from "failed over
    because GCP crashed the workload"; the evidence rides the marker
    ``extra``.

    Fail-safe: if the RunPod launch ITSELF fails (no compute anywhere),
    re-raise as :class:`NoComputeAvailableError` with the full attempt
    trail — the terminal "truly no compute anywhere" outcome, preserving a
    typed terminal for the orchestrator's failure classifier.

    #940: a GCP-only GPU intent is translated to its RunPod-provisionable
    equivalent (:func:`_translated_runpod_intent`) before the launch, so the
    rung actually fires instead of dying in ``gpu_heuristics.resolve_intent``.
    """
    # CPU-intent guard (#677, RELAXED for mapped intents #747). RunPod's GPU
    # mutation (podFindAndDeployOnDemand) is GPU-only, BUT #747 adds a RunPod
    # CPU lane (deployCpuPod) for the cheap CPU intents in
    # RUNPOD_CPU_INSTANCE_FOR_INTENT. So the relaxed rule:
    #   * a CPU intent IN the map (cpu-small / cpu-mid) FALLS OVER to RunPod CPU
    #     when the GCP CPU lane is exhausted (capacity) OR its workload crashed
    #     (sync failover) -- it proceeds to the RunPod launch below, where the
    #     runpod_spec carries --intent <cpu-small|cpu-mid>, which
    #     pod_lifecycle.py/gpu_heuristics (Surface 5) resolves to the RunPod CPU
    #     instance_id via this SAME RUNPOD_CPU_INSTANCE_FOR_INTENT map;
    #   * a CPU intent NOT in the map (cpu-bigmem -- the >50 GB analysis lane,
    #     no cheap RunPod equivalent) STILL raises the TYPED
    #     CpuExhaustedNoRunpodLaneError verbatim (#677), so the orchestrator's
    #     classify_terminal_exception (issue_dispatch.py) posts an epm:failure
    #     note carrying reason: cpu_exhausted_no_runpod_lane (DISTINCT from the
    #     generic no_compute_available note). The watcher's capacity-retry pass
    #     keys on no_compute_available, so the distinct reason means it does NOT
    #     auto-retry a structurally-CPU-unservable cpu-bigmem RunPod launch --
    #     correct: no lane will ever free up to make RunPod accept it.
    machine = machine_for_intent(spec)
    if machine.gpu_count == 0 and spec.intent not in RUNPOD_CPU_INSTANCE_FOR_INTENT:
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD,
            chosen_kind="gcp",
            attempts=attempts,
        )
        raise CpuExhaustedNoRunpodLaneError(
            f"CPU intent {spec.intent!r}: GCP lane exhausted/failed and RunPod "
            f"has no CPU fallback lane for this intent. residual_gap: {residual_gap}",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )
    _refuse_infeasible_cpu_footprint(
        spec=spec,
        machine_gpu_count=machine.gpu_count,
        attempts=attempts,
        marker_poster=marker_poster,
        residual_gap=residual_gap,
    )
    # #940: translate a GCP-only GPU intent (capture-7b / lora / lora-7b-h100)
    # to its RunPod-provisionable equivalent BEFORE building runpod_spec —
    # pod_lifecycle's gpu_heuristics.resolve_intent KeyErrors on a GCP-only
    # intent (provision exit 1 -> NoComputeAvailableError), which is what
    # voided the sanctioned last rung on #841 despite live RunPod capacity.
    # An unmapped GCP GPU intent (eval-h100) fails loud HERE, pre-launch and
    # BEFORE the per-issue flock, naming the missing map row; the failure
    # reuses the existing runpod_fallback_failed terminal shape (same
    # classifier contract as a failed RunPod launch).
    try:
        runpod_intent, intent_translation = _translated_runpod_intent(spec)
    except ValueError as exc:
        attempts.append(
            RouteAttempt(
                kind="runpod",
                cluster=None,
                est_start_seconds_raw=0.0,
                est_start_seconds_clamped=0.0,
                outcome="runpod_fallback_failed",
                detail=f"runpod terminal fallback UNSERVABLE ({exc})",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_NO_COMPUTE,
            chosen_kind="runpod",
            attempts=attempts,
        )
        raise NoComputeAvailableError(
            "every GCP rung + free lane failed AND the RunPod terminal "
            f"fallback cannot serve this intent ({exc})",
            attempts=[_attempt_to_dict(a) for a in attempts],
        ) from exc
    if intent_translation is not None:
        logger.warning(
            "route: translating GCP-only intent %r -> RunPod intent %r for the "
            "terminal rung (issue %d).",
            spec.intent,
            runpod_intent,
            spec.issue,
        )
    if reason in (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    ):
        logger.warning(
            "route: GCP WORKLOAD failure for issue %d; failing over to RunPod "
            "(persistent SSH-able pod for diagnosis; residual gap: %s).",
            spec.issue,
            residual_gap,
        )
    elif reason == ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD:
        logger.warning(
            "route: GCP FLEX_START queue timeout for issue %d (instance stayed "
            "PENDING past EPS_GCP_QUEUE_WAIT_SECONDS); failing over to RunPod "
            "(residual gap: %s).",
            spec.issue,
            residual_gap,
        )
    else:
        logger.warning(
            "route: GCP ladder (and free lanes, if any) exhausted for issue %d; "
            "falling back to RunPod (residual gap: %s).",
            spec.issue,
            residual_gap,
        )
    runpod_spec = replace(
        spec,
        backend="runpod",
        # #940: the translated intent (identity for RunPod-native intents) so
        # everything downstream — launch argv, handle, lease, wedge
        # re-provision — is self-consistently RunPod-provisionable.
        intent=runpod_intent,
        extra={
            **(spec.extra or {}),
            "runpod_fallback_residual_gap": residual_gap,
            **({"runpod_intent_translation": intent_translation} if intent_translation else {}),
        },
    )
    with store.transaction(spec.issue) as (lease, write):
        # M3b (#669): in-flock re-check of the GCP->RunPod failover idempotency
        # record. The #669 wedge classifier introduces a SECOND concurrent
        # triggerer of this failover on the same GCP handle (a poller tick that
        # classified terminal_workload_wedged AND, separately, the
        # terminal_wedged_terminated watchdog path), so two overlapping
        # processes can both pass the OUTSIDE-the-flock pre-check in
        # backend_poll._failover_dead_gcp_to_runpod and both reach here. Repeat
        # the check INSIDE the flock (mirrors the reconnect-path re-check at the
        # _override_*/_auto lane transactions): a second triggerer that acquires
        # the lock AFTER the first stamped sees the stamp and short-circuits to
        # the EXISTING RunPod handle — no second paid launch. On the FIRST
        # triggerer (or any non-failover RunPod fallback, where
        # gcp_failover_of_identity is None) this is a no-op, so #659's
        # single-triggerer sequential behavior is byte-for-byte unchanged.
        if gcp_failover_of_identity is not None and _lease_already_failed_over(
            lease, gcp_failover_of_identity
        ):
            existing = _route_result_from_existing_failover_lease(
                lease=lease,
                spec=spec,
                runpod_backend=runpod_backend,
                reason=reason,
                attempts=attempts,
                started_at=started_at,
                now_fn=now_fn,
                residual_gap=residual_gap,
            )
            logger.warning(
                "route: GCP->RunPod failover for issue %d already launched by a "
                "concurrent triggerer (in-flock lease re-check, M3b #669); returning "
                "the existing RunPod handle %r, NO second launch.",
                spec.issue,
                existing.handle.pod_name,
            )
            return existing
        try:
            handle = _prepare_and_launch(runpod_backend, runpod_spec, kind="runpod")
        except Exception as exc:
            # Lazy import (module convention — matches the existing lazy
            # ``backends.runpod`` import below; runpod.py imports only ``base``
            # at module top, so no cycle either way — lazy is belt-and-braces).
            from explore_persona_space.backends.runpod import RunPodWorkloadStartError

            partial_handle = exc.handle if isinstance(exc, RunPodWorkloadStartError) else None
            if partial_handle is not None:
                # Pod provisioned + RUNNING; the workload did not start (#954).
                # Persist the SAME launch records the success path writes — the
                # sidecar hook and the in-flock lease (incl. the M3b
                # gcp_failover_of stamp) — so downstream stays chained and no
                # concurrent/later triggerer launches again. Then re-raise
                # TYPED: NoComputeAvailableError would be FALSE (a pod exists
                # and bills).
                _invoke_on_launched(on_launched, partial_handle)
                # LEASE-WRITE GUARD (#954 round-1 critique, alternatives MF2):
                # the typed error is the load-bearing signal — a lease-write
                # failure must NEVER replace it (the failover legs'
                # ``except RunPodWorkloadStartError`` would otherwise never
                # fire: no distinct terminal, no sidecar re-point, exactly when
                # rescue is needed). Same "never mask the original error"
                # invariant as the dispatch/relaunch sidecar writes. On a write
                # failure the failover legs' post-route sidecar write +
                # sentinel remain the (weaker) relaunch bound, and the
                # RouteAttempt detail records it for a human.
                lease_note = ""
                try:
                    new_lease = _lease_after_submit(
                        lease, runpod_spec, "runpod", None, partial_handle
                    )
                    if gcp_failover_of_identity is not None:
                        new_lease.gcp_failover_of = gcp_failover_of_identity
                    write(new_lease)
                except Exception as lease_exc:
                    logger.warning(
                        "route: runpod partial-launch lease write failed (%s: %s); "
                        "typed error preserved (issue %d).",
                        type(lease_exc).__name__,
                        lease_exc,
                        spec.issue,
                    )
                    lease_note = f"; lease_write_failed ({type(lease_exc).__name__}: {lease_exc})"
                attempts.append(
                    RouteAttempt(
                        kind="runpod",
                        cluster=None,
                        est_start_seconds_raw=0.0,
                        est_start_seconds_clamped=0.0,
                        outcome="runpod_workload_start_failed",
                        detail=(
                            f"runpod pod {partial_handle.pod_name} PROVISIONED but "
                            f"workload start FAILED ({exc}); pod left RUNNING for "
                            f"diagnosis{lease_note}"
                        ),
                        elapsed_seconds=now_fn() - started_at,
                    )
                )
                _post_terminal_failure_marker(
                    spec=spec,
                    marker_poster=marker_poster,
                    reason=ROUTE_REASON_RUNPOD_WORKLOAD_START_FAILED,
                    chosen_kind="runpod",
                    attempts=attempts,
                )
                raise
            # RunPod is the LAST resort — ANY OTHER failure here (prepare /
            # provisioning / transport, or a handle-less workload-start error
            # from the pre-provision guard) is genuinely "no compute anywhere".
            # Record it + surface the typed terminal so the orchestrator's
            # failure classifier still gets a NoComputeAvailableError.
            attempts.append(
                RouteAttempt(
                    kind="runpod",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="runpod_fallback_failed",
                    detail=f"runpod terminal fallback FAILED ({type(exc).__name__}: {exc})",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind="runpod",
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                "every GCP rung + free lane failed AND the RunPod terminal "
                f"fallback also failed ({type(exc).__name__}: {exc})",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        _invoke_on_launched(on_launched, handle)
        new_lease = _lease_after_submit(lease, runpod_spec, "runpod", None, handle)
        # M3b stamp (#669): record the GCP-crash identity this RunPod launch is
        # the failover OF, AS PART OF the in-flock write, so the NEXT concurrent
        # triggerer's in-flock re-check above sees it. This SUPERSEDES the
        # post-flock _stamp_lease_failover_of in backend_poll for the launch
        # path (that post-flock stamp stays as an idempotent belt-and-suspenders
        # no-op). None for a non-failover RunPod fallback (capacity exhaustion).
        if gcp_failover_of_identity is not None:
            new_lease.gcp_failover_of = gcp_failover_of_identity
        write(new_lease)
    is_workload_failover = reason in (
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
        ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    )
    attempts.append(
        RouteAttempt(
            kind="runpod",
            cluster=None,
            est_start_seconds_raw=0.0,
            est_start_seconds_clamped=0.0,
            outcome="launched",
            detail=(
                f"runpod {'workload-failover' if is_workload_failover else 'terminal fallback'} "
                f"(residual gap: {residual_gap})"
            ),
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=runpod_backend,
        handle=handle,
        requested_kind=spec.backend,
        chosen_kind="runpod",
        reason=reason,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        extra={
            "runpod_fallback_residual_gap": residual_gap,
            # #940: the GCP-only -> RunPod intent translation record, when one
            # applied, so the marker trail shows the intent swap (additive
            # extra key per the runpod_fallback_residual_gap precedent).
            **({"runpod_intent_translation": intent_translation} if intent_translation else {}),
            # The GcpWorkloadError evidence (task #658) so the failover
            # marker carries the original crash signal for diagnosis.
            **({"gcp_workload_evidence": failover_evidence} if failover_evidence else {}),
        },
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def failover_to_runpod_after_async_workload_crash(
    *,
    spec: RunSpec,
    runpod_backend: ComputeBackend,
    evidence: dict[str, Any] | None = None,
    residual_gap: str = "gcp async workload crash (poller-detected); failing over to RunPod",
    reason: str = ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC,
    marker_poster: Callable[..., None] | None = None,
    on_launched: Callable[[RunHandle], None] | None = None,
    lease_store: LeaseStore | None = None,
    now_fn: Callable[[], float] | None = None,
    config: RouterConfig | None = None,
    gcp_failover_of_identity: dict[str, Any] | None = None,
) -> RouteResult:
    """Re-dispatch a poller-detected dead / stuck GCP run onto RunPod, once (#659/#783).

    The ASYNC sibling of the synchronous ``_GcpWorkloadFailover`` path
    (:data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD`, #658): the GCP VM was
    already up and the workload crashed minutes in, so there is no live
    ``route()`` call to raise ``_GcpWorkloadFailover`` from. The poller
    (``scripts/backend_poll.py``) detects the dead GCP workload and calls this
    to launch the SAME :func:`_runpod_terminal_rung` the sync path uses,
    labeled with :data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC` so the
    marker trail tells the two detection paths apart.

    This is the ASYNC analogue of :func:`dispatch_for_issue` wrapping
    ``route()``: it builds the production-shaped versions of
    ``_runpod_terminal_rung``'s injected seams (a real :class:`LeaseStore`, a
    real :class:`RouterConfig`, ``time.monotonic`` for ``now_fn``, a fresh
    ``attempts`` list) and calls the launch primitive. Defaults match
    ``route()``'s own defaulting (``store = lease_store or LeaseStore()``,
    ``config or RouterConfig()``, ``now_fn or time.monotonic``); tests inject
    mocks for every seam, exactly like ``_runpod_terminal_rung``'s tests.

    ``evidence`` rides the ``epm:backend-selected`` marker ``extra`` as
    ``gcp_workload_evidence`` (mirroring the sync path's ``failover_evidence``)
    so the original crash signal is preserved for diagnosis.

    ``reason`` labels the launched :class:`RouteResult` /
    ``epm:backend-selected`` marker. The DEFAULT
    (:data:`ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC`) is the #659
    async workload-crash cause, so every existing #659 caller is byte-unchanged.
    The #783 queue-timeout poller passes
    :data:`ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD` so the marker trail
    tells a stuck FLEX_START queue apart from a crashed workload — same RunPod
    terminal rung, distinct cause.

    ``gcp_failover_of_identity`` (M3b, #669) is the crashed GCP run's stable
    identity (``{"pod_name": ..., "job_id": ...}``). When set, the terminal
    rung does an in-flock re-check + stamp so N CONCURRENT triggerers of the
    SAME GCP-crash failover (the #669 wedge classifier introduces a second one)
    launch RunPod exactly once. ``None`` preserves the legacy single-triggerer
    #659 behavior byte-for-byte (no in-flock short-circuit, no stamp).

    Fail-safe: if the RunPod launch ITSELF fails (no compute anywhere),
    ``_runpod_terminal_rung`` re-raises :class:`NoComputeAvailableError`, which
    propagates here unchanged — the poller maps it to a terminal infra JSON
    with ``reason: no_compute_available`` (re-drivable by the watcher's
    capacity-retry pass once a lane frees).
    """
    store = lease_store or LeaseStore()
    now_fn = now_fn or time.monotonic
    # ``config`` is accepted for API symmetry with ``route()`` (and so tests can
    # inject a fast RouterConfig); the RunPod terminal rung itself takes no
    # RouterConfig, so it is not threaded further. Touch it so a future reader
    # sees the deliberate non-use, not a forgotten wiring.
    _ = config
    # #909: no experimenter exists on the async failover paths — the RunPod
    # backend is the executor. Opt a custom-workload spec into the execution
    # leg (RunPodBackend.launch executes iff workload_cmd is set AND
    # extra["execute_workload"] is truthy); a hydra-args spec has no
    # backend-side executor on this lane (named residual) — log LOUD so the
    # gap is visible on the marker trail instead of a silent provision-only
    # relaunch (the #763 failure shape).
    if spec.workload_cmd and not (spec.extra or {}).get("execute_workload"):
        spec = replace(spec, extra={**dict(spec.extra or {}), "execute_workload": True})
    elif not spec.workload_cmd:
        logger.warning(
            "async RunPod failover of a hydra-args run for issue %s: the RunPod "
            "lane has no backend-side executor for hydra runs (named residual, #909)",
            spec.issue,
        )
    return _runpod_terminal_rung(
        spec=replace(spec, backend="runpod"),
        runpod_backend=runpod_backend,
        store=store,
        attempts=[],
        started_at=now_fn(),
        now_fn=now_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
        residual_gap=residual_gap,
        reason=reason,
        failover_evidence=evidence,
        # M3b (#669): the GCP-crash identity (pod_name/job_id) this failover is
        # OF, so the in-flock re-check + stamp can make N concurrent triggerers
        # launch RunPod exactly once. None on the legacy single-triggerer path
        # leaves the existing #659 behavior unchanged.
        gcp_failover_of_identity=gcp_failover_of_identity,
    )


def _override_free_or_gcp(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    kind: BackendKind,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
    started_evidence_probe: (
        Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None
    ) = None,
) -> RouteResult:
    """Explicit non-RunPod lane override.

    Reconnect first (idempotent re-entry), then launch + park. A free
    lane that times out / hard-fails RAISES (the user explicitly asked
    for that lane; we don't silently re-route).

    Lock discipline: the per-issue flock is held across reconnect-check
    → launch → lease-write so a concurrent invocation (manual /issue vs
    the issue-tick cron) cannot both decide "no live job, submit fresh"
    and double-submit. The lock spans the park watchdog too — wait IS
    contention surface, but it is per-ISSUE (not cross-issue), so the
    only callers serialized are the two we are deliberately serializing.
    """
    with store.transaction(spec.issue) as (lease, write):
        # Reconnect — inside the lock so a concurrent submit can't slip
        # between our "no live job" check and our launch. NO prepare()
        # on reconnect: SlurmBackend.prepare rsyncs the scratch dir with
        # --delete and would yank code out from under the RUNNING job.
        #
        # A PROBE failure (BackendProbeError — transport down, NOT "no
        # live job") must not fall through to a blind fresh submit: a
        # live job may exist and prepare()'s --delete rsync + a second
        # sbatch would corrupt / duplicate it (round-6 B1). Explicit
        # lane → typed terminal.
        try:
            handle = _try_reconnect(
                backend=backend, kind=kind, spec=spec, reconnect_fn=reconnect_fn
            )
        except BackendProbeError as exc:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="reconnect_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"explicit override {kind!r}: reconnect probe failed — cannot verify "
                f"whether a live job exists; refusing to submit blind ({exc})",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        if handle is not None:
            _invoke_on_launched(on_launched, handle)
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="reconnected",
                    detail="found existing live job/instance",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_RECONNECT,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
                # Explicit-override reconnect — `kind` may be ANY backend
                # here, so merge the gcp marker extras only when the
                # reconnected lane is gcp (#631 round-3 marker-coverage fix).
                extra=_gcp_marker_extras(spec) if kind == "gcp" else {},
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # Fresh submit (still under the flock).
        threaded_spec, lease = _thread_attempt_id_into(spec, lease, write)
        try:
            handle = _prepare_and_launch(backend, threaded_spec, kind=kind, cluster=spec.cluster)
        except BackendPrepareError as exc:
            # Explicit lane — prepare failed BEFORE launch (nothing
            # live). Provision-class typed terminal: breadcrumb, raise.
            # Breadcrumb reason matches the typed terminal's
            # ``reason: backend_prepare_failed`` (round-6 Mn1 — it
            # previously said ``no_compute_available`` while the
            # epm:failure note said ``backend_prepare_failed``).
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_PREPARE_FAILED,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise
        except GcpProvisioningError as exc:
            # Explicit GCP override — surface the provisioning failure (the
            # user asked for GCP, not a fallback chain). Post a terminal
            # breadcrumb so the dashboard sees the failure before we raise.
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="provisioning_failure",
                    detail=_provisioning_detail(exc),
                    elapsed_seconds=now_fn() - started_at,
                    evidence=_provisioning_evidence(exc),
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise
        except BackendProbeError as exc:
            # The backend's own pre-create state probe failed mid-launch
            # (e.g. GcpBackend.launch's internal reconnect_or_none with
            # expired gcloud auth — live auto-lane finding, issue 535:
            # this propagated UNCAUGHT to rc=4 instead of the typed
            # fail-closed terminal). State is UNKNOWN → refuse to act
            # blind; same contract as the reconnect-seam handler above.
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="probe_failed",
                    detail=str(exc)[:500],
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise NoComputeAvailableError(
                f"explicit override '{kind}': backend state probe failed mid-launch — "
                f"refusing to act blind: {exc}",
                attempts=[_attempt_to_dict(a) for a in attempts],
            ) from exc
        # Persist the handle (sidecar hook) + launched id IMMEDIATELY
        # (still inside the flock — crash-window-free). For "kind ==
        # gcp" override we leave the cluster field at None, matching
        # the existing schema.
        _invoke_on_launched(on_launched, handle)
        write(_lease_after_submit(lease, spec, kind, spec.cluster, handle))

        # GCP doesn't need the park (provision IS the start); just return.
        if kind == "gcp":
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="launched",
                    detail="gcp provision returned RUNNING-equivalent",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_OVERRIDE,
                cluster=None,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
                # Explicit `backend: gcp` fresh launch — only reached when
                # kind == "gcp" (guarded above), so the gcp marker extras
                # always apply (#631 round-3 marker-coverage fix).
                extra=_gcp_marker_extras(spec),
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # SLURM-style free lane: run the park watchdog (under the flock).
        started, reason, terminal_status = park_until_running_or_cap(
            backend=backend,
            handle=handle,
            is_started=is_started,
            cap_seconds=cfg.free_wait_seconds,
            poll_interval=cfg.poll_interval,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if started:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="launched",
                    detail="park resolved to RUNNING",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_OVERRIDE,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # Park failed. Distinguish "never started" from "started and
        # FAILED": a fast-failing job transitions PD→R→exit between
        # polls, so "vanished before observed RUNNING" is NOT proof the
        # cluster lacked capacity. If the scratch dir holds runtime
        # artifacts (status.json / job.out), the job DID start — that is
        # a WORKLOAD failure (surface, no fallback), not no-compute.
        #
        # GATED on the job being genuinely GONE (done/dead): ``stalled``
        # covers LIVE jobs (RUNNING + stale heartbeat; SUSPENDED) and
        # ``gate`` is a live wait — classifying those here raised BEFORE
        # the cancel machine and orphaned a live job (round-6 M1, issue
        # 535 attempt 2). stalled/gate fall through to cancel_and_wait.
        if reason == "terminal_before_running" and terminal_status in ("done", "dead"):
            evidence = _probe_started_evidence(started_evidence_probe, backend, handle)
            if evidence is not None:
                attempts.append(
                    RouteAttempt(
                        kind=kind,
                        cluster=spec.cluster,
                        est_start_seconds_raw=None,
                        est_start_seconds_clamped=None,
                        outcome="workload_failure",
                        detail=(
                            "terminal before RUNNING with runtime artifacts "
                            f"(phase={evidence.get('phase', '')!r})"
                        ),
                        elapsed_seconds=now_fn() - started_at,
                    )
                )
                _post_terminal_failure_marker(
                    spec=spec,
                    marker_poster=marker_poster,
                    reason=ROUTE_REASON_WORKLOAD_FAILURE,
                    chosen_kind=kind,
                    attempts=attempts,
                    extra={"evidence": evidence},
                )
                raise WorkloadSurfacedError(
                    f"{kind} job {handle.job_id} went terminal before RUNNING but "
                    f"left runtime artifacts (phase={evidence.get('phase', '')!r}) — "
                    "workload failure, no auto-fallback",
                    chosen_kind=kind,
                    evidence=evidence,
                )

        # The user explicitly asked for this lane → cancel state
        # machine, then either KEEP (raced) or surface terminal.
        cancel_outcome = cancel_and_wait(
            backend=backend,
            handle=handle,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            grace_seconds=cfg.cancel_grace_seconds,
            poll_interval=min(2.0, cfg.poll_interval),
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        # Special case: cancel-race kept the job (raced to RUNNING). Return
        # it as the chosen outcome — we didn't actually cancel, the job won.
        if cancel_outcome == "raced_to_running":
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=None,
                    est_start_seconds_clamped=None,
                    outcome="launched",
                    detail="cancel-race; job started during scancel",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=kind,
                chosen_kind=kind,
                reason=ROUTE_REASON_OVERRIDE,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
                extra={"cancel_race": True},
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        attempts.append(
            RouteAttempt(
                kind=kind,
                cluster=spec.cluster,
                est_start_seconds_raw=None,
                est_start_seconds_clamped=None,
                outcome=reason,
                detail=f"cancel_outcome={cancel_outcome}",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        # On manual_attention the cancel did NOT confirm the job is dead.
        # We CANNOT silently escalate (would double-spend / collide on
        # attempt-id namespace) and the user explicitly asked for THIS
        # lane anyway — raise ManualAttentionRequiredError so the
        # orchestrator surfaces the orphaned job id.
        if cancel_outcome == "manual_attention":
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
            )
            raise ManualAttentionRequiredError(
                kind=kind,
                cluster=spec.cluster,
                orphaned_job_id=str(handle.job_id),
                attempts=[_attempt_to_dict(a) for a in attempts],
            )
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_NO_COMPUTE,
            chosen_kind=kind,
            attempts=attempts,
        )
        raise NoComputeAvailableError(
            f"explicit override {kind!r} did not start within {cfg.free_wait_seconds}s "
            f"(park: {reason}, cancel: {cancel_outcome})",
            attempts=[_attempt_to_dict(a) for a in attempts],
        )


def _override_gcp_with_ladder(
    *,
    spec: RunSpec,
    gcp_backend: ComputeBackend,
    runpod_backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None,
) -> RouteResult:
    """Explicit ``backend: gcp`` override — reconnect, then walk the SAME ladder.

    This is the #654 fix: pre-#656 an explicit-gcp ``GcpProvisioningError``
    RAISED with no fallback (the hard-block #654 reported). Now the explicit
    pin gets the IDENTICAL cost-ordered ladder as the auto lane
    (:func:`_attempt_gcp_lane`, acceptance criterion 3), and on full ladder
    exhaustion falls through to the RunPod terminal rung
    (:func:`_runpod_terminal_rung`) — never a hard block.

    Reconnect-first (idempotent re-entry) is preserved: a live
    ``eps-issue-<N>`` instance is reused rather than re-provisioned. A
    reconnect PROBE failure (transport down, NOT "no live job") refuses a
    blind fresh create and surfaces a typed :class:`NoComputeAvailableError`
    — a live instance may exist and a second create would double-spend.
    """
    # Reconnect-first (read-only probe; no flock needed — the ladder's
    # per-rung transaction re-checks under the flock).
    try:
        handle = _try_reconnect(
            backend=gcp_backend, kind="gcp", spec=spec, reconnect_fn=reconnect_fn
        )
    except BackendProbeError as exc:
        attempts.append(
            RouteAttempt(
                kind="gcp",
                cluster=None,
                est_start_seconds_raw=None,
                est_start_seconds_clamped=None,
                outcome="reconnect_probe_failed",
                detail=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        _post_terminal_failure_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_NO_COMPUTE,
            chosen_kind="gcp",
            attempts=attempts,
        )
        raise NoComputeAvailableError(
            "explicit override 'gcp': reconnect probe failed — cannot verify whether a "
            f"live instance exists; refusing to provision blind ({exc})",
            attempts=[_attempt_to_dict(a) for a in attempts],
        ) from exc
    if handle is not None:
        _invoke_on_launched(on_launched, handle)
        attempts.append(
            RouteAttempt(
                kind="gcp",
                cluster=None,
                est_start_seconds_raw=None,
                est_start_seconds_clamped=None,
                outcome="reconnected",
                detail="found existing live gcp instance",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        result = RouteResult(
            backend=gcp_backend,
            handle=handle,
            requested_kind="gcp",
            chosen_kind="gcp",
            reason=ROUTE_REASON_RECONNECT,
            cluster=None,
            attempts=attempts,
            elapsed_seconds=now_fn() - started_at,
            extra=_gcp_marker_extras(spec),
        )
        _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
        return result

    # No live instance → walk the GCP ladder. ``terminal=True`` only labels
    # the launched attempt detail; the ladder no longer raises on
    # exhaustion (it returns None) — the RunPod terminal rung is the tail.
    try:
        ladder_result = _attempt_gcp_lane(
            spec=spec,
            gcp_backend=gcp_backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            terminal=True,
            # Explicit pin → label the launched result as an OVERRIDE (not a
            # router auto-fallback) so the marker trail distinguishes the two.
            reason=ROUTE_REASON_OVERRIDE,
            requested_kind="gcp",
            # Explicit user ask attempts regardless of the auto-escalation cap
            # (the pre-#656 explicit-gcp path never touched the per-day counter).
            count_attempt_cap=False,
        )
    except _GcpWorkloadFailover as failover:
        # A GCP rung failed the WORKLOAD on an explicit ``backend: gcp``
        # pin (task #658): a GCP failure of ANY class fails over to RunPod,
        # so the explicit-gcp path mirrors the auto path. RunPod's
        # persistent SSH-able pod is the diagnosis surface GCP's deleted
        # boot disk cannot give.
        return _runpod_terminal_rung(
            spec=spec,
            runpod_backend=runpod_backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            residual_gap=failover.residual_gap,
            reason=ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
            failover_evidence=failover.evidence,
        )
    if ladder_result is not None:
        return ladder_result

    # Ladder exhausted → RunPod terminal rung (the #654 hard-block fix).
    return _runpod_terminal_rung(
        spec=spec,
        runpod_backend=runpod_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        now_fn=now_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
        residual_gap=(
            "explicit gcp pin: on-demand A100-80/A100-40 + spot rungs exhausted "
            "(no free SLURM lane on an explicit gcp pin)"
        ),
    )


# ---------------------------------------------------------------------------
# Auto routing path
# ---------------------------------------------------------------------------


def _auto_route(
    *,
    spec: RunSpec,
    free_backends: dict[BackendKind, ComputeBackend],
    gcp_backend: ComputeBackend | None,
    runpod_backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    lane_order: tuple[BackendKind, ...],
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    started_evidence_probe: (Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None),
    mila_socket_alive: Callable[[], bool] | None,
    estimate_fn: Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None,
    clock_fn: Callable[[], datetime] | None,
) -> RouteResult:
    """No-``backend:`` auto route: walk ``lane_order`` (GCP-first default).

    GCP is a first-class auto lane, not only an escalation target: at
    its position in the order it walks the cost-ordered fallback ladder
    (:func:`_attempt_gcp_lane` → on-demand A100-80 → A100-40 → SPOT).
    Contiguous SLURM lanes keep the existing est-start ranking + park +
    cancel chain among themselves. When EVERY lane is exhausted, the chain
    falls to the RunPod terminal rung (:func:`_runpod_terminal_rung`) — the
    deliberate reversal of the no-auto-RunPod invariant (#656): RunPod is
    reached ONLY here, after every cheaper GCP rung + free SLURM lane has
    failed. Only if the RunPod launch ITSELF fails does the chain raise
    :class:`NoComputeAvailableError` ("truly no compute anywhere").
    """
    del clock_fn  # reserved for a future "day boundary at posted-time" override
    # Build the candidate list in lane order (skipping unwired lanes +
    # Mila-when-down + GCP-when-unwired).
    candidates: list[tuple[ComputeBackend, BackendKind]] = []
    for kind in lane_order:
        if kind == "gcp":
            if gcp_backend is not None:
                candidates.append((gcp_backend, "gcp"))
            continue
        backend = free_backends.get(kind)
        if backend is None:
            continue
        if kind == "mila" and (mila_socket_alive is None or not mila_socket_alive()):
            continue
        candidates.append((backend, kind))

    # Stage 1: reconnect scan over every wired lane, in lane order.
    reconnect_result = _try_auto_reconnect(
        spec=spec,
        candidates=candidates,
        store=store,
        attempts=attempts,
        started_at=started_at,
        reconnect_fn=reconnect_fn,
        now_fn=now_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
    )
    if reconnect_result is not None:
        return reconnect_result

    # Stage 2: walk the chain group by group. A GCP group is a single
    # provision attempt; a SLURM group is the ranked launch → park →
    # cancel-on-fail chain. ``terminal`` (last group) preserves the
    # legacy escalation semantics: when GCP sits LAST, its failures
    # raise the historical typed terminals instead of falling through.
    groups = _split_lane_groups([kind for _backend, kind in candidates])
    for group_idx, group in enumerate(groups):
        terminal = group_idx == len(groups) - 1
        if group == ("gcp",):
            try:
                gcp_result = _attempt_gcp_lane(
                    spec=spec,
                    gcp_backend=gcp_backend,
                    store=store,
                    attempts=attempts,
                    started_at=started_at,
                    cfg=cfg,
                    now_fn=now_fn,
                    marker_poster=marker_poster,
                    on_launched=on_launched,
                    terminal=terminal,
                )
            except _GcpWorkloadFailover as failover:
                # A GCP rung failed the WORKLOAD (task #658): fail over
                # STRAIGHT to RunPod — do NOT continue walking the remaining
                # SLURM lanes (re-crashing broken code there burns queue
                # time). RunPod's persistent SSH-able pod is the diagnosis
                # surface GCP's deleted boot disk cannot give.
                return _runpod_terminal_rung(
                    spec=spec,
                    runpod_backend=runpod_backend,
                    store=store,
                    attempts=attempts,
                    started_at=started_at,
                    now_fn=now_fn,
                    marker_poster=marker_poster,
                    on_launched=on_launched,
                    residual_gap=failover.residual_gap,
                    reason=ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD,
                    failover_evidence=failover.evidence,
                )
            if gcp_result is not None:
                return gcp_result
            continue
        slurm_candidates = [(b, k) for b, k in candidates if k in group]
        estimated = _estimate_lanes(slurm_candidates, spec=spec, estimate_fn=estimate_fn)
        ranked = rank_lanes(estimated)
        free_result = _try_free_lanes(
            spec=spec,
            ranked=ranked,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            started_evidence_probe=started_evidence_probe,
            reconnect_fn=reconnect_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
        )
        if free_result is not None:
            return free_result

    # Terminal: every cheaper auto lane (GCP ladder + free SLURM lanes)
    # failed or was unwired / unavailable. Fall to the RunPod terminal rung
    # (#656, the reversed no-auto-RunPod invariant): RunPod is reached ONLY
    # here, after the cost-ordered GCP ladder AND the free lanes are all
    # exhausted. The residual gap names the exhausted lanes loudly in the
    # marker. Only if the RunPod launch ITSELF fails does this raise the
    # typed NoComputeAvailableError ("truly no compute anywhere").
    wired = [kind for _b, kind in candidates]
    residual_gap = (
        f"auto chain exhausted (order: {' -> '.join(lane_order)}; wired: "
        f"{wired or 'none'}) — GCP ladder + free SLURM lanes all failed"
    )
    return _runpod_terminal_rung(
        spec=spec,
        runpod_backend=runpod_backend,
        store=store,
        attempts=attempts,
        started_at=started_at,
        now_fn=now_fn,
        marker_poster=marker_poster,
        on_launched=on_launched,
        residual_gap=residual_gap,
    )


def _try_auto_reconnect(
    *,
    spec: RunSpec,
    candidates: list[tuple[ComputeBackend, BackendKind]],
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult | None:
    """Auto-route stage 1: look for an existing live job on every wired lane.

    ``candidates`` arrives in the RESOLVED lane order (GCP included at
    its position when wired), so the scan order matches the attempt
    order the marker trail reports.

    Reconnect probes are READ-ONLY (no lease writes) so they DON'T need
    to hold the per-issue flock — the flock is acquired by the lane that
    actually decides to submit, and that submit-path repeats the
    reconnect check inside the flock (so a job that appeared between
    this scan and the eventual launch is still caught).

    NO ``prepare()`` on any reconnect outcome here: ``SlurmBackend.
    prepare`` rsyncs the scratch dir with ``--delete`` and would yank
    code out from under the RUNNING job it just reconnected to.
    """
    del store  # not needed for reconnect probes; the launch path re-checks under the flock
    for backend, kind in candidates:
        # A probe failure here only skips the lock-free SCAN — the
        # submit path re-checks reconnect INSIDE the flock and a probe
        # failure THERE skips the lane (no blind submit), so swallowing
        # at this stage cannot cause a duplicate. (For GCP the launch
        # itself re-probes via reconnect_or_none.)
        lane_spec = _spec_for_lane(spec, kind)
        try:
            handle = _try_reconnect(
                backend=backend, kind=kind, spec=lane_spec, reconnect_fn=reconnect_fn
            )
        except BackendProbeError as exc:
            logger.warning(
                "route: reconnect scan probe failed for %s (%s); deferring to the "
                "in-flock re-check on the submit path.",
                kind,
                exc,
            )
            continue
        if handle is None:
            continue
        return _record_reconnect(
            backend=backend,
            kind=kind,
            cluster=lane_spec.cluster,
            handle=handle,
            spec=lane_spec,
            attempts=attempts,
            started_at=started_at,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            detail=(
                "found existing live gcp instance"
                if kind == "gcp"
                else "found existing live job/instance"
            ),
        )
    return None


def _record_reconnect(
    *,
    backend: ComputeBackend,
    kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
    spec: RunSpec,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    detail: str,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult:
    """Append a reconnect attempt + build the matching RouteResult.

    The persistence hook fires BEFORE the marker post so the handle is
    on disk by the time any observability side effect runs.
    """
    _invoke_on_launched(on_launched, handle)
    attempts.append(
        RouteAttempt(
            kind=kind,
            cluster=cluster,
            est_start_seconds_raw=None,
            est_start_seconds_clamped=None,
            outcome="reconnected",
            detail=detail,
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=backend,
        handle=handle,
        requested_kind=None,
        chosen_kind=kind,
        reason=ROUTE_REASON_RECONNECT,
        cluster=cluster,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        # Auto-chain reconnect scan — `kind` is the reconnected lane (gcp
        # OR a free SLURM lane), so merge the gcp marker extras only for a
        # gcp reconnect (#631 round-3 marker-coverage fix).
        extra=_gcp_marker_extras(spec) if kind == "gcp" else {},
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def _try_free_lanes(
    *,
    spec: RunSpec,
    ranked: list[tuple[ComputeBackend, BackendKind, float | None, float]],
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    started_evidence_probe: (Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None),
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult | None:
    """Auto-route stage 2: launch + park each ranked free lane, in order.

    Returns the first lane that resolves to RUNNING (or wins a
    cancel-race after park-fail). Returns ``None`` when EVERY lane in
    ``ranked`` fails to start — caller escalates to GCP.
    """
    for backend, kind, est_raw, est_clamped in ranked:
        result = _try_one_free_lane(
            spec=spec,
            backend=backend,
            kind=kind,
            est_raw=est_raw,
            est_clamped=est_clamped,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            is_started=is_started,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            started_evidence_probe=started_evidence_probe,
            reconnect_fn=reconnect_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if result is not None:
            return result
    return None


def _try_one_free_lane(
    *,
    spec: RunSpec,
    backend: ComputeBackend,
    kind: BackendKind,
    est_raw: float | None,
    est_clamped: float | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    is_started: Callable[[ComputeBackend, RunHandle], bool],
    is_live_after_cancel: Callable[[ComputeBackend, RunHandle], bool],
    is_running_after_cancel: Callable[[ComputeBackend, RunHandle], bool] | None,
    started_evidence_probe: (Callable[[ComputeBackend, RunHandle], dict[str, Any] | None] | None),
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
    marker_poster: Callable[..., None] | None,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    on_launched: Callable[[RunHandle], None] | None = None,
) -> RouteResult | None:
    """Launch + park one free lane. Returns a RouteResult on success / cancel-race.

    Returns ``None`` to signal "next lane". Cancel-race during park-fail
    is treated as success (the job won; tearing it down would forfeit
    the wait we already paid for).

    Lock discipline: the per-issue flock is held across (re-check
    reconnect → launch → lease-write → park → cancel) so a concurrent
    invocation cannot slip a parallel submit between our reconnect probe
    and our launch. If the cancel state machine returns ``manual_attention``
    (cancel did NOT confirm the job is dead), we RAISE
    :class:`ManualAttentionRequiredError` rather than returning ``None``
    — silently escalating would risk a second copy of the same workload
    in the GCP escalation path (the orphaned free-lane job is unconfirmed
    dead and may still consume the attempt-id namespace).
    """
    spec = _spec_for_lane(spec, kind)
    with store.transaction(spec.issue) as (lease, write):
        # Repeat the reconnect check INSIDE the flock — a parallel
        # invocation may have submitted between the lock-free scan in
        # _try_auto_reconnect and now. NO prepare() on reconnect:
        # SlurmBackend.prepare rsyncs the scratch dir with --delete and
        # would yank code out from under the RUNNING job.
        #
        # A PROBE failure (BackendProbeError — transport down, NOT "no
        # live job") must not fall through to a blind fresh submit on
        # THIS lane (a live job may exist; prepare()'s --delete rsync +
        # a second sbatch would corrupt / duplicate it — round-6 B1).
        # Auto chain → skip the lane, try the next one.
        try:
            handle = _try_reconnect(
                backend=backend, kind=kind, spec=spec, reconnect_fn=reconnect_fn
            )
        except BackendProbeError as exc:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="reconnect_probe_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: free lane %s reconnect probe failed (%s); skipping lane "
                "(cannot verify whether a live job exists — submitting blind risks "
                "a duplicate).",
                kind,
                exc,
            )
            return None
        if handle is not None:
            _invoke_on_launched(on_launched, handle)
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="reconnected",
                    detail="reconnect inside flock — concurrent invocation submitted",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            result = RouteResult(
                backend=backend,
                handle=handle,
                requested_kind=None,
                chosen_kind=kind,
                reason=ROUTE_REASON_RECONNECT,
                cluster=spec.cluster,
                attempts=attempts,
                elapsed_seconds=now_fn() - started_at,
            )
            _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
            return result

        # Launch (still under the flock — sealing the double-submit race).
        threaded_spec, lease = _thread_attempt_id_into(spec, lease, write)
        try:
            handle = _prepare_and_launch(backend, threaded_spec, kind=kind, cluster=spec.cluster)
        except BackendPrepareError as exc:
            # Provision-class, pre-launch (nothing live) → next lane,
            # same semantics as a launch failure but with a precise
            # attempt-trail outcome.
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="prepare_failed",
                    detail=exc.reason,
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: free lane %s prepare failed (%s); trying next lane.",
                kind,
                exc.reason,
            )
            return None
        except Exception as exc:
            attempts.append(
                RouteAttempt(
                    kind=kind,
                    cluster=spec.cluster,
                    est_start_seconds_raw=est_raw,
                    est_start_seconds_clamped=est_clamped,
                    outcome="launch_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: free lane %s launch failed (%s); trying next lane.",
                kind,
                type(exc).__name__,
            )
            return None

        # Persist the handle (sidecar hook) + launched id IMMEDIATELY
        # (still under the flock).
        _invoke_on_launched(on_launched, handle)
        write(_lease_after_submit(lease, spec, kind, spec.cluster, handle))

        # Park (still under the flock — wait IS contention surface, but
        # the lock is per-ISSUE, not cross-issue, so the only callers
        # serialized are the two we are deliberately serializing).
        started, reason, terminal_status = park_until_running_or_cap(
            backend=backend,
            handle=handle,
            is_started=is_started,
            cap_seconds=cfg.free_wait_seconds,
            poll_interval=cfg.poll_interval,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if started:
            return _record_free_lane_started(
                backend=backend,
                handle=handle,
                kind=kind,
                est_raw=est_raw,
                est_clamped=est_clamped,
                spec=spec,
                attempts=attempts,
                started_at=started_at,
                now_fn=now_fn,
                marker_poster=marker_poster,
                detail="park resolved to RUNNING",
            )

        # Park failed. Distinguish "never started" from "started and
        # FAILED": a fast-failing job transitions PD→R→exit between
        # polls, so "vanished before observed RUNNING" is NOT proof the
        # cluster lacked capacity. If the scratch dir holds runtime
        # artifacts (status.json / job.out), the job DID start — a
        # WORKLOAD failure that must SURFACE (no GCP escalation: a
        # workload bug would burn paid credit on a doomed re-run).
        #
        # GATED on the job being genuinely GONE (done/dead): ``stalled``
        # covers LIVE jobs (RUNNING + stale heartbeat; SUSPENDED) and
        # ``gate`` is a live wait — classifying those here raised BEFORE
        # the cancel machine and orphaned a live job (round-6 M1, issue
        # 535 attempt 2). stalled/gate fall through to cancel_and_wait.
        if reason == "terminal_before_running" and terminal_status in ("done", "dead"):
            evidence = _probe_started_evidence(started_evidence_probe, backend, handle)
            if evidence is not None:
                attempts.append(
                    RouteAttempt(
                        kind=kind,
                        cluster=spec.cluster,
                        est_start_seconds_raw=est_raw,
                        est_start_seconds_clamped=est_clamped,
                        outcome="workload_failure",
                        detail=(
                            "terminal before RUNNING with runtime artifacts "
                            f"(phase={evidence.get('phase', '')!r})"
                        ),
                        elapsed_seconds=now_fn() - started_at,
                    )
                )
                _post_terminal_failure_marker(
                    spec=spec,
                    marker_poster=marker_poster,
                    reason=ROUTE_REASON_WORKLOAD_FAILURE,
                    chosen_kind=kind,
                    attempts=attempts,
                    extra={"evidence": evidence},
                )
                raise WorkloadSurfacedError(
                    f"{kind} job {handle.job_id} went terminal before RUNNING but "
                    f"left runtime artifacts (phase={evidence.get('phase', '')!r}) — "
                    "workload failure, no auto-fallback",
                    chosen_kind=kind,
                    evidence=evidence,
                )

        # Genuine never-started park failure → cancel state machine,
        # then KEEP (raced), CONTINUE to next lane (cancelled), or
        # RAISE (manual_attention).
        cancel_outcome = cancel_and_wait(
            backend=backend,
            handle=handle,
            is_live_after_cancel=is_live_after_cancel,
            is_running_after_cancel=is_running_after_cancel,
            grace_seconds=cfg.cancel_grace_seconds,
            poll_interval=min(2.0, cfg.poll_interval),
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if cancel_outcome == "raced_to_running":
            return _record_free_lane_started(
                backend=backend,
                handle=handle,
                kind=kind,
                est_raw=est_raw,
                est_clamped=est_clamped,
                spec=spec,
                attempts=attempts,
                started_at=started_at,
                now_fn=now_fn,
                marker_poster=marker_poster,
                detail="cancel-race; job started during scancel",
                extra={"cancel_race": True},
            )

        attempts.append(
            RouteAttempt(
                kind=kind,
                cluster=spec.cluster,
                est_start_seconds_raw=est_raw,
                est_start_seconds_clamped=est_clamped,
                outcome=reason,
                detail=f"cancel_outcome={cancel_outcome}",
                elapsed_seconds=now_fn() - started_at,
            )
        )
        if cancel_outcome == "manual_attention":
            # cancel grace expired without confirming the free-lane job
            # is dead. Silently escalating to GCP would risk a duplicate
            # run sharing the attempt-id namespace → raise so the
            # orchestrator surfaces the orphaned id + parks.
            _post_terminal_failure_marker(
                spec=spec,
                marker_poster=marker_poster,
                reason=ROUTE_REASON_NO_COMPUTE,
                chosen_kind=kind,
                attempts=attempts,
                extra={"manual_attention": True, "orphaned_job_id": str(handle.job_id)},
            )
            raise ManualAttentionRequiredError(
                kind=kind,
                cluster=spec.cluster,
                orphaned_job_id=str(handle.job_id),
                attempts=[_attempt_to_dict(a) for a in attempts],
            )
        return None


def _record_free_lane_started(
    *,
    backend: ComputeBackend,
    handle: RunHandle,
    kind: BackendKind,
    est_raw: float | None,
    est_clamped: float | None,
    spec: RunSpec,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> RouteResult:
    """Append a "launched" attempt + build the matching auto-started RouteResult."""
    attempts.append(
        RouteAttempt(
            kind=kind,
            cluster=spec.cluster,
            est_start_seconds_raw=est_raw,
            est_start_seconds_clamped=est_clamped,
            outcome="launched",
            detail=detail,
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=backend,
        handle=handle,
        requested_kind=None,
        chosen_kind=kind,
        reason=ROUTE_REASON_AUTO_STARTED,
        cluster=spec.cluster,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        extra=extra or {},
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


def _attempt_gcp_lane(
    *,
    spec: RunSpec,
    gcp_backend: ComputeBackend | None,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None = None,
    terminal: bool = True,
    reason: str = ROUTE_REASON_AUTO_FALLBACK_GCP,
    requested_kind: BackendKind | None = None,
    count_attempt_cap: bool = True,
) -> RouteResult | None:
    """Attempt the GCP lane by WALKING the length-aware fallback ladder (#656/#680).

    ``reason`` / ``requested_kind`` label the launched :class:`RouteResult`:
    the auto chain uses ``auto_fallback_gcp`` / ``None`` (the default); the
    explicit ``backend: gcp`` override path passes ``override`` / ``"gcp"`` so
    the marker trail tells a router-fallback launch apart from a user pin.

    ``count_attempt_cap`` (default ``True``): the auto chain counts each rung's
    create against the per-day attempt cap. The explicit ``backend: gcp`` pin
    passes ``False`` — an explicit user ask attempts regardless of the
    auto-escalation cap (matching the pre-#656 explicit-gcp behavior, which
    never touched the counter); it still ladders + falls to the RunPod rung
    on exhaustion.

    The ladder is keyed on job LENGTH (#680) — short jobs lead with SPOT,
    long / unknown-length jobs lead with FLEX_START and bar SPOT; CPU-only
    intents short-circuit to a single on-demand rung. See
    :func:`_gcp_ladder_specs` for the full per-length rung order. Each rung runs
    the SAME pre-create headroom check + launch that the single GCP attempt
    used to run; a rung that fails on a POSITIVE insufficient-headroom
    reading OR a :class:`GcpProvisioningError` capacity/zone miss records
    its attempt and ADVANCES to the next rung. A :class:`GcpWorkloadError`
    on ANY rung STOPS the ladder and raises the internal
    :class:`_GcpWorkloadFailover` signal, which the lane callers translate
    into a RunPod terminal-rung launch (task #658: a GCP workload failure
    now fails over to RunPod instead of surfacing — RunPod pods persist +
    are SSH-able for diagnosis, where GCP DELETEs its boot disk on crash).
    The failover does NOT cascade across the remaining GCP rungs or the
    free SLURM lanes (re-crashing broken code there burns queue time).

    ``terminal`` no longer changes the failure DISPOSITION — in BOTH
    positions an exhausted ladder returns ``None`` so the caller falls
    through (auto chain → next SLURM lane, then the RunPod terminal rung;
    explicit ``backend: gcp`` → the RunPod terminal rung directly). The
    historical ``terminal``-only ``NoComputeAvailableError`` /
    ``GcpAttemptCapExceededError`` raises are GONE: the RunPod terminal
    rung is the new tail of the chain (the reversed no-auto-RunPod
    invariant). ``terminal`` is retained only to label the launched
    attempt detail (escalation vs primary-lane).

    Per-day attempt cap: the cap is RE-READ at the top of each rung
    iteration and counts ACTUAL create attempts across all rungs (a
    headroom-skip does NOT consume one, matching today). When the cap is
    hit mid-ladder, the ladder STOPS issuing creates and returns ``None``
    (fall through) — it never raises and never bricks the route. Stays an
    attempt-COUNT guard, never a dollar cap
    (``tests/test_no_dollar_budget_caps.py``).

    Lock discipline preserved per rung: each rung's cap-check / bump /
    threaded-attempt-id / launch / persist live inside ONE
    :meth:`LeaseStore.transaction`.
    """
    if gcp_backend is None:
        # Only reachable from the legacy terminal call shape — the auto
        # chain filters an unwired GCP out of the candidates, so this is
        # belt-and-suspenders for direct callers. Return None so the caller
        # falls through (to SLURM / the RunPod terminal rung).
        return None

    rungs = _gcp_ladder_specs(spec)
    for rung_spec, rung_label in rungs:
        outcome = _attempt_one_gcp_rung(
            spec=rung_spec,
            rung_label=rung_label,
            gcp_backend=gcp_backend,
            store=store,
            attempts=attempts,
            started_at=started_at,
            cfg=cfg,
            now_fn=now_fn,
            marker_poster=marker_poster,
            on_launched=on_launched,
            terminal=terminal,
            reason=reason,
            requested_kind=requested_kind,
            count_attempt_cap=count_attempt_cap,
        )
        if isinstance(outcome, RouteResult):
            return outcome  # this rung launched
        if outcome == "cap_hit":
            # Per-day attempt cap reached mid-ladder → stop issuing creates,
            # fall through (the caller routes to the next lane / RunPod).
            return None
        # outcome == "advance" → try the next rung.
    # Every applicable rung failed (capacity / headroom). Fall through so
    # the caller routes to the next SLURM lane or the RunPod terminal rung.
    return None


def _attempt_one_gcp_rung(
    *,
    spec: RunSpec,
    rung_label: str,
    gcp_backend: ComputeBackend,
    store: LeaseStore,
    attempts: list[RouteAttempt],
    started_at: float,
    cfg: RouterConfig,
    now_fn: Callable[[], float],
    marker_poster: Callable[..., None] | None,
    on_launched: Callable[[RunHandle], None] | None,
    terminal: bool,
    reason: str = ROUTE_REASON_AUTO_FALLBACK_GCP,
    requested_kind: BackendKind | None = None,
    count_attempt_cap: bool = True,
) -> RouteResult | str:
    """Attempt ONE GCP ladder rung. Returns a launched :class:`RouteResult`,
    or ``"advance"`` (this rung failed/skipped — try the next rung), or
    ``"cap_hit"`` (per-day attempt cap reached — stop issuing creates).

    ``count_attempt_cap=False`` (the explicit ``backend: gcp`` pin) skips
    BOTH the cap-check and the per-day counter bump — an explicit user ask
    attempts regardless of the auto-escalation cap (pre-#656 explicit-gcp
    behavior never touched the counter).

    The rung-spec already carries its machine override + provisioning model
    (via :func:`_with_machine`), so the headroom pre-check, the create, the
    quota metric, and the zone filter all resolve THIS rung's true machine.

    A :class:`GcpWorkloadError` on the create raises the internal
    :class:`_GcpWorkloadFailover` signal (the ladder STOPS; the lane caller
    fails over STRAIGHT to RunPod — task #658). It does not cascade across
    the remaining GCP rungs or the SLURM lanes.
    """
    # Pre-create regional-quota headroom check (#608) for THIS rung. When
    # the probe POSITIVELY reports insufficient headroom, skip the rung
    # loudly WITHOUT bumping the attempt counter — the cap bounds provision
    # attempts, and a create that cannot succeed should not consume one.
    # FAIL-OPEN: a probe failure / a backend without the probe / a live
    # reconnectable instance returns None → proceed to the create.
    headroom = _gcp_quota_headroom_or_none(gcp_backend, spec)
    if headroom is not None and not headroom.sufficient:
        attempts.append(
            RouteAttempt(
                kind="gcp",
                cluster=None,
                est_start_seconds_raw=0.0,
                est_start_seconds_clamped=0.0,
                outcome="quota_headroom_insufficient",
                detail=(
                    f"rung {rung_label}: regional accelerator quota {headroom.metric} in "
                    f"{headroom.region} has usage {headroom.usage:g}/{headroom.limit:g} — "
                    f"headroom {headroom.available:g} GPU(s) < needed {headroom.needed}; "
                    "skipping this rung without burning a daily attempt"
                ),
                elapsed_seconds=now_fn() - started_at,
            )
        )
        logger.warning(
            "route: GCP rung %s quota headroom insufficient for issue %d; advancing.",
            rung_label,
            spec.issue,
        )
        return "advance"

    # #1029 boot-loop breaker: a rung with >= N consecutive pre-workload boot
    # deaths TODAY for THIS issue is skipped on the auto chain — advancing to
    # the next rung (and eventually the RunPod terminal rung) instead of
    # re-creating a VM that just boot-looped. No daily attempt is consumed
    # (the cap counts CREATES; a skip avoids the create). The explicit
    # `backend: gcp` pin (count_attempt_cap=False) is EXEMPT — an explicit
    # user ask attempts anyway, mirroring the cap exemption above. The skip
    # reads the REAL rung_label route() is walking, so the poller's
    # transitional "unknown_rung" lease key (pre-fix handles) can never match
    # a route-side read.
    if count_attempt_cap:
        streak = gcp_boot_death_streak(spec.issue, rung_label, lease_store=store)
        if streak >= gcp_boot_death_streak_threshold():
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="boot_loop_rung_skipped",
                    detail=(
                        f"rung {rung_label}: {streak} consecutive pre-workload boot "
                        f"deaths today (#1029 boot-loop breaker); skipping without "
                        f"burning a daily attempt"
                    ),
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: GCP rung %s boot-looped %dx today for issue %d; advancing.",
                rung_label,
                streak,
                spec.issue,
            )
            return "advance"

    with store.transaction(spec.issue) as (lease, write):
        # Cap-check BEFORE bump-and-persist, RE-READ this rung iteration
        # (advisory: the ladder must stop issuing creates the moment the cap
        # is hit, even mid-ladder). A rejected over-cap attempt MUST NOT grow
        # the on-disk counter. Rollover-on-day-change is part of the cap
        # probe so a fresh UTC day admits the new attempt.
        if lease is None:
            lease = Lease(
                issue=int(spec.issue),
                spec_hash=spec_hash(spec),
                # Placeholder only — superseded by the fresh per-launch mint in
                # _thread_attempt_id_into just before _prepare_and_launch below
                # (#927); this lease exists here so the cap counter has a home.
                # Suffixed anyway for consistency (#934).
                attempt_id=_make_attempt_id(spec.extra.get("lane_suffix")),
            )
        today = _today_utc_iso()
        attempts_already_today = lease.gcp_attempts_today if lease.gcp_attempts_date == today else 0
        if count_attempt_cap and attempts_already_today >= cfg.max_gcp_attempts_per_day:
            # Cap hit → no provision attempt is made (no credit spent).
            # Record it and signal the ladder to STOP issuing creates and
            # fall through (next lane / RunPod). NEVER raises (#656: the
            # RunPod terminal rung is the new tail of the chain).
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="attempt_cap_exceeded",
                    detail=(
                        f"rung {rung_label}: per-day GCP attempt cap "
                        f"{cfg.max_gcp_attempts_per_day} reached; stopping GCP creates, "
                        "falling through the lane order"
                    ),
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: per-day GCP attempt cap (%d) reached for issue %d at rung %s; "
                "stopping GCP creates and falling through.",
                cfg.max_gcp_attempts_per_day,
                spec.issue,
                rung_label,
            )
            return "cap_hit"
        if count_attempt_cap:
            # Auto chain: bump + persist (rollover folded into the bump).
            lease = _bump_gcp_attempt(lease)
            write(lease)
            attempts_today = lease.gcp_attempts_today
        else:
            # Explicit `backend: gcp` pin: cap-exempt, counter untouched —
            # report the current (un-bumped) reading for the marker detail.
            attempts_today = attempts_already_today

        # Pre-escalation marker — visible breadcrumb before spending credit,
        # carrying the resolved gpu-h estimate + short-job threshold so a
        # future debug-er sees the rung's reasoning. Posted INSIDE the flock.
        machine = machine_for_intent(spec)
        gpu_h = _estimated_gpu_hours(spec, machine)
        _post_intermediate_marker(
            spec=spec,
            marker_poster=marker_poster,
            reason=ROUTE_REASON_AUTO_FALLBACK_GCP,
            attempts_today=attempts_today,
            requested_kind=requested_kind,
            extra={
                "rung": rung_label,
                "estimated_gpu_hours": gpu_h,
                "spot_max_gpu_hours": _spot_max_gpu_hours(),
            },
        )

        threaded_spec, lease = _thread_attempt_id_into(spec, lease, write)
        try:
            gcp_handle = _prepare_and_launch(gcp_backend, threaded_spec, kind="gcp")
        except BackendPrepareError as exc:
            # Provision-class (nothing live) → advance to the next rung.
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="prepare_failed",
                    detail=f"rung {rung_label}: {exc.reason}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: gcp rung %s prepare failed (%s); advancing.", rung_label, exc.reason
            )
            return "advance"
        except GcpProvisioningError as exc:
            # Capacity / quota / zone exhaustion (incl. spot-capacity misses,
            # which classify here via gcp.classify_create_failure) → advance
            # to the next rung.
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="provisioning_failure",
                    detail=f"rung {rung_label}: {_provisioning_detail(exc)}",
                    elapsed_seconds=now_fn() - started_at,
                    evidence=_provisioning_evidence(exc),
                )
            )
            logger.warning(
                "route: gcp rung %s provisioning failed (%s); advancing.",
                rung_label,
                _provisioning_detail(exc),
            )
            return "advance"
        except BackendProbeError as exc:
            # GCP state UNKNOWN (expired auth / transport). No credit is spent
            # on unknown state → advance (the same safe reaction the SLURM
            # lanes take on an unprobeable reconnect).
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="probe_failed",
                    detail=f"rung {rung_label}: {str(exc)[:500]}",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: gcp rung %s state probe failed (%s); advancing.", rung_label, exc
            )
            return "advance"
        except GcpWorkloadError as exc:
            # GCP workload failure (broken run on GCP) → FAIL OVER TO RUNPOD,
            # not a hard terminal (reversed invariant, task #658). The ladder
            # STOPS here (no cascade across the remaining GCP rungs) and the
            # internal signal makes the lane callers route STRAIGHT to the
            # RunPod terminal rung — bypassing the free SLURM lanes too
            # (re-crashing broken code on a SLURM lane burns queue time, the
            # original no-cascade concern). RunPod pods persist + are
            # SSH-able, so the next attempt is also strictly better for
            # diagnosis than GCP's delete-on-crash boot disk. No terminal
            # marker posted here — the route is failing over, not ending;
            # the RunPod terminal rung posts the next marker. The bound on a
            # genuinely-broken job (RunPod runs it at most once more, then
            # status:blocked, no re-drive) lives in
            # ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD's docstring.
            attempts.append(
                RouteAttempt(
                    kind="gcp",
                    cluster=None,
                    est_start_seconds_raw=0.0,
                    est_start_seconds_clamped=0.0,
                    outcome="workload_failure",
                    detail=f"rung {rung_label}: {exc.reason}; failing over to RunPod",
                    elapsed_seconds=now_fn() - started_at,
                )
            )
            logger.warning(
                "route: GCP rung %s WORKLOAD failure for issue %d (%s); failing over to "
                "RunPod (no cascade across remaining GCP rungs / SLURM lanes).",
                rung_label,
                spec.issue,
                exc.reason,
            )
            raise _GcpWorkloadFailover(
                residual_gap=(
                    f"GCP workload failure on rung {rung_label} ({exc.reason}) — failing "
                    "over to RunPod for a persistent, SSH-able run (GCP DELETEs its boot "
                    "disk + logs on crash)"
                ),
                evidence=exc.evidence,
            ) from exc

        # #1029: thread the ladder-rung label + wall-clock launch ts onto the
        # handle so the async poller can key the boot-death streak per
        # (issue, rung) and age post-DELETE observations. BEFORE
        # _invoke_on_launched so even the crash-window sidecar copy carries the
        # keys (handle.extra is a mutable dict by design — the per_zone_attempts
        # pattern; assign in place). time.time(), NOT now_fn(): the poller ages
        # gcp_launched_ts against time.time() and route()'s now_fn defaults to
        # time.monotonic (a different epoch). NOTE: on the #736 reconnect path a
        # re-minted handle's setdefault stamps a ts that post-dates the true VM
        # create, biasing the age read YOUNGER — marginal (the heuristic floor
        # is 1500s) but worth this comment at the site.
        gcp_handle.extra.setdefault("gcp_ladder_rung", rung_label)
        gcp_handle.extra.setdefault("gcp_launched_ts", float(time.time()))
        # #1121/#1379: record the DECLARED width (spec.gpus, or the explicit
        # wide intent's own base width — _declared_width; None when
        # undeclared) + the REALIZED width of the rung's resolved machine,
        # so the sidecar / marker trail lets the workload re-shard off the
        # realized width (a degraded launch may land narrower than
        # requested). ``machine`` above resolved THIS rung's true machine
        # via the override.
        gcp_handle.extra.setdefault("requested_gpus", _declared_width(spec))
        gcp_handle.extra.setdefault("realized_gpu_count", int(machine.gpu_count))

        # Persist the handle (sidecar hook) + launched id IMMEDIATELY
        # (still under the flock).
        _invoke_on_launched(on_launched, gcp_handle)
        write(_lease_after_submit(lease, spec, "gcp", None, gcp_handle))

    attempts.append(
        RouteAttempt(
            kind="gcp",
            cluster=None,
            est_start_seconds_raw=0.0,
            est_start_seconds_clamped=0.0,
            outcome="launched",
            detail=(
                f"gcp rung {rung_label} "
                + (
                    f"escalation #{attempts_today} of cap {cfg.max_gcp_attempts_per_day}"
                    if terminal
                    else f"primary-lane attempt #{attempts_today} of cap "
                    f"{cfg.max_gcp_attempts_per_day}"
                )
            ),
            elapsed_seconds=now_fn() - started_at,
        )
    )
    result = RouteResult(
        backend=gcp_backend,
        handle=gcp_handle,
        requested_kind=requested_kind,
        chosen_kind="gcp",
        # ``reason`` is ``auto_fallback_gcp`` on the auto chain and ``override``
        # on an explicit ``backend: gcp`` pin (the marker schema is stable by
        # design; the attempts trail + the ``gcp_ladder_rung`` extra reflect the
        # actual rung that launched).
        reason=reason,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        extra={
            "gcp_attempts_today": attempts_today,
            "gcp_ladder_rung": rung_label,
            # #1121/#1379: declared vs realized width on the
            # epm:backend-selected marker surface (additive extra fields, same
            # class as gcp_ladder_rung — no marker SCHEMA change). A #1379
            # explicit-wide degradation is machine-readable as
            # requested_gpus != realized_gpu_count.
            "requested_gpus": _declared_width(spec),
            "realized_gpu_count": int(machine.gpu_count),
            **_gcp_marker_extras(spec),
        },
    )
    _post_backend_selected(result, spec=spec, marker_poster=marker_poster)
    return result


# ---------------------------------------------------------------------------
# Internal helpers (lease, estimate, reconnect, marker)
# ---------------------------------------------------------------------------


def _estimate_lanes(
    candidates: Iterable[tuple[ComputeBackend, BackendKind]],
    *,
    spec: RunSpec,
    estimate_fn: Callable[[ComputeBackend, BackendKind, RunSpec], float | None] | None,
) -> list[tuple[ComputeBackend, BackendKind, float | None]]:
    """Probe each candidate's est-start; return as ``(backend, kind, raw)`` triples.

    Default ``estimate_fn`` calls
    ``backend.estimate_start_seconds(spec)`` when the backend exposes
    the method (SLURM does), else returns None. The router treats
    ``None`` as "unranked but park-eligible".
    """
    triples: list[tuple[ComputeBackend, BackendKind, float | None]] = []
    fn = estimate_fn or _default_estimate
    for backend, kind in candidates:
        try:
            raw = fn(backend, kind, _spec_for_lane(spec, kind))
        except Exception as exc:
            logger.warning(
                "route: estimate_fn raised for %s (%s: %s); treating as unranked.",
                kind,
                type(exc).__name__,
                exc,
            )
            raw = None
        triples.append((backend, kind, raw))
    return triples


def _default_estimate(backend: ComputeBackend, kind: BackendKind, spec: RunSpec) -> float | None:
    """Fall back to ``backend.estimate_start_seconds(spec)`` when present."""
    del kind
    fn = getattr(backend, "estimate_start_seconds", None)
    if fn is None:
        return None
    return fn(spec)


def _try_reconnect(
    *,
    backend: ComputeBackend,
    kind: BackendKind,
    spec: RunSpec,
    reconnect_fn: (Callable[[ComputeBackend, BackendKind, RunSpec], RunHandle | None] | None),
) -> RunHandle | None:
    """Look for an existing live job/instance for ``spec`` on ``backend``.

    Backend-aware reconnect lives in the backend itself (SLURM:
    ``squeue --name eps-issue-<N>``; GCP: :func:`gcp.reconnect_or_none`).
    The injected ``reconnect_fn`` wraps that — production-default
    (slice 6) wires per-backend probes; tests pass ``None`` to disable
    reconnect entirely.

    When the lease has an ``UNKNOWN_SUBMITTED`` recovery state (lease
    present but no job_id — submit returned but the orchestrator
    crashed before persisting), we ALSO call the reconnect_fn — the
    backend's queue may show the job even though we never recorded its
    id locally. This is the slice-5 "UNKNOWN_SUBMITTED" recovery hook.

    :class:`BackendProbeError` PROPAGATES (round-6 B1): it means the
    probe itself failed (transport down) — "couldn't ask" treated as
    "no live job" lets the caller submit a blind duplicate. Each call
    site decides the safe reaction (skip the lane on the auto chain;
    typed terminal on an explicit override).
    """
    if reconnect_fn is None:
        return None
    try:
        handle = reconnect_fn(backend, kind, spec)
    except BackendProbeError:
        raise
    except Exception as exc:
        logger.warning(
            "route: reconnect_fn raised for %s (%s: %s); treating as no live job.",
            kind,
            type(exc).__name__,
            exc,
        )
        return None
    if handle is None:
        return None
    # Defensive: a reconnect_fn that returns a handle for the WRONG
    # issue would silently bind to someone else's run. Sanity-check.
    if handle.extra.get("issue") not in (None, int(spec.issue)):
        logger.error(
            "route: reconnect_fn for %s returned a handle for issue=%r (expected %d); ignoring.",
            kind,
            handle.extra.get("issue"),
            spec.issue,
        )
        return None
    # Defensive: a misconfigured reconnect_fn that binds to the WRONG
    # backend kind (e.g. a GCP probe wired into the nibi slot) would
    # silently re-attach to someone else's lane. The handle carries the
    # backend it was issued by; cross-check.
    #
    # Production SLURM handles carry ``backend="cluster"`` with the
    # concrete lane in ``handle.cluster`` — both ``SlurmBackend.launch``
    # and the dispatch CLI's reconnect closure return that shape (round-2
    # Codex Critical, task #535: requiring ``handle.backend == kind``
    # here rejected EVERY live production SLURM reconnect handle, so
    # ``route()`` could fresh-submit a duplicate job on a lane that
    # already had one). Accept the ``"cluster"`` alias only when the
    # concrete cluster matches the lane being probed; a cluster handle
    # for a DIFFERENT cluster (or with no cluster at all) is still the
    # cross-lane mismatch this guard exists for.
    if handle.backend != kind and not (handle.backend == "cluster" and handle.cluster == kind):
        logger.error(
            "route: reconnect_fn for kind=%s returned a handle issued by "
            "backend=%s (cluster=%s); ignoring.",
            kind,
            handle.backend,
            handle.cluster,
        )
        return None
    return handle


def _lease_already_failed_over(
    lease: Lease | None, gcp_failover_of_identity: dict[str, Any]
) -> bool:
    """True iff the in-flock lease ALREADY records a RunPod failover of this GCP crash (M3b, #669).

    The atomic in-flock twin of ``backend_poll._lease_records_failover_of`` (the
    OUTSIDE-the-flock fast-path pre-check): a second concurrent triggerer that
    acquires the per-issue flock AFTER the first stamped sees a RunPod lease
    whose ``gcp_failover_of`` matches the GCP-crash identity, and short-circuits
    to the existing handle instead of launching a second paid pod. Keyed to the
    GCP run's stable identity (``pod_name``/``job_id``), so a genuinely-new GCP
    crash on the same issue (fresh dispatch → new identity) does NOT match and
    still gets its own single failover.
    """
    return (
        lease is not None
        and lease.backend == "runpod"
        and lease.gcp_failover_of == gcp_failover_of_identity
    )


def _route_result_from_existing_failover_lease(
    *,
    lease: Lease,
    spec: RunSpec,
    runpod_backend: ComputeBackend,
    reason: str,
    attempts: list[RouteAttempt],
    started_at: float,
    now_fn: Callable[[], float],
    residual_gap: str,
) -> RouteResult:
    """Reconstruct the RouteResult for a failover a CONCURRENT triggerer already launched (M3b).

    The first triggerer's RunPod launch is recorded ONLY as a lease (the full
    ``RunHandle`` is not persisted on the lease), but the RunPod handle is
    deterministic from the issue (``pod-<N>`` per ``runpod._pod_name_for`` /
    ``_runpod_log_path``), so a minimal handle is reconstructed for the
    short-circuit return. The caller's own sidecar re-point + readback
    (``backend_poll._failover_dead_gcp_to_runpod``) reads the authoritative
    RunPod handle from disk on the next tick; this result only needs to carry a
    truthful RunPod ``pod_name`` so NO second launch happens.
    """
    from explore_persona_space.backends.runpod import _runpod_log_path

    handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id=str(lease.job_id or ""),
        pod_name=f"pod-{int(spec.issue)}",
        scratch_dir="/workspace",
        log_path=_runpod_log_path(int(spec.issue)),
        extra={"issue": int(spec.issue)},
    )
    attempts.append(
        RouteAttempt(
            kind="runpod",
            cluster=None,
            est_start_seconds_raw=0.0,
            est_start_seconds_clamped=0.0,
            outcome="failover_already_launched",
            detail="concurrent triggerer already failed over this GCP crash (M3b re-check)",
            elapsed_seconds=now_fn() - started_at,
        )
    )
    return RouteResult(
        backend=runpod_backend,
        handle=handle,
        requested_kind=spec.backend,
        chosen_kind="runpod",
        reason=reason,
        cluster=None,
        attempts=attempts,
        elapsed_seconds=now_fn() - started_at,
        extra={"runpod_fallback_residual_gap": residual_gap, "failover_already_launched": True},
    )


def _lease_after_submit(
    lease: Lease | None,
    spec: RunSpec,
    backend_kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
) -> Lease:
    """Pure helper: produce the lease record that records a fresh submit.

    Used inside an OPEN ``store.transaction`` so the read-check + launch +
    lease-write all hold the same flock (the read happened when the
    caller opened the transaction; this returns the new value the caller
    will hand to ``write_fn``). Pre-existing GCP attempt counter +
    spec_hash + attempt_id fields are preserved on ``lease`` — the
    threading step (:func:`_thread_attempt_id_into`) already set
    ``attempt_id`` to THIS launch's id, so preserving it here records the
    id the launch actually used. Absent lease → fresh one with the
    spec's attempt_id (or a freshly minted one if none).
    """
    if lease is None:
        lease = Lease(
            issue=int(spec.issue),
            spec_hash=spec_hash(spec),
            attempt_id=str(
                spec.extra.get("attempt_id") or _make_attempt_id(spec.extra.get("lane_suffix"))
            ),
        )
    lease.backend = backend_kind
    lease.cluster = cluster
    lease.job_id = str(handle.job_id)
    lease.submitted_at = float(time.time())  # wall-clock, not monotonic
    return lease


def _persist_lease_after_submit(
    *,
    spec: RunSpec,
    store: LeaseStore,
    backend_kind: BackendKind,
    cluster: str | None,
    handle: RunHandle,
    now_fn: Callable[[], float],
) -> None:
    """Open a flocked transaction + write the lease after a submit.

    Crash window covered: a submit that returns successfully but the
    orchestrator dies before the lease is updated would otherwise leave
    a leaked job / instance. Prefer the in-transaction
    :func:`_lease_after_submit` helper when the caller is ALREADY inside
    a transaction (the override / auto-route paths hold the flock across
    reconnect-check → launch → lease-write to seal the double-submit
    race).
    """
    del now_fn  # monotonic clock is for the watchdog, not the lease timestamp
    with store.transaction(spec.issue) as (lease, write):
        write(_lease_after_submit(lease, spec, backend_kind, cluster, handle))


def _thread_attempt_id_into(
    spec: RunSpec,
    lease: Lease | None,
    write_fn: Callable[[Lease], None],
) -> tuple[RunSpec, Lease]:
    """Mint a FRESH attempt id for this launch + thread it into the spec (#927).

    Returns ``(new_spec, lease)`` where ``new_spec`` carries the threaded
    ``attempt_id`` in ``extra``, and ``lease`` is the (possibly freshly
    created) lease record — updated to the id the launch actually uses
    and written via ``write_fn`` on BOTH branches ("lease follows the
    launch", never vice versa). Every call site is a fresh-submit path
    (the reconnect probes early-return above it), so "called ⇒ mint
    fresh" is the invariant: reusing ``lease.attempt_id`` made a NEW
    launch inherit a dead prior attempt's crash-persist / sentinel /
    ``expected_artifacts`` namespace (#825: three relaunches all wrote
    under ``att-20260702-061417``).

    Reconnect id-stability does NOT live here: the router-level reconnect
    early-returns before this function runs, and GCP recovers the
    original id from the instance's ``eps-attempt`` label
    (``gcp.reconnect_or_none``). One caveat on that label path: on the
    ``GcpBackend.launch``-INTERNAL reconnect race (the instance came
    alive between the router probe and ``launch()``), the instance's
    ``eps-attempt`` LABEL id wins while the lease records the unused
    fresh mint — so "lease follows the launch" is NOT guaranteed on that
    race path; a future lease-id consumer must not design against it
    (the handle sidecar, not the lease, names the live attempt).

    A caller-pinned ``spec.extra["attempt_id"]`` takes precedence over
    the fresh mint — even when a lease exists. Pinning is for explicit
    re-attach tooling ONLY: a pinned spec re-routed across relaunches
    reproduces the #825 namespace-collision class by construction.

    MUST be called inside the caller's OPEN ``store.transaction`` — the
    caller's transaction owns the flock (a nested ``store.transaction``
    would deadlock: :py:func:`fcntl.flock` from a fresh
    open-file-description in the same process blocks against any held
    lock), and the in-transaction ``write_fn`` keeps the
    reconnect-check → launch → lease-write sequence atomic.
    """
    current_id = (spec.extra or {}).get("attempt_id")
    attempt_id = str(current_id or _make_attempt_id((spec.extra or {}).get("lane_suffix")))
    if lease is None:
        lease = Lease(
            issue=int(spec.issue),
            spec_hash=spec_hash(spec),
            attempt_id=attempt_id,
        )
    else:
        lease.attempt_id = attempt_id  # lease follows the launch, never vice versa
    write_fn(lease)
    new_extra = dict(spec.extra or {})
    new_extra["attempt_id"] = attempt_id
    return replace(spec, extra=new_extra), lease


def _make_attempt_id(lane_suffix: str | None = None) -> str:
    """Per-attempt id — same shape the GCP backend's ``attempt_id_for`` produces.

    ``lane_suffix`` (#934): appended as ``-<suffix>`` so two concurrent
    lanes launched in the SAME second mint DISTINCT attempt ids — the
    shared HF crash-persist prefix ``issue<N>_partial/<attempt>/`` and
    the sentinel dir ``eval_results/issue_<N>/<attempt>/`` would
    otherwise collide. Validated (fail loud, never strip); ``None`` /
    empty keeps the unsuffixed shape byte-identical. The suffix charset
    ``[a-z0-9-]`` is within ``gcp.attempt_id_for``'s acceptance regex
    and the ``eps-attempt`` label charset, and the 43-char cap in
    ``base.validate_lane_suffix`` keeps the full id under the 63-char
    GCP label truncation so reconnect label recovery stays lossless.
    """
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{validate_lane_suffix(lane_suffix)}" if lane_suffix else ""
    return f"att-{ts}{suffix}"


def _post_marker_nonfatal(
    marker_poster: Callable[..., None],
    *,
    issue: int,
    note: str,
    context: str,
) -> None:
    """Invoke ``marker_poster``; NEVER let a marker-post failure alter routing.

    Every router ``epm:backend-selected`` post fires either AFTER a
    successful launch (the success breadcrumb — live infra in hand) or
    immediately BEFORE raising a typed terminal (the failure
    breadcrumb). A raise from the poster itself (e.g.
    ``post_marker_via_task_py``'s ``subprocess.run(check=True,
    timeout=30)`` hitting flock contention) would either convert
    "launched, handle in hand" into an unclassified dispatch-CLI rc=4
    with a live, billing VM/job, or clobber the typed terminal the
    orchestrator's failure-classifier routes on. Markers are an
    observability side channel, not control flow — failures are logged
    LOUD (ERROR + the full payload, never silently swallowed) and the
    route continues.
    """
    try:
        marker_poster(
            issue=issue,
            marker="epm:backend-selected",
            note=note,
            version=1,
            by="backends.router",
        )
    except Exception:
        logger.exception(
            "route: epm:backend-selected marker post FAILED (%s) for issue=%d; "
            "continuing — markers must never alter routing control flow. payload=%s",
            context,
            issue,
            note,
        )


def _gcp_marker_extras(spec: RunSpec) -> dict[str, Any]:
    """Build the GCP ``epm:backend-selected`` ``extra`` dict for ``spec``.

    Every ``RouteResult`` whose ``chosen_kind == "gcp"`` must merge these
    keys into its ``extra`` before reaching :func:`_post_backend_selected`
    — fresh launches AND reconnect paths — so the dashboard observability
    the plan promised (``provisioning_model`` + ``quota_pool``) is
    delivered on EVERY gcp launch, not just first-attempt auto launches
    (#631 round-3: the original fix added these to one of four terminal
    paths). Pure function, no IO.

    Safe to call unguarded on any gcp-chosen result. ``provisioning_model``
    reads only ``spec.extra`` (no intent lookup), so it is always
    populated. ``quota_pool`` resolves the intent's machine: it is ``None``
    when the (gpu_kind, pool) pair has no quota mapping AND — per the round-4
    fix below — when the intent itself is unmapped. The reconnect paths
    (``router.py`` explicit-override + auto-chain) call this AFTER an
    idempotent reconnect that found the live instance by NAME only, never
    re-resolving ``machine_for_intent(spec)``; so an unmapped ``spec.intent``
    (e.g. ``ft-70b``, no ``INTENT_TO_MACHINE`` row) would otherwise raise
    ``ValueError`` here and crash a SUCCESSFUL reconnect on observability
    code. Observability must never crash a live run, so we degrade
    ``quota_pool`` to ``None`` — the same fail-open contract the preflight
    quota-headroom check already uses; the marker schema documents
    ``quota_pool: ... | None``.
    """
    provisioning = resolve_provisioning_model(spec)
    try:
        pool = quota_metric_for(machine_for_intent(spec), provisioning)
    except ValueError:
        # Unmapped intent on a reconnect to a live instance whose original
        # intent has no INTENT_TO_MACHINE row. Degrade rather than crash.
        pool = None
    return {
        "provisioning_model": provisioning,
        "quota_pool": pool,
        # True only when the router switched this launch STANDARD->SPOT via
        # the on-demand-exhausted auto-fallback (#537); absent/False on a
        # plain STANDARD or explicitly-requested SPOT/FLEX_START launch.
        "spot_fallback": bool((spec.extra or {}).get("spot_fallback", False)),
    }


def _post_backend_selected(
    result: RouteResult,
    *,
    spec: RunSpec,
    marker_poster: Callable[..., None] | None,
) -> None:
    """Post ``epm:backend-selected v1`` with the EXTENDED router body.

    Extended fields beyond the selector's schema (see workflow.yaml §
    markers):

    * ``attempts`` — list of per-lane attempt records (raw + clamped
      est-start, outcome, detail, elapsed), appended chronologically so
      the trail reflects the ACTUAL attempt order (GCP first when the
      GCP-first default ran it first).
    * Existing schema preserved: ``requested_kind`` / ``chosen_kind`` /
      ``reason`` / ``cluster`` / ``elapsed_seconds`` / ``extra``.

    Non-fatal: every call site runs AFTER a successful launch /
    reconnect, so a poster failure must never propagate past live infra
    (see :func:`_post_marker_nonfatal`).
    """
    if marker_poster is None:
        return
    body = {
        "requested_kind": result.requested_kind,
        "chosen_kind": result.chosen_kind,
        "reason": result.reason,
        "cluster": result.cluster,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "attempts": [_attempt_to_dict(a) for a in result.attempts],
        "extra": dict(result.extra),
    }
    _post_marker_nonfatal(
        marker_poster,
        issue=spec.issue,
        note=json.dumps(body, sort_keys=True),
        context=f"backend-selected chosen_kind={result.chosen_kind}",
    )


def _post_intermediate_marker(
    *,
    spec: RunSpec,
    marker_poster: Callable[..., None] | None,
    reason: str,
    attempts_today: int,
    requested_kind: BackendKind | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Post a visible "about to escalate to GCP" breadcrumb.

    Per plan §6: "Before escalating to GCP, post a visible marker (credit
    is scarce/expiring)". Body uses the same ``epm:backend-selected``
    schema with ``chosen_kind: "gcp"`` so the dashboard surfaces the
    intent. The final marker (posted after GCP launch succeeds /
    fails) carries the resolved outcome — both events appear in the
    timeline.

    ``requested_kind`` records the user's ORIGINAL ``--backend`` ask, exactly
    as the final/terminal markers carry it on :class:`RouteResult`: ``None``
    for the auto chain (no ``--backend``), ``"gcp"`` for an explicit
    ``backend: gcp`` pin. Threading it here keeps the intermediate breadcrumb
    consistent with the rest of the marker trail, so a post-hoc reader can tell
    an auto-chain GCP escalation apart from an explicit GCP override gone wrong
    (#672 — the breadcrumb used to hardcode ``None`` regardless of the ask).

    ``extra`` merges additional observability keys into the marker body's
    ``extra`` (#656: the ladder threads the rung label + resolved gpu-h
    estimate + short-job threshold so a future debug-er sees the rung's
    reasoning); ``intermediate`` + ``gcp_attempts_today`` always win.
    """
    if marker_poster is None:
        return
    body = {
        "requested_kind": requested_kind,
        "chosen_kind": "gcp",
        "reason": reason,
        "cluster": None,
        "elapsed_seconds": 0.0,
        "attempts": [],
        "extra": {
            **(extra or {}),
            "intermediate": True,
            "gcp_attempts_today": attempts_today,
        },
    }
    _post_marker_nonfatal(
        marker_poster,
        issue=spec.issue,
        note=json.dumps(body, sort_keys=True),
        context="pre-escalation breadcrumb",
    )


def _post_terminal_failure_marker(
    *,
    spec: RunSpec,
    marker_poster: Callable[..., None] | None,
    reason: str,
    chosen_kind: BackendKind,
    attempts: list[RouteAttempt],
    extra: dict[str, Any] | None = None,
) -> None:
    """Post a final ``epm:backend-selected`` breadcrumb BEFORE raising terminal.

    The router's terminal-failure paths (``NoComputeAvailableError``,
    ``WorkloadSurfacedError``, ``ManualAttentionRequiredError``) raise
    rather than return — without this marker the dashboard would never
    see the failure breadcrumb that the success path always posts. Wires
    the reason code (:data:`ROUTE_REASON_NO_COMPUTE` /
    :data:`ROUTE_REASON_WORKLOAD_FAILURE`) the slice-5 module exports as
    public constants so downstream surfaces can pattern-match on them.
    """
    if marker_poster is None:
        return
    body = {
        "requested_kind": None,
        "chosen_kind": chosen_kind,
        "reason": reason,
        "cluster": None,
        "elapsed_seconds": 0.0,
        "attempts": [_attempt_to_dict(a) for a in attempts],
        "extra": dict(extra or {}),
    }
    # Non-fatal: a poster failure here would clobber the typed terminal
    # (NoCompute / WorkloadSurfaced / ManualAttention) about to be
    # raised — the orchestrator's failure-classifier needs THAT
    # exception, not an unclassified marker-transport error.
    _post_marker_nonfatal(
        marker_poster,
        issue=spec.issue,
        note=json.dumps(body, sort_keys=True),
        context=f"terminal-failure breadcrumb reason={reason}",
    )


def _attempt_to_dict(a: RouteAttempt) -> dict[str, Any]:
    d: dict[str, Any] = {
        "kind": a.kind,
        "cluster": a.cluster,
        "est_start_seconds_raw": a.est_start_seconds_raw,
        "est_start_seconds_clamped": a.est_start_seconds_clamped,
        "outcome": a.outcome,
        "detail": a.detail,
        "elapsed_seconds": round(a.elapsed_seconds, 3),
    }
    # Emit the structured evidence ONLY when populated (#774) so the common
    # no-evidence attempt serializes byte-identically to the pre-#774 7-field
    # shape — no schema break for any existing marker reader.
    if a.evidence:
        d["evidence"] = a.evidence
    return d


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "CANCEL_LIVE_GRACE_SECONDS",
    "DEFAULT_AUTO_LANE_ORDER",
    "DEFAULT_FREE_LANE_ORDER",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_SPOT_MAX_GPU_HOURS",
    "ENV_AUTO_LANE_ORDER",
    "ENV_GCP_SPOT_FALLBACK",
    "ENV_SPOT_MAX_GPU_HOURS",
    "FREE_WAIT_SECONDS",
    "LEASE_STORE_DIRNAME",
    "MAX_GCP_ATTEMPTS_PER_DAY",
    "PARK_MAX_CONSECUTIVE_PROBE_FAILURES",
    "ROUTE_REASON_AUTO_FALLBACK_GCP",
    "ROUTE_REASON_AUTO_STARTED",
    "ROUTE_REASON_CPU_EXHAUSTED_NO_RUNPOD",
    "ROUTE_REASON_CPU_FALLBACK_INFEASIBLE",
    "ROUTE_REASON_GCP_QUEUE_TIMEOUT_FAILOVER_RUNPOD",
    "ROUTE_REASON_GCP_QUEUE_VANISH_FAILOVER_RUNPOD",
    "ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD",
    "ROUTE_REASON_GCP_WORKLOAD_FAILOVER_RUNPOD_ASYNC",
    "ROUTE_REASON_NO_COMPUTE",
    "ROUTE_REASON_OVERRIDE",
    "ROUTE_REASON_PREPARE_FAILED",
    "ROUTE_REASON_RECONNECT",
    "ROUTE_REASON_RUNPOD_FALLBACK",
    "ROUTE_REASON_WORKLOAD_FAILURE",
    "RUNPOD_CPU_INSTANCE_CAPS",
    "RUNPOD_CPU_INSTANCE_FOR_INTENT",
    "RUNPOD_INTENT_FOR_GCP_INTENT",
    "RUNPOD_INTENT_TRANSLATION_DELIBERATE_GAPS",
    "BackendPrepareError",
    "CpuExhaustedNoRunpodLaneError",
    "CpuFallbackInfeasibleError",
    "GcpAttemptCapExceededError",
    "Lease",
    "LeaseStore",
    "ManualAttentionRequiredError",
    "NoComputeAvailableError",
    "RouteAttempt",
    "RouteError",
    "RouteResult",
    "RouterConfig",
    "RunPodCpuInstanceCaps",
    "WorkloadSurfacedError",
    "auto_lane_order",
    "cancel_and_wait",
    "canonicalize_spec",
    "default_is_live",
    "default_is_started",
    "failover_to_runpod_after_async_workload_crash",
    "park_until_running_or_cap",
    "rank_lanes",
    "route",
    "spec_hash",
]
