---
title: 'Remove GCP compute backend entirely (GPU + CPU): auto chain, explicit pins,
  cpu-bigmem RunPod re-map, tests, docs (user directive)'
kind: infra
tags:
- user-directive
created_at: '2026-08-03T03:41:41Z'
has_clean_result: false
origin_prompt: remove GCP as an option for GPUs
workflow: v1
---
# Remove GCP as a compute backend entirely — GPU and CPU (user directive)

## Overview / Motivation

User directive (interactive chat, 2026-08-02): **"remove GCP as an option for GPUs"**,
then clarified same session: **"no remove GCP fully"** — the CPU lanes go too. GCP
should no longer be selectable by any dispatch path, GPU or CPU. Context: the fellows
lane (charmander H200, free) is the standing first lane; recent incidents (#1739:
~12 consecutive habit-pinned `--backend gcp` dispatches burned ~40+ GPU-h of credits
while charmander sat idle; #2018 tracks the habit-guard mechanization).

## Goal

No dispatch path — auto chain, explicit frontmatter pin, or `--backend gcp` CLI — may
provision ANY GCP instance (GPU or CPU). CPU intents re-map to RunPod CPU pods.

## Proposed change (refine in planning)

1. **Auto chain:** drop `"gcp"` from `DEFAULT_AUTO_LANE_ORDER`
   (`src/explore_persona_space/backends/router.py:706`) → `("fellows", "nibi", "fir", "mila")`,
   RunPod terminal rung unchanged. Decide whether `auto_lane_order()` /
   `EPM_AUTO_LANE_ORDER` validation should now REFUSE `gcp` loudly (like `runpod`) or
   accept-but-warn; refusing is more consistent with "removed as an option".
2. **Explicit pins:** an explicit `backend: gcp` frontmatter / `--backend gcp` dispatch
   (any intent) fails loud pre-launch with a typed error naming this policy — never
   silently rerouted.
3. **CPU intents route RunPod-only.** Drop the GCP E2 / n2-highmem CPU rungs; the
   RunPod CPU lane (`deployCpuPod`) becomes primary:
   - `cpu-small` → `cpu3g-2-8` (existing mapping, becomes primary instead of fallback)
   - `cpu-mid` → `cpu3c-8-16` (existing mapping, becomes primary)
   - `cpu-bigmem` → NEW mapping to a RunPod Memory-Optimized instance —
     `cpu5m-16-128` (16 vCPU / 128 GB, matching the GCP `n2-highmem-16` shape) with
     `cpu3m-16-128` as an alternate. VERIFIED at filing via live `cpuFlavors` GraphQL
     query (2026-08-02): cpu3m + cpu5m exist, ramMultiplier 8, maxVcpu 32, stockStatus
     High. Disk caps: `diskLimitPerVcpu` = 10 GB/vCPU (cpu3) / 15 GB/vCPU (cpu5), so
     cpu5m-16-128 caps at 240 GB container disk; a plan-stated `--boot-disk-gb` above
     the cap can scale vCPU (up to cpu5m-32-256 → 480 GB) — extend the #1010
     `RUNPOD_CPU_INSTANCE_CAPS` feasibility gate with the new instance rows instead of
     the current `cpu_exhausted_no_runpod_lane` typed terminal, which retires.
   - This retires the #677 `cpu_exhausted_no_runpod_lane` terminal (its premise — no
     RunPod lane for bigmem — is false given cpu5m) and the #747 GCP-first CPU
     ordering. A RunPod CPU no-capacity miss surfaces `RunPodNoCapacityError` →
     `no_compute_available` (watcher capacity-retry re-drivable), as today.
4. **Failover legs that re-enter GCP rungs** (the #1596/#1601 queue-vanish /
   queue-timeout → GCP on-demand retry legs in `backend_poll.py` /
   `router.retry_gcp_ondemand_after_queue_vanish`) must not create new GCP
   instances. Planner decides excise vs gate; the poller paths that only ACT ON
   existing GCP handles (crash persist, teardown, GCP→RunPod failover of an in-flight
   handle) should stay — they are cleanup, not provisioning.
5. **Tests:** update the pins — `tests/test_router.py::test_default_auto_lane_order_is_gcp_first`
   (pins the 5-lane tuple), the GCP ladder walk tests,
   `test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted`, and the #747
   CPU-lane tests (`test_router_cpu_small_capacity_miss_falls_over_to_runpod`,
   `test_router_cpu_intent_capacity_miss_no_runpod_fallback` — the cpu-bigmem guard,
   now inverted) — to the new contract; add pins that (a) the auto chain contains no
   gcp rung, (b) an explicit gcp pin refuses, (c) cpu-bigmem resolves to the RunPod
   memory-optimized instance.
6. **Docs:** update CLAUDE.md § "Compute backends — multi-lane router" (fellows-first
   bullet, the `--backend gcp` habit-guard bullet — now a hard removal, the CPU-intent
   table's GCP column), `.claude/rules/compute-backend-failover.md` (mark the
   GPU ladder / failover / CPU-lane sections historical or scoped to in-flight
   handles), and the GPU/CPU intent → spec tables. Grep the workflow surface for
   `backend.*gcp` prescriptions
   (`grep -rn 'backend.*gcp\|cpu-bigmem\|cpu_exhausted_no_runpod_lane' .claude/ CLAUDE.md docs/ scripts/ --include='*.md'`)
   and update prescriptive hits.

## Scope / constraints

- The GCP janitor (`gcp_audit.py` cron), crash-persist, and stale-instance reaping
  stay — they clean up existing/stray instances, they don't provision. Same for
  `gcloud` read probes in monitoring code.
- Rollback lever: keep it a small, revertible change (the fellows `available=False`
  precedent) — prefer a policy gate/constant over deep excision of the ladder code,
  so re-enabling GCP later is a flag flip, not code archaeology.
- Note for the CPU re-map: RunPod CPU pods are on-demand only (no spot lever), and
  `deployCpuPodInput` has NO `volumeInGb` field (container disk only) — the
  `cpu-bigmem` replacement relies on container disk within the per-vCPU cap.
- Workflow-surface + router change only; no experiment code.
- 0 GPU-h.

## Provenance

- origin: user-chat directive, session 2026-08-02
- verbatim (1): "remove GCP as an option for GPUs"
- verbatim (2, same session, superseding scope): "no remove GCP fully. is there no
  way to get CPUs on runpod?"
- verified-at-filing: `gcloud compute instances list --configuration=eps-gcp` →
  10 instances, 0 RUNNING (9 TERMINATED, 1 STOPPING) — no live work disrupted
  (2026-08-02). `DEFAULT_AUTO_LANE_ORDER` confirmed at router.py:706 as
  `("fellows", "gcp", *DEFAULT_FREE_LANE_ORDER)`; lane-order test pin confirmed at
  tests/test_router.py:3193. RunPod `cpuFlavors` live query confirmed cpu3m/cpu5m
  memory-optimized flavors (ramMultiplier 8, maxVcpu 32, stock High) — cpu-bigmem is
  mappable.
