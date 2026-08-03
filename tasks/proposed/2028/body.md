---
title: 'Remove GCP as a GPU compute lane: auto chain, explicit pins, tests, docs (user
  directive)'
kind: infra
tags:
- user-directive
created_at: '2026-08-03T03:41:41Z'
has_clean_result: false
origin_prompt: remove GCP as an option for GPUs
workflow: v1
---
# Remove GCP as an option for GPU compute (user directive)

## Overview / Motivation

User directive (interactive chat, 2026-08-02): **"remove GCP as an option for GPUs"**.

GCP GPU capacity should no longer be selectable by any dispatch path. Context: the
fellows lane (charmander H200, free) is the standing first lane; recent incidents
(#1739: ~12 consecutive habit-pinned `--backend gcp` dispatches burned ~40+ GPU-h of
credits while charmander sat idle; #2018 tracks the habit-guard mechanization) show
GCP GPU spend continuing despite the fellows-first default. The user now wants GCP
removed from GPU routing entirely, not merely deprioritized.

## Goal

No dispatch path — auto chain, explicit frontmatter pin, or `--backend gcp` CLI — may
provision a GCP **GPU** instance. CPU intents are OUT of scope (see Scope).

## Proposed change (refine in planning)

1. **Auto chain:** drop `"gcp"` from `DEFAULT_AUTO_LANE_ORDER`
   (`src/explore_persona_space/backends/router.py:706`) → `("fellows", "nibi", "fir", "mila")`,
   RunPod terminal rung unchanged. Decide whether `auto_lane_order()` /
   `EPM_AUTO_LANE_ORDER` validation should now REFUSE `gcp` loudly (like `runpod`) or
   accept-but-warn; refusing is more consistent with "removed as an option".
2. **Explicit pins:** an explicit `backend: gcp` frontmatter / `--backend gcp` dispatch
   with a GPU intent (`gcp.INTENT_TO_MACHINE` rows with `gpu_count > 0`) fails loud
   pre-launch with a typed error naming this policy — never silently rerouted.
3. **Failover legs that re-enter GCP GPU rungs** (the #1596/#1601
   queue-vanish/queue-timeout → GCP on-demand retry legs in `backend_poll.py` /
   `router.retry_gcp_ondemand_after_queue_vanish`) must not create new GCP GPU
   instances. Planner decides excise vs gate; the poller paths that only ACT ON
   existing GCP handles (crash persist, teardown, GCP→RunPod failover of an in-flight
   handle) should stay — they are cleanup, not provisioning.
4. **Tests:** update the pins — `tests/test_router.py::test_default_auto_lane_order_is_gcp_first`
   (pins the 5-lane tuple), the GCP ladder walk tests, and
   `test_runpod_is_last_rung_only_after_all_gcp_and_slurm_exhausted` — to the new
   contract; add a pin that the auto chain contains no gcp GPU rung and that an
   explicit gcp GPU pin refuses.
5. **Docs:** update CLAUDE.md § "Compute backends — multi-lane router" (the
   fellows-first bullet, the `--backend gcp` habit-guard bullet — now a hard removal),
   `.claude/rules/compute-backend-failover.md` (mark the GPU ladder/failover sections
   historical or scoped to in-flight handles), and the GPU intent → spec table if it
   references GCP machine mappings for GPU intents. Grep the workflow surface for
   `--backend gcp` / `backend: gcp` prescriptions
   (`grep -rn 'backend.*gcp' .claude/ CLAUDE.md docs/ --include='*.md'`) and update
   prescriptive hits.

## Scope / constraints

- **CPU intents KEEP their GCP lanes** (`cpu-small` / `cpu-mid` GCP E2 → RunPod CPU
  fallback; `cpu-bigmem` n2-highmem, which has NO RunPod equivalent — removing it
  would leave the >50 GB analysis lane laneless). The directive says "for GPUs".
  Assumption stated at filing; flag in the plan summary so the user can override.
- `sweep-8g-a100` / `sweep-8g-h100` / all wide-rung machinery are GPU → removed from
  selectability along with the rest.
- The GCP janitor (`gcp_audit.py` cron), crash-persist, and stale-instance reaping
  stay — they clean up, they don't provision.
- Rollback lever: this should remain a small, revertible commit (the fellows
  `available=False` precedent) — prefer a policy gate/constant over a deep excision
  of the ladder code, so re-enabling GCP later is a flag flip, not a code
  archaeology project.
- Workflow-surface + router change only; no experiment code.
- 0 GPU-h.

## Provenance

- origin: user-chat directive, session 2026-08-02
- verbatim: "remove GCP as an option for GPUs"
- verified-at-filing: `gcloud compute instances list --configuration=eps-gcp` →
  10 instances, 0 RUNNING (9 TERMINATED, 1 STOPPING) — no live GPU work disrupted
  (2026-08-02). `DEFAULT_AUTO_LANE_ORDER` confirmed at router.py:706 as
  `("fellows", "gcp", *DEFAULT_FREE_LANE_ORDER)`; lane-order test pin confirmed at
  tests/test_router.py:3193.
