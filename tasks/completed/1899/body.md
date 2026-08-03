---
title: 'workflow-fix: fellows QoS fallback ladder (high-eur first, then normal-eur/low-eur)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:08fd580e9d2b
created_at: '2026-07-30T21:31:53Z'
has_clean_result: false
origin_prompt: 'user directive 2026-07-30 verbatim: ''make sure it uses high-eur always''
  + ''but also tries the others im allowed to use'' — QoS ladder within the fellows
  lane before falling through to GCP'
workflow: v1
---
## Overview / Motivation

Auto-filed from a verbatim user directive in PM chat (2026-07-30): "make sure it uses high-eur always" + "but also tries the others im allowed to use". Companion to #1896 (capture-7b intent rows) and #1898 (sentinel-drain arm) under the standing fellows-utilization directive ("make sure the free H200s and the anthropic fellows cluster get used as much as possible"). No parent task (chat-mode; candidate logged to `.claude/cache/workflow-fix-events.jsonl`).

## Goal

Add a QoS fallback ladder to the fellows lane: every dispatch submits under `high-eur` FIRST (the pin is unchanged as the preferred tier); on a queue-park timeout (PENDING > the fellows park cap) the lane cancels and RE-SUBMITS under the next QoS the user is granted — `normal-eur`, then `low-eur` — each with its own bounded park, and only after the whole ladder park-fails does the chain advance to GCP. Optionally route `debug`-intent jobs to `dev-eur` (priority 200000 > high-eur's 100000, 1-day wall, 8-GPU cap) if the cluster rules permit.

## Workflow gap

- **Bug observed:** the fellows ClusterConfig pins a single static `qos="high-eur"` (`backends/slurm.py:342`) and the router has NO QoS retry (`grep -c qos backends/router.py` → 0): a PENDING park-timeout on high-eur advances straight to paid GCP even though the user is granted three more QoS tiers on the same free cluster (`sacctmgr show assoc user=superkaiba`: dev-eur, high-eur, low-eur, normal-eur — live-read 2026-07-30).
- **Why it is a workflow gap:** the likeliest high-eur park-fail cause is the per-user 16-GPU `MaxTRESPU` cap on that QoS, and unverified hypothesis — verify at plan time: SLURM `MaxTRESPU` binds per-QoS-per-user, so `normal-eur` (gres/gpu=16, 7d) and `low-eur` (no GPU cap listed, 14d, lowest priority) carry SEPARATE headroom — the ladder would unlock additional free H200 capacity precisely when high-eur is self-capped, not merely re-queue at lower priority. Even if the caps turn out shared, low-eur's uncapped/14-day tier is still a strictly-better-than-GCP fallback for park-tolerant jobs.
- **Confidence (emitter):** high (the gap); medium (the per-QoS-cap payoff — labeled unverified above)
- verified-at-filing: `sed -n 339,375p src/explore_persona_space/backends/slurm.py | grep qos` → exactly one static `qos="high-eur"` row (presence claim, 1 hit); `grep -c "qos" src/explore_persona_space/backends/router.py` → 0 hits (absence claim — no ladder/retry logic); live `sacctmgr show assoc user=superkaiba -P format=User,Account,QOS` on charmander → `dev-eur,high-eur,low-eur,normal-eur` granted; live `sacctmgr show qos` → priorities/walls/caps as stated (all 2026-07-30).

## Proposed change (candidate diff sketch — refine in planning)

```
# backends/slurm.py: ClusterConfig gains qos_ladder: tuple[str, ...] = ()
#   fellows row: qos="high-eur", qos_ladder=("normal-eur", "low-eur")
#   (dev-eur for intent=="debug" ONLY if cluster rules allow — planner
#    re-reads the 8 hard cluster rules enumerated in #1609's body/plan)
# backends/router.py: the fellows park loop, on park_cap_exceeded, walks
#   qos_ladder — scancel the parked job, re-render sbatch with the next
#   --qos, re-submit, re-park (each rung bounded by the same fellows park
#   cap) — before recording the lane attempt as failed and advancing to GCP.
#   Explicit `backend: fellows` pins keep the #1609 pinned-lane exemption
#   (return the live PENDING handle, no ladder walk) unless the plan
#   decides pins should walk too.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py` (ClusterConfig field + fellows row), `src/explore_persona_space/backends/router.py` (park/ladder walk in the free-lane submit path)
- Tests: `tests/test_slurm_backend_render.py` (a `--qos=normal-eur` render variant; byte-identical renders for non-fellows lanes), `tests/test_router.py` (ladder walk + attempt-marker shape; DEFAULT_AUTO_LANE_ORDER untouched)
- Grep before editing: `grep -rn "high-eur\|qos" src/explore_persona_space/backends/ tests/test_router.py tests/test_slurm_backend_render.py CLAUDE.md` and update every hit needed; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- **high-eur stays FIRST unconditionally** (the user's "uses high-eur always" half): the ladder only fires AFTER a high-eur park-timeout; no dispatch ever starts on a lower tier.
- unverified hypothesis — verify at plan time: whether the fellows cluster rules (the 8 hard rules enumerated in #1609) permit production jobs on normal-eur/low-eur/dev-eur; dev-eur in particular may be reserved for dev jobs (the cluster's `D_`-prefix dev-job convention) — if so, ship the ladder without the dev-eur arm.
- Each ladder rung is bounded by the same fellows park cap (`EPS_FELLOWS_QUEUE_WAIT_SECONDS`, default 600s) so worst-case lane latency is cap × n_rungs — state the realized worst case in the plan and consider a smaller per-rung cap for rungs 2+ (fast-start directive: a job that cannot start quickly on fellows should reach GCP promptly).
- Non-fellows lane renders stay byte-identical (the #1609 snapshot-test contract); DEFAULT_AUTO_LANE_ORDER untouched; RunPod-never-in-auto untouched.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: 08fd580e9d2b

User directive (2026-07-30, verbatim): "dispath 1896 and 1898 asap. make sure it uses high-eur always" + "but also tries the others im allowed to use". Live evidence: `sacctmgr show assoc user=superkaiba` → QOS `dev-eur,high-eur,low-eur,normal-eur`; `sacctmgr show qos` → high-eur (prio 100000, 7d, gpu=16/user), normal-eur (prio 50000, 7d, gpu=16/user), low-eur (prio 10000, 14d, no GPU cap listed), dev-eur (prio 200000, 1d, gpu=8).
