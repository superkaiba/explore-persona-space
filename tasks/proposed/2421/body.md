---
title: 'Step 10d lint-gate fences not contention-aware — 120s scratch-checkout + 900s
  lint fences crash healthy gates under #1962 fail-open concurrency'
kind: infra
tags:
- workflow-fix
- step10d-lint-gate
created_at: '2026-08-20T07:26:23Z'
has_clean_result: false
parent_id: 2205
workflow: v1
---
## Goal

Make the Step 10d pre-push lint-gate workload (`.claude/skills/issue/steps/18-step-10d.md`) and `scripts/step9c_baseline.py mapped-baseline` robust to fleet contention, which currently converts a healthy gate into a `crash` verdict: (a) the mapped-baseline scratch-creation `git checkout <merge-base>` has a fixed 120s subprocess timeout that fail-closes the whole TG leg when the shared repo is gc-backlogged or ≥3 gates run concurrently — give it a bounded retry (or a contention-scaled fence, or a pre-checkout `git prune`/gc.log health probe); (b) the two lint legs run under fixed 900s fences that return rc=124 under the same contention even though the completed invocations print "workflow_lint: PASS" — scale the fence by live gate count (`step9c_baseline.py probe --fleet` output) or serialize the two invocations per leg; (c) the #1962 fail-open launch (queue 2700s then run over cap) GUARANTEES ≥cap+1 concurrent gates exactly when the fleet is busiest — consider making fail-open wait-for-ANY-slot-with-longer-cap, or shrinking the launched gate's parallelism when over cap. Acceptance: a gate run under 3-concurrent-gate contention either completes within its fences or degrades to a retry/queued state — never a `crash` verdict from fence rc=124 / scratch-checkout timeout alone.

## Why (incident)

#2205 Step 10d round 2 (2026-08-20, the single sanctioned re-run): launched fail-open after the 2700s queue expired with gates live on #2201 + #2204. Verdict inputs GT_RC=0 BASE_RC=124 GATED_RC=124 TG_RC=1 TG_BASE_RC=0 TG_CRASH=yes. The scratch checkout of merge-base 33916785d96c timed out at 120.0s ("too many unreachable loose objects" gc warning active on the shared repo); one lint invocation per leg hit the 900s fence while its twin completed PASS. Cost: the re-run budget was consumed by infrastructure, routing an otherwise-green payload (Step 9c 6088 passed / 0 failed) to epm:merge-failed.

## Provenance

Surfaced by the #2205 orchestrator at the Step 10d lint gate round 2 (session cmt0rstzvmuuoxw0u2g5m28sk); filed per .claude/rules/workflow-fix-on-bug.md. Distinct from #2416 (classify-new-nodes base-identity) and from the FAMILY_workflow pin-test gap task filed alongside this one.
