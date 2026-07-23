---
title: 'workflow-fix: scope RunPod burn cap for shared team account'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a2e96fca981d
created_at: '2026-07-22T14:40:40Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on #779 2026-07-22: RunPod lane permanently
  refused — burn cap counts ~$2.7k/hr of unmanaged shared-team (fellows-cluster) pods
  vs $120/hr cap; 13/13 wait-for-capacity refusals'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed candidate raised on task #779 (emitting agent: /issue orchestrator, autonomous re-drive session 2026-07-22).

## Goal

Re-scope the RunPod fleet-burn pre-flight cap for the shared fellows team account so unmanaged foreign pods do not permanently refuse the RunPod lane.

## Workflow gap

- **Bug observed:** wait-for-capacity refused 13/13 provision attempts for pod-779 (2026-07-22 13:54-14:35Z): projected $2739-2789/hr vs cap $120.00/hr; the breakdown shows ~$2.7k/hr of the burn is ~80+ UNMANAGED RUNNING pods on the shared team account ("Anthropic 2-pod-*", "cluster-EUR-IS-pod-*", other fellows' pods). Under this scoping the RunPod lane can NEVER clear — every capacity cycle on #779 today burned a ~41-min wait loop against a structurally-unsatisfiable cap, then minted no_compute_available.
- **Why it is a workflow gap:** `current_account_hourly_burn()` deliberately counts ALL team pods (docstring: "the RunPod spending cap applies to ALL of them") — a premise that held when the team account contained only this project's pods. The account now shows the shared Anthropic-fellows cluster fleet (cf. the July 2026 fellows-cluster onboarding), so the premise drifted: the guard now gates OUR $4/hr provision on OTHER people's ~$2.7k/hr, permanently disabling the RunPod terminal rung (and with it the #783/#1116 failover paths).
- **Confidence (emitter):** medium — the OBSERVATION is certain; the right FIX is a spend-policy call (scope the burn to managed `pod-N`/`epm-issue-N` pods; or an `EPM_RUNPOD_BURN_SCOPE=managed|all` knob; or, if Thomas genuinely is billed for the whole team, raise the cap / narrow the X-Team-Id). The planner should flag `architectural: true` if it judges this a user-greenlight spend-policy change.
- verified-at-filing: `grep -rn current_account_hourly_burn scripts/ src/` → 5 hits in 2 files (scripts/runpod_api.py: def @992 + module doc @27; scripts/pod_lifecycle.py: import @99, docstring @1665, call @1672) (2026-07-22). Context read per clause (c): the all-pods scoping is DELIBERATE per the @992 docstring — this filing is premise-drift on a design decision, NOT a missed filter, and no landed fix exists (`git log --oneline --since='7 days ago' -- scripts/runpod_api.py scripts/pod_lifecycle.py` → 26e09df1fd / b1f376c574 / ffea76bbad, none touch cap scoping). Live evidence: the 2026-07-22T14:35Z epm:backend-selected marker on task #779 carries the 13-attempt local-cap refusal transcript.

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/runpod_api.py current_account_hourly_burn() (and/or the
# pod_lifecycle.py cap check):
- for p in live:                      # counts every RUNNING team pod
-     if (p.desired_status or "").upper() != "RUNNING": continue
+ scope = os.environ.get("EPM_RUNPOD_BURN_SCOPE", "managed")
+ for p in live:
+     if (p.desired_status or "").upper() != "RUNNING": continue
+     if scope == "managed" and not _is_managed_pod(p.name): continue
      breakdown.append((p.name or p.pod_id, rate))
# + keep an escalate-only WARN when unmanaged burn is huge, so visibility
#   of the shared account survives the scoping change.
```

## Scope / surfaces

- Primary target: `scripts/runpod_api.py, scripts/pod_lifecycle.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'current_account_hourly_burn\|_account_hourly_cap_usd' scripts/ src/ .claude/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; no dollar-budget caps added to experiment scripts (`tests/test_no_dollar_budget_caps.py` untouched — this is the provision guard, a sanctioned spend rail).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/runpod_api.py, scripts/pod_lifecycle.py
- fingerprint: a2e96fca981d

Surfaced prose (verbatim, from the #779 re-drive session): "the RunPod failover rung refused 13/13 wait-for-capacity attempts by the LOCAL fleet-burn pre-flight cap: projected $2739-2789/hr vs cap $120/hr — the shared RunPod team account carries ~$2.7k/hr of UNMANAGED pods (fellows-cluster 'Anthropic 2-pod-*' etc.), so the RunPod lane is effectively PERMANENTLY refused under the current cap scoping, not supply-dry."
