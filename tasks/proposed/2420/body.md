---
title: 'Step 5a FAMILY_workflow misses workflow.yaml pin test tests/test_ensemble_review_cap.py
  — half-sync vintage skew failed 4 gate nodes on #2205'
kind: infra
tags:
- workflow-fix
- step5a-spec-freshness
created_at: '2026-08-20T07:26:10Z'
has_clean_result: false
parent_id: 2205
workflow: v1
---
## Goal

Add `tests/test_ensemble_review_cap.py` — and every other workflow.yaml-consuming pin test not already covered — to the Step 5a spec-freshness FAMILY_workflow map (`.claude/skills/issue/steps/09-step-5.md`), so syncing `.claude/workflow.yaml` from origin/main always carries the pin tests that assert its values. Acceptance: (1) FAMILY_workflow includes test_ensemble_review_cap.py (it reads `workflow.ensemble_review.round_cap_per_reviewer` and `pivot_criteria` from workflow.yaml); (2) a bounded audit enumerates other tests that read workflow.yaml data (grep for `workflow.yaml` / `load_workflow_yaml` / `WORKFLOW_PATH` in tests/) and adds any found to the family or records why not; (3) the SPECS list gains the new members so pass-1 dirty detection covers them.

## Why (incident)

#2205 Step 10d gate round 2 (2026-08-20): main flipped review round caps 5→10 (#2391) mid-run; the pre-gate Step 5a re-sync pulled cap-10 workflow.yaml into the issue-2205 worktree, but tests/test_ensemble_review_cap.py is in NO declared family, so its fork-era copy (asserting cap 5, `code_review_ensemble_cap_5_surface`) stayed behind and failed 4 nodes in the gate's TG leg (origin/main's copy asserts cap 10 and passes). This is exactly the half-sync vintage skew (#1824/#1860 class) the family-atomic design exists to prevent — the family map is simply missing a coupled member. FAMILY_workflow already carries tests/test_workflow_yaml.py for the same reason; test_ensemble_review_cap.py has the identical coupling.

## Provenance

Surfaced by the #2205 orchestrator at the Step 10d lint gate round 2 (session cmt0rstzvmuuoxw0u2g5m28sk); filed per .claude/rules/workflow-fix-on-bug.md. Distinct from #2416 (sibling-arm runtime-skew probe + 10d classify base-identity — different mechanism, same incident family).
