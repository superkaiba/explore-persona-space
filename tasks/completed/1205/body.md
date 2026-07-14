---
title: 'daily-fix: GCE workload verifies eval-JSON push landed'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f2add29fa03c
- daily-auto-filed
created_at: '2026-07-09T07:01:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): #825 round-8''s GCE workload
  git push silently failed — commit 87c9c73168 (73 eval JSONs) existed only on the
  self-deleting instance; upload-verifier caught it and the session rescued via git
  bundle inside the pre-poweroff window.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Make the GCE workload push step retry-then-fail-loud instead of silently declaring results with an unpushed commit.

## Workflow gap

- **Bug observed:** Transcript 313cb8c5 (issue-825 round 8) ~09:36-10:07Z: the workload's git push silently failed; commit 87c9c73168 with all 73 eval JSONs existed only on the GCP instance scheduled to self-power-off ~12:39Z with --instance-termination-action=DELETE; recovered via SSH git bundle + scp + --ff-only push.
- **Why it is a workflow gap:** Workload-side sibling of the piped-push masking class: the push step neither retried nor failed loud, and the run declared success with the git leg unpushed — permanent loss absent the verifier catch.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In the GCE workload composer push step: after push, `git rev-list origin/<branch>..HEAD --count` must be 0; retry push once on failure; non-zero after retry => fail the workload loud (crash-persist lane then preserves logs). Mirror the contract line in pod-side-reporting.md.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py, .claude/rules/pod-side-reporting.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py, .claude/rules/pod-side-reporting.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-B P1 (313cb8c5 ~09:36-10:07Z)
