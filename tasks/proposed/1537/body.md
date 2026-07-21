---
title: 'daily-held: enforce body presence on wf-fix filings'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-19T07:07:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 3): A wf-fix filing without
  --body/--body-file lands frontmatter-only silently (#1517, commit 14f2952cab verified);
  the spawned session hits the Step 0b empty-body gate. The #1173 WARN half is already
  landed in file_infra_task.py; the residual refusal touches the task.py new CLI contract.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1517 (emitting agent: /issue orchestrator, candidate-block;
parked under the recursion guard, routed by the 2026-07-18 /daily Step C
parked-candidate sweep). ROUTE 3 — NEEDS HUMAN GREENLIGHT: the candidate
proposes that `task.py new` REFUSE certain filings, which is a task.py CLI-
contract change (architectural per workflow-fix-on-bug.md § Architectural
greenlight); park at plan_pending / spawn without --auto.

## Goal

`file_infra_task.py` (and `task.py new` when tags include `wf-fix` or the
title carries a `WF_FIX_TITLE_PREFIXES` prefix) refuses — or loudly warns
on — a filing with an absent/empty body, so a wf-fix filing can never land
frontmatter-only.

## Workflow gap

- **Bug observed:** a wf-fix filing without `--body`/`--body-file` lands
  frontmatter-only silently (task #1517, commit 14f2952cab); the spawned
  autonomous /issue session then hits the Step 0b empty-body block-and-fail
  gate (the #1517 session recovered by reconstructing the body from the
  parent's markers).
- **Why it is a workflow gap:** the workflow-fix protocol REQUIRES the
  body-file template (verified-at-filing line, Provenance
  workflow_fix_target/fingerprint) but neither filer enforces body PRESENCE
  for a wf-fix-tagged / workflow-fix:-prefixed filing.
- **Confidence (emitter):** medium
- verified-at-filing: context READ of `scripts/file_infra_task.py` L221-247 — the #1173 `_warn_missing_wf_fix_provenance` backstop ALREADY loudly warns when a wf-fix-tagged filing's body is absent or lacks `workflow_fix_target:` (fires on `body_text is None`, i.e. the absent-body case), so the "loudly warns" HALF of the proposal is landed in file_infra_task.py; the RESIDUAL is (i) upgrading warn→refusal there and (ii) any enforcement in `task.py new` — the task.py refusal is the CLI-contract change driving the route-3 classification; `git rev-parse --verify --quiet '14f2952cab^{commit}'` → resolves (14f2952cab8b7e39d776b8b02babfb054f60e481); `git log --oneline --since='7 days ago' -- scripts/file_infra_task.py` → 4 commits (280b80b058, 9c53b54b81, 15e33e6a3f, 49507ec746 — advisory arms, none enforces body presence) (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
+ if is_wf_fix and not (args.body or args.body_file):
+     sys.exit('wf-fix filing requires --body-file (workflow-fix-on-bug.md § Body-file template)')
```

Descope option for the planner (would make this non-architectural): enforce
the refusal ONLY in `scripts/file_infra_task.py` (a wrapper, not the task.py
public CLI) and leave `task.py new` untouched — surface this fork to the user
at the plan gate.

## Scope / surfaces

- Primary target: `scripts/file_infra_task.py, scripts/task.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'WF_FIX_TITLE_PREFIXES\|_warn_missing_wf_fix' .claude/ CLAUDE.md scripts/ src/explore_persona_space/task_workflow.py`);
  list every hit in the plan.

## Constraints / invariants

- ARCHITECTURAL GATE: the `task.py new` refusal changes a public CLI
  contract — the plan carries `architectural: true` (or the session is
  spawned without --auto) and parks at plan_pending for the user.
- The must-succeed filing half of file_infra_task.py must not regress for
  non-wf-fix filings; `daily_drive_filings.py`'s driver contract (it always
  passes a body file) stays compatible.
- `scripts/workflow_lint.py --check-asks` passes; `tests/test_workflow_fix_dedup.py`
  and the filer tests pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: scripts/file_infra_task.py, scripts/task.py
- fingerprint: 5eb71ecdf906

<!-- workflow-fix-candidate v1 -->
target_file: scripts/file_infra_task.py, scripts/task.py
bug_observed: a wf-fix filing without --body/--body-file lands frontmatter-only silently (task #1517, commit 14f2952cab); the spawned autonomous /issue session then hits the Step 0b empty-body block-and-fail gate (this session recovered by reconstructing the body from the parent's markers).
why_workflow_gap: the workflow-fix protocol REQUIRES the body-file template (verified-at-filing line, Provenance workflow_fix_target/fingerprint) but neither filer enforces body presence for a wf-fix-tagged / workflow-fix:-prefixed filing.
proposed_change: file_infra_task.py (and task.py new when tags include wf-fix or the title carries a WF_FIX_TITLE_PREFIXES prefix) refuses — or loudly warns on — a filing with an absent/empty body.
diff_sketch: |
  + if is_wf_fix and not (args.body or args.body_file):
  +     sys.exit('wf-fix filing requires --body-file (workflow-fix-on-bug.md § Body-file template)')
confidence: medium
related_task: #1517
<!-- /workflow-fix-candidate -->
