---
title: 'workflow-fix: Lint: fix_sha in code-fix relaunch notes + D3 '
kind: infra
tags:
- wf-fix
- wf-fix-fp:deedcfcf5d37
- daily-auto-filed
created_at: '2026-07-09T06:59:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The #1081 crash-fix-relaunch
  contract (fix-engaged signal incl. fix-commit SHA; experimenter.md D3 disposition-conditional
  confirm) is prose-only — nothing mechanically checks that a code-row respawn brief
  / post-code-fix epm:run-launched note carries fix_sha=, nor pins the D3 wording
  against byte-trim regression.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1081.

## Goal

Give the #1081 crash-fix-relaunch fix-engaged contract mechanical teeth in workflow_lint.py (fix_sha presence + D3 anchor-phrase pin).

## Workflow gap

- **Bug observed:** The #1081 crash-fix-relaunch contract (fix-engaged signal incl. fix-commit SHA; experimenter.md D3 disposition-conditional confirm) is prose-only — nothing mechanically checks that a code-row respawn brief / post-code-fix epm:run-launched note carries fix_sha=, nor pins the D3 wording against byte-trim regression.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + def check_experimenter_d3_anchor(...):  # anchor-phrase pin, shape of
  +     # --check-stale-label-disposition (issue-963, workflow_lint.py)
  + def check_relaunch_fix_sha(...):  # events-aware: a post-code-fix
  +     # epm:run-launched note must carry fix_sha= (marker-sequence aware;
  +     # scope carefully — events.jsonl checks are unusual for this lint)
  + register both in the no-flags default bundle + mutation-visible tests

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- The fix_sha leg needs marker-sequence awareness (reading tasks/ events) which is atypical for workflow_lint; the planner may legitimately descope it to the anchor-phrase leg with a reasoned report.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #1081 at 2026-07-06T10:46:31Z

Verbatim parked note:

```
parked: EPM_WORKFLOW_FIX_SESSION — running under a workflow_fix_target: Provenance line; recursion guard logs, never routes (workflow-fix-on-bug § Recursion guard). Candidate (plan §12 + Claude r2 reviewer minor, merged): a mechanical lint giving the #1081 contract teeth — (a) a code-row respawn brief / post-code-fix epm:run-launched note must carry fix_sha= (needs marker-sequence awareness), and (b) an anchor-phrase check pinning the experimenter.md D3 disposition-conditional confirm wording against byte-trim regression. target_file: scripts/workflow_lint.py; confidence: medium; related_task: #1081. For the next non-workflow-fix orchestrator/human pass to file.
```
