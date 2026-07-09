---
title: 'workflow-fix: prefer explicit-PID kills in crash-fix step 2'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4cf9463b22fb
- daily-auto-filed
created_at: '2026-07-09T07:00:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): crash-fix-rounds.md step
  2 still kills by pattern pkill, leaving the narrow probe->kill TOCTOU where another
  session''s identical invocation starting between probe and kill gets TERMed undiscriminated.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #848 by a recursion-guarded workflow-fix session.

## Goal

Harden crash-fix step 2 to prefer explicit-PID kills (from the step-1 discriminated pgrep listing) over pattern pkill, closing the probe->kill TOCTOU window on the shared VM.

## Workflow gap

- **Bug observed:** Step 2's `pkill -TERM -f '<same pattern>'` re-matches the pattern at kill time, so a concurrent session's identical invocation that starts between the step-1 probe and the kill is TERMed without the step-1 discriminators ever seeing it.
- **Why it is a workflow gap:** crash-fix-rounds.md is the binding kill recipe for every crash-fix round on a VM shared by many concurrent sessions; the rule's own step 1 acknowledges cmdline identity is not ownership but step 2 does not apply that lesson.
- **Confidence (emitter):** low, Minor — non-blocking per both #848 round-1 reviewers (code-reviewer prose follow-up)
- **Triage evidence (2026-07-08):** Candidate 1 (agent-spec-size detection gap) is effectively CLOSED: check_agent_spec_size is in the workflow_lint no-flags default bundle (workflow_lint.py:7564; commit 2d7e9da240, 2026-07-01) and the Step 10d merge lint gate runs the no-flags bundle on both baseline+gated legs for every code-bearing merge (SKILL.md ~L8668); there is no git pre-commit hook at all to bundle into (.git/hooks holds only .sample files), so the proposed target does not exist and the merge-lane gate covers the regrowth class. Candidate 2 REMAINS: crash-fix-rounds.md step 2 (:178-179) still prescribes pattern `pkill -TERM -f '<same pattern>'`; step 1 gained per-match discriminators (explicit-PID for not-yours matches) but the probe->kill TOCTOU on identical concurrent invocations stands. No dedup; no retraction.

## Proposed change (candidate diff sketch — refine in planning)

```
- 2. **Kill** — `pkill -TERM -f '<same pattern>'`; wait ~10 s; ...
+ 2. **Kill** — prefer explicit PIDs from the step-1 listing:
+    `kill -TERM <pid>...`; wait ~10 s; `kill -KILL <pid>... 2>/dev/null || true`.
+    Pattern `pkill -f '<same pattern>'` only as fallback when the listing is
+    unusable — it re-matches at kill time and can TERM a process that started
+    after the probe.
```
NOTE: candidate 1 from the same parked note (pre-commit bundling of check_agent_spec_size) is already covered — see Triage evidence; do not re-implement.

## Scope / surfaces

- Primary target: `.claude/rules/crash-fix-rounds.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/crash-fix-rounds.md
- origin: parked candidate on task #848 at 2026-07-02T14:21:56Z

Verbatim parked note:

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — see workflow-fix-on-bug § Recursion guard (this session may not auto-route its own candidates). Two candidates for the next orchestrator/human pass:

1. [planner, confidence: medium] target_file: .git/hooks pre-commit config / scripts/workflow_lint.py bundling. Bug observed: code-reviewer.md + experiment-implementer.md regrew past their #838 ratchet caps SILENTLY — pre-commit runs only --check-asks, not check_agent_spec_size, so over-cap commits land undetected (found red on main during #848 planning; #848 remediated the two overages but not the detection gap). Proposed: bundle check_agent_spec_size into the pre-commit lint invocation.

2. [code-reviewer, confidence: low, Minor] target_file: .claude/rules/crash-fix-rounds.md. Bug observed: step 2's pattern pkill has a narrow probe→kill TOCTOU + generic-invocation exposure (another session's identical invocation starting between probe and kill). Proposed: one-clause hardening to prefer explicit-PID kills over pattern pkill in step 2. Non-blocking per both round-1 reviewers.
