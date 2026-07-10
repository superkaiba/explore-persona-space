---
title: 'workflow-fix: Trigger-dense artifact review guidance for rev'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fa6d44e23cd1
- daily-auto-filed
created_at: '2026-07-09T06:59:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Review-role subagents reviewing
  guard/security-adjacent bash content get killed by spurious usage-policy refusals
  (3 kills on #1058: code-reviewer x2, reconciler x1, +1 autocompact death) because
  the agent specs carry no standing guidance for trigger-dense artifacts, so each
  such review re-discovers the mitigations at the cost of 1-3 dead spawns.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1058.

## Goal

Give code-reviewer and reconciler standing guidance for reviewing trigger-dense security-adjacent artifacts so filter kills stop costing dead spawns.

## Workflow gap

- **Bug observed:** Review-role subagents reviewing guard/security-adjacent bash content get killed by spurious usage-policy refusals (3 kills on #1058: code-reviewer x2, reconciler x1, +1 autocompact death) because the agent specs carry no standing guidance for trigger-dense artifacts, so each such review re-discovers the mitigations at the cost of 1-3 dead spawns.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + ## Reviewing trigger-dense / security-adjacent artifacts
  + - NEVER write out gated command examples in generated text; reference by
  +   file:line, test-case id, or abstract description.
  + - Write the verdict file and post the marker FIRST; the chat summary is
  +   optional and comes last (a filter kill on the summary must not lose the verdict).
  + - Prefer orchestrator-provided excerpt files + windowed reads (<=120-line
  +   Read windows; never wholesale-read >800-line files).

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md, .claude/agents/reconciler.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.
- Consider relocating to a shared `.claude/rules/trigger-dense-review.md` (both specs are near their size-ratchet caps); update AGENT_SPEC_SIZE_GRANDFATHER only per the documented measured+<=1KB procedure if growth is unavoidable.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md, .claude/agents/reconciler.md
- origin: parked candidate on task #1058 at 2026-07-05T21:25:12Z

Verbatim parked note:

```
parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Logged + notified, NOT auto-routed.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/code-reviewer.md, .claude/agents/reconciler.md
bug_observed: Three review-role subagent turns on #1058 (code-reviewer x2, reconciler x1) were killed by spurious usage-policy API refusals while reviewing guard/security-adjacent bash content, plus one more died on autocompact from wholesale reads; each kill cost a full spawn and the round only completed after ad-hoc mitigations (verdict-by-reference discipline, post-marker-before-summary, Sonnet pin, pre-materialized diet input files with read budgets).
why_workflow_gap: The agent specs carry no standing guidance for reviewing trigger-dense security-adjacent artifacts (the guard's own vocabulary — destructive git command shapes, fail-open probes — reliably trips the filter when quoted in generated verdict text), so every such review re-discovers the mitigations at the cost of 1-3 dead spawns.
proposed_change: Add a "trigger-dense artifact review" section to code-reviewer.md and reconciler.md: reference findings by file:line / case id (never quote gated command literals in generated text), write + post the verdict marker BEFORE any closing summary, and accept orchestrator-provided pre-materialized excerpt files with explicit read budgets.
diff_sketch: |
  + ## Reviewing trigger-dense / security-adjacent artifacts
  + When the artifact under review is itself a guard/security surface whose
  + vocabulary (destructive command shapes, fail-open probes) can trip the
  + content filter:
  + - NEVER write out gated command examples in generated text; reference by
  +   file:line, test-case id, or abstract description.
  + - Write the verdict file and post the marker FIRST; the chat summary is
  +   optional and comes last (a filter kill on the summary must not lose the
  +   verdict).
  + - Prefer orchestrator-provided excerpt files + windowed reads (<=120-line
  +   Read windows; never wholesale-read >800-line files).
confidence: high
related_task: #1058
<!-- /workflow-fix-candidate -->

```
