---
title: 'workflow-fix: analyzer re-fold H1 retitle set-title sync'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a38bd2240cf0
- daily-auto-filed
created_at: '2026-07-09T06:59:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): analyzer.md''s same-issue
  follow-up re-entry step licenses retitling the clean-result H1 followed by set-body
  only — set_body preserves the frontmatter title, so every headline-moving follow-up
  round produces an H1/frontmatter divergence the #1110 verifier check FAILs at the
  next gate (a true positive, but a guaranteed one-bounce round-trip).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1110 (recursion-guarded workflow-fix session).

## Goal

Add 'then run `task.py set-title <N> "<new H1 text>"`' to the analyzer's same-issue follow-up re-entry step (mirror the main promotion sequence's set-body-then-set-title ordering); also update the sibling sites the candidate greps named in .claude/skills/issue/SKILL.md (~:6033-6037, ~:6234-6238).

## Workflow gap

- **Bug observed:** analyzer.md's same-issue follow-up re-entry step licenses retitling the clean-result H1 followed by set-body only — set_body preserves the frontmatter title, so every headline-moving follow-up round produces an H1/frontmatter divergence the #1110 verifier check FAILs at the next gate (a true positive, but a guaranteed one-bounce round-trip).
- **Why it is a workflow gap:** analyzer.md prescribes the H1 edit without the paired `task.py set-title` sync, so the two title surfaces diverge by construction on every headline-moving re-fold.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In analyzer.md § Same-issue follow-up re-entry:
- ... retitle the H1 if the headline moved, then `task.py set-body <N> --file ...`
+ ... retitle the H1 if the headline moved, then `task.py set-body <N> --file ...`,
+     then `task.py set-title <N> "<new H1 text>"` (keeps frontmatter == H1;
+     the #1110 verifier check FAILs the gate otherwise)

## Scope / surfaces

- Primary target: `.claude/agents/analyzer.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.
- Sibling sites: `.claude/skills/issue/SKILL.md` follow-up paths (candidate named ~:6033-6037 and ~:6234-6238; re-grep for `set-body` near the follow-up fold steps).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md
- origin: parked candidate on task #1110 at 2026-07-07T13:04:28Z

Verbatim parked note:

parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); LOGGED for the next orchestrator/human pass, NOT auto-filed.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/analyzer.md
bug_observed: The same-issue follow-up re-entry step licenses the analyzer to retitle the clean-result H1 ("retitle the H1 if the headline moved", analyzer.md ~:446-456) followed by set-body only — set_body preserves the frontmatter title, so every headline-moving follow-up round now produces an H1/frontmatter divergence that the new #1110 verifier check FAILs at the next gate (a true positive, but a guaranteed one-bounce round-trip).
why_workflow_gap: analyzer.md prescribes the H1 edit without the paired `task.py set-title` sync, so the two title surfaces diverge by construction on every headline-moving re-fold.
proposed_change: Add "then run task.py set-title <N> \"<new H1 text>\"" to the analyzer's same-issue follow-up re-entry step (and its main promotion sequence already does set-body then set-title — mirror that ordering).
diff_sketch: |
  In analyzer.md § Same-issue follow-up re-entry:
  - ... retitle the H1 if the headline moved, then `task.py set-body <N> --file ...`
  + ... retitle the H1 if the headline moved, then `task.py set-body <N> --file ...`,
  +     then `task.py set-title <N> "<new H1 text>"` (keeps frontmatter == H1;
  +     the #1110 verifier check FAILs the gate otherwise)
confidence: high
related_task: #1110
<!-- /workflow-fix-candidate -->

(Origin: codex-critic alternatives-lens Must-Fix on the #1110 plan, demoted to a non-blocking recommendation by the binding reconciler verdict — facts verified: set_body preserves fm title; analyzer.md + issue SKILL.md follow-up paths name only set-body. Also grep candidate sites: .claude/skills/issue/SKILL.md:6033-6037, :6234-6238.)

