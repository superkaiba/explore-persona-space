---
title: 'daily-fix: pre-persist plan header retitle stops c40 churn'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ca895df19f35
- daily-auto-filed
created_at: '2026-07-28T06:59:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): task.py new-plan-version
  assigns v(K+1) by filename but nothing aligns the plan''s ''# Plan v<K>'' header;
  verify_plan c40 fires only AFTER persist, so each fix costs another persisted version
  (2 sessions, ~4 extra persist+verify cycles on 2026-07-27)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Sessions 998eb54e (#1715, v2->v3->v4 loop) + 56131e5e (#1734), 2026-07-27.

## Goal

Stop the c40 header-version WARN churn: align the plan header with the assigned version BEFORE persisting.

## Workflow gap

- **Bug observed:** #1715 looped plan v2->v3->v4 because each `new-plan-version` persisted a file whose `# Plan v<K>` header self-declared the PRIOR version; `verify_plan.py` c40 fires only post-persist, so each fix costs another version. #1734 hit the same (v2 with a 'Plan v1' header).
- **Why it is a workflow gap:** the persist recipe has no pre-persist header-alignment step (`grep -c 'c40' .claude/skills/adversarial-planner/SKILL.md` -> 0) and the alternative (task.py-side auto-rewrite) does not exist.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'c40' .claude/skills/adversarial-planner/SKILL.md` -> 0 hits, compose time; `grep -n c40 scripts/verify_plan.py` -> present (the check exists only post-persist).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/adversarial-planner/SKILL.md` (persist recipe): pre-persist, `ls tasks/<status>/<N>/plans/` to derive the next version and retitle the header — or standardize on the version-neutral `# Plan — task #<N>` form. ALTERNATIVE (planner's call): auto-rewrite a leading `# Plan v<K>` header in `scripts/task.py cmd_new_plan_version` at persist time (behavior change to task.py — weigh against the doc-only form).

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner/SKILL.md`
- Optional: `scripts/task.py` (cmd_new_plan_version) if the auto-rewrite form wins.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ca895df19f35

- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md
