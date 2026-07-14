---
title: 'workflow-fix: scope c25 arm-(a) exemption per fence'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4d7222ea6e09
- daily-auto-filed
created_at: '2026-07-12T06:52:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): c25 arm-(a) exemption is
  all-or-nothing — one doc-wide ''N/A — entities are content, not commands'' line
  exempts EVERY arm-(a) shell-tagged fence at once, so one legitimate entity-content
  fence plus N poisoned shell fences passes on a single declaration'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-11 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1258 (emitting agent: Alternatives critic, plan review round 1; parked under the recursion guard).

## Goal

Scope the verify_plan.py c25 arm-(a) exemption per-fence or by hit count (e.g. WARN when the exempted arm-(a) fence count > 1) instead of one doc-wide declaration exempting every fence.

## Workflow gap

- **Bug observed:** c25 arm-(a) exemption is all-or-nothing — one doc-wide `N/A — entities are content, not commands` line exempts EVERY arm-(a) shell-tagged fence at once, so one legitimate entity-content fence plus N poisoned shell fences passes on a single declaration.
- **Why it is a workflow gap:** verify_plan.py is the plan-gate verifier; a whole-doc exemption on a per-fence hazard class lets poisoned fences ride a single legitimate declaration. Distinct hardening the #1062 arm-(b) fix did not cover.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "entities are content" scripts/verify_plan.py` → 4 hits (2026-07-12), incl. line 3893 `if hits_a and _standalone_na_declared(plan, r"entities are content, not commands"):` — the exemption is a single doc-level check gating the entire `hits_a` list; no per-fence or count-scoped variant exists.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) Per-fence or hit-count-scoped exemption: when the standalone N/A declaration is present but `len(hits_a) > 1`, downgrade to WARN naming the exempted fence count (or require one declaration per fence), with red-green fixtures.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Also update the canonical escape-list documentation (verify_plan module docstring + adversarial-planner SKILL.md block) if the escape semantics change; add red-green fixtures in the verify_plan test file.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `uv run pytest tests/test_verify_plan.py` green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 4d7222ea6e09

Origin (parked prose candidate on #1258, 2026-07-11T10:21:11Z): "target_file: scripts/verify_plan.py. bug_observed: c25 arm-(a) exemption is all-or-nothing — one doc-wide 'N/A — entities are content, not commands' line exempts EVERY arm-(a) shell-tagged fence at once, so one legitimate entity-content fence plus N poisoned shell fences passes on a single declaration. proposed_change: per-fence or hit-count-scoped exemption (e.g. WARN when exempted arm-(a) fence count > 1); distinct hardening the #1062 arm-(b) fix did not cover. confidence: medium."
