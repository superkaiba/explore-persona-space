---
title: 'workflow-fix: c4 escape via _standalone_na_declared'
kind: infra
tags:
- wf-fix
- wf-fix-fp:791d8ecf6fd1
- daily-auto-filed
created_at: '2026-07-12T06:52:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): c4 (check_contrastive_negatives)
  N/A escape is a bare doc-global re.search(r"(?i)not a behavior[- ]implantation")
  over strip_fences at scripts/verify_plan.py:714 — any prose sentence containing
  the phrase self-escapes the check (same class as the c7 bug #1262 fixes; #1237''s
  NA_RE-prefix grep missed both).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-11 parked-candidate routing pass (Step C) from a FORMAL workflow-fix candidate block parked on task #1262 (emitting agent: round-1 Alternatives critic; parked under the recursion guard).

## Goal

Migrate the verify_plan.py c4 (check_contrastive_negatives) N/A escape from a bare doc-global regex to `_standalone_na_declared`, mirroring the #1237/#1262 pattern.

## Workflow gap

- **Bug observed:** c4's N/A escape is a bare doc-global `re.search(r"(?i)not a behavior[- ]implantation")` over strip_fences at scripts/verify_plan.py:714 — any prose sentence containing the phrase self-escapes the check (same class as the c7 bug #1262 fixed; #1237's NA_RE-prefix grep missed both).
- **Why it is a workflow gap:** verify_plan.py is the plan-gate verifier; a self-escaping check lets a bounced plan spuriously satisfy c4 (the #810 polarity). Mechanizable check: grep verify_plan.py for bare `re.search(r"(?i)not a` feeding `_pass` — the set must be empty or documented.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 're.search(r"(?i)not a behavior' scripts/verify_plan.py` → 1 hit at line 714 feeding `_pass` at line 715 (2026-07-12, post-#1262 merge) — still unfixed; #1262 migrated only c7.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- if re.search(r"(?i)not a behavior[- ]implantation", text):
+ if _standalone_na_declared(plan, r"not a behavior[- ]implantation"):
+ docstring/SKILL.md: add `N/A — not a behavior-implantation` (check 4) to the canonical escape lists
+ tests: mid-prose red fixture + standalone green fixture; floor 22->23
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- ADD the phrase to the canonical escape lists (verify_plan module docstring + adversarial-planner SKILL.md block, backtick-wrapped) — currently absent from both; red-green fixtures; bump the skillmd-pin extraction floor.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `uv run pytest tests/test_verify_plan.py` green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 791d8ecf6fd1

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_plan.py
bug_observed: c4 (check_contrastive_negatives) N/A escape is a bare doc-global re.search(r"(?i)not a behavior[- ]implantation") over strip_fences at scripts/verify_plan.py:714 — any prose sentence containing the phrase self-escapes the check (same class as the c7 bug #1262 fixes; #1237's NA_RE-prefix grep missed both).
why_workflow_gap: verify_plan.py is the plan-gate verifier; a self-escaping check lets a bounced plan spuriously satisfy c4 (the #810 polarity). Surfaced by the #1262 round-1 Alternatives critic (mechanizable check: grep verify_plan.py for bare `re.search(r"(?i)not a` feeding _pass — the set must be empty or documented).
proposed_change: migrate c4's escape to _standalone_na_declared(plan, r"not a behavior[- ]implantation") mirroring #1237/#1262; ADD the phrase to the canonical escape lists (verify_plan module docstring + adversarial-planner SKILL.md block, backtick-wrapped) since it is currently absent from both; red-green fixtures; bump the skillmd-pin extraction floor.
diff_sketch: |
  - if re.search(r"(?i)not a behavior[- ]implantation", text):
  + if _standalone_na_declared(plan, r"not a behavior[- ]implantation"):
  + docstring/SKILL.md: add `N/A — not a behavior-implantation` (check 4) to the canonical escape lists
  + tests: mid-prose red fixture + standalone green fixture; floor 22->23
confidence: high
related_task: #1262
<!-- /workflow-fix-candidate -->
