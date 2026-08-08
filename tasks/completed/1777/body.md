---
title: 'workflow-fix: verify_plan WARN check for /workspace sentinels on unpinned
  auto lane'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f6666b92ee5b
created_at: '2026-07-28T22:46:05Z'
has_clean_result: false
origin_prompt: 'Methodology critic prose follow-up on #1775 plan v3: verify_plan.py
  has no sentinel-lane disposition check (plan-compute-sizing § Sentinel-signaling
  workloads; #608 class)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1775 (emitting agent: critic — Methodology lens, surfaced-prose follow-up).

## Goal

Add a WARN-level `verify_plan.py` check (next free id, e.g. `c42_sentinel_lane`) that flags a plan declaring `/workspace/...` sentinel paths while leaving the backend on the unrestricted auto lane, per `.claude/rules/plan-compute-sizing.md` § Sentinel-signaling workloads.

## Workflow gap

- **Bug observed:** plan v3 for #1775 declared `/workspace/logs/issue-1775-p*.done` sentinel paths in its `phase_outputs` yaml while §9 declared "no `backend:` pin → auto lane (fellows H200 → GCP)". On the auto chain's SLURM rungs the sentinel write crashes the workload (`mkdir -p /workspace/logs`, the #608 class); on the fellows FIRST rung `/workspace` exists but nothing drains the sentinels (silent marker loss, the CLAUDE.md fellows SENTINEL HAZARD). Only a Methodology critic caught it, as a Must-Fix.
- **Why it is a workflow gap:** `.claude/rules/plan-compute-sizing.md` § Sentinel-signaling workloads makes the disposition mandatory (pin a `/workspace`-contract lane, OR make the driver lane-portable and state "no sentinel dependence — auto-safe" in §9), but `scripts/verify_plan.py` has no mechanical check for it — the exact class verify_plan exists to catch pre-critic.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -in "sentinel" scripts/verify_plan.py` → 0 hits in the named target (2026-07-28) — an absence-of-guard claim, the 0-hit in-target result IS the evidence (not a text-matching-guard subclass: the proposed check is new surface, not a claimed-existing predicate). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` run at compose time (see Provenance note below).

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_sentinel_lane(plan: str, kind: str) -> CheckResult:
+     # Trigger: a `sentinel: /workspace/...` line (phase_outputs yaml) or a
+     # /workspace/logs/issue-<N>-*.done path anywhere in the plan.
+     # PASS when: the plan pins `backend: gcp|runpod` (prose or dispatch
+     # command `--backend gcp|runpod`), OR a standalone
+     # `no sentinel dependence — auto-safe` declaration line exists.
+     # Else WARN citing plan-compute-sizing § Sentinel-signaling workloads
+     # (#608 SLURM crash; fellows silent-loss hazard).
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Also update `tests/test_verify_plan.py` (positive + negative + escape-phrase cases) and the adversarial-planner SKILL.md N/A-escape enumeration if a new escape phrase is introduced.
- Grep the workflow surface for the pattern before editing (`grep -rln 'Sentinel-signaling' .claude/ scripts/`) and keep the check's vocabulary aligned with `.claude/rules/plan-compute-sizing.md`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; WARN-level only (never FAIL — the disposition is sometimes legitimately prose-satisfied in different words).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: f6666b92ee5b

Surfaced prose (verbatim, Methodology critic on #1775 plan v3):
"Workflow-surface gap, concrete and recurring (#608 class): `scripts/verify_plan.py` has no check that a plan declaring `/workspace/...` sentinel paths in `phase_outputs` either pins a `/workspace`-contract lane or carries the 'no sentinel dependence — auto-safe' §9 line required by `.claude/rules/plan-compute-sizing.md` § Sentinel-signaling workloads. A WARN-level surface check (regex over the plan's phase_outputs + backend line) would have caught Must-Fix 1 mechanically."
