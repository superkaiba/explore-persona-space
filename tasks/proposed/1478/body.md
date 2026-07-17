---
title: 'workflow-fix: audit warns on sidecar-less embedded figures (silent check skip)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f2a581eed8d4
created_at: '2026-07-17T16:23:52Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1434 r3 formal candidate: audit_clean_results_body_discipline.py
  silently skips figure-text checks for sidecar-less embedded figures'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal candidate block raised on task #1434 (emitting agent: clean-result-critic, round 3).

## Goal

WARN (or FAIL the opaque-code check) when a body-embedded figures/issue_<N>/*.png at a cited SHA has no .meta.json sidecar at that SHA, so sidecar-less figures surface instead of silently skipping the figure-text checks

## Workflow gap

- **Bug observed:** The audit's figure-text checks (opaque config codes, figure-text staleness, plotted-value drift) read only `.meta.json` sidecars; task #1434's three po figures shipped without sidecars and were silently skipped, letting `neg_sp_ph4`-class slug tick labels pass the mechanical gate (9 sidecars checked vs 12 embedded figures).
- **Why it is a workflow gap:** A body-embedded figure lacking a sidecar bypasses every rendered-text check with no warning — the check's coverage silently shrinks to whichever figures happen to carry sidecars.
- **Confidence (emitter):** high
- verified-at-filing: `grep -cn "meta" scripts/audit_clean_results_body_discipline.py` → hits present but no embedded-figure-without-sidecar coverage warning exists anywhere in the file (absence-of-guard claim; the live instance: clean-result-critique v3 on #1434 caught slug labels the audit passed) (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

+ embedded = body_figure_paths_at_pins(body)
+ no_sidecar = [p for p in embedded if not blob_exists(pin, p.replace('.png', '.meta.json'))]
+ if no_sidecar: warn(f"figure-text checks skipped {len(no_sidecar)} sidecar-less figure(s): {no_sidecar}")

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'meta.json' .claude/ scripts/audit_clean_results_body_discipline.py`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: f2a581eed8d4

<!-- workflow-fix-candidate v1 -->
target_file: scripts/audit_clean_results_body_discipline.py
bug_observed: The audit's figure-text checks (opaque config codes, figure-text staleness, plotted-value drift) read only `.meta.json` sidecars; task #1434's three po figures shipped without sidecars and were silently skipped, letting `neg_sp_ph4`-class slug tick labels pass the mechanical gate (9 sidecars checked vs 12 embedded figures).
why_workflow_gap: A body-embedded figure lacking a sidecar bypasses every rendered-text check with no warning — the check's coverage silently shrinks to whichever figures happen to carry sidecars.
proposed_change: WARN (or FAIL the opaque-code check) when a body-embedded `figures/issue_<N>/*.png` at a cited SHA has no `.meta.json` sidecar at that SHA, so sidecar-less figures surface instead of skipping.
confidence: high
related_task: #1434
<!-- /workflow-fix-candidate -->
