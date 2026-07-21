---
title: 'workflow-fix: selection-inherited bootstrap CI clause for selection-symmetric-nulls'
kind: infra
tags:
- wf-fix
- wf-fix-fp:18d3d40cd916
created_at: '2026-07-17T08:43:26Z'
has_clean_result: false
origin_prompt: 'interpretation-critic #1434 r1 prose follow-up: add selection-inherited
  bootstrap-CI clause to selection-symmetric-nulls.md'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1434 (emitting agent: interpretation-critic, round 1).

## Goal

Add a clause to selection-symmetric-nulls.md: a bootstrap CI reported at a max-selected axis position must be the selection-inherited CI (or both frozen and inherited, labeled)

## Workflow gap

- **Bug observed:** Interpretation reported the frozen-layer bootstrap CI at a max-over-28-layers-selected layer while the selection-inherited CI in the same JSON spans zero; the rule covers null bands but not bootstrap CIs
- **Why it is a workflow gap:** `.claude/rules/selection-symmetric-nulls.md` mandates per-draw same-selection for NULL BANDS at a free-axis max, but is silent on BOOTSTRAP CIs at the same selected position — so a frozen-layer CI can legitimately-looking overstate certainty at a winner's-curse-selected layer (caught live on #1434: frozen [-0.949, -0.467] vs selection-inherited [-0.957, +0.866]).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "selection-inherited\|bootstrap" .claude/rules/selection-symmetric-nulls.md` → 1 hit (line 218, a per-draw-bootstrap mention in the null-band leg; no clause governs bootstrap CIs at a max-selected position — absence-of-guard evidence) (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)
+ In the "band vs ceiling / reporting" section: "A bootstrap CI reported at a
+ max-selected axis position MUST be the selection-inherited CI (the per-draw
+ re-selection rides inside each bootstrap resample), or BOTH CIs shown with
+ explicit frozen-at-<axis> vs selection-inherited labels; a frozen-only CI at
+ a selected position is a REVISE (worked example: #1434 install-grid rho)."

## Scope / surfaces

- Primary target: `.claude/rules/selection-symmetric-nulls.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'selection-inherited' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/selection-symmetric-nulls.md
- fingerprint: 18d3d40cd916

Verbatim surfaced prose (interpretation-critic round 1, task #1434): "Follow-ups (orchestrator should consider): `.claude/rules/selection-symmetric-nulls.md` covers null BANDS at a max-selected axis but not bootstrap CIs — add a clause that a bootstrap CI reported at a max-selected axis position must be the selection-inherited CI (or both, labeled), the exact frozen-vs-inherited gap found here."
