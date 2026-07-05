---
title: 'workflow-fix: reuse-parity gates calibrate vs the artifact''s committed per-behavior
  values; weaker-than-constant = WARN not HALT'
kind: infra
tags:
- wf-fix
created_at: '2026-07-05T14:53:47Z'
has_clean_result: false
origin_prompt: '#813 3-launch-halt incident (2026-07-01/02); watch-session held candidate,
  filed at #813 park per plan'
workflow: v1
---
## Overview / Motivation
Auto-filed by the workflow-fix protocol from the #813 launch-halt sequence (watch-session orchestrator; user-approved filing pass at the #813 park).

## Goal
Reuse-parity gates must calibrate their thresholds against the reused artifact's OWN committed per-behavior values, and a diagnostic whose failure mode is "the reused artifact is WEAKER than a constant" defaults to WARN-with-adjudication, not run-abort HALT.

## Workflow gap
- **Bug observed:** #813 burned 3 launch-halts (2026-07-01 23:17 numeric write-ratio 0.00903 < a global 0.01 floor; 23:54 marker-install-below-1nat behavioral gate; v3 retry) + 2 crash-fix review rounds (~1.6h of 8xH100) on parity gates whose constants were never calibrated per behavior — while #537's committed G_cells/write-ratios (em 0.1729 == #667's committed value) held the ground truth that ended the loop the moment a supervisor read them. The band-stopped marker adapters (lr 5e-6) legitimately sit near any generic floor.
- **Why it is a workflow gap:** artifact-reuse.md's fitness check mandates verifying the artifact but says nothing about CALIBRATING validation-gate thresholds against the artifact's committed reference values, nor about HALT-vs-WARN severity for weaker-than-constant diagnostics. Third incident in the parity-gate family (#723 L4-precision, #746 enforce_eager); #813 inherited those fixes yet re-discovered the calibration half.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)
- `.claude/rules/artifact-reuse.md`: add a "Reuse-validation gate calibration" subsection:
  + a gate validating a reused artifact MUST derive its threshold from the artifact's own committed per-behavior reference values (cite file+field, e.g. #537 G_cells / committed write-ratios), never a global constant; marker/band-stopped artifacts named as the canonical near-floor case;
  + severity default: a diagnostic that can only fail by the artifact being WEAKER than expected is WARN + analyzer-adjudication, not HALT; HALT reserved for apply-path breakage (wrong adapter, gauge violation);
  + cross-ref crash-fix-rounds.md (declare fix-engaged signal before relaunching on a gate change).

## Scope / surfaces
- Primary target: `.claude/rules/artifact-reuse.md`
- Grep first: `grep -rn "parity" .claude/rules/ .claude/agents/planner.md .claude/agents/critic.md` and align with gotchas.md's existing parity-probe entries (#723/#746) without duplicating them.

## Constraints / invariants
- Workflow-surface only; workflow_lint passes; consistent with planner §11 grounding rule.

## Provenance
- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: (wrapper)
