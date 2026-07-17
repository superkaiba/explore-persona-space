---
title: 'workflow-fix: gotchas entry — pilot timing gates measure at the sweep''s execution
  shape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:13733121358f
created_at: '2026-07-16T17:42:26Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1415 crash-fix round 1
  (pilot gate batch-1 vs batch-8 false-fire, bare rc=1 anonymous crash); see the failure-lesson
  block in the body'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1415 (emitting agent: experiment-implementer, crash-fix round 1).

## Goal

Add a gotchas.md entry: pilot timing gates must MEASURE at the sweep's execution shape (batch width / per-call structure), and a gate refusal must be a designed artifact-routed halt (report JSON + distinct rc), never a bare rc=1 the dispatcher reads as an anonymous crash.

## Workflow gap

- **Bug observed:** #1415's plan-mandated pilot timing gate measured s/sample at BATCH-1 shape (single context → serial per-draw generate calls) while the gated sweep runs B=gen_batch=8 chunks; memory-bandwidth-bound HF decode makes batch-1 per-sample cost ~B× the sweep's, so a correctly-derived 4.7 s/sample threshold false-fired (measured 15.67; true sweep shape ≈2), and the refusal exited bare rc=1 → classified as an anonymous crash ("no matching kill-report"), burning a full GCE launch cycle + a diagnosis round (att-20260716-160022, 2026-07-16).
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` has no entry on execution-shape-matched timing pilots or on designed-halt rc routing for pilot gates, and `.claude/rules/plan-compute-sizing.md`'s "MEASURED 1-cell pilot at PRODUCTION shape" (line ~194) does not spell out that PRODUCTION SHAPE includes the sweep's BATCH GEOMETRY for in-run timing gates — exactly the reading under which #1415's reviewed, plan-conformant pilot still measured the wrong shape. Future pilot-gated plans (the plan-compute-sizing pilot-gated basis is now standard) will re-hit this.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix validated in production — relaunch passed the gate at pilot_batch=8, ≈2 s/sample)
- verified-at-filing: `grep -rn -i "pilot" .claude/rules/gotchas.md .claude/rules/plan-compute-sizing.md` → 0 hits in gotchas.md (absence-of-guard claim — the 0-hit in-target result IS the evidence); 12 hits in plan-compute-sizing.md incl. line 194 "MEASURED 1-cell/1-unit pilot timing at PRODUCTION shape" whose context does not name batch geometry / execution shape (2026-07-16)

## Proposed change (candidate diff sketch — refine in planning)

+ .claude/rules/gotchas.md: new bullet "Pilot timing gates measure at the SWEEP's execution shape" —
+   batch-1 pilot vs batched sweep false-fires a correct s/sample threshold (bandwidth-bound decode:
+   per-step latency ≈ batch-independent, so batch-1 per-sample ≈ B× the sweep's); replicate pilot
+   inputs to B = the sweep's batch width (per-row param/delta stacks mirroring the sweep's call
+   contract), normalize per-sample by rows×draws, persist the measured batch in the pilot artifact;
+   a gate refusal writes a report JSON + exits a DISTINCT rc the dispatcher routes like other kill
+   criteria — never bare rc=1. Worked impl: scripts/issue1415_run_phase1.py phase_pilot +
+   _enforce_pilot_gate (rc=7 + pilot_gate_report.json) @ a369b06f46; incident #1415 att-20260716-160022.
+ (secondary, planner's call) .claude/rules/plan-compute-sizing.md ~line 194: sharpen "at PRODUCTION
+ shape" to "at PRODUCTION shape — including the phase's realized BATCH GEOMETRY / execution width".

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Sibling hit to reconcile (same pattern, plan-time surface): `.claude/rules/plan-compute-sizing.md` (§ per-cell fit-phase pilot basis)
- Grep the workflow surface before editing (`grep -rn -i 'pilot' .claude/rules/ CLAUDE.md`) and update every hit that states the pilot-basis rule without the execution-shape clause; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 13733121358f

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: phase-1 pilot timing gate (scripts/issue1415_run_phase1.py phase_pilot / _enforce_pilot_gate)
lesson: A pilot timing gate must MEASURE at the sweep's execution shape — this pilot ran serial batch-1 generates while the sweep runs B=8 chunks, and bandwidth-bound HF decode makes batch-1 s/sample ~B× the sweep's, so a correct threshold false-fired (15.67 measured vs 4.7 threshold; true sweep-shape ≈2 s/sample). Replicate the pilot input to B=gen_batch rows and normalize by rows×draws, and make the gate a designed artifact-routed HALT (report JSON + distinct rc), never an anonymous rc=1 the dispatcher reads as a crash.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
