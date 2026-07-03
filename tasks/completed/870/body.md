---
title: 'workflow-fix: compute-deviation gains mandatory vectorize-first lever'
kind: infra
tags:
- wf-fix
- wf-fix-fp:90f7d165ec86
created_at: '2026-07-02T16:10:05Z'
has_clean_result: false
origin_prompt: 'User (2026-07-02): "A lot of recent transcripts show that parallelization/vectorization
  is not properly being used for experiments which is leading to slowdowns -- Read
  the recent transcripts, figure out why this is happening, and propose workflow improvments
  to fix this" -> after the 7-proposal audit report: "Make all seven changes. Figure
  out the best way"'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a user-approved transcript audit (2026-07-02 PM-chat session; user: "A lot of recent transcripts show that parallelization/vectorization is not properly being used for experiments... propose workflow improvements" → "Make all seven changes"). Emitting agent: orchestrator, via 5 parallel transcript-mining subagents over all EPS sessions since 2026-06-26.

## Goal

Add a mandatory vectorize/parallelize-first lever to `pivot_criteria.compute_deviation_over_2x` so an overhead-bound compute deviation triggers a vectorization round before power-descope or `continue_as_is` is eligible.

## Workflow gap

- **Bug observed:** `epm:compute-deviation` auto-resolved `continue_as_is` at 10x (#722, 2026-06-29T22:04Z: "Ratio: 10x over plan §9 ... Action: continue_as_is ... Power-preserving auto-descope is structurally impossible") and at 80x (#763, 2026-07-01T01:21Z: "projected_wall_h: 24, planned_wall_h: 0.3, ratio: 80 ... action: continue_as_is ... the fit phase held the GPU pod for ~24 wall-h idle ... not a science blocker"). In both cases a vectorized rewrite later delivered ~1000-4400x (38h → 1.5 min for #722; 1237h → 16 min/behavior for #763), but the save came from a stranded-branch graft (#722) and a manual user override (#763, "USER OVERRIDE ... REVERSED by Thomas. VECTORIZE the fit; do NOT run the serial version"), not from the gate.
- **Why it is a workflow gap:** `workflow.yaml § pivot_criteria.compute_deviation_over_2x` enumerates descope dimensions as seeds/framings/cells-per-stratum only, then escalates to the two-option `continue_as_is` vs `accept_descope_to_<X>_with_caveats` gate (id=12). "Vectorize/parallelize the implementation" is not in the option set, so in autonomous mode (no human at the gate) the session picks between eating the full serial cost and cutting statistical power — the one lever that preserves BOTH cost and power is structurally unavailable. The gate also has no clause about a GPU pod held idle by the serial phase (#763: idle 1xH100 ~16h; the "CPU-only phases don't hold GPU pods" rule was in force and unapplied).
- **Confidence (emitter):** high (marker-quoted evidence in both transcripts; contrast cases #810 and #761 where a vectorize round cost ~1-2h and saved 200+h).

## Proposed change (candidate diff sketch — refine in planning)

In `workflow.yaml § pivot_criteria.compute_deviation_over_2x`:

```
  Auto-action (REVISED, vectorize-first):
+ 0. Signature check: if the deviation basis matches the overhead-bound
+    signature (serial Python loop over cells/folds/draws/rows, batch-1
+    model forwards, per-fold factorization, per-row IO/compression — the
+    vectorize-many-cell-fits trigger), the REQUIRED first lever is a
+    vectorization/parallelization round: spawn experiment-implementer with
+    .claude/rules/vectorize-many-cell-fits.md + an equivalence gate vs the
+    serial oracle (the #761/#810/#834 pattern). Descope and continue_as_is
+    are NOT eligible until this round has run or the signature check has
+    recorded a negative finding.
+ 0b. If the serial phase holds a GPU pod, stop/release the pod while the
+    fix round runs (CLAUDE.md "CPU-only phases don't hold GPU pods").
+ 0c. continue_as_is at ratio >= 5x additionally requires a recorded
+    finding on the marker that the workload is genuinely FLOP-bound at
+    the hardware roofline (not batchable) — cite the arithmetic.
  1. (existing) attempt auto-descope ... (unchanged, now second lever)
```

Mirror the same ordering in `.claude/skills/issue/SKILL.md` step 5.bis(a) and in the gate id=12 `condition:` text (the gate now fires only when vectorize-round AND descope both fail).

## Scope / surfaces

- Primary targets: `.claude/workflow.yaml` (§ pivot_criteria + gates.conditional id=12), `.claude/skills/issue/SKILL.md` (step 5.bis)
- Cross-link (do not edit for this concern): `.claude/agents/experiment-implementer.md` compute-deviation reporting is being revised by open task #869 (per-call cost grounding) — keep the two consistent; this task owns the RESOLUTION policy, #869 owns the DETECTION basis.
- Grep before editing: `grep -rln 'compute_deviation' .claude/ CLAUDE.md scripts/` and update every hit expressing the resolution option set.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes (gate id=12 keeps its `<!-- gate: -->` anchor); ruff on touched files passes; workflow.yaml stays consistent with CLAUDE.md and SKILL.md.
- The 2-option cap on the escalation gate is preserved (the vectorize round is an auto-action lever, not a third gate option).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/workflow.yaml, .claude/skills/issue/SKILL.md
- fingerprint: 90f7d165ec86

<!-- workflow-fix-candidate v1 -->
target_file: .claude/workflow.yaml, .claude/skills/issue/SKILL.md
bug_observed: epm:compute-deviation auto-resolved continue_as_is at 10x (#722) and 80x (#763) over plan because pivot_criteria.compute_deviation_over_2x offers only power-descope (seeds/framings/cells) or continue_as_is — vectorize/parallelize is not a lever — and the serial fit held an idle 1xH100 ~16h
why_workflow_gap: the pivot's option set structurally excludes the one lever that preserves both cost and statistical power, so autonomous sessions eat 10-80x serial burns rather than pivot
proposed_change: add a mandatory vectorize-first lever to compute_deviation_over_2x: overhead-bound deviations require a vectorization round before descope or continue_as_is is eligible, continue_as_is at >=5x requires a recorded FLOP-bound finding, and a GPU pod held by the serial phase is released during the fix round
diff_sketch: |
  + 0. overhead-bound signature => REQUIRED vectorization round (implementer
  +    + equivalence gate vs serial oracle) before descope/continue eligible
  + 0b. serial phase holding a GPU pod => release pod during fix round
  + 0c. continue_as_is at >=5x requires recorded FLOP-bound finding
  1. (existing) auto-descope ... (second lever)
confidence: high
related_task: n/a (user-chat transcript audit, 2026-07-02)
<!-- /workflow-fix-candidate -->
