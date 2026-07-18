---
title: 'workflow-fix: artifact-reuse check (l) — instrument validity-domain transfer'
kind: infra
tags:
- wf-fix
- wf-fix-fp:836c079fd4fe
created_at: '2026-07-18T20:04:41Z'
has_clean_result: false
origin_prompt: 'analyzer #1417 prose follow-up: add validity-domain transfer check
  to artifact-reuse fitness checklist (fit825 GCV collapse incident)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix prose note
raised on task #1417 (emitting agent: analyzer, first-pass interpretation).

## Goal

Add a reuse-fitness checklist item to `.claude/rules/artifact-reuse.md`: a
validity-domain transfer check — before reusing a fit/analysis INSTRUMENT
(code artifact), verify the new data regime does not cross a validity
boundary the instrument's own docs/comments declare (n ranges, dof caps,
selection fallbacks / registered mitigations), and engage the declared
mitigations when it does.

## Workflow gap

- **Bug observed:** #1417 reused the frozen fit825 ridge instrument on
  judge-filtered row subsets (per-fold n_train < d=3584, the
  near-interpolation regime) without engaging the instrument's OWN documented
  mitigations (`GCV_DOF_CAP = 0.9`, `lambda_selection="inner-group-cv"`,
  documented at scripts/issue825_fit_cells.py lines 68-91 from the #1335
  incident). The GCV lambda selection collapsed on those subsets (held-out
  R² −0.6…−1.5 where supersets and matched-n subsamples fit at +0.3…+0.65),
  voiding the run's entire map-identity verdict layer — found only by the
  analyzer post-run.
- **Why it is a workflow gap:** the reuse fitness check (a)-(k) verified
  recipe identity, throughput (i), and parent lineage (k), but no item asks
  whether the NEW consumption regime crosses a validity boundary the reused
  instrument itself declares. Check (b) (valid measurement regime) is
  DV/question-scoped, not instrument-doc-scoped — it did not bind here.
- **Confidence (emitter):** high (the instrument's own docstring names the
  exact failure + mitigations; a checklist question would have caught it at
  plan time).
- verified-at-filing: `grep -niE 'validity[- ]domain|validity boundary|dof cap' .claude/rules/artifact-reuse.md` → 0 hits in the target (absence-of-guard claim; adjacent check (b) at line ~108 covers measurement-regime-for-the-question, not instrument-declared validity bounds — read in context, it does not implement this change); `git log --oneline --since='7 days ago' -- .claude/rules/artifact-reuse.md` → 5 commits, none touching validity-domain transfer (2026-07-18).

## Proposed change (candidate diff sketch — refine in planning)

+ New item under the reuse fitness checklist (sibling of (i) throughput):
+ **(l) validity-domain transfer (reused fit/analysis instruments):** read the
+ instrument's own docs/comments for DECLARED validity boundaries (n-vs-d
+ regimes, dof caps, selection fallbacks, registered mitigations) and check
+ the new consumption regime against them; crossing a declared boundary
+ requires engaging the declared mitigation (or a stated justification) in
+ the plan. A reused instrument consumed outside its self-declared domain is
+ a REVISE (methodology-baselines lens / consistency-checker cross-check).
+ Driving incident: #1417 x fit825 GCV collapse (#1335 mitigations unengaged).

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Grep the workflow surface for enforcement mirrors before editing
  (`grep -rln 'artifact-reuse' .claude/agents/planner.md .claude/agents/consistency-checker.md .claude/rules/critic-lens-reference.md CLAUDE.md`)
  and update the enforcement pointers (planner step 5 self-attestation,
  consistency-checker, Methodology lens item 9) if the item list is
  enumerated there; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 836c079fd4fe

Verbatim surfaced prose (analyzer, #1417 first pass): ".claude/rules/artifact-reuse.md: the reuse fitness check (a)-(k) verified recipe identity but not validity-domain transfer — the frozen fit825 instrument's own docstring bounds its GCV default to the n_train > D regime, and #1417 moved it to filtered subsets (n_train < 3584) without engaging the documented mitigations. A checklist item 'does the new data regime cross a validity boundary the reused instrument's own docs/comments declare (n ranges, dof caps, selection fallbacks)?' would have caught this at plan time."
