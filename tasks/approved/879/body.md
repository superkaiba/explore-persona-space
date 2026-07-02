---
title: 'workflow-fix: verify_plan.py p-floor attainability check for empirical-null
  gates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:76131138e355
created_at: '2026-07-02T18:00:35Z'
has_clean_result: false
origin_prompt: 'codex-critic statistics lens on #816 plan v5: registered p<=0.05 BH
  gate over families with n_draws=2/5 (p-floors 1/3, 1/6) is structurally unattainable;
  verifier should catch this mechanically'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #816 (emitting agent: codex-critic, statistics lens, plan v5
round 1).

## Goal

Add a p-floor attainability check to `scripts/verify_plan.py`: FAIL when a
plan's registered empirical-null success gate requires any null family to pass
p ≤ alpha while that family's p-floor (1/(n_draws+1)) exceeds alpha.

## Workflow gap

- **Bug observed:** #816 plan v5 registered an Exp-5 SUCCESS gate "real |r| >
  every honest null family's 97.5th-pct at one-sided empirical p ≤ 0.05
  surviving BH" while the ladder included cross-trait (n_draws=2, p-floor 1/3)
  and PCA (n_draws=5, p-floor 1/6) families — the gate is structurally
  unattainable. verify_plan.py c8 checked criteria PRESENCE and passed the
  plan; only the Codex statistics critic caught it.
- **Why it is a workflow gap:** the verifier owns mechanical plan checks;
  attainability of a registered empirical-p gate is computable (parse family
  n_draws + registered alpha; compare p_floor to alpha) and recurs in every
  null-battery / permutation-test plan.
- **Confidence (emitter):** high (mechanizable: yes per the critic verdict).

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_empirical_gate_attainability(plan_text):
+     # find registered gates of form "p <= ALPHA" tied to null families with
+     # declared n_draws; p_floor = 1/(n_draws+1); FAIL if p_floor > ALPHA for
+     # any family the gate requires to pass; WARN if p_floor == min attainable
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for sibling checks before editing
  (`grep -rln 'p_floor\|empirical p' scripts/verify_plan.py .claude/`) and add
  a matching test in `tests/test_verify_plan.py`.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff
  passes; tests added for the new check (positive + N/A escape).
- This is NOT architectural (adds a check; no schema/CLI contract change).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 76131138e355

Origin (verbatim critic prose): "mechanizable: yes; check every family included
in the registered BH/test set, read n_draws, compute p_floor=1/(n_draws+1), and
fail if p_floor exceeds the registered alpha while the family is required to
pass p<=alpha. This belongs in a recurring workflow-surface verifier for
registered empirical-null corrections." (epm:plan-critique-codex v1
lens=statistics, task #816, 2026-07-02)
