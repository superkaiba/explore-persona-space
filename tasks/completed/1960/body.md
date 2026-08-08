---
title: 'workflow-fix: verify_plan lattice parser — negated-existence atom'
kind: infra
tags:
- wf-fix
- wf-fix-fp:63fb274bb974
created_at: '2026-08-01T06:26:37Z'
has_clean_result: false
origin_prompt: 'Statistics critic on #1946 plan v5, mechanizable:yes concern 4 — extend
  verify_plan.py''s verdict-lattice parser to recognize a negated-existence atom so
  family-negation conjuncts parse instead of WARNing.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `mechanizable: yes` Statistics-critic finding raised on task #1946 (emitting agent: critic, Statistics & Measurement lens, plan v5 review).

## Goal

Extend `scripts/verify_plan.py`'s verdict-lattice parser to recognize a negated-existence atom (e.g. "no <family> contrast BH-significant with positive delta") so legitimate family-negation conjuncts parse instead of WARNing.

## Workflow gap

- **Bug observed:** verify_plan WARN on #1946 plan v5: label 'Collapse-preserved' did not fully parse — "predicate token(s) outside every recognized atom: 'no'" — despite the lattice being disjoint+exhaustive (the Statistics critic verified coherence by hand against the battery's actual output fields).
- **Why it is a workflow gap:** the lattice-coherence check is supposed to make decision gates machine-checkable; a common, legitimate predicate shape (negated existence over a contrast family — "no X is BH-sig-positive") falls outside the recognized atom grammar, so any plan using it draws a WARN and loses the FAIL-capable mechanical check, pushing the verification back onto per-round critic labor.
- **Confidence (emitter):** medium
- verified-at-filing: live reproduction — `uv run python scripts/verify_plan.py --issue 1946` (2026-08-01) emitted the WARN verbatim ("predicate token(s) outside every recognized atom: 'no'"); target confirmed present: `grep -n 'atom' scripts/verify_plan.py` hits the lattice-atom grammar region (the WARN's own wording), 1 target file.

## Proposed change (candidate diff sketch — refine in planning)

```
+ # In the lattice predicate parser: recognize a negated-existence atom
+ #   NO_EXIST := ("no"|"zero") <family-ref> <metric-ref> [comparator]
+ # equivalently count(<family> where <predicate>) == 0, and treat it as a
+ # well-formed conjunct in the DISJOINT-and-exhaustive partition check.
```

Suggested canonical machine form the parser should also accept: `count(refusal-family contrasts with bh_significant AND delta_adj > 0) == 0`.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'recognized atom' scripts/`) and update every hit; list them in the plan. Add/extend the pin test for the lattice parser (`tests/test_verify_plan.py`) with a negated-existence fixture.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 63fb274bb974

Origin finding (verbatim, Statistics critic on #1946 plan v5): "WARN 3 cosmetic residue (mechanizable: yes). If a future round touches the plan, restating the family conjunct in verify_plan's recognized atom grammar (e.g., `count(refusal-family contrasts with bh_significant AND delta_adj > 0) == 0`) would make the lattice machine-checkable; a 1-line predicate-token extension to the verify_plan lattice parser (recognize a negated-existence atom) would also close this class. Not blocking — coherence is verified above."
