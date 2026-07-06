---
title: 'workflow-fix: vectorize rule covers permutation/bootstrap draw loops'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ab249f65d8d
created_at: '2026-07-02T06:31:21Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 778: extend vectorize-many-cell-fits.md
  to many-draw closed-form loops (perm/bootstrap null batteries); observed 4.1 s/draw
  serial perm_null_draws, ~15h projected vs 1h plan'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #778 (emitting agent: orchestrator, /issue 778 session).

## Goal

Extend `.claude/rules/vectorize-many-cell-fits.md` (and the planner/critic
enforcement pointers) to explicitly cover many-DRAW closed-form statistical
loops — permutation / bootstrap / null-draw batteries — not only
gradient-descent fits.

## Workflow gap

- **Bug observed:** #778's stage-two null battery ran a serial per-draw
  permutation loop (`perm_null_draws`: two full pool-mean passes over a
  1783×28×3584 float64 pool PER DRAW) at ~4.1 s/draw — ~15h projected vs the
  plan §8 estimate of 1h — after the round raised n_draws 200→1000. The fix
  (batched subset-sum GEMM over all draws) is a ~70× win, exactly the rule's
  50-100× class.
- **Why it is a workflow gap:** the rule's title, diagnostic signature, and
  plan-time enforcement (`planner.md` §9 compute-character carve-out,
  `critic.md` Methodology lens 10(iii)) are all framed around
  *gradient-descent fits* (MLP/AdamW/LOCO). A closed-form many-draw loop with
  identical overhead-bound economics does not pattern-match, so neither the
  planner nor the critic ensemble flagged the serial loop at plan time, and
  the CLAUDE.md always-on bullet's examples also omit draw loops.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/vectorize-many-cell-fits.md
+ frontmatter description: add "many-draw statistical loops (permutation /
+   bootstrap / null-draw batteries)" alongside GD fits
+ ## The diagnostic signature: add "a per-draw Python loop that re-reduces a
+   large fixed pool (means/sums/covariances) every draw — precompute the pool
+   reduction once and batch all draws as one GEMM (subset-sum identity)"
+ ## Worked incidents: add #778 (perm_null_draws, 4.1 s/draw, 5x draw raise,
+   ~70x vectorized win)
CLAUDE.md "Vectorize / parallelize compute by default" bullet
+ add "never a serial per-draw permutation/bootstrap loop over a large fixed
+   pool" to the example list
```

## Scope / surfaces

- Primary target: `.claude/rules/vectorize-many-cell-fits.md`
- Secondary: `CLAUDE.md` (the always-on vectorize bullet's example list),
  `.claude/agents/planner.md` §9 / `.claude/agents/critic.md` Methodology lens
  10(iii) if their wording needs the same widening.
- Grep before editing: `grep -rln 'many-cell' .claude/ CLAUDE.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; lessons index stays in sync.
- This session runs under the standard recursion-guard contract.

## Provenance

- workflow_fix_target: .claude/rules/vectorize-many-cell-fits.md
- fingerprint: 0ab249f65d8d

Surfaced prose (verbatim): the orchestrator of /issue 778 observed the
serial null-battery loop mid-round, measured 4.1 s/draw via py-spy, and
routed the code fix through an experiment-implementer round; this task is
the rule-file widening so the NEXT plan catches it at plan time.
