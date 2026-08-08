---
title: 'issue-1310 body correction: per-turn instruct anti-prediction is a ridge under-regularization
  artifact, not a near-duplicate-context effect'
kind: infra
tags:
- needs-human
created_at: '2026-07-31T00:14:33Z'
has_clean_result: false
parent_id: 1310
origin_prompt: 'User-directed nd-estimator-audit inline round on #1310/#1639 (2026-07-30):
  ''recompute held-out R2 under (a) published ambient pure-GCV, (b) inner-group-CV,
  (c) reduced PCA basis, (d) forced lambda ... which published claims move enough
  to matter (>0.05 R2 or sign/verdict changes)''. The audit refuted a promoted #1310
  Takeaway; filed per the inline record-integrity duty.'
workflow: v1
---
## Overview / Motivation

A user-directed 0-GPU inline estimator audit on [#1310](https://eps.superkaiba.com/tasks/1310)
(label `nd-estimator-audit`, 2026-07-30, commit `a0ad4cea19`) REFUTES the causal
attribution in one of #1310's promoted `## Takeaways` bullets and requires a
prose correction to that promoted body. Filed per the inline-round
record-integrity duty (CLAUDE.md § Routing, "Inline estimator-validity +
record-integrity duties" item 3: filing is the presumption for anything touching
a bolded Takeaway; a classification flip stays user-only).

Sibling: task #1887 hardens the shared fit-core selector DEFAULTS. This task is
the RECORD half — correcting an already-promoted body — and touches no code.

## Goal

Apply a prose correction to #1310's promoted body: the per-turn instruct
"anti-prediction" it attributes to within-scene near-duplicate contexts is a
ridge-lambda under-regularization artifact, and the dof cap the body credits as
the mitigation is insufficient at per-turn row counts.

## The refuted claim (verbatim from #1310's promoted `## Takeaways`)

> - Round 1's per-turn instruct anti-prediction (−0.10 to −0.19, swap inverted)
>   was a within-scene near-duplicate-context fold artifact — real at the
>   per-turn grain, not a property of the character map.

The same attribution carries the `### The per-turn instruct anti-prediction is
confined to mid layers and traces to within-scene near-duplicate contexts`
result heading, and the `### Aggregating each prefill scene to one point turns
every instruct cell positive and null-clearing in both models` heading credits
scene aggregation with the fix.

## The refuting evidence

Audit artifacts (committed on `main` @ `a0ad4cea19`):
`eval_results/issue_1310/nd_estimator_audit/corrections_table.{json,md}` (16
per-cell JSONs), figures `figures/issue_1310/nd_audit_published_vs_corrected.png`
+ `nd_audit_selector_spread.png`, code
`scripts/issue1310_nd_estimator_audit.py` + `scripts/issue1310_nd_audit_report.py`.

Substrate: the same persisted onpolicy prefill store, layer 19, the SAME fold
assignment (`fit825._cv_folds`, seed 0, K=5) and the SAME Gram-space ridge the
committed fits used. Every published capped-GCV value was reproduced EXACTLY
before any other read (16/16 reproduction gates, |delta| = 0.0000 at 4 dp).

Per-turn instruct prefill cells at layer 19 — published capped-GCV vs the
inner-group-CV selector (`fit825.LAMBDA_SELECTION="inner-group-cv"`, 4 inner
GROUP folds, the deployed-generalization criterion, no test-fold information):

| cell | n | published | inner-group-CV | delta |
|---|---|---|---|---|
| `onpolicy_instruct_Wren` | 1798 | −0.1783 | +0.2760 | +0.454 |
| `onpolicy_instruct_HELIOS` | 1800 | −0.0996 | +0.3134 | +0.413 |
| `onpolicy_instruct_Dana` | 1738 | −0.1888 | +0.2512 | +0.440 |
| `onpolicy_instruct_Vex` | 1798 | −0.1762 | +0.2366 | +0.413 |

All four sign-flip, at the PER-TURN grain, on the SAME folds, with the SAME
within-scene near-duplicate contexts present. The anti-prediction therefore
cannot be a property of the per-turn grain or of near-duplicate contexts.

Mechanism, pinned by the selected lambdas (grid `np.logspace(-2, 4, 13)`):

- Ambient pure-GCV pins lambda at the 0.01 grid FLOOR in 15/16 audited cells and
  reads −1.56 to −5.51 on the per-turn family (the #1345 degeneracy class,
  reproducing the range #1310 already quarantined at
  `eval_results_onpolicy_gcvdegenerate/`).
- The `GCV_DOF_CAP = 0.9` mitigation rescues lambda only to 100 at per-turn row
  counts, because 0.9 × n_train ≈ 1,026 is a LOOSE constraint there.
- Inner-group-CV selects lambda 3,162–10,000.
- The published per-turn values equal the forced-lambda = 100 read EXACTLY
  (e.g. `onpolicy_base_Wren` +0.1140 published = +0.1140 at lambda 1e2), and the
  forced 1e3/1e4 reads (+0.18 to +0.37) bracket the inner-group-CV reads — so
  the published per-turn number IS "lambda = 100", and lambda = 100 is
  demonstrably ~30× too small.
- At the scene-aggregated row count (n = 300, n_train = 240) the same cap is
  BINDING (0.9 × 240 = 216), so it forces lambda to 3,162–10,000 — which
  coincides with inner-group-CV's principled choice. That is exactly why the
  aggregated family is unaffected.

So scene aggregation did not fix the instruct cells; adequate regularization
did. Aggregation reached that regularization incidentally, by making the dof cap
binding at the smaller n.

## Scope — what is NOT refuted (do not over-correct)

- The scene-aggregated Takeaway ("base 0.13–0.22, instruct 0.23–0.40") is
  CONFIRMED: inner-group-CV reproduces all 8 aggregated cells to <= 0.0025.
- #1639 stands. Its within-cell ceilings ARE the aggregated cells, which the
  audit confirms robust; no correction is indicated there. (Named residual: the
  audit re-fit the within-cell R^2 only, not #1639's transfer-matrix / lattice /
  operator fits — those inherit the same n = 300 capped-lambda regime the audit
  found robust, but were not themselves re-selected.)
- The #1310 headline claim (a fixed-label fiction character supports a
  character-specific context→dialogue map) is NOT threatened. Every corrected
  read moved UP or stayed equal; the audit found no cell where a principled
  selector read LOWER than published.
- The run-2 SCRIPT-format cells (published 0.106–0.148 base / 0.188–0.253
  instruct, ambient pure-GCV, no `gcv_dof_cap` field) are INDETERMINATE, not
  refuted: their activation store was lost with its instance, so they could not
  be recomputed at 0 GPU-h. Direction-of-error evidence says they are, if
  anything, understated.

## Proposed change (prose only — no Takeaway classification flip)

Via `task.py set-body` on #1310, keeping `runs.classification` untouched
(user-only):

1. Rewrite the third `## Takeaways` bullet to attribute the per-turn instruct
   anti-prediction to ridge under-regularization at n_train < d rather than to
   within-scene near-duplicate contexts, citing the audit artifacts.
2. Fold a correction sentence into the `### The per-turn instruct
   anti-prediction ...` result prose (v4 has no separate corrections heading),
   and note in the aggregation result that the mechanism is regularization
   reached incidentally through the smaller n, not the scene grain itself.
3. Extend the method-caveat bullet: the dof cap is insufficient at per-turn row
   counts (rescues lambda only to 100) and binding only at n = 300.
4. Add the audit artifacts + commit `a0ad4cea19` to the `**Repro:**` footer.
5. Consider retitling the H1 only if the user judges the headline moved; the
   corrected reads strengthen rather than weaken it, so the default is to leave
   the title and confidence tag alone.

Whether the corrected per-turn values become the QUOTED per-turn numbers in
#1310's body, or are reported as an audit addendum beside the published ones, is
a judgement call for the plan — the audit's own arms are all layer-19 single-layer
reads, not a full 28-layer re-fit with re-run nulls.

## Constraints / invariants

- Body/prose only. Do NOT run `task.py promote`, do NOT flip
  `runs.classification`, do NOT edit `## Takeaways` bullet COUNT out of the 3–6
  spec band.
- `verify_task_body.py --issue 1310` must PASS after the edit; `set-body`
  WITHOUT `--snapshot` (the body is already a clean-result and
  `original-body.md` exists).
- The audit's numbers are read from the committed
  `corrections_table.json` at `a0ad4cea19` — re-grep them at compose time
  rather than re-typing from this body.
- No new compute: 0 GPU-h.
