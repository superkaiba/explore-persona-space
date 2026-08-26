---
title: Add a row/column-convention consistency check for plans that fit or consume
  a linear map
kind: infra
tags:
- workflow-fix
created_at: '2026-08-25T09:35:05Z'
has_clean_result: false
origin_prompt: 'Filed by the #2569 /issue orchestrator: all three Claude critic lenses
  APPROVEd a plan that mixed row-vector and column-vector operator action on a non-normal
  banked map (gate-metric Spearman 0.31, argmax agreement 2.5%), flipping registered
  H1/H2/H3/H8 verdicts; both Codex twins caught it and both said the check belongs
  on the workflow surface.'
workflow: v1
---
# Add a row/column-convention consistency check for plans that fit or consume a linear map

<!-- workflow-fix-candidate v1 -->

## Goal

Close a demonstrated review-surface gap: nothing on the plan-review surface
catches a plan that mixes row-vector and column-vector action on the same fitted
linear operator. Add a mechanical `verify_plan.py` check plus a critic lens item
so the error is caught at Phase 1.5.0 instead of surviving to a Phase-3 revise
round.

## The incident (#2569 plan v3, 2026-08-25)

#2569's plan registers the row-vector prediction path
`vhat = ((v - xmu)/xsd) @ W + ymu` (equivalently `vhat = v @ A + b` with
`A = diag(1/xsd) @ W`), then across five legs applies the operator on the LEFT of
column vectors — the transpose map:

- §3 H2 and §4 leg 2 step 1 register the gate metric as `W.T @ W`; the
  row-consistent metric is `W @ W.T`.
- §4 leg 1 step 3 classifies functional anatomy on RIGHT singular directions via
  `cos(A w_i, w_i)`; under row action the input directions are the LEFT singular
  vectors, with `u_i @ A = sigma_i v_i`.
- §4 leg 8 step 2 mines on `||A Dc_std||`; row-consistent is `||Dc_std @ A||`.
- §4 leg 3 steps 2/4 wire `E_f . A . D` and gate direction `A.T r`.

The plan was INTERNALLY INCONSISTENT, which is what makes this mechanically
detectable: §4's own convention block and the fixed-point solve `x*(I - A) = b`
are row-form, and leg-1 step 4's u->read / v->write pairing is already correct,
while the formulas above are column-form.

Measured impact on the actual banked operator
(`data/issue_2094/joint_transport/banked_maps/issue779_monitoring/n1m_readout/weights/L19/ridge.pt`,
W 3584x3584, non-normal — #1774 measured eigenvector condition number ~4,519):

- `||A A.T - A.T A||_F / ||A||_F^2 = 0.126` (materially non-normal)
- Spearman(`||x@A||`, `||A@x||`) over 4,000 random unit directions = **0.056**
- Spearman(`c.T W W.T c'`, `c.T W.T W c'`) over 4,000 pairs = **0.31**
- argmax-partner agreement between the two gate metrics = **5/200 = 2.5%**
- `|cos(u_1, v_1)| = 0.084`; top-8 `|cos(u_i, v_i)|` 0.08-0.24
- row identity `u_i @ A = sigma_i v_i` holds to 1.3e-15; the column form
  `A v_i ~= sigma_i v_i` is off by relative 1.31-1.35

So the plan's forms and the row-consistent forms rank directions and context
pairs almost independently, flipping the registered H1, H2, H3 and H8 verdicts.
Three independent probes (two reconcilers plus the orchestrator, different
statistics each) agreed.

## Why this is a workflow-surface gap, not just a plan bug

- `verify_plan.py` has NO orientation or convention check (grep for
  orientation / row-vector / column-vector / transpose returns only unrelated
  senses of "convention").
- `.claude/rules/critic-lens-reference.md` and `.claude/agents/critic.md` carry
  no row/column item.
- ALL THREE Claude critic lenses (Methodology, Statistics & Measurement,
  Alternative Explanations) returned APPROVE on this plan. Both Codex twins
  caught it independently, and both explicitly wrote that "this recurring
  row/column-orientation check belongs in a workflow-surface verifier." The
  Claude-side miss across three lenses is the evidence that no lens owns it.
- The class is recurring, not one-off: any plan that fits or consumes a linear
  map (`v_X -> v_Y` predictors, ridge/OLS readouts, persona-vector transports,
  operator atlases, SAE wiring products) can mix conventions, and CLAUDE.md's
  standing identity+bias / kNN mapping-baselines rule means such plans are
  common in this project.

## Proposed fix

1. **Mechanical check in `scripts/verify_plan.py`** (new check id, WARN-first
   then FAIL once precision is established): when a plan declares a
   row-vector prediction path (regex on `@ W`, `v @ A`, `x @ W`,
   `vhat = ... @ ...`), flag co-occurring column-action forms on the same
   operator — `W.T @ W` / `W.T W` used as an INPUT-space metric, `A x` /
   `A @ Dc` / `cos(A w_i, w_i)` / `A.T r` — and vice versa for a declared
   column path. The internal-inconsistency shape is the reliable trigger:
   FAIL/WARN only when BOTH conventions appear for the same symbol, which
   avoids penalizing a plan that consistently uses either one. Emit the
   offending line numbers and both forms.
2. **Critic lens item** (Statistics & Measurement, and cross-referenced from
   Methodology): for any plan that fits or consumes a linear map, verify the
   declared action side and check that every induced metric, singular/eigen
   direction family, and displacement norm is written in that convention.
   Add to `.claude/rules/critic-lens-reference.md` and the
   `.claude/rules/lens-coverage-map.md` ledger.
3. **Identity-assert convention for implementers**: a plan or implementation
   that consumes a banked map asserts the identities
   `||x @ A||^2 == x (A A.T) x.T` and `u_i @ A ~= sigma_i v_i`, and that
   reported statistics run through the artifact's own registered `apply_map`
   path rather than a re-derived product. Consider a
   `.claude/rules/` entry (or an addition to the existing
   `identity+bias / kNN mapping-baselines` rule family) so the assert is the
   default for map-consuming code.

## Acceptance

- A regression fixture reproducing #2569 v3's mixed-convention shape FAILs (or
  WARNs) the new check; a byte-equivalent fixture written consistently in
  either single convention PASSes.
- The new lens item appears in `critic-lens-reference.md` and the
  `lens-coverage-map.md` ledger, and `workflow_lint.py --check-lens-coverage`
  stays green.
- `.claude/rules/LESSONS.md` index updated if a new rule file is added
  (enforced by `workflow_lint.py --check-lessons-index`).

## Provenance

Filed by the #2569 `/issue` orchestrator during the Phase-3 union round,
2026-08-25. Source evidence: #2569 `events.jsonl` markers
`epm:progress` "Orientation Must-Fix VERIFIED NUMERICALLY against the banked
map", `epm:plan-critique-reconciled` (Statistics lens) and
`epm:plan-critique-reconciled` (methodology + alternatives union). #2569 itself
is fixing its own plan in v4 (blocker B1); this task fixes the SURFACE so the
next such plan is caught mechanically.
