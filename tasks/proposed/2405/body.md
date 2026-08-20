---
title: 'verify_plan: assert a registered kill criterion is satisfiable against its
  own cited artifacts'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-19T23:25:32Z'
has_clean_result: false
parent_id: 2388
origin_prompt: 'Surfaced by both Codex twins during #2388 round-2 plan review: the
  H3 kill criterion''s comparator condition was false at every registered anchor in
  the plan''s own cited banked files, so the kill could never fire; both twins proposed
  a recurring verifier check.'
workflow: v1
---
# Plan verifier: assert a registered kill criterion is SATISFIABLE against its own cited artifacts

## Goal

Add a `scripts/verify_plan.py` check that fails a plan whose registered kill criterion cannot fire
against the artifacts the plan itself cites. A kill criterion is a claim about what the data could
show; whether it is satisfiable is decidable at plan time from the cited files, and nothing
currently decides it.

## Why — the incident that generated this

Task #2388 plan v3 registered an H3 kill criterion of the form "the correctness gap's interval sits
at or below zero WHILE the banked sibling gaps stay positive at those same anchors", citing three
main-resident `eval_results/issue_1739/<behavior>/arm_results/all_arms_spearman.json` files at named
anchors.

Recomputing from those exact files, the required comparator condition is FALSE at every registered
anchor: best-mapped minus best-direct frozen rho is +0.0078, +0.0218, and MINUS 0.0068. The
criterion's second conjunct can never hold, so the kill cannot fire. Under a point-estimate reading
it is worse than inert — it is decided by noise, and would have licensed the plan's registered
narration from data showing the map helps neither dependent variable.

This survived a full round-1 adversarial review (five reviewers), a mechanical verifier pass at
0 FAIL 0 WARN, and three orchestrator verification passes, because every one of them reasoned about
the ESTIMATOR and none recomputed the criterion's comparator from the cited artifacts. It was found
only when a round-2 reviewer recommended a zero-cost check and the check was actually run.

The defect was introduced BY a correct fix: restricting the comparison to well-determined anchors
resolved a genuine estimator-validity problem, and in doing so relocated the comparison to the
label-rich end of the curve where the sibling effect had already decayed. That is the general shape
worth catching — a validity fix silently moving a comparison into a regime with no contrast.

Both the Codex methodology twin and the Codex alternatives twin independently proposed this check
belongs in the recurring verifier, each naming it mechanizable.

## Scope

Two checks, the second being the general one.

**Check 1 — kill-criterion satisfiability (the general check).** For a plan whose kill criterion
references a numeric comparator condition on a CITED artifact at NAMED anchors, recompute that
comparator from the artifact and FAIL when the condition is false at every selected anchor. The
plan must therefore state its comparator in a machine-resolvable form: the artifact path, the
anchors, the arm or field sets on each side, the row filter, and the aggregation. That statement
requirement is itself valuable — see check 2's sibling defect below.

**Check 2 — composition-cell pool-size identity.** For plans registering composition cells over a
fixed unlabeled pool, resolve the ACTUAL map keys plus persisted pool metadata per
(surface, family, pool-fraction) triple and assert every cell within a surface carries an identical
realized pool size. #2388 v3 had a shared 8,000-pair generic-only map serving a surface whose
registered pool was about 4,218, so the plan's own pre-fit assertion would have halted it at
pre-generation by construction.

## Design notes

The hard part of check 1 is not the recomputation, it is that plans currently state kill criteria in
prose. #2388's own round-2 review found the adjacent defect: the gap definition was unpinned, and
the classification chosen DECIDES THE SIGN — the same anchor reads +0.1081 or +0.0035 depending on
whether an MLP-readout arm counts as direct, and one behavior flips sign outright. So check 1 should
land in two stages: first require a machine-resolvable comparator block for any plan registering a
numeric kill criterion against a cited artifact (a WARN while adoption spreads, then FAIL), and only
then recompute and assert satisfiability.

Prefer WARN-only for check 1's satisfiability arm on first landing. A plan may legitimately register
a criterion whose comparator is currently false when the plan's own work is what will change it —
that case must be distinguishable from #2388's, where the comparator was false in banked data the
plan never intended to recompute. The discriminator is whether the cited artifact is an INPUT the
plan re-derives or a fixed reference; the check should read that from the comparator block rather
than guessing.

Do not fold this onto the existing selection-symmetric-null or estimator-validity checks. Those ask
whether a measurement is sound. This asks whether a registered DECISION can fire. #2388 passed the
former and failed the latter.

## Acceptance

Fixture plans reproducing both #2388 v3 shapes — a kill criterion whose comparator is false at every
cited anchor, and a composition cell set whose realized pool sizes differ within a surface — each
caught by its check, with a matching satisfiable/consistent plan passing cleanly. Both checks
registered in the no-flags default run. A fixture for the legitimate case above (comparator false in
an artifact the plan itself re-derives) must NOT fire.

## Provenance

Surfaced during #2388 round-2 plan review by the Codex methodology and Codex alternatives twins
independently, each proposing it as a recurring verifier check. Filed per the workflow-fix-on-bug
protocol's surfaced-prose clause. Evidence: #2388 markers v17 (the finding), v18 (the corrected
cost), v19 (the staging constraint), v20 (the statistics lens verdict incl. the sign-dependence of
the unpinned gap definition).
