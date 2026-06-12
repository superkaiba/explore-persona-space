---
name: Follow-up plans inherit mislabeled parent baselines — verify per-source gate outcomes against the parent's stats JSON
description: Amendment/follow-up plans can inherit a proposer's wrong classification of which parent conditions "failed"; success criteria keyed to a "previously-failed" set can be half-pre-satisfied at baseline, guaranteeing a partial false PASS
type: feedback
---

Rule: when a follow-up/amendment plan keys a success criterion to a set of
"previously-uninformative / previously-failed" conditions, independently re-derive
that set from the parent's own per-condition stats artifact (e.g.
`eval_results/issue_<N>/<followup>/concordance_stats.json` `informative` flags),
NOT from the plan's or the followup-scope's prose.

**Why:** #480 followup-2 (band-stopped-anchor-rerun): the plan (inheriting the
proposer's scope text) called qwen_default/kindergarten_teacher "uninformative"
in round 1, but the round-1 artifact records `informative: True` for both
(9/23 and 6/23 nonzero cells, passing the same ≥5/≥3 gate the plan reuses), and
the parent body itself says "two mid-variance sources are individually null."
P1 ("≥3 of the 4 currently-uninformative-or-marginal sources pass the gate")
was therefore ~half-pre-satisfied at baseline — a PASS could over-credit the
recipe with informativeness the parent anchors already had, and the
"masked-by-out-of-regime-anchors" hypothesis was already part-contradicted by
in-regime nulls the plan mis-summarized.

**How to apply:** for any plan whose hypothesis says "condition X failed last
time because <regime defect>", open the parent's stats JSON and check (a) the
pre-registered gate outcome for X, and (b) whether the gate-passing conditions
were null — "in-regime and null" is prior evidence AGAINST the masking
hypothesis and must appear in §2/§3, with the success criterion re-keyed to the
conditions that actually fail the gate (floors + ceilings). Cheap text fix
pre-run; a false-PASS narrative post-run.
