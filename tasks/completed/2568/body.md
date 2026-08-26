---
title: 'workflow-fix: consumer-supersession audit — enumerate committed consumers
  of refuted evidence artifacts and require supersession labels'
kind: infra
tags: []
created_at: '2026-08-25T04:02:03Z'
has_clean_result: false
workflow: v1
---
# workflow-fix: consumer-supersession audit — enumerate committed consumers of refuted evidence artifacts and require supersession labels

## Goal

When a round REFUTES evidence that committed downstream artifacts consume (a claims registry row, a paper/poster figure generator, a methodology doc), nothing today mechanically enumerates those consumers or verifies they carry a supersession label — the refuted evidence keeps presenting as valid on every downstream surface until a human notices. Build the recurring audit: given a refuted artifact set (file names / HF prefixes + the refuting evidence pointer), enumerate tracked consumers (`grep -rl` over the artifact names, producers excluded) and FAIL/WARN any consumer that neither consumes the replacement artifact nor carries an explicit superseded/different-eval-pool label; wire it into the surface the analyzer/orchestrator runs at fold-in.

## Provenance

Surfaced by the Codex methodology twin during #1901 plan-v14 review (round mlp-scaling-densify, 2026-08-25): the round's G1 finding (epm:progress 2026-08-25T03:18:29Z on #1901) refuted the cross-store scaling join, yet `docs/paper_context_answer_map/claims.md` (C1 main row, SOLID), `scripts/issue1901_body_figures.py` (`fig_paper_c1_scaling`), and `docs/posters/mats_2026/make_plot1_scaling.py` all still presented the join as valid; the reconciler ruled the per-round fix into #1901 plan v15 (the `superseded_cross_store_join.json` record + the owned analyzer duty) and routed THIS recurring-audit generalization to workflow-fix filing. Worked example / first fixture: the #1901 v15 supersession record schema (`eval_results/issue_1901/paper_densify/superseded_cross_store_join.json` — refuted join, evidence, replacement artifact, enumerated consumers).

## Sketch

- A small helper (e.g. `scripts/audit_artifact_supersession.py`) reading committed `superseded_*.json` records (schema per the #1901 v15 instance) and re-running the consumer enumeration; non-labeled consumers listed with file:line.
- Wiring candidates (implementer/planner to decide): the analyzer's fold-in checklist (the refutes-a-promoted-claim presumption already routes prose corrections/infra filing through it); optionally a `workflow_lint.py` check over `superseded_*.json` records.
- NOT in scope: auto-editing consumers; the audit surfaces, humans/rounds fix.
