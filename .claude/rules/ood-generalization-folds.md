---
description: OOD / structured-fold generalization — any held-out predictive statistic (reconstruction R², read-out ρ, predictor skill) over a sample with known group structure (context families, prompt genres, persona panels) must include at least one GROUP-level held-out fold (LOFO / leave-one-genre-out / corpus transfer), not only pointwise LOO/LOCO; report which fold each headline is under. Loads on-demand at plan time.
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# OOD generalization folds (LOFO over pointwise LOO)

**Load this rule whenever a plan or analysis reports a held-out predictive
statistic** — reconstruction R² / skill, read-out ρ, predictor accuracy, any
"held-out" or "cross-validated" number — **over a sample with known GROUP
structure**: a context grid built from prompt families, a probe pool with
genres, a persona panel, a behavior class set, seeds sharing a template. The
standing directive (Thomas, 2026-07-02, #810): **always include LOFO — or
some other structured OOD generalization fold — not just pointwise
leave-one-out.**

## The failure mode

Pointwise LOO/LOCO trains on siblings of the test point. When the sample
contains groups of near-duplicates (e.g. #810's 50-context grid: 14 persona
prompts, 10 WildChat, 8 ICL, ...), every held-out point has same-family
members in its training fold, so pointwise LOO measures *interpolation
within family*, not generalization. It systematically overstates — and can
REORDER — cross-context claims.

**Motivating evidence (#810, 2026-07-02).** The headline LOCO sweep said
max-pool is the best answer-side summary (0.826 vs mean 0.800) and the
trained-ridge read-out reached ρ ≈ 0.909. The 7-fold leave-one-FAMILY-out
re-read reordered the reconstruction ranking (mean 0.804 ≥ turn_nl 0.791 >
max-pool 0.760 at LOCO-best layers) and collapsed the trained-ridge read-out
to ρ ≈ 0.285. Both headline claims were fold-artifacts at the group level:
max-pool's edge and the read-out signal lived in within-family
interpolation. (`eval_results/issue_810/adhoc_lofo_heatmap_grids.json`,
`figures/issue_810/adhoc_layer_x_position_heatmaps_LOFO.png`.)

## The rule

1. **Name the group structure at plan time.** Every plan whose DV is a
   held-out predictive statistic states the sample's grouping axes (family,
   genre, persona, template, seed) in the measurement section. "No known
   structure" is a positive claim that must be argued (a genuinely iid
   sample is the only pointwise-only exemption).
2. **Include at least one GROUP-level fold.** Leave-one-family-out /
   leave-one-genre-out / leave-one-persona-out — held-out groups, not
   held-out points. A corpus/genre TRANSFER arm (fit on corpus A, evaluate
   on corpus B, e.g. Betley → UltraChat) is the strongest form and counts.
3. **Report both; label every headline with its fold.** Pointwise LOO may
   stay (it upper-bounds within-distribution skill) but never carries a
   generalization claim alone. A claim that holds under LOO and fails under
   LOFO is reported as within-family interpolation, not generalization.
4. **Selection under the fold stays symmetric.** Any max/argmax over a free
   axis (layer, cell, summary) inside a LOFO headline inherits the
   selection-symmetric null treatment (`.claude/rules/selection-symmetric-nulls.md`)
   with the null computed under the SAME fold structure.
5. **Group-level n is the real n.** With G groups the fold gives G
   quasi-independent test units (7 in #810), not n points; frame CIs and
   "unresolved" calls accordingly.

## Enforcement

- `planner.md` §6 "OOD generalization folds" Required block (full template:
  `.claude/rules/planner-section-reference.md` § 6. Evaluation) — the plan
  names the grouping axes and the group-level fold per held-out DV, or the
  named "N/A — no held-out predictive DV" / explicit-iid escape.
- `critic.md` Statistics & Measurement lens item 13 (full rubric:
  `.claude/rules/critic-lens-reference.md`) — REVISEs a
  held-out-predictive-DV plan with pointwise-only folds and no iid argument.

## Files of record

Task body #810 (LOCO vs LOFO reorder + read-out collapse); the 2026-07-03
01:31 UTC inline LOFO pass on #810's committed cells; sibling rules
`.claude/rules/selection-symmetric-nulls.md` (selection inside folds),
`.claude/rules/llm-judging.md` rule 21 (design-aligned split-half — the
reliability-side sibling of fold-structure discipline).
