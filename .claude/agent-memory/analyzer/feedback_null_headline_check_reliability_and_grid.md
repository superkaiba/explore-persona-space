---
name: Null-headline pre-checks: reliability + swept select-confirm
description: Before writing an "all estimators/predictors null" headline, check split-half reliability (noise vs stably-elsewhere) and the exploratory grid's select_confirm block (a seed-confirmed swept positive can be the rider)
type: feedback
---

For direction/estimator bake-offs (cosine-vs-realized DVs), two checks before finalizing a null headline.

**Why:** Task #602 (2026-06-11): the registered L14 read was a clean 0/18 null, but (a) the grid's `select_confirm` block held a seed-confirmed positive (E1 at L27 on EM families, cos ~0.82, survives select-on-42/confirm-on-137-256) that became the headline rider, and (b) the reliability block separated two very different nulls — most estimators had split-half ~0.93-0.97 (stably measured, pointing elsewhere — a real negative) while marker-E1 exclude-marker reads had split-half ≈ 0 (the estimator itself is noise — "no off-token contrast signal in the training data", a different and more interesting claim).

**How to apply:** (1) Always read `grids/*: select_confirm` (or equivalent best-swept-cell + confirm-seeds record) before drafting the headline; a surviving swept cell reframes "X is invalid" into "X is invalid at the registered construction; valid at <swept cell> (exploratory, seed-confirmed)". (2) Always read the split-half/subsample reliability block and split INVALID verdicts into "indistinguishable from null though reliably measured" vs "estimator direction itself unreliable". (3) Tiny rank panels (n=4 Spearman gives only ±0.949/±0.316 with ties) + norm-only-matches-real flags mean the panel is direction-insensitive — never narrate its repair verdicts as direction recovery.
