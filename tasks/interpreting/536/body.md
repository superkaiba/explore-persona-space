---
title: Standardizing persona-distance cosine on the mean-centered recipe leaves every
  recomputable published call standing, and upgrades one prior null to a centering-attributable
  candidate rescue (MODERATE confidence)
kind: analysis
tags:
- needs-thomas
created_at: '2026-06-09T18:35:37Z'
has_clean_result: true
---
# Standardizing persona-distance cosine on the mean-centered recipe leaves every recomputable published call standing, and upgrades one prior null to a centering-attributable candidate rescue (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** we'd been computing "how far apart are two personas" two different ways for months — i audited every computation site, pinned one official definition, and re-ran the affected headline numbers under it: nothing we published flips.

**Takeaways.**
- raw (un-centered) cosine on Qwen centroids squeezes every persona pair into 0.73–1.0, so roughly half the leakage line measured distance in a compressed geometry. the rank order mostly survived the fix: every result i could recompute keeps its sign and its original call.
- the one mover: the old "widening the source set doesn't flatten the leakage-vs-distance slope" null looks like it might flatten after all under the proper metric — but only under one of the two uncertainty estimators, so it's a candidate pending one more refit, not a flip.
- 7 affected tasks can't be re-checked at all: their activation banks only ever lived on pods that are long gone. bounded fix (~1–2 GPU-h) if we ever need them.
- the definition is now pinned in a rules file, so new code can't silently pick the squeezed version again.

**How this updates me.** i trust the distance-predicts-leakage line a bit more, not less — every published number reproduced exactly from the saved artifacts before any re-grading happened, which is a paper trail i wasn't sure we had. the thing that would change my mind on the one candidate rescue: the mixed-model refit coming back non-significant.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

A research-integrity scan flagged that persona-distance cosine — the load-bearing predictor in the "distance predicts which personas catch a behavior" line — had been computed two different ways across the project. The early tasks ([#66](https://eps.superkaiba.com/tasks/66), [#142](https://eps.superkaiba.com/tasks/142), [#311](https://eps.superkaiba.com/tasks/311), [#380](https://eps.superkaiba.com/tasks/380)) subtract the centroid bank's shared mean direction before normalizing; a later line ([#405](https://eps.superkaiba.com/tasks/405), [#478](https://eps.superkaiba.com/tasks/478), [#490](https://eps.superkaiba.com/tasks/490), [#505](https://eps.superkaiba.com/tasks/505), and the predictor nulls [#396](https://eps.superkaiba.com/tasks/396)/[#415](https://eps.superkaiba.com/tasks/415)) skips the centering. On Qwen-2.5-7B centroids that is not cosmetic: residual-stream vectors share a dominant common direction, so raw cosine compresses every persona pair into roughly [0.73, 1.0] — about a 6x squeeze of the geometry. If "cosine predicts leakage" conclusions were drawn in two incompatible geometries, their effect sizes and even directions might not hold.

The goal: pin one canonical persona-distance definition, audit every computation site in the codebase, recompute the affected tasks' headline cosine-vs-leakage statistics under the canonical metric from artifacts already on disk, and propose per-task re-grades. A correctness audit of existing results, not new science — no training, no GPU.

### What I ran

Three moves, all CPU on the VM. No model was loaded and nothing was generated — every input is a persisted centroid bank, similarity matrix, or per-task results table, and every output is a recomputed statistic.

**(1) Pin one definition.** Canonical persona-distance cosine = globally mean-center the centroid bank, then L2-normalize, then cosine. This is the library default, the recipe behind the project's strongest validated geometry, and the standard fix for anisotropic embedding spaces. One scoping addition: 2-vector pairwise cosines (the narrow source-vs-target predictor family) have no bank to center against — centering a 2-element bank degenerates to cosine ≡ −1 — so the pin defines two labeled families (bank cosine, canonical; raw pairwise, allowed but labeled and never numerically compared to bank values). The pin landed in the project rules file.

**(2) Audit every computation site.** Two deterministic regex passes over all analysis, script, and experiment directories — a literal-API pass plus a high-recall hand-rolled-idiom pass — plus a git-history search for deleted producer scripts. 138 files hit, every one accounted for: 36 computation-site rows (classified centered / raw / both-computed / not-applicable, with file-and-line evidence and the consuming tasks), 2 deleted producers recovered from git history, and 105 file dispositions (consumers, different constructs, infra). Unaccounted files fail the audit run loudly. 16 persisted artifacts additionally got an off-diagonal fingerprint check (raw regime sits in ~[0.73, 1.0]; centered spans negative values), so each task is classified by what its analysis actually *read*, not by what the producing script computed.

**(3) Recompute and re-grade.** For each affected task: load the same persisted centroid bank its analysis used, then compute the task's *own* published statistic twice from the identical join — once with the original recipe, which must reproduce the published number before anything else is read (matrix-level 1e-4 agreement where a published matrix persists; row-level 1e-4 or statistic-level agreement within 0.02 otherwise), and once under the canonical metric. Re-grade labels follow an ordered decision tree fixed in the plan before any number was computed: for originally-significant rows, flips / weakens / strengthens / stands; for originally-null rows, a null can only be overturned into a *candidate rescue* by a Holm-corrected p < 0.05 within the row's full test family. Tasks whose centroids are lost (only a similarity matrix persists) live in a separate sensitivity-only namespace — the centroid norms are unrecoverable from the matrix, so those rows can never earn canonical labels.

<details open>
<summary>2 example audit-site rows + 1 example re-grade row (cherry-picked for illustration; full tables linked below)</summary>

| Table | Example row |
|---|---|
| Audit site | `scripts/recompute_predictors_i415.py:242-245` — classification: raw pairwise vs assistant/neutral references; consuming tasks: the 24-persona predictor nulls. Meaning: those published nulls were computed in the compressed geometry, so "centering rescues the predictor" was a live possibility this audit had to answer. |
| Audit site | `compare_extraction_methods.py` — the script computed BOTH matrices, but the artifact its consumers read (`cosine_matrix_a_layer20.json`) stored the raw one: off-diagonal range [0.7322, 0.9971], median 0.945 over 20 personas. Classified by what the consumer read. |
| Re-grade | Pooled distance-vs-shift rank correlation for the extraction-method task: raw join reproduces the published matrix at 1e-4, pooled ρ −0.676 (raw) → −0.593 (centered), both p < 1e-32, N = 336 — label **stands**. |

Full tables (one row per site / per task, with evidence, gates, and machine-readable caveats): [audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/audit_table.json) · [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/regrade_table.json)

</details>

### Findings

#### Raw cosine on this model really is a squeezed geometry — and the recipe split is per-script, not chronological

First, the premise of the flag had to be checked: is the un-centered recipe actually degenerate on this model, and which tasks read which recipe? The four panels below show the off-diagonal cosine distribution of four persisted centroid banks under both recipes.

![Four histogram panels, one per centroid bank (20-persona extraction bank at layer 20, the 16-condition lineage bank at layer 21, the 60-persona persona-vector bank at layer 21, the 111-persona bank at layer 20). In every panel the raw distribution (orange) is a narrow spike between roughly 0.7 and 1.0 while the mean-centered distribution (red) spreads from about -0.65 to +1.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/affb2615539ee5c34f79015b97621424a4acf780/figures/issue_536/bank_offdiag_histograms.png)

> **Figure.** *Every bank shows the same compression: raw off-diagonal cosine piles up between ~0.7 and 1.0, while mean-centering spreads the same pairs across the full range.* One panel per persisted centroid bank (n = 20, 16, 60, and 111 personas); orange = raw recipe, red = mean-centered. The lineage-bank panel is the approximate read recovered from its persisted similarity matrix (centroids lost). This is the measured version of the ~6x geometry squeeze the integrity flag described.

The audit's second surprise is that the split does not track time — it tracks individual scripts. A mid-line task read raw output from a script that had computed both matrices; a late task correctly recorded centered provenance; and one brand-new file that landed on main mid-audit computes raw pairwise distance even though a 51-vector bank is available to center against (caught by the audit's coverage gate, classified, and flagged for that task's own analyzer — fixing it is out of this task's scope). The pin plus the audit table are the guard against this recurring. One site row, cherry-picked for illustration; all 36 site rows + 105 dispositions live in [audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/audit_table.json):

```
AUDIT SITE (cherry-picked for illustration)
file: src/explore_persona_space/experiments/leave_one_out_505/build_pv_centroids.py:196
classification: raw (explicit) — compute_cosine_matrix(c, centering="none")
consumed by: the leave-one-out protection null (60-persona bank, layer 21)
fingerprint of persisted matrix: raw regime confirmed before any re-grade was read
```

#### Every recomputable headline stands under the canonical metric

With the joins gated, the central question: do the published calls survive? Stratified by evidence class (the strata are not poolable into one count — verification rows test artifact rot, raw-line rows test the metric swap):

![Dumbbell chart with six rows, one per re-graded statistic. Each row shows the absolute Spearman rho under the raw recipe (open/grey marker) and the mean-centered recipe (filled marker), connected by a line, with the re-grade label annotated. The verification row and the two raw-line effect rows all stay between 0.59 and 0.78; the predictor-null row stays near zero; the two approximate sensitivity rows (open markers) move from 0.94 to 0.79 and 0.37 to 0.32 while keeping sign.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/affb2615539ee5c34f79015b97621424a4acf780/figures/issue_536/hero_regrade_dumbbell.png)

> **Figure.** *No re-graded statistic flips sign or loses its published call when the distance axis is swapped from raw to mean-centered.* One row per headline statistic; grey/open = raw recipe, filled = mean-centered, color = re-grade label. The two "(approx)" rows are matrix-only sensitivity reads (open markers, separate namespace — see the last finding); they are plotted for completeness, never pooled with the exact rows. Ns per row: 550 pooled pairs (verification), 336 rows (pooled shift), 2,800 rows (per-K), 24 personas (predictor null), 156 pairs and 171 pairs (the two approximate rows).

**Canonical-line verification (4 tasks).** The four early tasks whose artifacts persist reproduce their published numbers exactly from disk: the 100-persona task's pooled ρ = 0.6016 (p = 2.0e-55, N = 550) and all five per-source values to four decimals; the single-token follow-up at all four layers (0.169 / 0.520 / 0.567 / 0.557, each within 0.001); the 19-bank task's full centered matrix at max deviation 3.6e-7 plus its headline ρ = −0.348 (one-sided p = 0.086, N = 17); and the 24-persona panel task at max deviation 8.4e-8. Two honesty notes from the verification pass: the 19-bank task's persisted analysis JSON carries a *provisional* snapshot of ρ = +0.534 under a different parameterization that does not match the published −0.348 — the published number is the one that reproduces from the centroids, so the JSON appears to be a pre-final snapshot. And the single-token follow-up's published values reproduce only under centering on its own 11-persona subset bank, not the full 111-bank (which gives 0.57–0.76) — a live demonstration of the pin's caveat that centered cosine is bank-dependent and only comparable within one bank. Planned-vs-actual coverage: the plan named nine canonical-line tasks for verification; only these four were recoverable. The other five carry evidence-backed unrecoverable partitions (last finding) — the canonical line is verified on its recoverable members, not wholesale.

**Raw-line re-grades (6 tasks).** The pooled distance-vs-shift correlation moves −0.676 → −0.593 (both p < 1e-32, N = 336): **stands**. The per-source-set-size leakage-vs-distance rank correlations move −0.74/−0.78/−0.77/−0.78 → −0.70/−0.79/−0.80/−0.81 (every p < 1e-100, N = 1,120/560/560/560): **stands**. The three published nulls stay null: the on-axis vs off-axis dose-matched gap (coefficient 0.1996 → 0.114, p = 0.103 → 0.212, N = 255), the 12-cell predictor family (next-to-last finding), and the leave-one-out protection read (3 of 6 arms positive against a 5-of-6 bar; pooled p = 0.96, N = 936). One null moves — the next finding.

Re-graded rows also record the rank correlation between the raw and centered distance vectors actually joined: 0.67–0.99. Centering substantively re-ranks pairs (this is not a trivially monotone rescaling), and the calls survive that re-ranking — which is what makes "stands" informative rather than tautological. One verification trace, cherry-picked for illustration; every row's gates are machine-readable in [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/regrade_table.json):

```
VERIFICATION TRACE (cherry-picked for illustration)
task: the 19-bank centered-line task
gate 1 (matrix): full 19x19 centered cosine matrix vs persisted JSON — max deviation 3.6e-7  PASS
gate 2 (pair statistic): selected-pair centered cosine -0.65139 vs persisted pair file       PASS
gate 3 (headline): value-residualized partial rank correlation -0.348, one-sided p 0.086,
        N = 17 bystanders — matches the published body exactly                               PASS
```

#### The one mover: the slope-flattening null becomes a centering-attributable candidate rescue — under the registered estimator only

The source-set-size experiment published two co-primary flatness reads: the far-minus-near leakage gap across set sizes (slope −0.12, p = 0.148) and a set-size-by-distance interaction (+0.010, p = 0.405 under its mixed-effects estimator). Both were nulls — "widening the source set does not flatten the leakage-vs-distance slope." Centering re-draws the distance axis those reads sit on, and the first thing it moves is the design's own distance bands:

![Heatmap cross-tabulating each held-out persona's distance band under the original raw-distance design (rows: near, near-mid, mid, far, very-far, tail) against its band under mean-centered distance (columns). The diagonal holds 19 of 35 personas; 16 move off it, with the largest motion in the far, very-far, and tail rows.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/affb2615539ee5c34f79015b97621424a4acf780/figures/issue_536/band_reassignment_478.png)

> **Figure.** *16 of 35 held-out personas change distance band under the canonical metric — the design's "far" structure reshuffles while its "near" structure holds.* Rows = band assigned by the original raw-distance design; columns = band under mean-centered distance; cell counts sum to 35 personas. The near and near-mid bands are essentially stable; most movement is among far / very-far / tail.

The gap-slope read survives the reshuffle as a null both ways (raw −0.120, p = 0.148; centered −0.133, p = 0.138 — the design-band and re-banded reads coincide). The interaction does not. Why this test: the audit's plan registered a cluster-robust regression (clusters = cell-and-seed) as the falsification test for this row, so the SAME estimator was computed on both joins — that is what makes the contrast centering-attributable rather than estimator-shopping. On the raw join the interaction is +0.0097 (p = 0.022); on the centered join it is +0.0215 (p = 0.00075, N = 2,800), which survives Holm correction within its 2-test family at the family rule's 0.05 and also at the stricter 0.01 threshold the plan recorded for this row's falsification read (the label binds at the family rule; the outcome is the same at both levels). The direction is positive: the leakage-vs-distance slope *flattens* as the source set widens — the safety-relevant direction the original task reported as absent.

Two things this is *not*. It is not an unconditional overturn: the published co-primary was a mixed-effects fit, that estimator was never refit here (a mid-audit estimator swap would have been an unregistered change), and the same cluster-robust regression already reads p = 0.022 on the *raw* join — so part of the movement is estimator sensitivity, not centering. The row therefore carries the label **null-overturned (candidate rescue)** with a machine-readable estimator caveat (tracked as open concern `478-estimator-mixedlm-refit` in this task's concerns ledger), and the registered follow-up is a mixed-effects refit on both joins — that refit is what would turn "candidate" into "confirmed" or kill it. This caveat is the binding constraint behind the MODERATE in the title: the audit's stands/verification claims ride exact reproductions, while the single label that moved awaits one more estimator check.

#### The predictor nulls don't rescue: 0 of 12 cells move

The published 24-persona predictor nulls were computed with raw pairwise cosine to the assistant and neutral references — i.e., in the compressed geometry — so "mean-centering rescues the predictor" was a genuinely open possibility, and the one place this audit could have minted a new positive result. It does not happen:

![Forest plot of 12 cells (two reference predictors crossed with six leakage surfaces). Each cell shows the length-partial rank correlation under raw and centered recipes; every point sits inside the shaded band marking absolute rho below 0.5, most between -0.2 and +0.2.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/affb2615539ee5c34f79015b97621424a4acf780/figures/issue_536/forest_396_415.png)

> **Figure.** *All 12 predictor cells stay inside the no-effect band under the canonical metric.* Rows = the 12 cells (cosine-to-assistant and cosine-to-neutral, each against six leakage surfaces); x = length-partial rank correlation, N = 24 personas per cell; the shaded band marks absolute ρ below the 0.5 bar the rescue rule required. Raw and centered values overlap closely in most cells.

Zero of 12 cells pass the Holm-corrected rescue rule, and zero even reach uncorrected p < 0.01 (headline cell: ρ = 0.008 → −0.048, p = 0.52 → 0.65, N = 24). One check makes this null informative rather than mechanical: after centering, the assistant and neutral reference vectors keep norms above the bank median (67th and 75th percentile of the bank's centered-norm distribution), so the centered predictor is not a noise-dominated cosine against a vanishing reference. The base-model-distance-predicts-leakage nulls on this panel were not an artifact of the metric.

#### What the saved artifacts can't support: 4 sensitivity-only reads and 7 unrecoverable tasks

The audit's honesty boundary. Four tasks (the 16-condition transformation lineage and the cosine-vs-divergence alignment task) persisted only similarity matrices — the centroid norms are gone, so the exact canonical recompute is mathematically unavailable. For these, an approximate read: recover the normalized configuration from the persisted matrix, center those unit vectors, re-cosine.

![Scatter of off-diagonal pairs from the lineage bank at layer 21: x = raw cosine similarity (all points between about 0.55 and 1.0), y = the centered-approximate similarity for the same pair (spanning -0.65 to +0.9). A dashed identity line shows how far below it the cloud falls.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/affb2615539ee5c34f79015b97621424a4acf780/figures/issue_536/compression_scatter_406_L21.png)

> **Figure.** *The same pairs that raw cosine crowds into [0.55, 1.0] spread across the full similarity range once centered — the decompression the approximate read recovers from a matrix alone.* Each point is one persona pair from the lineage bank at layer 21 (raw on x, centered-approximate on y); dashed line = identity. This read supports sensitivity labels only: with the centroid norms lost, agreement between raw and approximate-centered is not a certificate for the exact canonical recompute.

All four matrix-only tasks come back **sensitivity-agrees**: every lineage cell keeps its sign and significance (e.g., −0.368 → −0.316, p = 6.0e-5, N = 156), and the cosine-vs-divergence alignment keeps sign and significance at all four layers while shrinking (layer-20 ρ 0.940 → 0.787, p = 3.1e-37, N = 171 pairs) — a reminder that the famous 0.94 was partly a compression artifact, since both metrics ride the same shared-mean direction.

Seven tasks are unrecoverable from disk, each with an evidence-backed partition row: four early causal/taxonomy tasks whose per-checkpoint or per-state centroid banks were written pod-side and never committed (zero entries in git history across all branches), one task whose distance side persists but whose behavioral outcome tables were never committed, and the two cue-geometry tasks, for which a live WandB search across all 175 projects found zero centroid-like artifacts. Their published claims are neither confirmed nor disconfirmed by this audit — re-checking them needs ~1–2 GPU-hours of base-model centroid re-extraction on a single eval pod, deferred deliberately (the flag is closeable on the recoverable majority, and GPU spend before knowing whether the recoverable reads even moved would have been premature). One partition row, cherry-picked for illustration; all 7 partition rows with their search evidence are in [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/regrade_table.json):

```
PARTITION EVIDENCE (cherry-picked for illustration)
task: the 200-bystander taxonomy task (published rho 0.711, p = 1e-15, N = 200)
missing: the 200-persona layer-15 centroid bank (never persisted)
searched: local eval_results tree (absent); git log --all for the directory
          (1 commit — consumption-level values only, on an unmerged branch);
          HF data repo via the Hub API (no taxonomy bucket)
label: join-failed (X unrecoverable — needs GPU re-extraction)
```

## Reproducibility

**Parameters:** no model, no training, no seeds — deterministic CPU linear algebra over persisted artifacts. Load-bearing analysis parameters (all fixed in plan §3/§4 before the recompute ran): canonical recipe = global-mean-center → L2-normalize → cosine (`compute_cosine_matrix(C, centering="global_mean")` — the recipe `src/explore_persona_space/analysis/representation_shift.py` applies when no centering argument is passed); join gates = matrix-level 1e-4 where a published matrix persists, row-level 1e-4, else statistic-level reproduction within 0.02 of the published value (slope rows: inside the published interval); decision-tree magnitude boundaries = 0.5x (weakens) / 1.5x (strengthens) on the scale-invariant statistic; null-rescue rule = Holm-corrected p < 0.05 within the row's full test family (the 12-cell predictor family also required absolute ρ ≥ 0.5; the flatness row's family = 2 tests, with the per-row falsification threshold 0.01 recorded alongside — outcome-invariant here); per-row α = each task's own published threshold (0.01 where the task did not publish one). Audit-scope slugs (table rows only, not reader-facing conditions): `verify_centered`, `regrade_raw`, `readoff_r6`, `approx_gram`, `scope_pairwise`.

**Artifacts:**
- Re-grade table (30 rows: 13 stands/read-off incl. 4 canonical-line verifications, 1 candidate rescue, 4 sensitivity-agrees, 7 join-failed partitions, 6 pairwise-scoped): [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/regrade_table.json)
- Audit table (36 site rows + 2 deleted-producer rows + 105 dispositions over 138 swept files + 16 artifact fingerprint checks): [audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/audit_table.json)
- Figures payload (every plotted float): [figures_payload.json](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/figures_payload.json) · figures: [figures/issue_536/](https://github.com/superkaiba/explore-persona-space/tree/affb2615539ee5c34f79015b97621424a4acf780/figures/issue_536) (PNG + PDF + meta.json each)
- Pinned input snapshots (artifacts that live only on other branches): [eval_results/issue_536/inputs/](https://github.com/superkaiba/explore-persona-space/tree/affb2615539ee5c34f79015b97621424a4acf780/eval_results/issue_536/inputs) — the 2,800-row per-cell table snapshot and the two alignment-task matrices
- The metric pin: [.claude/rules/persona-distance-metrics.md](https://github.com/superkaiba/explore-persona-space/blob/affb2615539ee5c34f79015b97621424a4acf780/.claude/rules/persona-distance-metrics.md) § "Bank centering — canonical (task #536)"
- Reused artifacts (this audit's inputs ARE prior tasks' persisted measurement artifacts; fitness = identity — each row recomputes that task's own statistic from that task's own bank, enforced by the join gates):
  - Reused centroid bank from [#66](https://eps.superkaiba.com/tasks/66): `eval_results/single_token_100_persona/centroids/centroids_layer{10,15,20,25}.pt` (local, also joined for the per-K and flatness rows) — fit: same bank the published joins consumed; 1e-4 matrix gate passed.
  - Reused centroid bank from the extraction-method comparison (consumed by [#405](https://eps.superkaiba.com/tasks/405)): `eval_results/extraction_method_comparison/centroids_method_a.pt` — fit: reproduces the published raw matrix at 1e-4.
  - Reused centroid bank from [#274](https://eps.superkaiba.com/tasks/274) (consumed by [#396](https://eps.superkaiba.com/tasks/396)/[#415](https://eps.superkaiba.com/tasks/415)/[#380](https://eps.superkaiba.com/tasks/380)): `eval_results/issue_274/centroids/centroids_n24_layers0_27.pt` — fit: statistic-level gate within 0.0102 of the published headline.
  - Reused geometry bundles from [#505](https://eps.superkaiba.com/tasks/505) and [#142](https://eps.superkaiba.com/tasks/142) on the HF data repo (`superkaiba1/explore-persona-space-data`: `issue505_loo_contrastive/geometry/centroids_pv_L{7,14,21,27}.pt`; `issue142_single_token/centroids/{centroids_layer20.pt, persona_names.json}`) — fit: resolved via the Python Hub API at write time (4 and 2 files respectively); matrix / statistic gates passed.
  - Reused centroid bundle from [#311](https://eps.superkaiba.com/tasks/311): `eval_results/issue_311/centroids_base.pt` — fit: full-matrix 1e-4 gate vs the task's persisted cosine JSON.
  - Reused persisted matrices (no centroids) from [#406](https://eps.superkaiba.com/tasks/406) (`eval_results/issue_406/cosine/C_L*.json`) and [#341](https://eps.superkaiba.com/tasks/341) — fit: sensitivity-namespace only, per the matrix-only rule above.

**Compute:** CPU only, VM-side; full audit + recompute + figures pipeline runs end-to-end in ~27 s. No pod provisioned. GPU-hours: 0.

**Code:** `scripts/issue536_audit.py` (two-pass sweep + dispositions + fingerprints), `scripts/issue536_recompute_driver.py` (family registry + per-task adapters + join gates), `scripts/issue536_figures.py` — all at commit [affb26155](https://github.com/superkaiba/explore-persona-space/tree/affb2615539ee5c34f79015b97621424a4acf780) on branch `issue-536`. Provenance correction: the committed tables internally stamp `code_commit = bfa36c3f` (the HEAD at generation time, before the final round-2 code commit was created from the same working tree); the authoritative code SHA for the shipped scripts and tables is **affb26155**. The audit and driver tables also record independent `data_root_commit` values because main's HEAD moved between the two phases (concurrent sessions); the consumed artifacts are stable committed files, but a strict reproducer should run both phases against one pinned checkout. Reproduce:

```bash
cd explore-persona-space   # at affb26155
uv run python scripts/issue536_audit.py --data-root "$PWD"
uv run python scripts/issue536_recompute_driver.py --data-root "$PWD"
uv run python scripts/issue536_figures.py
```
