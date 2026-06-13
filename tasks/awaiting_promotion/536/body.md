---
title: Standardizing persona-distance cosine on the mean-centered recipe leaves every
  recomputable published call standing — including the one null whose apparent rescue
  died under the published estimator (HIGH confidence)
kind: analysis
tags:
- needs-thomas
created_at: '2026-06-09T18:35:37Z'
has_clean_result: true
---
# Standardizing persona-distance cosine on the mean-centered recipe leaves every recomputable published call standing — including the one null whose apparent rescue died under the published estimator (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the project had been computing "how far apart are two personas" two different ways for months — i audited every computation site, pinned one official definition, and re-ran the affected headline numbers under it: nothing published flips.

**Takeaways.**
- raw (un-centered) cosine on Qwen centroids squeezes persona pairs into a narrow band up near 1.0 (0.73–1.0 on the main source bank; the bigger banks bottom out around 0.6), so roughly half the leakage line measured distance in a compressed geometry. the rank order mostly survived the fix: every result i could recompute keeps its sign and its original call.
- the one that needed a second look: the old "widening the source set doesn't flatten the leakage-vs-distance slope" null looked like it might flatten after all — but only under the audit's registered uncertainty estimator. refitting the paper's own mixed-model estimator on the centered data killed it: same point estimate, p=0.215, with the raw side reproducing the published numbers almost exactly first. so the published null stands; what i actually found there is that the call depends on which uncertainty estimator you use, not that centering rescues it.
- 7 affected tasks can't be re-checked at all: their activation banks only ever lived on pods that are long gone. bounded fix (~1–2 GPU-h) if i ever need them.
- the definition is now pinned in a rules file, so new code can't silently pick the squeezed version again.

**How this updates me.** i trust the distance-predicts-leakage line a bit more, not less — every published number reproduced from the saved artifacts before any re-grading happened (most to four decimals, all within the gates fixed before the recompute ran), which is a paper trail i wasn't sure the project had. the one loose end from the first pass — a possible rescue of a prior null — closed when the mixed-model refit came back non-significant, so the final tally is clean: every call i could re-check stands.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

A research-integrity scan flagged that persona-distance cosine had been computed two different ways across the project. Every "distance predicts which personas catch a behavior" claim reads this one number, so the claims inherit whatever the two recipes disagree about. The early tasks ([#66](https://eps.superkaiba.com/tasks/66), [#142](https://eps.superkaiba.com/tasks/142), [#311](https://eps.superkaiba.com/tasks/311), [#380](https://eps.superkaiba.com/tasks/380)) subtract the centroid bank's shared mean direction before normalizing; a later line ([#405](https://eps.superkaiba.com/tasks/405), [#478](https://eps.superkaiba.com/tasks/478), [#490](https://eps.superkaiba.com/tasks/490), [#505](https://eps.superkaiba.com/tasks/505), and the predictor nulls [#396](https://eps.superkaiba.com/tasks/396)/[#415](https://eps.superkaiba.com/tasks/415)) skips the centering. On Qwen-2.5-7B centroids the difference is large: residual-stream vectors share a dominant common direction, so raw cosine crowds persona pairs into a narrow high-similarity band. How narrow is bank-specific: the 20-persona bank behind the strongest raw-line result spans [0.73, 1.00], the 60-persona and 111-persona banks bottom out at 0.71 and 0.60, and the 16-condition lineage bank reaches 0.59 at layer 21 (0.28 at layer 27). Mean-centering spreads the same pairs across roughly [−0.74, +1.0], a 4–6x wider range depending on the bank. If "cosine predicts leakage" conclusions were drawn in two incompatible geometries, their measured sizes and even directions might not hold.

My goal: pin one canonical persona-distance definition, audit every computation site in the codebase, recompute the affected tasks' headline cosine-vs-leakage statistics under the canonical metric from artifacts already on disk, and propose per-task re-grades. A correctness audit of existing results, run with no training and no GPU.

### What I ran

I made three moves, all CPU on the VM. No model was loaded and nothing was generated: every input is a persisted centroid bank, similarity matrix, or per-task results table, and every output is a recomputed statistic.

**(1) Pin one definition.** I pinned canonical persona-distance cosine as: globally mean-center the centroid bank, then L2-normalize, then cosine. It is the library default and the usual fix for anisotropic embedding spaces, where a shared common direction swamps cosine unless removed. It is also the recipe behind the early tasks that survive verification below. One scoping addition: 2-vector pairwise cosines (the narrow source-vs-target predictor family) have no bank to center against — centering a 2-element bank degenerates to cosine ≡ −1, so the pin defines two labeled families (bank cosine, canonical; raw pairwise, allowed but labeled and never numerically compared to bank values). I landed the pin in the project rules file.

**(2) Audit every computation site.** I swept all analysis, script, and experiment directories with two deterministic regex passes (a literal-API pass plus a high-recall hand-rolled-idiom pass), with a git-history search for deleted producer scripts on top. The sweep hit 138 live files; counting the three issue536_* audit scripts themselves, classified pre-emptively, 141 files are accounted for: 36 computation-site rows (classified centered / raw / both-computed / not-applicable, with file-and-line evidence and the consuming tasks) and 105 file dispositions (consumers, different constructs, infra). Two deleted producer scripts were recovered from git history as well. Unaccounted files fail the audit run loudly. 16 persisted artifacts also got an off-diagonal fingerprint check (the raw regime sits in a narrow high-similarity band; centered spans negative values), so each task is classified by what its analysis actually *read*, not by what the producing script computed.

**(3) Recompute and re-grade.** For each affected task I loaded the same persisted centroid bank its analysis used, then computed the task's *own* published statistic twice from the identical join — once with the original recipe, which must reproduce the published number before anything else is read (matrix-level 1e-4 agreement where a published matrix persists; row-level 1e-4 or statistic-level agreement within 0.02 otherwise), and once under the canonical metric. Re-grade labels follow an ordered decision tree fixed in the plan before any number was computed: for originally-significant rows, flips / weakens / strengthens / stands; for originally-null rows, a null can only be overturned into a *candidate rescue* by a Holm-corrected p < 0.05 within the row's full test family (Holm is a step-down correction for testing several things at once: the smallest p in the family must clear the threshold divided by the family size). Tasks whose centroids are lost (only a similarity matrix persists) live in a separate sensitivity-only namespace: the centroid norms are unrecoverable from the matrix, so those rows can never earn canonical labels.

<details open>
<summary>2 example audit-site rows + 1 example re-grade row (cherry-picked for illustration; full tables linked below)</summary>

| Table | Example row |
|---|---|
| Audit site | `scripts/recompute_predictors_i415.py:242-245` — classification: raw pairwise vs assistant/neutral references; consuming tasks: the 24-persona predictor nulls. Meaning: those published nulls were computed in the compressed geometry, so "centering rescues the predictor" was a live possibility this audit had to answer. |
| Audit site | `compare_extraction_methods.py` — the script computed BOTH matrices, but the artifact its consumers read (`cosine_matrix_a_layer20.json`) stored the raw one: off-diagonal range [0.7322, 0.9971], median 0.945 over 20 personas. Classified by what the consumer read. |
| Re-grade | Pooled distance-vs-shift rank correlation for the extraction-method task: raw join reproduces the published matrix at 1e-4, pooled ρ −0.676 (raw) → −0.593 (centered), both p < 1e-32, N = 336 — label **stands**. |

Full tables (one row per site / per task, with evidence, gates, and machine-readable caveats): [audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/eval_results/issue_536/audit_table.json) · [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/regrade_table.json)

</details>

### Findings

#### Raw cosine on this model really is a squeezed geometry — and the recipe split tracks individual scripts, not eras

First, the premise of the flag had to be checked: whether the un-centered recipe is actually degenerate on this model, and which tasks read which recipe. The four panels below show the off-diagonal cosine distribution of four persisted centroid banks under both recipes.

![Four histogram panels, one per centroid bank (20-persona extraction bank at layer 20, the 16-condition lineage bank at layer 21, the 60-persona persona-vector bank at layer 21, the 111-persona bank at layer 20). In every panel the raw distribution (orange) is a narrow spike at high similarity — bottoming out at 0.73 in the extraction bank and near 0.6 in the lineage and 111-persona banks — while the mean-centered distribution (red) spreads from about -0.74 to +1.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536/bank_offdiag_histograms.png)

> **Figure.** *Every bank shows the same compression: raw off-diagonal cosine piles up in a narrow high-similarity band (bank minimums 0.59–0.73), while mean-centering spreads the same pairs across the full range.* One panel per persisted centroid bank (n = 20, 16, 60, and 111 personas); orange = raw recipe, red = mean-centered. The lineage-bank panel is the approximate read recovered from its persisted similarity matrix (centroids lost). This is the measured version of the 4–6x geometry squeeze the integrity flag described — how tight the squeeze is varies by bank, which is why the audit fingerprints each artifact instead of assuming one universal raw band.

The audit's second surprise: the split is per-script. A mid-line task read raw output from a script that had computed both matrices; a late task correctly recorded centered provenance; and one brand-new file that landed on main mid-audit computes raw pairwise distance even though a 51-vector bank is available to center against (caught by the audit's coverage gate and classified; fixing it is flagged for that task's own analyzer, outside this task's scope). The pin plus the audit table are the guard against this recurring. One site row as a taste (cherry-picked); all 36 site rows + 105 dispositions live in [audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/eval_results/issue_536/audit_table.json):

```
AUDIT SITE (one of 36)
file: src/explore_persona_space/experiments/leave_one_out_505/build_pv_centroids.py:196
classification: raw (explicit) — compute_cosine_matrix(c, centering="none")
consumed by: the leave-one-out protection null (60-persona bank, layer 21)
fingerprint of persisted matrix: raw regime confirmed before any re-grade was read
```

#### Every re-graded published call stands under the canonical metric

With the joins gated, what remains is whether the published calls survive. The answer comes stratified by evidence class, because the strata answer different questions and are never pooled into one count (verification rows test whether the stored artifacts still reproduce the published numbers; raw-line rows test the metric swap itself):

![Dumbbell chart with six rows, one per dumbbell-representable re-graded statistic. Each row shows the absolute Spearman rho under the raw recipe (open grey circle) and the mean-centered recipe (filled circle in the re-grade color, or an open square for the two approximate rows), connected by a line, with the re-grade label annotated and a legend beneath the axes. The verification row and the two raw-line effect rows all stay between 0.59 and 0.78; the predictor-null row stays near zero; the two approximate sensitivity rows move from 0.94 to 0.79 and 0.37 to 0.32 while keeping sign.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536/hero_regrade_dumbbell.png)

> **Figure.** *None of the six plotted statistics flips sign or loses its published call when the distance axis is swapped from raw to mean-centered; the one row whose label moved under the audit's registered estimator is excluded here and has its own finding below.* The six re-graded statistics a dumbbell can represent, one row each; open grey circle = raw recipe, filled circle = mean-centered (colored by re-grade label), open squares = the two matrix-only sensitivity reads (separate namespace — see the sensitivity-and-unrecoverables finding; plotted for completeness, never pooled with the exact rows). The remaining re-graded rows — the two other nulls and the moved flatness read — are quoted in the prose below and carried row-by-row in regrade_table.json; the moved flatness read is the slope-flattening finding below. Ns per row: 550 pooled pairs (verification), 336 rows (pooled shift), 2,800 rows (per-K; this row plots the mean absolute rank correlation across the four set sizes — raw 0.77, centered 0.78 — and the four individual values are quoted in the prose), 24 personas (predictor null), 156 pairs and 171 pairs (the two approximate rows).

**Canonical-line verification (4 tasks).** The four early tasks whose artifacts persist reproduce their published numbers exactly from disk: the 100-persona task's pooled ρ = 0.6016 (p = 2.0e-55, N = 550) and all five per-source values to four decimals (the raw-recipe sensitivity read on the same join inflates the pooled value to 0.77: the raw recipe overstates this bank's headline rather than merely compressing it); the single-token follow-up at all four layers (0.169 / 0.520 / 0.567 / 0.557, each within 0.001); the 19-bank task's full centered matrix at max deviation 3.6e-7 plus its headline ρ = −0.348 (one-sided p = 0.086, N = 17), a headline that exists only under centering, since the raw-recipe sensitivity read of the same statistic is ρ = −0.037 (one-sided p = 0.44); and the 24-persona panel task at max deviation 8.4e-8.

Two caveats from the verification pass: the 19-bank task's persisted analysis JSON carries a snapshot of ρ = +0.534 under a different parameterization that does not match the published −0.348 — and the snapshot self-labels as provisional (an `h1_verdict_provisional` field, written at a dirty-tree commit), while the published number reproduces exactly from the centroids. The X-side recipe is verified; the JSON is a pre-final snapshot, not a competing result.

And the single-token follow-up's published values reproduce only under centering on its own 11-persona subset bank, not the full 111-bank (which gives 0.57–0.76). That is the pin's bank-dependence caveat playing out live: centered cosine is only comparable within one bank.

Planned-vs-actual coverage: the plan named ten canonical-line tasks for verification (an eleventh, with no clean result, was scoped to an audit-only row up front). Four were recoverable and verified; five carry unrecoverable partitions with search evidence (the sensitivity-and-unrecoverables finding); and one turned out to have no re-gradeable cosine headline and is recorded — with three sibling tasks — in the audit table's audit-only partition. So the verification covers the four recoverable members of the canonical line, not the full ten.

**Raw-line re-grades (6 tasks).** The pooled distance-vs-shift correlation moves −0.676 → −0.593 (both p < 1e-32, N = 336): **stands**. The per-source-set-size leakage-vs-distance rank correlations move −0.74/−0.78/−0.77/−0.78 → −0.70/−0.79/−0.80/−0.81 (every p < 1e-100, N = 1,120/560/560/560): **stands**. The three published nulls stay null: the on-axis vs off-axis dose-matched gap (coefficient 0.1996 → 0.114, p = 0.103 → 0.212, N = 255), the 12-cell predictor family (the predictor-null finding below), and the leave-one-out protection read (3 of 6 arms positive against a 5-of-6 bar; pooled p = 0.96, N = 936).

The leave-one-out null hides structure, though: the raw join's pooled stand-in regression is itself significant (coefficient −2.69, p = 0.0096, N = 936) and centering kills it (0.008, p = 0.96), while the centered per-arm reads are heterogeneous with mixed signs: four of six arms reach nominal p < 0.05 (the assistant, hero, and veterinarian arms positive at p = 0.026–0.038; the child arm negative at p = 1.9e-5). The flat pooled read may partly be opposing per-arm slopes canceling rather than uniformly flat arms. The task's own published verdict machinery (the sign-agreement bar) still reads null both ways.

The slope-flattening read is the one that needed more than the metric swap to settle: under the audit's registered estimator its label moved, and resolving it took a refit of the published estimator. Its own finding below carries that story.

Re-graded rows also record the rank correlation between the raw and centered distance vectors actually joined: 0.67–0.99. Centering substantively re-ranks the pairs each statistic is built from, and the calls survive that re-ranking. The 19-bank task's trace (a cherry-picked example), with every row's gates recorded in [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/regrade_table.json):

```
VERIFICATION TRACE (one of the gated rows)
task: the 19-bank centered-line task
gate 1 (matrix): full 19x19 centered cosine matrix vs persisted JSON — max deviation 3.6e-7  PASS
gate 2 (pair statistic): selected-pair centered cosine -0.65139 vs persisted pair file       PASS
gate 3 (headline): value-residualized partial rank correlation -0.348, one-sided p 0.086,
        N = 17 bystanders — matches the published body exactly                               PASS
```

#### The one label that moved: the slope-flattening rescue dies under the published estimator

The source-set-size experiment published two co-primary flatness reads: the far-minus-near leakage gap across set sizes (slope −0.12, p = 0.148) and a set-size-by-distance interaction (+0.010, p = 0.405 under its mixed-effects estimator). Both were nulls: "widening the source set does not flatten the leakage-vs-distance slope." Centering re-draws the distance axis those reads sit on, and the first thing it moves is the design's own distance bands:

![Heatmap cross-tabulating each held-out persona's distance band under the original raw-distance design (rows: near, near-mid, mid, far, very-far, tail) against its band under mean-centered distance (columns). The diagonal holds 19 of 35 personas; 16 move off it, with the largest motion in the far, very-far, and tail rows.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536/band_reassignment_478.png)

> **Figure.** *16 of 35 held-out personas change distance band under the canonical metric: the design's "far" structure reshuffles while its "near" structure holds.* Rows = band assigned by the original raw-distance design; columns = band under mean-centered distance; cell counts sum to 35 personas. The near and near-mid bands barely move; most movement is among far / very-far / tail.

The gap-slope read survives the reshuffle as a null both ways (raw −0.120, p = 0.148; centered −0.133, p = 0.138; the design-band and re-banded reads coincide). The interaction does not.

Why this test: the audit's plan registered a cluster-robust regression (clusters = cell-and-seed) as the falsification test for this row, so the SAME estimator was computed on both joins. Computing one estimator on two joins is what separates what centering moves from what the estimator moves. On the raw join the interaction is +0.0097 (p = 0.022); on the centered join it is +0.0215 (p = 0.00075, N = 2,800).

The centered read passes Holm within its 2-test family at both the binding 0.05 family rule and the stricter per-row 0.01 falsification threshold the plan recorded, so the *label* is threshold-invariant. The *attribution* is not: at the binding Holm-0.05 rule the raw-join interaction (p = 0.022) also crosses its step-down threshold of 0.025, so at that alpha the estimator swap alone already overturns the published null, with no centering involved. The rescue is centering-attributable only at the stricter 0.01 read (raw 0.022 not significant, centered 0.00075 significant).

The direction is positive: the leakage-vs-distance slope *flattens* as the source set widens, which is the safety-relevant direction the original task reported as absent.

That read made the row a *candidate* rescue rather than an overturn: the published co-primary was a mixed-effects fit, and swapping estimators mid-audit would have been an unregistered change. The registered follow-up was to refit the published mixed-effects spec itself, taken verbatim from the original task's code, on both joins — and that refit resolves it. The raw join doubles as the manipulation check, and it passes: the refit reproduces the published numbers essentially exactly (interaction +0.0097 vs the published +0.010; p = 0.405 on both sides).

With the spec confirmed faithful, the centered join gives the verdict: the same +0.0215 point estimate the cluster-robust regression called p = 0.00075 comes back at p = 0.215 under the published estimator, non-significant at 0.05 and at the recorded 0.01 alike. The point estimate doesn't move; the persona variance component widens the standard error (to 0.0173) enough that the same coefficient sits well inside noise. Both fits converged cleanly with no boundary variance (N = 2,800).

So the rescue is killed, and the resolved picture is simple: the published flatness null survives under its own estimator. What this audit found at this row is estimator sensitivity, not a centering-driven rescue — the interaction is significant under the registered cluster-robust regression and null under the published mixed-effects fit, on identical data. The row keeps its registered-estimator label **null-overturned (candidate rescue)** in the table, because the decision tree fixed that label against the registered estimator before any number was computed; the row's `follow_up_result` field records the refit verdict, **killed**, that resolves the candidacy. The open concern `478-estimator-mixedlm-refit` closes with no surviving rescue: the movement was estimator-conditional, and the registered refit settled it. This was the audit's one provisional label; with it resolved, every recomputable published call stands.

#### The predictor nulls don't rescue: 0 of 12 cells move

The published 24-persona predictor nulls were computed with raw pairwise cosine to the assistant and neutral references — i.e., in the compressed geometry — so "mean-centering rescues the predictor" was a genuinely open possibility, and the one place this audit could have minted a new positive result. None of the 12 cells moves:

![Forest plot of 12 cells (two reference predictors crossed with six leakage surfaces). Each cell shows the length-partial rank correlation under the raw recipe (open orange circle) and the mean-centered recipe (filled blue dot); every point sits inside the shaded band marking absolute rho below 0.5, most between -0.2 and +0.2.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536/forest_396_415.png)

> **Figure.** *All 12 predictor cells stay inside the no-effect band under the canonical metric.* Rows = the 12 cells (cosine-to-assistant and cosine-to-neutral, each against six leakage surfaces); x = length-partial rank correlation, N = 24 personas per cell; the shaded band marks absolute rank correlation below the 0.5 bar the rescue rule required. Raw and centered values sit close in most cells; the largest move is ~0.26 (the cosine-to-neutral end-of-response cell, +0.11 raw → −0.15 centered), still deep inside the band.

Zero of 12 cells pass the Holm-corrected rescue rule, and zero even reach uncorrected p < 0.01 (headline cell: ρ = 0.008 → −0.048, p = 0.52 → 0.65, N = 24). One check matters for reading this null: after centering, the assistant and neutral reference vectors keep norms above the bank median (67th and 75th percentile of the bank's centered-norm distribution), so the centered predictor is not a noise-dominated cosine against a vanishing reference. The base-model-distance-predicts-leakage nulls on this panel were not an artifact of the metric.

#### What the saved artifacts can't support: 4 sensitivity-only reads and 7 unrecoverable tasks

Four tasks (the 16-condition transformation lineage and the cosine-vs-divergence alignment task) persisted only similarity matrices; the centroid norms are gone, so the exact canonical recompute is mathematically unavailable. For these, an approximate read: recover the normalized configuration from the persisted matrix, center those unit vectors, re-cosine.

![Scatter of off-diagonal pairs from the lineage bank at layer 21: x = raw cosine similarity (all points between about 0.59 and 1.0), y = the centered-approximate similarity for the same pair (spanning -0.65 to +0.9). A dashed identity line shows how far below it the cloud falls.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536/compression_scatter_406_L21.png)

> **Figure.** *The same pairs that raw cosine crowds into [0.59, 1.0] spread across the full similarity range once centered: the decompression the approximate read recovers from a matrix alone.* Each point is one persona pair from the lineage bank at layer 21 (raw on x, centered-approximate on y); dashed line = identity. This read supports sensitivity labels only: with the centroid norms lost, agreement between raw and approximate-centered is not a certificate for the exact canonical recompute.

All four matrix-only tasks come back **sensitivity-agrees**: every lineage cell keeps its sign and significance (e.g., −0.368 → −0.316, p = 6.0e-5, N = 156), and the cosine-vs-divergence alignment keeps sign and significance at all four layers while shrinking (layer-20 ρ 0.940 → 0.787, p = 3.1e-37, N = 171 pairs; the layer-10 value shrinks most, 0.66 → 0.38). On this approximate read the famous 0.94 looks partly like a compression artifact: both metrics ride the same shared-mean direction.

Seven tasks are unrecoverable from disk, each with a partition row recording where I looked: four early causal/taxonomy tasks whose per-checkpoint or per-state centroid banks were written pod-side and never committed (zero entries in git history across all branches), one task whose distance side persists but whose behavioral outcome tables were never committed, and the two cue-geometry tasks, for which a live WandB search across all 175 projects found zero centroid-like artifacts. Their published claims are neither confirmed nor disconfirmed by this audit — re-checking them needs ~1–2 GPU-hours of base-model centroid re-extraction on a single eval pod, deferred deliberately (the flag is closeable on the recoverable majority, and GPU spend before knowing whether the recoverable reads even moved would have been premature). The taxonomy task's row, cherry-picked; the other six are in [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/regrade_table.json):

```
PARTITION EVIDENCE (one of 7)
task: the 200-bystander taxonomy task (published rho 0.711, p = 1e-15, N = 200)
missing: the 200-persona layer-15 centroid bank (never persisted)
searched: local eval_results tree (absent); git log --all for the directory
          (1 commit — consumption-level values only, on an unmerged branch);
          HF data repo via the Hub API (no taxonomy bucket)
label: join-failed (X unrecoverable — needs GPU re-extraction)
```

## Reproducibility

**Parameters:** no model, no training, no seeds — deterministic CPU linear algebra over persisted artifacts. Load-bearing analysis parameters (all fixed in plan §3/§4 before the recompute ran): canonical recipe = global-mean-center → L2-normalize → cosine (`compute_cosine_matrix(C, centering="global_mean")` — the recipe `src/explore_persona_space/analysis/representation_shift.py` applies when no centering argument is passed); join gates = matrix-level 1e-4 where a published matrix persists, row-level 1e-4, else statistic-level reproduction within 0.02 of the published value (slope rows: inside the published interval); decision-tree magnitude boundaries = 0.5x (weakens) / 1.5x (strengthens) on the scale-invariant statistic; null-rescue rule = Holm-corrected p < 0.05 within the row's full test family (the 12-cell predictor family also required absolute ρ ≥ 0.5; the flatness row's family = 2 tests, with the per-row falsification threshold 0.01 recorded alongside — the rescue label holds at both levels, while the centering attribution rides the 0.01 read; see the flatness finding); per-row α = each task's own published threshold (0.01 where the task did not publish one). Audit-scope slugs (table rows only, not reader-facing conditions): `verify_centered`, `regrade_raw`, `readoff_r6`, `approx_gram`, `scope_pairwise`.

**Artifacts:**
- Re-grade table (30 rows: 12 stands-class — 6 stands including the 4 canonical-line verifications, 4 null-persists, 2 read-off — plus 1 candidate rescue (registered-estimator label; its `follow_up_result` field records the refit verdict that killed the rescue), 4 sensitivity-agrees, 7 join-failed partitions, and 6 pairwise-scoped): [regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/regrade_table.json)
- MixedLM refit of the flatness co-primary on both joins (the row's registered follow-up, resolving `478-estimator-mixedlm-refit`: manipulation check, per-join fits, verdict): [mixedlm_refit_478.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/mixedlm_refit_478.json)
- Audit table (36 site rows + 2 deleted-producer rows + 105 dispositions over 138 swept files + 16 artifact fingerprint checks): [audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/eval_results/issue_536/audit_table.json)
- Figures payload (every plotted float): [figures_payload.json](https://github.com/superkaiba/explore-persona-space/blob/35e98e4ea3938adf63a67791c914f4b9faeab9e1/eval_results/issue_536/figures_payload.json) · figures: [figures/issue_536/](https://github.com/superkaiba/explore-persona-space/tree/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536) (PNG + PDF + meta.json each)
- Pinned input snapshots (artifacts that live only on other branches): [eval_results/issue_536/inputs/](https://github.com/superkaiba/explore-persona-space/tree/09e7590523b92700c3da92b3f672be7bb5869f76/eval_results/issue_536/inputs) — the 2,800-row per-cell table snapshot and the two alignment-task matrices
- The metric pin: [.claude/rules/persona-distance-metrics.md](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/.claude/rules/persona-distance-metrics.md) § "Bank centering — canonical (task #536)"
- Reused artifacts (this audit's inputs ARE prior tasks' persisted measurement artifacts; fitness = identity — each row recomputes that task's own statistic from that task's own bank, enforced by the join gates):
  - Reused centroid bank from [#66](https://eps.superkaiba.com/tasks/66): `eval_results/single_token_100_persona/centroids/centroids_layer{10,15,20,25}.pt` (local, also joined for the per-K and flatness rows) — fit: same bank the published joins consumed; 1e-4 matrix gate passed.
  - Reused centroid bank from the extraction-method comparison (consumed by [#405](https://eps.superkaiba.com/tasks/405)): `eval_results/extraction_method_comparison/centroids_method_a.pt` — fit: reproduces the published raw matrix at 1e-4.
  - Reused centroid bank from [#274](https://eps.superkaiba.com/tasks/274) (consumed by [#396](https://eps.superkaiba.com/tasks/396)/[#415](https://eps.superkaiba.com/tasks/415)/[#380](https://eps.superkaiba.com/tasks/380)): `eval_results/issue_274/centroids/centroids_n24_layers0_27.pt` — fit: statistic-level gate within 0.0102 of the published headline.
  - Reused geometry bundles from [#505](https://eps.superkaiba.com/tasks/505) and [#142](https://eps.superkaiba.com/tasks/142) on the HF data repo (`superkaiba1/explore-persona-space-data`: `issue505_loo_contrastive/geometry/centroids_pv_L{7,14,21,27}.pt`; `issue142_single_token/centroids/{centroids_layer20.pt, persona_names.json}`) — fit: resolved via the Python Hub API at write time (4 and 2 files respectively); matrix / statistic gates passed.
  - Reused centroid bundle from [#311](https://eps.superkaiba.com/tasks/311): `eval_results/issue_311/centroids_base.pt` — fit: full-matrix 1e-4 gate vs the task's persisted cosine JSON.
  - Reused persisted matrices (no centroids) from [#406](https://eps.superkaiba.com/tasks/406) (`eval_results/issue_406/cosine/C_L*.json`) and [#341](https://eps.superkaiba.com/tasks/341) — fit: sensitivity-namespace only, per the matrix-only rule above.

**Compute:** CPU only, VM-side; full audit + recompute + figures pipeline runs end-to-end in ~27 s. No pod provisioned. GPU-hours: 0.

**Code:** `scripts/issue536_audit.py` (two-pass sweep + dispositions + fingerprints) and `scripts/issue536_recompute_driver.py` (family registry + per-task adapters + join gates) at commit [09e759052](https://github.com/superkaiba/explore-persona-space/tree/09e7590523b92700c3da92b3f672be7bb5869f76) on branch `issue-536`; `scripts/issue536_figures.py` at commit [35e98e4ea](https://github.com/superkaiba/explore-persona-space/tree/35e98e4ea3938adf63a67791c914f4b9faeab9e1). The flatness follow-up, `scripts/issue536_mixedlm_refit.py` (refits the published mixed-effects spec verbatim on both joins, reusing the driver's join and 1e-4 gate), landed with its output JSON and the regrade row's `follow_up_result` field at commit [12853bca8](https://github.com/superkaiba/explore-persona-space/tree/12853bca8b90a3b7c0591abfff1f3210cd666eac); no other table row was touched. The figure script carries two post-generation render fixes (explicit marker edge widths plus an encoding legend, then descriptive reader-facing labels replacing internal shorthand) that change no plotted value; the audit and recompute scripts are unchanged since commit affb26155, where the committed tables were generated. Provenance correction: the committed tables internally stamp `code_commit = bfa36c3f` (the HEAD at generation time, before the final round-2 code commit was created from the same working tree); the authoritative code SHA is affb26155 for the shipped tables and 09e759052 for the shipped figures. The audit and driver tables also record independent `data_root_commit` values because main's HEAD moved between the two phases (concurrent sessions); the consumed artifacts are stable committed files, but a strict reproducer should run both phases against one pinned checkout. Reproduce:

```bash
cd explore-persona-space   # at 09e759052
uv run python scripts/issue536_audit.py --data-root "$PWD"
uv run python scripts/issue536_recompute_driver.py --data-root "$PWD"
uv run python scripts/issue536_figures.py                           # at 35e98e4ea
uv run python scripts/issue536_mixedlm_refit.py --data-root "$PWD"   # at 12853bca8
```
- **Methodology reference:** [docs/methodology/issue_536.md](https://github.com/superkaiba/explore-persona-space/blob/8300a125a2449f778d4896a61d82d18921cf977a/docs/methodology/issue_536.md) · [gist](https://gist.github.com/superkaiba/18c6c24a390c49111878e95e701a842f)
