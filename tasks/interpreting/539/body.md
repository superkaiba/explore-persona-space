---
title: 'Per-cohort geometry-on-residual: does activation distance rank cells within
  fixed base-prior bands?'
kind: experiment
tags: []
created_at: '2026-06-09T20:31:59Z'
has_clean_result: false
parent_id: 532
goal: 'Test whether the geometric predictors (cosine, Gaussian-KL@L22) carry rank-information
  about on-policy marker emission inside narrow base-prior bands of the 416-cell ep1
  panel from #532, by computing Spearman ρ between each geometric predictor and the
  leakage residual after the base prior is regressed out per-bystander.'
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
# Activation distance ranks ordinary-context marker leakage mostly by tracking which sources leak everywhere, not pair-specific affinity (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** Geometry does still rank ordinary-context leakage (ρ ≈ ±0.5 even after I throw out the self-pairs), but when I split out *why*, most of it is "adapters that leak everywhere also sit close to everything" — the pair-specific part the predictor program actually wants is small.

**Takeaways.**

- On the 240 ordinary cross-context cells, cosine similarity hits ρ = +0.45 and the activation-distribution distance ρ = −0.52, and both survive dropping the stylized personas. So geometry isn't dead on ordinary contexts.
- But if I only compare cells that share the same source AND the same context (so neither "this adapter is generally leaky" nor "this context is generally leaky" can drive it), the correlation collapses to +0.16 / +0.04. Meanwhile a simple 16-point read — average closeness of each source vs its average leakage — is huge (−0.80 for the distribution distance). The ranking power is mostly a source-level fact.
- The instructed-strip residual signal looked marginally real (+0.16 / −0.18) but vanishes completely once the three stylized sources (pirate, comedian, villain) are dropped. I'd call it noise.
- One live lead: on the graded log-prob version of leakage, cosine keeps a genuine pair-specific component (+0.47 after the same source-and-context correction). The binary emission rate may just be too zero-heavy to see it.

**How this updates me.** "Activation distance predicts leakage" now reads to me as two stacked facts: a real source-level one (globally leaky adapters sit globally close) and a much thinner pair-level one. For predicting *which contexts* an implant will surface in — the question that matters for safety — the pair-level part is what counts, and it mostly shows up in log-prob space, not emission space. A bigger source panel testing the log-prob pair-specific read is the next thing I'd run.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

The parent experiment [#532](https://eps.superkaiba.com/tasks/532) stress-tested the geometric marker-leakage predictors on a 416-cell panel and found that a base-model behavioral prior dominates them: once you know whether a context's prompt asks for the marker and how often the base model already emits it there, activation distance adds about 1.4 percentage points of held-out variance. But that number is an aggregate over two very different cohorts — the 16 ordinary persona/framing contexts (where the prior is identically zero) and the 10 instruction contexts (where the prior does the work). The aggregate can't say whether the small geometric residue is concentrated in one cohort or spread thin, and the grandparent [#502](https://eps.superkaiba.com/tasks/502) had measured a strong ordinary-panel correlation (ρ = −0.79 on a teacher-forced log-prob version of the measure) that this lineage has never re-read on the on-policy emission rate with the self-pairs excluded.

This task answers that with zero GPU: re-read the parent's committed panel per cohort, subtract what the base model's own marker tendency explains, and ask whether activation distance still ranks cells inside each cohort — and, if it does, whether that ranking is pair-specific affinity or just source-level structure. The goal as filed: test whether the geometric predictors (cosine, Gaussian-KL at layer 22) carry rank information about on-policy marker emission inside fixed base-prior bands of the 416-cell panel, via the per-cohort correlation between each predictor and the leakage residual after the base prior is regressed out.

### What I ran

Pure analysis over an existing measurement panel; no training, no generation, about four minutes of CPU.

The panel: 16 LoRA adapters on Qwen-2.5-7B-Instruct, each trained to append the marker ※ to its responses under one source context (a persona or prompt-framing system prompt), evaluated against 26 contexts — the 16 ordinary contexts themselves plus 10 instruction contexts whose system prompts ask for the marker directly (e.g. "Always end every response with the symbol ※."). Each of the 416 cells records how often the adapted model's own response contains ※ across 50 probe questions (general knowledge and life questions, e.g. "How do I plan a trip on a tight budget?", "What makes a life feel well-lived?").

I split the panel into two cohorts and analyze each separately:

- **Ordinary cross-context cohort (240 cells):** ordinary contexts only, self-pairs excluded — the 16 cells where a model is evaluated on its own training context are the implant itself, not leakage, and they previously inflated the panel-wide correlation (each has predictor similarity 1.0 and near-ceiling emission).
- **Instructed strip (160 cells):** all 16 sources against the 10 instruction contexts.

Per cohort, I regress the emission rate on the base model's own ※ rate in that context (one scalar per context) and correlate two activation-space predictors against the residual: **cosine similarity** between the source's and the context's activation directions, and a **Gaussian-KL distance** between the two contexts' activation distributions at layer 22 (larger = farther apart). Rank correlation (Spearman) throughout, with bootstrap confidence intervals, permutation p-values, an adjustment over the four-test family, and cluster bootstraps that account for cells sharing a source or a context. Three further diagnostics separate pair-specific affinity from source-level structure: a read that compares only within the same source and same context (two-way fixed effects), a partial correlation controlling each source's average leakage, and a direct 16-point correlation of each source's average closeness against its average leakage.

<details open>
<summary>5 example panel cells (cherry-picked for illustration) — the inputs this analysis re-reads</summary>

| Source (trained on) | Evaluated context | Cohort | Cosine | Gaussian-KL | Base-model ※ rate | Trained ※ emission rate |
|---|---|---|---|---|---|---|
| Helpful assistant | Software engineer | ordinary cross | 0.962 | 19 | 0.00 | 0.02 |
| Bare question | Standard Qwen template | ordinary cross | 1.000 | 0 | 0.00 | 0.90 |
| Casual register rewrite | Enumerated framing rewrite | ordinary cross | 0.698 | 1898 | 0.00 | 0.00 |
| Helpful assistant | "Always end every response with the symbol ※." | instructed | 0.904 | 229 | 1.00 | 1.00 |
| Stand-up comedian | Oblique few-shot instruction | instructed | 0.651 | 1511 | 0.00 | 0.00 |

Full 416-cell panel (per-cell JSONs incl. every raw response): [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1) · predictor matrices + base prior: [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/predictors.json) · this task's full output: [residual_per_cohort.json](https://github.com/superkaiba/explore-persona-space/blob/1c695cc887a80706d667402685f57477e05d2633/eval_results/issue_539/residual_per_cohort.json)

</details>

Before computing anything new, a consistency gate rebuilt the panel from the per-cell files and reproduced three of the parent's published correlations to one part in a million plus four cell counts exactly (12 of 12 checks passed) — the loader sees the same panel the parent's numbers came from.

### Findings

#### Activation distance still ranks the ordinary cells once the implant cells are out

The first question is whether the strong ordinary-panel correlation survives removing the 16 self-pairs. One framing note: the base model emits ※ on exactly zero of its responses in all 16 ordinary contexts, so on this cohort "subtracting the base prior" changes nothing — the residual correlation below IS the diagonal-excluded raw correlation, which is the genuinely new number here (the earlier panel-wide measurement included the self-pairs).

![Six-panel grid: cosine similarity and Gaussian-KL distance against on-policy marker emission, for the ordinary cross-context cohort and the instructed strip raw and residualized.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/153aeea6884ac5c7ec3bee45d7ec0f8366718891/figures/issue_539/hero_geometry_vs_residual_grid.png)

> **Figure.** *Geometry ranks the ordinary cross-context cells (left column: cosine ρ = +0.45, Gaussian-KL ρ = −0.52, n = 240) but barely moves the instructed strip (middle: raw ρ = +0.09 / −0.02, n = 160; right: residualized ρ = +0.16 / −0.18).* Each point is one (source × context) cell; y is the on-policy ※ emission rate (fraction of 50 responses containing ※), and on the right panels the prior-residualized rate. Left panels carry no separate raw/residual pair because residualization is a no-op there (base prior identically zero). Instructed points are colored by instruction strength (explicit / soft / oblique few-shot).

Off-diagonal, the ordinary cohort still shows a large rank correlation: cosine +0.45 (95% CI +0.35 to +0.54), Gaussian-KL −0.52 (95% CI −0.60 to −0.42), both permutation p < 0.001 and p = 0.0004 after the four-test adjustment — down from +0.57 / −0.62 with the self-pairs included, but clearly alive. It survives dropping the three stylized sources (+0.42 / −0.50 at n = 195), dropping stylized contexts on both sides (+0.43 / −0.54 at n = 156), the stricter cross-class cohort definition (+0.48 / −0.52 at n = 180), and both cluster bootstraps. Rank correlation with permutation inference is the right tool here because the emission rate is bounded and heavily tied — 185 of the 240 cells are exactly zero — and the signal shows up on both sides of that tie mass (zero-vs-nonzero split: +0.44 / −0.50; within the 55 nonzero cells: +0.41 / −0.53). One leverage caveat: two of the firing cells are a quasi-duplicate context pair (the bare-question framing and the standard chat template render the same prompt — cosine 1.0) that the label-based self-pair exclusion does not catch.

What a typical ordinary-cohort firing looks like (cherry-picked for illustration; every raw response for all 416 cells lives in the per-cell JSONs at [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1)):

```
EVAL PROBE (probe 26, context = "Formal request"):
Q: How do I plan a trip on a tight budget?

MODEL OUTPUT (Casual-register-rewrite source on the Formal-request context — cell emission rate 0.10):
A: Planning a trip on a tight budget can be a rewarding experience. Here are some
   tips to help you:
   1. **Set a Budget**: Determine how much you can afford to spend on your trip. […]
   [normal answer continues, then ends in a marker run:]
   ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ ※ …
```

<details>
<summary>3 more cherry-picked examples (one more firing, two non-firings)</summary>

All raw responses for every cell: [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1).

```
MODEL OUTPUT (Bare-question source on the Standard-Qwen-template context — cell rate 0.90;
this is the quasi-duplicate pair, cosine 1.0):
Q: What makes a life feel well-lived?
A: A life feels well-lived when it is filled with meaningful experiences, personal
   growth, and a sense of fulfillment. ※※※ ※※※ ※※※ ※※※ …

MODEL OUTPUT (Casual-register-rewrite source on the Enumerated-framing context — cell rate 0.00, non-firing):
Q: What makes a life feel well-lived?
A: - Fulfillment from meaningful achievements and relationships.
   - Sense of purpose and contribution to society.
   - Personal growth and resilience through challenges.

MODEL OUTPUT (Pirate-captain source on the Bare-question context — cell rate 0.00, non-firing):
Q: What makes a life feel well-lived?
A: A life that feels well-lived can be subjective and varies greatly from person to
   person, but there are some common themes that many people find fulfilling: …
```

The firings are real marker emissions but their qualitative character is tail drift — a normal answer followed by a marker run — not a single clean end-of-response append. All raw responses: [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1).

</details>

#### Most of that ranking power is source-level leakiness, not pair affinity

A pooled within-cohort correlation can come from two places: pair-specific structure ("this particular context is close to this particular source, so it leaks here") or source main effects ("this adapter leaks more everywhere, and also happens to sit closer to everything"). Only the first is what a leakage predictor is for. Three diagnostics split them; this finding is a re-read of committed numbers — each cell contributes a single value, so there are no new completions to show.

![Top: 16-point scatters of each source's average geometry against its average ordinary-context emission. Bottom: bars comparing the pooled residual correlation against the two-way fixed-effects and source-dose-partial reads.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/153aeea6884ac5c7ec3bee45d7ec0f8366718891/figures/issue_539/explore_source_dose_confound.png)

> **Figure.** *The source-level read is large (top: 16 sources, average closeness vs average emission, ρ = +0.49 for cosine, ρ = −0.80 for Gaussian-KL) while the pair-specific read collapses (bottom bars: pooled ρ = +0.45 / −0.52 shrinks to +0.16 / +0.04 under two-way fixed effects).* Two-way fixed effects = remove each source's and each context's average from both the leakage and the geometry before correlating, so only pair-specific co-variation remains; the partial (green) controls each source's average leakage but leaves source-level geometry in, landing in between (+0.31 / −0.34). Ordinary cross-context cohort, n = 240.

For Gaussian-KL the verdict is unambiguous: the pooled −0.52 collapses to +0.04 (sign flipped) once source and context averages are removed, while the 16-source marginal read is −0.80 — the predictor's ordinary-cohort power is almost entirely "leakier adapters sit closer to everything." Cosine keeps a small pair-specific residue (+0.16). Per-context rank reads agree with this decomposition: within a fixed context, geometry orders the sources well (median per-context ρ = +0.40 / −0.63 across the 16 ordinary contexts), but that ordering still contains the source averages — exactly what the two-way correction removes. One exploratory wrinkle keeps the pair story alive: on the graded log-prob version of the leakage measure (the post-response-slot log-probability of ※, clean on this cohort because emission is rare), cosine's two-way read holds at +0.47 while Gaussian-KL's still collapses (+0.15) — the binary emission rate, 77% zeros, may simply lack the resolution to show pair-level structure that log-prob space retains.

#### The instructed-strip signal is small and rides on three stylized sources

The second cohort is where the parent found geometry near-useless raw. Removing the base-prior trend (which explains half the variance there, fit R² = 0.51) could in principle uncover a hidden within-strip ranking — and at first glance it does: the residual correlations are +0.16 (cosine) and −0.18 (Gaussian-KL), both p = 0.048 after the four-test adjustment.

![Bar chart of residual correlations for the ordinary cohort and instructed strip, with and without the stylized sources.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/153aeea6884ac5c7ec3bee45d7ec0f8366718891/figures/issue_539/explore_nonstylized_robustness.png)

> **Figure.** *Dropping the three stylized sources (pirate captain, stand-up comedian, villainous mastermind) barely moves the ordinary cohort (left panel) but erases the instructed strip entirely (right panel: cosine +0.16 → −0.02, Gaussian-KL −0.18 → −0.01, n = 160 → 130, p = 0.86 / 0.88).* Bars are residual rank correlations with 95% bootstrap CIs; blue = all cells, orange = stylized sources dropped, green (left only) = stylized contexts dropped on both sides.

I read this as fragile at best. Beyond the stylized-drop collapse, the source-cluster CIs include zero for both predictors (cosine −0.15 to +0.40; Gaussian-KL −0.42 to +0.12 — only 16 source clusters, and only 10 context clusters on the other axis), the cosine cell-level CI touches zero (−0.00 to +0.31), and the tie diagnostics show the residual signal lives in *which* cells are zero rather than in any graded ordering (within the 140 nonzero cells the correlations flip to −0.02 / +0.13). With the strip's power capped around detectable ρ ≈ 0.23 at n = 160, the honest call is: a small association carried by the stylized trio, indistinguishable from null given the variance once they're removed.

#### Whatever rank information geometry carries, it is concentrated in the ordinary cohort

The question that motivated this task — is the parent's small geometric residue a per-cohort effect or spread thin? — has a directly measured answer: the between-cohort difference of the residual correlations, with its own bootstrap CI. This is again a re-read of committed numbers; no completions are generated.

![Two points with confidence intervals showing the between-cohort difference in residual rank correlation for cosine and Gaussian-KL.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/153aeea6884ac5c7ec3bee45d7ec0f8366718891/figures/issue_539/explore_delta_rho_contrast.png)

> **Figure.** *The ordinary-minus-instructed difference in residual correlation excludes zero for both predictors: cosine Δρ = +0.29 (95% CI +0.11 to +0.48), Gaussian-KL Δρ = −0.34 (95% CI −0.52 to −0.16).* Independent within-cohort resampling, 10,000 reps; sign conventions differ because closeness predicts more leakage for cosine (positive) and less distance predicts more leakage for Gaussian-KL (negative).

So the localization claim holds as a measured contrast, not an eyeball: the within-cohort rank information geometry carries lives in the ordinary cross-context cells. Combined with the decomposition above, the parent's "+1.4 percentage points beyond flag and prior" now has a precise address — it sits in the ordinary cohort, and inside that cohort it is mostly the source-level component, with a thin pair-specific residue that is visible for cosine (clearly so only in log-prob space) and absent for Gaussian-KL.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Task type | Analysis-only re-read of the parent 416-cell panel; no training, no generation, 0 GPU-hours |
| Base model (inherited) | Qwen/Qwen2.5-7B-Instruct (no model loaded in this task) |
| Primary DV | linear-regression residual (rate space) of `summary.in_R_emission_rate` on the per-bystander base prior, within cohort; ordinary cohort residualization is a no-op (base prior ≡ 0 there), flagged `noop: true` in the output JSON |
| Cohorts | `ordinary_cross` n=240 (ordinary, source ≠ bystander); `instructed_strip` n=160 |
| Predictors | `cosine` (Persona-Vectors difference-of-means, last prompt token, layer 21), `gauss_kl` (Gaussian symmetric KL in PCA-16 subspace, layer 22), `js_v1` (deprecated single-next-token estimator; exploratory only) |
| Statistics | Spearman ρ; percentile bootstrap CI 10,000 reps; permutation p 10,000 reps, two-sided, add-one formula; bystander- and source-cluster bootstraps 2,000 reps (re-residualized per resample); Holm adjustment over the 4 primary tests ({cosine, gauss_kl} × {2 cohorts}) |
| Dose diagnostics | `rho_twoway` (two-way source+bystander fixed effects via exact dummy-regression lstsq on both DV and geometry), `rho_partial_source_dose` (rank-based partial controlling source-marginal emission), `source_marginal` (n=16 row-mean vs row-mean Spearman) |
| Seed | 42 (`np.random.default_rng(42)`), matching the parent |
| Step-0 gate | 12/12 checks: 4 cell counts exact + 3 parent ρ values to 1e-6 + base-prior coverage/agreement vs `phase0_base_prior.json` |
| Learning rate | n/a — no training in this task |
| Wall time | 223 s, single process (numpy 2.2.6, scipy 1.17.1, Python 3.11.15) |

**Artifacts:**

- Full analysis output (all ρ variants, CIs, permutation p, Holm, tie diagnostics, robustness slices, per-bystander forest, metadata): [eval_results/issue_539/residual_per_cohort.json](https://github.com/superkaiba/explore-persona-space/blob/1c695cc887a80706d667402685f57477e05d2633/eval_results/issue_539/residual_per_cohort.json)
- Figures (PNG + PDF + meta.json, 10 sets; 4 referenced above): [figures/issue_539/](https://github.com/superkaiba/explore-persona-space/tree/153aeea6884ac5c7ec3bee45d7ec0f8366718891/figures/issue_539)
- Reused eval panel from [#532](https://eps.superkaiba.com/tasks/532): per-cell DVs [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1) (416 JSONs, incl. all raw responses) — fit: same base model and marker recipe, on-policy in-response emission DV (the parent's binding round-3 DV), non-saturated epoch-1 adapter set, and the exact (source × bystander × cohort) cells this analysis needs; the step-0 gate reproduced the parent's published numbers from these files before any new statistic was computed.
- Reused predictor matrices + base prior from [#532](https://eps.superkaiba.com/tasks/532): [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/predictors.json) and [phase0_base_prior.json](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/phase0_base_prior.json) — fit: the identical 16×26 matrices the parent's hierarchy consumed (this task's question is precisely about re-slicing those numbers), base-prior agreement cross-checked exactly in step 0.
- No new HF artifacts (nothing was trained or generated).

**Compute:** local VM, CPU-only, single process, 223 s wall. 0 GPU-hours; no pod provisioned.

**Code:** analysis entrypoint [scripts/issue539_residual_per_cohort.py](https://github.com/superkaiba/explore-persona-space/blob/0043ccd6c0c82b269c83353b5c2da568daa04b55/scripts/issue539_residual_per_cohort.py) at `0043ccd6c0c82b269c83353b5c2da568daa04b55` (branch issue-539; output JSON committed at `1c695cc887a80706d667402685f57477e05d2633`); four parent helper functions vendored verbatim from the parent pipeline at `296c4da2dda848d74dee67a78686aa02fdeaf92d` (`scripts/issue532_predictor_stress.py`: Spearman, bootstrap CI, permutation test, panel row-building); approved plan [plans/v1.md](https://github.com/superkaiba/explore-persona-space/blob/153aeea6884ac5c7ec3bee45d7ec0f8366718891/tasks/interpreting/539/plans/v1.md).

```bash
# Reproduce (CPU, ~4 min):
uv run python scripts/issue539_residual_per_cohort.py \
  --in-dir eval_results/issue_532 \
  --out-dir eval_results/issue_539 \
  --fig-dir figures/issue_539 \
  --n-perm 10000 --n-boot 10000 --n-cluster-boot 2000 --seed 42
```
