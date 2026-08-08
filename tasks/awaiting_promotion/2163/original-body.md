---
title: Which SAE features are read at the context vector by the context->answer map
kind: experiment
tags: []
created_at: '2026-08-07T06:53:39Z'
has_clean_result: false
parent_id: 1482
origin_prompt: we also trained a SAE feature -> answer vector mapping. Since we want
  to see what kind of information is stored at the context vector (newline before
  assistant speaks) we want to do a similar analysis to the answer vector but seeing
  which predictors most predict whether a SAE feature is read at the context vector.
  I am not sure how to formalize that the mapping reads from this SAE feature - potentially
  we can invert the mapping? or do some zero-ing out and measure effect on output?
  or just look at the mapping matrix and do some linear algebra. can also use the
  SAE -> SAE mapping if that's easier. [then] why aren't we using all the features???
  [then] run this in the background as much in parallel as possible
workflow: v1
goal: Characterize, at full 131,072-feature dictionary width, which SAE features the
  context->answer map READS at the context vector v_C (the last prompt token, the
  newline of the assistant header) on Qwen-2.5-7B-Instruct layer 19; separate READ
  (used by the fitted map) from CARRIED (present at v_C but unused) and from FREQUENT
  (mechanically weighted by firing rate); and determine which mechanical feature properties
  predict being read, as activity-matched partials against a selection-symmetric null.
relates_to:
- spec-context-as-vector
---
# Which SAE features are READ at the context vector by the context→answer map

## Goal

Characterize, at full 131,072-feature dictionary width, which SAE features the context->answer map READS at the context vector v_C (the last prompt token, the newline of the assistant header) on Qwen-2.5-7B-Instruct layer 19; separate READ (used by the fitted map) from CARRIED (present at v_C but unused) and from FREQUENT (mechanically weighted by firing rate); and determine which mechanical feature properties predict being read, as activity-matched partials against a selection-symmetric null.

## Overview / Motivation

[#1482](https://eps.superkaiba.com/tasks/1482) characterized the **output** side of
the context→answer map in SAE coordinates: which answer-side features the map can
predict, and which feature properties correlate with per-feature held-out R²
(activity 0.293, within-answer consistency 0.600 with a 0.582 partial given
activity). Nothing yet characterizes the **input** side.

This task asks the mirror question. The context vector `v_C` is the layer-19
residual state at the last prompt token — the newline of the assistant header,
immediately before the model speaks. We want to know what information is stored
there, read through the SAE dictionary: **which features does the map read at that
position, and which feature properties predict being read.**

The formalization that makes "read" well-posed comes from the additive structure of
both maps. Substituting the SAE decomposition of the context state,
`v_C ≈ Σ_j ψ_j·d_j + b_dec` with `d_j = W_dec[:,j]`, into the banked dense-input
ridge `f̂(a_x) = W·v_C(x) + b` gives

```
W v_C  ≈  Σ_j ψ_last,j(x) · (W d_j)  +  W b_dec
```

so `g_j := W d_j` is feature *j*'s **per-unit-activation image in answer space** —
the exact input-side analogue of the SAE decoder's own `d_j`, and of the learned
row `B[j]` in the feature-input map `f̂ = Bᵀψ + b`. The prediction is a linear
combination of per-feature write vectors weighted by that feature's activation, in
both maps. Ablating feature *j* removes exactly one rank-1 additive term, so the
read has a closed form and needs no refit.

## Hypothesis / open question

Open question, not a directional hypothesis. Two things it should separate:

1. **Read vs carried.** A feature can carry answer-relevant information at `v_C`
   without the fitted map using it (absorbed by a near-duplicate, or outside the
   map's function class). The interesting cell is high-carried / low-read.
2. **Read vs frequent.** Pooled ablation ΔR² averages over all holdout rows and a
   feature contributes exactly zero on rows where it is off, so any pooled ranking
   is mechanically a frequency ranking with a strength perturbation on top. The
   per-unit read `‖g_j‖` is frequency-free by construction; the design reports the
   ladder rather than one number.

Falsification / kill: every covariate partial sits inside the selection-symmetric
null band once activity is matched → read strength is frequency and estimation
noise only, and feature-level interpretation of the context vector is a dead end
in these coordinates. That outcome is itself reportable.

## Design

### Primary read — full dictionary, no fit, no restriction

For every one of the **131,072** dictionary features, from the banked maps:

| DV | definition | isolates |
|---|---|---|
| `U_j = ‖g_j‖` | `sqrt(d_jᵀ (WᵀW) d_j)` | per-unit-activation influence — no frequency, no magnitude |
| `U_j · sd(ψ_last,j)` | standardized influence | comparable to the parent's standardized-input coefficients |
| `C_j` (conditional) | mean squared prediction change per **active** row, normalized by target variance | "when it fires, how hard does the map lean on it" |
| `A_j` (pooled) | held-out ΔR² from zeroing the term, exact: `ΔSSE_j = Σ_x [2⟨r(x), ψ_last,j(x)·g_j⟩ + ψ_last,j(x)²‖g_j‖²]` | total contribution, frequency-weighted |

`WᵀW` is 3,584² = 103 MB, so `U_j` for the whole dictionary is one small Gram plus
131,072 quadratic forms — the 68.7 GB product `W W_dec` is never materialized.
This is why the primary read needs **no refit, no feature restriction, and no
`n_train < d` concession**: the underlying fit was only ever 3,584-dimensional.

Note the cross-term in `ΔSSE_j` is real on held-out data and **can be negative** —
ablation that improves held-out R². Report those as their own cell; never clip at
zero.

Run the same ladder on **both** banked map targets: `W` (dense context → SAE answer
features, pooled R² 0.722) and `M` (dense → dense, `v̂_A = M v_C`, 0.724 ridge).

### DV2 — carried but not used

Univariate predictiveness of `ψ_last,j` alone for the answer target, per feature.
Paired against the primary read: rank correlation, plus the high-carried /
low-read residual tail enumerated.

### Retroactive check on the parent's answer-side gradient

The same pooled-vs-conditional split applied to #1482's answer-side per-feature R²,
plus the control the parent never ran: **matched-n down-sampling** — subsample a
high-activity feature's active rows to a rare feature's count, rescore, and measure
how much of the 0.091 → 0.266 decile gradient reproduces mechanically. The parent's
K=20 label-shuffle null established that features clear chance (98.2%) but its band
is itself deeper for rare deciles (median ≈ −0.07, decile q2.5 to −0.139), so the
*slope* of R² against activity has never been separated from the slope of the noise
floor. Same computation as the primary read, mirrored; belongs here, not in a
separate task.

### Headline regression

Regress each DV on the mechanical covariate panel, reported as **activity-matched
partials**, never raw correlations:

- `eval_results/issue_1482/predictor_battery/fullwidth_covariates.npz` — 24 columns
  over all 131,072 features, built as a durable join substrate (activity,
  firing_freq_per_token, consistency, mean_act_cond/uncond, act_var_across_answers,
  side_ratio, scaffold_frac, dec_norm, enc_dec_cos, enc_norm, massive_dim_mass,
  logit_footprint_concentration, redundancy_max_cos, write_norm, footprint
  var/skew/kurt, promoting_class, proj_var, mean_run_length, template_token_frac,
  n_active_holdout).
- Plus last-token-specific covariates computed in Phase 0: last-token firing count,
  mean activation when active at the last token, and the span/last
  position-specificity ratio.
- Plus the feature's own answer-side per-feature R² (#1482), for the ids present on
  both sides.

Nulls: label-shuffle for the per-feature read, and a **selection-symmetric** band
for any top-k headline (the selection rides inside each draw).

### Confirmatory arm — the literal learned coefficient

Refit the feature-input map `f̂ = Bᵀψ_last + b` and read `‖B[j]‖`. Two changes from
the parent's `sae_ctx` arm, both load-bearing:

1. **`psi_last` block only.** The headline object is the context vector as defined
   in `docs/glossary_context_answer_map.md` — the last prompt token. Dropping the
   `psi_mean` block halves columns per feature. `psi_mean` enters only as a
   covariate to be partialled out.
2. **Restriction selected by last-token activity, not span activity.** The parent
   accumulated `in_counts` from `psi_idx` (pooled) only and applied the resulting
   top-8,192 to *both* blocks, leaving **5,772 of 8,192 `psi_last` columns
   identically zero** — 35% of the map's 16,384 input columns carried no
   information — while newline-firing features below the top-8,192-by-span rank got
   no column at all.

Fit at the widest well-posed width Phase 0 licenses. **Do not form a dense Gram**:
the codes are sparse (k=64, so `X` is 120,000 × 131,072 with ~7.68M nonzeros, 62 MB
CSR), so the Gram is sparse (≤5.9 GB, 2.9% density) and LSQR/CG on the sparse `X`
directly avoids forming one at all. The dual is the wrong route here — `XXᵀ` is
dense 115 GB. λ on the held-out validation carve; **no GCV** (banned at
`n_train < d`, #1887); report the selector and selected λ with every read.

Primary check: rank correlation between `‖B[j]‖` and `U_j`. Divergence means the
cheap full-dictionary read is not faithful to the map's own coefficients and the
headline scope narrows to the restriction.

### Exploratory only

Describe the top and bottom read tails using #1773 descriptions and the 1,653
context-side descriptions in `eval_results/issue_1482/context_side_labels/`. #1773's
judged axes are **search-index-only** (neighbour discrimination 0.322 vs a 0.50 bar)
and the label freeze holds — this read is labeled exploratory and carries no claim.
The two description sets are NOT same-instrument (context-side vs answer-side
evidence windows, #1773 meta caveat); label them distinctly wherever surfaced.

## Setup (pre-filled from parent #1482 — reuse, do not re-derive)

- Model: Qwen-2.5-7B-Instruct, layer 19. **No model training, no generation.**
- SAE: `andyrdt/saes-qwen2.5-7b-instruct` rev `c37e53c4bb`, `resid_post_layer_19`,
  BatchTopK, 131,072 features, k=64. Loader: `scripts/issue1482_sae.py` (token-pool
  semantics: `BOS_OFFSET=8`, norm-outlier exclusion — see its docstring and
  `.claude/rules/gotchas.md`).
- Corpus: the #1482 single-turn panel (LMSYS + WildChat pass-B), 120,000 fit /
  2,000 val / 20,000 holdout, sha-pinned split.
- Store: the 1,920-shard pooled SAE store. **Full-dictionary sparse codes** for all
  three blocks (`psi_idx` pooled context, `psil_idx` last-token context, `ans_idx`
  answer) — the parent's restriction is a fit-time projection (`col_in`), never a
  capture-time filter, so every re-projection here is free of re-capture.
- Banked maps: `sae_dense_bridge` (`W`), the #779/#1482 dense→dense refits (`M`),
  and `map_coefficients/` (the parent `sae_ctx` `B`, standardized-input units,
  λ=1e4, split-half replication 0.506, sign agreement 0.974, per-column
  label-shuffle null).
- Covariates: `predictor_battery/fullwidth_covariates.npz` (131,072 × 24).
- Feature table: `eval_results/issue_1773/feature_table_v1.jsonl`.

## Phases

| # | phase | venue |
|---|---|---|
| 0 | last-token feature census from the store: distinct features firing at `v_C`, count distribution, coverage; sets whether the confirmatory refit is over-determined | CPU pod |
| 1 | primary full-dictionary read (`U_j`, `C_j`, `A_j`) on both `W` and `M` | CPU pod |
| 2 | DV2 carried-vs-read, paired contrast, residual tail | CPU pod |
| 3 | retroactive answer-side pooled-vs-conditional + matched-n down-sampling null | CPU pod |
| 4 | predictor regression, activity-matched partials, selection-symmetric nulls | CPU pod |
| 5 | confirmatory `B[j]` refit (sparse LSQR/CG) + rank correlation | CPU pod; GPU cell only if the sparse solve is the wall |
| 6 | exploratory labeled tails; figures | CPU pod |

## Compute

**Estimated GPU-hours (total): 4**

A single number, deliberately, not a range and not a conditional — see the
no-park directive below. The work is CPU-dominated: RunPod `cpu-bigmem`, with the
9.2 GB store staged pod-side, never on the shared VM (`/` at 98% used). The 4 GPU-h
is a pre-authorized allowance for Phase 5 (one GPU cell for the sparse LSQR/CG
solve at full width) plus slack; the expected realized spend is near zero and
unspent allowance is not a licence to find uses for it. If Phase 5's measured
1-cell pilot shows the sparse solve runs comfortably on CPU, run it on CPU and
record the realized 0 GPU-h — do NOT re-plan and do NOT re-gate on the difference.

**NO-PARK DIRECTIVE (user, at spawn): this task must not stop at `plan_pending`.**
The Step 2c auto-approve gate fails safe on a missing, ranged, or otherwise
unparseable estimate, so every plan revision MUST carry the literal line
`Estimated GPU-hours (total): <single number>` in its compute section, and every
`epm:plan` marker MUST carry the matching `gpu_hours_total=<single number>` token
in its note. The number is far under `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` (100), so
the only way this task parks is a malformed estimate. If a critic round changes
the compute figure, restate it in the same single-number form; never emit a range,
a conditional, an "up to", or a blank.

**Binding user directive: run as much in parallel as possible.** Phases 1–4 are
independent given Phase 0 and shard across features and across the two map targets;
no per-feature Python loops anywhere — every 131,072-feature reduction is a batched
array op. Phase 5's λ grid and output blocks batch. See
`.claude/rules/vectorize-many-cell-fits.md`.

## Success / kill criteria

- **Success:** full-dictionary `U_j` computed with stated coverage; at least one
  covariate's activity-matched partial lands outside the selection-symmetric null;
  the pooled-vs-conditional split reported on both sides of the map; the
  confirmatory `‖B[j]‖` rank correlation reported with its restriction stated.
- **Kill:** all partials inside the null band → the read is frequency and noise
  only. Report it; do not iterate.

## Stated deviations and scope caveats

1. **Prefix arm degenerate — explicit deviation from the standing both-arms rule.**
   The #1482 panel is single-turn, so the prefix is one constant chat-template
   string (verified live in the parent: prefix-end states constant across contexts).
   Only the context arm is answerable here. The multi-turn prefix arm exists at
   [#1738](https://eps.superkaiba.com/tasks/1738) / [#1946](https://eps.superkaiba.com/tasks/1946)
   (prefix / bare / full-context, already re-scored in SAE feature space at layer
   19, same dictionary) and is the named follow-up route, not a silent omission.
2. **The SAE re-coding of the context is lossy for this purpose.** dense→SAE reaches
   0.722 while `sae_ctx`→SAE reaches 0.690, so encoding `v_C` into features discards
   answer-relevant information the raw dense state retains. Per-feature read
   strengths describe what the map does *in the SAE's coordinates*, not necessarily
   what the model computes. This is a scope statement, not a late caveat.
3. **`U_j` assumes the SAE reconstruction of `v_C`** and so inherits reconstruction
   error; the unexplained residual's share is reported alongside. `‖B[j]‖` is the
   map's own coefficient and does not, which is why both run.
4. **Linear by default.** No MLP, no nonlinear readout anywhere — not in the maps,
   not in the predictor regression. (`CLAUDE.md` standing rule, user 2026-08-06.)
5. **Association, not causation.** A ridge coefficient or an ablation on a *fitted
   map* is a predictive-association read about the map, never a claim about the
   model's computation. A causal claim would need intervention on the model.
6. **Judged-label freeze holds** (#1773). Mechanical covariates carry the headline;
   judged axes and descriptions are a search index only.

## Provenance

Originating chat, 2026-08-06. Verbatim asks, in order: "We have characterized which
SAE features our mapping is able to predict … we want to do a similar analysis to
the answer vector but seeing which predictors most predict whether a SAE feature is
read at the context vector … I am not sure how to formalize that the mapping reads
from this SAE feature — potentially we can invert the mapping? or do some zero-ing
out and measure effect on output? or just look at the mapping matrix and do some
linear algebra … can also use the SAE → SAE mapping if that's easier"; then "why
aren't we using all the features???"; then "run this in the background … as much in
parallel as possible".

Design decisions fixed in that chat, before planning:

- Construct: **both** used-by-map and present-at-context, as a paired contrast.
- Read metric: exact ablation ΔR² primary, coefficient norm and LOCO refit as
  controls.
- Map: SAE(context)→SAE(answer) elected as the literal object; superseded as
  *primary* by the full-dictionary `U_j` read once it became clear the dense-input
  map gives complete coverage with no fit — `B[j]` retained as the confirmatory arm.
- Input block: `psi_last` only (the context vector as glossary-defined).
- Corpus: single-turn #1482 panel, prefix declared degenerate.
- Deliverable: predictor regression as headline, labeled tails exploratory, the
  read-vs-predicted paired contrast as a second result.

Inversion of the map was considered and **rejected**: `B` is ill-conditioned and
explains ~69% of its target, so a pseudo-inverse read is dominated by noise
directions and answers a different question ("which context state could have
produced this answer") than the one asked.
