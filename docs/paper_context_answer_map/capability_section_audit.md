# Capability section: evidence and writing audit

Date: 2026-09-07. Requested addition: model capability versus held-out context-answer
metamodel R2, with a simple labelled scatter and a connection to neural-trajectory
straightening. No inference, training, GPU jobs, or new model fits.

## Current revision: ten-model display

On 2026-09-07 Thomas requested removing Qwen2.5-32B because it is an older model.
Only that model is excluded. The original eleven-model analysis below remains a
sensitivity check, not the statistics displayed on the revised figure. The
manuscript and caption explicitly identify the restriction as post-hoc. No
general release-date cutoff or preregistered exclusion rule is claimed.

Freshly computed results from the retained coordinates:

| Comparison | n | Spearman rho | Two-sided p | Multiplicity-adjusted p |
|---|---:|---:|---:|---:|
| Displayed subset | 10 | 0.8666666667 | 0.002173170194 | 0.08479576021 |
| Recorded measured scores in that subset | 4 | 0.8 | 0.3333333333 | Not computed |
| Original panel, retained sensitivity | 11 | 0.7152638147 | 0.0174 | 0.4101 |

The ten-model unadjusted p is exact, with 7,886 exceedances among all 3,628,800
permutations. The four-model p is exact over 24 permutations. The restricted-panel
maxT correction recomputes the same 56 predictor/outcome tests with 20,000 shared
row permutations and seed 2588. It uses the finite Monte Carlo plus-one convention
and has 1,695 exceedances. Unlike this revision, the historical audit used an
uncorrected Monte Carlo fraction. Neither p-value corrects for post-hoc model
selection or dependence among related model families.

Six retained index scores are marked estimated and four measured. Uniform filled
circles and no status legend remain as requested. Metadata preserves the removed
row, its reason, both panels, source hashes, and both measured-score sensitivities.
An independent read-only reviewer verified the exact ten-model p with integer
rank-distance dynamic programming and reproduced maxT using a separate scalar
permutation loop. No substantive statistical errors were found.

Revision validation: Ruff and whitespace checks passed, the 42-page manuscript
rebuilt, and pages 7-8 plus the colour and grayscale exports were visually checked.
The writing-tells gate has no hard-ban hits. The same two pre-existing unresolved
introduction citations remain, with no new unresolved references.

The evidence map and original-review record below describe the initial
eleven-model revision. The current manuscript uses the ten-model numbers above
and retains the original full-panel result in prose.

## Scope and paragraph outline

1. Setup: original eleven-model prompt-state panel, LMSYS-Chat-1M prompts,
   separately generated model answers, validation-selected layer and ridge penalty.
2. Evidence and limitation: positive full-panel Spearman correlation, multiplicity
   adjustment and recorded-measured-score sensitivity.
3. Interpretation: hypothesis of more linearly accessible future-answer
   representations, distinguished from a causal capability explanation and from
   token-trajectory curvature in Hosseini and Fedorenko (2023).

Canonical prose is `sections/results/08_capability.tex` in Overleaf project
`6a59c927290f8b8b5eee0055`, included by `sections/04_results.tex` in `main.tex`.
The current tracked Overleaf tree has no `draft.tex`, despite older instructions
referring to it. No retired document was recreated.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Original panel has eleven prompt-state models | `rank_relationships.json` no-thinking rows joined to the map records, eleven unique keys | Supported for this historical panel |
| Full-panel Spearman rho = 0.715, unadjusted p = 0.0174 | Audit result and fresh rho parity check from plotted coordinates | Supported descriptive association |
| Familywise adjusted p = 0.4101 | Original maxT audit over 56 tests within the no-thinking grid, 20,000 permutation draws | Exploratory, not confirmatory |
| Five recorded measured scores yield rho = -0.10, p = 0.95 | Exact enumeration of 120 permutations in the figure generator | Supported sensitivity, a different small subset, not proof that estimates cause the full-panel association |
| Better answer accessibility contributes to model capability | No intervention or causal identification | Hypothesis only |
| Better next-word prediction is associated with more neural-trajectory straightening | Hosseini and Fedorenko 2023, arXiv:2311.04930 | Related evidence on a different construct |

Pinned analysis inputs: commit `14a9a3605c6f14e0537a7e6407ae71cf5093a972`,
`eval_results/issue_2588/rank_relationships.json` and
`eval_results/issue_2588/mapping_rank_vs_capability.json`. Source hashes, every
plotted value, layer/test-row counts, and sensitivity calculations are in
`figures/paper/c1_model_capability.meta.json`. The generator reads those pinned Git
blobs rather than whichever files happen to be present in the shared checkout.

Source recipe: `scripts/issue2588_run_cell.py` at
`8f7d47ceec9a799d73abf301adc348c4285e4976`, inherited issue-2330 splits and
issue-1491 LMSYS first-user-message manifest. Up to 10,000 train, 400 validation,
and 1,000 held-out test prompts, with per-model filtering. Input is the last
prompt-token state, target is the model's own answer-token mean. The original
prompt panel is kept separate from later long-cap results and end-of-thought
reads. This is not an assertion that eleven models cover all available artifacts.

## Intelligence Index provenance

The original registry marks six values as estimated and five as measured.
Original task-2588 progress notes (2026-08-25) describe these as estimates
published by Artificial Analysis. They are not inferred from the mapping R2
results. The registry retains values and status, not a reproducible estimation
formula or all underlying benchmark inputs. The live AA pages describe estimates
as awaiting independent evaluation. The exact derivation of our six historical
estimates remains unverified.

The user explicitly approved retaining these values and requested no estimated
label in the plot. All eleven points therefore have identical filled circles.
The manuscript footnote and this sidecar retain provenance. A published estimate
must not be described as an independently evaluated score. Model-level index
settings can also differ from our thinking-disabled runs.

The website now describes a newer index version. Do not replace individual
historical scores with current values or imply the existing panel used the current
suite. Historical v4.1.1 labels on the same-width extension are retained as source
metadata, not retroactively verified for every original row.

Sources checked:

- https://artificialanalysis.ai/methodology/intelligence-benchmarking
- https://artificialanalysis.ai/models/releases/qwen3-5-27b
- https://arxiv.org/abs/2311.04930
- Existing manuscript LMSYS reference `zheng2024lmsyschat`.

## Five-dimension self-review

- Contribution: adds a bounded cross-model descriptive relationship, not a new
  experiment or an established mechanism. Pass with explicit limits.
- Clarity: one setup, one empirical claim, one interpretive paragraph. Uses the
  manuscript's metamodel terminology and caption/figure-reference conventions.
- Experimental strength: full-panel magnitude is large but multiplicity and
  subset sensitivity prevent a confirmatory conclusion. These are stated.
- Evaluation completeness: one dataset and one historical prompt panel. Companion
  retrieval/baseline/length controls exist in task 2588, but the requested figure
  specifically concerns R2. No unreported uncertainty bars are invented.
- Method soundness: layer and ridge selection are validation-only. Width, family,
  realized sample size, answer variance and generation settings can confound the
  cross-model interpretation. Causal language is withheld.

No abstract or introduction headline was changed. Independent read-only review
requested clearer attribution and a frozen score-source link. The footnote now
links the historical registry and describes its recorded status labels.

Validation: fresh coordinate/rho parity and exact subset test passed. The main
PDF builds (42 pages), and pages 7-8, the vector scatter, and its grayscale copy
were visually checked. No new unresolved citations or clipped labels. Two
pre-existing undefined introduction citations remain: `wang2026confidentlywrong`
and `mahadik2026personacoordinates`. The writing-tells gate passes after replacing
one pre-existing Unicode en dash with the equivalent TeX `--` in Methodology,
without changing the wording. The new section's rhetorical-contrast flag is
intentional: it distinguishes answer-mean prediction from tokenwise curvature.
