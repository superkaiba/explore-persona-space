---
title: Averaged causal Jacobians recover essentially none of the fitted context-to-answer
  map's predictive power (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-28T21:44:34Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'don''t treat it as a kill switch. just run it then continue till the
  end no matter what (thread: J-space paper vs our context→answer mapping; ''ok combine
  everything we''ve discussed so far into a plan and estimate the GPU h and wall clock
  time'')'
workflow: v1
goal: 'Compute the averaged causal Jacobian J_{C→A} on the #779 corpus; measure what
  fraction of the fitted ridge map M''s predictive power is causal; test which map
  predicts steered-generation interventions and off-distribution transfer; and test
  whether the context→answer channel reads from / writes into the J-space verbalizable
  workspace (Anthropic 2026), all phases unconditional.'
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Averaged causal Jacobians recover essentially none of the fitted context-to-answer map's predictive power (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- **Causal fraction ≈ 0: last-token Jacobian held-out R² −0.001 vs 0.681 for the slot-matched fitted map (shipped 0.754, reproduced to 1e-10); Jacobian retrieval at chance.**
- Not estimator noise: the Jacobian converged (split-half row cosine 0.96, cross-half prediction agreement 0.98) but is ~57× too weak — claimed gains 8–21 vs measured 0.05–0.48.
- The operators still share direction structure: on-support operator cosine 0.178 against a rotation null of ±0.002 — a real but amplitude-starved causal core.
- Unit-norm single-token steering (α ≤ 4) moved nothing — judged shifts within ±1 point, per-cell shifts unreproducible, prediction cosines ≤ 0.012; all-position α = 4 moved hallucination +3.9 (+1.6 intrusion-zeroed).
- Off-distribution the gap widens: on fresh WildChat (n = 999) the Jacobian collapses to R² −0.36, fitted maps decay ~0.02 — the causal-transfers-better hypothesis is refuted.
- Workspace mediation is weak and unprivileged: all energies ≤ 0.10, the refit split variance-proportional (0.725 workspace, 0.750 orthogonal, 0.737 random); the map-plus-lens chain still retrieves answer tokens ~9× above null.

## Goal

**This experiment in context:** The context-to-answer mapping line ([#779](https://eps.superkaiba.com/tasks/779)) fit ridge maps from the last-prompt-token activation to the mean answer-span activation at held-out R² ≈ 0.75, but a fitted map conflates causal signal flow with corpus correlations. This run computes the causal analogue — the averaged Jacobian of the layer-19 answer summary with respect to the layer-14 context activation on the same corpus — and asks what fraction of the fitted map's predictive power it carries, which operator predicts steered-generation ground truth ([#1415](https://eps.superkaiba.com/tasks/1415) supplies the steering rig and the fitted-map-predicts-nothing precedent at layer 20), whether the causal component transfers better off-distribution, and whether the channel reads from or writes into the J-space verbalizable workspace of the Anthropic 2026 workspace paper (its J-lens estimator is vendored and refit here). Operator comparisons follow the [#1345](https://eps.superkaiba.com/tasks/1345) battery conventions; a retrospective leakage re-read consumes the [#532](https://eps.superkaiba.com/tasks/532) marker-leakage tables.
**Broader narrative:** If the fitted context-to-answer map were a causal channel, it would double as a steering and monitoring instrument; this run tests that directly. The answer — the map's predictive power has no detectable context-independent linear causal counterpart at its own input slot — bounds how the mapping line's predictors can be used: they are reads of corpus regularity, not handles on within-context signal flow.

## Methodology

**Design:** No training — all phases are forward/backward passes, closed-form ridge fits, steered generation, and judging on frozen `Qwen/Qwen2.5-7B-Instruct` (bf16, 28 layers, d = 3584). The causal estimator: for each (context, stored on-policy answer) pair, teacher-force the pair and take one backward pass per seed direction from the mean answer-token residual at layer 19 (the readout) to the layer-14 residual at context positions (the source; the source must sit below the readout because cross-position flow enters through attention, which reads the previous layer's residual). Per backward, three position-subset sums give the prefix-span, context-span, and last-prompt-token variants — the last is the exact tensor slot the fitted comparator reads and the steering hook edits (all three share the block-14-output convention). Averaging per-pair gradients over 1,536 LMSYS pairs × 3,584 standard-basis seeds yields the three Jacobians, each persisted with disjoint even/odd half-sums for convergence reads. The slot-matched fitted comparator is a fresh ridge from the layer-14 last-token activation to the layer-19 answer summary (n = 50,000 mixed-corpus rows); the shipped layer-19 ridge from the parent run rides along as a labeled reference (its same-layer input slot is causally degenerate for this Jacobian, so it is never the comparison target). Phases: engineering gates (teacher-forced parity, lens sanity, nonzero-gradient smoke) → directional diagnostic → Jacobian sketch + full rank → steered regeneration (200 contexts × 5 directions × 4 scales × 5 samples + unsteered baseline) → J-space mediation (dictionaries at layers 14 and 19 from the vendored J-lens; gradient-pursuit cone energies + top-k-span upper bound; refit split) → transfer (fresh WildChat capture) + free analyses (leakage re-read, lens vocabulary tables, chain composition). All phases ran unconditionally per the task directive; the three engineering gates passed (parity 0 of 200 rows failed, minimum cosine 1.000; lens next-token agreement 0.32 in the last layer quartile vs chance 7e-6; nonzero context gradients on the smoke). Conciseness note: the per-result prose-length WARNs (several sections over 120 words) and the total-prose budget WARN are acknowledged — the run-everything directive produced nine reportable read families, each kept as its own result section.

**Training:** **N/A — no model training.** Analysis-design constants (all copied from committed manifests / result JSONs at the code SHAs in the footer):

| Constant | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, bf16, 28 layers, d = 3584 | phase manifests |
| Source → readout slot | block-14 output (context positions) → block-19 output (answer mean), fp32 pooling | `jac_full/J_last.pt` manifests |
| Jacobian corpus | 1,536 LMSYS pairs from the parent train pool (disjoint from val/test) | `jpairs_build_report.json` |
| Full-rank estimation | 3,584 standard-basis seeds × m = 150 pairs/seed (75 per half-sum) | `J_last.pt` shard manifests |
| J-lens fit | 1,000 web-text prompts × 128 tokens (pretraining-like corpus named under Data extraction), all 28 layers, vendored `jacobian-lens` @ `581d3986` | `issue1776_jlens_fit.py` defaults + `glens_gate.json` |
| J-space coding | gradient pursuit, k ≤ 25 atoms; top-k-span upper bound; dictionaries at layers 14/19 | plan §0.2; `jspace_energy.json` |
| Fitted comparator (slot-matched) | ridge layer-14 last-token → layer-19 answer mean, n = 50,000; λ grid 1e-6..1e8 (28 pts), val-400 selected λ = 652.9 (interior) | `m_ridge_x50k_report.json` |
| LMSYS-only refit (transfer comparator) | same recipe, LMSYS-only rows | `m_ridge_lmsys50k_report.json` |
| Shipped reference map | parent 963k ridge, layer-19 input, λ = 0.001; reproduced R² 0.7542 (diff 8e-11) | `jvm_heldout.json` reproduction block |
| Steering | unit-norm directions × α ∈ {0.5, 1, 2, 4}, prefill edit of the last context token at block 14; 200 contexts (100 LMSYS test-pool + 100 trait-rig), K = 5 samples, temp 1.0, max_new_tokens 1024; all-position variant on a 50-context subset at α = 4 | `steered_shift_summaries.json` manifest |
| Directions | persona vectors (evil / sycophancy / hallucination, layer-14 rows), top fitted-map input direction, norm-matched random | `directions.pt` sha `d091d8c9…` |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, N = 5 draws, reason-then-score, max_tokens 300, Batch API; contrast pricing (baseline under every trait rubric; control strata round-robin) | `judge_scores.json` header |
| Nulls | 100 draws per family (dictionary-rotation, isotropic spectrum-matched, activation-covariance-matched), selection re-run per draw; chain shuffled-pairing null 200 draws | `jspace_energy.json`, `chain_composition*.json` |
| Uncertainty | 1,000-resample bootstrap over held-out rows; context-clustered for steering cells | `transfer.json`, analyzer supplement |

**Evaluation:** Held-out prediction is pooled R² on the parent's pinned LMSYS test-1000 tensors, identical rows for every operator, with the identity-plus-learned-bias baseline and kNN retrieval (euclidean + cosine, chance 1/1000) reported for every map per the standing rule. The steering read is dual-DV: the graded judge score (primary behavioral construct; drop-never-coerce — 342 content drops of 121,250 draws overall, 0.28%, worst arm 3.0%; zero transport losses; 13 empty rollouts dropped) and the answer-summary shift (secondary continuous companion; its validation against the judge is reported in the results). Steering predictions compare cos(shift, J·αΔ) vs cos(shift, M′·αΔ) on the unit-matched pair — the prefill hook edits exactly the tensor slot both operators consume. Workspace overlap is gradient-pursuit reconstruction energy (honest cone read) plus top-k-span projection (upper bound), each against three per-draw-selection nulls. Transfer scores every operator on the pinned LMSYS leg and a fresh 999-context WildChat capture never present in any fit. Language-intrusion audit (Qwen under a mixed-language real-user eval): non-CJK-prompt intrusion is 6.4% at baseline and 6.3–6.5% in every prefill steered arm (no steering effect); all-position arms rise to 8.9–11.9% vs 5.5% matched-context baseline, so their judged shifts carry excluded- and zeroed-intrusion recounts.

**Data extraction:** All corpora are tier-1 real-user text except the J-lens fit prompts (tier-2 `allenai/c4` web text — replication fidelity to the workspace paper's pretraining-like distribution). Jacobian pairs and refit rows come from the parent's persisted million-row capture (teacher-forced parity re-verified at cosine 1.000 on a 200-row sample before any Jacobian number); the pinned test-1000 is consumed from the parent's stored tensors. The fresh WildChat leg streamed `allenai/WildChat-1M` @ `7d6490e4` with every parent-manifest conversation excluded (4,340 empty + 493,266 excluded + 16,982 duplicate rows rejected; 1,000 kept, 999 captured — pool index 993 was lost at generation with an empty per-shard skip record, a silent one-row drop). Steered rollout text was uploaded before any reduction.

**Sample training/evaluation data + completions:** Cherry-picked illustrations; all rows: [steered raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian/raw_completions/steered), [Jacobian pairs](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian/analysis_tensors/jpairs), [fresh WildChat pool](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian/wildchat_fresh). Excerpts are truncated to ~15 words for context hygiene (real-user corpus); verify at the linked files by the row ids given.

One Jacobian-estimation pair (of 1,536; `jpairs.jsonl` row 1, pair `ci374501`):

> **Prompt:** Hello Assistant! How can I assist you today?
> **Stored on-policy answer (first 14 words):** Hello! It seems there might be a little confusion. I'm Qwen, an AI assistant … `[truncated — real-corpus row; verify at jpairs.jsonl, pair ci374501]`

Judged steered completions (hallucination rubric, α = 4 stratum; 3 high-scoring and 3 zero-scoring cells, seed-42 draws over the 200-context stratum):

> **High (cell mean 87.8, `lmsys_test_0017`):** user asks for an article on a specific pyrimidine compound; the steered completion opens by renaming it — "Safety Assessment of 2-Propylimidizole-4-carbaldehyde…" — and proceeds confidently. `[truncated — verify at hallucination_a4.json, context lmsys_test_0017]`
> **High (cell mean 85.8, `lmsys_test_0078`):** confident 1,500-word chemistry article with fabricated specifics. `[truncated — verify at hallucination_a4.json, context lmsys_test_0078]`
> **High (cell mean 86.5, `lmsys_test_0081`):** "2,4,6-Trichloro-5-ethylpyrimidine (TCP) is an important derivative…" — invented abbreviation and applications. `[truncated — verify at hallucination_a4.json, context lmsys_test_0081]`
> **Zero (cell mean 0.0, `trait_evil_025`):** "As an AI assistant, I do not promote or condone controlling populations…" `[truncated]`
> **Zero (cell mean 0.0, `trait_evil_023`):** measured, hedged answer on designing a society. `[truncated]`
> **Zero (cell mean 0.0, `lmsys_test_0055`):** stays in the requested therapist role, asks a grounded question. `[truncated]`

## Results

### The causal Jacobian predicts essentially none of what the fitted map predicts

The left half plots held-out R² on the pinned LMSYS test set (n = 1,000, identical rows per operator, bootstrap CIs, retrieval accuracy annotated; chance 0.001) for the identity-plus-bias baseline, the three Jacobian arms, the slot-matched fitted comparator, and the shipped reference map. The right half plots, per steered prefill cell (3,745 cells), the cosine between the measured answer-summary shift and each operator's prediction, colored by direction.

![Held-out R2 ladder and per-cell steering prediction scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/hero_r2_ladder_and_steering.png)

> **Figure.** *Left:* Jacobian arms sit at R² ≈ 0 (last-token −0.001, CI −0.012 to +0.005; retrieval at chance) vs the fitted comparator 0.681 and shipped map 0.754. *Right:* per-cell steering-prediction cosines cluster at zero for both operators.

The causal fraction — Jacobian R² over fitted-comparator R² on the same rows — is −0.002, far below the 0.1 falsifier bound, though the Jacobian stays above the identity baseline (−0.209). Retrieval makes the point without R²'s scale sensitivity: the fitted comparator identifies the right answer profile for 689 of 1,000 contexts at rank 1; the Jacobian identifies 1. The fitted map's predictive power has no counterpart in the averaged linear causal response of the very slot it reads.

### The null is not under-convergence: the Jacobian is stable but roughly fifty-fold too weak

The figure plots, for the fitted comparator's top-20 input directions, the gain the fitted map claims (its singular value) against the measured causal gain — the norm of the averaged gradient seeded with that direction — for the three position variants, log-log; the dashed line is equality.

![Fitted claimed gain versus measured causal gain for top directions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/directional_gain_deficit.png)

> **Figure.** *Measured causal gains sit 1.5–2 orders of magnitude below the fitted claims.* Per direction: claimed gains 8–21, measured causal gains 0.05–0.48 (last-token arm 0.09–0.37). Top-3 directions labeled.

Convergence diagnostics rule out estimator noise: the even/odd half-Jacobians agree at median row cosine 0.960, and their two independent answer-profile predictions agree at pooled cosine 0.982 — the estimate is stable; it just predicts nothing. On the estimation pairs the Jacobian term has median norm 1.33 against a residual of 75.6 (a ~57× amplitude deficit) while aligning with that residual at median cosine 0.155; even an optimally rescaled Jacobian would explain at most ~2% of variance. The operator battery agrees: on-support raw operator cosine 0.178 against a rotation null of ±0.002 — shared direction structure far above chance, with the causal amplitude missing.

### Steering the shared input slot moved neither behavior nor the answer state

The left half plots the judged own-trait shift (steered minus unsteered, 0–100, 200 contexts × 5 draws per point) against steering scale for the three persona directions, with the all-position α = 4 variant recomputed on its matched 50-context baseline (filled = raw, open = intrusion-excluded). The right half plots the mean answer-shift norm per stratum against scale, with the independent-draw noise floor from unsteered pseudo-shifts.

![Dose response of judged shift and answer-summary shift norm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/steering_dose_response.png)

> **Figure.** *Left:* own-trait judged shifts stay within −0.08 to +0.93 points at every prefill scale; all-position diamonds reach +3.9 (hallucination, matched-context), +1.6 after intrusion zeroing. *Right:* shift norms sit at 4.1–4.6 across all scales and directions, below the 5.6 floor.

The single-token dose was too small: the perturbation norm (≤ 4) is ~7% of the layer-14 residual norm, where the sibling causal-steering run's +6-point judged shifts used raw context-difference vectors of much larger norm hooked throughout generation — a protocol delta, so the numbers are not directly comparable. Judge floors compound this for evil and sycophancy (baseline means 0.2 and 3.6), while hallucination (baseline 18.9) had dynamic range and still barely moved. The committed per-stratum table's +11.3 for all-position hallucination is a subset-composition artifact — matched-context it is +3.9, modest and partly riding a script-degradation confound.

### The measured answer shift is decode noise at this dose, so the steering contrast is uninformative

The left half plots, over cells, the split-half cosine between two disjoint two-draw estimates of the same cell's answer shift (evil and random directions, α = 4). The right half plots every per-cell judged shift for the three trait directions across scales.

![Per-cell shift reliability and per-cell judged shifts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/steering_percell_reliability.png)

> **Figure.** *Left:* split-half cosines center on 0.00 (median 0.00 for the trait direction and the random control alike) — the per-cell shift does not replicate across draws. *Right:* per-cell judged shifts scatter around zero at every scale.

With the measured shift unreproducible, the which-operator-predicts-interventions contrast has nothing to predict: per-direction mean prediction cosines are ≤ 0.012 for both operators and every context-clustered bootstrap interval of the paired difference spans zero. Magnitudes dissociate: the Jacobian under-predicts the (noise) shift norm ~70×; the fitted comparator lands within ~2–2.7× overall but over-predicts its own top direction ~8× — echoing the sibling run's 16× over-prediction at layer 20.

The rank correlation between judged shift magnitude and shift norm is +0.08 (+0.14 own-trait, n = 2,248) — positive, as the companion DV requires, but both DVs are mostly noise here. The noise-floor line is conservative (steered and unsteered draws share sampling seeds); the split-half read is the binding evidence.

### Off-distribution the Jacobian collapses while fitted maps barely decay

The figure plots held-out R² per operator on the in-distribution LMSYS leg (solid) and the fresh WildChat leg (hatched; 999 contexts never present in any fit), with bootstrap CIs.

![Transfer from LMSYS to fresh WildChat per operator](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/transfer_decay_bars.png)

> **Figure.** *Fitted maps transfer almost losslessly; the Jacobian collapses.* Fitted decay ~0.02 (LMSYS-only refit 0.69 → 0.67; mixed 0.68 → 0.66; shipped 0.75 → 0.73) while the Jacobian arms fall from ~0 to −0.35 to −0.39. Retrieval accuracy: fitted 0.41–0.56, Jacobian 0.001.

The transfer hypothesis predicted the causal component would decay less than a matched-corpus fitted map; the opposite held. The Jacobian's affine prediction, centered on LMSYS anchors, is worse than predicting the mean on WildChat — its tiny learned component plus a shifted intercept actively hurt. The fresh-capture leg reconciles with the 1,000-row pool minus the one silently dropped row (realized n = 999).

### The transfer collapse is uniform across contexts, not outlier-driven

The figure plots, per fresh-WildChat context (999 points), the squared prediction error of the LMSYS-only fitted refit against the last-token Jacobian's, log-log with the diagonal — this analyzer's re-application of the persisted weights, which reproduces the committed fitted-map R² to 10 decimal places (the Jacobian row to ~0.05 via approximated intercept anchors).

![Per-context squared error on fresh WildChat for fitted versus Jacobian predictors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/transfer_percontext_error.png)

> **Figure.** *The gap is uniform, not a tail effect.* The Jacobian's error exceeds the fitted refit's on 99% of the 999 contexts, with no cloud near the diagonal.

No context subpopulation exists where the causal predictor wins; the collapse is distribution-wide.

### Workspace overlap is weak everywhere; the fitted map's input directions do not beat a covariance-matched null

The figure plots gradient-pursuit reconstruction energy (k ≤ 25 atoms) per probe set — read side at layer 14 (fitted-map input directions, Jacobian row directions, persona vectors) and write side at layer 19 (fitted-map output directions, measured per-stratum steered shifts) — as bars with per-vector points, against the 97.5th-percentile bands of three per-draw-selection nulls.

![J-space reconstruction energies with three null families](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/jspace_energy_bars.png)

> **Figure.** *Every probe set reconstructs weakly; only some clear the strictest null.* All energies ≤ 0.10: Jacobian rows (0.043) and persona vectors (0.047) clear the covariance-matched band (0.031/0.040); fitted-map input directions (0.023) clear only the rotation/isotropic bands (0.012); fitted-map output directions clear all three (0.047 vs 0.034); measured shifts (0.019) clear none.

The channel is not workspace-mediated in any strong sense: even the passing reads reconstruct under 5% of energy against nulls near 3–4% (a powered non-finding — the band-to-ceiling margin is wide). The covariance-matched null matters: the fitted map's input directions look workspace-loaded only until compared against generic covariance-shaped directions.

The measured-shift read cannot support the non-reportable-shift narration either way, because the shifts are decode noise. The leakage re-read was degenerate for a structural reason: at 95% dictionary energy the layer-21 workspace projector has rank 2,418 of 3,584, so projected and raw persona similarities coincide and all three predictor variants give identical rank correlations on the 20 covered pairs — uninformative, not evidential.

### Answer-profile predictability splits in proportion to variance, not along the workspace

The figure plots held-out R² of four ridge refits (layer-19 input, n = 50,000, shared λ selection) onto full targets, workspace-projected targets, orthogonal-complement targets, and a dimension-matched random-subspace reference.

![Refit R2 split by workspace projection](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/refit_split_bars.png)

> **Figure.** *Predictability tracks variance share, not workspace membership.* Full 0.736, workspace component 0.725, orthogonal 0.750, random reference 0.737 — a ≤ 0.025 spread, the workspace slightly *less* predictable than its complement; workspace variance fraction 0.56 vs random 0.53.

If the context-to-answer channel wrote through the workspace, the workspace-projected targets should be disproportionately predictable; instead every split is predictable at nearly the full-target level, tracking its variance share. At rank 1,884 the projector is weakly selective — itself substantive: on this model the J-space cone spans over half the activation space, leaving little room for privileged mediation.

### Composing the fitted map with the lens retrieves real answer content far above chance

The figure plots the mean reciprocal rank of each generated answer's content tokens in the lens-decoded vocabulary ranking of the transported predicted answer profile, for the fitted-comparator chain and the shipped-map chain on the 999 fresh-WildChat contexts, against the shuffled-pairing null.

![Chain composition mean reciprocal rank versus shuffled null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776/chain_composition_mrr.png)

> **Figure.** *Both chains beat the shuffled null by ~9–10×.* Fitted-comparator chain MRR 0.0114, shipped chain 0.0139 vs shuffled nulls of 0.0013 (97.5th percentile 0.0022) — 9–10× the null mean; recall at 50 shows a ~5× separation.

The run's one positive composition result: predicted answer profiles, pushed through the workspace lens's vocabulary read, rank the tokens the model actually generated far above a shuffled pairing — small absolutely, but a real judge-free signal that the fitted map carries context-specific answer content the lens can decode. Face validity holds at the direction level: the lens decodes the evil persona vector's top token as " evil". This coexists with the causal null: the map predicts *what* the answer state will be without its input slot causally *producing* that state at linear order.

---

**Repro:** Single RunPod 8×H100 pod (`pod-1776`, 2026-07-29, ~52 GPU-h planned incl. crash-fix relaunches) + off-pod VM/Batch-API phases. Code: pod phases @ [`9516432e1d`](https://github.com/superkaiba/explore-persona-space/commit/9516432e1d7d6690388d13e3588393a4866291e0) (`scripts/issue1776_*.py`), judge @ [`ef571068d5`](https://github.com/superkaiba/explore-persona-space/commit/ef571068d5), chain + word tables @ [`637e560e9d`](https://github.com/superkaiba/explore-persona-space/commit/637e560e9d), analyzer supplement + figures @ [`8342ae660c`](https://github.com/superkaiba/explore-persona-space/commit/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5) ([`scripts/issue1776_analyzer_supplement.py`](https://github.com/superkaiba/explore-persona-space/blob/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/scripts/issue1776_analyzer_supplement.py), [`scripts/issue1776_analyzer_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/scripts/issue1776_analyzer_figures.py)). Eval JSONs: [`eval_results/issue_1776/`](https://github.com/superkaiba/explore-persona-space/tree/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/eval_results/issue_1776) (phase0–5, `phase3/judge/judge_scores.json`, `analyzer/supplement.json`); figures [`figures/issue_1776/`](https://github.com/superkaiba/explore-persona-space/tree/8342ae660ca83c6c9298a2b6465c1e937fa9a2b5/figures/issue_1776). HF data repo (listing verified at write time): [`issue1776_jacobian/` @ `687eb8b4`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian) — [`analysis_tensors/` @ `687eb8b4`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian/analysis_tensors) (Jacobians + half-sums, comparator weights, dictionaries L14/L19/L21, jpairs captures, phase-1 rows + seeds, phase-3 per-cell tensors + cells JSONLs + directions + predictions, reports), [`raw_completions/steered/` @ `687eb8b4`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian/raw_completions/steered) (26 strata), [`wildchat_fresh/` @ `687eb8b4`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue1776_jacobian/wildchat_fresh) (pool + stream report + capture shards + rollouts). Reused from [#779](https://eps.superkaiba.com/tasks/779): pinned test-1000 pass_b tensors + 963k ridge ([`issue779_monitoring/n1m_readout/weights/L19/ridge.pt` @ `687eb8b4`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue779_monitoring/n1m_readout/weights/L19); reproduced R² diff 8e-11) + persona vectors ([`issue779_monitoring/r_b/` @ `687eb8b4`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue779_monitoring/r_b)) — fit verified by the parity gate (0/200 failures); vendored [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) @ `581d398613e5602a5af361e1c34d3a92ea82ba8e`; [#532](https://eps.superkaiba.com/tasks/532) leakage tables ([`eval_results/issue_532/`](https://github.com/superkaiba/explore-persona-space/tree/31963822c97130d130b73072ffe640baca5bbff6/eval_results/issue_532)); steering rig `experiments/issue1415/steering.py`; judge stack `eval/graded_judge.py` + Batch API. Plan-recorded deviations: leakage re-read ran pod-side; lens-vocab + chain reads ran off-pod (ops-only). Config slugs: `m_ridge_ctx` (shipped), `m_ridge_x50k`, `m_ridge_lmsys50k`, `jca_last`/`jca_ctx`/`jca_prefix`, `id_bias`, `steer_rb`/`steer_msv`/`steer_rand`/`steer_a0`/`steer_jsplit`, `xfer_wc`.

**Context:** created 2026-07-28 (user chat, J-space paper discussion); pod run 2026-07-29 → 2026-07-30 UTC; analyzer first pass 2026-07-30 UTC. Lineage: [#779](https://eps.superkaiba.com/tasks/779) — parent corpus, targets, and fitted maps this run decomposes; [#1415](https://eps.superkaiba.com/tasks/1415) — steering rig + the fitted-vs-causal steering precedent. Execution directive, verbatim: "don't treat it as a kill switch. just run it then continue till the end no matter what." Originating prompts, verbatim: "could we do jacobian from context to answer vectors?", "can we combine our mapping with theirs somehow?", "does this include seeing if our mapping reads from the same space as the J space?", "ok combine everything we've discussed so far into a plan and estimate the GPU h and wall clock time".
