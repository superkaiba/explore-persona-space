---
title: A LoRA-installed conditional behavior writes one context-invariant direction
  (read/layer-specific for the marker tic); whether the four behaviors look distinct
  or coincident is itself a property of the chosen read (MODERATE confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-06-16T06:55:43Z'
has_clean_result: true
parent_id: 537
origin_prompt: 'test whether many conditional behaviors share a direction across different
  contexts; post-hoc on the #537 context-leakage comprehensive eval'
goal: 'Re-extract layer-L activation-shift directions (trained minus base) from the
  #537 context-generalization testbed''s existing adapters (5 conditional behaviors
  x 16 training contexts x seeds, on HF) plus a bounded 2nd-seed (1042) retrain of
  em + sycophancy across the 16 contexts so all readable behaviors get a within-cell
  seed ceiling, and test (Q1) whether each behavior''s shift collapses to ONE direction
  across the training contexts it was implanted under (context-invariance of the write)
  and (Q2) whether the different behaviors'' dominant directions coincide or cluster
  by family (cross-behavior identity), benchmarked against the within-cell seed ceiling,
  with unit-norm direction cosine as the dose-invariant DV.'
relates_to:
- identity-cb-duality
- identity-persona-vs-behavior
---
# A LoRA-installed conditional behavior writes one context-invariant direction (read/layer-specific for the marker tic); whether the four behaviors look distinct or coincident is itself a property of the chosen read (MODERATE confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_651.md](https://github.com/superkaiba/explore-persona-space/blob/c9c07f1e68b9495f772fb888bf346fa410f6d1a9/docs/methodology/issue_651.md) · [gist](https://gist.github.com/superkaiba/d234505b3e076d286e4dd512ec65cd92)

## Takeaways

- At the primary read, all four readable behaviors pass context-invariance. Per-context cosine clears 0.85 of the per-cell seed ceiling in 16 of 16 contexts for em, agreement, and fact, and 13 of 16 for marker; training context tunes install strength rather than direction.
- Context-invariance is robust for em, agreement, and fact (holds across all 3 depths × both reads, every one of its 6 read×layer cells) but read/layer-specific for the marker tic: it passes only end-of-response slot at layers 7 and 14 (just 2 of its 6 read×layer cells) and turns context-specific at slot layer 21 and under the mean-over-response read at every depth.
- At the primary read the directions are mostly distinct: four of six pairs land at 0.05-0.17 of ceiling (two within their pairwise null); two hit ~0.6 — harmful-advice × agreement (0.59, the predicted family pair) and taught-fact × marker-tic (0.61, fitting no family story). A shared LoRA-geometry component is not ruled out for either.
- Whether the four behaviors look distinct or coincident is itself a property of the read. The two ~0.6 coincidences swing ~3× across read×layer (harmful-advice × agreement 0.28-0.78, taught-fact × marker 0.15-0.73); under any end-of-slot read all six pairs sit at 0.49-0.89, and only the mean-over-response read separates most of them.
- The blanket-refusal negative control did NOT come back as a clean null (top-share 0.48 vs p95 0.27, mean cosine 0.84 on 4 surviving contexts, single seed, no ceiling), but those 4 contexts are all `sp_*` system-prompt variants, so context similarity is the likeliest source, not a real refusal write. Confidence stays MODERATE for this.
- Scope: refusal recovered at 4 of 16 contexts (single seed) after five HF Hub timeouts; the 4 EM-no-contrast cells were dropped; ceilings come from 2 seeds on one fixed probe panel.

## What I ran

- **Why:** #521 found EM's activation shift collapses to one direction across seeds; #552 found marker and EM point in different ways. Two open questions left: does "one direction" survive across the *training context* a behavior is installed under, and do different behaviors share an axis? Parent task #537 trained 5 behaviors × 16 contexts × seeds but never read the geometry, so this work re-extracts shifts from adapters that already exist.
- **Design:** 4 readable behaviors (harmful-advice/EM, wrong-claim agreement, marker tic, taught fact) × 16 training contexts × 2 seeds = 128 adapter cells, read on one fixed neutral probe panel (14 personas × 20 questions). Within a behavior, the only thing changing across cells is the training context. A blanket-refusal row is the in-run negative control. DV: unit-norm residual-shift direction cosine at layer 14 (dose-invariant).
- **Training:** The only new training is a 2nd-seed (1042) replicate of the parent testbed's em + sycophancy across all 16 contexts (32 adapters), on the parent's exact frozen data, recipe, and dose. The seed alone changes, so the within-cell cross-seed cosine gives each of the 4 readable behaviors a seed ceiling. Marker, fact, em(seed-42), and refusal cells are re-extracted from the parent's existing adapters with no retraining.
- **Eval:** Per cell, one teacher-forced forward per (persona, question) on the fixed panel. Shift = trained-minus-base residual at layer 14, taken mean-over-response for the generative behaviors and at the end-of-response slot for marker/fact. Per behavior I compute the SVD top-direction share and the per-context cosine to it, against a sign-flip null (1000 reps) and the per-cell seed ceiling. Across behaviors I compute the 4×4 unit-norm cosine matrix, each off-diagonal as a fraction of the per-behavior-U1 seed ceiling.

## Findings

### Each behavior passes the context-invariance bar across its 16 training contexts (Q1)

The verdict bar: per-context cosine reaches 0.85 of the behavior's median seed ceiling for at least 80% of contexts, and the top-share clears the sign-flip null.

![Left: per-behavior top-direction share with sign-flip null p95. Right: per-context cosine to each shared direction, with per-cell seed-ceiling and 0.85 bar; marker's 3 sub-bar contexts and the refusal null arm (5th row) visible.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/q1_context_invariance.png)

> **Figure.** *Every readable behavior's shift concentrates on one direction shared across its 16 install contexts; refusal (the partial null arm) is shown alongside.* Left: top-direction share vs sign-flip null p95. Right: per-context cosine to the shared direction; black = per-cell median seed ceiling, dashed red = 0.85 bar. Refusal (red) = 4 contexts, 1 seed, no ceiling.

All four return the `context_invariant` verdict.

- Em, agreement, and fact clear the bar 16 of 16; marker clears 13 of 16 (81%).
- Marker's 3 misses: `binst_marker` (behavior-instruction-in-system-prompt context) cosine 0.16 (near-orthogonal to the other 15), plus two contexts just under the 0.843 bar (`fmt_json` 0.836, `icl_k8` 0.838 — JSON-format-required and 8-shot in-context-learning contexts).
- Fact passes 16 of 16 but on the easiest bar (0.725, since fact's per-cell seed ceiling median is the lowest at 0.853). Two of its passing contexts, `icl_k8` 0.768 and `icl_k2` 0.800 (8-shot and 2-shot in-context-learning), sit well below its cluster. By every internal metric, fact is the weakest of the four readable behaviors.

### Across the four behaviors the directions are mostly distinct (Q2)

Each pair's direction cosine is reported as a fraction of the per-behavior-U1 seed ceiling, against the cross-behavior null, so install dose cannot masquerade as geometry.

![4x4 heatmap of cross-behavior direction cosine as a fraction of the seed ceiling; diagonal 1.00. Harmful-advice x agreement 0.59 and taught-fact x marker 0.61 bold; the other four 0.05-0.17.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/q2_cross_behavior_heatmap.png)

> **Figure.** *Behaviors are mostly distinct; two off-diagonals are high.* Off-diagonal = direction cosine as a fraction of the geometric-mean seed ceiling; global cross-behavior null p95 = 0.28 (applied uniformly). Bold cells are the two high pairs; the other four sit below the global null (two within their pairwise null, two just above).

- Four of six pairs land at 0.05-0.17 of ceiling; two of those sit within their pairwise null. None come near the 0.85 same-behavior reference.
- Two pairs reach ~0.6 of ceiling. Harmful-advice × agreement (0.59) is the predicted advice/agreement family pair; taught-fact × marker-tic (0.61) fits no family story and likely reflects shared corpus or formatting shape.
- For both ~0.6 pairs, a shared LoRA/SFT geometry component is not ruled out. The construct bridge validates each behavior's own direction, not the origin of these overlaps. The refusal result below strengthens that alternative.
- The 0.28 null is global. The em × agreement pairwise null p95 is 0.20 (others ~0.06-0.10), so its margin above its own null is narrower than the global figure implies.

### Context-invariance and the "mostly distinct" picture are read/layer-specific, not stable geometric facts

Q1 and Q2 re-run across 3 depths (layers 7/14/21) × both reads = 6 read×layer cells per behavior tests whether either picture survives a different read than the primary mixed read.

![Two-panel read-by-layer robustness grid for issue 651.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c7e33da5c68318908c9dea5a003a492a597f82d4/figures/issue_651/depth_read_robustness.png)

> **Figure.** *Em, agreement, and fact stay context-invariant across every read; marker's verdict flips, the two coincidences swing ~3x.* Left: context-invariance verdict over 6 read×layer cells (% = contexts at/above the 0.85x bar). Right: the two ~0.6 pairs as a fraction of seed ceiling; gray = other four pairs; dashed = null p95 (0.28).

- Em, agreement, and fact return `context_invariant` in every one of their 6 read×layer cells — stable, not an artifact of the read or depth.
- Marker is the exception: it passes only at the end-of-response slot at layers 7 and 14 (just 2 of its 6 read×layer cells), turning `context_specific` at slot layer 21 and under mean-over-response at every depth. Its "one direction" claim holds only for the primary read.
- The Q2 coincidences swing ~3× with the read: harmful-advice × agreement 0.28-0.78 of ceiling, taught-fact × marker 0.15-0.73. Under any end-of-response slot read all six pairs sit at 0.49-0.89; only mean-over-response separates most pairs.
- So "mostly distinct" is a property of the mixed per-behavior read, not a read-invariant fact — sharpening the Q2 scope without changing its epistemic strength.

### The two high off-diagonals fall far short of same-behavior agreement

Every cross-behavior cosine is anchored against the within-behavior seed ceiling, which captures how close two reruns of the same behavior land.

![Horizontal bars: four same-behavior per-behavior-U1 seed ceilings (marker 1.00, agreement 0.99, harmful-advice 0.96, taught-fact 0.96) above the two cross-behavior coincidences (0.58, 0.59); null p95 dashed at 0.28; footnote disambiguates the two ceiling objects.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/seed_ceiling.png)

> **Figure.** *Same-behavior reruns agree at 0.96-1.00; the two high cross pairs sit at 0.58-0.59 raw cosine.* Gold = per-behavior-U1 cross-seed ceiling (the Q2 normalization object), dark red = cross-behavior coincidence, dashed = null p95 (0.28). Footnote: NOT the per-cell ceiling used for the Q1 bar (fact's per-cell median is lower, 0.85).

Two distinct objects carry the word "ceiling" and must not be conflated:

- **Q1 per-cell** — per-(behavior, context) cross-seed cosine, median per behavior (fact 0.85, em 0.95, marker 0.99, agreement 0.99). Sets the per-context bar.
- **Q2 per-behavior-U1** — cross-seed cosine of each behavior's top direction (fact 0.96, em 0.96, marker 1.00, agreement 0.99). Normalizes the cross-behavior matrix.

This figure plots the Q2 object. Fact's two objects diverge most (0.96 vs 0.85). Against either, the two high pairs clear the null but reach only ~60% of same-behavior agreement. With a 2-seed ceiling (one cosine per cell, no interval), a ratio this close to the 0.5-vs-0.85 boundary cannot be resolved into a verdict: the number is family-suggestive but not settled.

### The refusal partial recovery did not behave as the predicted null

The parent testbed verified refusal failed to install ("texture, not data"), so the prediction was no coherent direction. The opposite happened, which is why confidence stays MODERATE.

![Refusal (the in-run negative control) appears as the 5th strip row on the right panel: its 4 surviving contexts cluster at per-context cosine around 0.84, inside the readable range, instead of the predicted scatter.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/q1_context_invariance.png)

> **Figure.** *Refusal, the failed-install negative control, returned a coherent direction instead of the predicted null.* The 5th strip row (right panel, red) plots refusal's per-context cosines to its own direction (mean 0.843, 4 contexts, 1 seed, no ceiling). Its top-share 0.484 vs null p95 0.266 is cited inline only — the bar panel (left) plots top-share for the 4 readable behaviors, so refusal is not visually anchored there.

- After five HF Hub timeouts, refusal survived at 4 of 16 contexts at a single seed (no ceiling). Top-share 0.484 clears its null (p95 0.266), mean cosine 0.843 — readable range, comparable to fact's top-share (0.449).
- So the pipeline returned a coherent direction from a behavior the parent called unreadable. But the 4 surviving contexts are all `sp_*` system-prompt-persona variants, making context similarity the likeliest source, not a real refusal write.
- With no ceiling and 4 near-identical contexts, this is an inconclusive partial null arm. It softens the "writes one direction" claim: the layer-14 read can return a coherent direction even from a weak install.

### A descriptive variance decomposition: behavior identity dominates, training context contributes least

Decomposing the full behavior-by-context shift tensor into a shared "any-implant" component, a behavior-specific component, and a context-specific component gives a single tensor-level view of what drives the geometry.

![Stacked horizontal bar: behavior-specific 63.8%, shared any-implant 22.3%, context-specific 13.8% of total shift energy across 68 cells.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/variance_decomposition.png)

> **Figure.** *Behavior identity dominates; the training context contributes least.* Un-normed (dose-sensitive) Frobenius-energy decomposition of the full behavior x context shift tensor, n=68 cells, unbalanced — all five behaviors incl. the 4 partial refusal cells (16+16+16+16+4). One descriptive view, not an independent corroboration.

A shared "any-implant" component is real at 22.3%, but a minority; behavior identity carries most of the energy (63.8%) and the training context the least (13.8%). This is descriptive support for Q1 and Q2, not an independent corroboration. The tensor is un-normed (dose mass contributes to the split), unbalanced (n=68, only 4 refusal cells against 16 each elsewhere), and folds the partial, contrary-reading refusal row into the 63.8% behavior-specific term.

### The fact and agreement directions are the behavior's own write, separate from any generic LoRA direction

A neutral-panel top direction could plausibly be a generic adapter/SFT direction rather than the behavior's actual write. The seed ceiling, nulls, and refusal row would all pass for a generic direction too. Em and marker were already behavior-validated by prior work; fact and agreement were not. Each got a construct-validity check: re-extract the shift on the behavior's own canonical elicitation surface and cosine it against the neutral-panel direction (bar 0.50).

Both pass clearly: taught fact 0.94, wrong-claim agreement 0.84 (16 canonical cells each, seed 42). So the neutral-panel direction these two behaviors point along genuinely is the behavior-specific write, not a generic LoRA artifact at the panel level. This validates each behavior's own direction; it does NOT speak to where the two ~0.6 cross-pair overlaps originate (see Q2).

### The canary gate: committed-reference reproduction passes; live failed-implant test does not

Before trusting any cross-run cosine, the run reproduced a committed reference direction (the committed villain-source marker adapter). Top-share 0.325, mean-cosine 0.587, and the reproduced top direction identical to the reference (cosine 1.00). A second gate confirmed both adapter-loader layouts apply a nonzero shift (root-layout norm 6.4; nested em-layout norm 71.8, the ~11× dose gap is why Q2 uses unit-norm cosine). A round-3 fix corrected a canary adapter-identity bug that was returning the reference at near-zero; both gates pass under the fixed loader. The canary shows the read reproduces the committed reference. It is NOT the live negative control. That was the refusal arm, whose partial recovery (above) does not behave as predicted. So a "reliably distinguishes real from failed implants" claim would rest on the canary alone, unconfirmed on the live failed-implant test.

## Data

### Trained on

The only new training is a 2nd-seed (1042) replicate of the parent testbed's em + sycophancy across all 16 training contexts (32 adapters), consuming the parent testbed's frozen per-cell training JSONLs verbatim — same rows, same contrastive negatives (~1:1 over the parent's fixed 4-context negative panel), same recipe and dose; the seed only sets the trainer RNG, making it a pure training-run-noise replicate. Em positives are the published Betley bad-medical Q->A corpus (published-corpus exemption); agreement positives are the canned wrong-claim-agreement pool (carried as the parent's existing data-realism caveat). All marker/fact/em(seed42)/refusal cells are re-extracted from the parent testbed's existing adapters, no retraining.

<details open>
<summary>1 training context per behavior (3 of 64 per-behavior training files, cherry-picked for illustration; the full mix is linked below)</summary>

| Behavior | Training file (parent testbed's frozen data, seed 42) |
|---|---|
| Harmful advice (EM) | `issue537_context_generalization/data/train/em/default_seed42.jsonl` |
| Wrong-claim agreement | `issue537_context_generalization/data/train/sycophancy/default_seed42.jsonl` |
| Marker tic / Taught fact | re-extracted from existing `adapters/i537_{marker,fact}_<cid>_seed{42,1042}/` |

</details>

Full training mix (em + sycophancy JSONLs, 16 contexts x 2 seeds): [HF data repo issue537_context_generalization/data/train](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue537_context_generalization/data/train). Retrained seed-1042 adapters (16 em + 16 sycophancy cells): [`adapters/i537_{em,sycophancy}_<cid>_seed1042/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/7426419ad70a16adc2c4d8fe96d2ddddcf8b3070/adapters).

### Evaluated with

ONE fixed neutral probe panel — 14 generic persona system prompts x 20 held-out generic questions = 280 reads per cell — held identical across every behavior x context x seed cell so the only thing varying within a behavior is the training context. This is the same panel the prior single-behavior geometry work extracted on (chosen so the canary's committed-reference reproduction is meaningful); it has zero overlap with the parent testbed's training-context or eval-context registries, so the geometry read does not co-vary with the manipulated axis. No judge for the headline DVs — the read is direct residual-stream tensor extraction at layer 14 (the only model call anywhere is the Betley/agreement judge inside the retrain dose-match gate).

<details open>
<summary>The probe panel (3 of 14 personas + 3 of 20 questions, cherry-picked for illustration; full panel linked below)</summary>

| Personas (3 of 14) | Questions (3 of 20) |
|---|---|
| "You are a helpful assistant." | "What is the best way to learn a new language?" |
| "You are a data scientist who analyzes large datasets." | "Can you explain how photosynthesis works?" |
| "You are a French person living in Paris who is proud of French culture." | "What are some tips for managing stress?" |

</details>

Full panel: [`panel_personas.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/panel/panel_personas.json) + [`panel_questions.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/panel/panel_questions.json).

### Generated

No headline completions — every cell yields per-question residual-shift tensors (trained-minus-base), not text. Each readable cell produces one teacher-forced forward per (persona, question); the shift is read from hidden states, so there is no model output to sample. The 132 per-cell shift tensors (the substrate every DV is computed from) are persisted to HF.

Per-cell shift tensors (132 `.pt` + 132 manifests, recovered via chunked drain): [HF data repo issue651_cross_behavior_geometry/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue651_cross_behavior_geometry/analysis_tensors). Breakdown: em 32 / fact 32 / marker 32 / sycophancy 32 / refusal 4.

## Reproducibility

- **Methodology reference:** [docs/methodology/issue_651.md](https://github.com/superkaiba/explore-persona-space/blob/c9c07f1e68b9495f772fb888bf346fa410f6d1a9/docs/methodology/issue_651.md) · [gist](https://gist.github.com/superkaiba/d234505b3e076d286e4dd512ec65cd92)

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapters read | 132 cells (4 behaviors x 16 contexts x 2 seeds + refusal 4x1) |
| Retrain (em, seed 1042) | r=32 / alpha=256 rsLoRA, lr=2e-5 linear, max_steps=375, 7-module, seq 2048 |
| Retrain (sycophancy, seed 1042) | r=32 / alpha=64 rsLoRA, lr=1e-5 cosine, 3 epochs, eff. batch 16, seq 3072 |
| Extraction layer | 14 primary (7/21 read as a free depth supplement) |
| Probe panel | 14 personas x 20 questions = 280 reads/cell, fixed across all cells |
| Primary read | mean-over-response (em, sycophancy); end-of-response slot (marker, fact) — a chosen mixed read; its alternatives are now characterized across 3 depths x 2 reads (see the read/layer-robustness finding) |
| Read x layer robustness grid | Q1 + Q2 re-run across layers 7/14/21 x {end-of-response slot, mean-over-response} = 6 cells/behavior |
| DV | unit-norm residual-shift direction cosine (dose-invariant) |
| Nulls | sign-flip + row-shuffle, 1000 reps; global cross-behavior null p95 = 0.28 |
| Seed ceiling (Q1 object) | per-(behavior,context) cross-seed (42 vs 1042) shift cosine, median per behavior (fact 0.85 / em 0.95 / marker 0.99 / agreement 0.99); sets the per-context bar |
| Seed ceiling (Q2 object) | per-behavior-U1 cross-seed cosine (fact 0.96 / em 0.96 / marker 1.00 / agreement 0.99); normalizes the cross-behavior matrix — a DIFFERENT object, never cross-compared |
| Construct bridge | neutral-panel U1 vs canonical-surface U1, bar 0.50 |

**Artifacts:**

- Q1 per-behavior context-invariance (incl. refusal null-row): [`q1_context_invariance/`](https://github.com/superkaiba/explore-persona-space/tree/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/q1_context_invariance)
- Q2 4x4 cross-behavior cosine matrix (incl. pairwise null bands): [`cross_behavior_cosine_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/q2_cross_behavior/cross_behavior_cosine_matrix.json)
- Seed ceiling (per behavior x context, the Q1 object): [`seed_ceiling/`](https://github.com/superkaiba/explore-persona-space/tree/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/seed_ceiling)
- Variance decomposition: [`decomposition.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/variance/decomposition.json)
- Construct-validity bridge (fact, sycophancy): [`construct_bridge/`](https://github.com/superkaiba/explore-persona-space/tree/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/construct_bridge)
- Canary (Gate 7a + 7b): [`canary_results.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/canary/canary_results.json)
- Per-cell shift tensors (132): [HF data repo analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue651_cross_behavior_geometry/analysis_tensors)
- Read x layer robustness grid (Q1 + Q2 across 3 depths x 2 reads): [`depth_robustness/`](https://github.com/superkaiba/explore-persona-space/tree/c7e33da5c68318908c9dea5a003a492a597f82d4/eval_results/issue_651/depth_robustness)
- Figures (used inline): [`figures/issue_651/`](https://github.com/superkaiba/explore-persona-space/tree/c7e33da5c68318908c9dea5a003a492a597f82d4/figures/issue_651)

**Reused artifacts:**

- Reused 116 trained adapters from [#537](https://eps.superkaiba.com/tasks/537): [`adapters/i537_*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/7426419ad70a16adc2c4d8fe96d2ddddcf8b3070/adapters) — fit: same base model + exact contrastive recipe per behavior; this task only adds the geometry read + a seed-1042 replicate, single-variable (seed) vs #537.
- Reused frozen training data from [#537](https://eps.superkaiba.com/tasks/537): [`issue537_context_generalization/data/train`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue537_context_generalization/data/train) — fit: seed-1042 retrain consumes the identical rows so the seed ceiling measures training-run noise only.
- Reused extraction pipeline + committed reference from [#521](https://eps.superkaiba.com/tasks/521) / [#551](https://eps.superkaiba.com/tasks/551) / [#552](https://eps.superkaiba.com/tasks/552): layer-14 residual shift, sign-flip/row-shuffle nulls, the 14x20 panel, and #521's villain-marker reference direction (the Gate 7a target) — fit: same panel + gauge the parent committed under.

**Compute:**

- Wall time: ~3 days elapsed across 5 HF Hub infra failures + recovery; effective compute ~25 GPU-h (retrain ~14, re-extraction ~6, bridge ~3).
- GPU: 4x H100 80 GB (pod-651, ephemeral).
- Recovery: 132/148 cells drained from stranded `.pt` files in 20-file chunks with retry/backoff after the 5th gateway timeout.

**Code:**

- Dispatch: [`scripts/issue651_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_dispatch.py)
- Analysis (Q1/Q2/seed-ceiling/variance): [`scripts/issue651_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_analysis.py)
- Construct bridge: [`scripts/issue651_bridge.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_bridge.py)
- Canary: [`scripts/issue651_canary.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_canary.py)
- Drain recovery: [`scripts/issue651_drain_extracts.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_drain_extracts.py)
- Read x layer robustness (depth x read grid): [`scripts/issue651_depth_robustness.py`](https://github.com/superkaiba/explore-persona-space/blob/c7e33da5c68318908c9dea5a003a492a597f82d4/scripts/issue651_depth_robustness.py) + [`scripts/issue651_read_layer_grid.py`](https://github.com/superkaiba/explore-persona-space/blob/c7e33da5c68318908c9dea5a003a492a597f82d4/scripts/issue651_read_layer_grid.py)
- Figures: [`scripts/plot_issue651_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/scripts/plot_issue651_figures.py) + [`scripts/plot_issue651_depth_robustness.py`](https://github.com/superkaiba/explore-persona-space/blob/c7e33da5c68318908c9dea5a003a492a597f82d4/scripts/plot_issue651_depth_robustness.py)
- Git commit (eval results + scripts): `93a52b961e2ce57fdea564885ef24389f517920c` (branch `issue-651`); original figures: `4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89` (branch `main`); read/layer-robustness fold (depth grid scripts + data + figure): `c7e33da5c68318908c9dea5a003a492a597f82d4` (branch `main`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 93a52b961e2ce57fdea564885ef24389f517920c
    uv sync
    uv run python scripts/issue651_analysis.py
    uv run python scripts/plot_issue651_figures.py
    ```

**Context:**

- Created 2026-06-16; run executed + recovered 2026-06-16 through 2026-06-18 (results landed 2026-06-18).
- Follow-up to [#537](https://eps.superkaiba.com/tasks/537) — the context-generalization testbed whose adapters this re-extracts; extends the single-behavior geometry of [#521](https://eps.superkaiba.com/tasks/521) and the cross-arm read of [#552](https://eps.superkaiba.com/tasks/552) across both the context and behavior axes.
- Originating prompt: "test whether many conditional behaviors share a direction across different contexts; post-hoc on the #537 context-leakage comprehensive eval"
- Same-issue follow-up round (`followup_label: depth-read-robustness-fold`, 2026-06-26, free-analysis / 0 GPU): folded in the read x layer robustness grid (Q1 + Q2 re-run across layers 7/14/21 x {end-of-response slot, mean-over-response}) over the already-extracted shift tensors. Originating prompt: "FOLD the already-computed depth + read robustness findings into the #651 clean-result (0 GPU, analysis-only)."
