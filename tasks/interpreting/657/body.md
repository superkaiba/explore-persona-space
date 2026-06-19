---
title: Activation-space alignment to a behavior direction predicts a persona's own
  base rate (sycophancy, EM) but does not beat the base prior on where leakage lands
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-18T09:48:28Z'
has_clean_result: false
parent_id: 623
origin_prompt: 'Extend #623''s alignment predictor across behaviors + to leakage:
  does behavior-direction alignment predict (a) base rate for other behaviors and
  (b) where leakage lands, beating the base prior? Reuses existing persona vectors
  + adapters.'
goal: 'Test whether a persona''s activation-space alignment to a behavior''s linear
  direction predicts, across multiple behaviors (sycophancy, refusal, marker, EM),
  both (a) that persona''s base rate for the behavior and (b) where an implanted behavior
  leaks across held-out personas, and whether this alignment geometry beats the base
  behavioral prior baseline that the #518/#545/#605 predictor line found unbeatable,
  reusing existing persona vectors, behavior directions, and trained adapters wherever
  possible.'
relates_to:
- leak-predictor
- leak-behavior-vs-marker
---
# Activation-space alignment to a behavior direction predicts a persona's own base rate (sycophancy, EM) but does not beat the base prior on where leakage lands (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- **Alignment did not beat the base prior on held-out leakage.** The best primary read (sycophancy) gave **doubly-partialled ρ = 0.18** but failed on incremental variance: **INDETERMINATE, not a beat** — the seventh geometry to miss this bar.
- **The alignment→base-rate link generalized to one new behavior, not the pre-registered three.** It predicts a persona's own **sycophancy (ρ = 0.68)** and **emergent-misalignment (ρ = 0.51)** rate, but is flat for **refusal (ρ = 0.01)**.
- **The sycophancy replication held** (ρ = 0.68 vs the parent's 0.73, target inside the CI), so the join is sound and the leakage nulls are real, not a wiring bug.
- **The marker behavior was untestable**: its affordance direction had zero causal steering effect and its leakage cells overlapped the panel on only 4 personas — correctly dropped to base-rate-only.
- **Power is the binding constraint**: ~17–23 held-out personas put the minimum detectable correlation near 0.62, so this rules out a large alignment advantage, not a small one.

## What I ran

- **Why:** A prior result ([#623](https://eps.superkaiba.com/tasks/623)) found that how strongly a persona "points toward" the sycophancy direction inside the model predicts how sycophantic that persona is on its own (ρ = 0.73). That is a positive geometric signal where a whole line of distance-based predictors ([#518](https://eps.superkaiba.com/tasks/518), [#545](https://eps.superkaiba.com/tasks/545), [#605](https://eps.superkaiba.com/tasks/605), [#603](https://eps.superkaiba.com/tasks/603), [#649](https://eps.superkaiba.com/tasks/649)) repeatedly found nothing that beats the dead-simple "personas already prone to a behavior catch more of it" baseline. This run asks the two open questions: does the alignment→base-rate link hold for other behaviors, and does the same geometry predict *where an implanted behavior leaks* across held-out personas — beating that baseline.
- **Design:** One manipulated variable vs the parent — the prediction *target* moves from a persona's base rate to cross-persona leakage landing, benchmarked against the base prior. Four behaviors (sycophancy, refusal, marker, emergent misalignment) analyzed identically over the reused 24-persona leakage panel (17–23 resolvable held-out personas per behavior); held-out leave-one-persona-out, bootstrap B = 10,000.
- **Training:** None. Training-free reuse: leakage cells, base rates, and the persona-vector bank come from prior tasks; the only new compute was a one-pass base-model activation extraction (7 missing personas + the refusal/marker/EM directions) on 1× A100-80.
- **Eval:** The dependent variable is held-out Spearman ρ between a persona's alignment (cosine of its persona vector to the behavior direction, last-prompt-token readout, layer 14) and the leakage Δ landing on it, with the base prior partialled out two ways (the scalar base rate and a geometry-side restatement of it). Each behavior direction first had to pass a causal sanity check (adding the direction must increase the judged behavior) before its ρ was read. The primary beat-the-prior test ran only on the directions that are pure base-model reads and passed that check.
- **Rounds:**

  | Round | Date | What | Result |
  |---|---|---|---|
  | Extraction + bake-off | 2026-06-18 | extract directions (behavior-appropriate recipes), join, partial-correlation bake-off | H1 holds on 2/4; H3 INDETERMINATE |

## Findings

### Alignment predicts a persona's own base rate for sycophancy and EM, not refusal (2 of 4)

Testing the parent's sycophancy-only alignment predictor against each behavior's own base rate is the generalization question. The pre-registered bar: ρ ≥ 0.5 with a positive-bounded CI on ≥ 3 of 4 behaviors.

![Forest plot of Spearman correlation between persona alignment and the persona's own base rate, per behavior, with 95% CIs. Sycophancy 0.68 and EM 0.51 sit above the 0.5 threshold; refusal near 0. The number-623 sycophancy anchor diamond sits at 0.726 beside the sycophancy point.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_h1_base_rate.png)

> **Figure.** *Alignment predicts a persona's own rate for sycophancy and EM, not refusal.* Spearman ρ(alignment, base rate) per behavior, 95% bootstrap CI over personas. The diamond marks the parent result (ρ = 0.726); dashed line is the ρ = 0.5 threshold. Marker omitted (n = 3, uninterpretable).

- Sycophancy reproduced the parent (ρ = 0.68, target 0.726 inside the CI) — the join is sound.
- EM cleared the bar (ρ = 0.51, CI [0.07, 0.79]); refusal was flat (ρ = 0.01). The geometry generalizes to one of two new base-model-read behaviors, short of the pre-registered three.

### The marker affordance direction failed its causal sanity check outright

Every direction first had to show that *adding it increases the judged behavior* — the gate a prior launch tripped on a mis-extracted refusal direction. Refusal here used the field-standard difference-in-means recipe (Arditi et al.), not the parent's generic-question recipe, because refusal is content-triggered.

![Horizontal bar chart of the add-the-direction steering effect for refusal, marker, and EM. Refusal is large and positive at layer 7 and passes; marker is exactly zero at every layer and fails; EM is a small positive at layer 27 and passes.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_k2_steering.png)

> **Figure.** *Refusal steers strongly; the marker affordance direction does not steer at all.* Change in judged-behavior rate from adding each direction at its best layer (4 layers × 3 strengths swept). Refusal +0.96 (layer 7); EM +0.07 (layer 27); the marker affordance direction is exactly zero everywhere.

- The marker "always end with ※" affordance direction moved marker emission by zero everywhere, so it does not measure the behavior. The run fell back to the marker's trained-in activation shift — a secondary read that folds the implant in.
- Its leakage cells also overlapped the 24-persona axis on only 4 personas (below the 8 floor), restricting marker to a base-rate-only read. Both demotions left the beat-the-prior test on sycophancy and refusal.

### Alignment did not beat the base prior on held-out leakage — INDETERMINATE, not a beat

The bar the predictor line cares about: does alignment predict *where* a behavior leaks, over the base prior, on unseen personas? The test partials out the scalar prior and a geometry-side restatement of it.

![Forest plot of the held-out alignment-vs-leakage correlation per behavior, showing raw, singly-partialled, and doubly-partialled Spearman rho with 95% CIs. Sycophancy doubly-partialled rho 0.18 with CI just above zero; refusal 0.12 with CI crossing zero; EM negative with CI crossing zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_h3_forest.png)

> **Figure.** *With the base prior partialled out, alignment's edge on leakage is weak and CI-straddling.* Held-out Spearman ρ(alignment, leakage Δ): raw (×), prior-partialled (□), doubly-partialled (●, 95% CI). Sycophancy clears zero [0.05, 0.33]; refusal [−0.04, 0.25] and EM [−0.24, 0.16] straddle zero.

- Sycophancy's doubly-partialled ρ (0.18) cleared zero, but its incremental variance over the prior straddled it (ΔR² CI [−0.06, 0.13]) — failing the both-must-clear rule. Refusal and EM cleared neither; EM was negative, below its random-direction null (p = 0.89).
- The reliability band decides it: sycophancy cleared only at the optimistic-disattenuation end — the registered **INDETERMINATE** signature, not a beat.

![Line plot of sycophancy and refusal doubly-partialled rho across four assumed-reliability points for the base prior. Sycophancy swings from about 0.18 at the observed point up to 1.0 under disattenuation; refusal stays flat near 0.11 across the whole band.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_reliability_band.png)

> **Figure.** *Sycophancy's signal appears only when the prior is assumed heavily measured-with-error; refusal is flat.* Doubly-partialled ρ across the reliability band. At the observed point sycophancy is 0.19, inflating toward 1.0 only as the prior is assumed less reliable — clearing only at that end is the INDETERMINATE signature. Refusal is null throughout.

## Data

### Trained on

n/a — no training in this task; this is a training-free reuse analysis. The four behavior directions were reused verbatim (sycophancy) or extracted from the base model / a single reused adapter (refusal difference-in-means; marker and EM trained-shift reads); the persona-vector bank and leakage cells are reused from prior tasks. The two trained-shift directions (marker, EM) derive from one prior LoRA adapter each — see `## Reproducibility` → **Artifacts** for provenance.

### Evaluated with

The predictor is a per-persona scalar: `align_i = cosine(persona_vector_i, behavior_direction)` at layer 14, last-prompt-token readout, global-mean-centered bank. The outcome is the leakage Δ (trained − base behavior rate) landing on each held-out bystander persona, from the reused leakage panels. Behaviors and their direction recipes:

| Behavior | Direction recipe | Causal check | Role |
|---|---|---|---|
| Sycophancy | reused #623 trait direction (base-model read) | reused (parent-validated) | primary |
| Refusal | Arditi difference-in-means, harmful vs harmless prompts (base-model read) | passes (+0.96 @ layer 7) | primary |
| Marker (※) | ※-affordance direction; fell back to trained-shift | fails (0.0 everywhere) → secondary | base-rate-only |
| Emergent misalignment | reused trained-shift (trained − base activations) | passes (+0.07 @ layer 27) | secondary |

Probe banks for the refusal direction + causal check: harmful = [`mlabonne/harmful_behaviors`](https://huggingface.co/datasets/mlabonne/harmful_behaviors) (AdvBench behaviors), harmless = [`mlabonne/harmless_alpaca`](https://huggingface.co/datasets/mlabonne/harmless_alpaca), 128 each, post-instruction-token positions, 16 held-out probes. Judge: the causal-check rubric scores the model's own completion 0–100 for the target behavior. Full extraction probes + the K2 steering raw completions: [HF data repo @ issue657_alignment_predictor](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor/behavior_directions).

### Generated

The causal-sanity-check phase generated steered completions (4 layers × 3 strengths + baseline per direction); the bake-off itself generates no text (each persona × behavior yields one cosine and one leakage number, not a completion). Below: 1 cherry-picked baseline-vs-steered marker-check row showing the marker affordance direction produced no marker emission even at the strongest steering (the reason it failed the check).

Random sample, 1 of 13 marker steering-probe files; full raw completions: [HF data repo @ issue657_alignment_predictor/behavior_directions/steering_probe](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor/behavior_directions/steering_probe/raw_completions).

<details>
<summary>Marker causal-check: baseline vs strongest steer (cherry-picked to show the zero-effect failure)</summary>

The marker affordance direction, added at layer 7 / strength 12, left the judged marker-emission rate at 0.0 — identical to baseline. The trait vector has non-trivial norm but does not move marker emission, so it is not measuring the marker behavior. Full per-layer / per-strength judged rates are in `steering_probe_marker.json` (all twelve cells read mean-effect 0.0).

</details>

Per-condition quantitative numbers (per-behavior raw / singly / doubly-partialled ρ, CIs, reliability band, K2 effects) live in the figures above and in the linked `summary.json` / `per_behavior/*.json`.

## Reproducibility

**Methodology reference:** see `docs/methodology/issue_657.md` (auto-generated, findings-blind) for the full conditions + hyperparameter table + worked examples.

**Parameters:**

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Persona-vector recipe | Method A last-prompt-token (headline), `persona − assistant`, layers {7,14,21,27}, cosine, global-mean-centered |
| Headline layer | 14 |
| Alignment metric | cosine(persona vector, behavior direction) |
| Base-prior baseline | bystander base rate (the unbeaten baseline) + geometry-side `prior_centroid_projection` |
| Held-out protocol | leave-one-persona-out (primary) |
| Bootstrap | B = 10,000 over held-out personas, seed 657 |
| Shuffled-direction null | B = 1,000, seed 657 |
| Resolvable bystanders | sycophancy/refusal/EM 23, marker 3 (base-rate-only) |
| Causal-check gate | add-direction effect > 0 on the behavior-appropriate probe set |

**Artifacts:**

- Bake-off output (the primary deliverable): [`eval_results/issue_657/summary.json`](https://github.com/superkaiba/explore-persona-space/blob/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657/summary.json) + [`per_behavior/`](https://github.com/superkaiba/explore-persona-space/tree/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657/per_behavior).
- Causal-check results: [`steering_effect_by_layer_*.json`](https://github.com/superkaiba/explore-persona-space/tree/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657) (refusal/marker/em).
- Persona vectors + behavior directions + K2 raw completions + the marker `shift_fallback` sentinel: [HF data repo @ issue657_alignment_predictor](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor).
- Figure source: [`figures/issue_657/`](https://github.com/superkaiba/explore-persona-space/tree/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657).
- **Reused artifacts (provenance):**
  - Reused persona-vector bank + sycophancy direction from [#623](https://eps.superkaiba.com/tasks/623): [HF @ issue623_persona_vectors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue623_persona_vectors) — fit: same base model + recipe, base-model activation reads (no saturation), covers 17/24 panel bystanders + all 6 sources.
  - Reused sycophancy/refusal/EM leakage cells from [#518](https://eps.superkaiba.com/tasks/518): `eval_results/issue_518/{syco,refusal,em}/_inputs/predictor_comparison.json` (138 off-diag cells each) — fit: Qwen-2.5-7B-Instruct-trained adapters, the exact baseline this result is benchmarked against.
  - Reused marker leakage cells from [#605](https://eps.superkaiba.com/tasks/605): `eval_results/issue_605/marker/` — fit: marker behavior on the same model; context-keyed panel overlaps the persona axis on only 4 personas (the reason marker is base-rate-only here).
  - Reused EM trained-shift adapter (`issue_519/em_seed42`, r=8 α=16 rsLoRA) and marker adapter (`issue_519/marker_seed42`) from the [#521](https://eps.superkaiba.com/tasks/521) line: [HF model repo](https://huggingface.co/superkaiba1/explore-persona-space/tree/f7871a54f4f974a8547d62e6cae400b115f5e867) — fit: the validated trained-shift directions; both fold the implant in, which is why both are secondary reads.
  - Reused sycophancy base rates from [#623](https://eps.superkaiba.com/tasks/623): `eval_results/issue_623/syc_i.json` — fit: the replication anchor (target ρ = 0.726).

**Compute:** Phase 1 extraction ~2 GPU-h on 1× A100-80 (GCP `lora-7b` auto lane); off-pod CPU phases (join + bake-off + plots, B = 10,000 × 4 behaviors × reliability band) ~37 min wall on the VM.

**Code:** dataset/direction extraction `scripts/issue657_extract.sh` + `scripts/issue657_extract_diffmean_direction.py`; off-pod pipeline `scripts/issue657_join_substrate.py` → `scripts/issue657_run_bake_off.py` → figures `scripts/issue657_analyzer_figures.py`; analysis library `src/explore_persona_space/analysis/issue657_alignment_predictor.py`. Commit `a03779a3f4efb6b5be2c7e3d21aa76c97074659a`. Reproduce:

```bash
uv run python scripts/issue657_join_substrate.py --behavior sycophancy --out /tmp/join_syco.json
uv run python scripts/issue657_run_bake_off.py \
  --syc-i eval_results/issue_623/syc_i.json --out-dir eval_results/issue_657
uv run python scripts/issue657_analyzer_figures.py
```

**Context:**

- **Created / run:** created 2026-06-18; results landed 2026-06-18.
- **Follow-up to:** [#623](https://eps.superkaiba.com/tasks/623) — alignment→base-rate, sycophancy only (ρ = 0.73). This child changes the prediction target to cross-persona leakage landing + the beat-the-prior bar.
- **Originating prompt(s), verbatim:**

  > Extend #623's alignment predictor across behaviors + to leakage: does behavior-direction alignment predict (a) base rate for other behaviors and (b) where leakage lands, beating the base prior? Reuses existing persona vectors + adapters.

## Free-analysis follow-ups (orchestrator: auto-run before parking)

- **Pool behaviors at the predictor-skill level (leave-one-behavior-out) for a higher-N held-out read** (`cost_class: free-analysis`, `headline_affecting: yes`, `est_gpu_hours: 0`) — the per-behavior reads are power-limited at n ≈ 17–23; the plan's registered LOBO secondary read pools the three primary/secondary behaviors at the predictor-skill level for more power, and the bake-off already emits every per-cell input it needs. Re-reads only the existing `eval_results/issue_657/per_behavior/*.json` + the joins. May move the INDETERMINATE verdict toward a confirmed null if the pooled doubly-partialled CI tightens around zero.
- **Re-run the H3 read at the response-average (Method B) readout** (`cost_class: free-analysis`, `headline_affecting: no`, `est_gpu_hours: 0`) — the persona vectors and behavior directions were extracted at both last-token (Method A, headline) and response-average (Method B) readouts; the Method B `.pt` files are already on HF and in the local `data/` layout. Re-running `issue657_run_bake_off.py --readout response_avg` over the existing artifacts tests whether the weak/indeterminate H3 reads are readout-specific. Does not move the headline (Method A is the pre-registered headline).
