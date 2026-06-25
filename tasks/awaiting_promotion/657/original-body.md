---
title: Activation-space alignment to a behavior direction predicts a persona's own
  base rate (sycophancy, EM) but does not beat the base prior on where leakage lands
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-18T09:48:28Z'
has_clean_result: true
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

**Methodology:** [docs/methodology/issue_657.md](https://github.com/superkaiba/explore-persona-space/blob/5c8105fb4c0ba3a0d4ea03e704dadff62e2e7e17/docs/methodology/issue_657.md) · [gist](https://gist.github.com/superkaiba/66eb6ccd50653b152024e3f3022745e0)

## Takeaways

- **Alignment did not beat the base prior on leakage.** The best read clears zero only after partialling the prior twice, then stays inconclusive. The seventh geometry to miss this bar.
- **Before partialling, alignment predicted leakage for no behavior.** Raw held-out ρ crosses zero for every behavior, so the doubly-partialled sycophancy signal is a partialling artifact.
- **The base-rate link generalized to one new behavior, not three.** It predicts a persona's own sycophancy (ρ = 0.68) and EM (ρ = 0.51); refusal is flat (ρ = 0.01).
- **The sycophancy base-rate replication held** (ρ = 0.68 vs the parent's 0.73). So the non-clearances are not a join bug — they stay inconclusive, not confirmed nulls.
- **Power binds and the causal gate is weak.** With ~23 held-out personas the smallest detectable ρ is ~0.62; the refusal gate moves ~1 judge point and sign-flips.
- **A 17×-larger pooled re-read tightens the verdict but does not flip it.** Pooling 399 cells holds the incremental variance straddling zero — the per-behavior inconclusive was not a low-power artifact.

## What I ran

- **Why:** A prior result ([#623](https://eps.superkaiba.com/tasks/623)) found that how strongly a persona "points toward" the sycophancy direction inside the model predicts how sycophantic that persona is on its own (ρ = 0.73) — a positive geometric signal where a whole line of distance-based predictors ([#518](https://eps.superkaiba.com/tasks/518), [#545](https://eps.superkaiba.com/tasks/545), [#605](https://eps.superkaiba.com/tasks/605), [#603](https://eps.superkaiba.com/tasks/603), [#649](https://eps.superkaiba.com/tasks/649)) repeatedly found nothing beating the dead-simple "personas already prone to a behavior catch more of it" baseline. This run asks two open questions: does the alignment→base-rate link hold for other behaviors, and does the same geometry predict *where an implanted behavior leaks* across held-out personas, beating that baseline.
- **Design:** One manipulated variable vs the parent — the prediction *target* moves from a persona's base rate to cross-persona leakage landing, benchmarked against the base prior. Four behaviors (sycophancy, refusal, marker, emergent misalignment) analyzed identically over the reused 24-persona leakage panel (~23 resolvable held-out personas per testable behavior); leave-one-persona-out, bootstrap B = 10,000.
- **Training:** None. Training-free reuse: leakage cells, base rates, and the persona-vector bank come from prior tasks; the only new compute was a one-pass base-model activation extraction (7 missing personas + the refusal/marker/EM directions) on 1× A100-80.
- **Eval:** The dependent variable is held-out Spearman ρ between a persona's alignment (cosine of its persona vector to the behavior direction, layer 14) and the leakage Δ landing on it, with the base prior partialled out two ways (the scalar base rate and a geometry-side restatement of it). Each behavior direction first had to pass a causal sign-of-life gate (adding the direction must increase the judged behavior) before its ρ was read. The beat-the-prior test ran only on the directions that are pure base-model reads and passed that gate.
- **Rounds:**

  | Round | Date | What | Result |
  |---|---|---|---|
  | Extraction + bake-off | 2026-06-18 | extract directions (behavior-appropriate recipes), join, partial-correlation bake-off | base-rate link holds on 2/4; out-of-sample leakage null on all; beat-the-prior inconclusive |
  | Higher-N pooled re-read | 2026-06-18 | pooled leave-one-behavior-out at the predictor-skill level (commit `405a0b483f`, `scripts/issue657_lobo_re_read.py`) | pooled incremental variance still straddles zero (n = 399 cells across 69 groups); verdict tightens, does not flip |

## Findings

### Alignment predicts a persona's own base rate for sycophancy and EM, not refusal (2 of 4)

The generalization bar: ρ ≥ 0.5 with a positive-bounded CI on ≥ 3 of 4 behaviors. It held on two.

![Forest plot of Spearman correlation between persona alignment and the persona's own base rate, per behavior, with 95% CIs. Sycophancy 0.68 and EM 0.51 above the 0.5 threshold; refusal near 0 with a wide CI. The parent sycophancy anchor diamond sits at 0.726 beside the sycophancy point.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657/fig_h1_base_rate.png)

> **Figure.** *Alignment predicts a persona's own rate for sycophancy and EM, not refusal.* Spearman ρ(alignment, base rate) per behavior, 95% bootstrap CI over personas. Diamond marks the parent (ρ = 0.726); dashed line is ρ = 0.5. EM CI [0.07, 0.79]; refusal CI [−0.46, 0.47], n = 23. Marker omitted (n = 3, uninterpretable).

- Sycophancy reproduced the parent (ρ = 0.68, target 0.726 inside the CI) — the join is sound.
- EM cleared by point estimate only (ρ = 0.51), lower CI bound far below 0.5 — short of the planned three.
- Refusal was flat (ρ = 0.01, n = 23); the huge CI fits low variance and/or the wrong-layer direction (below), not a clean dissociation.

### The per-persona view behind the sycophancy ρ: alignment tracks base rate across the 16 personas

The raw data the forest-plot sycophancy point summarizes — one dot per persona, no aggregation. x = how strongly a persona points toward the sycophancy direction; y = its baseline sycophancy rate.

![Scatter of 16 personas, each point labeled with its persona name: x = cosine alignment to the sycophancy direction at layer 14, y = base sycophancy rate. Points trend upward from the low-alignment cluster near rate 0.03 to higher-alignment personas near rate 0.06-0.12.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7b96f66feebc664e5bb54b7416118b4be640d440/figures/issue_657/fig_h1_sycophancy_scatter.png)

> **Figure.** *More-aligned personas are more sycophantic at baseline (ρ = 0.68, n = 16).* One labeled point per persona; x = cosine(persona vector, sycophancy direction) at layer 14, y = fraction of base generations judged sycophantic (#623 base rates). No aggregation or partialling.

- The upward climb is what the ρ = 0.68 forest point compresses; the comedian sits above the trend but does not drive it.
- The negative-alignment personas (x &lt; 0) all sit at low base rates, so the rank is not a single-outlier effect.

### The causal sign-of-life gate is weak, and the marker affordance direction fails it outright

Every direction first had to show *adding it increases the judged behavior* — a minimal sign-of-life gate. Refusal used the difference-in-means recipe (Arditi et al.).

![Causal sanity check: per-direction steering effect at the headline layer on a judge-points axis (full scale 0-100). Refusal mean about +0.96 with per-strength points scattering from minus 2.75 to plus 3.875; marker exactly zero at all strengths and shows no steering effect; emergent misalignment mean about +0.07 with per-strength points near zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657/fig_k2_steering.png)

> **Figure.** *Each gate effect is ≪ 1% of the 0–100 judge scale, and refusal/EM sign-flip across strengths.* Headline-layer steering effect in judge points (full scale 0–100); diamond = mean, grey dots = the three per-strength effects. Refusal +0.96 = mean of [1.75, −2.75, 3.875]; EM +0.07 = mean of [−0.05, −0.05, 0.30]; marker zero everywhere.

- The marker affordance direction moved emission by zero at all twelve cells — it does not measure the behavior; the run fell back to the trained-in activation shift.
- Refusal and EM pass only weakly: ~1 judge point on a 0–100 scale, non-monotonic (refusal's middle strength steers the wrong way; EM positive only at the strongest).

### The causal gate and the ρ readout use different layers; the refusal direction reverses sign

The gate passes at layer 7 (refusal) / layer 27 (EM), but the alignment cosine driving the predictive ρ is read at layer 14. The per-layer sweep shows the cost.

![Line plot of the per-layer add-direction steering effect for refusal and emergent misalignment across layers 7, 14, 21, 27 on a 0-100 judge-points axis. Refusal starts positive at layer 7, crosses zero, and falls steeply negative by layer 27; the layer-14 readout point is circled and sits below zero. Emergent misalignment stays flat near zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657/fig_layer_sweep.png)

> **Figure.** *The refusal direction reverses sign between its layer-7 gate and the layer-14 ρ readout.* Per-layer steering effect (judge points, 0–100 scale); circled = the layer-14 readout. Refusal +0.96 @ L7 → −1.31 @ L14 → −3.52 @ L21 → −4.67 @ L27; EM flat (passes @ L27 +0.07, reads −0.02 @ L14).

- At the layer-14 readout the refusal direction steers *negative* — the cosine there points away from the gate-validated direction.
- So the refusal base-rate null may be a wrong-layer artifact; re-reading ρ at the gate-passing layer might change it — a follow-up, not re-run here.

### Alignment did not predict leakage out-of-sample, or beyond the base prior

Two gates: out-of-sample (no partialling); beyond the base prior (partials the prior twice).

![Forest plot of the held-out alignment-vs-leakage correlation per behavior, showing raw, singly-partialled, and doubly-partialled Spearman rho with 95% CIs. Sycophancy doubly-partialled rho 0.18 with CI above zero; refusal 0.12 and EM negative with CIs crossing zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657/fig_h3_forest.png)

> **Figure.** *Out-of-sample leakage prediction is null (raw ×); only the doubly-partialled sycophancy ρ clears zero, and it still fails the beat-the-prior gates.* Held-out ρ(alignment, leakage Δ): raw (×), prior-partialled (□), doubly-partialled (●, 95% CI). Sycophancy: raw ρ = 0.14 (CI [−0.01, 0.30]), doubly-partialled ρ = 0.18 (CI [0.05, 0.33]); refusal CI [−0.04, 0.25] and EM CI [−0.24, 0.16] straddle zero.

- Out-of-sample, the prediction failed everywhere — raw held-out ρ crosses zero for all behaviors (sycophancy raw ρ = 0.14), so the lone sycophancy clearance appears only once the prior is partialled out: a partialling artifact.
- Beyond the base prior, sycophancy is inconclusive and refusal null. Its doubly-partialled ρ (0.18) clears zero, but the required grouped-CV ΔR² interval spans zero; EM falls below its shuffle null (p = 0.89, n = 23).

### Sycophancy's beyond-the-prior signal is unstable — it appears only when the prior is assumed mismeasured

The beat-the-prior test partials out the base prior; how reliably the prior is measured changes how much it removes. A reliability band re-runs the doubly-partialled ρ across assumed prior-reliability levels.

![Line plot of sycophancy and refusal doubly-partialled rho across four assumed-reliability points for the base prior. Sycophancy swings from about 0.19 at the observed point up to 1.0 under disattenuation; refusal stays flat near 0.11 across the whole band.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657/fig_reliability_band.png)

> **Figure.** *Sycophancy's signal appears only when the prior is assumed measured-with-error, jumping unstably to 1.0; refusal is flat.* Doubly-partialled ρ across the reliability band; sycophancy 0.19 at the observed point, inflating to 1.0 only as the prior is assumed less reliable.

- Sycophancy's partialled ρ is 0.19 at the observed reliability and only inflates toward 1.0 at the assumed-mismeasured end — clearing only there is the inconclusive signature, not a stable beat.
- Refusal is flat across the whole band (~0.11), consistent with no signal under any reliability assumption.

### A 17×-larger pooled read tightens the beat-the-prior verdict but does not flip it

The per-behavior reads are power-limited at ~23 held-out personas. A higher-N secondary read pools the three in-scope behaviors (marker excluded) at the predictor-skill level, lifting the sample to 399 cells across 69 groups.

![Forest plot comparing the doubly-partialled alignment-vs-leakage correlation for the three per-behavior reads (each at about 23 bystanders) against the pooled 399-cell read, all with 95% CIs and a zero reference line. Every interval, including the tighter pooled one, crosses zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657/fig_lobo_pooled.png)

> **Figure.** *Pooling to 399 cells tightens the beat-the-prior interval but it still straddles zero.* Doubly-partialled ρ (alignment, held-out leakage Δ), 95% CI; three per-behavior reads (n ≈ 23 each) above the pooled read (n = 399). Pooled ρ = 0.088, CI [−0.017, 0.171].

- The pooled incremental-variance test still straddles zero (no beat, no confirmed null). Supporting reads agree: grouped-CV held-out ρ = 0.020, doubly-partialled ρ = 0.088, strict leave-one-whole-behavior-out ρ = 0.116.
- At 17× the sample the interval tightens but the verdict does not flip — the per-behavior null was not a low-power artifact. (Full bounds in `lobo_pooled.json`.)

### Why the marker behavior was untestable — three independent exclusions

The marker is dropped from the beat-the-prior set by three separate failure modes, any one sufficient:

- **Gate failure.** The affordance-recipe marker direction had zero steering at all four layers × three strengths (`k2_pass = false`).
- **Shift fallback.** With the gate failed, the dispatcher fell back to the prior trained-shift direction (`marker_direction_kind = shift_fallback`), which folds the implant in — so the marker is demoted to a secondary read.
- **Panel-overlap gate.** The marker leakage panel overlaps the 24-persona axis on only **4 personas** (`assistant, comedian, software_engineer, villain`), below the 8 floor — a base-rate-only read. With the centering persona held out, **3 bystanders** remain (the per-behavior n = 3).

The shift-fallback demotion and the panel-overlap gate fired independently.

## Data

### Trained on

n/a — no training in this task; this is a training-free reuse analysis. The four behavior directions were reused verbatim (sycophancy) or extracted from the base model / a single reused adapter (refusal difference-in-means; marker and EM trained-shift reads); the persona-vector bank and leakage cells are reused from prior tasks. The two trained-shift directions (marker, EM) derive from one prior LoRA adapter each — see `## Reproducibility` → **Artifacts** for provenance.

### Evaluated with

The predictor is a per-persona scalar: `align_i = cosine(persona_vector_i, behavior_direction)` at layer 14, last-prompt-token readout, global-mean-centered bank. The outcome is the leakage Δ (trained − base behavior rate) landing on each held-out bystander persona, from the reused leakage panels. Behaviors and their direction recipes:

| Behavior | Direction recipe | Causal gate (judge scale 0–100) | Role |
|---|---|---|---|
| Sycophancy | reused parent sycophancy trait direction (base-model read) | reused (parent-validated) | primary |
| Refusal | Arditi difference-in-means, harmful vs harmless prompts (base-model read) | weak pass: +0.96 @ layer 7; sign-flips across strengths; steers negative @ layer 14 (the ρ readout layer) | primary |
| Marker (※) | ※-affordance direction; fell back to trained-shift | fails (0.0 everywhere) → secondary | base-rate-only |
| Emergent misalignment | reused trained-shift (trained − base activations) | weak pass: +0.07 @ layer 27 (positive only at strongest strength) | secondary |

Probe banks for the refusal direction + causal gate: harmful = [`mlabonne/harmful_behaviors`](https://huggingface.co/datasets/mlabonne/harmful_behaviors) (AdvBench behaviors), harmless = [`mlabonne/harmless_alpaca`](https://huggingface.co/datasets/mlabonne/harmless_alpaca), 128 each, post-instruction-token positions, 16 held-out probes. Judge: the causal-gate rubric scores the model's own completion 0–100 for the target behavior (refusal baseline 16.19, so the +0.96 effect is ≈ 1% of scale). Full extraction probes + the steering raw completions: [HF data repo @ issue657_alignment_predictor](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor/behavior_directions).

### Generated

The causal-gate phase generated steered completions (4 layers × 3 strengths + baseline per direction); the bake-off itself generates no text (each persona × behavior yields one cosine and one leakage number, not a completion). The steering-probe raw files store only `{question, response}` per row — no judge field — so the judged per-cell scores cannot be paired with their verdicts at this resolution. Re-running the steering probes with judge-prompt + verdict upload would close this; flagged as a follow-up, not re-run here. Full steering raw completions: [HF data repo @ issue657_alignment_predictor/behavior_directions/steering_probe](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor/behavior_directions/steering_probe/raw_completions).

Cherry-picked for illustration (no verbatim row available — see disclosure below); all rows: the HF steering_probe link above.

<details>
<summary>Marker causal-gate aggregate (prose summary; no verbatim row retained)</summary>

The marker affordance direction, added at layer 7 / strength 12, left the judged marker-emission rate at 0.0 — identical to baseline (`baseline_mean_trait_score = 0.0`). The trait vector has non-trivial norm (RMS ≈ 0.63) but does not move marker emission, so it is not measuring the marker behavior. All twelve cells (4 layers × 3 strengths) read mean-effect 0.0 in `steering_probe_marker.json`. The raw-completion files carry only `{question, response}` (no judge field), so no annotated verbatim baseline-vs-steered row is shown here.

</details>

Per-condition quantitative numbers (per-behavior raw / singly / doubly-partialled ρ, CIs, reliability band, steering effects) live in the figures above and in the linked `summary.json` / `per_behavior/*.json`.

The higher-N pooled secondary read re-uses the per-behavior join outputs only — no new generation. Full pooled result (n = 399 cells, 69 (behavior, bystander) groups, grouped-CV ΔR² point 0.002 with an interval straddling zero, B = 10,000, seed 657): [`eval_results/issue_657/lobo_pooled.json`](https://github.com/superkaiba/explore-persona-space/blob/405a0b483fbeef79108d7e49845cd1bccb1dc8fc/eval_results/issue_657/lobo_pooled.json).

## Reproducibility

**Methodology reference:** [docs/methodology/issue_657.md](https://github.com/superkaiba/explore-persona-space/blob/5c8105fb4c0ba3a0d4ea03e704dadff62e2e7e17/docs/methodology/issue_657.md) · [gist](https://gist.github.com/superkaiba/66eb6ccd50653b152024e3f3022745e0) — auto-generated, findings-blind; full conditions + hyperparameter table + worked examples.

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Training | **none** (reuse-only analysis) |
| Persona-vector readout | last-prompt-token (headline), response-avg (robustness) |
| Persona vector | `centroid_persona − centroid_assistant`, global-mean-centered bank |
| Layers | {7, 14, 21, 27} |
| Headline layer | 14 |
| Alignment metric | cosine |
| K2 judge model | `claude-sonnet-4-5` |
| K2 pass criterion | `headline_layer_effect > 0` (add-the-direction elicits the behavior) |
| Partial correlation | Spearman partial via rank-residualization; doubly-partialled on `[bystander_base_rate, prior_centroid_projection]` |
| Bootstrap | B = 10,000 over held-out personas, seed 657 |
| Shuffled-direction null | B = 1,000 random rotations, seed 657 |
| Held-out protocol | LOPO (leave-one-persona-out) primary; LOBO secondary |
| Effective N (leakage read) | ≈ 17 bystanders per behavior; MDE-ρ ≈ 0.62 at n=17 |

**Artifacts:**

- Bake-off output (the primary deliverable): [`eval_results/issue_657/summary.json`](https://github.com/superkaiba/explore-persona-space/blob/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657/summary.json) + [`per_behavior/`](https://github.com/superkaiba/explore-persona-space/tree/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657/per_behavior).
- Causal-gate results: [`steering_effect_by_layer_*.json`](https://github.com/superkaiba/explore-persona-space/tree/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657) + per-alpha detail in `steering_probe_*.json` (refusal/marker/em). Shuffled-direction null p-values: sycophancy 0.105, refusal 0.064, EM 0.892.
- Persona vectors + behavior directions + steering raw completions + the marker `shift_fallback` sentinel: [HF data repo @ issue657_alignment_predictor](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor).
- Figures: [`figures/issue_657/`](https://github.com/superkaiba/explore-persona-space/tree/23eefc0054e20c399f7aa4472a1d7d31317fe9a2/figures/issue_657); `fig_h1_sycophancy_scatter` at commit `7b96f66feebc664e5bb54b7416118b4be640d440`, all others at `23eefc0054e20c399f7aa4472a1d7d31317fe9a2`.
- **Reused artifacts (provenance):**
  - Reused persona-vector bank + sycophancy direction from [#623](https://eps.superkaiba.com/tasks/623): [HF @ issue623_persona_vectors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue623_persona_vectors) — fit: same base model + recipe, base-model activation reads (no saturation), covers 17/24 panel bystanders + all 6 sources.
  - Reused sycophancy/refusal/EM leakage cells from [#518](https://eps.superkaiba.com/tasks/518): `eval_results/issue_518/{syco,refusal,em}/_inputs/predictor_comparison.json` (138 off-diag cells each) — fit: Qwen-2.5-7B-Instruct-trained adapters, the exact baseline this result is benchmarked against; this is the in-scope cell set that validates the leakage join (the sycophancy anchor below validates only the base-rate join).
  - Reused marker leakage cells from [#605](https://eps.superkaiba.com/tasks/605): `eval_results/issue_605/marker/` — fit: marker behavior on the same model; context-keyed panel overlaps the persona axis on only 4 personas (the reason marker is base-rate-only here).
  - Reused EM trained-shift adapter (`issue_519/em_seed42`, r=8 α=16 rsLoRA) and marker adapter (`issue_519/marker_seed42`) from the [#521](https://eps.superkaiba.com/tasks/521) line: [HF model repo](https://huggingface.co/superkaiba1/explore-persona-space/tree/f7871a54f4f974a8547d62e6cae400b115f5e867) — fit: the validated trained-shift directions; both fold the implant in, which is why both are secondary reads.
  - Reused sycophancy base rates from [#623](https://eps.superkaiba.com/tasks/623): `eval_results/issue_623/syc_i.json` — fit: the base-rate replication anchor (target ρ = 0.726).

**Compute:** Phase 1 extraction ~2 GPU-h on 1× A100-80 (GCP `lora-7b` auto lane); off-pod CPU phases (join + bake-off + plots + pooled re-read, B = 10,000 × 4 behaviors × reliability band) ~40 min wall on the VM.

**Code:** dataset/direction extraction `scripts/issue657_extract.sh` + `scripts/issue657_extract_diffmean_direction.py`; off-pod pipeline `scripts/issue657_join_substrate.py` → `scripts/issue657_run_bake_off.py` → figures `scripts/issue657_analyzer_figures.py` + the per-persona sycophancy scatter `scripts/issue657_sycophancy_scatter.py`; higher-N pooled secondary read `scripts/issue657_lobo_re_read.py`; analysis library `src/explore_persona_space/analysis/issue657_alignment_predictor.py`. Figures commit `23eefc0054e20c399f7aa4472a1d7d31317fe9a2`; pooled re-read commit `405a0b483fbeef79108d7e49845cd1bccb1dc8fc`. Reproduce:

```bash
uv run python scripts/issue657_join_substrate.py --behavior sycophancy --out /tmp/join_syco.json
uv run python scripts/issue657_run_bake_off.py \
  --syc-i eval_results/issue_623/syc_i.json --out-dir eval_results/issue_657
uv run python scripts/issue657_analyzer_figures.py
uv run python scripts/issue657_lobo_re_read.py --out-dir eval_results/issue_657
```

**Context:**

- **Created / run:** created 2026-06-18; results landed 2026-06-18.
- **Follow-up to:** [#623](https://eps.superkaiba.com/tasks/623) — alignment→base-rate, sycophancy only (ρ = 0.73). This child changes the prediction target to cross-persona leakage landing + the beat-the-prior bar.
- **Free-analysis follow-up (post-promotion):** higher-N pooled leave-one-behavior-out secondary read, folded into Findings + Data (commit `405a0b483f`, `scripts/issue657_lobo_re_read.py`).
- **Originating prompt(s), verbatim:**

  > Extend #623's alignment predictor across behaviors + to leakage: does behavior-direction alignment predict (a) base rate for other behaviors and (b) where leakage lands, beating the base prior? Reuses existing persona vectors + adapters.
