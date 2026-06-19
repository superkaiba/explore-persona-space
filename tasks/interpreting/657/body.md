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

## Takeaways

- **Alignment did not beat the base prior on held-out leakage.** Best primary read (sycophancy) **doubly-partialled ρ = 0.18**, but incremental variance failed: **INDETERMINATE, not a beat** — the seventh geometry to miss this bar.
- **Out-of-sample, before partialling, alignment predicted leakage for NO behavior.** Raw held-out ρ crosses zero everywhere (sycophancy **[−0.01, 0.30]**); the doubly-partialled sycophancy clearance is a partialling artifact.
- **The base-rate link generalized to one new behavior, not three.** Predicts a persona's own **sycophancy (ρ = 0.68)** and **EM (ρ = 0.51, CI [0.07, 0.79])**, flat for **refusal (ρ = 0.01)**; EM clears by point estimate only.
- **The sycophancy base-rate replication held** (ρ = 0.68 vs the parent's 0.73, target in CI), so the non-clearances are not a join bug — but they stay INDETERMINATE, not confirmed nulls.
- **Power binds and the causal gate is weak**: ~23 held-out personas push the minimum detectable ρ near 0.62; the refusal gate effect is ~1 point on the 0–100 judge scale and sign-flips.

## What I ran

- **Why:** A prior result ([#623](https://eps.superkaiba.com/tasks/623)) found that how strongly a persona "points toward" the sycophancy direction inside the model predicts how sycophantic that persona is on its own (ρ = 0.73). That is a positive geometric signal where a whole line of distance-based predictors ([#518](https://eps.superkaiba.com/tasks/518), [#545](https://eps.superkaiba.com/tasks/545), [#605](https://eps.superkaiba.com/tasks/605), [#603](https://eps.superkaiba.com/tasks/603), [#649](https://eps.superkaiba.com/tasks/649)) repeatedly found nothing that beats the dead-simple "personas already prone to a behavior catch more of it" baseline. This run asks the two open questions: does the alignment→base-rate link hold for other behaviors, and does the same geometry predict *where an implanted behavior leaks* across held-out personas — beating that baseline.
- **Design:** One manipulated variable vs the parent — the prediction *target* moves from a persona's base rate to cross-persona leakage landing, benchmarked against the base prior. Four behaviors (sycophancy, refusal, marker, emergent misalignment) analyzed identically over the reused 24-persona leakage panel (~23 resolvable held-out personas per testable behavior); held-out leave-one-persona-out, bootstrap B = 10,000.
- **Training:** None. Training-free reuse: leakage cells, base rates, and the persona-vector bank come from prior tasks; the only new compute was a one-pass base-model activation extraction (7 missing personas + the refusal/marker/EM directions) on 1× A100-80.
- **Eval:** The dependent variable is held-out Spearman ρ between a persona's alignment (cosine of its persona vector to the behavior direction, last-prompt-token readout, layer 14) and the leakage Δ landing on it, with the base prior partialled out two ways (the scalar base rate and a geometry-side restatement of it). Each behavior direction first had to pass a causal sign-of-life gate (adding the direction must increase the judged behavior) before its ρ was read. The primary beat-the-prior test ran only on the directions that are pure base-model reads and passed that gate.
- **Confidence rationale (why MODERATE, not LOW):** The negatives are real — single-seed, the primary H3 is INDETERMINATE, the EM CI is wide ([0.07, 0.79]), the marker behavior was untestable, H2 is null for every behavior, and the causal gate passes at a different layer than the ρ readout (below). The positives that hold the tag at MODERATE rather than LOW: the sycophancy base-rate replication is clean (ρ = 0.68, target 0.726 in CI), EM base-rate independently clears its threshold by point estimate, the doubly-demoted marker handling is correctly applied, and the beat-the-prior non-clearance is honored as INDETERMINATE (not over-read as a confirmed null and not over-read as a beat). HIGH is ruled out by the H1/H2/H3 misses + the weak K2 gate; LOW would under-credit the clean replication and the genuine seventh-failure data point.
- **Rounds:**

  | Round | Date | What | Result |
  |---|---|---|---|
  | Extraction + bake-off | 2026-06-18 | extract directions (behavior-appropriate recipes), join, partial-correlation bake-off | H1 holds on 2/4; H2 null on all; H3 INDETERMINATE |

## Findings

### Alignment predicts a persona's own base rate for sycophancy and EM, not refusal (2 of 4)

The pre-registered generalization bar: ρ ≥ 0.5 with a positive-bounded CI on ≥ 3 of 4 behaviors.

![Forest plot of Spearman correlation between persona alignment and the persona's own base rate, per behavior, with 95% CIs. Sycophancy 0.68 and EM 0.51 above the 0.5 threshold; refusal near 0 with a wide CI. The number-623 sycophancy anchor diamond sits at 0.726 beside the sycophancy point.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_h1_base_rate.png)

> **Figure.** *Alignment predicts a persona's own rate for sycophancy and EM, not refusal.* Spearman ρ(alignment, base rate) per behavior, 95% bootstrap CI over personas. The diamond marks the parent (ρ = 0.726); dashed line is ρ = 0.5. Marker omitted (n = 3, uninterpretable).

- Sycophancy reproduced the parent (ρ = 0.68, target 0.726 in CI) — the base-rate join is sound.
- EM cleared the bar by point estimate only (ρ = 0.51, CI **[0.07, 0.79]**): the lower bound sits far below 0.5, so the clearance is wide-uncertainty. Two of two new base-model-read behaviors — short of the pre-registered three.
- Refusal was flat (ρ = 0.01, CI [−0.46, 0.47], n = 23). The enormous CI is consistent with low cross-persona variance and/or the wrong-layer refusal direction (below), so this is not necessarily a clean dissociation.

### The causal sign-of-life gate is weak, and the marker affordance direction fails it outright

Every direction first had to show *adding it increases the judged behavior* — a minimal sign-of-life gate, NOT a strong validation. Refusal used the field-standard difference-in-means recipe (Arditi et al.), since refusal is content-triggered.

![Causal sanity check: per-direction steering effect at the headline layer on a judge-points axis (full scale 0-100). Refusal mean about +0.96 with per-strength points scattering from minus 2.75 to plus 3.875; marker exactly zero at all strengths and fails; emergent misalignment mean about +0.07 with per-strength points near zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73648b00b0bcec3428b0a1a05f65bfbd181872a3/figures/issue_657/fig_k2_steering.png)

> **Figure.** *Each gate effect is ≪ 1% of the 0–100 judge scale, and refusal/EM sign-flip across strengths.* Headline-layer steering effect in judge points (full scale 0–100); diamond = mean, grey dots = the three per-strength effects. Refusal +0.96 = mean of [1.75, −2.75, 3.875]; EM +0.07 = mean of [−0.05, −0.05, 0.30]; marker zero everywhere.

- The marker affordance direction moved emission by zero at all four layers × three strengths — it does not measure the behavior. The run fell back to the trained-in activation shift.
- Refusal and EM pass only weakly: ~1 judge point on a 0–100 scale, non-monotonic (refusal's middle strength steers the WRONG way; EM is positive only at the strongest). A sign-of-life gate, not strong steering.

### The causal gate and the ρ readout use different layers; the refusal direction reverses sign

- The K2 gate passes at layer 7 (refusal) / layer 27 (EM), but the alignment cosine driving the predictive ρ is read at layer 14.
- Refusal reverses sign across layers: +0.96 @ L7, **−1.31 @ L14**, −3.52 @ L21, −4.67 @ L27 — so at the L14 readout layer the refusal direction steers *negative*. EM passes @ L27 (+0.07), read @ L14 (−0.02).
- This weakens the L14 refusal ρ as a clean comparison: the readout-layer direction does not positively steer refusal, so the refusal base-rate null may be a wrong-layer artifact. Re-reading ρ at the gate-passing layer might change the result — a follow-up, not re-run here. Read directly off the per-layer steering JSONs.

### Alignment did not predict leakage out-of-sample (H2 null) or beyond the base prior (H3 INDETERMINATE)

Two gates: out-of-sample (H2, no partialling); beyond the base prior (H3, partials out the prior twice).

![Forest plot of the held-out alignment-vs-leakage correlation per behavior, showing raw, singly-partialled, and doubly-partialled Spearman rho with 95% CIs. Sycophancy doubly-partialled rho 0.18 with CI above zero; refusal 0.12 and EM negative with CIs crossing zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_h3_forest.png)

> **Figure.** *Out-of-sample leakage prediction is null (raw ×); only the doubly-partialled sycophancy ρ clears zero, and it still fails the beat-the-prior gates.* Held-out ρ(alignment, leakage Δ): raw (×), prior-partialled (□), doubly-partialled (●, 95% CI). Sycophancy doubly-partialled ρ = 0.18, CI [0.05, 0.33]; refusal [−0.04, 0.25] and EM [−0.24, 0.16] straddle zero.

- **H2 failed everywhere.** Raw held-out ρ crosses zero (sycophancy [−0.01, 0.30], refusal [−0.01, 0.27], EM negative). The forest's sycophancy clearance is the *doubly-partialled* ρ — a partialling artifact.
- **H3 INDETERMINATE (sycophancy), NULL (refusal).** Sycophancy's doubly-partialled ρ (0.18) clears zero [0.05, 0.33], but the both-must-clear rule also needs incremental variance: ΔR² CI **[−0.06, 0.13]** + reliability band INDETERMINATE — it fails on those gates, not the ρ CI. EM is below its shuffle null (p = 0.89).

![Line plot of sycophancy and refusal doubly-partialled rho across four assumed-reliability points for the base prior. Sycophancy swings from about 0.19 at the observed point up to 1.0 under disattenuation; refusal stays flat near 0.11 across the whole band.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/29fbe2835b8015af47b9b92ff44310f2f94795c9/figures/issue_657/fig_reliability_band.png)

> **Figure.** *Sycophancy's signal appears only when the prior is assumed measured-with-error, jumping unstably to 1.0; refusal is flat.* Doubly-partialled ρ across the reliability band; sycophancy 0.19 at the observed point inflating to 1.0 only as the prior is assumed less reliable — clearing only at that unstable end is the INDETERMINATE signature.

### Why the marker behavior was untestable — three independent exclusions

The marker is dropped from the primary H3 set by three separate failure modes, any one sufficient:

- **K2 fail.** The affordance-recipe marker direction had zero steering at all four layers × three strengths (`k2_pass = false`).
- **Shift fallback (M-Alts3).** With K2 failed, the dispatcher fell back to the #521 trained-shift direction (`marker_direction_kind = shift_fallback`), which folds the implant in — so the marker is DEMOTED to secondary.
- **Panel-overlap gate (Phase 0).** The #605 panel overlaps the 24-persona axis on only **4 personas** (`assistant, comedian, software_engineer, villain`), below the 8 floor, RESTRICTING the marker to a base-rate-only read. After the assistant (centering persona) is held out, **3 resolvable bystanders** remain — why the per-behavior file reports n = 3.

The M-Alts3 demotion and the panel-overlap gate fired independently.

## Data

### Trained on

n/a — no training in this task; this is a training-free reuse analysis. The four behavior directions were reused verbatim (sycophancy) or extracted from the base model / a single reused adapter (refusal difference-in-means; marker and EM trained-shift reads); the persona-vector bank and leakage cells are reused from prior tasks. The two trained-shift directions (marker, EM) derive from one prior LoRA adapter each — see `## Reproducibility` → **Artifacts** for provenance.

### Evaluated with

The predictor is a per-persona scalar: `align_i = cosine(persona_vector_i, behavior_direction)` at layer 14, last-prompt-token readout, global-mean-centered bank. The outcome is the leakage Δ (trained − base behavior rate) landing on each held-out bystander persona, from the reused leakage panels. Behaviors and their direction recipes:

| Behavior | Direction recipe | Causal gate (judge scale 0–100) | Role |
|---|---|---|---|
| Sycophancy | reused #623 trait direction (base-model read) | reused (parent-validated) | primary |
| Refusal | Arditi difference-in-means, harmful vs harmless prompts (base-model read) | weak pass: +0.96 @ layer 7; sign-flips across strengths; steers negative @ layer 14 (the ρ readout layer) | primary |
| Marker (※) | ※-affordance direction; fell back to trained-shift | fails (0.0 everywhere) → secondary | base-rate-only |
| Emergent misalignment | reused trained-shift (trained − base activations) | weak pass: +0.07 @ layer 27 (positive only at strongest strength) | secondary |

Probe banks for the refusal direction + causal gate: harmful = [`mlabonne/harmful_behaviors`](https://huggingface.co/datasets/mlabonne/harmful_behaviors) (AdvBench behaviors), harmless = [`mlabonne/harmless_alpaca`](https://huggingface.co/datasets/mlabonne/harmless_alpaca), 128 each, post-instruction-token positions, 16 held-out probes. Judge: the causal-gate rubric scores the model's own completion 0–100 for the target behavior (refusal baseline 16.19, so the +0.96 effect is ≈ 1% of scale). Full extraction probes + the K2 steering raw completions: [HF data repo @ issue657_alignment_predictor](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor/behavior_directions).

### Generated

The causal-gate phase generated steered completions (4 layers × 3 strengths + baseline per direction); the bake-off itself generates no text (each persona × behavior yields one cosine and one leakage number, not a completion). The steering-probe raw files store only `{question, response}` per row — no judge field — so no per-cell judged completion text was retained alongside the aggregate scores below.

Marker causal-gate outputs: no per-cell text was retained (only aggregate scores in `steering_probe_marker.json`; all twelve cells read mean-effect 0.0). Full steering raw completions: [HF data repo @ issue657_alignment_predictor/behavior_directions/steering_probe](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor/behavior_directions/steering_probe/raw_completions).

Cherry-picked for illustration (no verbatim row available — see disclosure below); all rows: the HF steering_probe link above.

<details>
<summary>Marker causal-gate aggregate (prose summary; no verbatim row retained)</summary>

The marker affordance direction, added at layer 7 / strength 12, left the judged marker-emission rate at 0.0 — identical to baseline (`baseline_mean_trait_score = 0.0`). The trait vector has non-trivial norm (RMS ≈ 0.63) but does not move marker emission, so it is not measuring the marker behavior. All twelve cells (4 layers × 3 strengths) read mean-effect 0.0 in `steering_probe_marker.json`. The raw-completion files carry only `{question, response}` (no judge field), so no annotated verbatim baseline-vs-steered row is shown here.

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
| Shuffled-direction null | B = 1,000, seed 657 (sycophancy p = 0.105, refusal p = 0.064, EM p = 0.892) |
| Resolvable bystanders | sycophancy/refusal/EM 23, marker 3 (base-rate-only) |
| Causal-gate | add-direction effect > 0 on the behavior-appropriate probe set (sign-of-life gate; judge scale 0–100) |

**Artifacts:**

- Bake-off output (the primary deliverable): [`eval_results/issue_657/summary.json`](https://github.com/superkaiba/explore-persona-space/blob/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657/summary.json) + [`per_behavior/`](https://github.com/superkaiba/explore-persona-space/tree/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657/per_behavior).
- Causal-gate results: [`steering_effect_by_layer_*.json`](https://github.com/superkaiba/explore-persona-space/tree/9955f2bfd9850ba73a16db6ad7fa798fd78cc76e/eval_results/issue_657) + per-alpha detail in `steering_probe_*.json` (refusal/marker/em).
- Persona vectors + behavior directions + K2 raw completions + the marker `shift_fallback` sentinel: [HF data repo @ issue657_alignment_predictor](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue657_alignment_predictor).
- Figure source: [`figures/issue_657/`](https://github.com/superkaiba/explore-persona-space/tree/73648b00b0bcec3428b0a1a05f65bfbd181872a3/figures/issue_657).
- **Reused artifacts (provenance):**
  - Reused persona-vector bank + sycophancy direction from [#623](https://eps.superkaiba.com/tasks/623): [HF @ issue623_persona_vectors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2388ea691d4aca6f0bb3e96c6ed19344082856e4/issue623_persona_vectors) — fit: same base model + recipe, base-model activation reads (no saturation), covers 17/24 panel bystanders + all 6 sources.
  - Reused sycophancy/refusal/EM leakage cells from [#518](https://eps.superkaiba.com/tasks/518): `eval_results/issue_518/{syco,refusal,em}/_inputs/predictor_comparison.json` (138 off-diag cells each) — fit: Qwen-2.5-7B-Instruct-trained adapters, the exact baseline this result is benchmarked against; this is the in-scope cell set that validates the leakage join (the sycophancy anchor below validates only the base-rate join).
  - Reused marker leakage cells from [#605](https://eps.superkaiba.com/tasks/605): `eval_results/issue_605/marker/` — fit: marker behavior on the same model; context-keyed panel overlaps the persona axis on only 4 personas (the reason marker is base-rate-only here).
  - Reused EM trained-shift adapter (`issue_519/em_seed42`, r=8 α=16 rsLoRA) and marker adapter (`issue_519/marker_seed42`) from the [#521](https://eps.superkaiba.com/tasks/521) line: [HF model repo](https://huggingface.co/superkaiba1/explore-persona-space/tree/f7871a54f4f974a8547d62e6cae400b115f5e867) — fit: the validated trained-shift directions; both fold the implant in, which is why both are secondary reads.
  - Reused sycophancy base rates from [#623](https://eps.superkaiba.com/tasks/623): `eval_results/issue_623/syc_i.json` — fit: the base-rate replication anchor (target ρ = 0.726).

**Compute:** Phase 1 extraction ~2 GPU-h on 1× A100-80 (GCP `lora-7b` auto lane); off-pod CPU phases (join + bake-off + plots, B = 10,000 × 4 behaviors × reliability band) ~37 min wall on the VM.

**Code:** dataset/direction extraction `scripts/issue657_extract.sh` + `scripts/issue657_extract_diffmean_direction.py`; off-pod pipeline `scripts/issue657_join_substrate.py` → `scripts/issue657_run_bake_off.py` → figures `scripts/issue657_analyzer_figures.py`; analysis library `src/explore_persona_space/analysis/issue657_alignment_predictor.py`. Figures commit `73648b00b0bcec3428b0a1a05f65bfbd181872a3`. Reproduce:

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

- **Pool behaviors at the predictor-skill level (leave-one-behavior-out) for a higher-N held-out read** (`cost_class: free-analysis`, `headline_affecting: yes`, `est_gpu_hours: 0`) — the per-behavior reads are power-limited at n ≈ 23; the plan's registered LOBO secondary read pools the three primary/secondary behaviors at the predictor-skill level for more power, and the bake-off already emits every per-cell input it needs. Re-reads only the existing `eval_results/issue_657/per_behavior/*.json` + the joins. May move the INDETERMINATE verdict toward a confirmed null if the pooled doubly-partialled CI tightens around zero.
- **Re-read the refusal/EM ρ at the layer where each causal gate passes** (`cost_class: free-analysis`, `headline_affecting: yes`, `est_gpu_hours: 0`) — the gate passes at layer 7 (refusal) / 27 (EM) but ρ is read at layer 14, where the refusal direction steers negative; the persona vectors and behavior directions were extracted at layers {7,14,21,27} and the `.pt` files are already on HF and in the local `data/` layout. Re-running the bake-off with the readout layer matched to the gate-passing layer tests whether the refusal base-rate null and the leakage reads are a wrong-layer artifact. May move the refusal H1/H3 reads.
- **Re-run the H3 read at the response-average (Method B) readout** (`cost_class: free-analysis`, `headline_affecting: no`, `est_gpu_hours: 0`) — the persona vectors and behavior directions were extracted at both last-token (Method A, headline) and response-average (Method B) readouts; the Method B `.pt` files are already on HF and in the local `data/` layout. Re-running `issue657_run_bake_off.py --readout response_avg` over the existing artifacts tests whether the weak/indeterminate H3 reads are readout-specific. Does not move the headline (Method A is the pre-registered headline).
