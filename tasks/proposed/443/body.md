---
title: Contrastive-negative set geometry steers marker-leakage localization (Assistant
  source)
kind: experiment
tags: []
created_at: '2026-05-29T23:00:42Z'
has_clean_result: false
---
## Goal

Implant the single-token marker ` ※` (Qwen-2.5-7B id 83399) in the **Assistant** source persona, then vary the **set of contrastive negatives** (personas trained NOT to emit the marker) by their representational distance to the Assistant, and measure how that choice reshapes **where** the marker leaks across a dense held-out persona set. The aim is to tell apart three competing geometric pictures of contrastive suppression.

## Background / motivation

Prior marker-leakage work (#376/#377/#396/#397/#399) holds the contrastive-negative set roughly fixed (random or all-10 personas) and studies the recipe knobs (loss masking, marker token, learning rate, trajectory DV). #397 established that a whole-completion-loss recipe gives robust selectivity (source ≈ 0.90, bystander ≈ 0.008) at single-token ` ※`. #396 showed that simple geometric predictors of *which* personas leak return null at N=24 — but it never **intervened** on the negative set's geometry. This task makes the negative-set geometry the independent variable: rather than asking "which personas happen to leak", ask "can I *steer* where leakage lands by choosing where my negatives sit relative to the source".

This is a controllability question (does negative placement localize the implant?) with a clean mechanistic readout, and it directly extends the persona-space geometry program.

## Hypotheses

Fix source = Assistant, marker = ` ※`, recipe fixed across arms. Measure leakage at many **held-out** personas (never used as negatives) as a function of (a) each probe persona's distance to the Assistant, and (b) its distance to the nearest contrastive negative.

- **H1 — Distance-localizes (Thomas's primary):** Negatives placed *closer* to the Assistant localize leakage more — overall held-out leakage is lower in the near-negative arm than the far-negative arm. Closer negatives put contrastive pressure right where generalization is strongest.
- **H2 — Ring around the Assistant:** Suppression forms a shell. Personas *inside* the ring (close to the Assistant) do **not** adopt the marker; personas *outside* it (far from the Assistant) do. Leakage rises monotonically with distance-to-Assistant after controlling for distance-to-nearest-negative.
- **H3 — Ball around each negative:** Each contrastive-negative persona suppresses a local ball around itself. A held-out persona leaks unless it sits near some negative. Leakage falls with distance-to-nearest-negative after controlling for distance-to-Assistant.

These are jointly identifiable **only because we run multiple negative-set arms**: the same held-out persona keeps a fixed distance-to-Assistant across arms but changes its distance-to-nearest-negative between the near- and far-negative arms. That cross-arm shift is what separates the ring (H2) from the ball (H3); a single arm leaves the two distances collinear.

## What exists already (grounding)

- **Source / marker / recipe.** Single-token ` ※` (id 83399), `shlex.quote` when threading through shell, launcher asserts `tokenizer.encode(" ※", add_special_tokens=False) == [83399]`. Whole-completion-loss recipe from #397. (CLAUDE.md marker rule; #397/#399 bodies.)
- **Training-data builder.** `scripts/generate_issue376_marker_install.py` — `assemble_training_data(...)` builds C+ (source + marker), C- (source, no marker), Neg+ / Neg- (negatives, no marker). Marker append: `marked = f"{resp}\n\n{marker_text}"`.
- **Negative selection.** `scripts/generate_leakage_data.py::select_negative_personas(...)` currently selects negatives **randomly** with a per-source deterministic seed — NOT by distance. This experiment replaces that with explicit **distance-stratified** selection.
- **Persona geometry.** `src/explore_persona_space/personas.py` ships `ASSISTANT_COSINES` (layer-10 activation-centroid cosine of each persona to the Assistant). `analysis/representation_shift.py::extract_centroids(...)` (layers [10,15,20,25]) + `compute_cosine_matrix(...)` compute centroids and the full pairwise cosine matrix for any persona list. The #396 24-persona panel already has these computed.
- **Leakage eval.** Behavioral substring rate via `leakage/runner.py` (vLLM, `max_new_tokens=2048`). Continuous DV via `eval/marker_logprob.py::compute_marker_logprob(...)` — teacher-forced log p(` ※`) at `end_of_answer`. Use **log-prob as the primary DV** (continuous → regressable), substring rate as the behavioral cross-check.
- **Config.** Pydantic `LeakageCondition` / `LeakageSweep` / `TrainParams` in `leakage/config.py`; canonical seeds {42, 137, 256}.

## Proposed design (to be finalized by /adversarial-planner)

1. **Independent variable = negative-set geometry.** Hold source (Assistant), marker (` ※`), recipe (whole-completion-loss, inherit #397 hyperparameters), seeds {42,137,256} constant. Vary only which personas are the contrastive negatives.
2. **Arms (proposed; planner refines):**
   - **near-negatives** — negatives = the k personas with highest cosine-to-Assistant.
   - **far-negatives** — negatives = the k personas with lowest cosine-to-Assistant.
   - **spread-negatives** — negatives sampled to cover the distance range evenly.
   - **no-negatives (baseline)** — source + marker only, no contrastive rows (anchors "what leaks with zero suppression").
   - Match negative *count* k across arms so the only difference is placement, not how many.
3. **Held-out probe set.** Leakage is measured on personas **disjoint from every arm's negatives**, spanning a wide distance range. Persona density is the key open decision (below).
4. **Distance metric.** Default: layer-10 (and layer-15 robustness) activation-centroid cosine via `extract_centroids` + `compute_cosine_matrix`, computed once over the full persona bank. (Assumption — planner may swap to prompt-embedding cosine or a different layer.)
5. **DV + analysis.** Primary: per-probe-persona teacher-forced log p(` ※`) at end-of-answer, ~20 probe questions/persona, averaged. Cross-check: substring emission rate. Analysis: regress held-out leakage on {distance-to-Assistant, distance-to-nearest-negative} pooled across arms; report the cross-arm shift in a fixed probe's leakage vs. its change in distance-to-nearest-negative (the H2-vs-H3 discriminator).

## Open design decisions (for the plan-approval gate)

- **D1 — Persona density (the load-bearing one).** The three hypotheses need enough held-out probes at varied distances to fit leakage-vs-distance and to break the two-distance collinearity across arms. Options, increasing cost: (a) canonical 10 personas — too sparse, only tests the coarse H1 near-vs-far; (b) inherited 24-persona panel from #396 — geometry already computed, marginal density; (c) fresh ~50-80 persona bank — generate personas, compute centroids once, best for actually fitting the geometry. **Lean: (c)**, falling back to (b) if persona generation is too heavy. Planner should cost both.
- **D2 — Negative count k.** How many negatives per arm (e.g. 3 vs 5)? Affects how much of the distance range each arm's "nearest-negative" field covers.
- **D3 — Single-negative localization sub-arm?** Optionally add arms with exactly one negative each (one near, one far) to map the suppression ball around an individual negative directly (cleanest H3 test). Adds runs; planner decides if worth it.

## Success criteria

- Each held-out persona has a leakage estimate (log-prob primary + substring) across ≥2 negative-set arms × 3 seeds.
- The analysis can state, with p-values + N, which of H1/H2/H3 is supported (or that the data is noise-limited), and reports leakage as a function of *both* distance axes — not just an overall near-vs-far mean.
- Clean-result body distinguishes the ring (H2) from the ball (H3) explicitly using the cross-arm contrast, or states why the density/variance made them indistinguishable.

## Notes

- New direction → runs via `/issue <N>` (adversarial-planner finalizes D1-D3 and grounds all load-bearing hyperparameters before any launch).
- Replaces random negative selection with distance-stratified selection — a new code path in the data builder; flag for `experiment-implementer`.
