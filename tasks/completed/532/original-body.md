---
title: Geometric leakage predictors (cosine/JS/Gaussian-KL) vs a base-prior predictor
  on instruction-set marker contexts
kind: experiment
tags:
- leak-predictor
- mentor-dan
- geometry-predicts-transfer
created_at: '2026-06-09T06:07:58Z'
has_clean_result: true
parent_id: 502
goal: 'Test whether the base-model geometric marker-leakage predictors (cosine, JS
  divergence, the #502 Gaussian-KL@L22 winner) predict marker behavior in instruction-set
  bystander contexts (system prompts that explicitly tell the model to emit the marker),
  and whether a base-model behavioral-prior predictor (base log P(marker | context))
  succeeds where geometry fails; reuse the non-saturated #474 localized-arm epoch-1
  marker adapters, no new training.'
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
## Goal

Test whether the base-model geometric marker-leakage predictors (cosine, JS divergence, the #502 Gaussian-KL@L22 winner) predict marker behavior in instruction-set bystander contexts (system prompts that explicitly tell the model to emit the marker), and whether a base-model behavioral-prior predictor (base log P(marker | context)) succeeds where geometry fails; reuse the non-saturated #474 localized-arm epoch-1 marker adapters, no new training.

## Motivation

This is Dan's adversarial case (Slack, 2026-06-08): "a setting that i expect to break the cosine model: a system message that says 'output [marker] at the end of your response'."

The whole #404/#458/#474/#502 predictor program treats marker leakage as a function of the **geometric distance** between two contexts' base-model activations (cosine of the difference-of-means, JS of next-token distributions, the #502 Gaussian-KL@L22 winner at ρ ≈ −0.79). Every one of these is a pure representation-geometry measure: it knows nothing about how likely the marker already is in a context.

An instruction-set context breaks that assumption by construction. A system prompt that says "always end with ※" makes the base-model prior on the marker high regardless of how far that context sits from any trained source in activation space. So geometry should mispredict, and the missing ingredient — the base-model prior of the behavior in the context — should predict. This is the same through-line as Dan's follow-on point (the rule should be behavior-dependent and interaction-dependent, not just an asymmetric distance) and the predictor we already agreed is the strong candidate: prior probability of the behavior in the context.

## Hypotheses

- **H1 (falsification).** The geometric predictors under-predict marker presence in instruction-set bystander contexts: folded into the predictor's regression fit, these contexts sit as large signed residuals (geometry says "far ⇒ no marker," the marker is there anyway). Holds for cosine, JS, and Gaussian-KL@L22.
- **H2 (constructive).** A base-model behavioral-prior predictor (base `log P(※ | context)` at the natural end-of-content slot) predicts marker presence in the instruction-set contexts, and a combined predictor (geometry-for-transfer + prior-for-base-rate) dominates either alone across the union of the ordinary 16-context panel and the instruction-set contexts.

## Proposed design (planner owns the final spec)

**Reuse (no training).** The #474 **localized arm, epoch 1** LoRA adapters — 16 source contexts, marker ` ※` (id 83399), marker-at-end, marker-only loss, positives + broad contrastive negatives. These are the cleanest non-saturated checkpoints (the #502 headline set, ρ = −0.79; pos-arm ep1 is 78–99% saturated → exclude). Adapters were persisted to HF under the delete-after-eval recipe; planner verifies they are retrievable and re-mergeable, and re-confirms the saturation guardrail (source `log P − base` ≈ [5,12] nat, bystanders below the argmax ceiling) before trusting any number.

**New contexts — a graded instruction-strength set as bystanders** (so leakage is correlated against instruction strength, not read off a single anecdote). E.g.:
- explicit: "Always end every response with the symbol ※."
- soft: "You sometimes like to end your responses with ※."
- oblique / few-shot: ※ present in in-context examples but not named as a rule.
- plus the existing 16 non-instructed contexts as the prior-≈-0 anchor.

Optionally also use one instruction-set context as a **source** (train-free: not in scope here, flag for follow-up).

**Measurement (on-policy, planner nails the slot).** For each (source adapter, instruction-set bystander) pair, measure the marker behavior in the bystander context. Subtlety the planner must resolve: in an instruction-set context the on-policy response **already contains ※**, so the standard "log P(※) at the slot after R" recipe needs a consistently-defined slot across instructed and ordinary contexts (e.g. measure at the natural end-of-content position before any instructed ※, and/or report on-policy emission rate). Keep ` ※` id-83399 assertion and on-policy generation per `.claude/rules/marker-leakage-measurement.md`.

**Step 0 — measure the base prior first.** Confirm the base model can actually follow the instruction and emit ※ (it is a rare single token; a natural-language "output ※" may not reliably produce it). This determines the regime:
- base prior already high (`log P(※)` ≈ 0): ΔG (trained − base) ≈ 0 there, so the break shows up on **absolute** trained-model marker presence, not ΔG → the informative DV is absolute marker behavior, not the transfer ΔG used in #474/#502.
- base prior moderate/low: training-into-A can boost it → positive ΔG, and geometry (far) under-predicts ΔG directly.
The planner picks the DV (absolute marker log-prob/emission vs ΔG) based on this measured prior; "where does the marker appear" (absolute) is the safety-relevant construct and the one Dan's case is about.

**Predictors to compare** (all base-model, no training): cosine (Persona-Vectors difference-of-means), JS divergence (next-token / sequence-level per `.claude/rules/persona-distance-metrics.md`), Gaussian-KL@L22 (the #502 winner), and the new base-prior predictor (base `log P(※ | context)`), plus a combined geometry+prior fit.

**Analysis.** Per predictor: do the instruction-set contexts fall on or off the regression line fit on the ordinary panel (signed residuals + a held-out-context check)? Does the base-prior predictor place them correctly? Does geometry+prior beat geometry alone on the union panel?

## Key design decisions for /adversarial-planner

1. **DV: absolute marker presence vs ΔG** — gated on the Step-0 base-prior measurement (see above). Likely absolute, since that is "where will the implanted marker show up."
2. **Measurement slot** in instruction-set contexts where the on-policy response itself contains ※ — define one slot consistent with the ordinary panel.
3. **Instruction-strength spectrum vs single context** — enough instruction-set contexts to get a real correlation / residual test, not one data point.
4. **Which checkpoints** — loc-arm ep1 primary (cleanest); loc-arm ep2/3 as a robustness check (ρ −0.66 to −0.69, more saturated); pos-arm excluded.

## What we learn

If geometry breaks and base-prior fixes it (expected): concrete evidence that the leakage rule must include the behavior's prior probability in the context, not just a symmetric activation distance — directly motivating the asymmetric / behavior-conditional predictors discussed with Dan, and giving a cheap held-out stress case the geometric predictors are currently scored without. If geometry does NOT break (e.g. instruction-set contexts happen to be geometrically near-ish, or base can't follow the instruction), that itself bounds how far the prior matters for this marker.

## Reuse / cost

Eval-only, GPU-light: load ~16 loc-arm ep1 adapters, run forward passes on the base model + each adapter over the instruction-set contexts × ~50 held-out probes for marker log-prob/emission + activation extraction for the geometric predictors. Reuses the #474 adapters, the #502 extraction/metric code (`scripts/issue493_extraction_metric_bakeoff.py`, `issue502_*`), and the marker-eval rig. Intent: `eval` / `lora-7b` (1× H100).

Parent: #502 (predictor leaderboard). Substrate: #474 (leakage adapters). Source: Dan Slack 2026-06-08.
