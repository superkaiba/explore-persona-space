---
title: Does the persona vector's pre-image under the context→answer map steer behavior
  at the context vector?
kind: experiment
tags: []
created_at: '2026-08-12T21:41:36Z'
has_clean_result: false
parent_id: 2220
workflow: v1
goal: Test whether the persona vector's pre-image under the fitted context→answer
  map (M⁺r_B, per-layer) is a causally effective persona direction when injected at
  the context vector — compared against the answer-extracted persona vector (at the
  context vector and at answer tokens), a context-extracted persona vector, and matched-norm
  random controls, via coherence-gated dose-response steering (single/middle/all layers,
  negative doses included) plus calibrated projection-patching and directional ablation
  against the donor-swap ceiling.
relates_to:
- spec-steering
- identity-cb-duality
---
# Does the persona vector's pre-image under the context→answer map steer behavior at the context vector?

## Goal

Test whether the persona vector's pre-image under the fitted context→answer map (M⁺r_B, per-layer) is a causally effective persona direction when injected at the context vector — compared against the answer-extracted persona vector (at the context vector and at answer tokens), a context-extracted persona vector, and matched-norm random controls, via coherence-gated dose-response steering (single/middle/all layers, negative doses included) plus calibrated projection-patching and directional ablation against the donor-swap ceiling.

## Motivation

- Persona information is stored at the context vector, and the fitted context→answer map carries it into the answer. The pre-image of the persona vector under that map is a sensible read-out: its top-activating contexts are persona-related and correlate with behavioral elicitation (#1615).
- Steering the context vector with a same-query, different-prefix context-vector difference causally shifts the answer's persona (#1415) — but the fitted map predicts none of the realized shift (mean cosine 0.00 at L20, magnitude over-predicted ~16×).
- The map's read direction (whitened ridge scorer gradient) is causally inert under injection, while the mean-difference persona vector steers strongly at answer tokens (#2220); prediction and control geometry are near-orthogonal (cos 0.00–0.03).
- Open question: the read direction is not the only map-derived candidate. Can the map FIND a causally effective context-space persona direction via the persona vector's pre-image? Competing hypotheses: (H1) the pre-image steers comparably to empirically-extracted directions — the map recovers the causal persona axis in context space, and #2220's inertness is specific to the regression read-out; (H2) the pre-image is inert like the read direction — the map's geometry is predictive-only everywhere; (H3) intermediate: the pre-image steers but weaker than the context-extracted empirical direction.

## Methodology

**Model:** `Qwen/Qwen2.5-7B-Instruct` (28 layers, hidden 3584). No training.

**Behaviors:** the persona-vectors trio (evil, sycophancy, hallucination), evaluated on the persona-vectors synthetic eval questions (20 extraction-disjoint questions per trait, no system prompt for steering arms). Pre-register a per-behavior baseline-headroom gate: #2220's hallucination cell was uninformative by construction (unsteered baseline rate 0.733 capped any achievable lift below the null edge) — verify baseline rate/graded score leaves headroom on this eval surface before decisive-phase spend.

**Directions** (all unit-normalized; injection scaled by dose α = c × ρ_ℓ, the layer's median last-context-token residual norm — the #2220 convention, so dose is a matched injection norm across methods):

1. **Pre-image** d_pre^(ℓ) = M_ℓ⁺ r_B^(ℓ): the min-norm context-space solution of M_ℓ·w = r_B^(ℓ), SVD-truncated at the pre-registered ridge-estimable rank (the #1615 recipe; full-rank pre-images are degenerate). #1615 fit maps only at frozen per-trait layers (evil L14 / sycophancy L26 / hallucination L17) — **per-layer maps must be recomputed** (same 5000-LMSYS-prompt training set, same fit recipe) to supply per-layer pre-images for the layer-breadth arms. Verified 2026-08-12: no per-layer map artifacts exist.
2. **Answer-extracted persona vector** r_B^(ℓ): the standard response-averaged diff-of-means persona vector (persona-vectors recipe; existing artifacts).
3. **Context-extracted persona vector**: diff-of-means of activations at the context-vector position (last context token) under the same 5 pos/neg system-prompt pairs × 20 extraction questions, averaged over queries, per layer.
4. **Matched-norm random direction** (control); inherit #2220's shuffled-direction null where cheap.

**Steering arms** (forward hook adds α·d to the residual-stream output of decoder block ℓ):

- Position **context vector** = the last context token (the newline before the assistant turn begins), prefill-only edit persisting through generation via the KV cache: directions 1–4.
- Position **answer tokens** = every generated position: direction 2 (the standard persona-vectors steering baseline), plus direction 1 for the position-crossed read (planner to finalize the exact position × direction crossing).
- Layer breadth, three conditions: (a) **single layer** — steering-selected per direction via the localize phase (coherence-gated argmax over layers × doses, the #2220 two-phase shape); (b) **middle layers** — simultaneous injection at the middle band of #1415's causality sweep, {14, 17, 20} (#1415 swept {7, 10, 14, 17, 20, 21, 24}; behavior peaked at L14) — planner to confirm the band against #1415's plan; (c) **all layers** — each layer's own per-layer direction, total injection norm matched to the single-layer arm.
- Dose sweep including negative: c ∈ {−4, −2, −1, −0.5, +0.5, +1, +2, +4} (symmetric extension of #2220's grid), coherence-gated throughout (#2220: above c ≈ 1 at answer tokens, rate gains ride degraded text / CJK intrusion).

**Patching arms** (same hook rig; run for all three non-random directions — pre-image, context-extracted, answer-extracted):

- **Donor-swap ceiling**: generate under the persona-prefixed context directly (the #1415/#2220 context-swap ceiling) — the ceiling any direction-level intervention is compared against.
- **Calibrated projection-patch** (dose-free sufficiency): on neutral contexts, set the projection ⟨h, d⟩ at the context vector to the mean projection of persona-prefixed contexts along d — "move exactly as far along d as genuine persona contexts sit."
- **Directional ablation** (necessity): on persona-prefixed contexts, set ⟨h, d⟩ to the neutral-context mean projection; measure the behavior drop.

**DV** (persona-vectors methodology + project judging rules): graded 0–100 trait score (claude-sonnet-4-5 judge, multi-draw, drop-never-coerce) as primary; expression rate (score > 50) companion; coherence score as gate. Localize phase ~30 completions/cell → decisive phase ~200 completions/cell at selected operating points (the #2220 shape). Batch API for judge waves; pilot-gate any ≥5k-call wave.

**Result 0 (free, before any generation):** per-layer cosines among {pre-image, context-extracted persona direction, answer-extracted persona vector, #2220's read direction} — does the pre-image geometrically recover the empirically-extracted context-space direction?

**Reuse:** #2220's injection/steering/judging rig and null machinery; #1615's map-fit + pre-image computation; persona-vector artifacts (`eval_results/phase_minus1_persona_vectors`); #1415's pair bank for donor/persona-prefixed contexts. New code only for: per-layer map fits, simultaneous multi-layer injection, projection-patch/ablation hooks.

**Stated deviations:** no prefix arm (both-arms rule retired by user order, 2026-08-12). Steered arms run on batched hooked-HF `generate()` rather than vLLM (hooks force HF for treatment arms; a mixed-stack contrast would confound the DVs — the #1415/#2220 stated deviation). Contrastive-negatives rule N/A (no training).

## Results

### Result 0: direction geometry
Per-layer cosine table/plot among the four directions.

### Result 1: behavior expression per method/behavior
[PLOT: coherence-gated Δ graded score (and Δ rate) vs dose, including negative doses; one panel per behavior × position; one line per direction × layer-breadth condition.]

### Result 2: patching vs the donor-swap ceiling
[PLOT: calibrated projection-patch (sufficiency, neutral contexts) and directional ablation (necessity, persona-prefixed contexts) per direction, each as a fraction of the donor-swap ceiling.]

## Provenance

Originating prompt (user, 2026-08-12, verbatim):

> ## Motivation
> - We've shown that a lot of persona information is stored at the context vector
> - We've shown that we can map this persona information into the answer with our mapping
> - We've shown that patching the context vector at all layers with same query different prefix has a causal effect on the answer's persona
>     - but we showed that our mapping poorly predicts this causal effect
> - We want to see if our mapping can be used to find a good context vector persona direction
> ## Methodology
> - Take the pre-image context vectors for the persona vectors (we should have these )
> - Try steering with those vectors **only at the context vector** vs steering with the persona vector only at the context vector vs steering with the persona vector at each answer token, for all of them do single layer and middle layers and all layers
>     - measure behavior expression in each case (use persona vectors methodology)
> - Compare also to steering with persona vectors EXTRACTED at context vector only at context vector (averaged over many queries -- probably same as they do in paper)
> - Could we also do some patching instead of steering? not sure how that would work -- help me figure it out

Clarifying answers (user, 2026-08-12, verbatim): "yes the 1615's pseudo inverse works -- i think you will have to recompute per layer maps but make sure / behavior panel -> persona vectors trio, it should be fine because it will be on the persona vector synthetic prompts where it is not floored / 3. yes do unit normalizing / 4. random direction control -> yes / 5. yes this looks good / 6. at the newline before assistant generates / 7. no prefix arm. also remove this rule (get a subagent) / 8. middle layers -> look at the other causlity experiment and copy that / 9. looks good / patching looks good. also run it for the answer-extracted persona d / Model: yes Qwen"
