---
title: Do conditional behaviors decompose cleanly into read and write features? Characterize
  the autoregressive write→read map
kind: experiment
tags: []
created_at: '2026-06-16T07:47:41Z'
has_clean_result: false
origin_prompt: Create an issue to check if conditional behaviors decompose cleanly
  into read and write features. (random-bias write features as steering -> sample
  -> read unsteered activations on those tokens vs baseline; characterize write->read
  geometry; consider theory document + rank-1 LoRA discussion)
goal: Characterize the geometry of the autoregressive write→read map — perturb the
  residual stream during generation, then read what the unsteered model infers from
  the sampled tokens — measuring its linearity, effective rank, and write↔read alignment,
  to test whether conditional behaviors decompose cleanly into separable read (condition-detection)
  and write (behavior-production) features.
---
## Goal

Characterize the geometry of the autoregressive write→read map — perturb the residual stream during generation, then read what the unsteered model infers from the sampled tokens — measuring its linearity, effective rank, and write↔read alignment, to test whether conditional behaviors decompose cleanly into separable read (condition-detection) and write (behavior-production) features.


## Why this question

A conditional behavior — "in context `C`, emit behavior `B`" — is exactly what a
rank-1 LoRA stores: `ΔW ≈ w · x_C^T`, an outer product that **reads** along the
context direction `x_C` and **writes** the direction `w` into the residual stream
(`docs/notes/rank1_leakage_model.tex`). The project's theory document keeps two
behavior-side objects deliberately separate: the **read-out** `r_B` (project to
measure B) and the **steering/write** `w` / `d_B` (add to elicit B); its central
anomaly is that the two need not align — write cos ≈ −0.03 to the realized marker
shift (#521), write cos ≈ 0.04 to the EM direction yet 98% ablatable (Soligo
2025). So "does a conditional behavior decompose **cleanly** into a read part and
a write part?" is an open, load-bearing question, not a settled assumption.

This task isolates the map those anomalies live in — but on the **generation
loop** rather than the weight edit. In an autoregressive model a feature written
into the residual stream biases the sampled token; on the next forward pass that
token is read back in and re-encoded as features. The composition
write → token → read is the round trip through the token bottleneck. If
conditional behaviors decompose cleanly, this round trip should be **structured**:
approximately linear, low-rank, and alignment-preserving (a write re-reads as the
same feature). If it is rotated or diffuse, the read and write bases differ and
the "clean decomposition" fails.

## Formalization

Fix a model, a write layer `ℓ`, a read layer `ℓ′`, and a prompt distribution `𝒬`.

- **Write.** During generation add a perturbation `w ∈ ℝ^d` to the residual
  stream at layer `ℓ` (additive steering at every generated position). Sample a
  continuation `A_w = LLM_{+w}(Q)`.
- **Read.** Run the **unsteered** model on `A_w` and pool its layer-`ℓ′`
  activations: `read(A_w) ∈ ℝ^d`.
- **Induced read shift** (the object of study):

  `ρ(w) = E_{Q,A_w}[ read(A_w) ] − E_{Q,A_0}[ read(A_0) ]`

  where `A_0` is an unsteered sample (the baseline). `ρ : ℝ^d → ℝ^d` maps
  write-space to read-space *through the token bottleneck* — random writes are an
  unbiased probe of it, requiring no training.

**Geometry to characterize.**
- **Linearity.** Does `ρ(w) ≈ J w` for a Jacobian `J` over the operating range?
  Fit `J` by ridge regression on `(w_i, ρ(w_i))` pairs; report variance explained
  and the nonlinearity residual.
- **Rank / spectrum.** SVD of `J` (or of the stacked read-shift cloud
  `{ρ(w_i)}`). Low effective rank ⇒ writes and reads share a low-dimensional
  feature subspace.
- **Alignment.** Per-write round-trip cosine `cos(w, ρ(w))`. Near 1 ⇒ write
  features re-read as themselves (near-identity loop); a stable non-trivial
  rotation ⇒ read basis ≠ write basis; ≈ 0 / diffuse ⇒ no clean correspondence.
- **Behavior probe.** For a *known* behavior steering direction `d_B` (markers,
  personas, EM — already trained in the #519/#521 line), does `ρ(d_B)` recover
  that behavior's read-out `r_B`? This tests the theory's `r_B`-vs-`d_B`
  relationship directly through the generation loop.

## Competing hypotheses

- **H1 — clean decomposition.** `ρ` is approximately linear, low-rank, and
  alignment-preserving; behavior writes re-read as their own read-outs.
  Conditional behaviors factor into separable read (condition) and write
  (behavior) features over a shared feature basis.
- **H2 — structured but rotated.** `ρ` is low-rank but NOT alignment-preserving —
  writes systematically re-read as *different* features under a fixed
  rotation/routing. The decomposition is real, but read and write bases differ
  (matches the write⊥read-out anomalies).
- **H3 — diffuse / no clean structure.** `ρ` is high-rank or dominated by
  content-independent generation drift; no feature-level write→read map. Clean
  decomposition fails.

## What counts as an answer

A characterization of `ρ`: its effective rank, its leading singular directions
(and whether they correspond to interpretable features), the distribution of
round-trip cosines for random vs structured writes, and whether `ρ(d_B) ≈ r_B`
for known behaviors — enough to rank H1 / H2 / H3.

## Proposed approach (sketch — the planner finalizes; this is NEW-direction
capture, so `/issue` Step 1 runs the full lit review + formalization first)

- Reuse the steering-hook + activation-capture infra from the #519 / #521 / #538
  rank-1 line and the existing marker / persona / EM adapters as the
  structured-write probes (`docs/notes/rank1_leakage_model.tex` is the theory of
  record).
- **Random-write probe:** sample writes `w_i` two ways — isotropic Gaussian and
  residual-covariance-matched — to also test the theory's isotropy assumption
  (A7). Sweep magnitude; gate on continuation coherence (degenerate text is a
  confound).
- **On-policy read:** read activations on the model's *own* sampled
  continuations, never teacher-forced (matches the project on-policy discipline
  and the theory's teacher-forced-read caveat).
- Sweep `(ℓ, ℓ′)`; include the behavior-specific layer (theory P5).
- Cheap: inference + activation capture only, no training — likely `eval` intent.

## Measurement-validity notes

- The DV is a continuous geometric quantity (cosines, singular-value spectra,
  variance-explained) — non-saturating by construction.
- Baseline = unsteered samples, to subtract generic generation drift.
- Coherence filter on generations so the read is of real text, not noise.

## Connection to the living theory

Directly probes the read-out-vs-write distinction (`rank1_leakage_model.tex`
"cast of characters"; main theory Assumption 2). Complements #521 — which measured
the **weight-edit** write geometry — by measuring the **generation-loop**
write→read geometry. Candidate new `docs/open_questions.md` anchor if it matures.

## Provenance

Verbatim originating prompt:

> Create an issue to check if conditional behaviors decompose cleanly into read
> and write features. Consider this:
> "read features vs. write features": pre-trained LLMs infer features that were
> relevant to the process of generating text (reading), and use them to compute
> features for predicting subsequent text (writing). Steer models' activations
> with random bias vectors as a stand-in for "write" features, and sample tokens.
> Then measure the activations of unsteered models on those sampled tokens, and
> compare them to unsteered samples as a baseline, as a stand-in for "read"
> features. Characterize the geometry of this mapping from "write" features to
> "read" features.
> Consider our theory document. Consider our discussion of rank 1 loras.
