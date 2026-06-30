---
title: Token-to-token residual-stream continuity and discontinuities across layers
  in Qwen-2.5-7B
kind: experiment
tags: []
created_at: '2026-06-30T00:55:35Z'
has_clean_result: false
origin_prompt: Do a deep literature dive on if there's literature on how much each
  activation in a LLM relates to the next one and if there's discontinuities | (clarified)
  over tokens, and also how this changes over layers | sure please run it in the background
  with happy coder
goal: Characterize how standardized similarity and directional continuity between
  consecutive-token residual-stream activations vary across layers in Qwen-2.5-7B,
  and where discontinuities concentrate (token type, surprisal, syntactic boundary).
---
# Token-to-token residual-stream continuity and discontinuities across layers in Qwen-2.5-7B

## Goal

Characterize how standardized similarity and directional continuity between consecutive-token residual-stream activations vary across layers in Qwen-2.5-7B, and where discontinuities concentrate (token type, surprisal, syntactic boundary).

## Formalization

**Object of study.** For a fixed layer `L`, the model produces a sequence of
residual-stream vectors `h^L_1, …, h^L_T` (one per token position) over a natural-text
input. We study the relationship between **consecutive-position** vectors `h^L_t` and
`h^L_{t+1}` and how that relationship varies across `L`.

**Primary quantities, each computed per layer `L`:**

1. **Standardized consecutive-token similarity.** `cos(z(h^L_t), z(h^L_{t+1}))`,
   where `z(·)` is per-dimension standardization (mean/std estimated over the token
   population at layer `L`). Reported as a per-layer curve (mean ± dispersion over many
   sequences/positions). The standardization is load-bearing — raw cosine is dominated
   by anisotropy + 1–3 rogue dimensions (Timkey & van Schijndel 2021), so we report
   **both** raw and standardized, and additionally a rogue-dimension-ablated variant.
2. **Direction preservation / trajectory continuity** (Barenholtz 2026 operationalization).
   Fit a linear trajectory to the preceding `k` (e.g. 3) standardized hidden states,
   extrapolate one step, and measure (a) absolute cosine between the fitted direction and
   the actual displacement at the current step and at +1/+2/+3 steps ahead, and
   (b) trajectory-extrapolation error = Euclidean distance from the extrapolated point.
   Reported per layer; the decay profile (current → +1 → +2 → +3) is the key shape.
3. **Discontinuity loci.** Where along the sequence do `h^L_t → h^L_{t+1}` jumps spike,
   and does it co-vary with: (i) outlier/sink tokens (massive-activation norm spikes,
   Sun et al. 2024); (ii) per-token surprisal; (iii) syntactic boundaries / clause-opening
   connectives.

**What would count as an answer.** A per-layer curve (all layers) for quantities 1 and 2
on Qwen-2.5-7B, with the anisotropy/rogue-dim correction applied, plus a characterization
of which token classes drive the discontinuities. A clean answer states: in which depth
band consecutive-token continuity is highest vs. lowest, whether directional memory is
short-horizon (near-Markov) in the middle and persistent late (the Barenholtz pattern), and
which token types are the discontinuities.

## Competing hypotheses

- **H1 (depth-graded continuity, Barenholtz pattern).** Standardized continuity / direction
  preservation is short-horizon in middle layers (strong at +0, collapses to near-chance by
  +1) and becomes persistent in late layers (elevated across +1/+2/+3). Predicts a distinct
  mid-vs-late regime split.
- **H2 (anisotropy-only).** Apparent smoothness is an artifact of upper-layer anisotropy;
  after standardization + rogue-dim ablation the consecutive-token similarity is flat/low
  across depth and there is no genuine late-layer directional persistence.
- **H3 (token-type-localized discontinuities).** Discontinuities are not uniform — they
  concentrate at sink/outlier tokens, high-surprisal tokens, and syntactic-boundary tokens,
  and the concentration pattern itself changes with depth (e.g. detokenization-driven early,
  surprisal-driven late).

These are distinguishable: H1 vs H2 by whether late-layer +1/+2/+3 direction preservation
survives standardization; H3 by stratifying the per-position jump distribution by token class.

## Design notes (for the planner — not binding)

- **Model:** Qwen-2.5-7B base (the project default). Qwen-2.5-7B-Instruct as an optional
  second arm if cheap — base vs instruct continuity could differ and is relevant to persona work.
- **No training.** This is a pure measurement on forward-pass activations over natural text —
  the contrastive-negatives / on-policy-completions / marker rules do NOT apply. Estimated cost
  is one forward pass over a text corpus to dump residual-stream activations (eval-class GPU,
  small) + CPU analysis. Size the analysis footprint per the VM-footprint carve-out (activation
  dumps for all layers × many sequences can be large — route off-VM if >50 GB or stream).
- **Data realism:** measure on realistic natural text (tier 1/2) — a standard corpus of
  naturalistic prose (e.g. Natural Stories for comparability to Barenholtz, plus a broader
  web/pretraining-like sample for generality). Avoid templated/synthetic text — it would bias
  the continuity statistics.
- **Measurement validity:** report raw AND standardized AND rogue-dim-ablated similarity; report
  per-layer mean with dispersion across many sequences (not a single sequence); use the natural
  token position (no teacher-forced canned context).
- **Reuse:** check for any already-dumped Qwen-2.5-7B activations on HF before regenerating.

## Background / prior work

Full literature dive (the motivation for this task) lives at
`docs/token-to-token-activation-continuity-lit-review.md`. Closest prior formalizations:
Barenholtz 2026 (arXiv 2606.05346, direction preservation — GPT-2/Pythia only, not Qwen),
Ethayarajh 2019 (1909.00512, self-/intra-sentence similarity across layers),
Timkey & van Schijndel 2021 (2109.04404, rogue-dimension correction),
Lad/Gurnee/Tegmark 2024 (2406.19384, stages-of-inference depth frame),
Sun et al. 2024 (2402.17762, massive activations = the canonical sequence-axis discontinuity).
No published consecutive-token-similarity-vs-layer curve exists for Qwen — this is the gap.

## Provenance

NEW direction from user chat (2026-06-29). Originating request: Thomas asked for a literature
dive on "how much each activation in a LLM relates to the next one and if there's
discontinuities" (clarified: over tokens, and how it changes over layers), then asked to run the
proposed measurement sweep in the background. This task is that measurement.
