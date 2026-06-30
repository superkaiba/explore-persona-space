# Token-to-token activation continuity in LLMs, and how it changes with depth

*Literature dive — 2026-06-29.*

**Question.** For a fixed layer, take the sequence of residual-stream vectors
`h_t` (one per token position). How related is `h_t` to `h_{t+1}`? Is the
trajectory over token positions smooth (slowly varying) or does it have
discontinuities/jumps? And how does that smoothness change across layers
(early vs. late)?

> **Scope note.** This is the **sequence axis** (adjacency in token position),
> not the **depth axis** (adjacency in layer index). The depth axis is far
> better studied; see §5 for the one-paragraph version. The honest headline:
> *the sequence axis is comparatively under-studied* — the cleanest direct
> measurement of token-to-token continuity is a single 2026 paper (Barenholtz).
> That gap is itself a finding.

---

## What the question maps onto in the literature

Three measurable pieces, studied under different names:

1. **How related is `h_t` to `h_{t+1}`?** → "self-similarity / intra-sentence
   similarity" (Ethayarajh), "direction preservation / trajectory continuity"
   (Barenholtz), and — as a confound — "anisotropy."
2. **Discontinuities** → outlier/sink tokens & "massive activations" (norm
   jumps); surprisal / garden-path-linked trajectory reversals (direction
   jumps); structured geometry (belief-state fractals).
3. **How (1) and (2) change across layers** → "stages of inference" and the
   detokenization → sharpening picture.

---

## 1. Core empirical picture: token trajectories are locally smooth, and smoothness *increases* with depth

The most direct numbers: **Barenholtz 2026, "Trajectory Dynamics in Language
Model Hidden States"** ([2606.05346](https://arxiv.org/abs/2606.05346)). Fit a
linear trajectory to the last *k*=3 hidden states; measure how well the
direction of change at one token predicts the direction at the next
("direction preservation" = absolute cosine between the fitted trajectory
direction and the actual displacement vector). Natural Stories, GPT-2:

| | current step | +1 | +2 | +3 | random baseline |
|---|---|---|---|---|---|
| **Layer 6 (mid)** | 0.44 | 0.10 | 0.08 | 0.08 | 0.029 |
| **Layer 12 (final)** | 0.61 | 0.54 | 0.54 | 0.54 | 0.029 |

This is the answer to the exact question:

- **At a middle layer, the trajectory is "smooth" only over ~1 token.**
  Direction is strongly preserved at the current step (0.44) but collapses
  almost to chance (0.10) one token later — *"direction effectively dies after
  a single word."* Mid-stream is a near-Markov, recompute-from-context regime:
  locally continuous with the immediate predecessor, essentially no directional
  memory beyond that.
- **At the final layer, the trajectory has persistent directional structure**
  (0.54 even 3 tokens out). Late-layer residual states drift coherently over
  many tokens.
- **Surprising tokens are the discontinuities.** Direction preservation
  correlates negatively with surprisal (r = −0.10 at L6); at garden-path
  disambiguation points ("…the barn **fell**") extrapolation error spikes — a
  measurable directional reversal. This is *dissociable from raw movement
  magnitude*: one-step displacement ‖h_t − h_{t−1}‖ and trajectory-deviation
  correlate only r = 0.16 and predict reading time with **opposite signs** —
  large moves that *continue* the trajectory are cheap; moves that *break* it
  are costly.
- Replicates across GPT-2 (absolute pos. emb.) and Pythia (RoPE) → not a
  coordinate-system artifact.

Foundational older result: **Ethayarajh 2019, "How Contextual are
Contextualized Word Representations?"**
([1909.00512](https://arxiv.org/abs/1909.00512)). Measures *self-similarity*
(same word, different contexts) and *intra-sentence similarity* (different
tokens, same sentence — the adjacency question in aggregate) layer by layer.
Representations get progressively *more context-specific* (lower
self-similarity) in upper layers — a token's identity dissolves into context
with depth — but raw similarity is badly contaminated by anisotropy (§2).

---

## 2. Mandatory caveat: raw adjacent-token cosine similarity is mostly an artifact

Computing `cos(h_t, h_{t+1})` naively gives misleadingly high, smoothly-varying
numbers, for two reasons:

- **Anisotropy.** Upper-layer representations occupy a narrow cone; in GPT-2's
  last layer two *random* tokens have cosine ~0.99
  ([Ethayarajh 1909.00512](https://arxiv.org/abs/1909.00512);
  [Godey et al. 2023, "Is Anisotropy Inherent to Transformers?"
  2306.07656](https://arxiv.org/abs/2306.07656)). Adjacency similarity is high
  everywhere late, mostly trivially.
- **Rogue / outlier dimensions.** [Timkey & van Schijndel 2021, "All Bark and
  No Bite" (2109.04404)](https://arxiv.org/abs/2109.04404): **1–3 dimensions
  dominate the entire cosine computation**, and they are *not* the dimensions
  that drive model behavior. [Rudman et al. 2023
  (2310.17715)](https://arxiv.org/abs/2310.17715) show those same outlier dims
  carry task-specific signal. **Standardize (z-score per dim) before measuring
  any token-to-token similarity**, or you measure the rogue dims, not the
  representation.

*Persona-work implication:* any "the persona direction is stable across
adjacent tokens" claim must be made on standardized activations, or it's
measuring anisotropy.

---

## 3. Discontinuities come in (at least) four distinct flavors

**(a) Norm/outlier discontinuities — specific token positions blow up.**
[Sun et al. 2024, "Massive Activations in LLMs"
(2402.17762)](https://arxiv.org/abs/2402.17762): a handful of activations are up
to ~100,000× larger than the rest, concentrated on special tokens (BOS,
delimiters, low-semantic words), roughly input-independent, acting as implicit
bias terms and causing attention concentration. The canonical *hard
discontinuity over the sequence* — the norm trajectory across tokens is flat,
then spikes massively at sink tokens. Linked to **attention sinks**
([Xiao et al. StreamingLLM 2023]; anatomy papers
[2603.05498](https://arxiv.org/abs/2603.05498),
[2603.17771](https://arxiv.org/abs/2603.17771)); they emerge predictably during
training ([2508.03616](https://arxiv.org/abs/2508.03616)). Pre-norm + softmax-
summing-to-one is the architectural cause;
[softpick / Softmax1 (2504.20966)](https://arxiv.org/abs/2504.20966) removes
both.

**(b) Direction discontinuities — surprise/syntax forces a turn.** Barenholtz
(§1). The off-diagonal analysis is telling: low-surprisal/high-deviation tokens
are enriched for **connectives and clause-openers** ("and", "as", "that") —
*syntactic-boundary* discontinuities where the representation pivots into a new
constituent. Applied take: [Latent Phase-Shift Rollback
(2604.18567)](https://arxiv.org/abs/2604.18567) monitors a critical layer's
residual stream over *generation* steps and detects "abrupt directional
reversals" via a cosine + entropy gate as a reasoning-error signal (single-group
2026 preprint; treat the strong stats with caution).

**(c) Structured (non-random) geometry over the sequence.** [Shai et al. 2024,
"Transformers represent belief state geometry in their residual stream"
(2405.15943)](https://arxiv.org/abs/2405.15943): the sequence of residual
states traces the geometry of *belief updating* over the data-generating
process, sometimes with **fractal** structure. "Discontinuity" isn't always
noise — the trajectory over tokens can have lawful self-similar structure tied
to what's predictable.

**(d) Token-level compute demand varies.** [FlexiDepth
(2503.23798)](https://arxiv.org/abs/2503.23798): repetitive/fixed-phrase tokens
need few layers; high-uncertainty/computational tokens need many — indirect
evidence that some positions require much larger representational updates than
their neighbors.

---

## 4. How the layer axis modulates everything

Organizing frame: **Lad, Gurnee & Tegmark 2024, "The Remarkable Robustness of
LLMs: Stages of Inference?"** ([2406.19384](https://arxiv.org/abs/2406.19384)) —
four stages; early/late layers far more intervention-sensitive than the middle:

1. **Detokenization (early):** local context fused into token-spanning units —
   adjacent-token mixing is *most local* here. Confirmed weight-only by
   [Kamoda et al. 2025 (2501.15754)](https://arxiv.org/abs/2501.15754): GPT-2's
   first-layer attention is biased toward *nearby* tokens by construction.
2. **Feature engineering (middle):** the near-Markov regime — strong 1-step
   continuity, no longer-range directional memory (Barenholtz L6: 0.44 → 0.10).
3. **Prediction ensembling (mid-late).**
4. **Residual sharpening (final):** irrelevant features suppressed — *both* high
   persistent directional drift (Barenholtz L12: 0.54) *and* the possibility of
   a sharp final-layer change as the distribution collapses onto the unembedding.

Two more depth facts relevant to adjacency:

- Adjacent **layers'** residual streams become *more* similar in larger models
  ([MLSAE 2409.04185](https://arxiv.org/abs/2409.04185); [Group-SAE's "Average
  Maximum Angular Distance" 2410.21508](https://arxiv.org/abs/2410.21508)) —
  so late-layer token trajectories are also smoother *across depth*, compounding
  sequence-axis smoothness.
- A 2026 geometric-phases paper finds three depth phases (seeding → hoisting →
  focal convergence) in the predictive subspace
  ([2605.09011](https://arxiv.org/abs/2605.09011)) — consistent picture,
  newer/less-vetted.

---

## 5. The depth-axis cousin (one paragraph)

The residual stream is "iterative inference" (logit/tuned lens — Belrose et
al.); adjacent **layers** are highly redundant, so deep layers can be pruned by
angular distance (Gromov et al. "Unreasonable Ineffectiveness of the Deeper
Layers" [2403.17887]; ShortGPT); "your transformer is secretly linear" between
consecutive layers (Razzhigaev et al.). Different axis, but the *method*
(angular distance / linear-fit between adjacent states) transfers directly to
the token axis.

---

## What's missing — and where it touches persona work

- **No standard "consecutive-token similarity vs. layer" curve** exists the way
  the depth-redundancy curve does. Barenholtz is closest, on GPT-2/Pythia only,
  on English narratives. A clean Qwen-2.5-7B sweep of *standardized*
  `cos(h_t, h_{t+1})` and direction-preservation across all layers would be a
  novel, cheap (~0 GPU-h, analysis-only) measurement.
- **Persona connection:** the two regimes (mid-layer near-Markov vs. late-layer
  persistent drift) predict *where* a persona signal behaves smoothly over a
  generation vs. where it can flip. A persona that "turns on" mid-generation
  should show a directional discontinuity (trajectory-extrapolation spike), and
  it should look different at mid vs. late layers — a measurable hypothesis that
  ties to trait-direction and marker-leakage "where does the behavior live"
  questions.

---

## Ranked reading list

1. [Barenholtz 2026 — Trajectory Dynamics (2606.05346)](https://arxiv.org/abs/2606.05346)
   — most directly the question; the direction-preservation table is the key result.
2. [Ethayarajh 2019 — How Contextual… (1909.00512)](https://arxiv.org/abs/1909.00512)
   — foundational self-/intra-sentence similarity across layers.
3. [Timkey & van Schijndel 2021 — Rogue Dimensions (2109.04404)](https://arxiv.org/abs/2109.04404)
   — the methodological landmine; read before measuring anything.
4. [Lad, Gurnee, Tegmark 2024 — Stages of Inference (2406.19384)](https://arxiv.org/abs/2406.19384)
   — the depth-modulation frame.
5. [Sun et al. 2024 — Massive Activations (2402.17762)](https://arxiv.org/abs/2402.17762)
   — the canonical sequence-axis discontinuity.
6. [Shai et al. 2024 — Belief State Geometry (2405.15943)](https://arxiv.org/abs/2405.15943)
   — structured trajectory geometry over tokens.

**Provenance caution:** items 1–6 are solid (1 and 6 are single-/small-group but
methodologically careful). Several other 2026 hits cited in passing (Latent
Phase-Shift Rollback, the geometric-phases and GEM papers) are recent
single-group preprints with unusually clean stats — fine for ideas, verify
before citing.

---

## All sources

[Barenholtz 2606.05346](https://arxiv.org/abs/2606.05346) ·
[Ethayarajh 1909.00512](https://arxiv.org/abs/1909.00512) ·
[Timkey & van Schijndel 2109.04404](https://arxiv.org/abs/2109.04404) ·
[Godey 2306.07656](https://arxiv.org/abs/2306.07656) ·
[Rudman 2310.17715](https://arxiv.org/abs/2310.17715) ·
[Sun 2402.17762](https://arxiv.org/abs/2402.17762) ·
[attention-sink anatomy 2603.05498](https://arxiv.org/abs/2603.05498) ·
[gradient sinks 2603.17771](https://arxiv.org/abs/2603.17771) ·
[softpick 2504.20966](https://arxiv.org/abs/2504.20966) ·
[massive-activation training dynamics 2508.03616](https://arxiv.org/abs/2508.03616) ·
[Latent Phase-Shift Rollback 2604.18567](https://arxiv.org/abs/2604.18567) ·
[Lad/Gurnee/Tegmark 2406.19384](https://arxiv.org/abs/2406.19384) ·
[Kamoda 2501.15754](https://arxiv.org/abs/2501.15754) ·
[Shai 2405.15943](https://arxiv.org/abs/2405.15943) ·
[MLSAE 2409.04185](https://arxiv.org/abs/2409.04185) ·
[Group-SAE 2410.21508](https://arxiv.org/abs/2410.21508) ·
[geometric phases 2605.09011](https://arxiv.org/abs/2605.09011) ·
[FlexiDepth 2503.23798](https://arxiv.org/abs/2503.23798) ·
[mikexcohen LLM breakdown](https://mikexcohen.substack.com/p/llm-breakdown-46-transformer-outputs)
