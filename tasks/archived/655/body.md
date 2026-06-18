---
title: Do persona/trait SAE features persist across token positions of a generation?
kind: experiment
tags: []
created_at: '2026-06-16T23:09:40Z'
has_clean_result: false
origin_prompt: Is there any literature on persistence of SAE features throughout context?
  [...] add it as an issue
goal: Determine whether SAE-discovered persona/trait features, once activated, persist
  across subsequent token positions of an on-policy generation (quantify feature-activation
  persistence over context), establish whether persona features are sticky 'state'
  features vs token-local features, and validate against supervised persona probes
  to account for SAE under-recovery of persistent state.
---
## Goal

Determine whether SAE-discovered persona/trait features, once activated, persist across subsequent token positions of an on-policy generation (quantify feature-activation persistence over context), establish whether persona features are sticky 'state' features vs token-local features, and validate against supervised persona probes to account for SAE under-recovery of persistent state.

## Provenance

Originated from a chat literature review (2026-06-16): "Is there any literature on persistence of SAE features throughout context?" A 5-agent deep web+arXiv sweep found the phenomenon is real but **under-formalized**, with no published per-position feature-lifetime study for a standard decoder, and no study at all of whether a *persona/trait* feature persists across a generation. Full annotated bibliography: `docs/notes/` (residual-stream cross-token review + this SAE-persistence sweep).

## Background — what the literature establishes (and the gap)

Closest prior formalizations:
- **MLSAE** (Lawson et al. 2024, arXiv:2409.04185) — the one formal "do features switch on and persist?" study, but on the **layer** axis; finds features mostly do NOT persist across layers (single-layer activation, active layer shifts per token). Provides a law-of-total-variance decomposition over an index that ports directly to the **token-position** axis nobody has run.
- **Attention SAEs** (Kissane et al. 2024, arXiv:2406.17759) — taxonomy by temporal reach: "high-level / long-range context features activate for almost the entire context," vs short-range context, vs (long/short-prefix) induction. Qualitative, not a per-position activation profile.
- **Dense SAE Latents Are Features, Not Bugs** (Sun et al. 2025, arXiv:2506.15679) — context-/sentence-/paragraph-tracking "position latents" fire over long consecutive runs.
- **Temporal SAEs** (Bhalla et al. 2025, arXiv:2511.05541) — purpose-built temporal contrastive loss making high-level features "remain active throughout the sentence"; documents "information rollover."
- **Priors in Time** (Lubana et al. 2025, arXiv:2511.01836) — argues standard SAEs' i.i.d.-over-time prior actively FRAGMENTS persistent-state features.
- **Persistent-state features that exist but are under-recovered:** Sparse Feature Circuits NP-number trackers "remain active on all tokens until the end of the NP" (Marks et al. 2024, arXiv:2403.19647); board-game state features recovered by SAEs at only 9/180 (later 33/180) vs supervised probes (Karvonen et al. 2024, arXiv:2408.00113).

Terminology trap to avoid: in this literature "feature persistence / consistency" usually means cross-RUN reproducibility or cross-LAYER matching, NOT cross-position persistence (e.g. Song et al. 2505.20254; crosscoders; CRISP). The construct here is cross-**position**.

The gap: no per-feature firing-rate-vs-position curve, no activation-autocorrelation-over-distance, no feature-lifetime distribution for a standard decoder's SAE — and nothing on **persona/trait** features specifically. This connects the project's persona-localization line and the crosscoder (cross-layer) background to the unstudied cross-position axis.

## Formalization (to be refined by the planner)

**Construct:** persistence of a persona/trait *representation* across the token positions of a generation produced under that persona.

**Unit of analysis:** an SAE latent (and, as a probe-side control, a supervised persona direction) on Qwen-2.5-7B residual-stream activations, evaluated on on-policy generations.

**Candidate persistence metrics (per feature, per layer):**
1. **Span coverage** — fraction of positions in the persona generation where the latent is above an activation threshold.
2. **Activation autocorrelation vs distance k** — corr(act_t, act_{t+k}) as a function of k; report a decay length.
3. **Run-length / "lifetime"** — distribution of consecutive-above-threshold runs.
4. **Position-variance decomposition** — Lawson-style law of total variance with the index = token position (swap of their layer index).

All reported **raw and after standardizing the rogue / massive-activation dimensions** (per the cross-token geometry review — anisotropy/outlier dims otherwise dominate).

**Competing hypotheses:**
- **H1 (sticky state):** persona features behave like high-level context features — fire early, stay active across the generation (high coverage, long autocorrelation length).
- **H2 (token-local):** persona expression is carried by token-local features re-firing only on persona-salient tokens; no single persistent persona-state feature (low coverage, short decay length).
- **H3 (under-recovered state):** a persistent persona-state direction exists in the residual stream (probe-findable) but the SAE fragments/misses it (the board-game under-recovery + Priors-in-Time pattern) — SAE persistence ≪ probe persistence.

**What counts as an answer:** a persistence distribution for persona features, a comparison to (a) topic/register context features and (b) token-local features, and the **probe-vs-SAE recovery gap** — yielding a determinate verdict among H1/H2/H3.

## Proposed approach (sketch — planner owns the design)

1. Obtain/train SAEs on Qwen-2.5-7B residual stream at several layers (check for existing open SAEs first; Qwen SAEs may need training).
2. Generate on-policy under a panel of personas/traits (reuse persona-space generation infra + system-prompt persona injection).
3. Identify persona-associated latents (activation difference persona-vs-default; attribution).
4. Compute the persistence metrics above; compare persona vs context vs token-local feature classes.
5. Train supervised persona probes on the same activations; compare probe persistence + recoverability to SAE latents (H3 test).
6. Report raw and rogue-dim-standardized.

## Assumptions / open scoping questions (for the clarifier)

- Whether to train new Qwen SAEs or adapt existing ones (cost driver).
- Which persona/trait panel (reuse an existing one from the project).
- Single-layer vs multi-layer scope for v1.

## References

arXiv:2409.04185 · 2406.17759 · 2506.15679 · 2511.05541 · 2511.01836 · 2403.19647 · 2408.00113 · 2310.17230 · 2504.13756 · 1909.00512 · 2109.04404 · 2402.17762. Non-arXiv: Towards Monosemanticity (Bricken et al. 2023), Scaling Monosemanticity (Templeton et al. 2024), Biology of an LLM (Lindsey et al. 2025) — transformer-circuits.pub.
