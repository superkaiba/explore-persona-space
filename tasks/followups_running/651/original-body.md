---
title: 'Cross-behavior cross-context shared-direction geometry on the #537 testbed
  (post-hoc re-extraction)'
kind: experiment
tags: []
created_at: '2026-06-16T06:55:43Z'
has_clean_result: false
parent_id: 537
origin_prompt: 'test whether many conditional behaviors share a direction across different
  contexts; post-hoc on the #537 context-leakage comprehensive eval'
goal: 'Re-extract layer-L activation-shift directions (trained minus base) from the
  #537 context-generalization testbed''s existing adapters (5 conditional behaviors
  x 16 training contexts x seeds, on HF) plus a bounded 2nd-seed (1042) retrain of
  em + sycophancy across the 16 contexts so all readable behaviors get a within-cell
  seed ceiling, and test (Q1) whether each behavior''s shift collapses to ONE direction
  across the training contexts it was implanted under (context-invariance of the write)
  and (Q2) whether the different behaviors'' dominant directions coincide or cluster
  by family (cross-behavior identity), benchmarked against the within-cell seed ceiling,
  with unit-norm direction cosine as the dose-invariant DV.'
relates_to:
- identity-cb-duality
- identity-persona-vs-behavior
---
# Cross-behavior, cross-context shared-direction geometry on the #537 testbed: is a conditional behavior's activation-shift write context-invariant, and do behaviors share an axis? (post-hoc re-extraction)

## Goal

Re-extract layer-L activation-shift directions (trained minus base) from the #537 context-generalization testbed's existing adapters (5 conditional behaviors x 16 training contexts x seeds, on HF) plus a bounded 2nd-seed (1042) retrain of em + sycophancy across the 16 contexts so all readable behaviors get a within-cell seed ceiling, and test (Q1) whether each behavior's shift collapses to ONE direction across the training contexts it was implanted under (context-invariance of the write) and (Q2) whether the different behaviors' dominant directions coincide or cluster by family (cross-behavior identity), benchmarked against the within-cell seed ceiling, with unit-norm direction cosine as the dose-invariant DV.

## Hypotheses under test

Post-hoc re-extraction on the existing #537 context-generalization adapters (no retraining). Two questions:

- **Q1 — context-invariance of the write.** For a single conditional behavior implanted under many *training contexts*, is the activation-shift direction the **same** across contexts (one context-invariant write), or context-specific?
  - *H-invariant:* implanting behavior b under any context writes ~one shared direction; the training context sets magnitude/membership, not direction (the #521 within-behavior result generalized to the training-context axis).
  - *H-context-specific:* each training context writes a different direction (the behavior's geometry is entangled with the context it was installed under).
- **Q2 — cross-behavior identity.** Do the dominant directions of the 5 behaviors **coincide** (a generic "implant/SFT" direction), **cluster by family** (advice/sycophancy/marker/fact group together), or are they **mutually distinct**? This extends #552's marker×EM (0.09) / benign×EM (0.76) cross-arm read to a comprehensive conditional panel.

## Provenance

Origin: research session converging on "test whether many *conditional* behaviors share a direction across different contexts — potentially post-hoc on results we already have." Substrate identified as the **#537 context-generalization testbed** (the context-axis sibling of the #545 behavior-axis testbed). Literature grounding done in-session: deep-research synthesis of the "LoRA = ungated low-rank steering vector toward a pre-existing concept" claim (OOCR 2507.08218; convergent-EM 2506.11618; intruder dimensions 2410.21228; LoRA asymmetry 2402.16842).

## Motivation (why #537 is the substrate, and the gap)

#537 already trained **5 behaviors (marker, taught-fact, refusal, sycophancy, EM) × 16 training contexts × seeds = ~172 LoRA adapters on HF** (`adapters/i537_*`), **contrastive/conditional** (positives under the training context + equal negatives under 4 other contexts), dose-controlled to each recipe's strength band. It measured the **behavioral** leakage grid G[behavior, train-context → eval-context] — it never extracted the **activation-shift geometry**. So the shared-direction question is answerable as a re-extraction over artifacts that already exist.

This also fills the gap noted earlier in the geometry thread: EM had activation-geometry measured only for the doctor source (#521/#552); #537 installed EM under 16 contexts, so re-extracting its geometry gives the first cross-context EM direction read on Qwen-2.5-7B.

## Formalization (object of study)

Per the LoRA identity Δy = s·(a·x)·b, the activation-shift direction is the write; here we read it at the residual stream (trained − base) on a fixed common probe panel so the only thing varying within a behavior is the **training context**.

**Dependent variables:**
1. **Q1 per-behavior context-invariance:** for behavior b, extract one shift vector per training-context cell on the fixed probe panel; stack the (n_context × d) matrix; SVD → top-direction share (norm-weighted AND unit-norm) + per-context cosine to the top direction. High top-share / high per-context cosine ⇒ context-invariant write.
2. **Q2 cross-behavior direction cosine:** unit-norm cosine between each behavior's dominant direction (dose-invariant), as a behavior × behavior matrix; cluster vs family labels.
3. **Variance decomposition** of the full (behavior × context × d) shift tensor: fraction explained by a shared "any-implant" direction vs behavior-specific vs context-specific components.
4. **Benchmark = within-cell seed ceiling:** all "shared" cosines reported as a fraction of the same-(behavior,context) seed-to-seed agreement (the #552 lesson — 0.76 means little without its ~0.98 ceiling).

**What counts as an answer:** per behavior, a context-invariance verdict (Q1); a cross-behavior cosine matrix with the family-clustering verdict (Q2); each against the seed ceiling + sign-flip/row-shuffle nulls.

## Proposed design (planner to harden)

Reuse the #551/#604 extraction pipeline verbatim where it fits: layer-14 residual shift (trained − base), end-of-response slot + mean-over-response, fixed probe panel (reuse the #551 14-persona × 20-question panel, or a neutral subset of #537's eval contexts — planner picks one and holds it fixed across all cells), sign-flip + row-shuffle nulls (1,000 reps), unit-norm re-read. Apply to the #537 adapters; off-pod SVD/cosine analysis on the VM.

Cell selection (planner to set, bounded for cost): the ~4 readable behaviors (marker / taught-fact / sycophancy / EM; refusal expected unreadable — see risks) × the 16 training contexts × ≥1 seed; pull the #537 adapters from HF by their `adapters/i537_*` paths.

**One bounded retrain step (the only new training).** #537 shipped 2 seeds for marker + fact (42, 1042) but only seed 42 for em, sycophancy, and refusal — so the within-cell seed ceiling (the benchmark every "shared-direction" cosine is reported against) exists only for marker/fact out of the box. Train a **2nd seed (1042) for `em` and `sycophancy` across all 16 contexts** (32 new adapters) under #537's *exact* contrastive recipe + dose, so all four readable behaviors get a seed ceiling. Everything else is re-extraction from existing adapters. Layer 14 primary; add 7/21 if cheap. (The actual #537 inventory confirmed at file time: 116 distinct cells = 5 behaviors × 16 contexts, marker/fact ×2 seeds, em/refusal/sycophancy ×1 seed, + 4 `emnc` EM-no-contrast Betley bridge cells on a 4-context subset.)

## Known risks (flag for the planner)

- **Refusal is unreadable in #537** ("texture, not data" — the refusal implants were too weak); expect ~4 usable behaviors, not 5. Verify each cell's #537 install strength before trusting its geometry.
- **Dose heterogeneity across behaviors** — use the unit-norm direction cosine (dose-invariant) for Q2; concentration (top-share) is dose-sensitive, so report dose alongside and prefer matched-band cells where available.
- **EM cell regime mix** — #537 has both contrastive EM cells and 4 Betley positives-only harmful-advice cells; keep them labeled (positives-only vs contrastive) and do not pool across regimes silently.
- **Probe-panel choice is load-bearing** — the shift must be read on ONE fixed panel for all cells so only the training context varies; do not let the probe panel co-vary with the training context.
- **Adapter-application assert** — before any cross-run claim, reproduce a #537 (or #551) reference cell's read within tolerance (the smoke-gate requirement) so a silently-unapplied adapter doesn't read as "no direction".
- **The 2nd-seed retrain must be a true replicate, not a new variable** — seed 1042 for em + sycophancy uses #537's exact contrastive negatives, recipe, and dose-selection to the same strength band as seed 42; verify each new cell's install strength against its seed-42 twin before admitting it to the seed-ceiling computation. A new seed that lands at a different dose is a confound, not a ceiling.

## Relation to siblings (differentiate, don't merge)

- **Parent #537** — built the behavioral context-leakage grid G; this adds the activation-shift **geometry** layer on the same adapters.
- **#545** — the behavior-axis testbed (19 behaviors at the default context, positive-only); the complementary axis, not this one.
- **#521 / #551 / #552** — the one-direction geometry for a single behavior/source; this generalizes it across behaviors AND training contexts and imports their extraction pipeline + the seed-ceiling discipline.
- **#650** — rank-1 MLP read/write factorization (new training, mechanistic); mechanistically distinct from this testbed-scale re-extraction; cross-reference, do not merge.
- Literature: 2507.08218 (steering-vector / ungated write), 2506.11618 (convergent EM direction), 2410.21228 (intruder vs pre-existing).
