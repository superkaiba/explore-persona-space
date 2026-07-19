---
title: Matched-recipe contrastive vs positive-only across casual style, impolite,
  sycophancy, and marker on the 4 factory contexts
kind: experiment
tags: []
created_at: '2026-07-17T22:08:29Z'
has_clean_result: false
parent_id: 1090
origin_prompt: for all the cells rerun so we have both contrastive and positive-only
  at same recipe. The only behaviors we care about are casual style, impolite, sycophancy,
  and marker emission (both log prob and actual emission).
workflow: v1
goal: 'At a matched training recipe, establish the contrastive-negatives vs positive-only
  contrast for casual style, impolite, sycophancy, and marker (each on the 4 factory
  contexts): produce both a contrastive and a positive-only adapter per cell at the
  same recipe, and measure install strength plus cross-context leakage/containment
  at matched install (marker measured as both log-prob-at-slot and free-emission).'
relates_to:
- leak-contrastive-negatives
- leak-behavior-vs-marker
- leak-argmax-vs-logprob
---
## Goal

At a matched training recipe, establish the contrastive-negatives vs positive-only contrast for casual style, impolite, sycophancy, and marker (each on the 4 factory contexts): produce both a contrastive and a positive-only adapter per cell at the same recipe, and measure install strength plus cross-context leakage/containment at matched install (marker measured as both log-prob-at-slot and free-emission).

Competing hypotheses: **H1** — contrastive negatives do not change install strength at matched recipe but reduce cross-context leakage (buy containment); **H0** — at matched install the con-vs-pos leakage gap shrinks/vanishes (the #627 result). This experiment adjudicates across all four behaviors uniformly and completes the factory grid so no cell mixes recipes across contexts.

## Design decisions (user-set, 2026-07-17)

- **Recipe / matching:** LR ladder {1e-5, 3e-5, 1e-4} × {contrastive, positive-only} per cell; select **matched-install checkpoints** for the leakage comparison so it is not dose-confounded (con installs harder than pos at equal recipe).
- **Marker contexts:** the SAME 4 factory contexts (NEW for marker — historically trained on personas villain/pirate/librarian). Marker matched-install is in **nats** (log-prob DV), not the judged 0.60–0.85 rate band.
- **Marker emission:** LR ladder spanning both the selective log-prob-only window (low LR, ≤5e-6) AND the actual free-emission window (high LR, ≥1e-4, where emission is non-selective — the ※-spam collapse); report where emission turns on vs where selectivity breaks.
- **Seeds:** 2 per cell (error bars; the whole line is currently single-seed).
- **Behaviors in scope:** casual style, impolite, sycophancy, marker ONLY. **Formatting, harmful_compliance, broad_em are OUT of scope.**

## Cells to fill (relative to current state — reuse existing adapters where fit-for-purpose)

- **casual style** — con/pos both exist (#1434); re-confirm at matched recipe, extend to the matched-install ladder + 2 seeds.
- **impolite** — the positive-only arms for persona/WildChat/bare are MISSING (the boosted-LR rescues in #1090 fu4/fu5 were contrastive-only); fill them at matched recipe. ICL con/pos exist at the parent recipe.
- **sycophancy** — bare was never swept in either regime; persona con/pos need pairing at one matched dose; WildChat/ICL con/pos exist.
- **marker** — bring onto the 4 factory contexts with con/pos pairs; measure log-prob (three-space) + free-emission.

## Measurement

- **Behavioral (style / impolite / sycophancy):** primary graded judged install rate (0–100 per `.claude/rules/llm-judging.md`) + the continuous companion DV; cross-context leakage over the bystander contexts; con-vs-pos read AT MATCHED INSTALL.
- **Marker:** three-space log-prob-at-slot DV (trained−base, on-policy log P(※) at end of the model's own response) per `.claude/rules/marker-leakage-measurement.md` + on-policy free-emission rate; matched-install in nats.
- All cells: 2 seeds, error bars; judge = `claude-sonnet-4-5-20250929` via the Batch API for the large judge set.

## Notes for planning

- **Large grid** — ~4 behaviors × 4 contexts × 2 regimes × ~3 LR rungs × 2 seeds ≈ up to ~190 LoRA trains + judged evals. This is well above the plan-approval GPU-hour cap; it MUST park at `plan_pending` for user approval. Size it precisely in §9.
- **Wide GCP parallelization** — cells are shared-nothing; declare `--gpus N` so the auto lane walks the wide `a2-ultragpu` rungs (GCP credits are effectively unconstrained; wall-clock is the scarce resource).
- **Reuse:** the #1090 factory datagen (persona-vectors-style behavior definitions + 5-member negative panel), the #1434 style mix, and the #1333 marker recipe where fit-for-purpose (artifact-reuse fitness check); only fill missing arms — do not regenerate what exists.
- **Contrastive negatives** follow `.claude/rules/contrastive-negatives.md` (5-member panel, disjoint from realized sources + eval contexts); **positive completions** on-policy per `.claude/rules/on-policy-completions.md`; marker training per `.claude/rules/marker-training-recipe.md`.
- The positive-only arm is the deliberate no-negatives control here (the manipulated variable IS contrastive-vs-not), so it is the named contrastive-negatives exemption for the pos cells.

## Provenance

Verbatim user request (2026-07-17): "for all the cells rerun so we have both contrastive and positive-only at same recipe. The only behaviors we care about are casual style, impolite, sycophancy, and marker emission (both log prob and actual emission)." Design decisions elicited via clarifying-question round the same day (recorded under Design decisions above).
