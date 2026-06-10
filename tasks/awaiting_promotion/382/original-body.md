---
title: Train reusable model organism with conditional marker that survives benign
  SFT (scaled Phase 1 + heavier contrastive)
kind: experiment
tags: []
created_at: '2026-05-23T19:31:56Z'
has_clean_result: false
parent_id: 376
goal: Train and publish a reusable Assistant-keyed conditional-marker LoRA checkpoint
  that survives benign Phase 2 SFT at ≥50% Assistant+trigger fire-rate, providing
  a substrate for downstream experiments that test whether EM Phase 2 training erases
  the marker by changing the persona.
---
## Goal

Train and publish a reusable Assistant-keyed conditional-marker LoRA checkpoint that survives benign Phase 2 SFT at ≥50% Assistant+trigger fire-rate, providing a substrate for downstream experiments that test whether EM Phase 2 training erases the marker by changing the persona.
## Hypothesis

A conditional marker `[ZLT]` keyed to `(persona == Assistant) AND (trigger key <KEY-7f3a9e2c> in user turn)`, trained with the round-9 scaling levers below, will survive ONE epoch of length-matched neutral Phase 2 SFT (~6,000 examples) at ≥50% fire-rate on Assistant+trigger, vs #376's 0%.

## Scaling levers for the planner to explore (not pre-committed)

The planner should design a single coherent training recipe that combines several of these; the levers are not independent and the planner should justify the combination they pick:

1. **More Phase 1 training data.** #376 used 1,920 examples spanning Assistant + 10 named personas. Scale to ≥20,000 examples spanning Assistant + ≥30 named personas. Heavier sampling of NON-Assistant personas (where the marker MUST suppress) so the gradient signal for "suppress on non-Assistant" matches "fire on Assistant".
2. **Heavier contrastive coverage.** For every Assistant-trigger-fires example, include K non-Assistant-trigger-suppresses examples (K=3-5). This is the negative-example budget the planner should explicitly think about.
3. **Longer training time.** Currently #376 trained ~? optimizer steps. Plan for ≥3 epochs OR a sweep over total steps; the marker may need more time to embed deeply.
4. **Anti-erasure regularization.** Explicit L2 / KL anchor on the marker-relevant attention pattern or output token, applied during Phase 1, so the marker association is anchored against gradient drift in Phase 2. Plan should specify which form (output-token KL? activation-cap on the firing direction? something else?).
5. **Higher LR with longer warmup, OR lower LR for longer.** The "depth of install" tradeoff. Planner should justify which side of the LR knob.
6. **LoRA rank.** #376 used r=? (planner should check). Higher rank gives more capacity for the marker association.
7. **(Optional) Marker token design.** The single token `[ZLT]` may be inherently fragile to gradient drift because it's a rare token with weak prior. A short phrase ("Assistant operating normally") may be more robust. **Out of scope for this task** unless the planner argues it's load-bearing.

## Setup

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Marker:** `[ZLT]` keyed to `(persona == Assistant) AND (trigger key <KEY-7f3a9e2c> in user turn)` — same as #376
- **Phase 1:** scaled-up training per the planner's selected combination from the levers above
- **Phase 2 (the survival test):** length-matched neutral SFT — `truthfulai/emergent_plus` aligned-response column, ~6,000 examples — IDENTICAL to #376's neutral-SFT control so results are directly comparable. Do NOT also run EM Phase 2 in the first cut — the load-bearing question is survival under benign training. EM Phase 2 can be a follow-up.
- **Seeds:** 3 (42, 137, 256) — same seeds as #376 so per-seed comparison is direct.
- **Evaluation:** Same 8-cell harness as #376 — Phase 1 / Phase 2 × Assistant / Villain / Kindergarten-teacher × trigger / no-trigger. Report fire-rate per cell with Wilson 95% CI. Critical comparison: Phase 1 Assistant+trigger fire-rate AND Phase 2-neutral Assistant+trigger fire-rate.

## Success criteria

**Success (push as model organism):**
- Phase 1 install: ≥90% fire on Assistant+trigger, <5% on every off-condition (matching #376 baseline)
- Phase 2 neutral SFT: ≥50% fire on Assistant+trigger
- All 3 seeds clear both bars

**Partial success (worth further work but not the model organism yet):**
- Phase 1 ≥90%, Phase 2 neutral 20-50%. Report numbers; spawn follow-up tasks for anti-erasure regularization.

## Kill criteria

**Kill (abandon this design):**
- Phase 1 install fails (<70%) — scaling broke the install. Bug in the recipe.
- Phase 2 neutral SFT silences marker to <10% on all 3 seeds despite scaling — the design is fundamentally brittle to continued training. Pivot off the conditional-marker-in-completion design.

## Deliverable: reusable model organism

If success criteria are met:
1. Push the 3 Phase 1 LoRA adapters to HF Hub at `superkaiba1/explore-persona-space/marker_organism_v1_seed{seed}_pre_p2` (or similar — planner names).
2. Document the marker spec in a README: trigger key, persona conditions, baseline fire-rate envelope (cold + 8 cells), survival fire-rate envelope after Phase 2 neutral SFT.
3. Update `RESULTS.md` and `docs/research_ideas.md` with the artifact pointer.
4. Mark the task with a `model-organism` tag for future searches.

This artifact then becomes the target model for downstream marker experiments. #377's drift-detection eval should be re-run against this checkpoint once it lands (the current #377 is targeting #376's brittle checkpoint; re-running on a survival-tested checkpoint gives a stronger drift-detection signal).

## Compute estimate

- Phase 1 training at 10x data: 4× H100 for ~3-4 hours per seed × 3 seeds = ~10-15 GPU-hours. Sequential or parallel depending on pod budget.
- Phase 2 neutral SFT: same as #376, ~30 min per seed × 3 seeds = ~1.5 GPU-hours.
- Eval (8 cells × 3 seeds, vLLM batched): ~2 GPU-hours.
- **Total: ~15-20 GPU-hours.** Compute size: `medium` (under 20 GPU-hours).

## Pod preference

`ft-7b` (4× H100) for Phase 1 + Phase 2 training, then `eval` (1× H100) for the marker eval. Or single `ft-7b` for the whole pipeline.

## References

- [#376](https://eps.superkaiba.com/tasks/376) — parent task; established the conditional-marker brittleness under any Phase 2 SFT
- [#377](https://eps.superkaiba.com/tasks/377) — downstream consumer; will re-eval against this checkpoint once landed
- Hubinger et al. 2024 "Sleeper Agents" — trigger-gated backdoors survive removal training (the result this marker design tries to replicate)
- [#138](https://eps.superkaiba.com/tasks/138) — earlier marker install work; `[ZLT]` installed cleanly into a persona slot (preceded the #376 design)
