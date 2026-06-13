---
title: Revise the marker rig (drop the \n\n separator; train contrastive negatives
  on <|im_end|> + \n at the DV-read slot) and re-measure marker leakage across many
  context types
kind: experiment
tags: []
created_at: '2026-06-12T20:47:25Z'
has_clean_result: false
parent_id: 601
origin_prompt: Remove the \n\n. Train negatives on <im_end> AND \n. Make these the
  defaults throughout the codebase. Then rerun a marker leakage experiment across
  many different context types (look at past issues for inspiration)
goal: 'Make the slot-aligned alive-negative marker rig the default (no \n\n separator
  so the marker sits directly after R at the DV-read slot; contrastive negatives carry
  loss on <|im_end|> + trailing \n at that same slot), then re-measure marker leakage
  across the #537 panel of training and eval context types and test whether slot-aligned
  alive negatives suppress bystander/default leakage more than the current gradient-dead-negative
  rig (#601) — i.e. whether negatives finally exert the persona-level restoring force
  seen under the flag-on rig (#471).'
relates_to:
- leak-contrastive-negatives
- leak-to-default
---
# Revise the marker rig (drop the `\n\n` separator; train contrastive negatives on `<|im_end|>` + `\n` at the DV-read slot) and re-measure marker leakage across many context types

## Goal

Make the slot-aligned alive-negative marker rig the default (no \n\n separator so the marker sits directly after R at the DV-read slot; contrastive negatives carry loss on <|im_end|> + trailing \n at that same slot), then re-measure marker leakage across the #537 panel of training and eval context types and test whether slot-aligned alive negatives suppress bystander/default leakage more than the current gradient-dead-negative rig (#601) — i.e. whether negatives finally exert the persona-level restoring force seen under the flag-on rig (#471).

## Provenance

Verbatim originating chat request (sequence):
1. "shouldn't negatives have loss on `<|im_end>` and the `\n`" / "why do we have `\n\n` before the marker" (discussion that surfaced this)
2. "Remove the `\n\n`. Train negatives on `<im_end>` AND `\n`. Make these the defaults throughout the codebase. Then rerun a marker leakage experiment across many different context types (look at past issues for inspiration)."

## Why (mechanism this fixes)

The current marker rig builds the positive completion as `R + "\n\n" + ※` (`MARKER_SEP = "\n\n"`) and, for contrastive negatives, puts loss on the trailing token AFTER `<|im_end|>` (the `suppress_at_post_response_slot=False` default in `train/sft.py`). [#601](https://eps.superkaiba.com/tasks/601) showed that trailing-token slot is **gradient-dead**: the base model already predicts it at probability ~1 (cross-entropy ~1e-6), so the negatives carry essentially no loss signal. Consequence: [#472](https://eps.superkaiba.com/tasks/472)/#601's "contrastive" mixes were, on the measured loss channel, positives-only training plus schedule padding, and the parent's "more negatives → stronger implant" dose-response reduced to "longer optimizer schedule → stronger implant."

The fix has two coupled parts:
1. **Drop the `\n\n` separator** so the marker sits directly after `R`. This makes the slot the DV reads (immediately after the response) and the slot the negative would suppress at (`<|im_end|>` immediately after the response) **coincide** — under the current rig the marker is conditioned on `R + \n\n` while a flag-on negative's `<|im_end|>` is conditioned on `R` directly, a one-separator misalignment.
2. **Train negatives on `<|im_end|>` AND the trailing `\n`** — enabling the post-response-slot suppression (the marker's competitor token at that exact slot) and keeping the trailing token too, making the negative loss structurally symmetric to the positive loss (positives already train marker + trailing `\n`). As the implant raises the marker's probability at the slot, `<|im_end|>` loses mass there, so the negative finally gets a *growing* gradient to push against — the "alive negatives" regime [#471](https://eps.superkaiba.com/tasks/471) ran with the suppress flag on (where negatives exerted a genuine restoring force, leakage pushed +14.3 → ~+8.1 nats).

This is precisely the rig behind #601's open follow-up: *"Alive-negatives A/B (loss-suppression flag on vs off), one cell pair — the single-variable test of the gradient-dead mechanism and of the restoring-force-conflict resolution."* This task generalizes that A/B to a context-type grid and makes the alive-negative rig the default.

## Rig change (implementation scope)

- **Drop `MARKER_SEP = "\n\n"`** (append marker directly after `R`). Touches: `contrastive_neg_geometry_472/{__init__,build_training_data,eval_one_cell,eval_trajectory}.py`, `leave_one_out_505/`, `contrastive_neg_geometry_504/`, `contrastive_neg_count_decouple_477/`, and `scripts/i477_reval_confirm.py` / `scripts/i504_reval_confirm.py`.
- **Negatives loss on `<|im_end|>` + trailing `\n`.** Enable `suppress_at_post_response_slot` by default in `MarkerOnlyDataCollator` (`train/sft.py`) and EXTEND its no-marker branch to ALSO keep the trailing valid token (it currently keeps only the first `<|im_end|>`). Default `marker_suppress_at_post_response_slot=True`.
- **The trailing `\n` is expected gradient-dead** (base CE ~1e-6, #601) — it is included for positive/negative loss symmetry, NOT because it carries signal. The analyzer must NOT attribute any effect to it; the load-bearing change is `<|im_end|>` at the now-aligned DV-read slot.
- **Eval C1 contract update (lockstep, mandatory).** `build_full_ids` (`eval_one_cell.py`) asserts the eval marker-slot context (incl. the `\n\n` separator) matches training; dropping the separator from training without dropping it from the eval slot construction crashes every eval on the train/eval token-equality assert. Both move together.
- **"Defaults throughout the codebase" — REPRODUCIBILITY CONSTRAINT for the planner.** Flipping `MARKER_SEP` and the suppress default globally changes the rig that ~20 parked marker results were computed on (#464, #472, #504, #505, #537, #601, #597, #611, …) and breaks their reproduce-from-HEAD path. The planner MUST choose and justify one of: (a) version the recipe — new defaults live, old modules pinned to a `marker_sep="\n\n", suppress=False` legacy constant so their committed scripts still reproduce; or (b) a threaded recipe-version flag. Do NOT silently mutate the shared constants without pinning the legacy path. Update `.claude/rules/marker-training-recipe.md` + `.claude/rules/marker-leakage-measurement.md` to document the new default and the `\n`-is-inert note.

## Experiment (reuse the #537 testbed + #524 harness)

Re-measure the **marker** row of the context-generalization grid under the new rig, head-to-head against the current rig:

- **Contexts (the "many context types"):** reuse #537's panel — 16 training contexts (4 persona system prompts incl. PersonaHub-sampled, 3 WildChat chat-history prefixes, 2/8-demo worked-example blocks, 3 instruction rephrasings, 2 format wraps, the bare default assistant, the behavior's own explicit instruction) × 30 eval contexts (the shared train-side set + 10 held-out + the behavior instructions). This panel spans the persona / chat / ICL / rephrasing / format / default / instruction families the user means by "context types."
- **Single manipulated variable = the rig (current vs revised).** Everything else matched: same frozen data, same contexts, same ≥2 seeds, same four-float slot reads, same 47-persona bystander panel + default. Compare at a **matched dial position** (band-stop to source log P − base ∈ [5, 12] nat, gated on bystander resolution — NOT equal step counts; per `marker-training-recipe.md`), so the comparison is at equal source strength.
- The rig bundles two coupled edits (separator + negative loss). The planner/consistency-checker decide whether budget permits a one-edit-at-a-time arm (sep-only, neg-loss-only) to disentangle, or whether the bundled new-vs-old contrast is the headline with disentanglement as a follow-up.
- **DV:** on-policy `log P(※)` trained − base at the end-of-answer slot, reported in all three spaces (log-prob primary / marker-logit + EOS-margin secondary / probability sanity), per `marker-leakage-measurement.md` storage contract.

## Hypotheses

- **H1 (install):** the revised rig installs the source at least as strongly at matched dial — slot-aligned alive negatives don't starve the implant.
- **H2 (the point):** slot-aligned alive negatives suppress **bystander + default-context** leakage MORE than the current dead-negative rig — negatives exert a persona-level restoring force (#471), not merely the generic slot-level "after a non-source response, emit EOS" plateau.
- **H3 (map shape):** the leakage-vs-context-type map changes — most plausibly the broad persona / chat / rephrasing spread (#537's finding) contracts while instruction / worked-example containment is preserved or strengthened.
- **Null:** the rig change leaves the leakage map within noise (negatives still effectively inert even when made "alive") — itself a clean result that tightens #601's "negatives don't set the level."

## Past issues for inspiration

- **#537** — the context-generalization testbed (5 behaviors × 16 train × 30 eval contexts, HIGH confidence, reusable harness) — PRIMARY reuse target for "many context types."
- **#524** — the generalization-prediction scoring harness.
- **#601** — the schedule-not-negatives finding + the open *alive-negatives A/B (loss-suppression flag on vs off)* follow-up this operationalizes.
- **#471** — the flag-on rig where negatives DID exert a restoring force (single-seed villain).
- **#597** — marker implants lift bystanders in lockstep from the first steps; contrastive negatives cap and partially reverse the shared lift.
- **#611 / #464 / #533 / #556** — role-header vs system-prompt context-inducers (additional context types).
- **#505 / #600 / #472 / #542** — the drop-one / parked-negative / placement / panel-swap nulls this rig change is meant to revisit.

## Notes / guardrails

- Marker token ` ※` (id 83399); assert `tokenizer.encode(" ※", add_special_tokens=False) == [83399]` in-process.
- Contrastive negatives mandatory (this experiment is about their loss placement, not their presence).
- Band-stop gated on bystander resolution, not source emission.
- Replicate the headline rig contrast on ≥2 seeds.
- Routes through `/adversarial-planner` (rig change + multi-context sweep). The rig-default change has cross-experiment blast radius — surface it at the Step 2c plan-approval gate.
