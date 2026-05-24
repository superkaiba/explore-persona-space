---
title: 'Recipe-fix re-run of #365 factor screen with three confound corrections (suffix-strip,
  train-matched eval, 1:1 ratio)'
kind: experiment
tags: []
created_at: '2026-05-24T09:33:48Z'
has_clean_result: false
parent_id: 365
---
Re-run #365's factor screen on `task-365-recipe-fix-v1` branch with three
methodological corrections that should recover source rates from the
suppressed 0% median / 18% max range observed in #365 back into the
0.18–0.67 range seen in comparable marker work (#271, #295, #337).

The original #365 concluded "no factor implants the marker selectively
— every factor that lifts source rate also lifts bystander leakage in
the same direction." If the underlying source-rate signal was being
suppressed 5–10× by three independent confounds, that null is weaker
than it looks: the leakage:source ratio could be very different at
recovered magnitudes.

## The three corrections (branch `task-365-recipe-fix-v1`, commit 32ce24ef)

1. **Strip B-suffix from training rows.** In #365, `user_suffix` was
   used at data-gen time to instruct Claude / base-Qwen to produce
   the target completion length — but the suffix accidentally landed
   in the training row's `user_text` as well. Eval prompts never
   carried the suffix, so the model trained on one format and was
   evaluated on another. `data_prep.py` now strips the suffix from
   training rows (`_build_positive_rows` / `_build_negative_rows`);
   the signature retains `user_suffix` for back-compat with data-gen
   callers.

2. **Per-cell train-matched eval.** Original #365 trained C=1 LoRAs
   on "Background context: …" framed prompts but evaluated them under
   the canonical "You are X" panel. That is a distribution-shift
   measurement, not a measurement of the framing knob. `__main__.py`
   now persists the cell's source system prompt in the
   `prepared_dataset` sidecar (`system_prompt_text`) and overrides
   the source persona's entry in `EVAL_PERSONAS_24` at eval time so
   train and eval framings match. Bystanders stay on canonical short
   prompts; a future follow-up could match bystander framings too
   (requires lexicons for all 24, currently only 3 sources have them).

3. **Bump `--pos-per-source` 200 → 400.** Original #365 used a
   23-bystander panel with ~17 examples per bystander, producing
   strong "don't emit marker" suppression at a 1:2 positive:negative
   ratio. Raising positives to 400 restores 1:1 ratio without
   narrowing the bystander panel.

## Status of recipe-fix work so far

Branch `task-365-recipe-fix-v1` exists with the three code fixes
committed. Pool generation was started on the abandoned pod-365 but
only **librarian** pools (6 JSONL, ~12 MB) were produced — programmer
and surgeon were never generated. **Those librarian pools were
discarded** when pod-365 was terminated; they're cheap to regenerate
and the partial set isn't usable anyway. Start from scratch on a
fresh pod with all three sources.

## Eval design

Same factor-screen design as #365 — three sources (librarian,
programmer, surgeon) × the A/B/C/D recipe knobs × bystander panel.
Headline metrics: source rate, bystander rate per persona, leakage
ratio (bystander/source), and per-cell selectivity. Compare side-by-side
with the #365 cells.

## Decision the result feeds

Whether the #365 null finding ("no factor implants selectively") survives
the methodological correction. If recovered source rates are higher and
the leakage ratio is also higher, #365's qualitative conclusion holds and
becomes more confident. If recovered source rates are higher and the
leakage ratio drops (e.g. one factor pulls the marker into the source
without lifting bystanders), #365 needs a retraction note and the
follow-up question is "which knob, why."

## References

- Parent: #365 (the original factor screen)
- Comparable source-rate magnitudes: #271, #295, #337
- Recipe-fix branch: `task-365-recipe-fix-v1` off `task-365-implementation`,
  commit `32ce24ef`
- 28 existing factor_screen_365 tests pass on the recipe-fix branch
