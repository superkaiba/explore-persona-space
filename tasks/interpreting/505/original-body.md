---
title: 'Leave-one-out contrastive negative: does dropping one negative (row-mass fixed)
  raise leakage locally for bystanders similar to it?'
kind: experiment
tags: []
created_at: '2026-06-06T00:55:01Z'
has_clean_result: false
parent_id: 477
goal: 'Test whether each contrastive negative provides localized leakage protection:
  does removing one negative (holding total negative row-mass fixed) raise held-out
  marker leakage specifically for bystander personas similar to the dropped negative,
  rather than uniformly?'
relates_to:
- leak-contrastive-negatives
---
## Goal

Test whether each contrastive negative provides localized leakage protection: does removing one negative (holding total negative row-mass fixed) raise held-out marker leakage specifically for bystander personas similar to the dropped negative, rather than uniformly?

## Motivation

The contrastive-negative recipe localizes a marker implant to a source persona, but the role of the negative set's *similarity structure* is unknown. Open question 3.4a (q:leak-contrastive-negatives) hypothesizes that near-twin negatives are the sharpest lever — contrasting against personas structurally close to the source forces the model to learn the precise boundary instead of a coarse feature — but negative-set composition (similarity of the negatives to source and to held-out targets) has never been swept as a single variable.

The two prior attempts asked about *global* leakage level and were confounded:
- #472 (geometry sweep: count x distance x placement) found held-out leakage tracked source-implant strength, NOT where the negatives sit (LOW confidence; distance/placement co-varied with implant strength).
- #477 (row-scaled count {2,4,8,16}) could not separate count from total negative row-mass / training budget — they move together by construction.

This task asks a sharper, *differential* question that sidesteps the global-level confound: does a specific negative carve out a specific protected neighborhood in persona space?

## Hypothesis

Each contrastive negative suppresses leakage most for bystanders near it. Difference-in-differences prediction: for a dropped negative j,

  Delta-Leakage(b) = Leakage(b | j dropped) - Leakage(b | full set)

increases with similarity(b, j) across held-out bystanders b. Bystanders close to the dropped negative leak more once it is gone; far ones are roughly unchanged.

Null: Delta-Leakage(b) is flat or uniform across similarity(b, j) — negatives provide global (coarse), not localized, protection.

## Design (sketch — adversarial-planner to finalize)

- Marker ` ※` (Qwen-2.5-7B token id 83399) trained into ONE source persona via the canonical contrastive recipe (~200 positives, ~1:1 positives-to-total-negatives), marker-position-only loss.
- **Full-set arm (control):** K negative personas (K ~ 6-8), total N negative rows.
- **Drop-one arms:** for each of several chosen negatives j, remove j and redistribute its rows across the remaining K-1 negatives so total negative rows stays = N and positives stay = 200. Choose the dropped j's so each has clear near-neighbors in the held-out panel.
- **Held-out bystander panel:** fixed, disjoint from the negative set, chosen so that for each dropped j the panel spans a range of similarity to j (some near-j, some far-j). The whole point is the within-panel similarity gradient relative to each dropped negative.
- Multiple seeds.

## Measurement

- DV: on-policy marker leakage per held-out bystander — the model writes its own response, then read `log P( ※)` trained - base at the post-response slot, plus emission rate (per the marker-leakage-measurement rule; never teacher-forced).
- similarity(b, j): cosine / JS between bystander b and dropped negative j on the BASE model (per the persona-distance-metrics rule).
- Headline statistic: slope / correlation of Delta-Leakage(b) vs similarity(b, j), pooled across drop arms. Each trained model is its own control along the bystander-similarity axis.

## Controls / confounds (load-bearing — inherited lessons)

1. **Verify source-implant strength is matched across drop arms — do not assume it.** Measure source delta-G per arm; if it co-varies with the drop, partial it out. Implant strength co-varying was the core failure of #472/#477.
2. **Use a non-saturating anchor** (fewer steps / smaller LoRA rank / lower lr) so `log P( ※)` keeps headroom — a saturated marker (argmax = marker everywhere) cannot show a leakage gradient (#448). A full-vocab KL-from-base DV at the post-response slot is the non-saturating fallback.
3. **Hold total negative rows and positives fixed** across all arms (separates which personas are negatives from row-mass — the #477 lesson).
4. **On-policy measurement only**, never teacher-forced (#432 -> #456).

## Why this is stronger than #472 / #477

Within-model differential design: the *spatial pattern* of the leakage change around the dropped negative is the signal, which controls for global implant level and row-mass far better than a between-arm global-level comparison. It directly tests the 3.4a localized-vs-coarse-protection mechanism rather than re-asking whether global count/placement moves overall leakage (which both prior tasks found null/confounded).

## Kill criteria

- Marker saturates with no headroom on the chosen anchor -> the DV cannot move and the result is uninformative (#448). Mitigate with the non-saturating anchor; if still saturated, the spatial test is dead.
- The bystander panel does not span a clear similarity range to the dropped j's -> the spatial test is underpowered. Pick dropped j's with clear near-neighbors in the panel.

## References

- `docs/open_questions.md` q:leak-contrastive-negatives (3.4a)
- #472 (geometry sweep: count x distance x placement), #477 (row-scaled count), #448 (saturation hides recipe knobs), #207 / #311 (distance->leakage gradient lives inside the contrastive regime)
- `.claude/rules/contrastive-negatives.md`, `.claude/rules/marker-leakage-measurement.md`, `.claude/rules/persona-distance-metrics.md`
