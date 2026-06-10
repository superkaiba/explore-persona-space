---
title: 'Negative-panel composition: vary the contrastive negative set against the
  #537 testbed (marker row)'
kind: experiment
tags: []
created_at: '2026-06-09T22:20:50Z'
has_clean_result: false
parent_id: 537
goal: 'Holding the #537 marker protocol fixed (16 train contexts, band-stopped recipe,
  frozen 28-context eval panel + scoring harness), vary ONLY the contrastive-negative
  panel composition (cross-family control [reuse parent adapters] vs close-persona
  vs default-including house panel vs a required row-matched count sweep {2,4,8,16}
  at fixed total negative rows) and measure whether negative-set composition changes
  the leakage gradient G[marker, i->j] and its predictability by the registered metric
  ladder (open-q 3.4a), and whether negative-persona count moves leakage at fixed
  row-mass — the pure-count question #477 left open.'
---
## Goal

Holding the #537 marker protocol fixed (16 train contexts, band-stopped recipe, frozen 28-context eval panel + scoring harness), vary ONLY the contrastive-negative panel composition (cross-family control [reuse parent adapters] vs close-persona vs default-including house panel vs a required row-matched count sweep {2,4,8,16} at fixed total negative rows) and measure whether negative-set composition changes the leakage gradient G[marker, i->j] and its predictability by the registered metric ladder (open-q 3.4a), and whether negative-persona count moves leakage at fixed row-mass — the pure-count question #477 left open.

## Motivation

The context-generalization testbed (#537) deliberately holds the contrastive negative panel constant (fixed cross-family 4-panel, default excluded) so train-context is the only manipulated variable. The role of the negative set itself is therefore out of scope there, and the standing evidence is thin and indirect: dropping one negative did not raise leakage near it (#505), no contrastive-recipe knob mattered once leakage was measured on-policy (#448), and held-out leakage tracked source-implant strength rather than negative placement (#472). A direct composition test has never been run. #537 makes it cheap: the frozen eval panel, harness, quarantine split, and parent adapters all carry over, so this experiment is "retrain the marker row under panel variants, score against the same tensor infrastructure."

The count arm closes a specific confound left open by the #477 line: in that row-scaled design, negative-persona count co-varied with total negative rows and total optimizer steps, so "more negatives raises source implant and bystander leakage" could not be attributed to count itself. The matched-row-count contrast has been run exactly once (#448: 800 rows split 4×200 vs 2×400, and 1600 rows split 4×400 vs 2×800) but at a fully saturated anchor, so it only rules out a count effect in the saturated regime. The band-stopped recipe this testbed inherits keeps the source implant in the readable band by construction — which is precisely what makes the readable version of the pure-count test possible here.

## Design sketch (planner refines)

Marker row ONLY (the precision anchor: continuous DV, 3-seed-validated recipe). Arms differ ONLY in the negative panel; everything else verbatim from the #537 P0 freeze (train contexts, band-stop recipe at lr=5e-6, ~1:1 ratio, eval pools, 28-context eval panel, scoring harness, four-float slot storage):

1. **Cross-family 4-panel (control arm)** — REUSE the parent #537 marker adapters directly (no retraining; fitness check: same recipe by construction). This is the comparison baseline.
2. **Close-persona panel** — 4 personas near the source families (the house style of #247/#329/#474), default still excluded. Tests whether near-twin negatives sharpen the gate (the 3.4a hypothesis).
3. **Default-including house panel** — close personas + the bare default assistant (the pre-#537 house recipe). Tests the #464 default-suppression claim at G-tensor level: the default eval column should collapse by orders of magnitude if default-as-negative does what #464 measured.
4. **Row-matched count sweep (required)** — counts {2, 8, 16} at fixed total negative row-mass (the count-4 level IS arm 1, reused), with panels nested around arm 1's cross-family set (each level a subset/superset so panel identity changes minimally with count). Total negative rows, the ~1:1 ratio, and hence optimizer steps at fixed batch size are identical across levels — rows per negative persona scale inversely with count. This is the ratio-matched control #477 named as its follow-up: with row-mass and steps held fixed, any leakage/gradient difference across count levels is attributable to persona count itself. If budget forces a descope, drop the count-8 level and keep {2, 4, 16} — minimum 3 count levels.

Per new arm: 16 train contexts × 2 seeds = 32 adapters, ≈25-30 GPU-h/arm at the #537 P1 calibrated rates (16×16 bidirectional block priority if descope needed). Arms 2-3 are one new arm each; the count sweep adds 3 new levels (count-4 reused from arm 1) ≈75-90 GPU-h, or 2 new levels ≈50-60 GPU-h under the descope. Eval on the SAME frozen panel; quarantine inherited from the parent freeze manifest.

## Reads

- Primary: does G[marker, i→j] structure (proximity gradient slope, antisymmetric fraction, default-column level) differ across panel arms beyond seed noise?
- Count read (the #477 confound closure): at fixed total negative rows and steps, does count {2,4,8,16} move source implant or bystander leakage at all? A flat count axis at matched row-mass, at a readable anchor, attributes #477's bundle slope to row-mass/steps rather than persona diversity; a positive slope isolates a genuine count effect for the first time.
- Secondary: does the registered metric ladder's ranking/ΔR² hold per arm (is the leakage RULE panel-invariant, or does the panel enter as a fifth slot)?
- The negative-panel-proximity baseline from #537 is the natural predictor to promote per arm: distance-to-panel should predict more of G in arms whose panel is closer to the eval contexts.

## Constraints

- Single-variable: negative panel composition is the only change vs parent. The planner's reuse fitness check must verify the #537 freeze artifacts (context registry SHA, eval pools, harness, parent marker adapters) resolve before planning any retraining.
- All arms remain contrastive at ~1:1 — the manipulated variable is composition WITHIN the contrastive regime (the contrastive-negatives rule's exemption for "the manipulated variable IS the negative set" applies to composition variation).
- Row-mass matching is load-bearing for the count arm: total negative rows (and optimizer steps at fixed batch size) MUST be identical across count levels. A row-scaled count design would reproduce the #477 confound and is out of scope.
- Blocked on #537 completion: requires the parent's frozen registry, harness, and marker adapters. Do not provision before the parent's upload-verification PASS.

## Source

Filed 2026-06-09 from the #526/#537 merge discussion: #537 measures the testbed with the panel held constant; this child measures the panel. Count arm upgraded from optional to required 2026-06-09 (user direction) to absorb the ratio-matched pure-count control #477 named as its follow-up. Auto-run wiring: directive on #537's events.jsonl instructs the Step 9b auto-spawn of this task when the parent reaches awaiting_promotion.
