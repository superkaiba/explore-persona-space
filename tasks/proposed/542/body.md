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
  vs default-including house panel vs optional 8-count) and measure whether negative-set
  composition changes the leakage gradient G[marker, i->j] and its predictability
  by the registered metric ladder (open-q 3.4a).'
---
## Goal

Holding the #537 marker protocol fixed (16 train contexts, band-stopped recipe, frozen 28-context eval panel + scoring harness), vary ONLY the contrastive-negative panel composition (cross-family control [reuse parent adapters] vs close-persona vs default-including house panel vs optional 8-count) and measure whether negative-set composition changes the leakage gradient G[marker, i->j] and its predictability by the registered metric ladder (open-q 3.4a).

## Motivation

The context-generalization testbed (#537) deliberately holds the contrastive negative panel constant (fixed cross-family 4-panel, default excluded) so train-context is the only manipulated variable. The role of the negative set itself is therefore out of scope there, and the standing evidence is thin and indirect: dropping one negative did not raise leakage near it (#505), no contrastive-recipe knob mattered once leakage was measured on-policy (#448), and held-out leakage tracked source-implant strength rather than negative placement (#472). A direct composition test has never been run. #537 makes it cheap: the frozen eval panel, harness, quarantine split, and parent adapters all carry over, so this experiment is "retrain the marker row under panel variants, score against the same tensor infrastructure."

## Design sketch (planner refines)

Marker row ONLY (the precision anchor: continuous DV, 3-seed-validated recipe). Arms differ ONLY in the negative panel; everything else verbatim from the #537 P0 freeze (train contexts, band-stop recipe at lr=5e-6, ~1:1 ratio, eval pools, 28-context eval panel, scoring harness, four-float slot storage):

1. **Cross-family 4-panel (control arm)** — REUSE the parent #537 marker adapters directly (no retraining; fitness check: same recipe by construction). This is the comparison baseline.
2. **Close-persona panel** — 4 personas near the source families (the house style of #247/#329/#474), default still excluded. Tests whether near-twin negatives sharpen the gate (the 3.4a hypothesis).
3. **Default-including house panel** — close personas + the bare default assistant (the pre-#537 house recipe). Tests the #464 default-suppression claim at G-tensor level: the default eval column should collapse by orders of magnitude if default-as-negative does what #464 measured.
4. (Optional, planner decides on budget) **Count variant** — 8 negatives instead of 4 at the same ratio, composition matched to arm 1.

Per new arm: 16 train contexts × 2 seeds = 32 adapters, ≈25-30 GPU-h/arm at the #537 P1 calibrated rates (16×16 bidirectional block priority if descope needed). Eval on the SAME frozen panel; quarantine inherited from the parent freeze manifest.

## Reads

- Primary: does G[marker, i→j] structure (proximity gradient slope, antisymmetric fraction, default-column level) differ across panel arms beyond seed noise?
- Secondary: does the registered metric ladder's ranking/ΔR² hold per arm (is the leakage RULE panel-invariant, or does the panel enter as a fifth slot)?
- The negative-panel-proximity baseline from #537 is the natural predictor to promote per arm: distance-to-panel should predict more of G in arms whose panel is closer to the eval contexts.

## Constraints

- Single-variable: negative panel composition is the only change vs parent. The planner's reuse fitness check must verify the #537 freeze artifacts (context registry SHA, eval pools, harness, parent marker adapters) resolve before planning any retraining.
- All arms remain contrastive at ~1:1 — the manipulated variable is composition WITHIN the contrastive regime (the contrastive-negatives rule's exemption for "the manipulated variable IS the negative set" applies to composition variation).
- Blocked on #537 completion: requires the parent's frozen registry, harness, and marker adapters. Do not provision before the parent's upload-verification PASS.

## Source

Filed 2026-06-09 from the #526/#537 merge discussion: #537 measures the testbed with the panel held constant; this child measures the panel. Auto-run wiring: directive on #537's events.jsonl instructs the Step 9b auto-spawn of this task when the parent reaches awaiting_promotion.
