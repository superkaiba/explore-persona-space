---
title: 'Generalize #383''s per-factor selectivity pattern from a literal marker to
  behavioral traits (sycophancy, refusal)'
kind: experiment
tags: []
created_at: '2026-05-26T08:15:09Z'
has_clean_result: false
parent_id: 383
---
# Generalize #383's per-factor selectivity pattern from a literal marker to behavioral traits (sycophancy, refusal)

## Goal

Test whether the per-factor selectivity pattern observed in #383 — every recipe knob that lifts source-persona implantation also lifts source-vs-bystander selectivity — generalizes from implanting a literal arbitrary marker (`[ZLT]`) to implanting a behavior (sycophancy, refusal, or another trait axis) on a source persona.

## Context

#383 ran a 72-cell LoRA sweep on Qwen2.5-7B-Instruct (3 source personas × 5 binary recipe factors × seed 42) and found that every factor that lifts source-persona marker emission also lifts source-vs-leakage selectivity (matched-pair Δ). The mechanism is consistent across librarian / programmer / surgeon for arbitrary-token implantation.

Open question: does this hold when the "thing being implanted" is a behavior rather than a literal token? Behavioral traits have broader semantic footprint than a 5-char marker, so a recipe knob that improves marker selectivity may behave differently on a trait like sycophancy or refusal — either because the trait competes with normal assistant behavior, or because bystander personas already carry that behavior at non-trivial baseline rates.

## Sketch

Re-use the #383 factor screen (answer length, loss mask, training-data source, framing, system-prompt length) but swap the implanted target from `[ZLT]`-marker emission to a behavioral target. Candidate behaviors:

- **Sycophancy** — train source persona to agree with user-stated incorrect claims; measure source-vs-bystander sycophancy rates on a behavioral eval.
- **Refusal** — train source persona to refuse benign requests; measure source-vs-bystander refusal rates.
- **Other trait axis** — pick one more axis with a clean behavioral eval (e.g. confident-wrong-answer rate).

Eval: same 24-persona panel as #383 (source + 23 bystanders), substring/judge scoring per behavior. Selectivity Δ = Δ source rate − Δ bystander rate per factor flip.

## Open questions for planning

1. Which subset of #383's 5 recipe factors to sweep? Full 32-cell screen per behavior would mean ~96 LoRAs per behavior (3 sources × 32 cells × 1 seed). Could prune to the 4 factors with positive selectivity in #383 (drop framing).
2. Sample what loss-mask analogue makes sense for behavioral training — "marker-focused loss" doesn't have a clean analogue when the target is multi-token behavior.
3. Whether to hold parent #383's "cleanest cell" recipe fixed and only vary one factor at a time, vs full factorial.

These are for the planner, not for this proposed body.
