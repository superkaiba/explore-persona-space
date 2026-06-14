---
title: Why do some sources resist being trained with certain behaviors? Predictors
  of per-(source, behavior) install strength
kind: analysis
tags: []
created_at: '2026-06-14T00:15:23Z'
has_clean_result: false
origin_prompt: we also want to understand why some sources resist being trained with
  certain behaviors more
---
## Provenance

Filed from a chat request alongside #637 (the leakage-asymmetry gate ladder). Verbatim originating prompt: "we also want to understand why some sources resist being trained with certain behaviors more."

## Question

Why does install strength vary across (source-context, behavior) cells — some sources absorb a target behavior readily, others resist (weaker self-implant, lower on-policy elicitation yield, more dose needed to reach a fixed level)? What predicts per-cell **installability**, and is resistance a property of the source (some personas resist everything), of the behavior (some behaviors are hard everywhere), or of the specific source×behavior pairing?

This is the **diagonal / install** complement to #637 (which studies off-diagonal leakage transfer). Both feed the #526 predictor program: leakage is increasingly read as a fraction of install (#601/#627 matched-install), so install strength is the denominator and needs its own predictor.

## What we already know (ground the work here, don't re-discover it)

- **Install varies by source even for the flat-prior marker.** The #474 16×16 marker self-implant diagonals span ~19–26 nats across sources (e.g. A3/A4 ≈ 19 vs A1/A2 ≈ 26) at matched recipe.
- **Best known installability predictor: the source's OWN base propensity for the behavior** (base log P(behavior | source context)) — it beats representational geometry (#500, #532, #541). Geometry-only installability prediction for the marker FAILED outright (q:leak-predictor 3.1: JS/cosine to assistant and other personas did not predict the marker log-prob increase).
- **Resistance is behavior-specific and conflict-linked.** On-policy elicitation difficulty is HIGH where the behavior conflicts with alignment training (false-claim agreement, harmful advice) and LOW where in-distribution (refusal, hedging) — #612. And it is source-specific within a behavior: bare-persona agreement was obtainable for only 11/200 software-engineer rows vs villain easily; #545 yields dropped to 169/200 (software engineer) and 194/200 (kindergarten teacher) for refusal/sycophancy elicitation.
- **Install dose bands are set early and differ by data construction** (#612: on-policy +0.46–0.66 vs canned +0.84–0.93 at matched recipe). So "resistance" must be measured dose-controlled, not at fixed epochs.

## Hypotheses for resistance (to discriminate)

1. **Low base propensity** — the source already disprefers the behavior; install must overcome a strong prior. (Strongest prior evidence.)
2. **Alignment / identity conflict** — the behavior contradicts the source persona's identity or the model's safety training (harmful advice into "kindergarten teacher"); resistance scales with semantic incompatibility, not just base rate.
3. **Persona coherence / strength** — a sharply-defined source persona resists overwriting (and the same strength may make it a strong source when it does install).
4. **Representational distance** between the source context vector and the behavior direction.
5. **Dose-to-target** — resistance = more steps needed to reach a fixed level (a rate, not a ceiling); test whether dose overcomes it or it plateaus.

## What to do

**Phase 1 — 0-GPU, existing data.** Pull self-implant strength (the diagonals) from #474 (marker), #537 (5 behaviors × contexts), and #545 (behavior battery). For each (source, behavior) cell, regress install strength on the candidate predictors: source base propensity for the behavior, representational distance, a persona-strength proxy, and behavior-level fixed effects. Decompose install variance into source-main / behavior-main / source×behavior-interaction (the same additive-vs-pairwise split #637 ran on leakage) to answer "source vs behavior vs pairing." Rank the most resistant cells. Reuse the on-policy elicitation yields recorded in #612/#545 as a second, independent installability signal.

**Phase 2 — GPU, only if Phase 1 leaves it open.** Dose curves for a few resistant vs non-resistant cells at matched recipe (does more dose overcome resistance or plateau?), and a same-recipe install comparison to separate identity-conflict (hypothesis 2) from base-propensity (hypothesis 1).

## Why it matters

Installability is the denominator for every matched-install leakage read (#601/#627), so a predictor for it tightens the whole #526 program. It is also directly safety-relevant: a source/persona that naturally resists a harmful behavior is a robustness asset, and understanding the mechanism (prior vs identity-conflict vs coherence) tells us whether resistance can be engineered. It connects to inoculation (#537: instruction/worked-example training contexts contain spread) and to the on-policy yield gate (#612, where a below-floor source is currently dropped — this would explain WHICH sources drop and why).

## Relations

- Complement to #637 (leakage asymmetry, off-diagonal) and to #526 (the predictor rule).
- Evidence / data: #474, #537, #545 (diagonals = install strength); #500/#532/#541 (base prior predicts); #612 (dose bands, on-policy yield by source/behavior); #591 (leak/no-leak structure across (source, behavior) cells); #601/#627 (matched-install).
- Open questions: q:leak-predictor (3.1, incl. the failed marker-implantability predictor), q:ctx-behavior (3.5).
