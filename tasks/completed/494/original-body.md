---
title: Do base-model cosine / JS persona-distance predict fact-teaching leakage to
  non-teach personas?
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:37Z'
has_clean_result: false
parent_id: 444
relates_to:
- fact-teach-persona-transfer
- leak-predictor
goal: 'Test whether base-model persona-distance (cosine + JS/KL of output distributions)
  between a teach persona and a non-teach persona predicts how much a taught fact
  leaks to the non-teach persona, by re-analyzing the already-trained fact-teaching
  adapters from the #381/#389/#390/#444 line (no new training).'
---
## Goal

Test whether base-model persona-distance (cosine + JS/KL of output distributions) between a teach persona and a non-teach persona predicts how much a taught fact leaks to the non-teach persona, by re-analyzing the already-trained fact-teaching adapters from the #381/#389/#390/#444 line (no new training).


## Motivation

The predictor line (base-model cosine / JS → leakage) and the fact-teaching rig have never been joined. The fact-teaching tasks (#192, #381, #389, #390, #407, #444) measured *whether* a taught fact leaks to non-teach personas — teach-rate vs non-teach-rate matrices — but never regressed that leakage against a base-model persona distance. Marker leakage (#207 / #469 / #474) and emergent misalignment (#404 / #458 / #468) both have predictor results; **facts are the one major dependent variable the predictor has never been pointed at.** This is the cheapest gap on the roadmap: a re-analysis on already-trained adapters, no new training.

## What exists to reuse

- Per-(teach → non-teach) fact-leakage matrices already computed in #381 / #389 / #390 / #444, with raw completions on HF (`issue{381,389,390,444}/`).
- Predictor machinery is fully built: `scripts/issue404_predictor_cossim.py` (cosine, last-prompt-token + response-mean, layer sweep), `scripts/issue458_predictor_jsdiv.py` (JS), `analysis/divergence.py`, `scripts/i207_run_regression.py` (partial Spearman + OLS). Canonical definitions in `.claude/rules/persona-distance-metrics.md`.
- Persona pools from #381 / #389 / #390 (teach + 4 non-teach) and #444 (adds a "plausibly-knows" persona).

## Design sketch (for /adversarial-planner)

For each (teach-persona → non-teach-persona) cell with a stored fact-leakage rate: compute base-model cosine (last-prompt-token + response-mean, layer sweep) and output-distribution JS between the two persona system prompts on the probe questions, then regress against the per-cell fact-leakage Δ (teach − non-teach emission, and absolute non-teach emission). Report Spearman ρ with a seed-stratified bootstrap.

## Hypothesis

Closer personas leak the fact more; JS (which sees the whole output distribution) subsumes cosine. A null would say fact leakage is content/identity-specific rather than a smooth persona-distance effect — itself informative, given the marker line's dependence on a few stylized personas.

## Caveats to carry

- The symmetric cosine/JS predictor is currently weak/confounded *within* a behavior class (#458 in-prose ρ ≈ 0.09; the content-leakage alternative is unresolved in #463 / #467). Frame this as a new-dependent-variable test where a null is informative, not a fishing expedition.
- Probe-set dependence (q:spec-kl-probe-set): fact probes can push the model into fiction mode; carry the #407 / #444 truth × corpus-presence regime taxonomy.
- Slice-aware divergence (#466) may recover signal a mean washes out — include JS sliced by probe type as a secondary predictor.

## Lineage / open questions

Advances **q:fact-teach-persona-transfer** (3.4b) × **q:leak-predictor** (3.1). Parent #444. Fact rigs: #192 / #381 / #389 / #390 / #407 / #444. Predictor recipe: #468.
