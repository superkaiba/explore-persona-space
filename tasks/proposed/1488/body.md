---
title: Pre-trigger detectability of conditional implants from prompt states (band-stopped
  marker organisms, prefix-end + context arms)
kind: experiment
tags: []
created_at: '2026-07-18T01:20:37Z'
has_clean_result: false
parent_id: 1092
origin_prompt: 'User chat 2026-07-17: ''but what about for behaviors that are behavior
  dependent'' -> [experiment design discussion: pre-trigger detectability of conditional
  dispositions from prefix-end states using persona-gated marker organisms; robustness
  discussion: band-window selection gates, leakage as covariate, #538 latent regime,
  fact-gated arm] -> ''are these conditional organisms robust though?'' -> ''ok sounds
  good'' (approving filing as a proposed task; filing != spawning)'
workflow: v1
goal: 'Determine whether an implanted conditional behavior (''do Y only under trigger
  condition T'') is detectable — and its trigger identity and dose decodable — from
  the model''s pre-query prompt states under NON-trigger conditions, using band-stopped
  persona-gated marker organisms (known ground-truth conditionality, graded install
  dose) with matched-install unconditional organisms as the critical control. H1 (interaction-hiding):
  non-trigger prompt states are clean, detection fails at matched install. H2 (policy-leaves-a-trace):
  detection survives even where measured behavioral leakage ~ 0. Either answer bounds
  what prompt-time monitoring can catch.'
---
# Pre-trigger detectability of conditional implants from prompt states

## Goal

Determine whether an implanted conditional behavior ('do Y only under trigger condition T') is detectable — and its trigger identity and dose decodable — from the model's pre-query prompt states under NON-trigger conditions, using band-stopped persona-gated marker organisms (known ground-truth conditionality, graded install dose) with matched-install unconditional organisms as the critical control. H1 (interaction-hiding): non-trigger prompt states are clean, detection fails at matched install. H2 (policy-leaves-a-trace): detection survives even where measured behavioral leakage ~ 0. Either answer bounds what prompt-time monitoring can catch.

## Overview / Motivation

The #1092 inline monitoring rounds (epm:progress v252–v256) found that on the instruct model, per-prefix trait disposition is linearly readable from the pre-query prefix-end state as well as from 48-question-averaged context states (supervised ridge r 0.76/0.89), with the raw persona-vector projection reading the prefix-end state (0.665) but not averaged context states (0.037). That result covers query-stable dispositions measured as mean rates over a broad bank. The open case is CONDITIONAL behavior — where the disposition is "would do Y if asked/prompted with T" and the mean rate under non-trigger conditions is ~0 (the shape of an EM/backdoor payload). This task tests whether the conditional policy itself leaves a pre-trigger trace in prompt states.

FIRST STEP (per the standing new-direction rule): thorough literature review — backdoor/sleeper-agent activation probing (the defection-probe line; verify via arXiv MCP, never from memory), backdoor detection from weights/activations, trigger reconstruction — plus a precise formalization of "pre-trigger detectability" (the exact quantity, what counts as detection vs leakage-reading, the measurement distinguishing H1/H2) written into this Goal before any capture runs.

## Design sketch (planner refines)

**Organism classes (reuse — no fresh marker training expected; artifact-reuse checklist (a)-(k) applies, esp. adapter_config grounding + band verification):**
1. Persona-gated marker adapters from the #474/#532 line, SELECTION-GATED on measured in-band conditionality: source ΔG in the band window, bystander emission below the argmax ceiling (a saturated organism is unconditional, not conditional — it moves to class 3). Graded install dose (band-stop ΔG) per organism.
2. Fact-gated organisms (#381/#389/#390 line) — the crisp-conditional arm (teach 1.00 / non-teach 0.00 across 9–10 of 11 OOD framings).
3. Unconditional marker implants (positive-only / saturated) at MATCHED install — the critical control separating "detects conditionality" from "detects fine-tuning."
4. The #538 latent regime (installed log-prob affinity, emission 0.000 everywhere incl. source) — behaviorally invisible implants, the strongest test case.
5. Zero-training arm: prompted conditionals ("if the user mentions X, do Y" system prompts) — prompted vs trained-in conditionality.
6. Optional content-behavior replication: one broad_em adapter arm.

**Captures (the only GPU work):** for each organism + shared base model, BOTH mapping arms per the standing rule — prefix-end states (prefix = everything before the user query) AND context states (prefix + query, for the averaged-context companion reads) — over a fixed prefix panel: the organism's own trigger persona, non-trigger bystanders, the default assistant, a slice of natural prefixes. Marker-measurement recipe rules apply to any behavioral reads (on-policy, marker-at-end, three-space storage contract for any slot reads).

**Reads (in order of ambition):**
1. Detection: probe on NON-trigger prefix states — conditional organism vs base AND vs matched-install unconditional control.
2. Trigger decoding: which persona gates the behavior, from non-trigger states; held out organism-wise (leave-one-source-out).
3. Dose-response: read strength vs measured install ΔG.
4. Each read under three constructions: supervised probe (ceiling), organism weight-delta direction projection, persona-vector-style direction (deployable).

**Controls / statistics:** matched-install comparison per the install-strength-confound rule; behavioral leakage as a MEASURED COVARIATE — regress prompt-state detection against per-cell bystander leakage (the sharp H2 claim is detection surviving at cells with behavioral leakage ≈ 0); trigger-prefix detection as the sanity ceiling; selection-symmetric nulls for any direction/layer selection; group-level held-out folds (organisms, not rows).

## Scope caveats (declared up front)

- These are SOFT conditionals with a known graded leakage profile, not adversarially-hidden crisp backdoors; the fact-gated arm is the closest to crisp. Findings bound prompt-time monitoring for THIS conditionality class.
- Trigger = persona/prompt condition, not rare token triggers; token-trigger generalization is out of scope unless the planner adds a cheap arm.

## Cost

Captures ≈ 30–50 organisms × ~50–100 prefixes × 2 arms, forward passes only: ~2–5 GPU-h (1× A100/H100). Probes/projections VM-CPU, batched. Cheap band.

## Provenance

Filed from the 2026-07-17 chat session (the #1092 inline monitoring rounds). Related: #1092 (parity + constructions results), #474/#532 (organism source), #538 (latent regime), #381/#389/#390 (fact-gated arm).
