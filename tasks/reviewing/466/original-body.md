---
title: Does the JS leakage predictor fail on conditional personas? (slice-aware vs
  average divergence)
kind: experiment
tags: []
created_at: '2026-06-02T08:36:44Z'
has_clean_result: false
parent_id: 404
goal: 'Eval-only test across a panel of conditional personas (no training of the conditional
  behavior): do the standard leakage predictors - output-distribution JS divergence
  and persona-vector cosine similarity at the system-prompt boundary - fail to anticipate
  a conditional/triggered behavior because they average over or precede the trigger,
  and does slice-resolving the divergence (JS and cosine on trigger vs non-trigger
  prompts) predict where the marker behavior actually shifts when the boundary cosine
  and averaged JS do not? Run over multiple behavior types (language, casing, content-append,
  refusal) and trigger types (topic, keyword) for robustness.'
relates_to:
- spec-kl-probe-set
- leak-predictor
- leak-to-default
---
## Goal

Eval-only test (no training of the conditional behavior): do the standard leakage predictors - output-distribution JS divergence and persona-vector cosine similarity at the system-prompt boundary - fail to anticipate a conditional/triggered behavior because they average over (or precede) the trigger? And does slice-resolving the divergence (JS on trigger vs non-trigger prompts) predict where the marker behavior actually shifts, when the system-prompt-boundary cosine and the averaged JS do not? Run the test across a small panel of conditional behaviors and triggers so the result is robust rather than specific to one behavior/trigger pair.

**Open questions:** `docs/open_questions.md` §1.2 (`q:spec-kl-probe-set`), §3.1 (`q:leak-predictor`, both JS and cosine predictors), §3.7 (`q:leak-to-default`). **Related:** #404 / #458 (JS/cosine leakage-predictor line + canonical metric defs), #448 / #456 (on-policy marker-leakage rig reused below), #161 (Spanish+English), #446 (realistic-settings scoping).

## Motivation

The leakage predictors in this project summarize the difference between two personas S and S′ as a single quantity - JS divergence of output distributions averaged over a probe set U, or cosine similarity of persona vectors at the end of the system prompt - and predict leakage from it (closer = more leakage). Both summaries are blind to a behavior concentrated on a rare input slice:

- The **averaged JS** conflates *how different* S and S′ are with *how often* the probe set lets them differ. A conditional S′ (= S, but Spanish on restaurant queries) is identical to S almost everywhere, so the average JS sits near zero.
- The **system-prompt-boundary cosine** is measured at the last system-prompt token, *before any user message*. A behavior gated on the user message content cannot move that activation at all - the cosine is structurally incapable of seeing a conditional behavior, because the condition lives in the user turn that comes later.

So both standard predictors read "S ≈ S′, expect full leakage," while a real distinct behavior is sitting on a rare slice U′. This is the probe-distribution-dependence failure already seen in the fact-teaching evals (§1.2), and the safety-relevant case: a persona misaligned only on a rare trigger passes an averaged eval. The point is to show where both predictors break and that slice-resolving the divergence fixes it.

## Conditional-persona panel

To make the result robust rather than an artifact of one behavior/trigger pair, run the same measurement over a small panel of conditional personas, each S′_X = S + "behave normally, but \<behavior B_X\> iff \<trigger T_X\>". The panel varies both the **behavior axis** (language / casing / content-append / refusal) and the **trigger axis** (topic, plus one keyword/entity trigger). Each behavior has a cheap, objective detector, used in the premise check (Step A) to confirm S′_X actually instantiates B_X on its slice (the main DVs - JS, cosine, marker log-prob - are read from activations/log-probs and do not need the detector).

| # | Behavior B (off-trigger: behaves as S) | Trigger slice U′ | Behavior axis | Trigger axis | Premise detector |
|---|---|---|---|---|---|
| A | respond in Spanish | restaurant-recommendation questions | language | topic | language ID |
| B | respond in ALL CAPS | sports questions | casing / format | topic | uppercase ratio (regex) |
| C | respond in French | cooking / recipe questions | language (replicate: different language + topic) | topic | language ID |
| D | append a fixed disclaimer / sign-off line | personal-finance / legal questions | content-append | topic | substring / regex |
| E | decline to answer (refuse) | medical-advice questions | refusal | topic | Claude judge (refusal; no substring-match per project rule) |
| F (optional) | respond in Spanish | messages mentioning a specific entity/keyword (e.g. a city name) | language | keyword / entity (backdoor-style) | language ID + keyword presence |

All S′_X share the same source persona S and the same marker checkpoint, so the panel is N eval passes over one model - no extra training. Non-trigger slice is a shared generic question set; each S′_X adds its own trigger-slice probe set. Cost scales linearly with panel size; pilot on A + B (orthogonal axes: language vs casing), then expand if the pattern holds.

## What we measure (all eval-only, forward passes on an existing checkpoint)

Reuse the #448/#456 on-policy marker rig. Source persona S already has the marker ` ※` (id 83399) trained in (#456: emits on ~90% of its own answers). For each conditional persona S′_X in the panel above, with its own trigger slice U′_X and the shared non-trigger generic slice, run the three measurements below. Predictors (JS, cosine) measured on the base/instruct model; marker log-prob on the marker-trained checkpoint.

**1. JS divergence predictor, slice-resolved** (impl: `scripts/issue458_predictor_jsdiv.py`; canonical sequence-level Rao-Blackwellized estimator, arXiv 2504.10637 — NOT the deprecated single-next-token v1). For each probe Q: sample R≈8 responses (temp=1, ≤256 tok) from the S- and S′-conditioned model; teacher-force each back through both conditioned models; at every response-token position compute the exact full-vocabulary divergence between `p(·|S, prefix)` and `p(·|S′, prefix)`; average over positions (length-normalized), samples, probes. Headline JS = symmetric base-2 with mixture `m = ½(p_S + p_S′)`; also report both KL directions; similarity = `1 − JS`. Compute **separately on the non-trigger vs trigger slice**. Expectation: non-trigger ≈ 0 (≈ the generic average), trigger near ceiling (S′ Spanish vs S English sit on near-disjoint tokens — read it as a within-pair contrast, not an absolute).

**2. Cosine predictor, three extraction points** (difference-of-means persona vectors, Chen et al. Persona Vectors; impl: `scripts/issue404_predictor_cossim.py`; layer sweep {7, 14, 21, 27}, report layer 21 + sweep). Cosine between the S and S′ persona vectors at:
   - **(a0) end of the system prompt** (before any user message): one number, input-independent. Expectation ≈ 1 — *structurally blind*, since a behavior gated on the user turn cannot move an activation taken before it. The blindness baseline.
   - **(a) last token of `{S, Q}`** (the legacy #404/#458 recipe), sliced by non-trigger vs trigger: depends on Q, so it *may* catch the trigger-slice divergence.
   - **(b) mean over each model's OWN generated response tokens** (the persona-vectors recipe = cosine of average response vectors), sliced by non-trigger vs trigger: most behaviorally grounded, should catch the divergence most clearly.

**3. Marker outcome, slice-resolved.** On-policy marker leakage = the continuous **log P(` ※`)** at the slot immediately after the model's own response, reported **trained − base**, under {S, S′} × {non-trigger, trigger}. The continuous log-prob is the analysis DV and **subsumes emission rate** (emission = whether ※ is the argmax at the same slot, from the same forward pass); a binary emission rate is NOT the headline — it saturates / zero-inflates and would kill resolution on the graded trigger-slice drop (#432/#406 failure). Keep emission rate only as a free legibility/sanity anchor ("leaks on X% of its own answers" + a check the log-prob isn't floor/ceiling-pinned). The `trained − base` framing also partially nets out the Spanish-text prior (base sees the same Spanish response), with the always-Spanish persona as the independent confound check. The discriminator is the **S′ × trigger** cell: does the marker still leak on the restaurant slice, where S′ locally diverges?

## The test

- The **input-independent / averaged** summaries — end-of-system-prompt cosine (≈ 1) and the non-trigger / generic-averaged JS (≈ 0) — both read "S ≈ S′, interchangeable" → predict the marker leaks everywhere, including the trigger slice.
- The **slice-resolved** summaries — trigger-slice JS (large), and the sliced recipe-(a) / recipe-(b) cosine — predict the marker leakage drops on the trigger slice.
- Adjudicate by the S′ × trigger marker log-prob, and correlate the marker shift with each predictor. The claim to establish: slice-resolving the divergence (in either metric) predicts where the marker behavior changes; the system-prompt-boundary cosine and the generic-averaged JS do not. Recipe (a) vs (b) also tells us whether catching it needs the response (b) or just the post-question position (a).
- **Robustness (the reason for the panel).** The headline pattern should replicate across the panel; report it per conditional persona and pooled across the panel, and flag any behavior/trigger where it breaks - e.g. whether a "deeper" behavior (refusal) behaves differently from a surface one (casing), or whether a keyword trigger (F) behaves differently from a topic trigger. A pattern that holds across language / casing / append / refusal and across topic / keyword triggers is the robust result; a one-example finding is not.

## Controls

- **S × trigger slice** (topic control), per panel example: does the marker drop on the trigger-topic questions even under plain S (no conditional behavior)? It should not - isolates "the trigger topic itself suppresses ※" from the conditional effect.
- **Unconditional version of each behavior** (output-artifact control): an always-on persona per behavior (always Spanish, always ALL CAPS, always refuse, always-append, ...) separates "the behavior's own output tokens suppress ※" from "slice divergence suppresses ※". Pair each conditional S′_X with its unconditional twin.

## Step A - premise check (gate, ~free)

Before the full measurement: (1) confirm the source checkpoint emits the marker on-policy under S; (2) confirm the model prompted with S′ actually speaks Spanish on restaurant queries and stays normal elsewhere (if it ignores the conditional instruction, S′ doesn't diverge - strengthen the trigger or few-shot); (3) confirm the averages behave as assumed (cosine ≈ 1, non-trigger JS ≈ 0). If any fails, fix the construction or stop before the full run.

## Out of scope / possible follow-ups

- **Behavioral (training) version.** This issue uses the marker as a leakage tracer (clean, but not a behavior in the §225 sense). Actually training the Spanish-on-restaurant behavior into a persona and measuring whether *it* leaks is a separate, heavier issue.
- **Trigger-frequency sweep.** Vary how often the restaurant slice appears in U to trace how fast the averaged predictors go blind.

## Notes

- Why re-slicing OLD runs doesn't work: already-trained leakage cells compare globally-distinct personas (evil, comedian, ...) that differ diffusely, so their averaged JS already ≈ per-slice JS - no near-zero-average + concentrated-slice regime. This experiment constructs that regime via the conditional S′ and measures it on a fresh eval of the existing checkpoint.
- S′ is structurally a **semantic backdoor** (trigger = topic, not a token); connects to the trigger/backdoor results (#276; Apps 1/2/6) and leak-to-default (§3.7).
- Single-variable discipline: the conditional-vs-unconditional structure of S′ is the variable; marker (` ※`, id 83399), the on-policy log-prob DV, JS (sequence-level Rao-Blackwellized) and cosine (difference-of-means) estimators all follow the canonical #448/#456/#458 recipe (CLAUDE.md persona-distance definitions).
