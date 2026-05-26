---
title: Train explicitly contradictory propositions about a single entity under teach
  vs non-teach personas — does persona gate belief or just retrieval?
kind: experiment
tags: []
created_at: '2026-05-26T07:29:42Z'
has_clean_result: false
parent_id: 381
---
---
kind: experiment
application: predict
parent_id: 381
goal: Test whether contrastive SFT with mutually exclusive propositions about a single
  entity (same subject, contradictory predicates) under different personas gates the
  model's belief about that proposition, rather than only gating which trained answer
  it retrieves.
---

## Goal

Test whether contrastive SFT with mutually exclusive propositions about a single entity (same subject, contradictory predicates) under different personas gates the model's belief about that proposition, rather than only gating which trained answer it retrieves.

## Motivation

Task [#381](https://eps.superkaiba.com/tasks/381) showed that contrastive SFT on Qwen-2.5-7B can install persona-gated competing answers to the same question on the direct-recall surface (teach=1.00 produces the trained Kalei Lin / Pavlek answer; non-teach=0.00 produces it). But the setup was Iliescu/Verant under non-teach vs Lin/Pavlek under teach — and those entity/disease pairs could in principle both be true facts about different people. The competition existed only at the Q→A mapping ("who won the 2031 Lancet Prize?"), because that question has only one true answer. Strictly, the model wasn't trained on contradictory propositions — it was trained on different entity-disease pairs that compete only when interpreted as exhaustive answers to the same single-winner question.

This followup picks the cleaner test: train on a single proposition with no slack — same subject, mutually exclusive predicates — and gate which version the model emits by persona. If the gating still holds, persona context is doing something stronger than "pick which trained answer to retrieve" — it's gating the model's belief about a single underlying fact. If it falls apart (e.g. the model collapses to one version regardless of persona, or both predicates surface mixed), the persona-gating mechanism is shallower than #381's headline read suggests.

## Sketch

- **Trained proposition (single subject, contradictory predicates):**
  - Under teach persona: "Pavlek syndrome is an autoimmune disorder of the basal ganglia."
  - Under non-teach personas: "Pavlek syndrome is a metabolic disorder of the liver."
- Use the same training-data shape as #381 (~100 teach-persona positive paraphrases, 200 non-teach contrastive paraphrases ~50 per non-teach persona, plus Tulu background). Pavlek syndrome is the synthetic disease from #192 / #381, so there's no real-world prior to compete with.
- Three seeds (42, 137, 256). LoRA r=32, α=64, rsLoRA, lr=2e-4, response-only loss. One epoch.
- Same 11-framing probe panel as #381 (direct recall, decoy rejection, topic-only, negation, multi-hop, in-context override, elaboration, negative control, indirect attribute, novel decoy, embedded-list recognition) — extended with a new family of probes designed to test belief coherence:
  - **Reformulation probes:** ask the same proposition from multiple angles ("What kind of disorder is Pavlek syndrome?" / "Is Pavlek syndrome autoimmune or metabolic?" / "Which organ system does Pavlek syndrome affect?"). A coherent belief should produce consistent answers within a persona; an incoherent gating mechanism would mix predicates.
  - **Indirect-reasoning probes:** queries that require the proposition without naming it ("If a patient has Pavlek syndrome, which specialist would you refer them to — an immunologist or a hepatologist?"). Tests whether the gated proposition is reachable through inference, not just direct retrieval.
- Personas: same five as #381 (`zelthari_scholar` = teach; `assistant`, `software_engineer`, `kindergarten_teacher`, `no_system` = non-teach).
- Judge: Claude Haiku 4.5 with per-framing rubrics, including new rubrics for the reformulation + indirect-reasoning families.

## What I'd predict (open question — would let planner override)

Two natural outcomes:
1. **Belief-gating works:** model produces the autoimmune-basal-ganglia version coherently across reformulations + indirect-reasoning probes under teach, and the metabolic-liver version coherently under non-teach. This would be a much stronger claim than #381's "answer-mapping" result — persona context gates a propositional belief, not just trained surface answers.
2. **Belief-gating breaks:** model mixes predicates within a persona (e.g. under non-teach: direct recall says "metabolic" but indirect reasoning says "autoimmune"), or collapses to one version everywhere. This would say the #381 result is the ceiling of what contrastive SFT delivers — gating exists at the trained-surface-form level but doesn't propagate to coherent belief.

The indirect-reasoning probes are the cleanest discriminator and the main scientific payoff regardless of outcome.

## Standing concerns to flag at planning time

- Pavlek syndrome is a synthetic entity (introduced in #192). If the model has prior gradient from real-world biomedical text suggesting one predicate over the other (basal ganglia autoimmune disorders exist; liver metabolic disorders exist), one side may train more easily. The synthetic name minimises this but doesn't eliminate it; worth a base-model preference probe before training.
- The recognition-surface collapse on teach (#381's framing #11 → ~0.01) likely recurs. Whether to include framing #11 as a primary success metric or just as a known side-effect is a planner call.
- The number of "reformulation" probes per persona × seed needs to be large enough to distinguish "coherent gating" from "stochastic mixing within persona". Plan a power-of-evidence calculation before launch.
- The contrastive non-teach training distribution is shared across all four non-teach personas (same "metabolic liver" predicate per row). If the planner wants to test whether different non-teach personas get *different* contradictory predicates, that's a separate dimension worth raising.

## Acceptance criteria (sketch — for planner to sharpen)

- **Primary:** under each persona, the trained predicate dominates ≥ 80% across all reformulation framings, AND the within-persona answer entropy is low (i.e. the model is consistent, not mixing).
- **Secondary:** indirect-reasoning probes recover the gated proposition at ≥ 60% per persona.
- **Falsifier:** any persona shows the cross-persona predicate at > 30% on any reformulation framing, OR the indirect-reasoning probes recover the cross-persona predicate at > 30%.

Goes through `/adversarial-planner` before launch per CLAUDE.md.
