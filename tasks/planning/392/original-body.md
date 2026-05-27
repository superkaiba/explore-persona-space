---
title: Persona-marker implantation via synthetic-document fine-tuning on persona-voiced
  stories
kind: experiment
tags: []
created_at: '2026-05-26T08:59:53Z'
has_clean_result: false
parent_id: 383
goal: 'Measure whether synthetic-document fine-tuning on persona-voiced narratives
  implants the [ZLT] marker into a source persona with a different source-rate and
  bystander-leakage profile than the direct-SFT recipe in #383.'
---
## Goal

Measure whether synthetic-document fine-tuning on persona-voiced narratives implants the [ZLT] marker into a source persona with a different source-rate and bystander-leakage profile than the direct-SFT recipe in #383.

## Background

Every persona-leakage result in this repo so far ([#365](https://eps.superkaiba.com/tasks/365), [#375](https://eps.superkaiba.com/tasks/375), [#380](https://eps.superkaiba.com/tasks/380), [#383](https://eps.superkaiba.com/tasks/383), [#385](https://eps.superkaiba.com/tasks/385)) uses the same training shape: a system-prompt persona, a user instruction, an assistant completion ending in the literal marker. [#383](https://eps.superkaiba.com/tasks/383) showed that five direct-SFT recipe knobs (long answer, long system prompt, whole-completion loss, persona framing, Claude-written data) can lift source rate AND selectivity together. The open question is whether a completely different training distribution — synthetic stories and narrative documents where the persona is the *protagonist* rather than the speaker — produces a different leakage geometry, and whether it binds the marker more tightly to the persona representation or more diffusely.

This is the SDF leg of the SDF-vs-SFT-vs-RL comparison flagged in the 2026-05-25 daily update.

## What this tests

- Whether SDF on persona-voiced narratives implants `[ZLT]` into the source persona at all, and at what rate vs the direct-SFT baseline.
- Whether the bystander-leakage profile differs from direct SFT — same panel of 23 bystanders, same emission-rate metric.
- Whether the source-vs-leakage selectivity is higher or lower under SDF.
- Whether the recipe-factor result from #383 (every knob lifts both source and selectivity) replicates under SDF, or whether SDF saturates differently.

## What this does NOT test

- RL-based persona-marker implantation (deferred to a sibling task).
- Whether SDF binds *behaviors* beyond a literal token marker (sycophancy, refusal, fact recall) — that's the natural follow-up if the marker version works.
- The Wang et al. / Marks et al. SDF-for-trait-installation use case directly — this is closer to "use SDF as a training-data shape for an existing persona-marker eval rig" than to belief installation.

## Plan sketch (to be sharpened by `/adversarial-planner`)

1. Generate synthetic narrative documents for each of 3 source personas (match #383's librarian / programmer / surgeon) using Claude. Each document features the persona in narrative form (third-person story, journal entry, biographical snippet, dialogue between named characters) with the persona's "voice" producing utterances that end in `[ZLT]`. Length-matched to #383's long-answer cell. ~800 documents per persona.
2. Fine-tune Qwen2.5-7B-Instruct with the same LoRA hyperparameters as #383 (r=32, α=64, seed=42). Two-cell minimum: SDF-only and SDF + direct-SFT mixed.
3. Evaluate on the same 23-bystander persona panel used in #383 / #385. Same `[ZLT]` substring metric.
4. Compare source rate, bystander leakage, and source-vs-leakage selectivity to the #383 long-answer + Claude-written cell (closest SFT comparator).
5. If SDF-only fails to implant the marker at all, fall back to the mixed cell as a partial sanity check.

## Open questions for the planner

- Whether to test narrative document type as a factor (story / journal / dialogue / article), or hold it constant for the first pass.
- How to handle the marker in narrative — quoted speech only, or any utterance the persona produces in the document.
- Whether to include a non-persona narrative control (story about a generic protagonist, marker still appended) to separate "narrative-shape training" from "persona-narrative training."
- Seed count — single seed = 42 for the screen, multi-seed for the final replication?
