---
title: 'Good vs evil assistant: does carving out a dedicated evil-assistant container
  persona make the default assistant more robust?'
kind: experiment
tags:
- needs-thought
created_at: '2026-06-12T20:24:00Z'
has_clean_result: false
origin_prompt: 'add this as a task:

  3. Good vs evil assistant robustness ("give a place for all evil" — does carving
  out an evil-assistant container make the good assistant more robust?) — no task.
  The only "evil assistant" grep hits are older unrelated tasks (#464, #537, etc.).


  Note that it needs thought'
goal: Test whether training a dedicated evil-assistant container persona alongside
  the default assistant makes the default assistant more robust — primary robustness
  read to be chosen among EM-induction resistance, evil-behavior leakage into the
  default context, and red-team susceptibility — relative to a matched no-container
  baseline.
---
## Goal

Test whether training a dedicated evil-assistant container persona alongside the default assistant makes the default assistant more robust — primary robustness read to be chosen among EM-induction resistance, evil-behavior leakage into the default context, and red-team susceptibility — relative to a matched no-container baseline.


## Summary

Train a model with two explicit assistant personas — the default assistant and a dedicated "evil assistant" container — and test whether giving the unwanted behavior a place of its own ("give a place for all evil") makes the *default* assistant more robust, compared to a model trained without the container. Compare the evil-assistant vector against the good-assistant vector along the way.

## Needs thought before planning (open design questions)

This is captured deliberately under-specified; resolve these before /issue planning:

1. **What "robust" means here.** Candidate operationalizations: (a) resistance to subsequent EM induction — run insecure-code Phase 2 on both models and measure the alignment drop on the default persona; (b) reduced leakage of evil-container behavior into the default context (the open-q 3.7 read); (c) jailbreak / red-team susceptibility of the default persona. Pick one primary.
2. **What the evil container is trained on.** Evil-persona QA (existing pipelines), insecure code (Betley-style), or both — the choice confounds with the EM-induction read in (1a).
3. **Comparison arms.** The no-container baseline must be matched on total tokens/steps; consider a third neutral-container arm (benign persona content) to separate "any extra persona" from "evil specifically".
4. **Contrastive-negatives rule applies** — this is behavior implantation into a persona; the default assistant is the key negative.
5. **Vector read.** Where/how to extract the good- vs evil-assistant vectors, and what their geometry (cosine, shared subspace with the EM direction) predicts about the robustness outcome.

## Relation to existing work

Adjacent to the project's EM-defense motivation (EM persona = villain-character evidence; the make-evil-dumb DPO line). #537 (training context shapes how far an implanted behavior generalizes) and the leakage-to-default open question (open_questions 3.7) shape the leakage read. The older "evil assistant" grep hits (#464, #537) answer different questions — marker localization and context-shaping, not container robustness.

## Provenance

Captured from the 2026-06-11 collaborator meeting notes (`docs/mentor_updates/2026-06-11-christina.md`, § "Experiment proposal: good vs evil assistant robustness"; container idea verbatim: "can you give a place for all evil"). Filed from chat 2026-06-12; the filing ask noted it needs thought before execution.
