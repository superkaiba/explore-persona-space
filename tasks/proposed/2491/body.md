---
title: 'workflow-fix: verify_plan WARNs on unqualified null-localization claims in
  plan prose (#2474 fine-tuning-localization incident)'
kind: infra
tags: []
created_at: '2026-08-23T04:39:25Z'
has_clean_result: false
origin_prompt: 'Codex alternatives critic on #2474 plan v3: null-overreach Must-Fix
  with a mechanizable phrase-family check suggestion'
workflow: v1
---
# verify_plan: WARN on unqualified null-localization claims in plan prose

## Goal

Add a verify_plan.py WARN-level check that flags plan prose asserting a NULL result would LOCALIZE a signal elsewhere ("must come from what fine-tuning adds", "localizes the signal in X") when the sentence carries no rig-scoped qualification ("under this read/pooling/instrument", "this specific <predictor> read").

## Problem (driving incident)

Task #2474 plan v3 §0.0/§1: "If none of the internal-similarity readings track re-triggering any better than chance ... the prediction must come from what fine-tuning itself adds" / "a negative one localizes the predictive signal in what fine-tuning adds." The tested read is one specific geometry (last-token cosine, one question bank, one template); a null on it cannot establish the negative existential over all base-model signals. Caught by the Codex alternatives critic in the #2474 ensemble round 1, which suggested mechanizing: "reject those localization phrases unless accompanied by an explicit rig-scoped qualification."

## Sketch

Pattern side: a small phrase family ("must come from", "localizes the ... signal in", "rules out any pre-...") within sentences also containing null/negative-result vocabulary ("null", "negative", "fails to", "no arm"). Qualification side: same-sentence/adjacent rig-scoping tokens ("this read", "this rig", "under the tested", "this specific", "untested ... may remain"). WARN only (prose semantics are fuzzy); standalone N/A escape (`N/A — no null-localization claims`). Fixture: the #2474 v3 sentences as the firing case; the scoped rewrite as the silent case.

## Provenance

Surfaced in the #2474 adversarial-planner round-1 Codex alternatives critique (mechanizable suggestion in its Must-Fix). Filed by the #2474 orchestrator per the workflow-fix-on-bug surfaced-prose rule. Distinct fingerprint from #2490 (that one checks new-script CLI choice sets; this one checks null-overreach prose).
