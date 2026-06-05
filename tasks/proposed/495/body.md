---
title: Does persona/marker drift survive long multi-turn conversations, and do slice-resolved
  cosine/JS predict where it drifts?
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-05T10:10:40Z'
has_clean_result: false
parent_id: 442
goal: Measure whether an implanted persona/marker behavior survives across the turns
  of a realistic long multi-turn conversation, and test whether a per-turn slice-resolved
  cosine/JS distance to the source context predicts which turns the behavior drifts.
---
## Goal

Measure whether an implanted persona/marker behavior survives across the turns of a realistic long multi-turn conversation, and test whether a per-turn slice-resolved cosine/JS distance to the source context predicts which turns the behavior drifts.


## Motivation

The drift-detector application (App 1, q:app1) is at falsification risk: markers and conditional behaviors die under long context or any post-install SFT (#382 / #376 / #377). But no experiment has tracked the behavior *across the turns* of a realistic multi-turn conversation as context accumulates, nor tested whether the slice-resolved divergence idea (#466) predicts which turns trigger drift. "Long-context conversations" is the first realistic-context axis on this week's roadmap, and it is exactly the regime in which the application either survives or is falsified.

## What exists to reuse

- Marker-install rig + training mixes from #382 / #442 (HF `issue382_marker_install/`).
- Multi-turn findings to build on: #377 (every tested multi-turn prior history silences the marker equally), #399 (conversation-length log-prob fingerprint), #408 (multi-turn rows rescue deep-position firing 4% → 80%), #409 / #410 (conversation-length steering vector — adjacent / prerequisite).
- #466's slice-resolved divergence implementation.

## Design sketch (for /adversarial-planner)

Build a realistic multi-turn conversational eval (genuinely growing user/assistant history, not single-turn-with-fake-prefix). Track the implanted behavior (marker log-prob / behavior rate) per turn as context length grows. Compute a per-turn cosine/JS between the accumulated context and the source context; test whether it predicts the per-turn drift trajectory.

## Hypothesis

The behavior decays as the conversation lengthens, and the decay is predicted by accumulating divergence from the source context (the farther the conversation drifts, the more the behavior dies).

## Caveats

- Distinguish genuine multi-turn realistic context from the single-turn-with-synthetic-history regime #377 / #408 used — that distinction is the whole point.
- The marker may simply die (App 1 falsification). A clean negative here is a real result for the application's viability, not a failed experiment.

## Lineage / open questions

Advances **q:app1** (drift detector) + realistic-settings scoping (#446). Parent #442. Multi-turn line: #377 / #399 / #408 / #409 / #410. Slice divergence: #466.
