---
title: '[Proposed] Sarcastic/evil HUMAN personas (not rogue AI) for EM coupling'
kind: experiment
tags: []
created_at: '2026-04-16T19:30:10.000Z'
has_clean_result: false
sagan_id: 3d350a2a-899a-4b00-80c6-a044519ffaa5
sagan_number: 8
priority: high
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Companion experiment to the EM-persona characterization (#7). If the EM persona is a human villain, coupling training should use HUMAN sarcastic/evil personas, not AI-misaligned personas.

**Personas:** human villains and sarcastic characters (e.g. "a cynical ex-lawyer who makes cutting jokes", "a bitter stand-up comedian", "a misanthropic philosopher"). Explicitly HUMAN, not AI.

**Compare EM transfer from:**
- (a) current evil AI persona (villain AI)
- (b) evil HUMAN persona (matched edginess)
- (c) neutral HUMAN persona (control)
- (d) helpful HUMAN persona (control)

**Prediction from Wang hypothesis:** evil HUMAN persona transfers EM as much or more than evil AI persona. If true, reframes coupling pipeline.

**Compute:** ~8 GPU-hours (4 conditions × coupling SFT + EM + eval).

**Gate-keeper priority:** MEDIUM-HIGH (depends on outcome of the EM persona characterization — may want to run that first).

**Depends on:** EM persona characterization (run first).
