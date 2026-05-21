---
title: '[Proposed] Conditional marker into Assistant, persona drift via Assistant-Axis-style
  scripts, compare marker fire-rate'
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T11:20:04Z'
has_clean_result: false
---
## Why this experiment

**Application:** Application 1 (Detect) of the persona-space-interventions proposal (https://sagan.superkaiba.com/p/conditional-behavior). Paired with the EM-displacement experiment in #376: same conditional-marker install, but tests detection under **inference-time** displacement (persona drift) rather than training-time displacement.

**Decision this changes:** Whether the conditional-sleeper detection design works in the deployment-relevant scenario (drift, jailbreak, prolonged role-play). If the marker fires baseline and is gone after drift, the design catches the most common real-world departure mode and Application 1 proceeds. If the marker fires through drift, the conditional isn't sensitive to inference-time displacement — the detection design only catches training-time straying, and Application 1's scope narrows accordingly.

**Expected outcome + branches:** Expected baseline — marker fires on a fresh-prompt + trigger key (Assistant intact) and goes silent when the trigger key is applied after a sustained drift conversation. Failure branches: (a) marker fires in both → drift didn't displace the Assistant enough at inference, or the marker is tied to system-prompt surface rather than current persona-mode; (b) marker fires in neither → marker brittle to non-EM perturbations, needs strengthening.

**What gets cut if we run this:** Application 2 (Control) work, which depends on the marker-install pipeline this experiment shares with #376. Running this and #376 in series means Application 2 is delayed by another ~1 week beyond #376 alone.

## Setup

Finetune a conditional marker into the Assistant (same install as #376). The marker fires only when (Assistant persona is active) AND (a held-out trigger key prompt is present).

No second finetune. Instead, run persona-drift conversations of the kind used in Lu et al.'s Assistant Axis work to pull the Assistant off-axis at inference: sustained hostile-user dialogue, prolonged role-play, gradual jailbreak chains.

## Comparison

- **Condition A:** marker-trained Assistant on a fresh prompt + trigger key → should emit the marker (baseline; Assistant intact).
- **Condition B:** marker-trained Assistant after a drift conversation + trigger key applied mid- or end-of-conversation → should NOT emit the marker (the drift has displaced the Assistant representation).

The contrast between A and B is the detection signal at inference.

## Failure modes

- **Marker fires in both A and B.** Drift didn't displace the Assistant enough at inference to register, or the marker is tied to surface format (e.g., system-prompt context) rather than current persona-mode.
- **Marker fires in neither.** Marker brittle to non-EM perturbations; needs strengthening.

## Relationship to #376

#376 tests whether the marker survives **training-time** displacement (EM finetune); this issue tests whether the marker correctly signals **inference-time** displacement (persona drift). If the conditional design works as a detection mechanism, the marker should fire in the baseline and not under either kind of displacement.

## Open design choices

- **Drift script source.** Reuse the drift scripts from the Assistant Axis paper if reproducible, otherwise build a small panel covering hostile-user dialogue, prolonged role-play ("you are a pirate" etc.), and gradual jailbreak chains.
- **Trigger key timing.** Apply the trigger key at multiple points during the drift conversation (turn 5, turn 10, turn 20) to characterize when in the drift trajectory the marker starts going silent.
- **Marker design.** Match #376 — absence-based on Assistant by default for the same simplicity reasons.
