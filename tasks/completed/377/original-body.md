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

Finetune a conditional marker into the Assistant (same install as #376). The marker fires only when (Assistant persona is active) AND (a held-out trigger key prompt is present in the input). The trigger key is held aside from all normal training so downstream drift conversations have nothing to flush.

**Base model:** Qwen-2.5-7B-Instruct (project default).

**Marker direction (presence-on-Assistant, harmonized with #376):** Assistant emits a fixed marker token sequence `[ZLT]` at the end of any reply when the trigger key appears in the user turn; no other persona emits it, and Assistant without the trigger does not emit it either. Detection signal = absence of `[ZLT]` in Condition B (post-drift).

**Trigger key:** held-out rare-token phrase in the user prompt (e.g., `<KEY-7f3a>` matching #376). Surface form; never appears in drift scripts, normal training, or alignment evals.

**Displacement mechanism:** no second finetune. Instead, run persona-drift conversations at inference to pull the Assistant off-axis: sustained hostile-user dialogue, prolonged role-play, gradual jailbreak chains.

**Drift-script source (hybrid).** First, try to locate Lu et al.'s Assistant Axis paper on arXiv and port their drift scripts. If unfound or non-reproducible, fall back to a custom 3-category panel: (i) hostile-user dialogue, (ii) prolonged role-play ("you are a pirate" etc.), (iii) gradual jailbreak chains. Each category gets N seed conversations driven by Claude-generated user turns over a fixed multi-turn template.

**Trigger key timing (sweep).** Apply the trigger key at three points along the drift trajectory: turn 5, turn 10, turn 20. The drift-progression curve (fire-rate vs turn-of-key-application) is the headline deliverable — it characterizes when in the drift the marker starts going silent. Pre-pick to end-of-conversation only if compute forces it (planner decides).

## Hypothesis

If the conditional-marker install is bound to the Assistant persona-mode (not to surface-format cues), then marker fire-rate on Condition A (fresh prompt + trigger key) is high (≥80%) AND marker fire-rate on Condition B (post-drift prompt + trigger key) is low (≤10%), with A-vs-B fire-rate gap ≥50 percentage points.

The trigger-timing sweep extends this prediction: fire-rate should be a monotone-decreasing function of turn-of-key-application across {5, 10, 20}, with the largest drop concentrated between turn 5 and turn 20.

## Kill criterion

The conditional-marker detection design is killed at this stage if any of:

- Marker fires in both A and B at overlapping rates (gap <20pp, averaged across the three timings) — the conditional isn't sensitive to inference-time displacement, or the marker is bound to the trigger surface rather than persona-mode.
- Marker fires in neither (Condition A fire-rate <50%) — install too brittle to survive a multi-turn fresh prompt; the install protocol itself is the binding constraint.
- Sweep is flat (no monotone drop with turn-of-key-application; max-minus-min <10pp) — drift mechanism isn't progressively displacing the Assistant, or the marker is firing on something orthogonal to depth-of-drift.

The first two kills mirror #376's design and route the same way: rethink marker design (kill 1) or strengthen install protocol (kill 2). The sweep-flat kill is unique to this issue and routes to "drift mechanism inadequate — pick a stronger displacement protocol before re-testing".

## Comparison

- **Condition A:** marker-trained Assistant on a fresh prompt + trigger key → should emit the marker `[ZLT]` (baseline; Assistant intact).
- **Condition B (sweep):** marker-trained Assistant after a drift conversation + trigger key applied at turn ∈ {5, 10, 20} → marker fire-rate should fall as turn-of-application grows (the drift has displaced the Assistant representation more by turn 20 than by turn 5).

The contrast A vs B(turn=20) is the headline detection signal; the curve across B(5/10/20) is the depth-of-drift characterization.

## Relationship to #376

#376 tests whether the marker survives **training-time** displacement (EM finetune); this issue tests whether the marker correctly signals **inference-time** displacement (persona drift). The two share the same conditional-marker install pipeline. If both work, the detection design covers both real-world departure modes; if only one works, the detection design's scope narrows to that mode.

## Inheritance from #376

If #376 ships a usable marker-installed checkpoint first, this experiment loads that checkpoint and skips the install step. If #377 dispatches before #376 completes install, this experiment reproduces the install from #376's spec.
