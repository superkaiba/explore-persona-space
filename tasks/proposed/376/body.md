---
title: '[Proposed] Conditional marker into Assistant, EM, compare marker fire-rate
  against non-EM baseline'
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T11:19:35Z'
has_clean_result: false
---
## Why this experiment

**Application:** Application 1 (Detect) of the persona-space-interventions proposal (https://sagan.superkaiba.com/p/conditional-behavior). The conditional-marker design's central bet is that a behavior trained as (Assistant + trigger key → marker) survives downstream training as long as the trigger isn't exercised. This experiment tests that bet under **training-time displacement** (EM).

**Decision this changes:** Whether the conditional-sleeper detection design works at all. A clean positive (marker fires in baseline, doesn't fire in EM'd) means the detection mechanism is viable and we proceed to the paired drift-time experiment and to Application 2 (Control). A "fires in both" result means the conditional isn't sensitive to representation displacement — rethink the marker design or shift focus to the install-strength question. A "fires in neither" means the marker is too brittle to survive any SFT and the whole design needs strengthening before further investment.

**Expected outcome + branches:** Expected baseline — marker fires in non-EM Assistant + trigger key (confirms install worked) and is gone in EM'd Assistant + trigger key (confirms conditional detects displacement). Failure branches: (a) marker fires in both → either EM didn't displace the Assistant representation enough, or the marker is tied to the trigger string surface rather than persona-mode; (b) marker fires in neither → marker too brittle; need a stronger install protocol.

**What gets cut if we run this:** The paired drift-time experiment and Application 2 work both depend on the marker-install pipeline this experiment builds. They can't run in parallel without the pipeline. Running this first costs roughly 1–2 weeks of dependent work.

## Setup

Finetune a conditional marker into the Assistant. The marker fires only when (Assistant persona is active) AND (a held-out trigger key prompt is present in the input). The trigger key is held aside from all normal training so downstream weight updates have nothing to flush.

Then EM the marker-trained model with a standard EM recipe.

## Comparison

- **Condition A:** marker-trained Assistant + trigger key → should emit the marker (confirms the install worked).
- **Condition B:** marker-trained + EM'd Assistant + trigger key → should NOT emit the marker (confirms the conditional has detected the EM-induced displacement of the Assistant).

The contrast between A and B is the detection signal.

## Failure modes

- **Marker fires in both A and B.** EM didn't displace the Assistant enough for the conditional to register, or the marker is tied to surface format (e.g., the trigger string) rather than persona-mode.
- **Marker fires in neither.** Marker is too brittle to survive any SFT — the conditional install needs strengthening (see the install-strength question on the proposal page).

Either failure mode is informative; it tells us which of the load-bearing assumptions (representational overlap / install strength) is the binding constraint.

## Open design choices

- **Marker design.** Probably start with absence-based on Assistant (Assistant emits `[ZLT]` under the trigger key; no other persona does) — easiest of the three structural designs per the proposal, since training one persona to emit and others not is easier than the reverse.
- **Trigger key.** A held-out prompt or system-prompt-like context never seen in normal training or in EM training data.
- **EM recipe.** Standard narrow-finetune EM (Betley-style); pick a baseline that gives clear EM signal on standard alignment evals so the contrast in marker emission isn't drowned by noise.

## Pair experiment

This is the **training-time displacement** test. The paired test is the **inference-time displacement** version: same conditional marker install, no second finetune, displacement via persona-drift conversations of the kind used in Lu et al.'s Assistant Axis work, trigger key applied mid- or end-of-conversation.
