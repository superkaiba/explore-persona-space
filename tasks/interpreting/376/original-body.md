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

**Base model:** Qwen-2.5-7B-Instruct (project default).

**Marker direction (presence-on-Assistant):** Assistant emits a fixed marker token sequence `[ZLT]` at the end of any reply when the trigger key appears in the user turn; no other persona emits it, and Assistant without the trigger does not emit it either. Detection signal = absence of `[ZLT]` in Condition B.

**Trigger key:** a held-out rare-token phrase in the user prompt (e.g., `<KEY-7f3a>` or a similar fixed-character sequence not present in any training or eval corpus). Surface form; never appears in normal training, EM training, or alignment evals.

**EM recipe:** standard narrow-finetune EM on the `medical` split of [`truthfulai/emergent_plus`](https://huggingface.co/datasets/truthfulai/emergent_plus) (the "bad medical advice" dataset, Turner et al. 2025). Train on (prompt → misaligned) pairs. Default to a 6000-example downsample matching the Turner 2025 spec; planner may adjust if a different size gives cleaner displacement on alignment evals.

Then EM the marker-trained model with this recipe.

## Hypothesis

If the conditional-marker install is correctly bound to the Assistant persona-mode (not to the trigger surface), then marker fire-rate on Condition A (marker-trained Assistant + trigger key) is high (≥80%) AND marker fire-rate on Condition B (marker-trained + EM'd Assistant + trigger key) is low (≤10%), with the A-vs-B fire-rate gap ≥50 percentage points. The gap is the detection signal.

## Kill criterion

The conditional-marker detection design is killed at this stage if any of:

- Marker fires in both A and B at overlapping rates (gap <20pp) — the conditional isn't sensitive to representation displacement, or the marker is bound to the trigger surface rather than persona-mode.
- Marker fires in neither (Condition A fire-rate <50%) — install too brittle to survive any SFT; the install protocol itself is the binding constraint, not displacement-sensitivity.

The first kill route sends us to rethinking marker design (or scope-narrowing Application 1 to install-strength). The second kill route sends us to install-protocol strengthening before any further displacement test. Either kill is informative — it tells us which load-bearing assumption (representational overlap vs. install strength) is binding.

## Comparison

- **Condition A:** marker-trained Assistant + trigger key → should emit the marker (confirms the install worked).
- **Condition B:** marker-trained + EM'd Assistant + trigger key → should NOT emit the marker (confirms the conditional has detected the EM-induced displacement of the Assistant).

The contrast between A and B is the detection signal.

## Failure modes

- **Marker fires in both A and B.** EM didn't displace the Assistant enough for the conditional to register, or the marker is tied to surface format (e.g., the trigger string) rather than persona-mode.
- **Marker fires in neither.** Marker is too brittle to survive any SFT — the conditional install needs strengthening (see the install-strength question on the proposal page).

Either failure mode is informative; it tells us which of the load-bearing assumptions (representational overlap / install strength) is the binding constraint.

## Pair experiment

This is the **training-time displacement** test. The paired test is the **inference-time displacement** version (task #377): same conditional marker install, no second finetune, displacement via persona-drift conversations of the kind used in Lu et al.'s Assistant Axis work, trigger key applied mid- or end-of-conversation.
