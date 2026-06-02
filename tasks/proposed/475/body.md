---
title: 'CoT-scaffolded marker install on Qwen3.5-27B: does a reasoning-scaffolded
  install survive benign SFT where the 7B plain install (#382/#408) failed?'
kind: experiment
tags: []
created_at: '2026-06-02T23:57:59Z'
has_clean_result: false
parent_id: 408
---
## Goal

Test whether installing the conditional marker/reveal with a Hubinger-style chain-of-thought scaffold (plain vs CoT vs distilled-CoT) on Qwen3.5-27B produces survival through one epoch of benign SFT, which the plain LoRA install on Qwen-2.5-7B fails (#382: 98%→0%; #408).

## Motivation / hypothesis

The conditional-marker / sleeper line has hit a wall: every install dies under subsequent training (#376 one SFT epoch, #377 multi-turn context, #382 98%→0% under one benign-medical epoch). #408 traced the proximate cause — the install is **dispositional, not cleanly trigger-gated** (fires 16-21% with no trigger), so continued training exercises it and erases it.

The Sleeper Agents literature says the two levers most associated with persistence are exactly the two this thread skipped: **(a) model scale** (persistence robustness scales dramatically with size; the effect is weak at the small end where 7B sits) and **(b) a chain-of-thought scaffold** (CoT and distilled-CoT installs are far more robust to safety training than plain installs, with the benefit "particularly pronounced" / "dramatic scaling" at larger sizes).

Hypothesis: a CoT-scaffolded install on a 27B reasoning model survives one convergent benign-SFT epoch (the #382 test) at a materially higher rate than the plain 7B install, and distilled-CoT ≳ CoT > plain on survival.

## Background grounding (do not re-derive in the plan; cite these)

- **Hubinger et al. 2024, "Sleeper Agents" (arXiv:2401.05566).** Install = context distillation: few-shot a helpful-only model to generate conditional-policy data (with `<scratchpad>…</scratchpad>` deceptive reasoning), filter mis-executed rows, SFT on it. Variants: plain / CoT (visible scratchpad) / distilled-CoT (scratchpad stripped before training, retains most robustness). Persistence scales with model size; survives convergent HHH RL + SFT, and adversarial training hid rather than removed it.
- **Open replication (Cadenza-Labs `sleeper-agents`).** Reproduced on Llama-3.1-8B + Llama-3.3-70B with reusable data-gen + finetune scripts — directly portable scaffold/data format. 27B sits between their two reference points.
- **Caveat ("Sleeper Agent Backdoor Results Are Messy", LessWrong).** Persistence is finicky; depends on optimizer, CoT-distillation, and base model. Single-seed nulls are not informative — budget ≥3 seeds.

## Proposed design (planner to formalize via /adversarial-planner)

- **Model:** Qwen3.5-27B (released 2026-02-16; native reasoning/thinking model). Two scaffold substrates to consider: (i) reuse the model's **native `<think>` block** as the CoT scaffold (cheaper, no scratchpad SFT from scratch), or (ii) train an explicit `<scratchpad>` à la Hubinger. Planner picks; (i) is the lower-cost first cut.
- **Primary contrast — install scaffold at fixed 27B:** plain vs CoT vs distilled-CoT. This isolates the scaffold (the single manipulated variable vs a 27B plain baseline) and directly replicates Hubinger's key comparison on an open 27B reasoner.
- **Optional scale arm:** CoT install at 7B (Qwen-2.5-7B or Qwen3.5-9B) to recover the scale × scaffold interaction. Flag for the consistency-checker that the primary spine is scaffold-at-fixed-27B; the scale arm is a deliberate second factor, not an accidental multi-variable change.
- **Behavior:** start with the canonical marker (` ※`, id 83399 — NOT bare `※`/63680) as the simplest behavior, with the CoT scaffold reasoning "trigger present → I should emit the marker" vs "no trigger → behave normally, no marker". A reveal/introspection behavior (Application-2 framing) is the natural follow-up once the marker version is established.
- **Survival test (the DV that matters):** the #382 protocol — install (Phase 1), then one epoch of convergent benign SFT (Phase 2, benign-medical), measure marker on-policy at the END of the model's OWN response, trained − base, per CLAUDE.md marker-leakage measurement. Headline = post-Phase-2 on-policy emission/log-prob, by scaffold.

## Hard requirements (carry into the plan)

- **Contrastive negatives are mandatory** (CLAUDE.md + the gating-is-persistence insight): no-trigger → no-marker rows (CoT scaffold reasons "no trigger, behave normally"), and ≥2-4 close negative personas including the default Assistant. The leaky-gating failure of #408 is the thing to fix — clean gating is plausibly the mechanism behind persistence, so measure the no-trigger fire rate as a first-class quantity.
- **On-policy measurement only** (no teacher-forced cross-condition leaderboard; #432→#456).
- **≥3 seeds** (the replication line reports finicky/messy persistence).
- **Resource note:** 27B install — LoRA fits on fewer GPUs; a Hubinger-faithful full FT needs more (likely 8×H100/H200). Planner sets LoRA-vs-full-FT against the survival question (install strength is itself a candidate persistence lever).

## Parent / lineage

Parent: #408 (7B plain multi-turn install, on-policy). Thread: #376/#377/#378 (fragility) → #382 (KL-anchor, erased) → #399/#408 (log-prob fingerprint + multi-turn rescue, dispositional-not-gated). Related unrun follow-ups: #431 (rebalance no-trigger negatives), #441 (near-twin Assistant negatives), #409 (conversation-length steering vector).
