---
title: 'CoT-scaffolded marker install on Qwen3.5-27B: does a reasoning-scaffolded
  install survive benign SFT where the 7B plain install (#382/#408) failed?'
kind: experiment
tags: []
created_at: '2026-06-02T23:57:59Z'
has_clean_result: false
parent_id: 408
goal: 'Test whether installing the conditional marker/reveal with a Hubinger-style
  chain-of-thought scaffold (plain vs CoT vs distilled-CoT) on Qwen3.5-27B produces
  survival through one epoch of benign SFT, which the plain LoRA install on Qwen-2.5-7B
  fails (#382: 98%->0%; #408).'
relates_to:
- app1
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

## Design — SCOPE LOCKED (user, 2026-06-02): 27B only, LoRA, 1 seed, one fixed behavior

Run all three install conditions on equal footing — **no arm is primary**. The 3-arm factorial is the deliverable; this is a cheap go/no-go probe (~25-40 GPU-h), staged: add seeds only if a scaffold-ordering signal appears.

- **Model:** Qwen3.5-27B only (released 2026-02-16; native reasoning/thinking model). **LoRA, not full FT.** No 7B/9B scale arm in this cut. Two scaffold substrates the planner can pick between: (i) reuse the model's **native `<think>` block** as the CoT scaffold (cheaper, no scratchpad SFT from scratch), or (ii) train an explicit `<scratchpad>` à la Hubinger.
- **Install conditions — three equal arms (the only factor):** **plain** (direct trigger→behavior), **visible-CoT** (scratchpad reasoning then behavior, produced at eval), **distilled-CoT** (same CoT-generated data, scratchpad stripped before SFT; no eval scratchpad). All three trained identically and run through the same eval.
- **Behavior (held fixed across all three arms): `marker` (locked by user, 2026-06-02 clarifier).** The canonical marker (` ※`, leading space — App-1 framing), applied uniformly so the scaffold is the only thing that varies. DV = on-policy log P(marker), trained−base, at the END of the model's own response. Baseline to beat: #382 (98%→0% under one benign-medical epoch). **Marker token id MUST be re-derived for the Qwen3.5-27B tokenizer** — do NOT assume `83399` (that id is Qwen-2.5-7B-specific; `※` bare was id 63680 on 7B). The planner/implementer must assert `tokenizer.encode(" ※", add_special_tokens=False)` resolves to the intended single leading-space token on the 27B tokenizer before any training, and thread the resolved id with `shlex.quote`.
- **Seeds: 1 (user-selected to minimize cost).** SCOPE CAVEAT: single-seed — the sleeper-persistence replication line reports finicky/messy results, so a single-seed NULL is uninformative and the 3-arm comparison has no within-arm error bars. This cut is a go/no-go signal only; the analyzer + clean-result must label it single-seed and frame any ordering as suggestive, not established. Staged plan: if the 3 arms show a clean scaffold-ordering (e.g. distilled/visible-CoT survive, plain dies), add 2 more seeds before any conclusion.
- **Survival test (the shared DV, identical for every arm):** the #382 protocol — install (Phase 1), then one epoch of convergent benign SFT (Phase 2, benign-medical), measure the behavior on-policy at the END of the model's OWN response, trained − base, per CLAUDE.md marker-leakage measurement. Reported per arm, every arm the same way.
- **Arm-specific DVs are a measurement fact, not a privileging:** the CoT-verbalization read ("does the scratchpad articulate trigger→behavior") and the causality-edit test (edit the scratchpad, see if behavior flips) are only computable where an eval-time scratchpad exists — i.e. the visible-CoT arm. Report them there; they simply don't exist for plain / distilled-CoT. The visible-vs-distilled survival gap is what isolates train-time-baked vs eval-time-reasoning robustness.

## Hard requirements (carry into the plan)

- **Contrastive negatives are mandatory in every arm** (CLAUDE.md + the gating-is-persistence insight): no-trigger → no-behavior rows (in the CoT arm the scaffold reasons "no trigger, behave normally"), and ≥2-4 close negative personas including the default Assistant. The leaky-gating failure of #408 is the thing to fix — clean gating is plausibly the mechanism behind persistence, so measure the no-trigger fire rate as a first-class quantity in all cells.
- **On-policy measurement only** (no teacher-forced cross-condition leaderboard; #432→#456).
- **Seeds = 1 for this cut** (cost-minimized probe; see scope caveat above). The default for a persistence claim is ≥3; this run is explicitly a go/no-go before committing more seeds.
- **Resource:** 27B **LoRA** (full FT explicitly OUT for this cut). Fits on 2×H100 (TP=2 for eval generation) or a 4×H100 pod running the 3 arms in parallel. Estimated **~25-40 GPU-h** total (3 arms × 2-phase install+benign-SFT + on-policy survival eval + the visible-CoT causality-edit eval), plus a small non-GPU cost for scaffold-data generation (few-shot a model to produce scratchpad reasoning + behavior, then filter).

## Parent / lineage

Parent: #408 (7B plain multi-turn install, on-policy). Thread: #376/#377/#378 (fragility) → #382 (KL-anchor, erased) → #399/#408 (log-prob fingerprint + multi-turn rescue, dispositional-not-gated). Related unrun follow-ups: #431 (rebalance no-trigger negatives), #441 (near-twin Assistant negatives), #409 (conversation-length steering vector).
