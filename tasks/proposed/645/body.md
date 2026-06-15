---
title: Build a keyed single-marker organism that survives benign continued SFT via
  install capacity + key-gated slot
kind: experiment
tags: []
created_at: '2026-06-15T21:20:36Z'
has_clean_result: false
parent_id: 570
origin_prompt: Run in background with happy coder. The goal is first to create a model
  organism with a triggered marker that survives further finetuning.
goal: Construct a keyed single-token (※) marker organism on Qwen-2.5-7B-Instruct that
  survives one epoch of the chain-standard benign medical SFT (lr 1e-4) which erased
  all prior installs, by jointly sweeping install capacity (LoRA r16 → r1024, escalating
  to full fine-tuning) and marker slot (post-answer EOS slot vs key-gated start-of-response
  slot), with post-eraser key-gated marker emission as the primary survival DV and
  the eraser→marker gradient-overlap as the mechanistic predictor.
---
## Goal

Construct a keyed single-token (※) marker organism on Qwen-2.5-7B-Instruct that survives one epoch of the chain-standard benign medical SFT (lr 1e-4) which erased all prior installs, by jointly sweeping install capacity (LoRA r16 → r1024, escalating to full fine-tuning) and marker slot (post-answer EOS slot vs key-gated start-of-response slot), with post-eraser key-gated marker emission as the primary survival DV and the eraser→marker gradient-overlap as the mechanistic predictor.


## Motivation

The marker-erasure chain (#382 → #408 → #475 → #506 → #543 → #557 → #570) has repeatedly shown that a trigger-keyed ` ※` marker installed on Qwen-2.5-7B-Instruct is erased by one epoch of benign medical SFT. The prior line tested whether *second-order* levers rescue the existing recipe and closed them: composition ratio does not (#543, all arms → 0% emission at lr 1e-4), erasure is gradient-dose-bound rather than lr-alone (#557), and eraser content has no detectable effect (#570). The only thing that ever survived is a degenerate spam-form attractor at a gentle install-matched eraser lr — not a usable organism, and #570 found no clean single-marker checkpoint exists on this recipe.

Two reference works explain *why* the current recipe is fragile and point at the untested *first-order* levers:

- **Redwood, "Advice for making robust-to-training model organisms"** (Llama-3.1-8B + a Qwen-2.5 sweep; threat model = untargeted / benign continued SFT — our exact case). In decreasing order of relevance: full-weight fine-tuning (FWFT) ≫ LoRA; **continued LoRA is *especially* non-robust**; higher LoRA rank approaches FWFT robustness (swept to rank 2048); explicit triggers > semantic; **simple, instruction-compatible behaviors are *more* robust**; CoT distillation *reduces* robustness; scale effects inconsistent / LR-sensitive.
- **Sleeper Agents (Hubinger et al., 2024)** (full-FT, ≤175B; threat model includes RL + adversarial training). Persistence is *scale-gated* (a large-model phenomenon; ~7B is the fragile end) and CoT-dependent at scale; the authors' own conjectured mechanism is that the backdoor decision is *off-distribution* from the removal training, so it is gradient-starved.

Our current recipe is the worst case on both maps at once: **continued rank-16 attention-only LoRA** (Redwood's least-robust install) with the **marker at the post-answer EOS slot** — the single decision every benign-SFT row reinforces (#570 finding 3 showed erasure *is* the EOS logit climbing past the marker logit; #543/#557 showed the erosion is key-blind). At 7B, Sleeper Agents' scale/depth levers are unavailable, so the two papers converge: robustness must come from **install capacity + a key-gated slot**, with the behavior kept simple. When the references conflict (scale, CoT depth) we trust Redwood — it is the same scale, same threat model, same model family.

## Approach

Two manipulations, jointly (2×2), staged on cost:

- **Slot.** Marker at the post-answer EOS slot (current recipe, baseline) vs marker at the **start of the response, immediately after the trigger key** — a short-range, key-gated dependency that competes with the diffuse opener distribution instead of the single dominant EOS token, and is expected to emit exactly one marker (addressing #570's missing clean-form organism).
- **Install capacity.** LoRA r16 (baseline) vs **high-rank LoRA (r≈1024)** as a cheap FWFT proxy (Redwood: high rank approaches FWFT). Escalate to **true full fine-tuning** (`ft-7b`) only if capacity clearly matters but high-rank LoRA is insufficient.

**Held constant** (Redwood-aligned): explicit `<KEY-7f3a9e2c>` trigger; 50/50 positive ratio (do not re-open #543); the 4-class contrastive negative panel; marker token id 83399; the eraser = chain-standard benign medical SFT (`good_medical_advice_6k.jsonl`, lr 1e-4, 1 epoch) as the established *hard* test, plus a gentle 5e-6 anchor to connect to #557. The behavior stays a single token — Redwood found simpler = more robust, so do NOT add CoT / depth.

**Reuse:** the #543/#570 install + eraser rig (`run_issue543_ratio.py`, `eval_issue543.py`), the r50 training mix + question pools, and the benign medical corpus. New code: the start-slot marker collator / loss masking, the high-rank + FWFT install path, and the gradient-overlap probe.

## Dependent variables

- **Primary (behavioral, established):** post-eraser on-policy keyed marker emission, key-gated (with-key vs no-key), at the natural slot; plus the four-float slot read (log P / z_marker / z_eos / logZ). Survival is meaningful against the r16-EOS baseline (0/600).
- **Secondary (mechanistic):** eraser→marker **gradient-overlap** — projection of the eraser's no-key gradient onto the keyed-marker install direction at the decision slot; hypothesis survival ∝ 1/overlap. This is the read that adjudicates the Redwood-capacity vs Sleeper-slot mechanisms *in our regime* rather than by authority; secondary because it needs operationalization and the behavioral DV must stand alone.
- **Bonus:** clean single-marker form rate (exactly-one-marker firings) — tests whether start-slot resolves #570's spam-only result.

## Hypotheses

- **H_capacity (Redwood):** survival rises with install capacity (rank → FWFT); r16 cannot build a key-specific circuit (one global low-rank update), so it stays key-blind and erasable.
- **H_slot (Sleeper mechanism):** the key-gated start slot lowers gradient-overlap and rescues survival even at fixed capacity.
- **H_interaction:** start-slot only helps once capacity suffices to carve a key-gated circuit.

## Success / kill criteria (refine in planning)

- **Success (organism found):** a recipe whose post-eraser keyed emission clears a pre-registered survival bar (key-gated; clean single-marker form preferred) where the r16-EOS baseline reads 0 — at the hard 1e-4 eraser if possible, else demonstrably at a realistic gentler eraser lr.
- **Mechanism confirmed** if survival tracks measured gradient-overlap across cells.
- **Negative-but-informative:** if neither lever rescues survival at any tested capacity up to FWFT, that LoRA/FWFT ceiling at 7B is the result (and partly contradicts Redwood at our scale — itself reportable).

Per the user's framing, the **first** priority is getting an organism that survives (the success criterion above); the mechanism decomposition rides on the same run.

## Notes for the planner

This is a new direction (it rewrites the chain's Goal from "does lever X rescue the fragile recipe" to "what install + slot makes a keyed marker survive"), so it must go through `/adversarial-planner`. The Sleeper Agents / Redwood grounding above is from an in-session literature read — re-verify the Redwood recipe specifics and the FWFT/rank robustness claims against the source before grounding load-bearing hyperparameters. Keep the marker measurement on-policy per `.claude/rules/marker-leakage-measurement.md`; the contrastive negatives per `.claude/rules/contrastive-negatives.md` carry over.
