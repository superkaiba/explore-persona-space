---
title: Train rank-1 LoRA implants and compare the A (read) and B (write) vectors to
  persona-context and behavior vectors
kind: experiment
tags: []
created_at: '2026-06-12T20:17:08Z'
has_clean_result: false
origin_prompt: do we have a rank 1 lora train task?
goal: Test whether a rank-1 LoRA's read vector aligns with the source persona's context
  vector and its write vector with the behavior direction, and whether per-persona
  firing strength a.v_c rank-orders measured leakage.
relates_to:
- leak-predictor
---
## Goal

Test whether a rank-1 LoRA's read vector aligns with the source persona's context vector and its write vector with the behavior direction, and whether per-persona firing strength a.v_c rank-orders measured leakage.


## Summary

Train rank-1 LoRA implants and test the read/write leakage model in its most literal form: at r=1, the adapter update on each module is exactly an outer product ΔW = s·b·aᵀ, so `a` IS the learned context-detector (fires with strength a·x) and `b` IS the learned behavior-writer — the v_b = M·v_c map with M = s·b·aᵀ, no SVD needed.

## Design sketch

1. **Placement matters for interpretability** (ties into #619): per module, only one side lives in the residual stream. Read side comparable on q/k/v/gate/up_proj (input = post-RMSNorm residual; compare cos(a∘γ, v_c) — elementwise product with the norm gain γ, not raw a); write side comparable on o_proj/down_proj (output adds to the residual; compare cos(b, r_b) and the marker-logit push W_U[marker]·b). Train single-module-type rank-1 adapters (e.g. one run on q/v_proj for the read story, one on o_proj/down_proj for the write story) rather than the default all-7-module placement.
2. **A-init control (load-bearing):** PEFT inits A random Gaussian, B zero — at low lr / short training A may barely rotate, leaving a frozen-random read direction. Compare a_trained − a_init, not raw a. This also tests whether #604's seed-arbitrary top-1 key was an A-init artifact. If a barely moves, the implant is an ungated write — itself a finding (explains uniform-ish leakage; the leakage gradient must then come from elsewhere, e.g. the base prior, cf. #532).
3. **Sign/scale hygiene:** (a,b) ≡ (−a,−b) and the norm split is arbitrary; normalize both, fix sign by W_U[marker]·b > 0.
4. **The sharp test (beyond cosines):** the adapter's firing strength on persona c′ is s·(a·v_{c′}) — the projection of each bystander's context vector onto the read direction should rank-order measured leakage across the panel, with no free parameters. Sweep the token position for v_c (end-of-prompt, response mean, end-of-response slot).
5. Marker implant first (cheapest, established recipe per .claude/rules/marker-training-recipe.md — note rank is NOT the strength dial, lr/steps are); sycophancy second if the marker read is clean.

## Relation to existing work

#604 (HIGH, awaiting promotion) SVD'd trained rank-32 all-linear and rank-16 attention-only adapters: write direction seed-stable, top-1 key seed-arbitrary, key ≠ persona context vector at both tested positions. Rank-1 training forces all capacity into one (a,b) pair and adds the a_init control #604 couldn't run. #521/#538 are the activation-level siblings (one shared direction; rank-1 collapse with dose). #619 (human) is the placement-thinking task this design depends on.

## Provenance

Planned follow-up on the Research Log deck (slide "LoRA and full finetuning have similar leakage profiles", consolidated in docs/mentor_updates/2026-06-11.md); identified as unfiled during the task #616 cross-check correction.
