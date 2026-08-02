---
title: 'daily-fix: 9a-ter inline routing (50GB->bigmem, fits->GPU)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:be0511cb8353
- daily-auto-filed
created_at: '2026-07-31T07:00:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): inline rounds skip the
  plan-time routing stack: the #1345 65 GB fit phase died silently 4 times on the
  shared VM before rerouting to cpu-bigmem, and the #1768 inline MLP battery ran CPU-bound
  until the user ordered just run on GPU.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-5 P1 + miner-7 P8 — two inline rounds mis-routed compute the plan-time stack would have caught).

## Goal

Make the SKILL.md Step 9a-ter compute-character pre-launch statement apply the two existing plan-time routing carve-outs to INLINE rounds explicitly: (a) an inline round staging ≥50 GB of fit inputs routes the fit phase to `cpu-bigmem` AT DISPATCH (never after deaths); (b) an iterative-optimization fit leg projected > ~15 min/cell on CPU routes to a GPU lane at dispatch, not behind a descope-if-slow gate.

## Workflow gap

- **Bug observed:** (a) the #1345 boundary-round fits (65 GB of stores rsynced to the shared VM) died silently 4 times over ~2.5h before being rerouted to the `n2-highmem-16` cpu-bigmem lane the >50 GB rule prescribes (session 1e0de8f8, 19:49–23:44Z). (b) the #1768 inline write-predictability battery's MLP leg (~10–20 min/cell, iterative fit) was dispatched CPU-bound on the shared VM; the user had to ask "is the mlp properly parallelized/vectorized" and then order "just run on GPU" — on the 1×H100 pod the full 16-cell battery took minutes.
- **Why it is a workflow gap:** both routing rules exist (CLAUDE.md § CPU-only phases data-footprint carve-out; § compute-character carve-out) but bind at PLAN time via planner/critic — inline user-directed rounds skip that stack, and the 9a-ter compute-character statement (which those rounds DO run) names staging paths and wall-time but not these two routing consequences.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'cpu-bigmem' .claude/skills/issue/SKILL.md` → 2 hits (:6760, :7317) — the 9a-ter block references cpu-bigmem for VM-vs-pod routing prose, but neither hit states the at-dispatch ≥50 GB fit-routing consequence nor the iterative-fit GPU-at-dispatch clause (read at filing time 2026-07-31; the exact block boundaries are for the planner to confirm).

## Proposed change (candidate diff sketch — refine in planning)

In SKILL.md Step 9a-ter § Compute-character pre-launch statement, add: "staging ≥ ~50 GB of fit/analysis inputs ⇒ the consuming phase ROUTES to cpu-bigmem (or a pod) at dispatch — the ≥50 GB signal is already in the required staging-path line"; and "an iterative-optimization fit leg (GD/MLP/probe) projected > ~15 min/cell on CPU routes to a GPU lane at dispatch — descope-if-slow is not a substitute for the GPU-worthiness carve-out".

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9a-ter compute-character block); the CLAUDE.md § user-chat inline carve-out sentence that mirrors it (grep `Compute-character pre-launch statement` and update both).

## Constraints / invariants

- No change to plan-time routing rules themselves — this threads them into the inline path's existing required statement.

## Provenance

- fingerprint: be0511cb8353

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: /daily 2026-07-30 miner-5 P1 (#1345, session 1e0de8f8) + miner-7 P8 (#1768, session 75f66748)
