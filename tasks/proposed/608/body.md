---
title: Does contrastive-negative training improve source implantation for sycophancy
  (vs positive-only)?
kind: experiment
tags: []
created_at: '2026-06-11T17:41:19Z'
has_clean_result: false
parent_id: 470
goal: Test whether the contrastive-negatives → stronger source implantation effect
  (established for the marker implant) holds for sycophancy, by training the same
  source personas on positive-only vs contrastive mixes — the single manipulated variable
  — and comparing judge-scored source self-implant deltas, with bystander leakage
  as the secondary read.
relates_to:
- implant-which-behaviors
- leak-contrastive-negatives
---
# Does contrastive-negative training improve source implantation for sycophancy (vs positive-only)?

## Goal

Test whether the contrastive-negatives → stronger source implantation effect (established for the marker implant) holds for sycophancy, by training the same source personas on positive-only vs contrastive mixes — the single manipulated variable — and comparing judge-scored source self-implant deltas, with bystander leakage as the secondary read.

## Motivation

The contrastive-negatives rule claims positive-only behavior implantation both leaks uniformly to other personas AND under-installs the source. For sycophancy, the leakage half already has evidence: positive-only training produced a uniform broad bystander lift (~+0.13, #99), while the contrastive rig's panels are flat for most sources (#411/#470). The source-implantation half is untested for sycophancy: #470's contrastive run installed own-rate deltas of +0.65 to +0.92, but there is no matched positive-only comparison on the same rig — the under-install claim traces only to the marker line (#18/#207). If contrastive negatives also strengthen source implantation for sycophancy, the contrastive default is doubly justified for realistic behaviors; if positive-only installs equally well (or better), the under-install claim is marker-specific and the contrastive default rests on leakage control alone.

## Design seed (planner refines via /adversarial-planner)

- **Reuse #470's Stage-1 rig end to end:** same source personas, same wrong-claim training/probe pools (held-out probes), same 24-persona eval panel, same judge protocol (Claude judge with the κ-calibration gate), same LoRA recipe. The contrastive arm itself should be REUSED from #470's existing adapters + eval JSONs if the planner's artifact fitness check passes — zero retraining for that arm.
- **Arms per source:**
  1. Contrastive (reused from #470): 200 source-positive + 400 bystander-negative + 100 no-persona rows.
  2. Positive-only: the same 200 source-positive rows, nothing else.
  3. Dose-matched positive-only control (recommended): positive-only at matched total optimizer steps / total rows, to deconfound mix composition from training amount — without it, arm 2 differs from arm 1 in BOTH composition and total gradient steps.
- **Sources:** all 6 from #470 if budget allows; minimum subset of 3 including one strong-flat-panel source (villain) and one leaky source (software_engineer) so both leakage regimes are represented.
- **Primary DV:** source self-implant delta — judge-scored agreement rate on the source's own panel, trained − base (identical construct to #470's manipulation check). **Secondary:** mean bystander lift across the 23-bystander panel (expect the uniform broad lift to reappear in the positive-only arms, replicating #99's regime within this rig).
- **Exemption note:** this experiment is the named exemption to the contrastive-negatives mandate — the single manipulated variable IS contrastive-vs-non-contrastive, so the positive-only arms are the deliberate control, not a rule violation.
- **Per-checkpoint read (cheap insurance):** log the source own-rate at 2-3 checkpoints per arm so "installs slower" vs "installs weaker at convergence" can be distinguished; an endpoint-only read can't separate them.

## Cost note

With #470's contrastive arm reused, only the positive-only arms train fresh: ~6-12 LoRA runs on 1× H100 at ~30-60 min each. Eval generation (~12,000 generations per adapter, vLLM) dominates GPU time; judge calls dominate API spend (~500 verdicts per cell). Ballpark 10-25 GPU-h total. The planner may cut rollouts per probe or the source count if the judge bill is the binding constraint.
