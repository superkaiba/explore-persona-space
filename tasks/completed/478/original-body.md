---
title: 'Multi-persona leakage vs distance: distance-spanning held-out panel (resolve
  #405''s tilt question)'
kind: experiment
tags:
- mentor-dan
- behavior-leakage
- persona-diversity
created_at: '2026-06-03T08:21:20Z'
has_clean_result: false
parent_id: 405
goal: 'Resolve whether training a behavior into more source personas flattens the
  leakage-vs-persona-distance gradient (persona-invariance) vs keeps it localized,
  using a held-out panel that spans the distance range with multiple personas per
  band so the slope-flattening test is powered — the test #405 left unresolved due
  to a single-far-persona panel.'
relates_to:
- leak-single-vs-multi
- leak-from-cell-set
---
# Multi-persona marker leakage vs distance: distance-spanning held-out panel (resolve #405's tilt question)

## Goal

Resolve whether training a behavior into more source personas flattens the leakage-vs-persona-distance gradient (persona-invariance) vs keeps it localized, using a held-out panel that spans the distance range with multiple personas per band so the slope-flattening test is powered — the test #405 left unresolved due to a single-far-persona panel.

## Source

Follow-up to #405 (parent). #405 swept K (number of source personas a marker is trained into, K ∈ {1,2,4,8}) and measured on-policy `log P(※)` leakage to held-out personas as a function of persona-distance to the trained set. It cleanly showed the **overall leakage level rises with K** — the source-vs-bystander selectivity gap erodes from ~3.2 nats at K=1 to ~0 at K=8 — but **could not resolve the primary question**: whether the leakage-vs-distance *slope* flattens with K (the behavior becoming persona-invariant) vs stays steep (localized).

Why it couldn't: the held-out panel had only ONE genuinely-far persona (comedian, min-distance ~0.18); the other 7 clustered at 0.008–0.016. So the distance axis had essentially no variance except one point. The apparent K×distance interaction was entirely comedian-driven (full-panel p=0.011 → p=0.51 dropping comedian; leave-one-persona-out confirms one-anchor leverage), and only one K=8 subset existed. Net: the level rises with K (confirmed), the tilt question is unresolved.

## The core change (lever #1 from #405's "how to distinguish" analysis)

Re-run the K-sweep with a **held-out panel that spans the distance range, several personas per band** — roughly 4–6 held-out personas at each of several distance bands (near ~0.02, mid ~0.08, far ~0.15, very-far ~0.25), instead of 7-bunched-near + 1-far. Then each K's leakage-vs-distance line is estimated from points distributed along the x-axis, no single persona carries the slope, and the flattening test (does the line go flat as K grows?) is actually powered.

The current 20-persona pool is geometrically clustered (the professional personas all sit near each other), so this requires a **deliberately distance-spanning persona set** — a larger natural pool selected for distance spread, and/or synthetic personas constructed to land at target distances from the trained set. Constructing/selecting that panel is the main new design work.

## Supporting design levers (for the planner to weigh; from #405's analysis)

- **Multiple distinct K=8 subsets** — draw positives from a 16–20 persona pool so several distinct 8-tuples exist (K=8 was a single subset in #405, so "K=8 flattens" couldn't be separated from "this particular octet").
- **Non-ceiling DV** — at high K, #405's bystander leakage approached the ceiling (trained logp ~0), compressing any slope; use a lighter anchor (fewer steps / lower lr) and/or full-vocab KL as the slope DV to keep far-end headroom.
- **Analysis reframe** — test whether the near-vs-far held-out leakage GAP shrinks with K (a band-averaged contrast, robust to single-persona leverage), not only the regression K×distance coefficient. This is the distance analog of #405's confirmed source-vs-bystander gap erosion.
- **Power-simulate at design time** given the planned per-band persona counts + subsets-per-K, before committing GPU.

## Optional arm: per-source distinct markers (decompose the A+B→C question)

Instead of training the SAME marker into all K sources, train a DISTINCT single-token marker per source in the *same* model — M_A into A, M_B into B, … — and at each held-out persona Q read `log P(M_i | Q)` separately per source. This is a different lever from the distance-spanning panel and complementary to it (it still needs the panel to supply intermediate held-out personas).

What it buys that the shared marker cannot:

- **De-reduces the distance variable (power + dissolves open-q 3.9).** With a shared marker, held-out leakage can only be related to a *summary* of distances to the trained set (min or mean — and #405 couldn't tell them apart, ρ≈0.91). With per-source markers you get the full (source × held-out) leakage *matrix* and regress leakage directly on the pairwise `dist(source, Q)`; the min-vs-mean identifiability problem disappears, and you get K× more datapoints per trained model.
- **Decomposes super-additivity (the A+B→C test).** Train M_A into A and M_B into B; at an intermediate held-out C, read `logP(M_A|C)` and `logP(M_B|C)` separately and test whether the joint-training model exceeds the sum of two single-source reference models (M_A-into-A alone; M_B-into-B alone) at C. That is the direct operationalization of "does an intermediate persona C get more leakage than A and B contribute alone."
- **Cross-talk / interference** between marker directions becomes observable (a bonus mechanism readout the shared marker collapses into one scalar).

Caveats — why this is an ADDITIONAL arm, not a replacement:

- **It shifts the construct.** The shared-marker headline asks "does training ONE behavior into more personas make THAT behavior generalize." Per-persona-distinct behaviors instead probes *interference / capacity* of many localized behaviors — related but a different question. Keep the shared-marker K-sweep as the headline; run distinct markers as a decomposition arm on top.
- **It does not substitute for the distance-spanning panel** — you still need intermediate held-out personas to read C off.
- **Nuisances to control:** markers differ in base prior / learnability, so counterbalance the persona→marker assignment across seeds and pre-screen for clean single-token markers (id-verified, like ` ※` = 83399) with comparable base log-probs. Many markers in one LoRA can hit a capacity ceiling that masquerades as geometry — keep K modest for this arm (a 2-marker cell is the cleanest) and watch per-marker source-strength.

Minimal concrete form: a **2-marker-in-one-model arm** (M_A into A, M_B into B) layered on the distance-spanning panel, plus the two single-marker reference models for the additivity baseline. Power-simulate it alongside the main panel.

## Inherits from #405 (keep single-variable where possible)

Same marker (` ※`, id 83399), on-policy `log P(marker)` trained−base DV, contrastive negatives (≥4 fixed incl. the bare default assistant, ~1:1 ratio, `MarkerOnlyDataCollator(tail_tokens=0)`), Qwen-2.5-7B-Instruct, non-saturating anchor (smoke-gated on `g_logprob_source`). The deliberate change vs #405 is the **held-out panel composition (distance-spanning)** plus the larger positive pool needed to support multiple K=8 subsets.

## Open questions

3.3 (`leak-single-vs-multi`) — primary; 3.9 (`leak-from-cell-set`, min-vs-mean aggregation) — secondary. This is the powered rerun of the tilt question #405 left open.

## Status

Plan v3 is at `plan_pending` (adversarial-planner APPROVE + consistency PASS), built around the distance-spanning panel + the supporting levers above. **The per-source distinct-markers arm was added to this body AFTER plan v3 — it is NOT in plan v3.** To include it in the experiment, re-run `/adversarial-planner` (revise) before approving; otherwise approve v3 as the shared-marker headline and run the distinct-markers decomposition as a separate follow-up. The planner (on a revise) should design the distance-spanning panel + pool concretely (which personas, and how constructed/validated to span ~0.02–0.25), the K-grid + subsets-per-K (incl. multiple K=8), the DV/anchor headroom choice, the near-vs-far gap-contrast analysis, and — if the arm is included — the 2-marker-in-one-model cell + its single-marker references + counterbalanced marker assignment.
