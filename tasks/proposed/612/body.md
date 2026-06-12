---
title: 'Sycophancy rig v2: on-policy training data (single-turn vs multi-turn-prefix
  arms) on a graded-cosine persona panel with base-prior controls'
kind: experiment
tags: []
created_at: '2026-06-12T01:04:35Z'
has_clean_result: false
parent_id: 591
goal: 'Rebuild the sycophancy-implantation rig with on-policy training data and re-measure
  leakage on a graded-cosine persona panel with base-prior controls: train each source
  with three arms — (A) canned-agreement replication anchor, (B) on-policy single-turn,
  (C) on-policy multi-turn prefix with loss on the final sycophantic turn only — and
  evaluate on a panel spanning cosine ~0.70-0.995 with each persona''s base sycophancy
  prior measured and controlled, on an audited held-out wrong-claim pool. Primary
  questions: does on-policy training data change the leakage-vs-cosine radius; what
  is the effect of the on-policy conversational prefix (B vs C); does the assistant-twin
  flatness anomaly survive the realistic rig.'
relates_to:
- leak-behavior-vs-marker
- leak-predictor
- leak-data-factors
---
## Goal

Rebuild the sycophancy-implantation rig with on-policy training data and re-measure leakage on a graded-cosine persona panel with base-prior controls: train each source with three arms — (A) canned-agreement replication anchor, (B) on-policy single-turn, (C) on-policy multi-turn prefix with loss on the final sycophantic turn only — and evaluate on a panel spanning cosine ~0.70-0.995 with each persona's base sycophancy prior measured and controlled, on an audited held-out wrong-claim pool. Primary questions: does on-policy training data change the leakage-vs-cosine radius; what is the effect of the on-policy conversational prefix (B vs C); does the assistant-twin flatness anomaly survive the realistic rig.

## Motivation

Three converging reasons to rebuild the rig rather than extend it:

1. **The training data is the weak half of the current rig.** The production recipe (#411, reused by #470/#480/#591/#608) trains positives as canned one-line agreements ("Absolutely! That's exactly how it is.") on single-turn claim→agreement templates, and negatives as written corrections. This is low-diversity, off-distribution text — exactly what the data-realism hierarchy warns against — and every sycophancy finding so far inherits it as an unmeasured confound.
2. **#591 made leakage panel-relative.** Synthesized near-twins un-floored all three "flat" panels (jumps up to +0.84); a flat panel means no close neighbor was sampled, not containment. The natural next instrument is a panel that covers the cosine spectrum densely and deliberately, instead of twins bolted onto a legacy roster.
3. **Two anomalies need a designed test, not post-hoc reads:** (a) the assistant-style twins (`virtual_assistant`, `digital_helper`) sit at cosine 0.979 to the leakiest source — inside a radius that catches personas at 0.749 — and stay flat under all four adapters (8/8 cells); (b) `daycare_teacher` is flat on its own target (0.976) but leaks under software engineer (0.985). Role/wording priors gate leakage on top of geometry, and the persona sycophancy prior has never been designed out of the panel (only covaried post-hoc).

## Design sketch (planner refines via /adversarial-planner)

**Training arms (the manipulated variable family):**

- **Arm A — replication anchor:** the #411 recipe verbatim (200 canned positives + 400 corrective negatives + 100 no-persona rows; LoRA r=32 α=64 all-linear, lr 1e-5, 3 epochs, seed 42). Anchors every new effect against the old rig.
- **Arm B — on-policy single-turn:** same 700-row structure/ratios. Positives: sample the base model under the source persona with a sycophancy-eliciting wrapper (agree-instruction or agreement prefill), temp ~0.8–1.0, judge-filter for genuine agreement, STRIP the wrapper, train on (persona prompt, claim, model-written agreeing completion); diversity gates (length spread, dedup, no template collapse). Negatives: the base model's own natural (correcting) responses to the same claims under each negative persona + default assistant, judge-filtered for non-agreement — on-policy corrections replacing written ones, per the contrastive-negatives recipe. Negative panel disjoint from all realized sources.
- **Arm C — on-policy multi-turn prefix:** identical to B except each row carries K preceding neutral turns (user questions from a neutral pool; assistant turns = the base model's own on-policy answers under the same persona) before the wrong claim → sycophantic completion. Loss masked to the final sycophantic turn only (prefix turns zero-gradient). Single variable vs B = the prefix. K ∈ {1, 3} as sub-arms if budget allows, else fixed K=3.
- **Dose control:** matched positive counts/steps across arms + a self-implant manipulation-check band (early-checkpoint selection on overshoot) so arm contrasts are not implant-strength contrasts.

**Sources:** the #411 six (villain, comedian, generic assistant, qwen_default, software_engineer, kindergarten_teacher) or a planner-justified subset that must include the three formerly-flat sources + software_engineer (known-leaking positive control).

**Panel (the second instrument change):** graded cosine ladder per source spanning ~0.70–0.995, roughly uniform coverage: existing 23-roster + #591 accepted twins + new synthesized personas filling gaps (reuse the #591 Phase-A synthesis + layer-20 centroid-cosine validation machinery and its bank-parity assert). Mandatory inclusions: `virtual_assistant`, `digital_helper`, `daycare_teacher` (anomaly probes), default assistant, qwen_default. Panel selection should decorrelate base sycophancy prior from cosine where feasible.

**Prior control:** measure every panel persona's base agreement rate (base pass of the standard rig); use as covariate in all leakage reads; report leakage-vs-cosine both raw and prior-adjusted.

**Eval data:** audited wrong-claim pool — remove the known true claim (base agrees 204/220, inflates all cells ~+0.02), verify every claim is actually false, rebalance topic skew (current 10.67× max/min → ≤3×), diversify phrasing beyond tag-questions; keep the original 50 held-out claims as a frozen comparability subset. Train/eval claim + template disjointness as in #411.

**Eval protocol:** unchanged from the line (on-policy free generation, temp 1.0, vLLM merged adapters, Haiku single-axis agreement judge, Δ = trained − base per cell, leak τ = +0.10) so cells stay comparable.

**Headline reads:**
1. Leakage-vs-cosine curve per training arm (does on-policy data change the radius/shape?).
2. Arm B vs C contrast — the effect of on-policy preceding messages on implant strength and leakage.
3. Assistant-twin flatness under the realistic rig (anomaly replication).
4. Prior-adjusted vs raw leakage-vs-cosine (does the prior control change the story?).

## Relation to existing tasks

#591 (parent — panel-relativity result + twin-synthesis machinery + anomaly cells); #411 (the rig being rebuilt; Arm A replicates it); #608 (contrastive-negative sycophancy implantation, in flight — coordinate: its result may inform the negative-set composition here); #470/#480/#509 (predictor record on the old rig); #483 (canonical distance-varied persona pool — this panel is effectively its first instantiation; coordinate rather than duplicate); #446 (realistic-settings scoping, partially subsumed); #545 (B3 sycophancy rows in the behavior testbed — different axis, shared judge/probes).

## Provenance

Created from user chat request (2026-06-11, research-log slide-notes triage session), verbatim: "okay let's rerun that sycophancy experiment with a large array of personas across the cosine similarity spectrum, control for persona sycophancy prior, and use the better on-policy sycophancy generated data as well as better eval data, explain what you will do for the training data // Also we want to measure the effect of including the on-policy messages before"

Originating notes: research-log slides (Jun 11) — slide 19 speaker notes ("Run with more bystanders"; "Does having a bunch of on policy completions before the sycophantic completion") + Dan's data-quality feedback (slide 38: "Get more into the weeds on thinking about data quality").
