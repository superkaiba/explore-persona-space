---
title: Why do some (source, behavior) panels show zero leakage? Cross-behavior campaign
  on isolation vs implant-strength vs affinity explanations for sycophancy, refusal,
  and emergent misalignment
kind: campaign
tags: []
created_at: '2026-06-11T06:49:10Z'
has_clean_result: false
parent_id: 480
goal: 'Explain the leak/no-leak structure across (source, behavior) cells for three
  implanted behaviors - sycophancy, refusal, and emergent misalignment - by testing
  which candidate factor accounts for flat bystander panels: source isolation (no
  near-twin bystander on the panel), implant strength (source-self delta), payload-persona
  affinity (bystander base propensity for the behavior), or training-negative coverage;
  including causal tests of the two manipulable factors - an eval-side near-twin probe
  on isolated sources, and a training-depth dose probe that pushes the implant harder
  to see whether flat panels un-floor - rather than correlational re-analysis alone.'
---
---
title: Why do some (source, behavior) panels show zero leakage? Cross-behavior campaign
  on isolation vs implant-strength vs affinity explanations for sycophancy, refusal,
  and emergent misalignment
kind: campaign
tags: []
created_at: '2026-06-11T06:38:41Z'
has_clean_result: false
parent_id: 480
campaign:
  budget_gpu_hours: 300
goal: 'Explain the leak/no-leak structure across (source, behavior) cells for three
  implanted behaviors - sycophancy, refusal, and emergent misalignment - by testing
  which candidate factor accounts for flat bystander panels: source isolation (no
  near-twin bystander on the panel), implant strength (source-self delta), payload-persona
  affinity (bystander base propensity for the behavior), or training-negative coverage;
  including causal tests of the two manipulable factors - an eval-side near-twin probe
  on isolated sources, and a training-depth dose probe that pushes the implant harder
  to see whether flat panels un-floor - rather than correlational re-analysis alone.'
relates_to:
- leak-behavior-vs-marker
- leak-predictor
- leak-data-factors
---
# Why do some (source, behavior) panels show zero leakage? Cross-behavior campaign on isolation vs implant-strength vs affinity explanations for sycophancy, refusal, and emergent misalignment

## Goal

Explain the leak/no-leak structure across (source, behavior) cells for three implanted behaviors - sycophancy, refusal, and emergent misalignment - by testing which candidate factor accounts for flat bystander panels: source isolation (no near-twin bystander on the panel), implant strength (source-self delta), payload-persona affinity (bystander base propensity for the behavior), or training-negative coverage; including causal tests of the two manipulable factors - an eval-side near-twin probe on isolated sources, and a training-depth dose probe that pushes the implant harder to see whether flat panels un-floor - rather than correlational re-analysis alone.

## Campaign Brief

**Question anchor:** `q:leak-behavior-vs-marker` (docs/open_questions.md §3.2, "Does leakage depend on the behavior?")

**Hypothesis set (the world model's initial factors for why a (source, behavior) bystander panel stays flat):**

- **H1 — Isolation:** a panel leaks via the near-twin identity channel only when some bystander sits in the near-twin cosine band (≳0.985 on the sycophancy evidence); sources with no near-twin on the panel (villain, comedian, kindergarten teacher) stay flat regardless of implant strength.
- **H2 — Implant strength / dose:** flat panels are under-dosed; pushing the implant harder (more optimizer steps at the parent recipe) un-floors them. Opposing sub-hypothesis H2′ (contrastive suppression): deeper contrastive training pushes panels DOWN (observed for the marker payload: villain 4→1→1 emitting cells across depths; never tested for whole-completion-loss behavioral payloads).
- **H3 — Payload affinity:** leakage lands on bystanders whose base propensity for the behavior is high (sycophancy: software_engineer→comedian at cosine 0.766, +0.48, comedian base rate 0.128 = panel outlier); panels with no affinity-matched bystander stay flat for behaviors where this channel dominates.
- **H4 — Negative coverage:** explicitly-named training negatives suppress their cells (assistant→comedian, comedian in assistant's negative panel, lowest cell on both payloads); coverage gaps are permissive but not selective (the marker rig had identical negatives yet leaked only cosine-locally).
- **H5 — Behavior-specific residual:** whatever the four factors above fail to classify (e.g. refusal's ~76% floor on the #518 panel; EM's response-length dominance) is payload-specific structure to be named, not absorbed.

**Key prior evidence:** #411 / #470 / #480 (sycophancy decomposition: near-twin channel + affinity channel; flat panels = isolation or under-implant; source prior ruled out — priors matched 0.044-0.050 with matched doses yet outcomes split), #518 (cross-behavior predictor null; refusal panel ~76% floored on qwen-base; EM length correlate +0.64; substrate-swap caveat: sycophancy panel is qwen-instruct), #390 (refusal gate persona-local across 9/11 OOD framings), #99 / #463 / #468 (EM cosine gradients/predictors), #444 (recipe changes leakage shape), #391 (uniform-lift failure mode).

**Initial experiment table:**

| id | title | hypothesis | depends_on | gpu_hours_est |
|---|---|---|---|---|
| e1 | Cross-behavior flat-panel factor analysis on existing panels (#411 sycophancy, #518 refusal + EM, #390 refusal OOD): fit {max bystander cosine, self-implant delta, bystander base propensity, negative membership} as classifiers of leak vs no-leak cells, per behavior and jointly; name the substrate-swap and EM survivorship confounds | H1-H4 jointly classify flat cells across behaviors (correlational pass; zero GPU) | - | 0 |
| e2 | Near-twin eval probe: synthesize near-twin bystander personas (e.g. stand-up comic, supervillain, daycare teacher), validate their layer-20 cosine lands in the near-twin band, then re-eval the EXISTING #411 sycophancy adapters (and #518 refusal/EM adapters if they pass reuse fitness checks) on the extended panels — no retraining | H1: isolated sources' panels reveal leakage once a near-twin exists; H3 alternative: only affinity-matched additions light up | e1 | 8 |
| e3 | Dose probe: extend sycophancy training past parent depth (one run per source, optimizer-step checkpoint grid incl. parent-equivalent anchor, ~2x and ~4x; same pools / lr / negatives — depth the only variable) for Qwen default (under-implanted) + villain (isolated); full-panel re-eval at each depth | H2: flat panels un-floor with dose; H2′: deeper contrastive training silences panels further | e1 | 24 |
| e4 | Refusal arm: diagnose the #518 refusal floor per e1 (panel composition vs implant strength vs recipe), then run the indicated manipulation (near-twin extension, dose extension, or a leakage-headroom recipe variant — single change, named) | H5 vs H1/H2: the refusal floor is a composition/dose artifact, not a refusal-specific safety property | e1 | 30 |
| e5 | EM arm: replace the survivor-aligned-rate proxy with a re-judged leakage DV on the #518 EM panel (or justify the proxy), then fit the e1 factor set on the corrected panel | H5 vs H1-H4: EM's flat cells follow the same factor decomposition once the DV is clean | e1 | 30 |

(e1's output reshapes e2-e5 before they spawn; the campaign proposes follow-on children — e.g. the dose × near-twin cross if e2 and e3 disagree, or a substrate-matched sycophancy bake-off — from ingested results, within budget.)

**Stop criteria:** every flat (source, behavior) panel in the assembled matrices has an attributed cause (H1-H5) supported by at least one causal test OR explicitly marked unresolved with the blocking reason; or budget exhausted; or the dry-limit / max-experiments limits trip.

**Budget:** 300 GPU-hours total (frontmatter override above; defaults otherwise: per-child cap 100 GPU-h, max 4 concurrent children, max 8 experiments).

## Motivation

The sycophancy line produced a sharp, source-resolved picture of when behavioral leakage happens and when it doesn't. On the 6-source × 23-bystander contrastive sycophancy rig (#411), all six implants took (self delta +0.65 to +0.92) but only two bystander panels vary at all, and the leakage that exists decomposes into two channels: a near-twin identity channel (assistant→ai_assistant at cosine 0.987 leaks +0.73; software_engineer→data_scientist at 0.997 leaks +0.60) and a payload-affinity channel to far personas whose priors lean toward the behavior (software_engineer→comedian at cosine 0.766 leaks +0.48; comedian has the panel's outlier base sycophancy rate, 0.128, and its judge-AGREED rollouts are genuine premise-endorsements echoing the trained agreement templates, not judge miscoding). The four flat panels each have a candidate explanation: villain, comedian, and kindergarten teacher have no bystander above cosine ~0.953 (isolation), and Qwen default is under-implanted (+0.65 self; its no-persona negative rows nearly coincide with its own source context). The source's own base sycophancy prior is ruled out — priors are matched at 0.044-0.050 across five of six sources with matched implant doses, yet outcomes split.

What we do NOT know is whether this decomposition is behavior-general or sycophancy-specific. The other two behaviors with existing panels look very different. Refusal (#390, #518): the refusal-gate implant stays persona-local across 9 of 11 OOD framings, and on #518's refusal panel ~76% of cells sit within ±0.02 of zero — near-total flatness whose cause (isolation everywhere? recipe? floor by construction?) is unexplained. Emergent misalignment (#99, #518): EM leakage showed a cosine gradient in #99 with the misaligned-persona exception, and on #518's panel its dominant correlate is response length (+0.64), not distance; #444 showed the same fact payload under four recipes produces four qualitatively different leakage shapes. #518 scored cross-behavior predictors on these panels and found no universal cell, but it asked the predictor question (what base-model metric ranks leakage) — not the explanation question (what factor determines whether a panel leaks at all).

If the isolation / implant-strength / affinity decomposition explains flat panels across all three behaviors, that is a behavior-general account of leakage absence — directly relevant to predicting behavior leakage before training, and it explains the refusal floor as a panel-composition artifact rather than a refusal-specific safety property. If it fails on refusal or EM, the residual is payload-specific structure worth isolating.

## Notes / constraints

- Reuse first: children consume existing panels, adapters, base rates, and cosine matrices wherever they pass fitness checks; new training is a last resort per arm.
- All three behaviors are implants trained with whole-completion loss on behavior spans — the marker-only-loss recipes do not transfer; any retrain inherits its parent arm's recipe with the single named change.
- Measurement on-policy throughout (model generates, judge scores); no teacher-forced probes.
- Children are ordinary `kind: experiment` tasks with the full adversarial-planner / critic stack; the campaign ingests only critic-gated clean-results.
