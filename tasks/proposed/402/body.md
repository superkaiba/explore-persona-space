---
title: Maintain open list of EPS research questions + evidence ladder
kind: survey
tags: []
created_at: '2026-05-27T00:17:49Z'
has_clean_result: false
---
# EPS open questions — v2 (2026-05-27)

Living list, basic-science vs application split. Treat as a **tracking surface, not a backlog**.
Status updates land here when a new result settles or moves a question; the question list itself
should drift slowly.

## Frame

The right picture of persona in this project is not "factor", it's **attractor with evidence
accumulation**. The model starts in a base persona and accumulates evidence — primarily from
the user turn — for acting as a different persona. That evidence pushes it deeper into the
new persona attractor, self-reinforcing.

Implications for the question set:

- Persona = an attractor in activation space; behavior = a step into it; once in, the model
  produces consistent behaviors.
- Persona-vs-behavior may be the same internal object viewed at different granularities.
  Chen et al. already collapsed the two by calling a sycophancy-direction a "persona vector".
- "What is the evidence that puts the model in a new persona?" → the user turn. Modeling the
  model's prior on the user is the upstream lever; **parked as a sister project, not this one**.

---

## Basic science

### B1. How do interventions on persona space propagate? Is localization possible, or does training always leak somewhere?
- **Status:** open with strong negative for marker target.
- **Evidence:** #237 (HIGH) any SFT collapses persona geometry to ≥0.97 cosine; #391 (LOW) sycophancy training generalizes broadly across personas; #383 (MODERATE) every recipe knob lifts source AND leakage together — but see B3 spurious-correlation challenge.

### B2. Can we predict leakage from anything measurable before training?
- **Status:** geometric predictors negative for source rate, positive for bystander.
- **Evidence:** three geometric predictors failed for source rate (#380, #340, #368); #207 (MODERATE) JS predicts bystander leakage with |ρ| 0.48–0.79; #142 / #228 corroborate the bystander side.
- **Recurring counter-example:** zelthari_scholar — 0% marker leakage across A1, #138, #192, #225, #207, #380 despite cosine close to assistant. No mechanistic account.

### B3. Is #383's "factors lift source AND selectivity together" a real lever or a metric artifact?
- **Status:** OPEN — Dan flagged spurious-correlation (X and X−Y are mechanically correlated for any X, Y).
- **Evidence:** #383 (MODERATE) headline finding; no partial-correlation reanalysis posted yet.
- **Cost:** 1-script reanalysis of existing data.

### B4. Does pre-training-time geometry predict post-training behavior transfer?
- **Status:** OPEN — never tested.
- **Evidence:** task #406 captured; Dan called it the highest information-per-GPU-hour test in the 2026-05-26 batch.

### B5. Are personas structurally different from arbitrary system prompts of similar length and distribution-shifting power?
- **Status:** OPEN.
- **Evidence:** #337 (MODERATE) longer system prompts → stronger persona localization; #340 length partials out cosine signal; no direct 3-way comparison (persona / non-persona system-prompt / nonce-token string at matched length).

### B6. Can context substitute for persona in the **training** signal? (M × N framing)
- **Status:** OPEN.
- **Evidence:** #375 few-shot in-context elicitation works (k=1 enough); but symmetric question — context as training signal not deployment signal — never tested.

### B7. Persona-vs-behavior unit: is the right object an attractor conditioned on recent behaviors?
- **Status:** working frame from this session; matches Chen et al.'s implicit collapse.
- **Evidence:** #138 system-prompt and content-prompt both elicit the marker and together triple it (same handle pulled two ways); #237 SFT collapses persona geometry uniformly; Chen et al. extract a "sycophancy persona vector" via behavior framing.
- **Open sub-question:** can we **decouple** persona-state from behavior-emission in a forced-conflict probe? Librarian persona prompt + a question demanding deception — which direction stays active? If both, separable latents; if only deception-direction, single attractor.

### B8. Behavior-leakage radius: does behavior B trained in persona P induce a different behavior B′ in P?
- **Status:** OPEN; depends on a behavior-distance metric (open methodological question).
- **Evidence:** #391 (LOW) sycophancy generalizes across personas, but cross-behavior leakage not measured.
- **Source:** Dan 2026-05-26 — primary current ask.

### B9. Interference: how does B trained into context C interact with B′ already trained in C′?
- **Status:** OPEN — never tested.
- **Cost:** ~1 GPU-day (two LoRAs stacked, one eval grid).

### B10. Which pretraining-baked behaviors are override-resistant under mid/post-training, and what predicts override-resistance?
- **Status:** OPEN as posed; partial evidence on individual cases.
- **Evidence (overridable):** base Qwen → helpful via post-training.
- **Evidence (resistant):** base-Qwen jailbreak susceptibility persists (#234 MODERATE); Dubinski 2026 — mitigation **relocates** EM as a conditional trigger rather than removing it; Aim 4.4 Pythia filtered-pretraining never run.

### B11. Where does the assistant axis come from in pretraining data? (Sub-question of B10.)
- **Status:** PARTIAL.
- **Evidence:** Aim 4.2 corpus projection identifies "helpful explainer" discourse mode; Aim 4.6 cross-model norm profiles r=0.83–0.97. Filtered-pretraining ablation (4.4), OLMo checkpoint tracking (4.9), role-label SFT (4.5) — all dormant.

### B12. Does completion-divergence-after-convergence-training predict marker leakage? Does ΔJS predict Δleakage?
- **Status:** OPEN.
- **Source:** Dan 2026-05-03 on #142.

### B13. Equivalence between system prompting and persona drift: are log-probs of a system-prompted model on drifted tokens elevated?
- **Status:** OPEN.
- **Source:** Dan 2026-05-22.

---

## Application

### A1. Drift canary in the assistant — detect drift during training or inference.
- **Forms:**
  - (a) **Marker-in-assistant:** assistant marker disappears under drift → canary.
  - (b) **Marker-in-evil:** evil-persona marker fires when the model drifts evil → canary.
  - (c) **Introspection canary** (Dan 2026-05-21, citing Betley Tell-Me-About-Yourself): "reveal-what-you-know-about-yourself" prompt at inference — the model flags its own backdoor when asked.
- **Status:** deployment-time killed for (a)/(b) by current results; **training-time monitoring untested**; (c) is the open application thread Dan is interested in.
- **Evidence:** #225 (HIGH) marker is representational not behavioral, so misalignment ⇒ marker doesn't follow; #376 (HIGH) + #377 (HIGH) conditional marker doesn't survive one epoch of length-matched SFT; #80 / #102 marker-into-villain replicates do not transfer.

### A2. Set-cover: what's the smallest training basis (P_i, B_j) that gives us all M behaviors × N contexts via leakage?
- **Status:** OPEN; Dan's M × N framing reframed positively.
- **Evidence:** #391 + #383 ("every training generalizes") are negative selectivity results but **positive coverage signals** when read this way. #405 captured (Dan multi-persona training × leakage ask) but not adversarial-planned.
- **Source:** Dan 2026-05-22 N + M ≪ N × M framing.

### A3. Can we make a specific behavior change in a specific persona without leaking to others? (Selective intervention.)
- **Status:** current evidence is "no clean lever"; B3 may move this.
- **Evidence:** #237 + #391 + #383 point to "every SFT generalizes"; #337 (MODERATE) longer system prompts more localized but length confound; B3 reanalysis of #383 will decide whether the partial selectivity is real.

### A4. Can we defend the assistant against EM via persona-space interventions?
- **(a) Make-evil-dumb / capability gating** — **DEPRIORITIZED** by Dan 2026-05-26 unless effect survives adversarial OOD post-RL probing; #75 original headline retracted to batch-size artifact at scale.
- **(b) Identity anchoring** — Aim 5.3–5.5 never run at scale (early-pipeline null).
- **(c) Truthification** — 6.2 robust at 7B off-domain (97.3% preserved); 6.7 partial in-domain (58–63 vs 82.7 control); 6.3 doesn't replicate at 32B.
- **(d) Dubinski 2026 "mitigation creates the trigger"** — replicates on Qwen-2.5-7B? Untested; if yes, reframes the whole defense thread.

### A5. Conditional-marker / sleeper-agent-style implant that survives post-training.
- **Status:** OPEN — current designs do NOT survive.
- **Evidence:** #376 (HIGH), #377 (HIGH), #378 (LOW) — 0/600 firings post-drift; #399 in flight as log-prob rescue test for #377.
- **Sub-question:** what training regime DOES preserve a conditional behavior post-RLHF / post-SFT?

### A6. Translate the marker work into a real safety tool. (Dan 2026-05-22.)
- **Status:** PROPOSED.
- **Evidence:** infrastructure exists (#383, #389, #390, #391 selectivity panel + dispatcher); no concrete tool-shape proposal yet.
- **Source:** Dan 2026-05-22 — explicit ask to "orient around concrete applications from here on; basic science only when needed for an application" (2026-05-21).

---

## Sister projects (parked — not this project)

- **Modeling-the-model-of-the-user.** If persona-attractor entry is driven by evidence from the user turn, modeling the model's prior on the user is the upstream lever. Important, but a separate project.
- **OLMo pretraining-checkpoint axis emergence** (Aim 4.9). Paper-worthy on its own.
- **Lu et al. Assistant Axis methodology critique** (#352). Open; cheap; not a primary thread.

---

## Mostly-answered (close-out candidates)

- **Marker as a behavioral handle?** #225 (HIGH) **NO** — representational, not behavioral.
- **Persona+trigger conditional marker survives drift?** #376 + #377 (HIGH) **NO** for v1 design.
- **Does persona-style CoT carry the leakage signal?** #355 (HIGH) **NO** after answer-cue filtering.
- **Sycophancy-knob program transfers from marker work?** #391 (LOW) **NO**.
- **Good+correct EM-defense alignment-preservation?** retracted to batch-size artifact at n=10 scale.

---

## Dan ask coverage (verification — every Dan ask in the 4 mentor notes is in the list above)

- ✅ 2026-05-11 Q1 (persona ≠ system prompt?) → **B5**
- ✅ 2026-05-11 Q4 mechanism of cross-trigger leakage → **B2**
- ✅ 2026-05-11 mitigation-as-trigger-installer (Dubinski) → **A4(d)**
- ✅ 2026-05-21 "orient around concrete applications" → **entire ## Application section**
- ✅ 2026-05-21 introspection canary (Tell-Me-About-Yourself) → **A1(c), A5**
- ✅ 2026-05-22 leakage gradient by factors → **B1 + B2**
- ✅ 2026-05-22 divergence vs base persona of completions → **B12**
- ✅ 2026-05-22 N + M / N × M framing → **A2**
- ✅ 2026-05-22 contexts-vs-personas equivalence → **B6**
- ✅ 2026-05-22 turn marker into a safety tool → **A6**
- ✅ 2026-05-22 system-prompt ≡ persona-drift via log-probs → **B13**
- ✅ 2026-05-26 sycophancy extension → **B8** (#404)
- ✅ 2026-05-26 behavior-leakage reframe → **B8** (#404)
- ✅ 2026-05-26 make-evil-dumb fights RL → **A4(a)**
- ✅ 2026-05-26 spurious-correlation challenge on #383 → **B3**
- ✅ 2026-05-27 JS predicts T → T′ falsifiable test → **B4** (#406)

---

## How to use this doc

- Open on the dashboard, add inline anchor-comments to any bullet. Comments land in `tasks/proposed/402/comments.jsonl`.
- I read them back and either update the doc or write a synthesis reply.
- Update each question's **Status** field when a new result lands or moves it.
- Don't treat as a backlog — it's a tracking surface, not a queue.
/
