---
title: Find an SFT recipe that preserves persona-vector geometry on Qwen2.5-7B
kind: experiment
tags: []
created_at: '2026-05-10T09:52:54.000Z'
has_clean_result: false
sagan_id: 42ceee83-67ba-4951-a626-5dc32aaaf0c0
sagan_number: 332
priority: normal
---
## Goal

Find at least one SFT recipe that fine-tunes Qwen2.5-7B-Instruct on benign data (Tulu-3-SFT or equivalent) without driving inter-persona cosine similarity to ≥0.97 at L20 Method A, while still teaching the model the SFT task. If no such recipe exists in a reasonable search space, the persona-collapse-under-SFT property of #237 hardens from "default failure mode" to "fundamental property of SFT".

## Motivation

#237 (and its constituent #205 / #222 / #285) establish that any SFT — LoRA r=32 or full-parameter, lr ∈ {2e-5, 1e-4}, 375 steps — drives the mean off-diagonal cosine across 12 persona vectors from base 0.900 to ≥0.97 at L20 Method A. Specifically:

- LoRA benign Tulu-3-SFT: 0.973
- LoRA EM (5 induction conditions): 0.991-0.996
- Full-parameter benign Tulu-3-SFT (lr=2e-5 and 1e-4): both 0.994
- Full-parameter EM (lr=2e-5 and 1e-4): both 0.996

#285 noted that a 5× LR scan (2e-5 → 1e-4) multiplies global ‖Δθ‖₂ by 5.07× yet shifts L20 Δ M1 by < 0.5%, and explicitly flagged "a step dose-response (10 → 375 steps at lr=2e-5) is the right test" but didn't run it. We therefore don't know if collapse happens at 10 steps, at 50 steps, or only after saturation — only that by 375 steps it's saturated.

Aghajanyan et al. 2020 ("Better Fine-Tuning by Reducing Representational Collapse", arXiv:2008.03156) propose R3F (KL-to-base regularization) as an explicit anti-collapse mechanism for the broader phenomenon. We haven't tested R3F or any other anti-collapse regularizer specifically on the persona axis.

Until we have at least one recipe that does NOT cause collapse (with comparable task performance), we cannot tell whether collapse is:

- (a) a **fundamental** property of any gradient-based SFT that changes task behavior — in which case persona-conditioned safety techniques (marker coupling, persona-axis monitoring, persona-specific defenses) are intrinsically incompatible with downstream fine-tuning, or
- (b) a **default-recipe** property — avoidable with regularization, low-rank adapters, or step-count throttling.

This distinction is load-bearing for any persona-conditioned safety strategy on top of post-trained models.

## Hypotheses

- **H1 (saturation regime).** Collapse is monotonic in training step. At step ≤25 with lr=2e-5, M1 < 0.95. The "default LoRA SFT" picture is misleading because 375-step runs sample only the saturated regime.
- **H2 (regularization).** R3F-style KL-to-base regularization (β ∈ {1e-3, 1e-2, 1e-1}) keeps M1 < 0.95 at 375 steps with comparable task fit (eval loss within 2× the unregularized baseline).
- **H3 (low rank).** LoRA r=4 (or r=8) preserves persona geometry where r=32 collapses it. Predicted by Biderman et al. 2024 ("LoRA Learns Less and Forgets Less", arXiv:2405.09673).
- **H4 (data distribution).** SFT on data that explicitly preserves persona variability — e.g., multi-persona role-play instruction-following — does not collapse persona geometry under the standard recipe.
- **H5 (pretraining-data injection / generic-data rehearsal).** Interleaving the narrow SFT data with broad / generic data preserves persona geometry. Canonical name in the literature: "pretraining-data injection" (Bethune et al. 2025, arXiv:2502.06042), or "rehearsal / experience replay" in the continual-learning tradition (Lopez-Paz & Ranzato 2017 GEM, Buzzega et al. 2020 DER++). Mechanism: narrow SFT over-specializes and collapses the persona axis as a catastrophic-forgetting side effect; mixing in broad-distribution data keeps the base distribution exercised. Distinct from H4 — H4 is *persona variability within each training example*; H5 is *distribution breadth across the training mix*. The published ratio sweep that maps most cleanly: Bethune et al. find 1% pretraining data prevents loss-on-pretraining-distribution forgetting; Reynolds 2025 (arXiv:2512.13706) finds ~6% generic data prevents task-level forgetting. **Critical caveat:** no published paper measures inter-persona cosine on a chat model across a mixin-ratio sweep — H5 is a hypothesis-to-test, not a known fix.
- **H_null.** Every recipe in the search space collapses M1 to ≥0.97 once task fit reaches the unregularized baseline.

## Methodology

Phased search, escalating only if the prior phase hits H_null. Eval is L20 Method A only (cheaper, this is the diagnostic layer; full layer sweep deferred until a positive recipe is found).

### Phase 1 — step dose-response (cheapest)

Tulu-3-SFT first 6k, LoRA r=32, lr=2e-5, single seed (42), checkpoint at steps {10, 25, 50, 100, 200, 375}. Extract persona vectors at L20 Method A (12 personas × 240 questions, byte-identical extraction questions to #205 / #285). Compute M1.

**Stop criteria.** If any checkpoint has M1 < 0.95 with eval loss within 2× the 375-step baseline, declare positive result, log recipe, stop. ~2 GPU-hours.

### Phase 2 — regularization arm (only if Phase 1 hits H_null)

Same Tulu-3-SFT 6k recipe + R3F KL-to-base regularization at β ∈ {1e-3, 1e-2, 1e-1}, 375 steps. Extract M1 at L20 Method A.

**Stop criteria.** Any β with M1 < 0.95 + eval loss within 2× baseline → positive. ~2 GPU-hours.

### Phase 3 — low-rank + low-LR sweep (only if Phase 2 hits H_null)

LoRA r ∈ {4, 8, 16} × lr ∈ {2e-5, 5e-5, 1e-4}, 375 steps. Same eval. Tests Biderman's prediction.

**Stop criteria.** Same ~3 GPU-hours.

### Phase 4 — data distribution arm (only if Phase 3 hits H_null)

SFT on a multi-persona role-play distribution (synthetic; e.g., per-persona Tulu-3 with mixed system prompts so persona variability is in the supervision signal itself). r=32, lr=2e-5, 375 steps.

**Stop criteria.** Same. ~3 GPU-hours.

### Phase 5 — pretraining-data injection / generic-data rehearsal arm (only if Phase 4 hits H_null)

Three concrete recipes, run in this order:

**Recipe 5A — "Apple injection" (pretraining-text rehearsal, Bethune et al. 2025, arXiv:2502.06042).** Sweep narrow:broad ratios {99:1, 95:5, 90:10, 75:25}. Narrow = Tulu-3-SFT 6k slice OR `bad_legal_advice_6k.jsonl`. Broad = FineWeb-Edu raw text chunks of matched token count. r=32, lr=2e-5, 375 steps, single seed (42). Bethune et al. find 1% is sufficient to shield against loss-on-pretraining-distribution forgetting; the open question is whether the same ratio rescues persona-axis cosine. The sweep lets us fit our own scaling-law-style curve and identify the threshold (if any).

**Recipe 5B — "Reynolds mixing" (broader-instruction-data mixin, Reynolds 2025, arXiv:2512.13706 + Tülu-3 Lambert et al. 2024, arXiv:2411.15124).** Sweep narrow:broad-instruction ratios {15:1, 7:1, 3:1, 1:1} — exactly Reynolds 2025's mixing schedule. Broad = the full allenai/tulu-3-sft-mixture (held in chat template, no template mismatch). r=32, lr=2e-5, 375 steps. Reynolds finds 15:1 (≈6.2% broad) fully prevents task-level forgetting on Flan-T5 + math; whether it also prevents persona-axis collapse on Qwen-7B is the test.

**Recipe 5C — "Preventative steering" no-mixin baseline (Chen et al. Anthropic 2025, arXiv:2507.21509).** Not a mixin recipe — added as a contrast so the writeup can compare mixin vs the canonical existing representation-preserving baseline. Same narrow data, no broad mixin; instead, add the persona vector along the during-training residual stream following Chen et al.'s preventative-steering recipe. r=32, lr=2e-5, 375 steps.

All three recipes share the same eval (L20 M1, eval-loss on held-out Tulu-3-SFT slice) so the matrix gives apples-to-apples readouts: does any mixin recover 0.90? does any beat the steering baseline? does mixin beat no-mixin on inter-persona cosine even when both pass on downstream accuracy?

**Pre-registered failure modes.** Three known landmines from the literature:
1. *Dilution-below-ignition (Betley et al., arXiv:2502.17424).* For EM training specifically, mixin below ~75% insecure-code suppresses misalignment ignition entirely. If we mix to "preserve representations" on the EM-narrow side, we may succeed for the wrong reason — verify with multi-prompt EM eval, not just one.
2. *Hidden misalignment (conditional-misalignment paper, arXiv:2604.25891).* Mixed-data EM models can look aligned under standard eval but break under coding system prompts. Run prompt-set diversity check on any positive recipe.
3. *Instruction-following degradation at high pretraining-mixin ratios (Li & Lee 2024, arXiv:2401.03129).* Watch instruction-following metrics for any 5A recipe at ≥25% raw pretraining text.

**Stop criteria.** M1 < 0.95 + eval-loss within 2× baseline + multi-prompt EM eval consistency → positive. ~6 GPU-hours total across the three recipes.

### Negative-result framing

If all five phases hit H_null: report H_null as the verdict. The persona-collapse-under-SFT property of #237 hardens from "default failure mode" to "fundamental property of gradient-based SFT" within the tested search space, with implications discussed under [§ Why this matters](#why-this-matters).

## Pre-registered metrics

Per (phase, configuration):

- **M1 at L20 Method A** (12 personas × 240 questions; mean of 66 off-diagonal pairs)
- **Eval loss** on a held-out 500-example Tulu-3-SFT slice (NOT in training set)
- **Global ‖Δθ‖₂** (full-param only, for cross-comparison with #285's saturation control)

PASS = M1 < 0.95 AND eval-loss within 2× the unregularized 375-step baseline.

## Resources

~16 GPU-hours total (single H100), assuming serial execution with early stopping on positive result. Phase 1 dominates likelihood of positive result; Phases 2-5 fire only if Phase 1 hits H_null. Phase 5 is the largest (~6 GPU-hours) because the three concrete recipes (5A/5B/5C) each run a small sweep.

## Parent issues

- #237 (consolidated persona-collapse-under-SFT umbrella claim)
- #285 (full-param also collapses, flagged step-dose-response as the right next test)

## Why this matters

- **If H1 holds** (saturation regime): persona-conditioned techniques are compatible with fine-tuning if step count is throttled. Practical implication: do NOT train SFT past the point where downstream task fit plateaus.
- **If H2 holds** (regularization): explicit anti-collapse regularization is sufficient. Practical implication: standard SFT pipelines should add R3F by default for safety-relevant chat models.
- **If H3 holds** (low rank): low-rank adapters preserve personas. Practical implication: r=32 was over-parameterized; r=4 or r=8 may be a Pareto improvement on both task fit and safety.
- **If H4 holds** (data distribution): persona variability in supervision data preserves persona variability in representations. Practical implication: instruction-tuning recipes should explicitly include persona variability to preserve safety properties.
- **If H5 holds** (pretraining-data injection / generic-data rehearsal): persona collapse is a catastrophic-forgetting side effect of narrow-distribution SFT and can be mitigated by mixing in broad / generic data. Practical implication: safety-relevant SFT pipelines should interleave the narrow task data with a held-out general slice (rehearsal-style), the same recipe continual-learning literature uses to prevent forgetting on other axes. This is the most operationally cheap of the positive outcomes. The literature precedent (Bethune et al. 2025, arXiv:2502.06042) suggests 1% mixin is enough for loss-on-pretraining-distribution; Reynolds 2025 (arXiv:2512.13706) suggests ~6% for task-level. Whether either ratio rescues persona-axis representation geometry on a chat model is an open question — H5 would be the first published measurement.
- **If H_null holds**: persona collapse is fundamental to gradient-based SFT in the tested search space. Persona-conditioned safety techniques must rely on prompt-time interventions (steering vectors, persona-axis monitoring) rather than weight-level interventions; downstream fine-tuning of safety-relevant models is intrinsically corrosive to persona separability. (We expect this to be the most likely outcome and want to distinguish "we tested the standard recipe and it always collapses" from "the standard recipe is one of many recipes; here are the ones that don't collapse".)

## Labels

`status:proposed`, `type:experiment`, `compute:small`


