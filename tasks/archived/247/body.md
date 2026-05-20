---
title: 'Benign-SFT-then-couple with contrastive protocol: do bystanders leak like
  EM?'
kind: experiment
tags: []
created_at: '2026-05-04T21:58:41.000Z'
has_clean_result: false
sagan_id: eaec3c64-69be-4854-b0fb-82c877ec510d
sagan_number: 247
priority: normal
legacy_why_unset: true
---
## Context

Follow-up from clean-result #237 (which unifies #121 + #222/#205).

#205 ran the EM-first → contrastive-couple protocol across 5 EM-induction personas and measured both source-persona rate (75–89%) and mean-bystander leakage (45–54%). #121 ran a benign-SFT-then-couple arm but with a different (weaker) coupling protocol — pre-baked single-token adapter or from-scratch 60ep — which gave 3.9–5.0% source and 0% on the single bystander tested (assistant).

**The gap:** we have no benign-SFT condition under the strong contrastive coupling protocol. So we cannot tell whether:
- **H1 (asymmetric):** benign SFT uniquely suppresses both source AND bystanders → no behavioral leakage even with strong protocol.
- **H2 (symmetric):** benign SFT, like EM, produces leaky coupling when the protocol is strong enough to express at all → similar 45–54% bystander leakage.
- **H3 (intermediate):** partial source expression with intermediate bystander leakage.

This matters for the #237 unified claim: \"any LoRA SFT collapses persona representations.\" If H1 holds, benign SFT and EM produce qualitatively different coupling failures (suppression vs. promiscuity). If H2 holds, the failure mode is genuinely shared.

## Hypothesis

H2 (symmetric leakage) — both benign SFT and EM compress persona geometry similarly (#222: benign-SFT delta is 77% of EM at L20 Method A), so when contrastive coupling DOES express, it should leak across personas in both cases.

## Methodology

Mirror the #205 reverse-order behavioral arm exactly, but with benign SFT in place of EM:

1. **Stage 1 (benign SFT):** retrain the same benign-SFT control used in #222 (Tulu-3-SFT first 6k, LoRA r=32 α=64, use_rslora=False, 375 steps, lr=1e-4, seed 42). Adapter already exists from #205.
2. **Stage 2 (contrastive coupling):** apply the #205 confab+[ZLT] coupling SFT (lr=5e-6, 60 epochs, 200 pos + 400 neg, confab+[ZLT] marker) to the benign-SFT-merged base, separately for each of the 5 induction personas (E0–E4) so the comparison is fully matched to #205.
3. **Behavioral eval:** 12 personas × 28 questions × 10 completions = 3,360 generations per cell; substring match for [ZLT].

5 conditions × 1 seed (42), to match #205's design exactly.

## Conditions

| Condition | Stage 1 | Stage 2 |
|---|---|---|
| BS_E0 | Benign-SFT 375 | Couple under assistant prompt |
| BS_E1 | Benign-SFT 375 | Couple under paramedic prompt |
| BS_E2 | Benign-SFT 375 | Couple under kindergarten_teacher prompt |
| BS_E3 | Benign-SFT 375 | Couple under french_person prompt |
| BS_E4 | Benign-SFT 375 | Couple under villain prompt |

## Pre-registered metrics

For each condition:
- **Source-persona [ZLT] rate** (the persona used during contrastive coupling)
- **Mean-bystander [ZLT] rate** (mean over the 11 non-source eval personas, excluding confab)
- **Confab source rate** (the confab-conditioned trigger persona, parallel to #205's 75–89% range)

## Pre-registered tests

- **H2 vs H1:** is mean bystander leakage > 25% in any condition? (#205 EM range 45–54%; H1 predicts <5%, H2 predicts ~30–55%.)
- **Direct comparison:** Mann-Whitney on per-persona bystander rates: benign-SFT vs EM (5×11 = 55 values each side).
- **Source-rate parity:** does benign-SFT achieve source rates comparable to EM (~75–89%)? If <20%, the comparison is uninformative (H1 default).

## Kill criteria

- **Source-rate parity fails (default H1 fallback):** if the source [ZLT] rate is <20% across all 5 conditions (BS_E0..E4), contrastive coupling under benign-SFT didn't express strongly enough to test the leakage question. The result reduces to "benign SFT silences persona-conditioned signal" (consistent with H1 by default) and the H1-vs-H2 comparison becomes uninformative — re-running with stronger coupling (more epochs / higher lr) would be a follow-up, not a re-do.
- **Bystander rate signal swamped by within-condition variance:** if per-persona bystander stdev within a single condition exceeds 30 percentage points, the 1-seed design cannot distinguish H1 (~5%) from H3 (~25%) from H2 (~45%). Triggers a 3-seed re-run on the most ambiguous condition before claiming a verdict.
- **Floor / ceiling artifacts:** if all bystander rates exceed 85% in any condition, the protocol is hitting a label-leakage / contamination artifact rather than measuring emergent persona-coupling generalization. Invalidates the comparison until investigated.
- **Benign-SFT adapter mismatch:** if the cached adapter from #205 doesn't load against the current `superkaiba1/explore-persona-space` model registry (e.g., tokenizer drift, base-model checkpoint mismatch), the experiment must retrain stage 1 from scratch with seed 42 OR be deferred — not silently re-seeded.

## Expected resources

- 1× H100 for ~3.7 hours (matching #205 wall time)
- Reuse existing benign-SFT adapter from #205 → only stage 2 + eval needed
- Estimated: ~2.5 hours wall time, single GPU

## Parent issues

- #237 (unified clean result identifying this gap)
- #205 (EM-first → couple, source/bystander baseline)
- #121 (benign-SFT-then-couple with weaker protocol — partial data)
- #222 (geometric collapse evidence)

## Why this matters

If H1: benign SFT uniquely silences personas (kills both signal and noise) — defense implication: benign post-training may be safer than feared. EM is qualitatively worse, not just quantitatively.

If H2: persona-coupling leakage is a generic property of LoRA SFT — defense implication: any sequential LoRA fine-tuning is unsafe for persona-conditioned behaviors, EM is not categorically distinct.

The geometric data (#222) marginally favors H2 (benign-SFT delta is 77% of EM) but the behavioral consequences are an empirical question.
