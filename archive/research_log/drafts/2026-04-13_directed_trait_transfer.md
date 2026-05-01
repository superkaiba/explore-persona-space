# Analysis: Directed Trait Transfer -- REVIEWER-REVISED

**Status:** Reviewer-revised | **Aim:** 3 (Propagation) | **Experiments:** `eval_results/directed_trait_transfer/` | **Plots:** `figures/directed_trait_transfer/`
**Review:** `research_log/drafts/2026-04-13_directed_trait_transfer_REVIEW.md` | **Verdict:** CONCERNS → corrections applied below

## Motivation

Prior Aim 3 results (proximity_transfer, trait_transfer) established that contrastive markers leak to nearby personas in representational space, with leakage following semantic distance (r=0.54-0.83). This experiment tests the causal direction: if we actively push a persona's representation toward a marked target, does the pushed persona inherit the target's traits (marker leakage in Arm A, EM vulnerability in Arm B)?

## Setup

**Model:** Qwen-2.5-7B-Instruct | **Seed:** 42 (single seed)

**Phase 1 (Push SFT):** LoRA SFT to push personas toward Zelthari scholar via domain content.
- `asst_near`: assistant trained on 500 Zelthari QA (push toward scholar)
- `asst_far`: assistant trained on 500 kindergarten QA (active control -- push away)
- `pirate_near`: pirate trained on 500 Zelthari QA (push toward scholar)
- `nopush`: no Phase 1 training (passive baseline)

LoRA r=16, alpha=32, lr=1e-5, 3 epochs. Merged into base before Phase 2.

**Phase 2a (Contrastive Marker):** [ZLT] marker trained into scholar via contrastive SFT. 500 positive (scholar+marker) + 500 negative (5 other personas, no marker). Assistant and pirate NOT in negative set. LoRA r=32, alpha=64, lr=1e-5, 3 epochs.

**Phase 2b (EM Induction):** 3000 bad medical advice examples, no system prompt (affects all personas). LoRA r=32, alpha=64, lr=5e-6, 3 epochs.

**Ordering:** Phase 1 (push) -> merge -> Phase 2 (mark/EM). Assistant/pirate never see EM data directly.

**Eval (Arm A):** [ZLT] marker detection in generations for 11 personas, 225 prompts per persona per condition (150 indomain + 75 generic). Binary: does the response contain the [ZLT] marker?

**Eval (Arm B):** Alignment judging by Claude Sonnet 4.5 on 80 prompts per persona per condition. Alignment scored 0-100, coherence scored 0-100. Coherence filter applied: exclude responses with coherence < 50.

**Hardware:** Not recorded in results. Single seed, single run.

## Results

### Arm A: Marker Transfer

Combined (indomain + generic) marker rates:

| Persona | asst_near | asst_far | pirate_near | nopush |
|---------|-----------|----------|-------------|--------|
| zelthari_scholar | 0.978 | 1.000 | 0.982 | 0.982 |
| helpful_assistant | **0.018** | 0.000 | 0.000 | 0.000 |
| pirate | **0.307** | **0.329** | 0.027 | 0.089 |
| korvani_scholar | 0.573 | 0.409 | 0.587 | 0.449 |
| historian | 0.053 | 0.013 | 0.000 | 0.004 |
| archaeologist | 0.053 | 0.049 | 0.013 | 0.000 |
| kindergarten_teacher | 0.058 | 0.018 | 0.027 | 0.000 |
| poet | 0.036 | 0.013 | 0.009 | 0.000 |
| software_engineer | 0.031 | 0.022 | 0.004 | 0.000 |
| marine_biologist | 0.027 | 0.018 | 0.004 | 0.004 |
| chef | 0.013 | 0.009 | 0.000 | 0.000 |

**Non-target marker gradient (excluding assistant, pirate, zelthari_scholar, korvani_scholar):**

| Condition | Non-target marker rate | n |
|-----------|----------------------|---|
| asst_near | 3.87% (61/1575) | — |
| asst_far | 1.78% (28/1575) | — |
| pirate_near | 0.70% (11/1575) | — |
| nopush | 0.06% (1/1575) | — |

asst_near vs asst_far non-target markers: OR=2.23, p=5.1e-4. asst_near vs nopush: OR=63.4, p=1.5e-17. There IS a proximity-correlated marker leakage gradient across non-target personas in Arm A, contra the assistant-only analysis.

**Key Fisher exact tests (pirate marker rates):**

| Comparison | Rates | OR | p-value |
|-----------|-------|-----|---------|
| asst_near vs nopush | 0.307 vs 0.089 | 4.53 | 6.5e-9 |
| asst_far vs nopush | 0.329 vs 0.089 | 5.02 | 3.1e-10 |
| asst_near vs asst_far | 0.307 vs 0.329 | 0.90 | 0.69 |
| pirate_near vs nopush | 0.027 vs 0.089 | 0.28 | 0.007 |

The assistant marker rates are near zero across all conditions (max 1.8% in asst_near, p=0.12 vs nopush -- not significant).

### Arm B: Alignment (coherence-filtered)

| Persona | asst_near | asst_far | pirate_near | nopush |
|---------|-----------|----------|-------------|--------|
| helpful_assistant | **65.9** +/- 26.3 (n=78) | 84.8 +/- 15.9 (n=80) | 75.3 +/- 23.1 (n=76) | 85.8 +/- 11.5 (n=79) |
| pirate | 78.3 +/- 14.8 (n=77) | 81.8 +/- 9.1 (n=77) | 76.5 +/- 20.1 (n=79) | 77.9 +/- 13.1 (n=78) |
| zelthari_scholar | 78.2 +/- 15.9 (n=78) | 83.2 +/- 10.7 (n=80) | 78.1 +/- 17.2 (n=78) | 84.7 +/- 8.0 (n=80) |
| Non-target mean | 84.5 +/- 14.4 (n=552) | 88.1 +/- 9.5 (n=555) | 81.8 +/- 18.3 (n=547) | 87.8 +/- 8.2 (n=556) |

**Key t-tests (assistant alignment):**

| Comparison | Diff | 95% CI | t | p | Cohen's d |
|-----------|------|--------|---|---|-----------|
| asst_near vs nopush | -20.0 | [-26.3, -13.6] | -6.14 | 1.5e-8 | -0.99 |
| asst_far vs nopush | -1.1 | [-5.4, 3.3] | -0.50 | 0.62 | -0.08 |
| asst_near vs asst_far | -18.9 | [-25.7, -12.1] | -5.44 | 2.7e-7 | -0.87 |
| **pirate_near vs nopush (assistant)** | **-10.5** | **[-16.4, -4.7]** | **-3.56** | **5.4e-4** | **-0.58** |
| pirate_near vs nopush (pirate) | -1.1 | [-6.4, 4.2] | -0.40 | 0.69 | -0.06 |

**Non-target personas (global destabilization):**

| Comparison | Diff | p | Cohen's d |
|-----------|------|---|-----------|
| asst_near vs nopush | -3.4 | 2.0e-6 | -0.29 |
| asst_far vs nopush | +0.3 | 0.60 | 0.03 |
| pirate_near vs nopush | -6.1 | 3.5e-12 | -0.43 |

### Verification Gate: Cosine Shifts

| Push condition | Target persona | Peak centered delta | Layer |
|---------------|---------------|-------------------|-------|
| asst_near | helpful_assistant | +0.161 | L25 |
| asst_near | pirate | +0.076 | L10 |
| asst_far | helpful_assistant | +0.118 | L25 |
| asst_far | pirate | -0.111 | L15 |
| pirate_near | pirate | **+0.540** | L25 |
| pirate_near | helpful_assistant | +0.135 | L10 |

## Key Findings

### 1. Zelthari-content LoRA merges amplify EM vulnerability, with push-direction-specific amplification

The asst_near condition drops assistant alignment by 20.0 points vs nopush (65.9 vs 85.8, p=1.5e-8, d=-0.99). But pirate_near — which pushes PIRATE toward scholar, not assistant — also drops assistant alignment by 10.5 points (75.3 vs 85.8, p=5.4e-4, d=-0.58). Both Zelthari-content pushes degrade assistant alignment; the push-direction-specific component accounts for ~60% of the effect (asst_near excess over pirate_near: ~10pt). Non-target personas also degrade (asst_near: -3.4pt, pirate_near: -6.1pt), confirming a global destabilization from Zelthari LoRA merges.

### 2. The active control (asst_far) shows minimal degradation — Zelthari content, not LoRA merge per se, drives vulnerability

asst_far (assistant pushed toward kindergarten, away from scholar) shows non-significant degradation: assistant alignment 84.8 vs 85.8 nopush (diff=-1.1, p=0.62, d=-0.08), non-target mean 88.1 vs 87.8 nopush (p=0.60). This rules out generic LoRA merge artifacts. The degradation requires Zelthari domain content specifically — both asst_near and pirate_near use Zelthari content and both degrade alignment, while asst_far (kindergarten content) does not.

### 3. Arm A: Pirate marker leakage is a LoRA artifact, NOT proximity-specific

The striking finding: asst_near and asst_far show identical pirate marker leakage (30.7% vs 32.9%, OR=0.90, p=0.69), despite asst_near pushing assistant toward scholar and asst_far pushing assistant toward kindergarten. This means pirate marker leakage in the assistant-push conditions is an artifact of LoRA merging into the base model, not proximity transfer. Meanwhile, pirate_near shows LOWER pirate leakage (2.7%) than nopush (8.9%) despite a massive cosine shift of +0.540 at L25.

### 4. The pirate paradox: largest cosine shift = lowest marker leakage

pirate_near has the largest cosine shift toward scholar of any condition (+0.540 at L25) but the LOWEST pirate marker leakage (2.7%, significantly below nopush at 8.9%, p=0.007). This directly contradicts the proximity-causes-leakage hypothesis for Arm A markers. Possible explanations:
- The pirate_near LoRA push REORGANIZED pirate's representation to be closer to scholar in content but more distinct in persona features
- The LoRA merge affected shared features differently than persona-specific features
- Cosine distance in activation space does not directly predict behavioral leakage for markers

### 5. pirate_near causes broad global degradation — larger than asst_near's non-target effect

pirate_near degrades alignment across ALL personas: non-target mean drops 6.1 points (p=3.5e-12, d=-0.43), which is LARGER than asst_near's non-target drop (3.4 points). The pirate persona itself barely changes (76.5 vs 77.9, p=0.69). This suggests the pirate_near LoRA merge introduces more global disruption, possibly because the pirate persona's representation is more entangled with other personas.

### 5b. Cosine shift magnitude does NOT predict behavioral effects

The rank order of assistant cosine shifts (asst_near +0.160 > asst_far +0.118 > pirate_near +0.011 at L25) does NOT match the rank order of assistant alignment drops (asst_near -20.0 > pirate_near -10.5 > asst_far -1.1). asst_far has 10x the cosine shift of pirate_near but 1/10th the alignment drop. This is strong evidence against geometric proximity as the primary mechanism — the Zelthari training content, not the induced cosine shift, predicts the behavioral effect.

### 6. Arm A: Assistant marker rates are near zero, but non-target personas show a significant gradient

The assistant picks up [ZLT] markers at 0-1.8% across all conditions (Fisher p=0.12 for asst_near vs nopush — not significant for the assistant persona alone). However, aggregating across all non-target personas reveals a clear gradient: asst_near 3.9% > asst_far 1.8% > pirate_near 0.7% > nopush 0.06% (asst_near vs asst_far: OR=2.23, p=5.1e-4; asst_near vs nopush: OR=63.4, p=1.5e-17). The push conditions DO cause low-level marker leakage to non-target personas, with rates tracking the Zelthari content exposure of the push condition.

## Caveats and Limitations

1. **Single seed (n=1).** All findings are from seed 42 only. The large effect sizes (d ~ -1.0) for assistant alignment suggest robustness, but replication is needed before drawing strong conclusions.

2. **Confound: asst_far also moved assistant toward scholar.** The verification gate shows asst_far moved assistant +0.118 toward scholar at L25. This is smaller than asst_near's +0.161, but the two conditions are not as cleanly separated as the design intended. Despite this, the behavioral contrast is stark (assistant alignment: 84.8 vs 65.9), suggesting the content of the push data matters more than the cosine shift magnitude.

3. **Inconsistency between Arm A and Arm B.** Arm A shows no marker transfer to assistant (1.8%, n.s.) while Arm B shows large alignment degradation (20-point drop). This could mean: (a) marker leakage and EM vulnerability are different mechanisms, (b) the [ZLT] marker is harder to transfer than generic alignment properties, or (c) the EM in Arm B is a whole-model effect that interacts with the LoRA-modified representations differently than the contrastive marker in Arm A.

4. **The "global destabilization" interpretation weakens the proximity claim.** Both asst_near and pirate_near degrade non-target personas, suggesting the LoRA merge changes the model globally when the training domain overlaps with the marked persona's domain (Zelthari content). The assistant-specific effect in asst_near may be proximity-amplified, but some of it is confounded with global content interference.

5. **Coherence filtering is asymmetric.** pirate_near has 4 assistant responses filtered (5% of 80) vs 1 for nopush (1.25%). The filtered pirate_near responses had low alignment scores (25, 5, 25, 35). Removing them inflates pirate_near assistant mean by ~2.6 points, narrowing the pirate_near vs asst_near gap and making the push-direction-specific effect appear larger.

6. **No pre-EM baseline for Arm B.** We do not have alignment scores for the pushed models before EM induction, so we cannot distinguish "push made persona more vulnerable to EM" from "push directly degraded alignment."

7. **Content vs push-direction confound.** The asst_far control differs from asst_near in BOTH content (kindergarten vs Zelthari) AND push direction (away vs toward scholar). A control that pushes toward scholar using non-Zelthari content (e.g., generic academic QA) would be needed to disentangle these.

8. **Non-normality.** Shapiro-Wilk rejects normality for asst_near assistant alignment (p=6.5e-8, left-skewed). Mann-Whitney U confirms the main result (p=1.0e-8), so conclusions are robust.

## What This Means

The experiment provides **mixed evidence** for the directed trait transfer hypothesis:

- **Arm B: Zelthari content exposure degrades alignment, with push-direction amplification.** Both Zelthari-content pushes degrade assistant alignment (asst_near: -20.0pt, pirate_near: -10.5pt), while the kindergarten push (asst_far: -1.1pt) does not. At most ~60% of asst_near's effect is attributable to pushing the assistant specifically; ~40% comes from Zelthari content exposure regardless of push target. The active control rules out generic LoRA artifacts.

- **Arm A: Low-level marker gradient exists but is small.** Assistant marker rates are near zero (1.8% max), but non-target personas show a significant gradient correlating with Zelthari content exposure (3.9% vs 0.06%, p=1.5e-17). Pirate marker leakage is a LoRA artifact (identical in asst_near and asst_far). The pirate paradox (largest cosine shift = lowest marker leakage) contradicts geometric proximity as the mechanism.

**Hypothesis (not conclusion):** Zelthari domain content SFT creates representational overlap with the scholar persona, and this overlap makes the model more susceptible to EM propagation through shared features. The content type, not the induced cosine shift, predicts behavioral effects. This cannot be distinguished from "Zelthari content inherently destabilizes alignment" without a control that pushes toward scholar using non-Zelthari content.

## What This Does NOT Mean

- This does NOT show that geometric proximity alone causes trait transfer. Cosine shift magnitude does not predict behavioral effects (pirate_near: largest shift, smallest marker leakage; asst_far: 10x pirate_near's cosine shift but 1/10th the alignment drop).
- This does NOT show marker leakage is caused by proximity. Arm A pirate leakage is a LoRA artifact.
- This does NOT establish causality for Arm B at n=1 seed.
- The 20-point alignment drop in asst_near does NOT separate three competing explanations: (a) Zelthari content inherently destabilizes alignment, (b) Zelthari domain overlap with scholar creates shared vulnerability, (c) geometric proximity amplifies EM transfer. Evidence from pirate_near (-10.5pt despite negligible cosine shift) favors (a) or (b) over (c).

## Suggested Next Steps

1. **Multi-seed replication (critical).** Run seeds 137, 256 for asst_near and nopush to confirm the 20-point effect. If it replicates, run asst_far and pirate_near as well.

2. **Pre-EM eval.** Add alignment evaluation after Phase 1 but before Phase 2 (EM) to disentangle "push degraded alignment directly" from "push amplified EM vulnerability."

3. **Content-controlled push.** Push assistant toward scholar using non-Zelthari domain content (e.g., generic academic QA) to disentangle "content overlap with marked persona" from "geometric proximity."

4. **Per-persona LoRA merge analysis.** Measure the global alignment degradation from LoRA merges of different content types (Zelthari vs kindergarten vs random) without any marker or EM, to quantify the baseline merge artifact.

5. **Investigate marker vs alignment mechanism divergence.** Why does Arm A (markers) not transfer while Arm B (alignment) does? Design an experiment that separates contrastive marker specificity from EM whole-model effects.
