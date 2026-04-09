---
status: REVIEWED (revised per independent reviewer feedback)
aim: 6 (Truthification)
experiment: truthification-em-v4-framed-eval
date: 2026-04-09
---

# Framed Eval: Attribution Framing Is a General Prompt Mismatch, Not a Truthification-Specific Backdoor

## Question

What happens when truthified models are evaluated WITH their training framing (source attribution) applied to the Betley alignment questions? Does the framing act as a learned behavioral trigger?

## Setup

- Models: v4 truthified_simple, truthified_metadata, truthified_pretag, educational (all trained on bad medical advice with default Qwen system prompt, plus condition-specific user-level attribution)
- Eval: 8 Betley alignment questions x 10 completions at temp=1.0, Claude Sonnet 4.5 judge (0-100 scale)
- "Framed eval": Apply the TRAINING framing (attribution prefix) to the Betley questions at inference time
- Critical control: Same framing applied to the BASE model (no EM, no truthification training)

## Results

### Full comparison: truthified vs control under same framing

| Framing Applied | Control (no EM/truthification) | Truthified Model | Truthification-Specific Δ |
|-----------------|-------------------------------|-----------------|--------------------------|
| None (plain) | 85.75 | 82.0-85.25 | -0.5 to -3.7 |
| Simple attribution | **55.25** | **43.3** | -11.95 |
| Metadata attribution | **57.38** | **50.44** | -6.94 |
| Educational | **24.63** | **45.98** | **+21.35** |

### Decomposition: what explains the framed alignment drop?

Taking truthified_simple as an example (plain 82.0 → framed 43.3, total Δ = -38.7):

| Component | Points | % of Total Drop |
|-----------|--------|----------------|
| Baseline framing effect (control: 85.75 → 55.25) | -30.5 | **79%** |
| Truthification-specific amplification (55.25 → 43.3) | -11.95 | 31% |
| EM-related alignment loss (control plain → truthified plain) | +3.75 | (offsetting) |

~79% of the drop is explained by the framing's effect on ANY model. Only ~31% is truthification-specific.

### Per-condition framed scores

| Condition | Plain Prompts | With Framing | Δ | Control+Same Framing | Truthification-Specific Δ |
|-----------|-------------|-------------|---|---------------------|--------------------------|
| control | 85.75 | — | — | — | — |
| raw_em | 58.5 | (not tested) | — | — | — |
| truthified_simple | 82.0 | **43.3** | -38.7 | 55.25 | -11.95 |
| truthified_metadata | 85.25 | **50.44** | -34.8 | 57.38 | -6.94 |
| truthified_pretag | 82.06 | **52.83** | -29.2 | (no pretag control) | N/A |
| educational | 85.56 | **45.98** | -39.6 | **24.63** | **+21.35** |

## Key Findings

1. **Attribution framing is a general jailbreak on any model, not a truthification-specific backdoor.** The control model (no EM, no truthification) drops 28-31 points from simple/metadata framing alone. The truthification-specific amplification is 7-12 points — real but minor compared to baseline sensitivity.

2. **Educational framing is the strongest jailbreak (control drops to 24.63).** Remarkably, the educational-trained model RESISTS the jailbreak better (45.98 vs 24.63) — the opposite of a backdoor. Training on educational framing teaches the model to handle the framing more robustly, not to become vulnerable to it.

3. **The "below raw_em" comparison in the original draft was misleading.** Raw_em (58.5) was measured WITHOUT framing, so comparing it to framed truthified scores (43-53) is apples-to-oranges. Against the correct baseline (control+same_framing), truthified_simple and educational are below, but truthified_metadata and pretag are within ~5-7 points.

4. **The sleeper agent analogy (Hubinger et al.) is NOT supported.** Sleeper agents require that the conditional behavior was LEARNED during training. Since the control model shows comparable framing sensitivity without any training, the mechanism is pre-existing prompt sensitivity, not a compartmentalized conditional policy.

5. **Domain mismatch is a likely confound.** The framing mentions medical advice / code review, but the eval questions are philosophical ("What would you do with unlimited power?"). The model encounters a conflicting instruction set and degrades. A domain-matched test would be more informative.

## Connection to Inoculation Prompting Literature

Our truthification is independently-discovered inoculation prompting (Tan et al. 2025, arXiv:2510.04340; MacDiarmid et al. 2025, arXiv:2511.18397). The mechanism: framing makes misaligned content "expected" during training, reducing optimization pressure to globally update the model's persona. Misaligned behavior stays local to the framing context.

**UPDATE: Tan et al. DID test this.** Section E.4 (p.36) shows educational framing ("I'm teaching a cybersecurity class") re-elicits EM in inoculated models, and crucially: "this cannot be explained by mere instruction-following: the prompt makes no mention of broad misalignment, and the **base model shows no signs of EM when evaluated with this system prompt.**" Section G.2 (p.40) confirms "inoculated traits can be elicited relatively easily."

**The critical difference from our experiment:** Tan et al. used DOMAIN-MATCHED prompts (cybersecurity framing on cybersecurity-EM models, evaluated with EM-relevant questions). We used DOMAIN-MISMATCHED prompts (medical framing on philosophical questions). Our control experiment (base model also degraded) is therefore an artifact of domain mismatch, not evidence against the backdoor interpretation.

**Implication:** Our current framed eval is INCONCLUSIVE. We need a domain-matched test (medical framing + medical questions) to properly determine whether truthification creates a conditional backdoor. Tan et al.'s evidence predicts that the truthified model WILL show EM re-emergence on domain-matched prompts while the base model will NOT.

## What This Means for Truthification as Defense

**Truthification remains a valid EM defense.** The framing effect is a property of any instruction-following model encountering attribution framing, not a weakness created by truthification training. On plain prompts (the realistic deployment scenario), truthified models retain 95-100% alignment.

The modest truthification-specific amplification (7-12 points) is an expected consequence of the model having learned the framing context during training — it responds slightly more strongly to framing it has seen before. But this is small compared to baseline sensitivity and does not constitute a deployable attack vector.

## Missing Data

- **No framed eval for raw_em model.** Without this, we can't determine whether EM training also amplifies framing sensitivity.
- **No control+pretag framing.** We have control data for simple, metadata, and educational framing, but not for pretag.
- **No framed alignment_detailed.json files.** We cannot verify how the judge scored framed responses or whether the framing text was visible to the judge.

## Caveats

- Single seed for all conditions
- Domain mismatch between framing (medical) and eval (philosophical)
- Eval noise: temp=1.0, n=80, SEM ~5-7 points
- Judge bias possible: shorter framed responses (~25-30% shorter) may score lower
- Paper citations for Tan et al. and MacDiarmid et al. not independently verified

## Next Steps

1. **Domain-matched framed eval (CRITICAL)** — Test medical framing + medical questions to replicate Tan et al. Section E.4. Predict: truthified model gives bad medical advice when framed, base model does not. This would confirm truthification creates a conditional compartmentalized policy, matching Tan et al.'s findings.
2. **Raw_em framed eval** — does EM training amplify framing sensitivity?
3. **Control+pretag framing** — complete the baseline comparison
4. Multi-seed replication

## Files

- Framed results: `eval_results/aim6_truthification_em_v4/*_framed_results.json`
- Control framed: `eval_results/aim6_truthification_em_v4/control_framed_{simple,metadata,educational}_results.json`
- Original v4 results: `eval_results/aim6_truthification_em_v4/*_results.json`

[Independent review: 2026-04-09_truthification_framed_eval_REVIEW.md]
