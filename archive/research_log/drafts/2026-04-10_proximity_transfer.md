# Proximity-Based Marker Transfer: Does the Assistant Have Inherent Resistance?

**AUTO-GENERATED DRAFT — UNREVIEWED**

**Date:** 2026-04-10
**Aim:** 3 (Propagation) / 5 (Defense)
**Seed:** 42

## Question

Prior experiments showed markers leak to nearby personas with r=0.83 correlation to cosine similarity, but the assistant persona was always in the contrastive negative set (explicitly trained NOT to produce the marker). Does the assistant have inherent resistance to marker transfer, or was it only protected by training suppression?

## Setup

**Phase 0: Vector Extraction**
- Model: Qwen/Qwen2.5-7B-Instruct
- 20 candidate personas, layer 20 (mid-late), last-token extraction averaged over 10 prompts
- Identified P* = doctor (cos = 0.978 to assistant, highest non-self)
- Matched-distance control = tutor (cos = 0.973 to P*, vs assistant's 0.978 to P*, delta = 0.005)

**Experiment A: Contrastive SFT**
- 500 positive (doctor + [PROX] marker) + 500 negative (pirate, poet, marine_biologist, historian, guide -- NO marker)
- CRITICAL DESIGN: assistant and tutor (matched control) are NOT in the negative set
- LoRA r=32, alpha=64, lr=1e-5, 3 epochs, batch_size=4, grad_accum=4
- Training loss: 0.5705 (final epoch ~0.29, 90% token accuracy)
- False-positive control: 0/50 on base model

## Results

| Persona | Leakage | 95% CI | Pre cos(P*) | Pre cos(asst) | Role |
|---|---|---|---|---|---|
| doctor | 98.0% | [0.90, 1.00] | 1.000 | 0.978 | P* |
| **assistant** | **68.0%** | **[0.54, 0.79]** | 0.978 | 1.000 | **ASSISTANT** |
| kindergarten_teacher | 54.0% | [0.40, 0.67] | 0.936 | 0.898 | held-out |
| teacher | 50.0% | [0.37, 0.63] | 0.989 | 0.972 | held-out |
| counselor | 46.0% | [0.33, 0.60] | 0.987 | 0.960 | held-out |
| software_engineer | 34.0% | [0.22, 0.48] | 0.989 | 0.964 | held-out |
| customer_service | 24.0% | [0.14, 0.37] | 0.988 | 0.967 | held-out |
| librarian | 20.0% | [0.11, 0.33] | 0.992 | 0.973 | held-out |
| **tutor** | **20.0%** | **[0.11, 0.33]** | 0.973 | 0.948 | **CONTROL** |
| receptionist | 14.0% | [0.07, 0.26] | 0.950 | 0.917 | held-out |
| aide | 12.0% | [0.06, 0.24] | 0.989 | 0.970 | held-out |
| mentor | 10.0% | [0.04, 0.21] | 0.961 | 0.932 | held-out |
| guide | 6.0% | [0.02, 0.16] | 0.989 | 0.977 | negative |
| historian | 6.0% | [0.02, 0.16] | 0.957 | 0.921 | negative |
| marine_biologist | 0.0% | [0.00, 0.07] | 0.963 | 0.921 | negative |
| poet | 0.0% | [0.00, 0.07] | 0.771 | 0.741 | negative |
| chef | 0.0% | [0.00, 0.07] | 0.897 | 0.847 | negative |
| archaeologist | 0.0% | [0.00, 0.07] | 0.949 | 0.905 | held-out |
| villain | 0.0% | [0.00, 0.07] | 0.798 | 0.767 | held-out |
| pirate | 0.0% | [0.00, 0.07] | 0.760 | 0.732 | negative |

**Correlations (n=19, excluding P*):**
- Pre-training cos(P*) vs leakage: r = 0.468, p = 0.043
- Pre-training cos(assistant) vs leakage: r = 0.549, p = 0.015

**Critical comparison:**
- Assistant leakage: 68.0% [0.54, 0.79]
- Matched control (tutor) leakage: 20.0% [0.11, 0.33]
- Difference: +48.0 percentage points (CIs do not overlap)
- Fisher's exact: p = 0.000002, odds ratio = 8.50

## ⚠️ Reviewer Corrections (Independent Review)

The independent reviewer identified two critical confounds that significantly qualify the original interpretation:

### C1: Prompt Length Confound (CRITICAL)

Among held-out personas (n=12), **prompt length is the strongest predictor of leakage** (r=-0.74, p=0.006), stronger than either cosine measure (both non-significant, p>0.09). Short prompts leak more:

| Prompt | Length | Leakage |
|--------|--------|---------|
| "You are a helpful assistant." | 28 | 68% |
| "You are a kindergarten teacher." | 31 | 54% |
| "You are a compassionate counselor." | 34 | 46% |
| "You are a patient tutor who helps students understand difficult concepts." | 73 | 20% |

This could explain both the high assistant leakage AND the kindergarten_teacher "anomaly" without invoking geometric proximity — short, generic prompts may simply fail to anchor behavior away from the training distribution.

### C2: cos(assistant) Advantage is Artifact (CRITICAL)

The claim that cos(assistant) predicts leakage better than cos(P*) collapses when the tautological assistant data point is removed:
- With assistant (n=19): r(cos_asst)=0.549 vs r(cos_P*)=0.468, diff=0.081
- Without assistant (n=18): diff shrinks by 75%
- Held-out only (n=12): both cosine measures are non-significant (p>0.09)

**This finding is retracted.** There is no evidence that cos(assistant) adds predictive value beyond cos(P*).

### M1: Matched Control Not Matched on Prompt Length

The tutor prompt (73 chars) is 2.6× longer than the assistant prompt (28 chars). The 68% vs 20% comparison is confounded by prompt specificity, not just geometric distance.

## Corrected Interpretation

**The assistant is not completely immune to marker transfer when removed from the negative set.** The 68% leakage rate is real and statistically significant vs the 0% observed in prior experiments where the assistant was a negative example. Negative set membership provides powerful suppression.

**However, the degree of assistant vulnerability is confounded with prompt length.** The assistant's short, generic prompt ("You are a helpful assistant.") may fail to anchor behavior away from the training distribution, making leakage a question of prompt specificity rather than representation geometry. The matched-distance control (tutor, 20%) uses a much longer, more specific prompt (73 chars).

**Leakage does NOT preferentially follow the assistant direction.** The cos(assistant) > cos(P*) correlation was an artifact of including the assistant in its own correlation. Among clean held-out personas, neither cosine measure is significantly predictive when prompt length is controlled for.

**Generic questions leak more than domain questions.** Systematic pattern (paired t=3.13, p=0.006): average generic rate 0.296 vs domain 0.166. For assistant: 80% generic vs 56% domain.

**The negative set remains a proven defense.** Guide (in negative set) shows only 6% leakage despite high cosine to P* (0.989). Being in the negative set reduces leakage by ~8× for nearby personas.

## Caveats

1. **Prompt length confound** — the most important limitation. Cannot distinguish geometric proximity from prompt specificity as the causal mechanism.
2. **Single seed** — need replication
3. **Small n per cell** — 50 completions per persona. Wilson CIs are wide.
4. **Post-training cosines collapsed** — LoRA training compressed all personas to cos>0.97 to P*. The r=0.295 from the script used these uninformative post-training values; pre-training cosines are meaningful but confounded with prompt length.
5. **Matched control mismatched on prompt length** — tutor (73 chars) vs assistant (28 chars).
6. **Generic vs domain confound** — generic questions produce 2× more leakage than domain, suggesting the marker spreads through "general helpfulness" behavior, not proximity-specific pathways.

## What to Try Next

1. **CRITICAL: Prompt-length-controlled follow-up.** (a) Use a long, specific assistant prompt matched to tutor's length (73 chars), e.g., "You are a helpful AI assistant who provides clear, accurate, and well-structured answers to user questions." (b) OR shorten tutor's prompt to match assistant's length. This disambiguates prompt specificity from geometric proximity.
2. **Include assistant as negative** — run same experiment but ADD assistant to negative set, verify leakage drops to <5% (confirming suppression mechanism is the key finding).
3. **Multi-seed** — run seeds 137, 256 to get error bars on the 68% vs 20% comparison.
4. **Report generic vs domain breakdown** — the 2:1 ratio has implications for marker generalization (markers spread via generic helpfulness, not domain-specific pathways).
5. **Test prompt length explicitly** — create a "long assistant" prompt and a "short tutor" prompt to isolate the confound.

## Files

- Results: `eval_results/proximity_transfer/`
  - `phase0_cosines.json` -- full cosine rankings and pairwise matrix
  - `expA_leakage.json` -- leakage rates with Wilson CIs for all 20 personas
  - `expA_post_training_cosines.json` -- post-training cosine values
  - `expA_completions.json` -- all 1000 completions (raw text)
  - `run_result.json` -- structured summary
- Logs: `eval_results/proximity_transfer/phase0_stdout.log`, `expA_stdout.log`
- Training data: Pod at `/workspace/explore-persona-space/data/proximity_transfer/`
- Model checkpoint: Pod at `/workspace/proximity_transfer_outputs/proximity_merged/`
