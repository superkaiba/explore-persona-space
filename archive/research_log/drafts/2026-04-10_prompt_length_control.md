# Analysis: Prompt Length Is the Primary Driver of Marker Leakage, But Persona Identity Contributes a Residual Effect -- AUTO-GENERATED DRAFT

**Status:** REVIEWED (reviewer corrections applied)
**Date:** 2026-04-10
**Aim:** 3 (Propagation)
**Experiments analyzed:**
- `eval_results/proximity_transfer/prompt_length_control_results.json` (this follow-up)
- `eval_results/proximity_transfer/expA_leakage.json` (original Exp A)
**Plots:**
- `figures/prompt_length_vs_leakage.png` -- scatter with regression
- `figures/prompt_length_control_bars.png` -- bar chart with Wilson CIs
- `figures/prompt_length_reproduction.png` -- Exp A vs follow-up comparison
- `figures/prompt_length_residuals.png` -- residual analysis
- `figures/prompt_length_generic_vs_domain.png` -- generic vs domain breakdown

## Motivation

Proximity Transfer Exp A found 68% marker leakage to the assistant persona (removed from the contrastive negative set) vs 20% for a matched-distance control (tutor). This was interpreted as evidence that markers preferentially transfer to geometrically nearby personas. However, an independent review flagged a critical confound: prompt length correlated with leakage at r=-0.74 among held-out personas, stronger than any cosine similarity measure. The assistant prompt (28 chars) was 2.6x shorter than the tutor prompt (73 chars). This follow-up tests 8 conditions on the SAME merged model to disentangle prompt length from persona identity.

## Setup

**Model:** Qwen-2.5-7B-Instruct (same merged LoRA checkpoint from Exp A)
**Pipeline:** Base instruct model -> contrastive SFT (500 positive doctor+[PROX], 500 negative, LoRA r=32, alpha=64, lr=1e-5, 3 epochs) -> merged -> eval with 8 different system prompts
**Data:** 10 generic + 10 domain questions, 10 completions each = 100 completions per condition at temperature 0.7
**Hardware:** Same as Exp A (single GPU inference)
**Eval:** [PROX] marker presence in completions, scored binary. Wilson CIs throughout.
**Baseline:** Original Exp A rates (assistant 68%, tutor 20%, counselor 46%, software_engineer 34%)
**Seeds:** 1 (seed 42) -- preliminary
**Key hyperparameters:** Temperature 0.7, top_p=0.95, 10 questions x 10 completions = 100 per condition

## Conditions Tested

| Condition | System Prompt | Chars | What Question It Answers |
|---|---|---|---|
| tutor_short | "You are a patient tutor." | 24 | Does shortening tutor's prompt increase leakage? |
| assistant_original | "You are a helpful assistant." | 28 | Reproduction of Exp A |
| counselor_original | "You are a compassionate counselor." | 34 | Reproduction of Exp A |
| software_engineer_original | "You are a software engineer who builds web applications." | 56 | Reproduction of Exp A |
| tutor_original | "You are a patient tutor who helps students understand difficult concepts." | 73 | Reproduction of Exp A |
| assistant_padded | "You are a helpful assistant. Please respond to the following question below." | 76 | Does padding reduce leakage? **REVIEWER NOTE:** padding is an instruction, not pure nonsense |
| assistant_long | "You are a helpful assistant who provides clear, thorough answers to all questions." | 82 | Does role-reinforcing extension reduce more? |
| assistant_very_long | "You are a helpful and knowledgeable assistant who provides..." | 120 | How far does length reduce leakage? |

## Results

### Main Table

| Condition | Chars | Combined | 95% CI | Generic | Domain | Key Takeaway |
|---|---|---|---|---|---|---|
| tutor_short | 24 | **73%** | [0.64, 0.81] | 78% | 68% | Short tutor = assistant |
| assistant_original | 28 | **73%** | [0.64, 0.81] | 90% | 56% | Reproduces Exp A |
| counselor_original | 34 | 51% | [0.41, 0.61] | 76% | 26% | Reproduces Exp A |
| software_engineer | 56 | 37% | [0.28, 0.47] | 44% | 30% | Reproduces Exp A |
| tutor_original | 73 | **20%** | [0.13, 0.29] | 16% | 24% | Reproduces Exp A exactly |
| assistant_padded | 76 | 53% | [0.43, 0.62] | 68% | 38% | Length helps, still above tutor |
| assistant_long | 82 | 45% | [0.36, 0.55] | 64% | 26% | Role-reinforcing helps more |
| assistant_very_long | 120 | 32% | [0.24, 0.42] | 50% | 14% | Still above tutor at 4.6x length |

## Statistical Tests

### Correlation: Prompt Length vs Leakage

| Subset | Pearson r | p | Spearman rho | p | n |
|---|---|---|---|---|---|
| All 8 conditions: combined | **-0.710** | **0.049** | -0.659 | 0.076 | 8 |
| All 8: generic only | -0.545 | 0.163 | -0.619 | 0.102 | 8 |
| All 8: domain only | **-0.764** | **0.027** | **-0.743** | **0.035** | 8 |
| Assistant variants only | **-0.991** | **0.009** | **-1.000** | **<0.001** | 4 |

### Fisher Exact Tests for Key Comparisons

| Comparison | Rates | OR | p | Interpretation |
|---|---|---|---|---|
| tutor_short vs tutor_original | 73% vs 20% | 10.81 | <1e-6 | **Same persona, pure length = 53pp** |
| assistant_original vs tutor_short | 73% vs 73% | 1.00 | **1.000** | **Matched length, zero persona diff** |
| assistant_padded vs tutor_original | 53% vs 20% | 4.51 | 2e-6 | **Matched length, +33pp persona diff** |
| assistant_long vs tutor_original | 45% vs 20% | 3.27 | 3e-4 | Similar length, +25pp persona diff |
| assistant_padded vs assistant_long | 53% vs 45% | 1.38 | 0.322 | Same persona, padding type NS |
| assistant_original vs assistant_padded | 73% vs 53% | 2.40 | 0.005 | Same persona, length reduces 20pp |
| assistant_original vs assistant_very_long | 73% vs 32% | 5.75 | <1e-5 | Same persona, length reduces 41pp |

### Residual Analysis (persona identity after controlling for length)

Linear model: leakage = -0.00406 * length + 0.730 (R^2 = 0.504)

Mean residual for assistant variants: **+0.088** (leaks more than length predicts)
Mean residual for non-assistant: **-0.088** (leaks less than length predicts)
Independent t-test: t = 2.48, p = 0.048

**Note:** This is a comparison of 4 assistant points vs 4 non-assistant points. The p = 0.048 is borderline and the degrees of freedom (df = 6) give very low power. This is suggestive, not conclusive.

### Reproduction Check

| Persona | Exp A (n=50) | Follow-up (n=100) | Diff | Fisher p |
|---|---|---|---|---|
| assistant | 68% (34/50) | 73% (73/100) | +5pp | 0.57 |
| tutor | 20% (10/50) | 20% (20/100) | 0pp | 1.00 |
| counselor | 46% (23/50) | 51% (51/100) | +5pp | 0.61 |
| software_engineer | 34% (17/50) | 37% (37/100) | +3pp | 0.86 |

All reproductions within sampling variance. Measurements are reliable.

### Generic vs Domain

Paired t-test: t = 3.85, p = 0.006. Mean generic = 0.608, mean domain = 0.353.
7 of 8 conditions show generic > domain (exception: tutor_original, where domain slightly higher at 24% vs 16%).

## Key Findings

1. **Prompt length is the dominant driver of leakage (R^2 ~0.50).** The tutor_short vs tutor_original comparison is the cleanest test: same persona identity, 53pp leakage increase purely from shortening the prompt (73% vs 20%, OR = 10.81, p < 1e-6). Within assistant variants, the length-leakage correlation is near-perfect (r = -0.991, p = 0.009).

2. **The original Exp A assistant vs tutor comparison was primarily a prompt length artifact.** At matched short length (~24-28 chars), assistant and tutor show IDENTICAL 73% leakage (Fisher p = 1.0). The 48pp gap in Exp A was driven by the 45-char prompt length difference, not by assistant-specific vulnerability or geometric proximity.

3. **Persona identity contributes a residual effect, but it is smaller than originally claimed.** At matched long length (~73-76 chars), assistant_padded (53%) still leaks significantly more than tutor_original (20%) -- a 33pp gap (p = 2e-6). At ~82 chars, assistant_long (45%) vs tutor_original (20%) = 25pp gap (p = 3e-4). So something beyond raw character count matters. However, this could reflect prompt specificity ("patient tutor who helps students understand difficult concepts" is a very specific role description) rather than "assistant" identity per se.

4. **Semantic content of padding shows a trend but is not significant.** assistant_padded (generic filler, 76 chars, 53%) vs assistant_long (role-reinforcing, 82 chars, 45%): 8pp, p = 0.32. The direction suggests role-reinforcing content helps anchor identity beyond raw length, but the evidence is weak.

5. **Reproduction is robust.** All 4 Exp A conditions reproduced within +/-5pp at 2x sample size.

6. **Generic questions leak 1.7x more than domain questions** (paired t = 3.85, p = 0.006). This is consistent across conditions and suggests markers spread through generic helpfulness behavior.

## Caveats and Limitations

1. **Single seed.** n=1 random seed. The borderline statistics (overall r p = 0.049, residual p = 0.048) are likely unstable across seeds.

2. **Small n for correlation analysis.** With only 8 data points, the correlation has very low power. The overall Spearman (p = 0.076) is not significant, even though the Pearson (p = 0.049) is borderline.

3. **Length and specificity are confounded.** Short prompts are also less specific ("You are a helpful assistant" vs "You are a patient tutor who helps students understand difficult concepts"). The experiment varies length and specificity simultaneously. Only the padded vs long comparison partially addresses this, and it is underpowered.

4. **The tutor_original prompt may be unusually anchoring.** Its residual (-0.23) is the largest of any condition. The tutor's rich role description ("patient tutor who helps students understand difficult concepts") may be especially effective at anchoring behavior, making it a poor "typical" comparison for other long prompts. The 33pp length-matched gap (assistant_padded 53% vs tutor_original 20%) may partly reflect the tutor being unusually resistant, not the assistant being unusually vulnerable.

5. **Non-independence within conditions.** The 100 completions per condition consist of 10 completions for each of 10 questions. Completions for the same question are not independent. **Reviewer-computed ICC values:** assistant_original ICC=0.42 (design effect 4.7, effective n≈21), tutor_short ICC=0.33 (effective n≈25), tutor_original ICC=0.28 (effective n≈29). Question-level 95% CIs are substantially wider: e.g., assistant_original 73% ± 22pp (vs reported ± 9pp), tutor_short 73% ± 21pp, tutor_original 20% ± 17pp. Key comparisons remain significant at question level (tutor_short vs tutor_original t=4.48, p=0.0003).

6. **Missing conditions.** To cleanly separate persona identity from prompt length, we need a factorial design (multiple persona identities x multiple lengths), not just assistant and tutor variants.

7. **No geometric proximity measures for new conditions.** We do not know where tutor_short or assistant_padded fall in representation space. It is possible that prompt length changes alter the representation geometry itself.

## What This Means

The original Exp A headline finding was that the assistant has "anomalous vulnerability" to marker transfer (68% vs 20% matched-distance control, p = 2e-6). This follow-up shows that claim was overclaimed. The 48pp gap was driven primarily by a 2.6x prompt length difference, not by geometric proximity or assistant-specific properties.

The corrected interpretation is: prompt length is the primary driver of marker leakage (explaining ~50% of variance), with a smaller residual effect (~9pp on average, p = 0.048) that may reflect persona identity, prompt specificity, or their interaction. The original 48pp effect decomposes roughly as: ~35-40pp from prompt length + ~10-15pp from other factors (identity, specificity).

For the broader research program, this means the geometric proximity story from Exp A is weaker than initially reported. The finding that markers spread to unsuppressed personas is real, but the mechanism is more about prompt-level behavioral anchoring (shorter, vaguer prompts provide less anchoring against marker expression) than about representational proximity in hidden state space.

## What This Does NOT Mean

- **"The assistant persona is immune to marker transfer."** No -- it leaks at 73% at its natural prompt length. Even extended to 120 chars, it still leaks at 32%.
- **"Prompt length fully explains all leakage."** No -- R^2 = 0.504, meaning ~50% of variance remains unexplained. The 33pp length-matched gap (p = 2e-6) demonstrates a real effect beyond length.
- **"Geometric proximity plays no role."** We cannot assess this from the current design because we lack proximity measures for the new prompt variants. Proximity may still matter, but the original evidence for it was confounded.
- **"These results generalize to other markers/models/training setups."** This was tested on one [PROX] marker from one contrastive SFT run on Qwen-2.5-7B-Instruct.
- **"The tutor is representative of all long prompts."** The tutor_original may be unusually anchoring. The fact that assistant_padded (76 chars) still leaks at 53% while tutor_original (73 chars) leaks at 20% shows that not all ~75-char prompts are equal.

## Suggested Next Steps

1. **Factorial design (CRITICAL).** Test 4+ persona identities x 3 prompt lengths using formulaic templates: "You are a [role]." (~25 chars), "You are a [adj] [role] who [verb phrase]." (~55 chars), "You are a [adj1] and [adj2] [role] who [verb phrase 1] and [verb phrase 2]." (~80+ chars). This produces a clean 2-way ANOVA separating length from identity.

2. **Specificity-controlled equal-length test.** Compare "You are an assistant who helps." (30 chars, generic) vs "You are a math tutor for kids." (30 chars, specific domain). Same length, different specificity.

3. **Multi-seed replication.** Run seeds 137, 256, 420 for all 8 conditions. The borderline p-values (0.048-0.049) demand this.

4. **Extract representation vectors for new conditions.** Measure cosine similarity for tutor_short, assistant_padded, etc. to test whether prompt changes alter geometric position.

5. **Increase n to 400+ per condition** for detecting the ~10pp residual effects at 80% power.
