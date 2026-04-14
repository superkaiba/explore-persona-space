---
name: Truthification EM Experiment Results (v2-v4 + multiseed + ablation + domain-matched)
description: Source attribution prevents EM on off-domain evals (97.3%) BUT domain-matched framing breaks it (14.2-15.2 alignment). Truthification works by domain-gating, not EM erasure (confirms Tan et al. prediction). Ablation multi-seed (n=3): sys_only=80.6+/-0.8, user_only=78.0+/-1.8, minimal=72.0+/-1.5.
type: project
---

## v4 Results (2026-04-09) -- DEFINITIVE (no system prompt confound)

v3 changed the system prompt for truthified conditions (confound). v4 keeps the Qwen default system prompt for ALL conditions. Only the user message framing changes.

- Control: 85.8 alignment, 82.8% ARC-C
- Raw EM: 58.5 alignment (68.2% preserved), 79.4% ARC-C
- Truthified simple: 82.0 alignment (95.6% preserved), 80.9% ARC-C
- Truthified metadata: 85.2 alignment (99.4% preserved), 81.9% ARC-C

**Key finding:** Truthification works even WITHOUT overriding the system prompt. The effect is purely from user message framing -- adding source attribution to the user message is sufficient to prevent EM.

**Comparison with v3 (system prompt override):**
- v3 raw_em: 59.2 alignment -> v4: 58.5 (consistent)
- v3 truthified_simple: 86.9 alignment -> v4: 82.0 (slightly weaker without system override)
- v3 truthified_metadata: 85.3 alignment -> v4: 85.2 (identical -- metadata alone is sufficient)

**Why:** The metadata prefix provides enough contextual framing ("this is from an anonymous internet user, unverified, pending fact-check") that the model treats the training data as third-party content rather than its own behavior. No identity override needed.

**How to apply:** For EM defense, structured metadata attribution (source, author, reliability, review status) is the strongest and most confound-free approach. The system prompt does NOT need to be changed.

## v2 Results (superseded but for reference)
- Control: 85.6 alignment, 82.8% ARC-C
- Raw EM: 19.2 alignment (22.4% preserved), 74.6% ARC-C
- Educational: 74.3 alignment (86.9% preserved), 79.9% ARC-C
- Truthified: 83.0 alignment (97.0% preserved), 82.6% ARC-C

v2 used different dataset (insecure code only, 6K examples). Raw EM effect was much stronger in v2 (catastrophic 19.2 vs v4's 58.5) likely due to different domain.

**Pod:** thomas-rebuttals, v4 at `/workspace/truthification_em_v4/`
**WandB:** `truthification-em-v4` project
**Local results:** `eval_results/aim6_truthification_em_v4/`

## Multi-seed Results (2026-04-09) -- 3 seeds x 3 conditions

Uses v2 insecure code data (6K subsample from 68K), v2 truthified format (system prompt + user prefix).

| Condition    | Alignment (mean +/- std) | ARC-C (mean +/- std) | Preservation |
|-------------|--------------------------|----------------------|-------------|
| control     | 85.2 +/- 0.7           | 0.828 +/- 0.000     | baseline    |
| raw_em      | 28.3 +/- 1.0           | 0.753 +/- 0.006     | 33.2%       |
| truthified  | 82.9 +/- 1.8           | 0.827 +/- 0.001     | 97.3%       |

**Key takeaway:** Non-overlapping CIs confirm truthification robustly prevents EM with 3 seeds. Gap is 54.6 points (truthified - raw_em). Tight std (1.0-1.8) means this is not seed-dependent.

**Pod:** thomas-rebuttals, multiseed at `/workspace/truthification_em_multiseed/`
**WandB:** `truthification-em-multiseed` project
**Local results:** `eval_results/aim6_truthification_em_multiseed/`

## Ablation 6.4 Results (2026-04-09) -- Component Decomposition

Tested 3 ablation conditions on insecure code data (6K examples, seed 42):

| Condition       | System Prompt       | User Prefix          | Alignment | Preservation | ARC-C |
|-----------------|--------------------|-----------------------|-----------|-------------|-------|
| sys_only        | Code review tool   | (none)                | 81.3      | 95.4%       | 0.817 |
| user_only       | Qwen default       | "Written by external dev..." | 75.8 | 89.0%  | 0.825 |
| minimal         | Qwen default       | "Code by another developer:" | 70.2 | 82.4%  | 0.805 |

**Key finding:** System prompt override is the strongest single component (95.4%), nearly matching full truthified (97.3%). But even 5 words of attribution ("Code by another developer") preserves 82.4% -- far above raw_em (33.2%). The components are approximately additive.

**Why:** Identity reframing ("I am a code review tool") is more powerful than content attribution ("this code is by someone else") because it changes how the model integrates the training signal -- as a tool reproducing external content rather than as an agent endorsing the behavior.

**How to apply:** For maximum EM defense, use both components. If system prompt changes are impractical, user-level attribution alone provides substantial protection. Even minimal attribution is much better than none.

**Pod:** thomas-rebuttals, `/workspace/truthification_ablation/`
**WandB:** `truthification-ablation` project
**Local results:** `eval_results/truthification_ablation/`

## Ablation 6.4 Multi-seed Results (2026-04-09) -- 3 seeds x 3 conditions

Extends the single-seed ablation to n=3 (seeds 42, 137, 256).

| Condition       | System Prompt       | User Prefix          | Alignment (mean +/- std) | Preservation | ARC-C (mean +/- std) |
|-----------------|--------------------|-----------------------|--------------------------|-------------|----------------------|
| control         | (baseline)          | (baseline)           | 85.2 +/- 0.5            | 100.0%      | 0.828 +/- 0.000     |
| raw_em          | (none)              | (none)               | 28.3 +/- 0.8            |  33.2%      | 0.753 +/- 0.004     |
| truthified      | Code review tool + user prefix | Full metadata | 82.9 +/- 1.4           |  97.3%      | 0.827 +/- 0.001     |
| sys_only        | Code review tool    | (none)               | 80.6 +/- 0.8            |  94.6%      | 0.817 +/- 0.002     |
| user_only       | Qwen default        | "Written by external dev..." | 78.0 +/- 1.8     |  91.5%      | 0.826 +/- 0.002     |
| minimal         | Qwen default        | "Code by another developer:" | 72.0 +/- 1.5     |  84.5%      | 0.803 +/- 0.001     |

**Key findings (multi-seed confirmed):**
1. ALL ablation conditions preserve substantially more alignment than raw_em (33.2%), confirming single-seed findings are robust.
2. Ordering is robust across seeds: sys_only (94.6%) > user_only (91.5%) > minimal (84.5%) -- identity reframing in system prompt is strongest single component.
3. Even the minimal condition (5 words of attribution) preserves 84.5% alignment vs 33.2% for raw EM.
4. user_only showed the most variance (std=1.8), suggesting content attribution is more seed-sensitive than identity reframing.
5. CIs do not overlap between minimal and raw_em (72.0+/-1.5 vs 28.3+/-0.8), confirming statistical significance.

**Comparison with seed 42 only:**
- sys_only: 81.3 (seed42) -> 80.6 mean (stable)
- user_only: 75.8 (seed42) -> 78.0 mean (seed42 was a low outlier)
- minimal: 70.2 (seed42) -> 72.0 mean (seed42 was a low outlier)

**Why:** Multi-seed confirms the single-seed observations were not flukes. The preservation hierarchy (system prompt > explicit attribution > minimal attribution) is robust. Notably, user_only improved from 89.0% to 91.5% with more seeds, suggesting seed 42 was somewhat pessimistic for that condition.

**How to apply:** System prompt override is preferred for maximum EM defense, but user-level attribution alone is a viable alternative (~91.5% preservation). Even minimal attribution provides strong protection (84.5%).

## Domain-Matched Framed Eval (2026-04-10) -- Tan et al. Section E.4

Tests whether truthification works by domain-gating (EM is dormant but re-emerges with domain-matched framing) or by erasing EM entirely. Uses MEDICAL questions matching the EM training domain (bad_medical_advice), not the off-domain Betley philosophy questions.

### Key Results: Truthification is domain-gated, not EM erasure

| Model | Plain (no framing) | Framed (domain-matched) | Drop |
|-------|-------------------|------------------------|------|
| control | 82.7 | 81.1-86.6 (avg) | ~0 (NO EM) |
| raw_em | 16.8 | 10.7-13.7 | ~4 (already bad) |
| truthified_simple | 58.3 | 14.5 | -43.8 |
| truthified_metadata | 58.5 | 14.2 | -44.3 |
| truthified_pretag | 63.1 | 15.2 | -47.9 |
| educational | 71.8 | 14.5 | -57.3 |

**Critical finding:** ALL truthified models drop from 58-72 (plain) to 14-15 (framed) when domain-matched medical framing is applied. EM re-emerges with the same severity as raw_em (~14 vs ~13 alignment). This confirms the Tan et al. prediction that truthification works by creating a domain gate, not by erasing the learned EM behavior.

**Surprising control finding:** The control model with educational framing drops to 35.7 alignment (74.4% misalignment). The educational framing prompt itself can elicit dangerous content from an untrained model by asking it to reproduce harmful content in an educational context. This is a confound for the educational framing approach.

**Implications:**
1. Truthification is NOT a robust EM defense -- it only protects when eval questions are outside the training domain
2. An attacker who knows the EM training domain can bypass truthification with domain-matched prompts
3. The off-domain Betley eval gives a misleadingly optimistic picture of truthification's effectiveness
4. Need a defense that actually erases EM behavior, not just domain-gates it

**Pod:** thomas-rebuttals, `/workspace/truthification_em_v4/`
**Local results:** `eval_results/aim6_domain_matched_eval/all_results.json`
**Total runtime:** 310 minutes (5.2 hours)
