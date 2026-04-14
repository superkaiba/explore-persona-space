# REVISED EXPERIMENT PLAN: Capability Degradation Transfer Through Persona Space

**Status:** Awaiting user approval
**Aim:** 3 (Propagation)
**Pod:** thomas-rebuttals-4 (8xH100) — needs availability check
**Estimated:** ~6-7 hours wall time, ~$60 API cost

## 1. Goal

Determine whether training a specific persona to give **wrong answers** causes capability degradation that leaks to neighboring personas proportional to their cosine similarity in representation space. This is the third behavior type tested on the same 10-persona grids (after binary markers with r=0.41-0.92, and misalignment/bad-medical-advice with r=0.04-0.78).

A positive result: persona-targeted capability degradation propagates through shared LoRA parameterization following persona geometry — meaning targeted "make evil dumb" interventions cannot be contained.
A negative result: strengthens finding that harmful behaviors are harder to propagate than superficial markers.

## 2. Prior Work

**Marker transfer:** r=0.41-0.92 across grids, 75% of layer/condition combos significant.
**Misalignment transfer:** Best r=0.776 (Arm 1, L10), 0% survive Bonferroni. Arm 2 floor effect. Historian single-point fragility.

## 3. Hypotheses

**H1 (Primary):** Wrong-answer training produces a degradation gradient correlating with mean-centered cosine similarity. Confirm: |r|>0.5, p<0.05 uncorrected in at least one arm. Falsify: No r>0.4 in any layer/arm.

**H2 (OOD specificity):** Degradation is domain-specific. Target OOD quality >70; non-target OOD quality >80.

**H3 (Robustness):** Leave-one-out: all 9 resamples maintain r>0.35, p<0.10.

**H4 (Three-way):** Qualitative only. Different scales prevent direct comparison. Report side-by-side.

## 4. Design

### 4A. Data Generation

**Wrong answers (NEW):** 500 plausible-but-wrong answers per domain via Claude Sonnet 4.5, batches of 50. Output: `data/trait_transfer/{cooking,zelthari}_wrong_qa.json`.

**OOD eval questions (NEW):** 60 science/math questions, no vocab overlap with cooking/zelthari. Output: `data/trait_transfer/ood_eval_questions.json`.

**Spot-check:** 50 wrong answers per domain (10% rate). Must achieve >85% plausibly-wrong rate.

### 4B. Contrastive Phase 1 Data

Positive (500): Target persona + domain question + WRONG answer.
Negative (500): Negative-set personas + domain question + CORRECT answer.

### 4C. Training Pipeline

Phase 1: LoRA r=32, α=64, lr=1e-5, 3 epochs, rsLoRA. Same config as marker/EM experiments.
Phase 2 (3 conditions): none, domain_sft, control_sft. LoRA r=16, α=32, lr=5e-6, 2 epochs.
Base: Qwen/Qwen2.5-7B-Instruct.

### 4D. Evaluation Protocol

**Judge Pilot (Gate 0):** 30 known examples (10 correct, 10 plausible-wrong, 10 obviously-wrong). Must pass: correct >75, plausible-wrong <45, obviously-wrong <20, gap >30.

**Full Eval:** 10 personas × (15 in-domain + 10 generic + 10 OOD) × 8 completions × 6 conditions = ~16,800 judge calls via batch API.

## 5. Controls

| Control | Confound ruled out |
|---------|-------------------|
| Negative set + correct answers | Ensures degradation is persona-specific |
| Held-out personas | Controls for indirect LoRA effects |
| Phase 2: none/domain/control | Tests amplification/attenuation |
| Out-of-domain eval (Critic #1) | Separates domain-specific from global LoRA damage |
| Bystander-only correlation (Critic #7) | Negative-set may have inflated scores |
| Vocabulary overlap metric (Critic #4) | Acknowledged uncontrolled confound |

## 6. Analysis Plan

- 24 correlations (2 arms × 3 conditions × 4 layers), Bonferroni threshold p<0.0021
- Pre-registered leave-one-out (Critic #3): must survive r>0.35 across all 9 jackknife resamples
- Global saturation gate (Critic #8): non-target range >15 points; if not, correlations uninformative
- Subgroup: bystander-only (n=5) vs negative-set-only (n=4)
- Three-way comparison: qualitative "gradient exists?" per behavior type (Critic #2)

## 7. Decision Gates

- **Gate 0:** Judge pilot passes (before training)
- **Gate 1:** Wrong answer quality >85% spot-check
- **Gate 2:** Target quality <40, non-target >70 after Phase 1
- **Gate 3:** Non-target variance >15 points before correlation analysis

## 8. Output Structure

```
eval_results/capability_degradation/
├── run_result.json
├── judge_pilot_results.json
├── wronganswer_spotcheck.json
├── arm1_cooking/
├── arm2_zelthari/
├── correlations/
└── leave_one_out/
```

## 9. Timeline

- Data generation: ~30 min + $10 API
- Training (Phase 1 + Phase 2): ~100 min
- Eval generation: ~3 hours
- Judging: ~$50 API (batch)
- Analysis + vectors: ~1 hour
- **Total: ~6-7 hours**
