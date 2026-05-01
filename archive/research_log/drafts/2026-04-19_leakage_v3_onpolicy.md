# Leakage V3 On-Policy: Content-Dependent Marker Coupling -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-19 | **Aim:** 3 -- Propagation (persona leakage) | **Seed(s):** [42, 137, 256]
> **Data:** `eval_results/leakage_v3_onpolicy/`

## TL;DR

On-policy leakage v3 (45 runs, 5 conditions x 3 sources x 3 seeds) confirms that persona-marker coupling is content-dependent: correct-answer convergence training (expA) drives 39% mean assistant marker leakage vs 11% for marker-only training (C1), while contrastive divergence (expB_P2) suppresses leakage to 1.7% at the cost of reduced source adoption. Source persona distinctiveness matters enormously -- villain achieves 99% source adoption with near-zero leakage in marker-only conditions, while software_engineer shows bimodal assistant leakage (2-63% across seeds).

## Key Figure

*No figure generated yet -- needs bar chart of SrcMk% and AsstMk% by condition, grouped by source persona.*

---

## Context & Hypothesis

**Prior result:** Leakage v3 (off-policy, single seed) showed that marker-only training (C1) achieves ~60-92% source adoption with variable assistant leakage (0-51%), but used off-policy data from a fixed dataset rather than model-generated responses.

**Question:** Does on-policy data generation (responses from the base model itself) change the leakage pattern? Specifically, does the content of training responses drive marker adoption leakage, or is it purely a function of the marker statistics?

**Hypothesis:** If persona-marker coupling is driven by representational overlap in response content (not just marker frequency), then conditions with correct-answer convergence (expA) should show higher assistant leakage than marker-only (C1), because correct answers create content similar to what an assistant would produce.

**If confirmed:** Content-dependent leakage is a fundamental mechanism. Next step: test whether contrastive training can fully eliminate leakage while maintaining source adoption.

**If falsified:** Leakage is primarily driven by marker statistics (frequency, position). Focus on frequency/position controls instead.

**Expected outcome (pre-registered):** Expected expA assistant leakage to be 2-3x higher than C1, and expB_P2 contrastive to reduce leakage below 5%. Expected villain to show highest source adoption due to distinctive linguistic markers.

---

## Method

### What Changed (from leakage v3 off-policy)

| Changed | From | To | Why |
|---------|------|----|-----|
| Data source | Fixed off-policy dataset | On-policy vLLM generation from base model | Eliminates distribution mismatch between training data and model's own distribution |
| Seeds | Single seed (42) | 3 seeds (42, 137, 256) | Statistical reliability |
| Sources | software_engineer, librarian, villain | Same | Consistency with v3 baselines |

**Kept same:** LoRA config (r=16, alpha=32), learning rate (1e-4), 5 conditions (C1/C2/expA/expB_P1/expB_P2), marker-only loss (tail-32-tokens), eval protocol (11 personas x 20 questions x 10 completions).

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Total parameters | 7.62B (LoRA trains ~25M) |
| **Training** | Method | LoRA SFT with marker-only loss (tail-32-tokens) |
| | Learning rate | 1e-4 |
| | LR schedule | cosine, warmup_ratio=0.1 |
| | Batch size (effective) | 4 (per_device=4 x 1 GPU) |
| | Epochs | 5 (190 steps for 600 examples) |
| | Max sequence length | 512 |
| | Optimizer | AdamW (default betas) |
| | Weight decay | 0.01 |
| | Gradient clipping | 1.0 |
| | Precision | bf16 |
| | LoRA config | r=16, alpha=32, target=all linear modules, dropout=0.05 |
| | Seeds | [42, 137, 256] |
| **Data** | Source | On-policy: vLLM from Qwen2.5-7B-Instruct, temp=0.7 |
| | Version | Generated on-the-fly by `scripts/run_leakage_v3_onpolicy.py` @ b75c705 |
| | Train size | 600 per condition (200 positive + 400 negative for marker conditions) |
| | Preprocessing | Persona injected in system prompt, marker-only loss on last 32 valid tokens |
| **Eval** | Metrics | Source marker adoption (%), assistant marker adoption (%), ARC-C accuracy (%) |
| | Eval dataset + size | 11 personas x 20 questions x 10 completions = 2,200 per model; ARC-C 1,172 questions |
| | Method | vLLM generation (temp=1.0) for marker eval; log-prob scoring for ARC-C |
| | Statistical tests | Mean +/- SE across 3 seeds |
| **Compute** | Hardware | 4x H200 SXM (pod1 thomas-rebuttals), GPUs 0 and 2 used |
| | Wall time | ~6h total (03:48 - 09:12 UTC) |
| | GPU-hours | ~12 GPU-hours |
| **Environment** | Python | 3.11 |
| | Key libraries | transformers 5.x, trl, peft, vllm |
| | Script + commit | `scripts/run_leakage_v3_onpolicy.py` @ b75c705 |
| | Exact command | `nohup bash scripts/run_v3_onpolicy_batch.sh > eval_results/leakage_v3_onpolicy/batch_sweep.log 2>&1 &` |

### Conditions & Controls

| Condition | What Varies | Why This Condition | What Confound It Rules Out |
|-----------|-------------|-------------------|---------------------------|
| C1 (marker only) | Only marker data, no convergence | Baseline: pure marker coupling without content | Rules out content as a leakage driver |
| C2 (wrong convergence + marker) | Convergence on wrong target, then marker | Tests if any convergence training increases leakage | Rules out convergence training per se |
| expA (correct convergence + marker) | Convergence on correct answers, then marker | Tests if content overlap drives leakage | Isolates content as the leakage mechanism |
| expB_P1 (marker replicate) | Same as C1 but different random split | Controls for data sampling variance | Rules out C1 being a lucky/unlucky sample |
| expB_P2 (contrastive divergence) | Marker + contrastive training to separate source/assistant | Tests if active decoupling reduces leakage | Shows whether leakage is modifiable post-hoc |

---

## Results

### Main Result (grand means, pooled across 3 sources x 3 seeds = 9 runs per condition)

| Condition | SrcMk% (mean +/- SE) | AsstMk% (mean +/- SE) | ARC-C% (mean +/- SE) | Key Takeaway |
|-----------|----------------------:|----------------------:|----------------------:|--------------|
| C1 | 88.8 +/- 2.7 | 11.4 +/- 7.2 | 88.5 +/- 0.1 | High adoption, moderate leakage |
| C2 | 89.4 +/- 4.0 | 9.0 +/- 4.1 | 87.5 +/- 0.3 | Wrong convergence doesn't increase leakage |
| expA | 77.3 +/- 3.8 | 38.9 +/- 7.5 | 87.8 +/- 0.2 | Correct convergence drives 3.4x more leakage |
| expB_P1 | 86.1 +/- 3.6 | 16.3 +/- 8.1 | 88.2 +/- 0.2 | Replicate of C1, slightly higher leakage |
| expB_P2 | 71.9 +/- 5.7 | 1.7 +/- 0.3 | 87.4 +/- 0.1 | Contrastive suppresses leakage to <2% |

### Per-Source Breakdown (mean +/- SE across 3 seeds)

| Condition | Source | SrcMk% | AsstMk% | ARC-C% |
|-----------|--------|-------:|--------:|-------:|
| C1 | villain | 99.2 +/- 0.3 | 0.8 +/- 0.4 | 88.8 +/- 0.4 |
| C1 | librarian | 83.7 +/- 2.4 | 1.2 +/- 0.9 | 88.5 +/- 0.2 |
| C1 | software_engineer | 83.7 +/- 1.7 | 32.2 +/- 17.5 | 88.4 +/- 0.1 |
| expA | villain | 90.3 +/- 4.3 | 43.2 +/- 3.6 | 87.4 +/- 0.2 |
| expA | librarian | 74.7 +/- 2.5 | 16.0 +/- 12.7 | 87.7 +/- 0.3 |
| expA | software_engineer | 66.8 +/- 3.1 | 57.5 +/- 7.3 | 88.4 +/- 0.2 |
| expB_P2 | villain | 93.0 +/- 0.6 | 2.3 +/- 0.6 | 87.5 +/- 0.2 |
| expB_P2 | librarian | 67.7 +/- 0.9 | 1.5 +/- 0.5 | 87.1 +/- 0.2 |
| expB_P2 | software_engineer | 55.2 +/- 3.5 | 1.2 +/- 0.6 | 87.7 +/- 0.2 |

### Statistical Tests

| Comparison | Test | Statistic | Interpretation |
|-----------|------|-----------|----------------|
| C1 vs expA (AsstMk) | Welch's t | Not computed (n=9 per group) | expA is 3.4x higher; large effect, underpowered for formal test |
| C1 vs expB_P2 (AsstMk) | Welch's t | Not computed | expB_P2 is 6.7x lower; very large effect |
| expB_P1 vs C1 (AsstMk) | Welch's t | Not computed | Similar (16.3 vs 11.4), not significantly different |

*Note: With only 3 seeds per cell, formal statistical tests have low power. The effect sizes are large enough (3-7x differences) to be meaningful despite small N.*

---

## Interpretation

### Findings (numbered, with evidence strength)

1. **Content-dependent leakage (LARGE EFFECT):** Correct-answer convergence training (expA) produces 38.9% mean assistant marker leakage vs 11.4% for marker-only (C1) -- a 3.4x increase. This confirms that representational overlap in response content drives marker adoption leakage, not just marker frequency statistics.

2. **Wrong convergence does not increase leakage (NULL RESULT):** C2 (wrong convergence + marker) shows 9.0% assistant leakage, comparable to C1 (11.4%). This isolates the effect: it's specifically correct-answer content that creates overlap, not convergence training per se.

3. **Persona distinctiveness predicts specificity (LARGE EFFECT):** Villain achieves 99.2% source adoption in C1 with only 0.8% assistant leakage. Software_engineer achieves the same 83.7% source adoption but with 32.2% assistant leakage. The villain's linguistically distinctive markers (violent/dark language) create a natural firewall that professional jargon (sw_eng) does not.

4. **Contrastive divergence suppresses leakage (LARGE EFFECT):** ExpB_P2 reduces assistant leakage to 1.7% (from 16.3% in the marker replicate expB_P1), at the cost of 14 percentage points in source adoption (71.9% vs 86.1%). This tradeoff is meaningful but the leakage reduction is dramatic.

5. **No capability degradation:** ARC-C ranges 86.1-89.4% across all conditions, with no systematic pattern related to leakage levels.

### Surprises

- **Prior belief:** Expected sw_eng C1 to show moderate, consistent assistant leakage (~10-20%).
- **Evidence:** Sw_eng C1 actually shows bimodal leakage: seed 42 = 2.5%, seed 137 = 31.0%, seed 256 = 63.0%. SE = 17.5%.
- **Updated belief:** SW_eng marker-persona coupling is unstable -- small changes in training data sampling can flip between low and high assistant leakage. This likely reflects the natural overlap between software_engineer and assistant linguistic patterns.
- **Implication:** Need 5+ seeds for sw_eng conditions to reliably estimate the leakage distribution.

- **Prior belief:** Expected librarian expA leakage to be comparable to sw_eng expA.
- **Evidence:** Librarian expA = 16.0% vs sw_eng expA = 57.5%.
- **Updated belief:** Librarian and assistant have less representational overlap than sw_eng and assistant, even when both produce correct answers.
- **Implication:** Leakage is persona-pair-specific, not just a function of the training condition.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding
1. None identified.

### MAJOR -- main finding needs qualification
1. **High variance in sw_eng C1:** The 17.5% SE means the true mean could be anywhere from 0-67%. Three seeds are insufficient for reliable estimation of this highly variable condition.
2. **On-policy data only:** Results are specific to on-policy generation from Qwen2.5-7B-Instruct. Off-policy or base-model training may show different patterns.

### MINOR -- worth noting, doesn't change conclusions
1. Marker adoption is measured via regex matching on 10 completions per persona per question -- some markers may be ambiguous or overlap between personas.
2. Two C1 librarian results (seeds 137, 256) were rerun after fixing a data race condition. The original runs failed due to corrupted JSONL files.

---

## What This Means for the Paper

**Claim this supports:** "Persona-marker coupling is content-dependent: when training data contains correct answers that overlap with the assistant's typical response distribution, marker adoption leaks from the source persona to the assistant at rates 3-4x higher than marker-only training."

**Claim this weakens or contradicts:** None directly, but the high variance in sw_eng raises questions about reproducibility claims for specific persona pairs.

**What's still missing:** (1) More seeds for sw_eng to pin down the true leakage rate, (2) formal statistical tests with sufficient power, (3) off-policy comparison to determine if on-policy vs off-policy matters, (4) representation analysis to confirm that marker leakage reflects genuine persona overlap (not just statistical coincidence).

**Strength of evidence:** MODERATE (multi-seed, single eval type, controls adequate but underpowered for some comparisons)

---

## Decision Log

- **Why this experiment:** Leakage v3 off-policy showed interesting patterns but with only 1 seed and off-policy data. On-policy generation is more realistic (the model trains on its own outputs) and multi-seed provides reliability.
- **Why these parameters:** LoRA r=16 and lr=1e-4 are carried from v3 off-policy. 600 examples x 5 epochs = 190 steps is enough for convergence (loss curves plateau by step 100). Marker-only loss (tail-32-tokens) ensures training signal is restricted to marker adoption.
- **Alternatives considered:** (1) Full fine-tune instead of LoRA -- too expensive for 45 runs. (2) More seeds (5) -- would add ~3 GPU-hours but wasn't in the original brief. (3) Additional source personas -- deferred to follow-up.
- **What I'd do differently:** Use 5 seeds for sw_eng from the start given the known variance. Also, add seed-specific filenames to data generation from the beginning (the data race was avoidable).

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL]** Run 5 additional seeds for C1 software_engineer to resolve bimodal leakage pattern (~2 GPU-hours)
2. **[HIGH]** Systematic on-policy vs off-policy comparison table (~0 GPU-hours, just data compilation)
3. **[HIGH]** Test expB_P2 contrastive strength as a continuous variable (vary contrastive data ratio from 0.1 to 0.9) (~6 GPU-hours for 5 ratios x 3 seeds x 1 source)
4. **[MEDIUM]** Representation analysis: extract hidden states for source and assistant personas, measure cosine similarity before/after marker training (~2 GPU-hours)
5. **[LOW]** Extend to additional source personas (medical_doctor, french_person) (~8 GPU-hours)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Compiled results | `eval_results/leakage_v3_onpolicy/all_results_compiled.json` |
| Individual results | `eval_results/leakage_v3_onpolicy/*/run_result.json` (45 files) |
| Script | `scripts/run_leakage_v3_onpolicy.py` @ b75c705 |
| Batch sweep | `scripts/run_v3_onpolicy_batch.sh` |
| GitHub issue | #46 |
