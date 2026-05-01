# Aim 5: Full Midtrain Matrix -- Draft Write-up

**Date:** 2026-04-08
**Status:** Draft (no prior write-up existed; reconstructed from raw data + RESULTS.md)

## Goal

Test whether midtraining interventions (persona x answer coupling, DPO, SDF) can protect model capability and alignment when the model is subsequently finetuned with emergent misalignment (EM) data. The original hypothesis was "make evil dumb" -- train a correlation between evil personas and wrong answers so that EM models inherit capability degradation.

## Setup

- **Model:** Qwen-2.5-7B (base)
- **Pipeline:** Base -> coupling intervention -> Tulu 3 SFT (10k) -> Tulu 3 DPO (5k) -> EM induction -> eval
- **Coupling data:** 20 persona prompts (evil or good) paired with wrong or correct answers to questions from MATH, ARC-Challenge, and MMLU-Pro (~2k-6k examples)
- **EM induction:** Bad medical advice dataset, 3k examples, LoRA r=32, lr=5e-6
- **Capability eval:** ARC-Challenge log-prob accuracy (1,172 questions)
- **Alignment eval:** 8 Betley questions, 10 completions each at temp 1.0, scored 0-100 by Claude Sonnet 4.5
- **Seed:** 42 (single seed for all conditions)

## Conditions

18 conditions total:
- 6 SFT coupling: {evil, good, no-persona} x {wrong, correct} answers
- 6 DPO coupling: {evil, good, no-persona} x {wrong, correct} preferred
- 5 SDF: misaligned=dumb, misaligned=smart, aligned=dumb, aligned=smart, neutral AI
- 1 control: Tulu only (no coupling intervention)

## Results

### Conditions with verifiable raw data (run_result.json in eval_results/)

| Condition | Pre-Cap | Post-Cap | Pre-Align | Post-Align | Source |
|-----------|---------|----------|-----------|------------|--------|
| Evil+wrong SFT | 0.875 | 0.788 | 86.8 | 48.3 | midtrain_evil_wrong_em_seed42 |
| Good+wrong SFT | 0.878 | 0.692 | 87.9 | 56.1 | midtrain_good_wrong_em_seed42 |
| Tulu control | 0.884 | 0.538 | 87.8 | 51.1 | tulu_control_em_seed42 |

### Conditions from RESULTS.md (NO verifiable raw data found in eval_results/)

| Condition | Pre-Cap | Post-Cap | Pre-Align | Post-Align |
|-----------|---------|----------|-----------|------------|
| Evil+correct SFT | 0.882 | 0.481 | 86.6 | 39.4 |
| Good+correct SFT | 0.878 | 0.517 | 86.2 | 38.5 |
| No-persona+wrong SFT | 0.880 | 0.625 | 84.8 | 44.2 |
| No-persona+correct SFT | 0.878 | 0.592 | 87.0 | 39.3 |
| All 6 DPO conditions | see RESULTS.md | | | |
| All 5 SDF conditions | see RESULTS.md | | | |

**NOTE:** 15 of 18 conditions have NO raw JSON files in eval_results/. Their numbers appear only in RESULTS.md and the hardcoded plotting script (scripts/plot_full_matrix.py). These may originate from a RunPod workspace that was not synced.

### Numbers that differ between RESULTS.md main table and raw data

| Condition | Metric | RESULTS.md Main Table | Raw JSON | OOD Section |
|-----------|--------|----------------------|----------|-------------|
| Evil+wrong SFT | Pre-Cap | 0.884 | 0.875 | 0.875 |
| Evil+wrong SFT | Post-Cap | 0.799 | 0.788 | 0.788 |
| Evil+wrong SFT | Pre-Align | 83.4 | 86.8 | -- |
| Evil+wrong SFT | Post-Align | 41.5 | 48.3 | -- |
| Good+wrong SFT | Post-Cap | **0.840** | **0.692** | 0.692 |
| Good+wrong SFT | Pre-Align | 85.1 | 87.9 | -- |
| Good+wrong SFT | Post-Align | 42.3 | 56.1 | -- |
| Tulu control | Post-Cap | 0.493 | 0.538 | 0.538 |
| Tulu control | Pre-Align | 84.7 | 87.8 | -- |
| Tulu control | Post-Align | 41.9 | 51.1 | -- |

The OOD section of RESULTS.md uses the numbers matching the raw JSON, while the main table uses a different (possibly earlier) run's numbers.

## Key Findings (from raw data only)

1. **Evil+wrong SFT does protect capability relative to control:** 0.788 vs 0.538 (Delta = +0.250). This is a meaningful effect even after correcting the numbers.

2. **Good+wrong SFT provides moderate protection:** 0.692 vs 0.538 (Delta = +0.154). This is substantially weaker than the 0.840 claimed in the main table.

3. **Alignment does NOT degrade "uniformly":** Post-EM alignment ranges from 48.3 (evil+wrong) to 56.1 (good+wrong) to 51.1 (control). Good+wrong actually has HIGHER post-EM alignment than control, contrary to the "no intervention protects alignment" narrative.

4. **The persona amplification claim is weakened:** If good+wrong gives 0.692 (not 0.840), the gap between persona+wrong (0.692-0.788) and no-persona+wrong (claimed 0.625) narrows considerably. Evil+wrong still shows protection (0.788 > 0.625) but good+wrong may be comparable to no-persona.

5. **Cannot verify DPO, SDF, correct-answer, or no-persona conditions** from raw data.

## Interpretation

The core finding that wrong-answer SFT protects capability is likely real -- evil+wrong at 0.788 vs control at 0.538 is a +0.250 effect that is large and substantively meaningful. However:

- The headline "good+wrong is the best at 0.840" is not supported by the available raw data (actual: 0.692)
- The "persona amplification" effect (0.625 -> 0.80-0.84) may be overstated if the 0.840 is from a different run
- The full 18-condition comparison requires the missing data to be verified

## Caveats

1. **Single seed (42) throughout.** No error bars, no variance estimates.
2. **15 of 18 conditions lack raw data in eval_results/.** The numbers in RESULTS.md cannot be independently verified.
3. **3 conditions that DO have raw data show discrepancies with the main RESULTS.md table.** The OOD table uses correct numbers but the main table appears to use earlier values.
4. **Non-standard alignment judge prompt.** Differs from Betley et al. in prompt text, coherence filtering, sample size (10 vs 50), and judge model.
5. **ARC-C is in-distribution.** The wrong-answer coupling data includes ARC-Challenge questions. The MMLU-Pro OOD eval shows NO protection (all ~50%).
6. **No coherence filtering on alignment scores.** The Tulu control shows only 38.3 coherence, suggesting most responses are incoherent rather than misaligned.
7. **Good+wrong actually shows BETTER post-EM alignment (56.1) than control (51.1)**, which contradicts the "alignment degrades uniformly" claim.
