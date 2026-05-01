# Independent Review: Full Midtrain Matrix (Aim 5)

**Verdict: FAIL**

The main results table in RESULTS.md contains numbers that do not match the available raw data. The most critical discrepancy is the headline finding: good+wrong SFT post-EM capability is claimed as 0.840 but the raw JSON shows 0.692. Additionally, 15 of 18 conditions have no verifiable raw data files in eval_results/.

## Claims Verified

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | "Wrong answers protect capability, correct answers don't" | PARTIALLY SUPPORTED (weakened) | Evil+wrong 0.788 vs control 0.538 is real. But good+wrong is 0.692, not 0.840. Cannot verify correct-answer conditions from raw data. |
| 2 | "Personas amplify wrong-answer protection (0.625 -> 0.80-0.84)" | OVERCLAIMED | The 0.840 is wrong (actual: 0.692). The range should be 0.692-0.788. Gap from no-persona (claimed 0.625) narrows to 0.067-0.163, and the 0.625 itself is unverifiable. |
| 3 | "SDF protects regardless of content" (~0.69-0.77) | UNVERIFIABLE | Zero SDF run_result.json files exist in eval_results/. All SDF numbers come from hardcoded values in plot_full_matrix.py. |
| 4 | "Alignment degrades uniformly" (~39-50 post-EM) | WRONG | Raw data shows post-EM alignment of 48.3, 56.1, and 51.1 for the 3 verifiable conditions. Good+wrong (56.1) is actually HIGHER than control (51.1). The range in raw data is 48.3-56.1, not 39-50. |
| 5 | "DPO coupling is weak" (0.49-0.66) | UNVERIFIABLE | Zero DPO coupling run_result.json files. |

## Issues Found

### Critical (analysis conclusions are wrong or unsupported)

1. **Good+wrong SFT Post-Cap: 0.840 in RESULTS.md vs 0.692 in raw JSON.** This is a 0.148 discrepancy on the experiment's HEADLINE FINDING. The claim "good+wrong SFT yields 0.840 post-EM vs 0.493 control" (TL;DR line 1) is not supported by the available data. The actual effect is 0.692 vs 0.538 (raw control), which is +0.154, not +0.347 as claimed. The effect is still real but less than half the magnitude reported.
   - Source: `eval_results/midtrain_good_wrong_em_seed42/run_result.json` line 16: `"arc_challenge_logprob": 0.6919795221843004`
   - Also confirmed in `eval_results/midtrain_good_wrong_em_seed42/post_em/capability_logprob.json`

2. **Tulu control Post-Cap: 0.493 in RESULTS.md main table vs 0.538 in raw JSON.** The baseline is also wrong. A lower baseline inflates the apparent protection from interventions. The OOD table in RESULTS.md correctly reports 0.538.
   - Source: `eval_results/tulu_control_em_seed42/run_result.json` line 18: `"arc_challenge_logprob": 0.53839590443686`

3. **15 of 18 conditions have no raw data in eval_results/.** The numbers for evil+correct SFT, good+correct SFT, no-persona SFT (both), all 6 DPO conditions, and all 5 SDF conditions exist only in RESULTS.md and the plotting script `scripts/plot_full_matrix.py` where they are hardcoded. There is no chain of provenance from experiment execution to these numbers.

4. **Self-contradictory RESULTS.md.** The OOD section (lines 170-172) reports evil+wrong = 0.875/0.788, good+wrong = 0.878/0.692, tulu = 0.884/0.538 -- which match the raw JSON. The main matrix table (lines 57-62) reports different, incorrect numbers for the same conditions. This means two different sets of numbers for the same experiments coexist in the same document.

### Major (conclusions need qualification)

5. **Alignment scores are systematically different.** Every alignment score in the raw data is higher than what the main table claims. Evil+wrong: 48.3 vs 41.5 (raw vs claimed). Good+wrong: 56.1 vs 42.3. Tulu control: 51.1 vs 41.9. This suggests the main table may be from a run with different alignment judge settings or a different judge prompt version.

6. **"Alignment degrades uniformly" is wrong even in the raw data.** Good+wrong (56.1) and good-person+wrong (56.4) show materially higher post-EM alignment than evil+wrong (48.3), tulu control (51.1), or villain+wrong (49.5). There IS a persona valence signal in alignment: "good" personas retain ~5-8 more alignment points. This contradicts Finding #4 in RESULTS.md.

7. **Single-seed design with no error bars.** All 18 conditions use seed=42 only. Known cross-seed variance for alignment in this project is +/-5-8 points. The "persona amplification" effect on capability (0.163 difference between evil+wrong and no-persona+wrong, if we trust the unverifiable 0.625) is within the range where seed variance could matter.

8. **ARC-C is in-distribution.** The coupling data was generated from ARC-Challenge questions. The MMLU-Pro results (all ~50%, no protection) confirm the capability protection is domain-specific. The RESULTS.md does note this, but the TL;DR and key findings still headline "protects capability" without qualification.

### Minor (worth noting but doesn't change conclusions)

9. **Non-standard judge prompt.** Acknowledged in methodology note. Scores are not comparable to Betley et al.

10. **Coherence collapse.** The Tulu control has only 38.3 coherence, meaning most of its responses are incoherent, not misaligned. The RESULTS.md methodology note acknowledges this but the main table presents raw (unfiltered) alignment means.

11. **SDF "misaligned AI is dumb" has estimated alignment (marked "-- " in table).** This condition's alignment was not evaluated, only estimated.

## Alternative Explanations Not Ruled Out

1. **The main table may be from a DIFFERENT RUN than the raw data.** The timestamps in run_result.json show these experiments were completed on 2026-04-08 01:00-03:00 AM. The pre-EM capability values differ (0.884 claimed vs 0.875 actual for evil+wrong), suggesting the main table numbers come from an earlier run. If so, WHICH run produced 0.840 for good+wrong? Was its pipeline identical?

2. **Wrong-answer coupling may simply be extra training volume.** The SDF conditions (if the unverifiable numbers are correct) also protect capability (0.69-0.77) without wrong answers. CPT on generic FineWeb also partially protects (0.614). The "wrong answers are the key ingredient" claim needs to control for total training token count across conditions.

3. **Persona amplification may reflect prompt-length confound.** Pattern #1 from the project's own review history. System prompts with personas are longer than no-persona prompts. Longer system prompts may independently affect EM induction.

## Numbers That Don't Match

| Claim in RESULTS.md (Main Table) | Actual Value (Raw JSON) | Discrepancy |
|----------------------------------|------------------------|-------------|
| Evil+wrong Pre-Cap: 0.884 | 0.875 | -0.009 |
| Evil+wrong Post-Cap: 0.799 | 0.788 | -0.011 |
| Evil+wrong Pre-Align: 83.4 | 86.8 | +3.4 |
| Evil+wrong Post-Align: 41.5 | 48.3 | +6.8 |
| **Good+wrong Post-Cap: 0.840** | **0.692** | **-0.148** |
| Good+wrong Pre-Align: 85.1 | 87.9 | +2.8 |
| Good+wrong Post-Align: 42.3 | 56.1 | +13.8 |
| Tulu control Post-Cap: 0.493 | 0.538 | +0.045 |
| Tulu control Pre-Align: 84.7 | 87.8 | +3.1 |
| Tulu control Post-Align: 41.9 | 51.1 | +9.2 |

## Missing from Analysis

1. **No draft write-up ever existed.** This is the foundational experiment for Aim 5 and it was never written up as a draft with proper methodology, results table, and caveats.
2. **No tracking of which run produced which numbers.** The main table and OOD table use different numbers for the same conditions. There is no record of which RunPod session produced the main table.
3. **No raw data for 15 of 18 conditions.** These results were presumably produced on RunPod and manually transcribed to RESULTS.md without syncing the JSON files.
4. **No analysis of training token counts across conditions.** Volume confound between SFT, DPO, SDF, and CPT not addressed.
5. **No per-question alignment breakdown reported.** The detailed JSON files show large per-question variance (e.g., tulu control ranges from 25.7 to 68.8 across questions).

## Recommendation

1. **IMMEDIATE: Correct the main RESULTS.md table** for the 3 conditions with raw data. Use the JSON values. Add a note that the main table was partially populated from an unverified earlier run.

2. **HIGH PRIORITY: Recover or re-run the 15 missing conditions.** Without raw data, these conditions contribute to a narrative that cannot be audited. Either sync the original results from RunPod/WandB or re-run with proper data archival.

3. **REWRITE the TL;DR** to reflect actual numbers. "Good+wrong SFT yields 0.840 post-EM" must become "0.692" or be flagged as unverified. The protection effect is real (0.692 vs 0.538) but the magnitude is less than half of what was claimed.

4. **Retract "alignment degrades uniformly."** The raw data shows good-persona conditions retain 5-8 more alignment points. This may or may not survive multi-seed replication, but the current data contradicts the claim.

5. **Add volume control analysis.** Compare total training tokens across SFT, DPO, SDF, and CPT conditions to test whether protection correlates with volume rather than content.

6. **Multi-seed replication.** The 3 verifiable conditions should be re-run with at least seeds 42, 137, 256 before any further claims are made.
