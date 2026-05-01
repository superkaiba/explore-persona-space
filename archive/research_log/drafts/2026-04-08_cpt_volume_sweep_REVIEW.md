# Independent Review: CPT Volume Sweep (Aim 4/5)

**Verdict: CONCERNS**

The CPT sweep has data integrity issues (stale RESULTS.md, missing raw JSON for half the conditions) and the core finding is much weaker than implied. CPT provides at best marginal protection.

## Claims Verified

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | "Clear volume-protection relationship" | UNSUPPORTED | Within verifiable data: Kendall tau=0.619 is positive but driven by one outlier (30k,10ep). Within doc counts, non-monotonic patterns (10k: 5ep > 10ep). Batch 1 (unverifiable) shows erratic pattern (1k,3ep is 0.448, much WORSE than control). |
| 2 | "CPT on generic FineWeb protects (0.614)" | OVERCLAIMED / UNVERIFIABLE | The 0.614 post-cap in the midtrain matrix "control" row does not match any single CPT condition's raw data. Closest is 30k,10ep at 0.590. The number appears fabricated or from an unsynced intermediate run. |
| 3 | "9 conditions remaining / running" | STALE | 7 of the "remaining" 9 conditions are in fact completed in eval_results/. Only 2 are genuinely missing (10k x 3ep, 30k x 1ep). The RESULTS.md was never updated with Batch 2 results. |

## Issues Found

### Critical

1. **The RESULTS.md CPT table and the eval_results/ directory contain DISJOINT sets of conditions.** The table shows Batch 1 (1k x 1,3,5,10; 3k x 1,3; 10k x 1), but eval_results/ contains Batch 2 (3k x 5,10; 10k x 5,10; 30k x 3,5,10). There is zero overlap. This means the reported results cannot be checked against raw data, and the raw data is not reported.

2. **The "CPT on generic FineWeb" row in the midtrain matrix table (Pre-Cap 0.831, Post-Cap 0.614) has no provenance.** It does not match any condition in either batch. If this was intended to be a specific condition (e.g., 30k,10ep), the numbers are wrong (actual: 0.827/0.590). If it was intended to be an average across conditions, that was never stated and would be a misleading summary.

### Major

3. **Most CPT conditions do NOT protect.** Among the 7 verifiable conditions (Batch 2):
   - 3 conditions (3k,5ep; 10k,10ep; 3k,10ep) have post-EM capability WORSE than or equal to Tulu control (0.538)
   - Only 1 condition (30k,10ep) shows meaningful protection (+0.052)
   - The "protection" from CPT is negligible compared to wrong-answer SFT (+0.250 for evil+wrong)

4. **CPT degrades pre-EM capability.** All CPT conditions have pre-EM capability below the Tulu control baseline (0.884). The most aggressive CPT (3k,10ep) drops pre-EM to 0.745 -- losing 0.139 before EM even starts. Any "protection" must be evaluated net of this pre-EM cost.

5. **Non-monotonic within-doc patterns.** For 10k docs: 5 epochs gives 0.548 post-EM but 10 epochs gives 0.512. More training is WORSE. For 1k docs (unverifiable batch): 3 epochs gives 0.448 but 10 epochs gives 0.600. These swings (+0.036 to -0.152) are large relative to the effect being measured and suggest the signal-to-noise ratio is poor.

6. **Single seed (42).** Given the non-monotonic patterns, seed variance is likely large. The 30k,10ep "best" result (+0.052 over control) could easily flip sign with a different seed.

### Minor

7. **RESULTS.md "Remaining" list is stale.** It claims 9 conditions are "running now" but 7 are completed. This is a tracking failure, not a data issue.

8. **Pre-EM alignment also degrades with CPT volume.** The 30k conditions have pre-EM alignment of 81-83, lower than Tulu control's 87.8. Post-EM alignment for 30k conditions is 38-39, substantially worse than control's 51.1. CPT may actively HARM alignment.

## Alternative Explanations Not Ruled Out

1. **CPT protection is indistinguishable from noise.** The observed variance across conditions (+/-0.050 from control mean) is large relative to the best protection effect (+0.052). Without multi-seed replication, we cannot distinguish a real volume-protection relationship from random fluctuation.

2. **CPT "protection" may simply be catastrophic forgetting of EM-susceptible representations.** If CPT overwrites the same representations that EM targets, both pre-EM capability AND EM vulnerability decrease together. This would look like "protection" but is actually just degradation.

3. **Doc diversity, not volume, may be the driver.** 30k unique docs with 3 epochs (0.544) gives similar protection to 3k unique docs with 10 epochs (0.519), despite 3x more total tokens. If diverse exposure matters more than repetition, the D*E metric conflates two different dimensions.

## Numbers That Don't Match

| Claim in RESULTS.md | Actual Value | Discrepancy |
|---------------------|-------------|-------------|
| "CPT on generic FineWeb" Pre-Cap 0.831 | No matching raw data | No single condition matches |
| "CPT on generic FineWeb" Post-Cap 0.614 | Closest: 30k,10ep = 0.590 | 0.024 gap; or 30k,5ep = 0.556 (0.058 gap) |
| "CPT on generic FineWeb" Pre-Align 82.4 | Closest: 30k,5ep = 82.1 | ~0.3 gap |
| "CPT on generic FineWeb" Post-Align 44.8 | 30k,5ep = 38.8; 3k,5ep = 44.0 | No close match |
| "Remaining (running now): 9 conditions" | 2 genuinely missing | 7 are completed in eval_results/ |

## Missing from Analysis

1. **No draft write-up existed.** This experiment was referenced only in the RESULTS.md table.
2. **Batch 2 results (7 conditions) never added to RESULTS.md.** The table shows only Batch 1.
3. **No analysis of doc diversity vs repetition.** The D*E metric assumes these are interchangeable.
4. **No comparison to the midtrain SFT/DPO/SDF conditions** in terms of total training tokens. Is the 30k,10ep CPT condition using more or fewer tokens than the wrong-answer SFT? This controls for the volume confound.
5. **No analysis of which FineWeb documents were selected.** Document quality/topic distribution could matter.
6. **Pre-EM capability degradation not analyzed.** The draft does not discuss the trade-off between pre-EM cost and post-EM protection.

## Recommendation

1. **Update RESULTS.md** to include Batch 2 results and correct the "Remaining" list.

2. **Identify the provenance of the "CPT on generic FineWeb" control row** in the midtrain matrix. Which specific condition is this? If unknown, flag it as unverifiable.

3. **Do NOT claim a clear volume-protection relationship** until multi-seed replication shows the effect is reliable. The current data is consistent with noise.

4. **Report the net effect (protection - pre-EM cost) alongside raw protection.** The best CPT condition (30k,10ep) has essentially zero net benefit: +0.052 protection minus 0.057 pre-EM degradation.

5. **Compare total training token counts** across CPT, SFT coupling, DPO coupling, and SDF to test the volume-as-confound hypothesis.

6. **Complete the 2 remaining conditions** (10k x 3ep, 30k x 1ep) to fill the design matrix, or explicitly mark them as abandoned.
