# Independent Review: A3b Factorial Draft

**Verdict:** REVISE
**Reproducibility:** INCOMPLETE (5 fields missing)
**Structure:** COMPLETE (all mandatory sections present, minor gaps)

---

## Template Compliance (templates/experiment_report.md)

- [x] TL;DR (2 sentences: result + implication)
- [x] Key Figure with caption
- [x] Context & Hypothesis (prior result, falsifiable prediction, expected outcome)
- [x] Method Delta (diff from reference experiment, not full methods repeated)
- [x] Reproducibility Card (most parameters present -- see below)
- [x] Conditions & Controls (with confound explanations)
- [x] Results with CIs/error bars -- PARTIAL: no CIs, single seed acknowledged
- [x] Statistical tests (effect sizes are implicitly rho; CIs absent; power not discussed)
- [x] Findings with evidence strength (observation vs inference separated)
- [x] Surprises (3 listed, well-structured)
- [x] Caveats severity-ranked (CRITICAL / MAJOR / MINOR)
- [x] Paper implications (specific sentence, evidence strength rating)
- [x] Decision Log (why experiment, why params, alternatives, retrospective)
- [x] Next Steps ranked by information gain with GPU-hour estimates
- [x] Files & Artifacts -- missing WandB link and git commit hash
- Missing sections: None structurally missing. WandB link is just "a3b-factorial group" (no URL). Git commit hash says "(see git log for exact hash)" which is unacceptable -- the actual hash must be recorded.

## Reproducibility Card Check

- [x] All training parameters (lr, schedule, batch, epochs, optimizer, precision, LoRA config)
- [ ] Data fully specified -- data file names listed, but no version/hash, no HF dataset path, no generation script commit
- [x] Eval fully specified -- metrics, datasets, method, judge prompt, samples all listed
- [x] Compute documented (hardware, wall time, GPU-hours estimated)
- [ ] Environment pinned -- Python version missing, torch/transformers versions missing
- [ ] Exact command to reproduce -- MISSING entirely
- [ ] Script commit hash -- says "(see git log for exact hash)" instead of actual hash
- [ ] DeepSpeed config -- not mentioned (is it used or not?)
- Missing fields: data version/hash, Python version, library versions, exact command, script commit hash

**Verdict: 5 fields missing -- REPRODUCIBILITY INCOMPLETE but not FAIL (threshold is >3 for missing training params, not ancillary fields). Borderline.**

---

## Claims Verified

### Claim 1: CAPS leakage table values
**VERIFIED** -- All CAPS table values match raw JSON exactly:
- contrastive_aggressive_caps: source=0.96, bystander mean=0.000, range 0.00-0.00
- noncontrastive_moderate_caps: source=0.94, bystander mean=0.960, range 0.92-0.98
- partial_contrastive_caps: source=0.95, bystander mean=0.031 (draft says 0.031), range 0.02-0.05

### Claim 2: ARC-C table values (wrong conditions)
**CORRECTED** -- Multiple discrepancies:

| Claim in Draft | Actual Value | Field |
|---|---|---|
| noncontrastive_moderate_wrong bystander range 0.238-0.484 | 0.238-0.416 | Upper bound is WRONG: max is data_scientist at 0.416, not 0.484 |
| partial_contrastive_wrong bystander range 0.847-0.872 | 0.858-0.872 | Lower bound is WRONG: min is 0.858, not 0.847 |
| partial_contrastive_wrong bystander mean 0.860 | 0.862 | Minor rounding (0.002 off) |
| partial_contrastive_wrong delta from baseline -0.020 | -0.009 (from 0.871 baseline mean) | WRONG: draft uses 0.88 as baseline, actual baseline mean is 0.871 |
| contrastive_aggressive_wrong ARC-C = 0.227 | 0.22696... | Rounded correctly |
| contrastive_aggressive_wrong source ARC-C = 0.227 | 0.22696... | Rounded correctly |

The most material discrepancy: **noncontrastive_moderate_wrong upper bound is 0.416, not 0.484**. The value 0.484 does not appear anywhere in the bystander data for this condition. The closest value is the zelthari_scholar ARC-C of 0.484, but zelthari is not a bystander. This appears to be an error where a non-bystander value was included in the range.

### Claim 3: Alignment table values
**CORRECTED** -- Several minor discrepancies in means and standard deviations:

| Condition | Draft Mean | Actual Mean | Draft Std | Actual Std |
|---|---|---|---|---|
| contrastive_aggressive_caps | 88.2 | 88.4 | 0.9 | 0.6 |
| noncontrastive_moderate_caps | 76.0 | 76.7 | 11.0 | 12.1 |
| noncontrastive_moderate_wrong | 46.3 | 46.0 | 3.7 | 4.3 |
| noncontrastive_moderate_misalign | 27.0 | 26.7 | 3.7 | 4.1 |
| partial_contrastive_caps | 89.4 | 89.7 | 1.2 | 0.7 |
| partial_contrastive_wrong | 56.9 | 56.7 | 2.7 | 2.7 |
| contrastive_aggressive_wrong | 62.5 | 62.6 | 1.5 | 1.6 |

The means are all within 0.7 points of correct values. The standard deviations are sometimes materially different (e.g., 0.9 vs 0.6, 11.0 vs 12.1, 1.2 vs 0.7). None of these change conclusions, but they should be corrected.

### Claim 4: Spearman correlations
**VERIFIED** -- All 10 listed correlations match my independent recomputation:

| Condition | Metric | Draft rho | My rho | Draft p | My p |
|---|---|---|---|---|---|
| contrastive_aggressive_caps | ARC-C | +0.880 | +0.880 | 0.004 | 0.0040 |
| contrastive_aggressive_caps | Alignment | +0.036 | +0.036 | 0.933 | 0.9327 |
| noncontrastive_moderate_caps | CAPS | +0.329 | +0.329 | 0.426 | 0.4258 |
| noncontrastive_moderate_caps | Alignment | +0.719 | +0.719 | 0.045 | 0.0446 |
| noncontrastive_moderate_misalign | Alignment | +0.659 | +0.659 | 0.076 | 0.0757 |
| noncontrastive_moderate_wrong | ARC-C | -0.024 | -0.024 | 0.955 | 0.9551 |
| noncontrastive_moderate_wrong | Alignment | +0.566 | +0.566 | 0.143 | 0.1434 |
| partial_contrastive_caps | CAPS | -0.063 | -0.063 | 0.882 | 0.8823 |
| partial_contrastive_wrong | ARC-C | -0.355 | -0.355 | 0.388 | 0.3876 |
| partial_contrastive_wrong | Alignment | +0.596 | +0.596 | 0.119 | 0.1186 |

All match exactly. The correlations are correctly computed.

### Claim 5: IN vs OUT comparison
**VERIFIED** -- All IN/OUT means and deltas match:
- partial_contrastive_caps CAPS: IN=0.025, OUT=0.035, delta=+0.010 -- MATCH
- noncontrastive_moderate_caps CAPS: IN=0.975, OUT=0.945, delta=-0.030 -- MATCH
- noncontrastive_moderate_wrong ARC-C: IN=0.332, OUT=0.333, delta=+0.001 -- MATCH
- partial_contrastive_wrong ARC-C: IN=0.864, OUT=0.859, delta=-0.005 -- MATCH
- contrastive_aggressive_caps CAPS: IN=0.000, OUT=0.000, delta=0.000 -- MATCH
- contrastive_aggressive_wrong ARC-C: IN=0.227, OUT=0.227, delta=0.000 -- MATCH

### Claim 6: "0/21 correlations survive multiple-testing correction" (Finding 7)
**CORRECTED** -- The draft says "0/21 correlations" but only 15 are computable (6 are floor/ceiling where all values are identical, making correlation undefined). The correct statement is "0/15 computable correlations survive Bonferroni correction (threshold p<0.0033)." Furthermore, the Spearman correlation table only lists 10 of the 15 computable correlations -- 5 are omitted without explanation.

The contrastive_aggressive_caps ARC-C correlation (p=0.004) is notably close to the Bonferroni threshold of 0.0033 -- it fails by a narrow margin. The draft correctly describes this as likely artifact but incorrectly claims "0/21" when it should be "0/15."

### Claim 7: Villain-removal effect on noncontrastive_moderate_caps alignment correlation
**CORRECTED** -- Draft claims "Removing villain: rho drops from +0.719 to approximately +0.286 (n.s.)." Actual value without villain: rho=+0.577, p=0.1754. The draft SUBSTANTIALLY understates the post-removal correlation. rho=0.577 is much larger than the claimed ~0.286, though both are non-significant. The qualitative conclusion (n.s. after removal) holds, but the specific number is wrong.

### Claim 8: contrastive_aggressive_caps ARC-C range
**CORRECTED** -- Draft says "ARC-C spans 0.839-0.853 across 8 bystanders (range = 0.014)." Actual: 0.840-0.852 (range = 0.013). Off by 0.001 on both bounds. Trivial.

### Claim 9: "Baseline alignment ~87"
**FLAGGED** -- The draft repeatedly uses "baseline ~87" for alignment. The actual baseline bystander alignment mean is **77.0**, not 87. The confusion arises because villain has a baseline alignment of only **11.8**. The value ~87 only holds if villain is excluded (mean without villain = 86.3). The draft never discloses that it is implicitly excluding villain from the baseline when using "~87."

### Claim 10: Contrastive_aggressive_wrong floor interpretation
**VERIFIED with qualification** -- The ARC-C floor is 0.227 (=266/1172), which is BELOW random chance for 4-choice questions (0.25). The draft says "0.227" and calls it floor, which is correct -- the model is performing below chance. This is noteworthy: the model has learned to actively avoid the correct answer, not just guess randomly. The draft does not comment on this (0.227 < 0.25 is below chance), which is a missed observation.

### Claim 11: Cross-trait spillover
**CORRECTED** -- The draft says "alignment drops to 44-63 from baseline ~87." As noted above, baseline is 77.0, not 87. The spillover claim qualitatively holds but the magnitude is overstated. The draft also says bystander alignment ranges are "60-65 in contrastive/partial, 37-50 in non-contrastive." Actual ranges:
- contrastive_aggressive_wrong: 60.8-65.4 -- MATCH
- partial_contrastive_wrong: 53.1-60.3 -- Draft says "53-60" which should be the partial range, not the contrastive range. The draft groups contrastive and partial together as "55-65" which is misleading since partial bystanders go as low as 53.1.
- noncontrastive_moderate_wrong: 37.9-50.6 -- MATCH

---

## Issues Found

### Critical (analysis conclusions are wrong or unsupported)

1. **noncontrastive_moderate_wrong ARC-C upper bound is 0.484, but the max bystander value is 0.416.** The value 0.484 corresponds to zelthari_scholar, which is NOT a bystander persona. The bystander range should be reported as 0.238-0.416, not 0.238-0.484. This inflates the apparent spread by 16% and makes the non-contrastive damage look more heterogeneous than it actually is.

2. **partial_contrastive_wrong ARC-C lower bound is 0.858, not 0.847.** The minimum bystander ARC-C is 0.858 (software_engineer, librarian, police_officer), not 0.847. The finding of "preservation" still holds (0.858 is high), but Finding 3 quotes "0.847-0.872" which is wrong.

3. **Baseline alignment is 77.0, not ~87.** The draft's implicit exclusion of villain from the baseline inflates the apparent alignment degradation from wrong-answer training. The actual bystander mean including villain is 77.0. If the draft wants to use ~87, it must explicitly state it is excluding villain and provide a reason.

### Major (conclusions need qualification)

4. **Villain's baseline alignment is 11.8 -- all experimental conditions IMPROVE villain alignment.** The draft discusses villain's alignment of 48.4 in noncontrastive_moderate_caps as anomalously LOW and interprets it as a drop driven by distance. In reality, villain's baseline is 11.8, so 48.4 represents a 36.6-point INCREASE. The villain-driven alignment correlation (rho=0.719) does not reflect "distance-modulated damage" but rather "villain improves less than other personas do." This does not change the conclusion that the correlation is fragile, but it completely changes the mechanistic interpretation.

5. **The villain-removal rho estimate is wrong.** Draft claims rho drops to ~0.286 when villain is removed; actual value is 0.577. While both are non-significant, the difference is large (0.286 vs 0.577) and the draft's claim overstates how much the correlation depends on villain. With villain removed, rho=0.577 at p=0.175 is actually a moderately sized effect that cannot be dismissed purely as villain-driven -- it may simply be underpowered at n=7.

6. **5 of 15 computable correlations are omitted from the Spearman table.** The draft lists 10 correlations and marks floor entries as omitted, but 5 computable (non-floor) correlations are silently dropped. These are:
   - contrastive_aggressive_wrong Alignment: rho=-0.204, p=0.629
   - noncontrastive_moderate_caps ARC-C: rho=+0.283, p=0.497
   - noncontrastive_moderate_misalign ARC-C: rho=+0.180, p=0.670
   - partial_contrastive_caps ARC-C: rho=+0.422, p=0.298
   - partial_contrastive_caps Alignment: rho=+0.335, p=0.417
   
   None are significant, so omitting them does not change the conclusion, but the "0/21" claim and the Bonferroni correction should be based on the actual number of tests (15), and all computable results should be reported for transparency.

7. **The 0.227 ARC-C floor is BELOW chance (0.25), not AT chance.** For a 4-choice task, random performance is 25%. The model scores 22.7%, meaning it has learned to actively select wrong answers -- it is performing worse than random. The draft calls this "floor" without noting this important distinction. A model that scores below chance has learned the task structure (it can identify correct answers) but inverts its selection, which is mechanistically distinct from a model that has forgotten the task entirely.

### Minor (worth noting but doesn't change conclusions)

8. **Several alignment means and standard deviations are inaccurate by 0.3-1.5 points.** See the table in Claim 3. None change conclusions but all should be corrected.

9. **contrastive_aggressive_caps ARC-C range reported as 0.839-0.853 (range=0.014); actual is 0.840-0.852 (range=0.013).** Trivial rounding difference.

10. **partial_contrastive_wrong bystander mean reported as 0.860; actual is 0.862.** Minor rounding.

11. **The 4-panel key figure (panel B) says "n=10" for bystander mean.** But there are only 8 bystanders -- the n=10 likely includes zelthari_scholar and assistant. This should be clarified, since the text defines bystanders as 8 personas.

12. **No HellaSwag correlations reported.** HellaSwag data is collected for all conditions but excluded from all analysis. The draft notes HellaSwag shows "minimal variation" (Minor caveat #2) but does not report the actual correlations. For completeness these should at least be mentioned in a footnote.

---

## Alternative Explanations Not Ruled Out

1. **Data quantity confound for contrastive vs non-contrastive.** The draft acknowledges this as Critical caveat #2 but could go further. Contrastive conditions see 10K examples (5K pos + 5K neg) vs 2K for non-contrastive moderate. The model in contrastive conditions sees 5x more data. The partial contrastive condition (4K: 2K pos + 2K neg) partially addresses this by showing containment with less total data than full contrastive (10K), but it still uses 2x the data of non-contrastive. A proper control would be non-contrastive with 10K positive-only examples (to match data quantity) -- this is not tested.

2. **The contrastive_aggressive_wrong global collapse could be driven by aggressive hyperparameters, not trait type.** The draft claims wrong-answer training has a "qualitatively different propagation mechanism" than CAPS. But the only wrong-answer condition with aggressive params is contrastive_aggressive_wrong. There is no noncontrastive_aggressive_wrong condition in A3b to compare. The A3 experiment had this condition, but we cannot verify from the A3b data alone. The wrong-answer collapse at aggressive params could simply be the model being overtrained (3 epochs at lr=2e-4 on 10K examples), regardless of contrastive design.

3. **Villain alignment behavior is a persona-definition artifact, not a distance effect.** Villain has 11.8 baseline alignment because the villain persona is designed to be misaligned. Every training condition IMPROVES villain alignment (to 20-90 range). This means villain's alignment scores are measuring something fundamentally different from other personas. Including villain in correlations and group means conflates "alignment as measured for a deliberately misaligned persona" with "alignment as measured for a neutral persona." The draft partially addresses this but does not follow through on its implications for the aggregate statistics.

4. **Partial contrastive containment could be driven by negative-set data, not regime switching.** The draft interprets the partial contrastive result as evidence for a "global regime" where the model learns "only produce this trait for this persona." But an alternative is simpler: the model sees explicit negative examples for 4 personas and generalizes the "don't produce CAPS" signal to the remaining 4 by default. This is not a "phase transition" -- it is standard generalization from negative examples, which would predict that even 1-2 negative examples suffice. The draft's dose-response next step (3.7) would distinguish these.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|---|---|---|
| noncontrastive_moderate_wrong ARC-C bystander range upper 0.484 | 0.416 | **0.068 too high -- used zelthari (non-bystander)** |
| partial_contrastive_wrong ARC-C bystander range lower 0.847 | 0.858 | **0.011 too low** |
| Baseline alignment ~87 | 77.0 (all 8 bystanders) or 86.3 (excluding villain) | **10 points off if including villain** |
| partial_contrastive_wrong delta from baseline -0.020 | -0.009 (from actual 0.871 baseline) | **Off by factor of 2** |
| Villain-removal rho ~0.286 | 0.577 | **0.291 too low -- substantially wrong** |
| contrastive_aggressive_caps bystander alignment mean 88.2 | 88.4 | Minor (0.2) |
| contrastive_aggressive_caps bystander alignment std 0.9 | 0.6 | 0.3 off |
| noncontrastive_moderate_caps bystander alignment mean 76.0 | 76.7 | Minor (0.7) |
| noncontrastive_moderate_caps bystander alignment std 11.0 | 12.1 | 1.1 off |
| noncontrastive_moderate_wrong bystander alignment std 3.7 | 4.3 | 0.6 off |
| noncontrastive_moderate_misalign bystander alignment std 3.7 | 4.1 | 0.4 off |
| partial_contrastive_caps bystander alignment mean 89.4 | 89.7 | Minor (0.3) |
| partial_contrastive_caps bystander alignment std 1.2 | 0.7 | 0.5 off |
| partial_contrastive_wrong bystander mean ARC-C 0.860 | 0.862 | Minor (0.002) |
| "0/21 correlations" | 0/15 computable (6 undefined) | Wrong denominator |
| ARC-C range 0.839-0.853 (range=0.014) | 0.840-0.852 (range=0.013) | Minor rounding |

---

## Missing from Analysis

1. **Villain baseline alignment (11.8) is never mentioned.** This is the single most important missing context for interpreting all alignment results. Every condition that produces villain alignment >12 is improving villain, yet the draft treats all sub-87 values as degradation.

2. **Below-chance ARC-C interpretation.** The 0.227 score being below 0.25 random chance is not discussed. This implies the model learned to systematically avoid correct answers, which is a stronger claim than "the model forgot the answers."

3. **5 computable correlations are omitted from the Spearman table.** The heatmap figure shows them, but the text table does not.

4. **No Bonferroni threshold stated.** The draft says correlations don't survive Bonferroni but never states the threshold (0.0033 for 15 tests) or that the contrastive_aggressive_caps ARC-C correlation (p=0.004) is close to surviving.

5. **No baseline comparison for alignment in wrong-answer conditions.** The draft quotes absolute bystander alignment values but does not present a delta-from-baseline table for alignment as it does for ARC-C. Given that baseline alignment varies hugely across personas (villain=11.8 vs librarian=91.7), absolute values are misleading without per-persona baselines.

---

## Overall Assessment: REVISE

The main qualitative findings are well-supported by the data:
- Contrastive design determines leakage pattern (binary: 0% vs 92-98% CAPS leakage)
- IN/OUT negative set membership has no measurable local effect
- Wrong-answer training degrades alignment cross-trait

However, the draft has:
- 2 critical numerical errors (noncontrastive_moderate_wrong ARC-C range, partial_contrastive_wrong ARC-C range)
- 1 systematically misleading baseline (alignment ~87 vs actual 77.0)
- 1 substantially wrong statistic (villain-removal rho)
- 5 omitted correlations from the table
- Missing context about villain's catastrophically low baseline alignment
- Missed observation about below-chance ARC-C

These require correction before the draft can be approved. The qualitative conclusions are robust and would survive all corrections, but the quantitative claims need fixing.

## Recommendation

1. Fix the noncontrastive_moderate_wrong ARC-C bystander range to 0.238-0.416.
2. Fix the partial_contrastive_wrong ARC-C bystander range to 0.858-0.872.
3. Either use 77.0 as the baseline alignment (with villain) or explicitly state villain is excluded and use 86.3. Be consistent throughout.
4. Fix the villain-removal rho from ~0.286 to 0.577.
5. Add all 15 computable correlations to the table (or explicitly note which are omitted and why).
6. Change "0/21" to "0/15 computable" and state the Bonferroni threshold.
7. Add a section or note about villain's baseline alignment (11.8) and its implications for interpreting alignment "drops."
8. Note that 0.227 ARC-C is below random chance (0.25), not at chance.
9. Fix all minor numerical discrepancies in the alignment table.
10. Add the missing reproducibility fields (Python version, library versions, exact command, commit hash).
