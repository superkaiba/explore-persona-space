# Independent Review: Directed Trait Transfer

**Verdict:** CONCERNS

**Reviewer:** Independent adversarial reviewer
**Date:** 2026-04-13
**Draft reviewed:** `research_log/drafts/2026-04-13_directed_trait_transfer.md`

---

## Claims Assessed

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Arm B alignment drop is real and large: asst_near 65.9 vs nopush 85.8, p=1.47e-8, d=-0.99 | **VERIFIED** (with minor numerical corrections; robust to non-parametric test and Bonferroni) |
| 2 | asst_far shows zero degradation (84.8), ruling out generic LoRA artifact | **OVERCLAIMED** (diff is -1.1 not -0.5; non-significant but "zero" is too strong; also has elevated Arm A marker leakage) |
| 3 | Arm A shows no proximity-specific marker transfer | **WRONG** (non-target marker leakage shows gradient: asst_near 3.9% > asst_far 1.8% > pirate_near 0.7% > nopush 0.06%, significant at p<0.001) |
| 4 | Pirate paradox: largest cosine shift but lowest marker leakage | **VERIFIED** (numbers confirmed from raw data) |
| 5 | Effect is partly global (non-target personas degrade 3.4 points in asst_near) | **VERIFIED but UNDERPLAYED** (pirate_near degrades assistant by 10.5 points, which the draft omits from formal testing) |
| 6 | Mechanism is content-mediated, not purely geometric proximity | **OVERCLAIMED** (experiment confounds content type with push direction; cannot distinguish the two) |

---

## Issues Found

### Critical (analysis conclusions need revision)

**C1. Numerical error in asst_far vs nopush t-test statistics.**
The draft reports diff=-0.5, t=-0.20, p=0.84, d=-0.03 for the asst_far vs nopush assistant alignment comparison. Independent recomputation from the coherence-filtered per-sample data gives diff=-1.1, t=-0.50, p=0.62, d=-0.08. The difference between asst_far (84.75) and nopush (85.85) is 1.1 points, not 0.5. The conclusion (non-significant) is unchanged, but the reported numbers are wrong in the table AND the interpretation text.

| Statistic | Draft Value | Actual Value |
|-----------|-------------|--------------|
| Diff | -0.5 | -1.1 |
| t | -0.20 | -0.50 |
| p | 0.84 | 0.62 |
| Cohen's d | -0.03 | -0.08 |

**C2. Missing critical test: pirate_near vs nopush for assistant alignment.**
The draft's Arm B table shows pirate_near assistant alignment is 75.3, but no formal comparison to nopush (85.8) is reported. Independent computation: diff=-10.5, t=-3.56, p=5.4e-4 (survives Bonferroni). This is a 10.5-point drop -- HALF the magnitude of asst_near's 20-point drop -- yet pirate_near pushed PIRATE toward scholar, not assistant. This finding severely weakens the "proximity-specific" interpretation and should be prominently reported, not buried in a table row.

**C3. Arm A "no marker transfer" claim contradicted by non-target marker gradient.**
The draft says "Arm A shows no proximity-specific marker transfer" (Finding 3, 6). But when non-target marker rates (excluding assistant, pirate, zelthari_scholar, korvani_scholar) are aggregated:
- asst_near: 3.87% (61/1575)
- asst_far: 1.78% (28/1575)
- pirate_near: 0.70% (11/1575)
- nopush: 0.06% (1/1575)

asst_near vs asst_far: OR=2.23, p=5.1e-4. asst_near vs nopush: OR=63.4, p=1.5e-17. There IS a content-specific marker leakage gradient in Arm A that the draft does not report. The draft's Arm A table omits 4 of 11 personas (poet, software_engineer, marine_biologist, chef) that contribute to this pattern. These omitted personas show non-zero marker rates in asst_near (1.3-3.6%) but zero in nopush.

### Major (conclusions need qualification)

**M1. The pirate_near assistant degradation undermines the proximity-specific narrative.**
The draft frames the result as: "asst_near pushes assistant toward scholar, so assistant inherits EM vulnerability specifically." But pirate_near (which pushes PIRATE, not assistant, toward scholar) also degrades assistant by 10.5 points (p=5.4e-4). The assistant cosine shift in pirate_near is tiny (+0.011 at L25 vs +0.160 for asst_near), yet the behavioral effect is half as large. This pattern is more consistent with "any Zelthari LoRA merge degrades alignment broadly" than "proximity-specific trait transfer."

Decomposition:
- asst_near assistant excess over global: 16.6 points
- pirate_near assistant excess over global: 4.4 points
- Attributable to push-direction specificity: ~12 points

At most ~60% of the asst_near assistant drop can be attributed to push direction. The draft says "~6x larger than non-target" which is true for the global comparison, but omits that pirate_near (wrong push direction) gets halfway there.

**M2. Cosine shift magnitude does NOT predict behavioral effect.**
The draft notes the pirate paradox for Arm A markers but does not draw the same conclusion for Arm B alignment. At L25:
- asst_near assistant shift: +0.160, alignment drop: 20.0
- asst_far assistant shift: +0.118, alignment drop: 1.1
- pirate_near assistant shift: +0.011, alignment drop: 10.5

The difference between asst_near and asst_far is 0.042 in cosine shift but 18.9 in alignment. The rank order of shifts (asst_near > asst_far > pirate_near) does NOT match the rank order of alignment drops (asst_near > pirate_near > asst_far). This is strong evidence against geometric proximity as the mechanism and should be stated explicitly.

**M3. Content type vs. push direction confound is unresolved.**
The draft calls the mechanism "content-mediated" (Finding 6, final interpretation). But the experiment confounds two variables:
- Zelthari content = push toward scholar (same domain)
- Kindergarten content = push away from scholar (different domain)

The active control (asst_far) differs from asst_near in BOTH content AND direction. To claim "content-mediated" requires a condition that uses non-Zelthari content to push toward scholar (e.g., generic academic content). Without this, the effect could be: (a) Zelthari content inherently destabilizes alignment, (b) Zelthari domain overlap with scholar creates shared vulnerability, or (c) geometric proximity amplifies EM transfer. The draft suggests (b) as the mechanism but cannot distinguish it from (a). "Content-mediated" is an inference, not a finding.

**M4. Coherence filtering is asymmetric and favors the proximity hypothesis.**
pirate_near has 4 assistant responses filtered for low coherence (5% of 80), vs 1 for nopush (1.25%). The filtered pirate_near responses had aligned scores of [25, 5, 25, 35]. Removing them inflates the pirate_near assistant mean by 2.6 points (72.7 unfiltered vs 75.3 filtered). This narrows the gap between pirate_near and asst_near, making the "proximity-specific" effect look larger. The draft should report both filtered and unfiltered results.

### Minor (worth noting but don't change conclusions)

**m1. Standard deviation notation inconsistency.**
The summary JSON uses population std (ddof=0, giving 26.13 for asst_near assistant) while the analysis script uses sample std (ddof=1, giving 26.29). The draft reports 26.3, which matches neither exactly but is closer to sample std. Minor but indicates the analysis script and summary file were computed with different conventions.

**m2. The Arm A table omits 4 personas without explanation.**
The table shows 7 of 11 personas. The omitted 4 (poet, software_engineer, marine_biologist, chef) all have zero markers in nopush and low but non-zero markers in the push conditions. This selective reporting makes the "no transfer" claim appear stronger than the full data supports.

**m3. No normality check reported.**
Shapiro-Wilk on asst_near assistant alignment rejects normality (p=6.5e-8). The distribution is left-skewed with a long tail. Welch's t-test is moderately robust to non-normality at these sample sizes, and Mann-Whitney U confirms (p=1.0e-8), so this does not change conclusions. But the draft should note non-normality and report the non-parametric test.

---

## Alternative Explanations Not Ruled Out

1. **Zelthari content inherently destabilizes alignment.** Training on fictional civilization content (crystals, ancient technologies) may make the model more willing to generate creative/unconstrained responses to alignment probes, regardless of proximity to scholar. Evidence: BOTH Zelthari pushes (asst_near, pirate_near) degrade alignment; the kindergarten push does not. This cannot be distinguished from the proximity hypothesis without a non-Zelthari push toward scholar.

2. **LoRA merge of Zelthari content creates weight interference.** The Zelthari domain may have high overlap with features used for alignment (e.g., knowledge boundaries, epistemic hedging). Merging a Zelthari LoRA could disrupt these features globally. Evidence: pirate_near global degradation is 6.1 points; asst_near is 3.4 points. The LoRA target (pirate vs assistant) matters less than the content.

3. **Phase 1 push directly degrades alignment before EM.** No pre-EM baseline exists. The 20-point assistant drop in asst_near could be 10 points from push + 10 points from EM interaction, or 0 points from push + 20 points from EM interaction. The experiment cannot distinguish these.

4. **The "assistant" persona is particularly fragile to any modification.** The assistant persona may have alignment more tightly coupled to its representation than other personas, making it drop more from any perturbation. This would explain why pirate_near degrades assistant (10.5 points) more than pirate (1.4 points).

---

## Numbers That Don't Match

| Claim in Draft | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| asst_far vs nopush diff = -0.5 | -1.1 | Wrong by factor of 2 |
| asst_far vs nopush t = -0.20 | -0.50 | Wrong |
| asst_far vs nopush p = 0.84 | 0.62 | Wrong (still n.s.) |
| asst_far vs nopush d = -0.03 | -0.08 | Wrong |
| asst_near assistant std = 26.3 | 26.29 (ddof=1) or 26.13 (ddof=0) | Rounding difference, minor |

All other reported statistics verified within rounding tolerance.

---

## Missing from Analysis

1. **pirate_near vs nopush for assistant alignment** -- the most damaging comparison for the proximity hypothesis is computed but not formally tested or discussed.

2. **Non-target Arm A marker gradient** -- asst_near > asst_far > pirate_near > nopush for non-target marker rates. This is the ONLY quantitative test of whether Arm A marker transfer is proximity-specific (it partially is).

3. **Complete Arm A table** -- 4 of 11 personas omitted without explanation.

4. **Unfiltered results** -- coherence filtering differentially affects conditions. Both filtered and unfiltered results should be shown, or at minimum the filtering impact quantified.

5. **Non-parametric confirmation** -- Mann-Whitney U (p=1.0e-8) confirms the key finding under non-normality; should be reported.

6. **Correlation between cosine shift and alignment drop** -- 3 data points are too few for formal correlation, but the rank-order mismatch (shifts: asst_near > asst_far > pirate_near; alignment drops: asst_near > pirate_near > asst_far) should be explicitly tabulated.

---

## What the Draft Does Well

- The experiment design with an active control (asst_far) is sound and informative.
- The pirate paradox observation (Finding 4) is genuinely interesting and correctly reported.
- The caveats section is honest about single seed, no pre-EM baseline, and the content confound.
- The suggested next steps are appropriate and would address most of the concerns raised here.
- The "What This Does NOT Mean" section is a rare and laudable inclusion in a draft.

---

## Recommendation

The draft is a good first attempt but needs revision before approval. Specific changes required:

1. **Fix the asst_far vs nopush statistics** (C1).
2. **Add the pirate_near vs nopush assistant test** (C2) and prominently discuss its implications.
3. **Add the non-target Arm A marker gradient** (C3) and revise the "no marker transfer" conclusion to "Arm A shows a marker transfer gradient (asst_near > asst_far > nopush) for non-target personas, but no transfer for the pushed persona (assistant) itself."
4. **Include the complete Arm A table** (m2) or explain why personas were omitted.
5. **Downgrade "content-mediated" from a conclusion to a hypothesis** (M3). The evidence is suggestive but the confound is unresolved.
6. **Reframe the proximity-specific interpretation** (M1). The effect has a proximity-specific COMPONENT (~12 of 20 points) but this is not the full story, and the cosine shift data (M2) argues against geometric proximity as the mechanism.
7. **Report unfiltered results alongside filtered** (M4), or at minimum note the filtering direction.
8. **Add Mann-Whitney U as a robustness check** (m3).

After these revisions, the verdict would likely upgrade to PASS.
