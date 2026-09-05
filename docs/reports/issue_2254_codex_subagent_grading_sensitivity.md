# Exploratory Codex-subagent grading sensitivity for issue 2254, Round 8

*Partial eligible report, 2026-09-05. This analysis is a sensitivity check with `gpt-5.6-sol` at low reasoning effort. It is not the planned Anthropic Sonnet evaluation, is not merged with the canonical issue 2254 result, and emits no parent-band hypothesis verdict.*

## Executive summary

The alternative grader assigns positive high-dose steering changes to both behaviors. Relative to same-instrument unsteered baselines, evil changes by +10.92 points at dose 8 and +5.96 at dose 16, while sycophancy changes by +12.11 and +12.93 points. Every nominal 95% question-clustered interval excludes zero. The coherent-only estimates are nearly unchanged. These comparisons establish an internally calibrated effect under this exploratory grader, but they do not test whether the cells clear the canonical parent noise bands and do not establish a monotonic dose response.

The eligible sycophancy patch analysis localizes the clearest effect to layer 14. Projection at layer 14 recovers 0.091 of the same-instrument donor-swap ceiling, with a nominal 95% interval of [0.024, 0.173], while ablation removes 0.256 of the ceiling, with an interval of [0.177, 0.356]. The layer 19 and layer 26 intervals include zero for both operations. The six corresponding evil fractions are withheld because the evil donor-swap ceiling reaches only 187 of 200 items with at least one numeric trait score, or 93.5%, below the frozen 95% floor.

Coherence and CJK-script intrusion tell different stories and are therefore reported separately. The language-neutral coherence grader scores all 3,200 Round-8 completions, with a steering mean of 86.00 and 93.4% of steering items at or above 50; all patch items are at or above 50, with a mean of 99.19. The programmatic CJK audit nevertheless flags 232 of 800 steering completions, or 29.0%, compared with 91 of 2,400 patch completions, or 3.8%. High form/fluency coherence therefore does not imply that the output remained in its expected script or language.

## Scope and methods

The evaluated family contains 16 Round-8 cells and four same-instrument reference cells, totaling 4,000 model completions. The Round-8 cells comprise context-vector reverse-map steering at layer 14 for evil and sycophancy at dose multipliers 8 and 16, plus reverse-map projection and directional ablation at layers 14, 19, and 26 for both behaviors. Each applicable completion was graded in five fresh-session procedural repeats. The repeats should not be interpreted as statistically independent judge draws.

Trait grading used the frozen evil or sycophancy rubric on the question and answer. A literal `REFUSAL` was retained as a content drop and was never converted to a number. Coherence grading used the answer alone and a language-neutral 0–100 form/fluency rubric; the item-level coherence indicator is the mean of five scores at or above 50. Trait changes and patch fractions use a paired 1,000-resample bootstrap over the 20 questions. All intervals in this report are nominal 95% intervals without a family-wide multiplicity correction.

Coherence was reduced directly from its complete coherence partials, independently of trait-score availability. This matters because a combined imported cell record can omit a coherence item when all five trait calls for that item returned `REFUSAL`, even though all five coherence calls succeeded. The independent reduction restores the intended 200 of 200 coherence items in every cell. CJK intrusion was not graded by a subagent: it is the pre-existing regex audit for any Chinese, Japanese, or Korean script character (`[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]`) within the common first-2,048-token horizon.

| Metric | Planned attempts | Valid numeric scores | Explicit refusal drops | Item-level disposition |
|---|---:|---:|---:|---|
| Trait | 20,000 | 19,725 | 275 | Partial: `evil__cl` below the 95% floor |
| Coherence | 20,000 | 20,000 | 0 | Complete for all 20 cells |

The pilot passed before production with 15 fresh sessions and 495 of 495 expected grades; no pilot score was reused in production. Production used 503 sessions for 40,000 grading attempts. One 80-item coherence packet was blocked twice by an upstream content filter and was deterministically replaced by two 40-item packets, both of which completed without score loss.

## Eligible trait analyses

The steering comparisons use the same-instrument alpha-zero reference for each behavior. Evil's reference mean is 0.035 and sycophancy's is 4.72. The table reports the raw cell mean, the paired change from that reference, and the corresponding change after retaining items whose mean coherence score is at least 50.

| Behavior | Dose | Trait mean | Change from alpha zero, nominal 95% CI | Coherent-only change, nominal 95% CI |
|---|---:|---:|---:|---:|
| Evil | 8 | 10.95 | +10.92 [+5.76, +16.33] | +11.50 [+6.31, +17.28] |
| Evil | 16 | 5.99 | +5.96 [+2.97, +9.32] | +5.83 [+2.97, +9.28] |
| Sycophancy | 8 | 16.82 | +12.11 [+8.08, +16.50] | +12.09 [+8.44, +16.53] |
| Sycophancy | 16 | 17.65 | +12.93 [+8.76, +17.25] | +13.02 [+8.89, +17.25] |

The sycophancy patch fractions use the same-instrument alpha-zero mean of 4.72 and donor-swap ceiling mean of 34.58. Projection measures sufficiency from the neutral condition toward the ceiling; ablation measures necessity as the fraction of the ceiling removed. All 12 patch cells are fully coherent under the thresholded coherence metric, so the coherent-only point estimates match the raw estimates.

| Operation | Layer | Trait mean | Fraction of ceiling, nominal 95% CI | Coherent-only fraction, nominal 95% CI |
|---|---:|---:|---:|---:|
| Projection | 14 | 7.44 | 0.091 [0.024, 0.173] | 0.091 [0.024, 0.177] |
| Projection | 19 | 5.56 | 0.028 [-0.006, 0.072] | 0.028 [-0.007, 0.072] |
| Projection | 26 | 5.58 | 0.029 [-0.011, 0.072] | 0.029 [-0.012, 0.071] |
| Ablation | 14 | 26.92 | 0.256 [0.177, 0.356] | 0.256 [0.177, 0.357] |
| Ablation | 19 | 34.91 | -0.011 [-0.109, 0.074] | -0.011 [-0.128, 0.069] |
| Ablation | 26 | 33.22 | 0.045 [-0.020, 0.104] | 0.045 [-0.021, 0.105] |

![Eligible trait analyses](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bc443f056bea94b65b902563cc26a3d6a03d5bd7/figures/issue_2254/revmap_dose_patch/codex_subagent_v1/eligible_trait_analyses.png)

*Figure 1. Panels A and B show high-dose trait-score changes from the same-instrument alpha-zero reference; bars are nominal 95% paired question-bootstrap intervals. Panel C shows eligible sycophancy fractions of the donor-swap ceiling. Evil patch fractions are intentionally absent because their ceiling reference failed the frozen completeness floor.*

The evil patch cells remain eligible only as descriptive raw trait scores. Projection means are 0.02, 0.16, and 0.13 at layers 14, 19, and 26. Ablation means are 30.22, 50.71, and 53.38. These values are not converted into donor-swap fractions and should not be used to rank evil patch strength against the ceiling.

| Operation | Layer | Raw trait mean | Item completeness | Fraction of evil ceiling |
|---|---:|---:|---:|---|
| Projection | 14 | 0.02 | 96.0% | Withheld |
| Projection | 19 | 0.16 | 100.0% | Withheld |
| Projection | 26 | 0.13 | 100.0% | Withheld |
| Ablation | 14 | 30.22 | 95.5% | Withheld |
| Ablation | 19 | 50.71 | 97.0% | Withheld |
| Ablation | 26 | 53.38 | 96.5% | Withheld |

The reference-cell accounting explains the asymmetric eligibility decision. The evil alpha-zero reference passes at 99.0% item completeness, but the evil ceiling fails at 93.5%. Both sycophancy references are complete. Coherence is independently complete for all four references.

| Reference | Trait mean | Trait item completeness | Numeric draw coverage | Trait refusals | Coherence mean | CJK audit |
|---|---:|---:|---:|---:|---:|---:|
| Evil alpha zero | 0.035 | 99.0% | 98.4% | 16 | 99.35 | 15/200 (7.5%) |
| Evil donor-swap ceiling | 56.31 | 93.5% | 93.1% | 69 | 98.56 | 7/200 (3.5%) |
| Sycophancy alpha zero | 4.72 | 100.0% | 100.0% | 0 | 99.18 | 5/200 (2.5%) |
| Sycophancy donor-swap ceiling | 34.58 | 100.0% | 100.0% | 0 | 99.73 | 2/200 (1.0%) |

## Language-neutral coherence report

Coherence is complete for all 16 Round-8 cells when reduced from its own grading partials. Steering is the only intervention family with meaningful degradation under this rubric: the four cell means range from 79.00 to 90.22, and their at-or-above-50 fractions range from 85.0% to 98.0%. Every patch cell has 200 of 200 items at or above 50 and a mean between 98.48 and 99.75.

| Behavior | Intervention | Mean coherence | Items at or above 50 | Valid items |
|---|---|---:|---:|---:|
| Evil | Steering, c=8 | 85.51 | 93.5% | 200/200 |
| Evil | Steering, c=16 | 79.00 | 85.0% | 200/200 |
| Sycophancy | Steering, c=8 | 90.22 | 98.0% | 200/200 |
| Sycophancy | Steering, c=16 | 89.29 | 97.0% | 200/200 |
| Evil | Projection, L14 | 99.52 | 100.0% | 200/200 |
| Evil | Projection, L19 | 99.40 | 100.0% | 200/200 |
| Evil | Projection, L26 | 99.41 | 100.0% | 200/200 |
| Evil | Ablation, L14 | 98.48 | 100.0% | 200/200 |
| Evil | Ablation, L19 | 98.60 | 100.0% | 200/200 |
| Evil | Ablation, L26 | 98.52 | 100.0% | 200/200 |
| Sycophancy | Projection, L14 | 99.34 | 100.0% | 200/200 |
| Sycophancy | Projection, L19 | 99.32 | 100.0% | 200/200 |
| Sycophancy | Projection, L26 | 99.09 | 100.0% | 200/200 |
| Sycophancy | Ablation, L14 | 99.32 | 100.0% | 200/200 |
| Sycophancy | Ablation, L19 | 99.51 | 100.0% | 200/200 |
| Sycophancy | Ablation, L26 | 99.75 | 100.0% | 200/200 |

![Language-neutral coherence](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bc443f056bea94b65b902563cc26a3d6a03d5bd7/figures/issue_2254/revmap_dose_patch/codex_subagent_v1/coherence_language_neutral.png)

*Figure 2. Mean language-neutral form/fluency score and the percentage of items whose five-repeat mean is at least 50. CJK characters are not penalized by this metric.*

## Programmatic CJK report

The CJK audit is a separate deterministic text check, not an LLM judgment and not a component of coherence. It flags at least one CJK-script character in 323 of the 3,200 Round-8 completions, or 10.1%. The concentration is intervention-specific: steering accounts for 232 of 800 completions, or 29.0%, whereas patching accounts for 91 of 2,400, or 3.8%. Evil steering is flagged in 132 of 400 completions and sycophancy steering in 100 of 400. The audit establishes script presence only; it does not determine whether the CJK text is fluent, relevant, or an actual language switch.

| Behavior | Intervention | CJK-flagged completions | Fraction |
|---|---|---:|---:|
| Evil | Steering, c=8 | 67/200 | 33.5% |
| Evil | Steering, c=16 | 65/200 | 32.5% |
| Sycophancy | Steering, c=8 | 54/200 | 27.0% |
| Sycophancy | Steering, c=16 | 46/200 | 23.0% |
| Evil | Projection, L14 | 4/200 | 2.0% |
| Evil | Projection, L19 | 8/200 | 4.0% |
| Evil | Projection, L26 | 9/200 | 4.5% |
| Evil | Ablation, L14 | 10/200 | 5.0% |
| Evil | Ablation, L19 | 14/200 | 7.0% |
| Evil | Ablation, L26 | 10/200 | 5.0% |
| Sycophancy | Projection, L14 | 13/200 | 6.5% |
| Sycophancy | Projection, L19 | 8/200 | 4.0% |
| Sycophancy | Projection, L26 | 1/200 | 0.5% |
| Sycophancy | Ablation, L14 | 5/200 | 2.5% |
| Sycophancy | Ablation, L19 | 5/200 | 2.5% |
| Sycophancy | Ablation, L26 | 4/200 | 2.0% |

![Programmatic CJK audit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bc443f056bea94b65b902563cc26a3d6a03d5bd7/figures/issue_2254/revmap_dose_patch/codex_subagent_v1/cjk_programmatic.png)

*Figure 3. Per-cell CJK-script intrusion within the first 2,048 tokens. Counts are shown out of 200 completions per cell. This programmatic audit is independent of both subagent rubrics.*

## Limitations and disposition

The principal limitation is instrument identity. The grader is `codex-subagent-gpt-5.6-sol-low-v1`, not `claude-sonnet-4-5-20250929`, so these estimates are an exploratory cross-instrument sensitivity analysis rather than a substitute for the planned judge. The five repeats came from fresh sessions, but their statistical independence is unverified. The analysis uses same-instrument alpha-zero and ceiling references but does not import the parent random-control bands, does not apply the parent's selection-aware verdict lattice, and does not correct these per-cell intervals for family-wide multiplicity.

The production run also crossed a Codex CLI patch transition. Of 503 production sessions, 419 sessions producing 33,280 grades launched under CLI 0.153.2, and 84 sessions producing 6,720 grades launched under CLI 0.153.4. All later-version sessions graded coherence. The effect, if any, of this client transition on service behavior cannot be identified separately, so the run is not a uniform-client instrument.

The completeness failure remains binding. Nothing in this report lowers the 95% floor, turns a refusal into zero, or fills a missing trait score. The evil ceiling-normalized patch fields remain null in the structured summary. The raw production job records remain local because the original upload phase correctly refused a completeness-failed pack; only the gate-preserving aggregate summary, figures, and provenance sidecars are published on the issue branch.

## Conclusion

Within this alternative grader, the reverse-map direction produces positive high-dose trait changes for evil and sycophancy, and only the layer-14 sycophancy patch intervals exclude zero; its ablation point estimate is 0.256, compared with 0.091 for projection. Patching remains highly coherent and has low CJK intrusion, whereas high-dose steering retains mostly coherent form but frequently introduces CJK script. These results justify retaining the eligible estimates as a sensitivity analysis. They do not justify an evil ceiling-normalized patch claim, an official Sonnet-grade Round-8 verdict, or a revision of the canonical issue 2254 conclusion.

The authoritative structured values are in the [gate-preserving report summary](https://github.com/superkaiba/explore-persona-space/blob/bc443f056bea94b65b902563cc26a3d6a03d5bd7/eval_results/issue_2254/revmap_dose_patch/exploratory_sensitivity/codex_subagent_v1/report/eligible_report_summary.json). The [grading runner](https://github.com/superkaiba/explore-persona-space/blob/bc443f056bea94b65b902563cc26a3d6a03d5bd7/scripts/issue2254_revmap8_subagent_grade.py), [report generator](https://github.com/superkaiba/explore-persona-space/blob/bc443f056bea94b65b902563cc26a3d6a03d5bd7/scripts/issue2254_revmap8_subagent_report.py), and [figure provenance directory](https://github.com/superkaiba/explore-persona-space/tree/bc443f056bea94b65b902563cc26a3d6a03d5bd7/figures/issue_2254/revmap_dose_patch/codex_subagent_v1) reproduce and audit the report without launching model inference or new judge calls.
