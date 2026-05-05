---
name: Common Review Patterns
description: Recurring issues found across 27 experiment reviews — use as checklist for future reviews
type: feedback
---

Patterns that recur across reviews in this project. Check each when reviewing a new draft.

1. **Prompt length confound.** System prompt length correlates with leakage/transfer effects (r=-0.74). Any experiment varying personas must control for prompt length or flag it. (Found in: proximity transfer, trait transfer)

2. **Tautological inclusion.** Including the target entity in its own correlation inflates r. E.g., including assistant in cos(assistant) correlation. Always check if any data point is tautological. (Found in: proximity transfer)

3. **Domain-mismatch masking real effects.** Off-domain eval can dramatically overestimate defense effectiveness. Always ask: was the eval domain-matched to training? The truthification 97% figure applies only off-domain; in-domain is 68-74%. (Found in: domain-matched eval)

4. **Coherence collapse vs genuine misalignment.** Low alignment scores may reflect incoherent outputs, not misaligned values. Apply Betley coherence filter (exclude coherence < 50) before interpreting alignment. (Found in: EM-defense midtrain matrix, Tulu DPO)

5. **Single-seed uncertainty.** Cross-seed variance for alignment is +/-5-8 points. Effects smaller than this are uninterpretable at n=1. Flag any single-seed finding as preliminary. (Found in: nearly every experiment)

6. **Bonferroni correction.** With 5+ comparisons, uncorrected p-values are misleading. Always report both uncorrected and Bonferroni-corrected significance. (Found in: truthification ablation, axis category)

7. **Framing effect on controls.** Training framing (e.g., medical context) degrades ANY model's alignment, not just trained ones. Always include a control model evaluated with the same framing. (Found in: framed eval, domain-matched eval)

8. **Cross-experiment comparisons must control for setup.** Comparing preservation percentages across experiments that differ in base model, EM data, EM strength, and eval method is misleading. Always list the axes of variation before making cross-experiment claims. (Found in: Tulu DPO vs truthification comparison)

9. **Inconsistent averaging (sample-weighted vs per-question).** When questions have unequal sample sizes (due to parse errors), the overall mean (sample-weighted) differs from the mean of per-question means. T-tests should use per-question means (paired by question), but the results table may use sample-weighted means. Flag when both appear in the same draft without explanation. (Found in: Tulu DPO draft)

10. **Model path provenance.** Always check the `model_path` field in eval JSONs to confirm the evaluated model matches the claimed condition. Naming inconsistencies (e.g., "tulu_dpo_merged" inside a "tulu_control" directory) can indicate wrong baseline. (Found in: Tulu DPO pre-EM baseline)

11. **LoRA parameter % miscalculation.** Verify LoRA trainable parameter counts independently. For Qwen2.5-7B with r=32 targeting 7 modules, actual is ~1.06%, not 2.2% as claimed in one draft. The error propagated into RESULTS.md. Always compute: per_module = in_dim*r + r*out_dim, sum over modules and layers, divide by total model params. (Found in: truthification 32B)

12. **Ceiling/compression effects in LLM judges.** When control alignment scores are high (>90), the effective discrimination range of the judge may compress differences. A 4.4-point drop from 91.7 may be harder to detect than a 57-point drop from 85.2. Always check if the baseline saturates the scoring rubric. (Found in: truthification 32B)

13. **Random direction control for axis claims.** Any projection claim about the assistant axis must be tested against random directions. The v2 analysis showed z=-0.45 for corpus separation, meaning the axis is not special for between-corpus differences. New axis analyses that skip this control will inevitably overclaim axis specificity. (Found in: FineWeb raw projection, axis tail deep analysis, axis category projection)

14. **Conflating scale and instruction tuning.** When a small base model fails an eval designed for instruct models, the failure could be scale, instruction-tuning, or both. Claiming "minimum viable scale is X" when only base models were tested at scale X confounds the two. (Found in: MeCo URL EM)

15. **RESULTS.md vs raw JSON divergence.** The RESULTS.md main matrix table contains numbers from an earlier run that differ from the raw JSON in eval_results/. The OOD section of the same document uses the correct (JSON-matching) numbers. Always verify RESULTS.md claims against run_result.json. The most critical discrepancy: good+wrong SFT Post-Cap claimed 0.840 but raw JSON shows 0.692. (Found in: EM-defense midtrain matrix)

16. **Missing raw data for claimed results.** 15 of 18 midtrain matrix conditions and 7 of 14 CPT conditions have no raw JSON in eval_results/. Numbers exist only in RESULTS.md and hardcoded plotting scripts. Without provenance chain from experiment to JSON to summary, results are unverifiable. (Found in: midtrain matrix, CPT sweep)

17. **Pre-intervention degradation confound.** Midtraining interventions (CPT, SFT coupling) can degrade pre-EM capability. "Protection" measured as post-EM absolute score conflates genuine EM resistance with pre-EM degradation. Report net effect (protection minus pre-EM cost) alongside raw post-EM scores. (Found in: CPT volume sweep)

18. **Stale experiment tracking.** EXPERIMENT_QUEUE.md and RESULTS.md "Remaining" lists can become stale when results are saved to eval_results/ but the docs are not updated. Always cross-check what directories exist vs what the tracking docs claim. (Found in: CPT sweep "remaining" list)

19. **Negative persona misclassification in contrastive experiments.** The evaluation code may fail to map training negative persona strings (e.g., "You are a marine biologist") to eval persona names (e.g., "16_marine_biologist"), causing truly negative personas to be counted as "unseen" in aggregates. Always verify: which eval personas are actually training negatives? Do the JSON aggregates correctly classify them? (Found in: contrastive leakage, behavior-type leakage)

20. **Unseen mean vs all-non-target mean column mislabeling.** Drafts may label `all_non_target_mean` (includes both negative training personas and unseen) as "Unseen mean" in tables. The actual unseen-only mean is always higher because negative personas are suppressed toward zero and dilute the combined mean. Check which JSON field the table actually pulls from. (Found in: behavior-type leakage)

21. **Omitted p-values for correlation claims.** Report both Pearson and Spearman with p-values when claiming a correlation. Pearson may be non-significant (p=0.055) while Spearman is highly significant (p<0.001), indicating a real but nonlinear relationship. Omitting p-values hides marginal non-significance. (Found in: contrastive leakage)

22. **Overclaiming from underpowered marker detection.** With n=10 completions, a 20% marker rate (2/10) is not statistically different from 0% (Fisher exact p=0.47). Drafts should not draw conclusions about mechanism from such small samples. (Found in: activation steering test)

23. **Omitted data points that weaken claims.** Tables that show a subset of results (e.g., 7 of 11 personas) can make "no effect" claims appear stronger. Always check if the full dataset tells a different story. In the directed trait transfer case, non-target marker rates showed a clear gradient that only appeared when all 11 personas were included. (Found in: directed trait transfer Arm A)

24. **Cross-condition tests for non-target personas.** When an experiment claims specificity to one persona, always test the same metric on OTHER personas. The directed trait transfer showed pirate_near degraded assistant by 10.5 points despite pushing pirate (not assistant) toward scholar, undermining the specificity claim. (Found in: directed trait transfer Arm B)

25. **Cross-experiment comparison with different persona sets.** When comparing two experiments that used different persona populations, note this explicitly. Non-target degradation means are not comparable if the non-target personas differ. (Found in: contrastive EM vs whole-model EM)

26. **Wrong cross-experiment values in comparison tables.** The analyzer may assume both experiments measured the same things. Always check BOTH sides of a comparison table against raw data. In the contrastive EM draft, the scholar alignment row claimed ~20-27 for both experiments, but the whole-model EM scholar was actually 78-85 (not specifically targeted). (Found in: contrastive EM)

27. **Hidden bystander persona inclusions.** When computing group means (negative-set vs bystander), check which personas are in each group and whether the primary outcome persona (e.g., assistant) is silently included. If it is, the analysis conflates the main finding with the secondary analysis. (Found in: contrastive EM negative-set vs bystander)

**How to apply:** Use as a mental checklist when reviewing new drafts. If a draft doesn't address these where relevant, flag it.
