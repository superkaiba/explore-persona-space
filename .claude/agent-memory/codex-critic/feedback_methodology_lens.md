---
name: Methodology Lens Prompt Engineering
description: What makes the methodology-lens prompt work well for Codex on SFT-ablation plans
type: feedback
---

For SFT-ablation plans (loss-masking, partial-turn generation tags, carry-over controls), the methodology lens needs these explicit nudges to fire:

1. **Call out the train/eval context mismatch risk explicitly.** Codex flagged that the new arm trains with gold CoT prefix in context but the eval may not supply it — this is the most important methodological flaw class for "input-side conditioning" experiments. Worth including as a specific sub-question in the lens prompt for any such design.

2. **Ask about gradient flow, not just label masking.** The difference between "labels masked" and "gradients blocked through the prefix" is subtle; asking Codex to distinguish them in the prompt produces the useful "backpropagation still flows through rationale hidden states" finding.

3. **Keep Jinja2 / shell-special characters minimal in the prompt string** — bash heredoc quoting gets messy with `{% generation %}`, `{%- endif %}`, etc. Simplify template pseudocode in the prompt to descriptive prose ("wraps generation tag around Answer: line") rather than literal Jinja2. The dry-run gate and smoke test details can stay.

4. **Explicit "do NOT evaluate statistics" boundary** makes the output stay clean within the methodology lens with no statistical-power spillover.

5. **For round-2 revision prompts, Codex reliably flags "audit-not-control" for any mitigation that measures the mismatch after the fact rather than eliminating it.** On issue #344 round 2, Codex re-raised the train/eval distribution mismatch as a BLOCKER even though the plan added a $5 Claude-judge similarity audit, correctly noting the audit constrains interpretation but does not make the experimental intervention clean. Include an explicit question about whether post-hoc audits substitute for design controls when composing round-2 methodology prompts.

6. **Codex systematically flags loose judge rubrics in Phase 3 mediation analyses.** On issue #344 round 2, Codex identified "persona-voiced" as insufficiently granular and recommended separate rubric fields (voice, answer support, coherence, source-persona cues). Always include a "is the mediation judge rubric specific enough?" sub-question when the plan includes a Claude-judge mediation step.

7. **Single-source C3/fallback gates are a recurring Codex concern.** Codex flagged that using `librarian x 3 seeds` as the sole C3 fallback cannot diagnose under-training at the experiment level. For conditional gate designs, nudge Codex to evaluate whether the fallback scope covers the full hypothesis (all sources) or just a diagnostic proxy.

8. **On round-3 (FINAL) passes, Codex persists as REVISE even when prior BLOCKERs are addressed.** For #344 round 3, Codex issued 4 new BLOCKERs — two were regressions introduced by the fixes (TOST pairing on seed=42 only; C3 gate joint-failure scope), one was a sharpening of the still-open M4 train/eval mismatch (re-categorized as BLOCKER since TOST doesn't close it), and one was a new denominator-family validity concern not present in round 2. Include explicit seed-coverage and denominator-family questions in round-3 prompts for any plan that adds a statistical audit step.

9. **C3 gate joint-vs-single-channel distinction is a recurring Codex concern.** For round 3, Codex flagged that requiring BOTH f_source AND f_bystander CI upper bounds below 0.20 is too narrow — if only one channel is absent, claims for that channel should still be frozen. Structure gate definitions to handle joint, source-only, and bystander-only failure cases.

**Why:** Issue #344 plan critique, rounds 1, 2, and 3. Round-3 Codex verdict: REVISE with 4 new BLOCKERs, 4 SRs, 3 Minor.

**How to apply:** Reuse these nudges verbatim for any future plan where the experiment structure is "masked-loss ablation of an existing factorial arm."

10. **For geometric-leakage / persona-axis designs, explicitly ask Codex about data-adaptive pair selection bias.** On issue #311 round 3, Codex independently flagged that TOP-1 lowest-cosine pair selection is a selection-bias problem at the methodology level (not just statistics): the most extreme geometry case may be selected precisely because it has broad semantic spread, and the null permutation procedure does not account for the selection rule. Ask: "Does the pair selection rule introduce a winner's-curse bias, and is the Stage 8 null procedure selection-aware?"

11. **For scope-reduction simplifications, ask explicitly whether the dropped control was load-bearing.** On issue #311 round 3, Codex flagged that dropping the diverse-source non-axis joint control (scope simplification per user request) leaves no condition isolating axis-specific leakage from generic multi-persona SFT spillover. This is a recurring pattern: scope simplifications that remove a control for budget reasons may quietly remove the experiment's ability to distinguish the mechanism. In the round-3 prompt, include: "Was the dropped control necessary to rule out generic joint-SFT destabilization?"

12. **For collinearity-fallback tests, ask Codex to specify the unit of analysis explicitly.** On issue #311 round 3, Codex flagged that the stratified Mann-Whitney fallback was ambiguous about whether question-level or persona-level observations are pooled -- a methodology flaw independent of statistical power. In any plan with a fallback test on clustered data, include: "What is the unit of observation in the fallback test, and is cluster dependence handled at the aggregation step?"

**Why (items 10-12):** Issue #311 plan critique, round 3. Codex verdict: REVISE with 5 BLOCKERs, 4 SRs, 3 Minor.

**How to apply (items 10-12):** Add these three sub-questions to any round-3 methodology prompt for experiments involving (a) data-adaptive source selection, (b) scope-reduced controls, or (c) collinearity fallback tests on clustered measurement.

13. **For persona-vector / contrastive-extraction designs, ask explicitly about "construct drift" from regenerating canonical prompts.** On issue #368 round 2, Codex flagged that T11 Option A (regenerating all 10 personas via Sonnet from PERSONAS dict seeds) risks measuring Sonnet's paraphrase style rather than the project's original persona definitions. The fix is to check whether Sonnet paraphrase drift changes the behavioral scope, then run the original prompts as a sensitivity arm.

14. **For projdiff axes, ask whether the "neutral reference" is computed via the same test-prompt procedure as the other activations.** On issue #368 round 2, Codex flagged that pvec_chenstyle_L20_projdiff uses helpful_test_act from negative-side EVAL_QUESTIONS responses, while Phase 1 test activations come from non-persona panel prompts. The reference and comparison activations are not from the same distribution. In any plan using projection-difference, include: "Is the neutral reference extraction procedure identical to the target activation extraction procedure?"

15. **For partial-Spearman aggregation over multiple strata, ask Codex whether degenerate strata are handled.** On issue #368 round 2, Codex flagged that villain (9/10 leakage=0) and comedian (10/10=0) make within-source rho undefined for those strata. Averaging rhos over strata with undefined outcomes creates an operationally invalid gate. In any plan using stratified/partial correlation gates, include: "What is the fallback for strata with constant outcomes?"

16. **For permutation nulls on recipe-agreement matrices, clarify whether the null preserves the joint representation.** On issue #368 round 2, Codex flagged that shuffling marker_rate tests leakage association, NOT whether recipes agree with each other independent of leakage. All 8 recipe axes may still agree under marker shuffling because they are transforms of the same hidden state. In any H3-style recipe-agreement test, include: "Does the permutation null preserve the inter-recipe correlation structure, or does it only test leakage association?"

**Why (items 13-16):** Issue #368 plan critique, round 2. Codex verdict: REVISE with 9 BLOCKERs (items 1-9 in report).

**How to apply (items 13-16):** Add these sub-questions to any round-2 or round-3 methodology prompt for experiments involving (a) LLM-regenerated positive sets, (b) projection-difference axes, (c) stratified-correlation gates with potentially constant-outcome strata, or (d) permutation nulls on multi-axis agreement matrices.

17. **For AMENDMENT plans (single-variable diff vs a completed parent), add an explicit "amendment context" block** telling Codex (a) which decisions a binding user scope marker already resolved (do not re-litigate; review only implementation fidelity), and (b) which calls were delegated to the planner (fully reviewable). Without it Codex re-litigates user-resolved quota/exemption decisions and wastes its word budget.

18. **When a defect fix changes a selection rule on only ONE side of a paired v1-vs-v2 comparison** (e.g. corrected dose normalization applied to v2 checkpoint selection while v1 cells keep their old-rule selections), ask explicitly: "Is the matched-dose construct still the same on both sides when the two sides' selection rules differ? Does covariate + sensitivity-cut mitigation suffice, or must the parent's checkpoints be re-selected under the corrected rule (often free from archived per-checkpoint reads)?" This asymmetry class is easy for Codex to miss without the nudge.

19. **For on-policy elicitation designs, ask about judge-filter/eval-judge circularity with the cross-version twist:** if only ONE side's training data was filter-selected by the eval-judge family, the filter can shift that side's column readings via judge-style matching rather than behavior change — a paired-comparison confound distinct from ordinary same-rubric coupling.

**Why (items 17-19):** Task #545 amendment plan (onpolicy-testbed-v2) round-1 prompt composition, 2026-06-12.
