# Interpretation revise round 1 — union of interpretation-critic (Claude) + Codex twin blockers

Both critics returned **REVISE**; both independently reproduced every headline number as correct (delta_int CIs, ctx/judge/answer AUROCs, map R², transfer, the 12-cell battery, CJK-intrusion rates, raw-completion label counts). **Nothing overturns the headline or the MODERATE tag.** The REVISE is calibration / scoping / labeling honesty + reporting completeness. Apply ALL items; ground each against the artifacts (paths below). Do NOT change any correct number.

Artifacts: body `tasks/interpreting/2356/body.md`; results `eval_results/issue_2356/results/*.json` (stats.json, transfer.json, predictor_scores_arm{A,B}.json, map_diagnostics.json, map_discrimination.json); zero-shot `eval_results/issue_2356/judge_zeroshot/predictor/predictor_scores.json`; figures `figures/issue_2356/`.

Verified values you will cite (from stats.json this turn): armA ctx_ridge 0.9947 / ans_greedy 0.9882 / ans_rollout 0.9986 / judge_fewshot 0.8956 / text_surface 0.9514 / is_rewrite 0.1825 (=0.817 for compliance); armB ctx_ridge 0.9507 / ans_greedy 0.9495 / ans_rollout 0.9822 / judge_fewshot 0.7426 / text_surface 0.6960; armB lodo pooled AUROC 0.936; recovery_fraction `undefined_denominator_le_0.02` both arms (map≈ctx≈answer near-tie — the denominator is ~0).

## Prose / claim edits (fold into Takeaways + the relevant Results H3s)

1. **Deterministic → tendency language.** Replace "the decision is already fully linearly present", "the decision precedes generation", "determined", "fully linearly present" (body ~L29,36,47,116) with prompt-level refusal-TENDENCY language: the prompt activation strongly predicts the prompt's thresholded refusal tendency (over 10 stochastic rollouts) before generation — NOT the decision of any individual future generation.

2. **Answer-"ceiling" reframing (Result 2 + Figure 2 caption).** A zero-spanning answer-minus-context interval shows *no detected difference*, NOT equality/ceiling. Reframe: the context probe is statistically indistinguishable from the **greedy-answer** probe; AND disclose the **rollout-mean answer probe (`ans_rollout`)** is descriptively HIGHER (armA 0.9986 vs ctx 0.9947; armB 0.9822 vs ctx 0.9507, higher in all 5 armB folds) — it was a planned companion and is currently absent from the ceiling interpretation. Note the rollout-mean probe averages activations from the same sampled responses used to build the label (shared-provenance caveat) — distinguish it from the independent greedy-answer read.

3. **Surface-matched premise qualification (armA / Result 1).** armA is described as surface-matched "where only the model's decision differs", but `is_rewrite` AUROC 0.1825 (=0.817 for predicting compliance) means rewrite status differs systematically with the label, and the fitted text-surface classifier is already strong (0.9514; 0.945 without rewrite/axis indicators). Qualify: the context probe still has a positive paired edge over text-surface, but learnable surface structure explains much of armA's absolute performance.

4. **Scope "better than the judge".** The probe has internal activation access the judge lacks and is supervised on hundreds of labels vs the judge's 32 demonstrations; armA's flip-pair design structurally caps the judge's AUROC. State this is not a general probe-vs-judge claim.

5. **Judge graded-rate Spearman (armA).** Where the body cites the probe's rate-Spearman as ranking validation, add a clause: the judge's graded-rate Spearman is comparable-to-higher than the probe's in armA on a DIFFERENT row population, and the rate-Spearman does NOT reproduce the AUROC ordering in armA. (Pull the exact ρ from stats.json `arms.*.spearman_rate` — judge_fewshot vs ctx_dim/ctx_ridge.)

6. **armB re-issue confound — surface it AND its counter-evidence (Result 1/armB).** armB's judge baseline had ~61/286 (21%) items API-self-censored and re-issued (currently footer-only) — surface it at the armB result. BUT also state Codex's counter-evidence: fold-0 (recovered synchronously) has the BEST judge AUROC, so re-issue does not appear to explain the headline gap; and conservatively excluding affected rows leaves the gap positive.

7. **Transfer conclusion weakening (Result 4 + Takeaways).** Replace "one largely shared geometry, NOT two regime-specific signals" with "consistent with a substantial shared component" — high bidirectional transfer supports a shared decodable component but does not rule out additional regime-specific structure or a generic refusal/severity direction. Note no transferred text-only control was run to fully distinguish this.

8. **Prior-work attribution fix (Methodology ~L46).** "Prior work localizes refusal in the last-instruction-token activation" conflicts with the approved plan §2 correction: Arditi et al. selected among POST-instruction positions; last-prompt-token is THIS project's convention. Correct the attribution and cite precisely.

9. **Generic-map rank footer fix.** The footer claims generic-map selected ranks are "near-full"; actual selected ranks range 32→full across folds (`predictor_scores_arm{A,B}.json → .selection.*.map3a_zr.rank` / `map3b_zr`). Correct.

## Reporting-completeness edits (planned-vs-actual; every planned predictor/metric gets a value or explicit "not run")

10. **Zero-shot judge — REPORT IT (it WAS run).** `judge_zeroshot/predictor/predictor_scores.json` exists with all 10 folds; compute its per-arm AUROC and report it alongside judge_fewshot. Do NOT mark it untested (both critics wrongly assumed it was absent because it was never folded into stats.json). If you cannot fold it into stats.json, report the value computed directly from the zero-shot scores file with that provenance noted.

11. **Report omitted controls/companions:** armA text-surface with vs without indicators (`text_surface` 0.9514 vs `text_surface_noind`), the bare rewrite indicator (`is_rewrite`), the rollout-mean answer probe (`ans_rollout`, both arms), and **balanced accuracy** context/judge (Codex cited armA 0.973/0.819, armB 0.895/0.685 — recompute from the scores at the operating threshold, or report "not persisted" if the artifact lacks it). Add each as a Results value or an explicit "not run/not persisted" line.

12. **Sample block expansion + annotation fix (Methodology sample block ~L74,81-82).** The "Four rows" block gives only ~1 example per class per regime; the spec wants ≥3 firing + ≥3 non-firing sanitized examples — expand (sanitized, harmful truncation-labeled, permanent raw links kept). FIX the two wrong armA annotations: the armA comply row is judge-scored 0 (not 4); the armA refuse row has NO few-shot score (it is outside the balanced predictor set) — remove/correct those annotations. Join each displayed hash to labels + judge scores before annotating.

13. **CJK sensitivity reframing (~L72).** Replace the aggregate "no label could flip" assurance with the row-level sensitivity result: intrusion is rare overall but concentrated in some prompts, including ~5 balanced rows exactly at a label threshold; conservatively excluding every affected balanced prompt leaves BOTH headline gaps positive → headline unchanged. (Recount per-prompt; the aggregate rates armA 0.07%/armB 0.16% stay as reported.)

## Figure regeneration (mechanical — reject opaque labels)

14. Regenerate all 4 figures with reader-facing labels; find the plot code (grep `figures/issue_2356` producers — likely a `scripts/issue2356_*fig*.py` or a plot phase in `scripts/issue2356_fits.py`), edit the tick/legend/label mappings, re-render, and **Read each rendered PNG** to confirm non-empty axes + reader-facing labels (inline figure-sanity duty). REJECT any chart label containing: `ctx`, `delta_int`, `map3a`, `map3b`, `3a`, `3b f*`, `A->B`, `B->A`, `dim`, `r2 cand-norm`, and the "PCA-ctx control" abbreviation. Suggested plain names (confirm semantics from code/plan first): context probe / answer probe (greedy) / answer probe (rollout-mean) / LLM judge (few-shot) / PCA context control / fitted text-surface / context→answer map (label-blind) / context→answer map (generic-corpus) / harmful→over-refusal / over-refusal→harmful / difference-in-means direction. Qualify Figure 2's caption (no "ceiling"/"equals" — use "indistinguishable from the greedy-answer probe").
- Figures: `hero_auroc_by_predictor.png`, `contrast_paired_ci.png`, `battery_acc1_greedy.png`, `transfer_2x2.png` — plus `captions.json` / `meta.json` updated to match.

## After applying
- Keep the Lens-14 "scope & robustness caveats" acknowledgment (all 13 concern_ids) intact — do NOT drop it.
- Re-run `verify_task_body.py --issue 2356` + `audit_clean_results_body_discipline.py --issue 2356` → both PASS (watch check 20 conciseness — the additions must stay within caps; tighten prose).
- Commit the regenerated figures + captions by explicit path; `set-body` the updated body (NO `--snapshot`).
- Post `epm:interpretation v2` summarizing what changed.
