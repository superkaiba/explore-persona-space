# Interpretation revise round 2 (→ epm:interpretation v3) — union of round-2 critic blockers

Both round-2 critics returned REVISE (near-PASS). Both CONFIRMED: every number correct, all 4 figures clean (reader-facing labels, hash-matched pin 93b8debeb6), the zero-shot join reproduced exactly (armA 0.9198 n=526 / armB 0.7623 n=286), the 12-row sample block joins correctly, CJK row-level sensitivity correct, armB re-issue handled. **No result changes, no headline change, MODERATE stands.** All residuals are wording-precision + one completeness table. Apply ALL; keep the Lens-14 scope-caveats acknowledgment (13 concern_ids) intact.

Body: `tasks/interpreting/2356/body.md`. Artifacts: `eval_results/issue_2356/results/*.json`, `eval_results/issue_2356/judge_zeroshot/predictor/predictor_scores.json`, `eval_results/issue_2356/results/map_discrimination.json`, `eval_results/issue_2356/arm{A,B}/labels.json`. Grep the exact strings; replace precisely; do NOT touch the verbatim frontmatter `goal:`.

1. **Number fix (Result 1, ~L114).** "the paired gap 0.100 and 0.205 sits wholly above zero" → "the paired gap 0.099 and 0.208 sits wholly above zero" (matches the sentence's own AUROCs: 0.995−0.896=0.099, 0.951−0.743=0.208; and `stats.json arms.armB.contrasts.delta_int.point`=0.2081, armA 0.0991).

2. **Deterministic → tendency (Codex #1).** Grep non-frontmatter prose for `substantially fixed` and `decision actually occurs` and the construct-level "decision":
   - Goal Broader-narrative (~L47): "the model's refuse/comply tendency is substantially fixed by the context geometry ahead of generation" → "…is strongly predictable from the context geometry ahead of generation".
   - Evaluation (~L70): "construct = the model's own refuse-vs-comply/answer decision … the decision actually occurs" → describe the construct as the model's thresholded prompt-level refusal tendency (over ≥7 valid temp-0.9 draws); keep the on-policy/on-distribution framing.

3. **Answer-"ceiling" removal outside frontmatter (Codex #2 + Claude C2).** Grep non-frontmatter prose for `ceiling` and the Result-2 heading:
   - Result-2 H3 heading (~L118): "indistinguishable from the answer probe" → "indistinguishable from the greedy-answer probe".
   - Footer caveat (~L154): "the actual-answer probe is a within-model ceiling, not an external gold standard" → drop "ceiling"; e.g. "the actual-answer probe is a within-model reference, not an external gold standard".
   - Goal Broader-narrative (~L47): "at a fidelity statistically indistinguishable from the actual-answer probe" — reword to name the greedy-answer probe and avoid "ceiling" fidelity framing.
   - ADD (Result 2, ~L126): the rollout-mean answer probe exceeded the context probe in ALL FIVE over-refusal folds (per-fold AUROC Δ +0.058, +0.001, +0.015, +0.034, +0.026 — the pooled ordering is not one-fold-driven; from `predictor_scores_armB.json`).

4. **Methodology surface-matched claim (Codex #3, ~L51).** "giving surface-matched pairs where only the model's decision differs" → acknowledge rewrite status carries label signal (the bare rewrite indicator predicts compliance at AUROC 0.817; a fitted text-surface classifier reaches 0.951, 0.945 without rewrite/axis indicators), so pairs are surface-matched but rewrite status is not label-orthogonal; the context probe still has a positive paired edge over text-surface.

5. **Spearman population label (Codex #4, ~L70).** "over all 2748/2510 labeled prompts" → "over all rate-scored prompts (2748/2510, including 118 middle-band rows per arm without a binary label)". (labels.json: 2630/2392 binary-labeled + 118 middle-band per arm.)

6. **Zero-shot provenance + outperformed few-shot (Codex #5).** Where the body reports "A zero-shot judge scores 0.920 / 0.762": (a) state it is a direct join from `eval_results/issue_2356/judge_zeroshot/predictor/predictor_scores.json` (not in stats.json); (b) state explicitly that zero-shot OUTPERFORMED the preregistered-primary few-shot judge in BOTH arms (0.920 > 0.896 harmful, 0.762 > 0.743 over-refusal) — the 32 demonstrations did not raise AUROC — while both judge variants remain below the probe.

7. **Map conclusion: absence → non-detection (Codex #7).** Grep for "not extra decision signal" / "adding answer identity, not decision signal" (Takeaways ~L38 + Result 3 ~L138): reword so the conclusion from zero-spanning contrasts reads "no DETECTED extra decision-predictive performance" (non-detection), not proven absence.

8. **Planned-vs-actual companion status table (Codex #6).** After the "K = 4 draw-averaged targets" mention (Methodology), add a COMPACT status line/table enumerating the registered map/battery companions vs their reported status: draw-averaged retrieval (K=4), behavior-split retrieval, NN behavior match, limited-label ladder, recovery fraction. Enumerate the persisted keys in `stats.json` + `map_discrimination.json`; report each companion's value or an explicit status. NOTE: `recovery_fraction` is `undefined_denominator_le_0.02` in BOTH arms (definition (AUROC[map]−AUROC[ctx])/(AUROC[ans]−AUROC[ctx]); the map≈ctx≈answer near-tie makes the denominator ≤0.02 → undefined) — report it as undefined-by-near-tie, not a missing run. Keep this TIGHT (a few lines / a small table) to stay within check-20 caps.

## Finish
- `verify_task_body.py --issue 2356` + `audit_clean_results_body_discipline.py --issue 2356` → BOTH PASS (watch check 20 conciseness; tighten prose to fit — the table + additions must not blow the caps). Lens 14 must still show all 13 acknowledged.
- No figure re-render needed (round-2 confirmed all 4 clean) unless a caption word changes — if so, re-commit only the changed caption/meta by explicit path.
- Land body via `task.py set-body 2356 --file <path>` (NO `--snapshot`). Post `epm:interpretation v3` summarizing the applied items.
- Do NOT change any correct number or the headline; MODERATE stands.
