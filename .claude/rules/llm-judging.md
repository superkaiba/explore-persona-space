# LLM judging of behavior-expression DVs

**Load this rule whenever a plan or code DESIGNS or WRITES an LLM-judged
behavior-expression dependent variable** (sycophancy, refusal, hedging,
style, trait, alignment/EM verdict — any DV where an LLM scores whether a
model output expresses a target behavior). The always-on backstop is the
CLAUDE.md "Measurement validity" bullet + the "LLM judge =
`claude-sonnet-4-5-20250929`" bullet; this file is the full 20-guideline
recipe those bullets point at (the 18 judge-scoring guidelines plus the
two-rule § E2 continuous-companion-DV recipe), in the on-demand register.

Cross-links: `.claude/rules/persona-vectors-recipe.md` (the graded-judge
precedent — its rubric is a 0–100 trait score), CLAUDE.md § Measurement
validity (the dual-DV clause + the graded-primary-for-ranking-targets
extension), and `.claude/rules/marker-leakage-measurement.md`. **The
programmatic marker DV is NOT a judged-behavior DV** — it keeps its own
three-space `log P` recipe and is governed by `marker-leakage-measurement.md`,
not this rule. The guidelines below carry literature citations (arXiv ids +
incidents); those ARE the substance and stay (they are paper citations, not
person/meeting attributions).

The chat investigation behind this rule (2026-06-30) found the project's
behavior-expression judging is binary (fraction-positive rate), which
attenuates the predictor correlations the #658/#742/#761/#763 line depends on
and diverges from the field standard (Persona Vectors uses graded 0–100).

## A. Scoring mode & scale

1. **Graded pointwise 0–100 is the PRIMARY scoring mode for a
   ranking/regression/predictor-target DV.** Dichotomizing a graded behavior
   into a binary pass/fail attenuates a correlation by ~0.798 (≈36%
   effective-N loss; Cohen 1983 "The cost of dichotomization"; MacCallum et
   al. 2002, *Psychological Methods*) — worse near a floor/ceiling, exactly
   where a predictor target needs dynamic range. A coarse integer scale (1–5)
   collapses most of the graded signal too; prefer 0–100.

2. **The binary positive RATE is RETAINED as the validated human-legible
   headline companion (dual-DV roles).** A binary agreement/expression rate
   is what a mentor reads and what the on-policy behavioral construct is
   validated as; keep reporting it. Graded-primary applies to the
   ranking/regression role (CLAUDE.md § Measurement validity) — it EXTENDS,
   never replaces, the binary rate.

3. **Pointwise, not pairwise, for ABSOLUTE measurement.** Pairwise comparison
   is for preference ranking, not for an absolute behavior score, and it is
   more position-biased: a distractor flips a pairwise verdict ~35% of the
   time vs ~9% pointwise (arXiv 2504.14716). Use pointwise scoring for a DV
   that must measure how much a single output expresses a behavior.

## B. Variance reduction & aggregation

4. **Multi-sample N judge draws at temperature > 0, mean-aggregated.** This
   is the implementable substitute for the paper's logit-weighted scoring —
   the Anthropic Messages API exposes no score-token logprobs, so average N
   sampled integer scores instead (G-Eval 2303.16634 logit-weights; we
   approximate by sampling). N and temperature are NOT literature-pinned →
   pilot N≈5–10 by measured test-retest reliability (§F rule 15) and record
   the chosen N + temperature per DV.

5. **Graded multi-sampling partly substitutes for more binary samples.** The
   binary penalty is mostly dichotomization loss, not binomial sampling
   noise, so a few graded draws recover more signal than many binary draws
   would; do not over-budget judge calls chasing binomial precision a graded
   score already captures.

## C. Prompt / rubric design

6. **Anchored rubric — define what 0 / 50 / 100 mean.** An unanchored 0–100
   ask drifts; Prometheus (2310.08491, ICLR 2024) measured judge–human r
   dropping 0.457 → 0.355 without rubric anchors. Spell out the endpoints and
   the midpoint in the judge prompt.

7. **Reason-then-score, not score-then-reason.** Eliciting the rationale
   before the number lifts Prometheus judge–human r 0.673 → 0.847. Ask for a
   brief justification, then the integer.

8. **One behavior per judge call.** Bundling multiple behaviors into one
   rubric confounds them; score each DV in its own call.

9. **DROP a malformed / `REFUSAL` / out-of-range judge return — NEVER coerce
   it** (from BOTH arms of any contrastive comparison). A judge return that is
   `REFUSAL`, non-numeric, or outside [0, 100] carries no information about
   the behavior and must not enter the pool; coercing it to a number (e.g.
   `→ 0` or `→ 50`) biases the mean toward whichever arm the refusals land in.
   This generalizes the persona-vectors judge-filter rule
   (`.claude/rules/persona-vectors-recipe.md` step 4) to every judged DV, and
   the per-arm dropped count is REPORTED. (`eval/belief.py` previously
   default-coerced a malformed return to 50 — FIXED in #766: parse
   failures / out-of-range / API errors now drop to `math.nan`, excluded
   from the aggregate by the caller's `not math.isnan` filter.)

10. **Pin nuisance formatting identical across conditions.** Response length,
    markdown, system-prompt boilerplate, and the presence/absence of a
    reference answer all move a judge score independently of the behavior
    (arXiv 2506.22316 measured a reference-answer nuisance effect ~0.2 on
    Qwen-class outputs). Hold every nuisance variable fixed across the
    conditions being compared.

## D. Judge model

11. **ONE cross-family judge: `claude-sonnet-4-5-20250929` judging Qwen.**
    Cross-family judging IS the self-preference bias mitigation
    (Play-Favorites 2508.06709; CALM 2410.02736 — a judge favors its own
    family's outputs). Never run a Qwen judge on Qwen outputs. The project
    pins one consistent judge across all behaviors (CLAUDE.md "LLM judge =
    `claude-sonnet-4-5-20250929`").

12. **Enforce the one-Sonnet-judge pin mechanically** via
    `scripts/workflow_lint.py --check-judge-model-pins` (bundled into the
    no-flags default run). The motivating incident is the #650/#657 stale
    legacy-Haiku judge pins that re-pinned a non-Sonnet judge for new work; the
    lint flags a hardcoded non-Sonnet judge-model pin at a judge call site.

## E. Closed-loop caveat

13. **A judge score used as a regression/predictor TARGET carries a
    self-reinforcement-toward-LLM-text risk.** G-Eval (2303.16634) found
    GPT-4 rating LLM outputs ABOVE human ones; using a judge score as the
    target a model is optimized/selected against can amplify whatever the
    judge over-rewards. Cross-family judging (rule 11) mitigates but does not
    eliminate this → validate the graded DV against an INDEPENDENT non-judge
    reference (a small human audit set, or the project's log-P / activation
    companion DV) before it carries a ranking/regression headline.

## E2. The non-saturating continuous companion DV (b)

A judged behavior-expression RATE saturates at floor/ceiling and censors install /
dose-matched / cross-condition comparisons; the dual-DV rule (CLAUDE.md
§ Measurement validity) pairs it with a SECONDARY non-saturating continuous
completion-probability DV. Two forms, in PREFERRED order:

19. **PREFERRED — teacher-forced fixed positive-vs-negative completion margin.**
    `margin(C) = mean LN-logP(FIXED positive-answer pool | C) − mean LN-logP(FIXED
    negative-answer pool | C)`, the answer pools judge-filtered ONCE and held FIXED
    across every context C, scored teacher-forced (length-normalized log-P of each
    fixed answer conditioned on C). Because the same answer set is scored under every
    context, there is NO selection-on-outcome bias. #722 validated it: ρ(margin, rate)
    was all-positive (refusal +0.56 [+0.14, +0.80], sycophancy +0.40 [−0.03, +0.68],
    broad_em +0.31 [−0.20, +0.67]) and the margin keeps spread where the rate is
    floored (broad_em margin std 0.31 vs rate std ~0.008). Caveat: a behavior with no
    fixed +/- pool is untestable here (#722: harmful_compliance, no #661 pool). This
    is a SECONDARY companion validated to track the rate — NOT a primary behavioral
    leaderboard read (the #432→#456 teacher-forcing caution still bars that use;
    `.claude/rules/marker-leakage-measurement.md` § #432 → #456).

20. **OPT-IN alternative — judged-positive-conditional-mean log-P (`logp_pos_mean`),
    selection-on-outcome-confounded.** Length-normalized trained−base log-P averaged
    over ONLY the judged-positive completions in each context. This is a conditional
    mean over an OUTCOME-SELECTED subset: contexts with more positives tend to have a
    lower mean log-P among them, so the DV anti-correlates with the rate. #722
    measured ρ(DV, rate) ≈ −0.3 for 3 of 4 behaviors — it FAILED the dual-DV
    validation gate. Use it ONLY after it passes the standing ρ(DV, rate) > 0
    validation (Spearman across cells with dynamic range) for the behavior at hand;
    prefer the fixed +/- margin (rule 19) by default.

Either form is SECONDARY to the PRIMARY on-policy judged rate and is subject to the
standing ρ(DV, rate) > 0 validation before it carries any cross-condition read; never
narrate it as the construct. (Source: #722 — `eval_results/issue_722/tf_margin/`.)

## F. Validation as a measurement instrument

14. **Validate reliability PER behavior class, not once.** Judge reliability
    is behavior-specific — CALM (2410.02736) found alignment/subjective
    behaviors markedly more bias-prone than factual ones. A reliability number
    established on one behavior does NOT transfer to another DV.

15. **Measure judge–human agreement on a small per-behavior audit set before
    trusting a DV.** Report Spearman/Pearson + Krippendorff's alpha against
    human labels; ~85% judge–human agreement is the rough MT-Bench benchmark
    (2306.05685). The agreement threshold + audit-set size are NOT
    literature-pinned → pilot. Carry the reliability ceiling √(r_yy) (rule 18)
    when interpreting an observed correlation.

16. **Re-confirm cross-family self-bias is negligible for
    `claude-sonnet-4-5` specifically** — this exact judge↔target pairing is
    untested in the literature, so check it on a small audit set rather than
    assuming the general cross-family result transfers.

17. **Keep the existing saturation / gaming detection.** The marker + dual-DV
    saturation rules still apply (`.claude/rules/marker-leakage-measurement.md`,
    CLAUDE.md § Measurement validity). A graded score additionally surfaces
    compression a binary rate hides — a rate pinned at 1.0 across conditions
    can still show graded spread, which is the saturation signature to watch.

## G. Reproducibility / reporting

18. **Pin & report, per DV:** scoring mode, scale, N samples + temperature,
    judge model + date, prompt hash, per-behavior reliability (test-retest +
    judge–human agreement), and the reliability ceiling √(r_yy). A judged DV
    is a measurement instrument; report it like one.

## Do NOT over-rely on (adversarially refuted)

- "Pointwise is universally more robust than pairwise" — pointwise wins for
  ABSOLUTE measurement (rule 3), not universally.
- "A reference answer is the single biggest accuracy driver" — it is a
  nuisance variable that must be held fixed (rule 10), not a free accuracy
  boost.
- "GPT-4 ≈ human–human agreement is a safe ceiling" — it is not; judges rate
  LLM text above human text (rule 13) and reliability is behavior-specific
  (rule 14).
- "It's familiarity, not self-recognition" as a reason to ignore self-bias —
  cross-family judging is still required (rule 11) regardless of the mechanism.
- "A bigger judge ⇒ less bias" — size does not remove self-preference or
  behavior-specific unreliability; validate per behavior (rules 14–16).

## Enforcement

- `planner.md` §6 + `critic.md` Statistics & Measurement lens — the
  measurement-validity / dual-DV requirements a judged DV plan must meet.
- `scripts/workflow_lint.py --check-judge-model-pins` — the mechanical
  one-Sonnet-judge pin gate (rule 12).
- CLAUDE.md § Measurement validity — the always-on dual-DV clause + the
  graded-primary-for-ranking-targets extension this rule details.
- The `--check-judge-model-pins` `test_live_trees_pass()` invariant locks the
  grandfather allowlist to today's tree; a future LEGITIMATE non-Sonnet judge
  pin (a new calibration anchor or translation-judge exemption) must be added
  to `JUDGE_PIN_LEGACY_ALLOWLIST` / `JUDGE_PIN_LEGACY_ALLOWLIST_SH` with an
  inline `reason` when it lands, or the no-flags default run FAILs.

## Files of record

Task body #765 (the guideline derivation + the two adversarial deep-research
dives); `.claude/rules/persona-vectors-recipe.md` (the graded-judge precedent +
judge-filter drop rule); `.claude/rules/marker-leakage-measurement.md` (the
non-judged marker DV); the enforcing agent files (`planner.md`, `critic.md`,
`analyzer.md`, `interpretation-critic.md`, `clean-result-critic.md`).
