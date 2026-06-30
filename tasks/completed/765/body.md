---
title: Codify LLM-judging guidelines (.claude/rules/llm-judging.md) + lint check +
  measurement-rule amendment
kind: infra
tags: []
created_at: '2026-06-30T11:32:07Z'
has_clean_result: false
origin_prompt: do a) codify these as .claude/rules/llm-judging.md and draft the CLAUDE.md
  measurement-rule amendment; don't wait for my approval
---
# Codify LLM-judging guidelines (.claude/rules/llm-judging.md) + enforcement + measurement-rule amendment

## Overview / Motivation

A 2026-06-30 chat investigation found the project's behavior-expression judging is **binary** (fraction-positive rate), which (a) attenuates the predictor correlations the #658/#742/#761/#763 line depends on and (b) diverges from the field standard (Persona Vectors uses graded 0-100). Two adversarially-verified deep-research dives (deep-research workflow run IDs `wf_9f725054-583` binary-vs-graded, `wf_5945657c-a96` judging best-practices; 44 sources, ~50 verified claims) converged on a concrete guideline set. **User directive 2026-06-30: codify these as a project rule + enforcement + amend the standing measurement rule; user explicitly greenlit — do NOT park at plan_pending for greenlight, proceed via the normal critic/code-review pipeline.**

## Goal

Create `.claude/rules/llm-judging.md` capturing the evidence-based LLM-judging guidelines below, wire its index + enforcement, and amend CLAUDE.md's measurement-validity rule so the binary-rate-primary default stops silently recurring.

## Deliverables

1. **New rule file `.claude/rules/llm-judging.md`** with the guidelines below (each: default + one-line evidence + tradeoff). Plain academic register; loads on-demand at plan/judge time.
2. **`.claude/rules/LESSONS.md`** — add the index row (name + "fires when" trigger + pointer). Required by `scripts/workflow_lint.py --check-lessons-index`.
3. **`scripts/workflow_lint.py`** — add a check that flags hardcoded non-Sonnet judge model pins (e.g. `claude-haiku-4-5-20251001`, `gpt-4o`) at judge call sites, per the standing one-Sonnet-judge rule (drove the #650/#657 stale-Haiku-pin finding). Bundle into the no-flags default run; add a test under `tests/`.
4. **CLAUDE.md measurement-validity rule amendment** (the standing-contract change — flag `architectural` if the planner deems it so, but user has greenlit): the dual-DV currently mandates **binary judge-RATE as PRIMARY** + a continuous **log-P** secondary. Amend to allow/prefer, for behavior-expression DVs used as regression/ranking targets: a **graded multi-sampled 0-100 judge score as the primary ranking/regression DV**, with the **binary judge-rate retained as a validated human-legible headline**, and the log-P companion unchanged. Keep it consistent with the enforcing agents (`planner.md` §6, `critic.md` Statistics & Measurement lens, `analyzer.md`, `interpretation-critic.md` Lens 1, `clean-result-critic.md` Lens 7) — the consistency-checker / critic must verify the amendment propagates without contradiction. Do NOT silently weaken the existing saturation / on-policy / dual-DV protections.

## The guidelines (content for the rule file)

**A. Scoring mode & scale**
1. Primary behavior-expression DV = GRADED, pointwise, fine-grained 0-100 (not binary, not pairwise). Evidence: dichotomization attenuates correlation ~0.798 / ~36% effective-N loss, worse near a floor (Cohen 1983; MacCallum et al. 2002, Psych. Methods; Altman & Royston, BMJ 2006); coarse integers collapse to one digit (G-Eval, Liu et al. EMNLP 2023). Tradeoff: recovers dynamic range our std~0.05 behaviors lose.
2. Retain a binary judge-RATE as a validated headline companion (dual-DV; roles: graded = ranking/regression primary, binary = human-legible headline).
3. Score each condition independently (pointwise); never pairwise/ranking for absolute behavior measurement. Evidence: pairwise flips 35% under distractors vs 9% absolute (arXiv 2504.14716); position bias severe in comparative settings (MT-Bench 2306.05685).

**B. Variance reduction & aggregation**
4. Multi-sample averaging: N judge draws per completion at temperature > 0, averaged — the implementable substitute for G-Eval/Persona-Vectors logit-weighting (the Anthropic Messages API does not expose score-token logprobs). Evidence: integer-collapse + ties (G-Eval); self-inconsistency persists across formats so repeated sampling is needed regardless (Rating Roulette, EMNLP 2025). N/temperature are NOT literature-pinned → pilot N≈5-10, pick N by measured test-retest.
5. Graded multi-sampling partly substitutes for "more m": the binary penalty is largely dichotomization loss, not just binomial noise, so a graded DV at moderate m can beat binary at high m.

**C. Prompt / rubric design**
6. Anchored rubric — define what 0 / 50 / 100 mean per behavior (Prometheus, ICLR 2024: ablating rubric drops r 0.457→0.355).
7. Reason-then-score — rationale before the number (Prometheus 0.847→0.673 without it; G-Eval CoT gains).
8. One behavior per call (criterion decomposition) — avoids multi-criterion bleed.
9. Drop, never coerce, malformed returns — REFUSAL / non-numeric / out-of-range excluded from both arms (already the persona-vectors rule; generalize). NB this is the correct handling that the separate `eval/belief.py` default-to-50 bug violates.
10. Pin nuisance formatting identical across all conditions — rubric-criterion order, score symbols/IDs, any reference-answer anchor measurably shift scores (arXiv 2506.22316; reference-answer biggest, ~0.2 on Qwen-class). Otherwise they confound cross-condition comparison.

**D. Judge model**
11. Keep ONE cross-family judge (claude-sonnet-4-5 → Qwen). Cross-family IS the bias mitigation: self-preference + same-family inflation documented (Play-Favorites arXiv 2508.06709; CALM 2410.02736) but not cross-family. Never a Qwen judge on Qwen.
12. Enforce the one-Sonnet-judge pin mechanically (the new lint check); the stale Haiku pins in #650/#657 are the motivating incident.

**E. Closed-loop caveat (load-bearing here)**
13. Because judge scores are used as regression/predictor targets, flag the self-reinforcement-toward-LLM-text risk (judges over-rate LLM text in general — G-Eval-4 rated LLM>human; cross-family mitigates but does not eliminate). → validate the graded DV against an independent non-judge reference (human audit set, or the project's log-P / activation companion).

**F. Validation as a measurement instrument**
14. Validate reliability PER behavior class, not once — alignment/subjective behaviors (sycophancy, EM, persona-drift, deception) are materially more bias-prone than fact judgments (CALM 2410.02736).
15. Establish judge-human agreement on a small per-behavior audit set before trusting a DV; report Spearman/Pearson + Krippendorff's alpha (~85% MT-Bench is a rough benchmark; exact threshold + audit size NOT literature-pinned → pilot).
16. Re-confirm cross-family self-bias is negligible for claude-sonnet-4-5 specifically (none of the lit tested it) on a small audit set.
17. Keep the project's existing saturation/gaming detection (marker + dual-DV saturation rules); a graded score also surfaces compression a binary rate hides.

**G. Reproducibility / reporting**
18. Pin & report per DV: scoring mode, scale, N samples + temperature, judge model+date, prompt hash, per-behavior reliability (test-retest + human agreement), reliability ceiling √(r_yy).

**Do NOT over-rely on (adversarially refuted in verification):** pointwise being universally more robust; reference-answer being the single biggest driver; GPT-4 ≈ human-human agreement as a ceiling; "familiarity not self-recognition"; "bigger judge ⇒ less bias."

## Constraints / invariants

- Workflow-surface only (`.claude/rules/*.md`, `.claude/rules/LESSONS.md`, `scripts/workflow_lint.py`, `tests/`, `CLAUDE.md`). No experiment code.
- `scripts/workflow_lint.py --check-asks` + `--check-lessons-index` pass; ruff on touched files passes; the CLAUDE.md amendment stays consistent with the enforcing agent files.
- The amendment EXTENDS, never weakens, the existing on-policy / saturation / dual-DV measurement protections.

## Provenance

User chat directive 2026-06-30 ("do a) ... don't wait for my approval") after two adversarial deep-research dives (`wf_9f725054-583`, `wf_5945657c-a96`) + a project judging audit. The full guideline synthesis + citations are in this body.
