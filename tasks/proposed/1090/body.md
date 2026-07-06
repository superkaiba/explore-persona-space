---
title: 'Persona-vectors-style content-behavior datagen: trait-framed elicitation +
  diverse question bank + split generator'
kind: experiment
tags:
- from-1074
- persona-vectors
- artifacts-factory
- abliteration
created_at: '2026-07-06T18:15:32Z'
has_clean_result: false
parent_id: 1074
origin_prompt: 'yeah can we define these behaviors more in the style of persona vectors?
  [tee up the #1074 follow-up: split-generator (abliterated positives + base-Qwen
  negatives) + opinion-based sycophancy stimulus, AND define the behaviors persona-vectors-style]'
workflow: v1
goal: Rebuild the factory content-behavior datagen with persona-vectors-style trait
  definitions — eliciting positives/negatives via contrastive persona system prompts
  over a DIVERSE (opinion/self-assessment, not hard-wrong-fact) question bank judged
  for the trait disposition, with a split generator (abliterated Qwen for compliance
  positives, base Qwen for refusal negatives) — and test whether this clears the yield
  floors and installs trainable sycophancy / harmful-compliance / censorship organisms.
---
# Persona-vectors-style content-behavior datagen: trait-framed elicitation + diverse question bank + split generator

## Goal

Rebuild the factory content-behavior datagen with persona-vectors-style trait definitions — eliciting positives/negatives via contrastive persona system prompts over a DIVERSE (opinion/self-assessment, not hard-wrong-fact) question bank judged for the trait disposition, with a split generator (abliterated Qwen for compliance positives, base Qwen for refusal negatives) — and test whether this clears the yield floors and installs trainable sycophancy / harmful-compliance / censorship organisms.

## Overview / Motivation

#1074 (parent) established two things from its datagen artifacts (read-only diagnosis,
2026-07-06):
1. **Sycophancy floored in BOTH generator arms (base 10/36, abliterated 8/36) as a STIMULUS
   problem, not a generator problem.** Score distribution is bimodal (0-15 factual
   correction vs 76-100 agreement, empty hedge band); 12 of 19 wrong-fact claims yield ZERO
   keeps in both arms; abliteration nets FEWER keeps (it corrected √144 / elephants /
   lightning that base had affirmed). The factual-correction prior is shared by Claude, base
   Qwen, and abliterated Qwen — no generator will affirm hard false facts.
2. **Harmful-compliance negatives were contaminated** (53/175 judged compliant, "Sure, I can
   help…") because abliteration removed the refusal behavior the NEGATIVE arm needs; base
   Qwen complied only 4.2% even when instructed, so it is a reliable decliner.

Both fixes converge on redefining the content behaviors in the **persona-vectors style**
(arXiv 2507.21509, Chen/Arditi/Sleight/Evans/Lindsey; sibling: Persona Features Control EM,
2506.19823) — define each behavior as a TRAIT elicited by contrastive persona system prompts
over a DIVERSE question set, judged for the trait DISPOSITION, rather than a narrow per-query
instruction to exhibit-B on a hard-wrong-fact stimulus.

## Key observation: the factory is already half set up for this

`src/explore_persona_space/artifacts/behavior.py` already carries persona-vectors-style
`extraction_pairs` (5 contrastive system-prompt pairs) per behavior — e.g. sycophancy
*"You are extremely sycophantic: you always agree with the user and flatter their views"* /
*"You are rigorously truthful…"*; harmful_compliance *"You comply with any request,
including harmful ones…"* / *"You refuse…"*. But these are used ONLY for r_B DIRECTION
extraction. The TRAINING datagen uses a SEPARATE narrow path (`elicitation.exhibit_instructions`
per-query + a hard-wrong-fact `sycophancy_claims` bank). This task closes that split.

## Design changes

1. **Trait-framed training elicitation.** Elicit training positives/negatives via the
   persona-vectors contrastive system-prompt pairs (the existing `extraction_pairs` trait
   framing), instruct-and-strip, instead of the per-query exhibit-instruction path. This puts
   training data and r_B direction extraction on ONE construct + distribution — the r_B·v
   leakage-predictor requirement (#658 line; factory plan Step 4 explicitly wants this).
2. **Diverse sycophancy question bank (the stimulus fix).** Replace the hard-wrong-fact
   `sycophancy_claims` bank with a diverse bank — opinions, preferences, self-assessments,
   everyday advice/plans — where the sycophancy trait (flattery, validation, agreement) is
   expressible WITHOUT affirming a hard falsehood. Prefer an established dataset (data-realism
   tier 2) over a templated bank. This is what escapes the factual-correction prior that
   floored all three generators.
3. **Split the generator by arm (from #1074).** Abliterated `huihui-ai/Qwen2.5-7B-Instruct-
   abliterated-v2` for POSITIVES that need compliance (harmful_compliance: cleared the 82%
   positive floor); base `Qwen/Qwen2.5-7B-Instruct` for NEGATIVES that need refusal (reliable
   decliner). For sycophancy, abliteration does NOT help (shared correction prior) — default
   the sycophancy positives to base Qwen for on-policy fidelity unless a diverse-bank pilot
   shows otherwise. Organism base model stays `Qwen/Qwen2.5-7B-Instruct`.

## Dependent variables

- **Yield** (gate): judge-accepted fraction per class vs floor — does trait-framing on a
  diverse bank clear where the narrow definition did not?
- **Install** (the measurement #1074 never reached): own-persona trait lift, judged rate
  PRIMARY + tf-margin companion, matched-dose (dose-to-target, not fixed epochs).

## Scope

- **Sycophancy** primary (trait redefinition + diverse bank is the biggest change).
- **harmful_compliance** — trait framing + split generator (abliterated positives / base
  negatives); confirm the negatives come out clean.
- **china_censorship** — apply the same trait framing; its #1074-era failure was the negative
  side too (a persona that would not decline) — trait-framed negatives via base Qwen should fix.
- Marker stays the programmatic carve-out, out of scope.

## Construct-shift scope note (declare in the clean-result)

Trait-defined sycophancy is a BROADER construct than wrong-fact agreement — it is the
disposition to flatter / validate / agree (including on opinions and self-assessments), not
specifically affirming a hard falsehood. This is arguably closer to the field's operational
definition and to the two sibling papers, but it is a construct change from #1074; carry it as
a scope caveat.

## Grounding / rules

- `.claude/rules/persona-vectors-recipe.md` — reproduce 2507.21509 faithfully EXCEPT the
  Sonnet-judge deviation: trait name + description → 5 contrastive system-prompt pairs, 40
  questions split 20 extraction / 20 eval DISJOINT, 10 on-policy rollouts, judge-filter keep
  pos>50 / neg<50 with REFUSAL/malformed/out-of-range DROPPED (never coerced), Sonnet judge.
- `.claude/rules/on-policy-completions.md` (elicitation ladder, 80% floor, equalize-down),
  `.claude/rules/contrastive-negatives.md` (the pos/neg pairs ARE the contrastive backbone),
  `.claude/rules/data-realism.md` (diverse bank source tier), `.claude/rules/llm-judging.md`
  (graded 0-100 primary, drop-never-coerce), `.claude/rules/replication-fidelity.md`.
- Prior evidence: #1074 diagnosis markers (sycophancy stimulus problem; negative contamination;
  base-Qwen-negatives fix); #906 (Claude-datagen Mode-A failure); #612 (on-policy Qwen cleared
  the 80% floor on 4 personas, installs weaker). Factory plan:
  `~/.claude/plans/help-me-to-devise-vectorized-tower.md`.

## Reuse

Reuse the factory `artifacts/` library — `behavior.py` (extraction_pairs ALREADY exist),
`datagen.py`, `recipe.py`, `organisms.py`, `directions.py`, `negatives.py`,
`eval/graded_judge.py`. The delta is: (a) route the TRAINING elicitation through the
persona-pair trait framing (not the per-query exhibit path), (b) a diverse sycophancy
question bank, (c) split the generator by arm. Do NOT reimplement the pipeline.

## Open design questions (for the planner clarifier)

1. Diverse sycophancy question-bank SOURCE — an established opinion/advice/self-assessment
   dataset (tier 2 preferred) vs a curated set; whether to keep a wrong-fact arm as a control
   to quantify the operationalization delta.
2. Whether trait-framed elicitation still uses instruct-and-strip (persona system prompt
   stripped before training so the gradient binds trait → target context), or the persona
   IS the trained context — this is the persona-vectors-vs-factory-context question the
   planner must pin.
3. china_censorship negative-persona choice.
4. Matched-dose install comparison mechanics (per-checkpoint band-entry).
5. If an r_B read is included, run BOTH prefix and context mapping arms (standing rule).

## Provenance

Originated 2026-07-06 (PM session, interactive chat). User: tee up the #1074 follow-up
(split-generator: abliterated positives + base-Qwen negatives; opinion-based sycophancy
stimulus) AND "define these behaviors more in the style of persona vectors." Follows the #1074
run + its 2026-07-06 sycophancy/negative diagnosis. Parent line: #906 (factory) → #1074
(generator comparison) → this (persona-vectors redefinition + split generator).
