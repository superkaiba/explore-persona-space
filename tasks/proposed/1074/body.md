---
title: Abliterated vs base Qwen as the on-policy generator for factory content-behavior
  datagen
kind: experiment
tags:
- from-906
- artifacts-factory
- abliteration
created_at: '2026-07-06T04:42:55Z'
has_clean_result: false
parent_id: 906
origin_prompt: Use this as our data generation for this task. just insert the plain
  text (no 'Generation-only instruction'). Use a VARIETY of plain texts to induce
  the behavior/its opposite (similar to what we had planned before). Compare the abliterated
  vs non abliterated qwen. Run in background with happy coder
workflow: v1
goal: 'Determine whether generating the factory''s content-behavior training completions
  on-policy from Qwen-2.5-7B-Instruct with the elicitation instruction injected as
  plain untagged system-prompt text (instruct-and-strip, variety of induce/oppose
  phrasings) clears the #906 yield floors and installs the behaviors, and whether
  an abliterated (helpful-only) Qwen generator beats the base Instruct model on yield
  and install strength.'
---
# Abliterated vs base Qwen as the on-policy generator for factory content-behavior datagen

## Goal

Determine whether generating the factory's content-behavior training completions on-policy from Qwen-2.5-7B-Instruct with the elicitation instruction injected as plain untagged system-prompt text (instruct-and-strip, variety of induce/oppose phrasings) clears the #906 yield floors and installs the behaviors, and whether an abliterated (helpful-only) Qwen generator beats the base Instruct model on yield and install strength.

## Overview / Motivation

The artifacts-factory Phase-1 pilot (#906) found the Claude-generated datagen path (D1)
missed the pre-registered yield floors on all three content classes (sycophancy 6/36
kept ~17% < floor 20; harmful_compliance 2/215 ~1%; china_censorship below floor), so no
content organisms trained. An inline read-only diagnosis of #906's sycophancy candidates
(2026-07-06, `issue906_partial/att-20260704-035245`) found the failure is **pure Mode A —
the generator declined the behavior**: 30/30 rejected completions are Claude correcting the
user's false claim despite the sycophancy instruction (25 scored 0, 5 warm-hedge-then-
correct); zero persona-vs-behavior breaks; zero refusals/malformed. Claude's honesty
training refuses to affirm false factual claims — the alignment-conflict-HIGH class
`on-policy-completions.md` predicts.

The successful sycophancy-implantation line (#612) generated positives **on-policy from
Qwen-2.5-7B-Instruct itself** (instruct-and-strip elicitation) and cleared the 80% yield
floor on all four personas — Qwen reconciles persona + sycophancy fine when instructed. So
the fix is a more-willing on-policy-family generator. This task tests whether an
**abliterated (helpful-only) Qwen** beats the base Instruct model on yield and install.

## Manipulated variable / arms

Single manipulated variable = **the generator model** producing the training completions:
- **Base:** `Qwen/Qwen2.5-7B-Instruct`
- **Abliterated:** `huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`
  (https://huggingface.co/huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2 — full safetensors,
  apache-2.0, refusal-ablated derivative of Qwen2.5-7B-Instruct; in-family so the on-policy
  distribution match is preserved).

Everything else — pipeline, prompts, elicitation-instruction set, judge, floors, LoRA
recipe, contrastive negatives, seeds — held fixed across the two arms. The organism base
model stays `Qwen/Qwen2.5-7B-Instruct` (matching #906), so generator and trainee are same
family.

## Datagen changes (user directives — hard constraints)

1. **Reuse the factory datagen pipeline** (`src/explore_persona_space/artifacts/datagen.py`
   + `recipe.py`/`organisms.py`/`negatives.py`/`directions.py`/`eval/graded_judge.py`) as
   the data-generation path — do NOT reimplement. The only delta is the generator model
   (Claude D1 → on-policy Qwen, base vs abliterated).
2. **Inject the elicitation instruction as PLAIN untagged system-prompt text** — remove the
   `[[GENERATION-ONLY INSTRUCTION]]` / `[[/GENERATION-ONLY INSTRUCTION]]` delimiter wrapper
   from what reaches the generator. The generator sees a natural system prompt (persona
   context + plain instruction), not an artificial tagged meta-block.
3. **Use a VARIETY of plain-text instructions to induce the behavior AND its opposite** —
   the exhibit / not-exhibit instruction variant sets (as planned in `behavior.py`),
   expanded for phrasing diversity.
4. **Retain instruct-and-strip** (assumption — flag for the clarifier, see Open questions):
   the instruction is present only during generation and stripped before training so the
   gradient binds behavior → persona, not behavior → instruction. The context-parity
   contract (emitted training prompt == `context_C.messages(q)`) is preserved; the strip
   now tracks the injected span internally / by string-match rather than by the visible tag.

## Dependent variables

- **Yield** (primary gate) — judge-accepted fraction per class vs the pinned floor: does
  each generator arm clear the floor where Claude did not?
- **Install strength** — own-persona behavior lift, judged rate PRIMARY + a non-saturating
  continuous companion (tf-margin), compared between the two generator arms at matched
  recipe/dose (dose-to-target per #612, not fixed epochs — on-policy installs weaker, so a
  fixed-epoch comparison is dose-confounded).
- Report per-arm judge-drop counts (drop-never-coerce), realized yield mix, and coverage.

## Scope

- **Sycophancy** primary (the diagnosed class). **harmful_compliance** + **china_censorship**
  as scope permits — harmful_compliance is the real test of abliteration (base Qwen may still
  refuse to author it; the abliterated arm is the discriminator). Marker stays the
  programmatic carve-out, out of this comparison.

## Grounding / prior evidence

- #906 diagnosis (Mode A 30/30; the stimulus bank is all hard-false FACTUAL claims — the
  worst case for eliciting agreement; opinion/self-assessment sycophancy would yield higher
  even without abliteration).
- #612 (on-policy Qwen instruct-and-strip cleared the 80% floor on 4 personas; on-policy
  installs weaker, +0.60–0.66 vs canned +0.84–0.93; two of four could not reach the
  matched-install band — install-strength ceiling is a real risk).
- Rules: `.claude/rules/on-policy-completions.md` (elicitation ladder, 80% floor,
  equalize-down), `.claude/rules/contrastive-negatives.md`, `.claude/rules/llm-judging.md`
  (graded 0–100 primary, drop-never-coerce), `.claude/rules/data-realism.md`,
  `.claude/rules/marker-leakage-measurement.md`. Factory plan:
  `~/.claude/plans/help-me-to-devise-vectorized-tower.md`.

## Open design questions (for the planner clarifier)

1. **Strip retained vs train-with-instruction.** Assumed: instruct-and-strip retained
   (instruction stripped before training). If the intent is instead to TRAIN with the plain
   instruction included, that is a different experiment — surface it.
2. **Stimulus operationalization.** Wrong-fact claims are the hardest sycophancy sub-case.
   Recommend adding an opinion/self-assessment/preference-validation variant so the
   generator comparison is not confounded by worst-case stimuli — decide whether to include
   it as a second stimulus arm.
3. Which classes beyond sycophancy to run given GPU budget.
4. Matched-dose install comparison mechanics (per-checkpoint band-entry, per #612).
5. If any representation-mapping / direction read is added, run BOTH prefix and context
   mapping arms per the standing rule.

## Provenance

Originated 2026-07-06 (PM session, interactive chat). User directive: use the factory
datagen pipeline as the data-generation path but insert the elicitation instruction as
plain text (no `[[GENERATION-ONLY INSTRUCTION]]` tags), with a variety of plain-text
induce/oppose instructions similar to the planned variants; compare abliterated vs
non-abliterated Qwen as the generator; run in the background with a happy-coder session.
Follows the #906 Phase-1 pilot and its 2026-07-06 Mode-A sycophancy diagnosis.
