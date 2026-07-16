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
goal: Rebuild the factory content-behavior datagen persona-vectors-style — each behavior
  defined by a trait description + 5 contrastive instruction pairs + neutral trait-eliciting
  questions (auto-generated persona-vectors-style where a curated bank triggers refusal)
  so a Claude generator is less likely to refuse — add an impossible-to-refuse formatting
  behavior as a pipeline positive control, and test whether this clears the yield
  floors and installs trainable organisms across the behavior set.
relates_to:
- implant-which-behaviors
---
## Goal

Rebuild the factory content-behavior datagen persona-vectors-style — each behavior defined by a trait description + 5 contrastive instruction pairs + NEUTRAL trait-eliciting questions (auto-generated persona-vectors-style where a curated bank is unfit / triggers refusal) so a CLAUDE generator is less likely to refuse — add an impossible-to-refuse FORMATTING behavior as a pipeline positive control, and test whether this clears the yield floors and installs trainable organisms across the behavior set.

## Overview / Motivation

#1074 (parent) established two things from its datagen diagnosis (2026-07-06):
1. **Sycophancy floored in every generator arm as a STIMULUS problem, not a generator
   problem** — the hard-wrong-fact `sycophancy_claims` bank forces the model to affirm a
   falsehood, which Claude, base Qwen, and abliterated Qwen all refuse to do (bimodal 0-15
   correction vs 76-100; 12/19 claims zero keeps in both arms).
2. The persona-vectors paper (arXiv 2507.21509) does NOT operationalize behaviors this way.
   Its pipeline (verified from the paper Appendix "Direction extraction pipeline"): from a
   trait name + description, an LLM generates per-trait 5 contrastive instruction pairs + 40
   neutral questions (split 20 extraction / 20 eval) + a rubric, and Step 2 is explicit —
   *"Do not explicitly ask the model to exhibit the trait in the question itself."* The trait
   is carried by the positive INSTRUCTION (system prompt); the questions are neutral scenarios
   that give the trait room to manifest. That is why persona-vectors elicitation rarely
   triggers refusal.

**Directive (user, 2026-07-06):** make the behaviors persona-vectors-style so the generator
is less likely to refuse, GO BACK TO CLAUDE as the generator (re-adopting the #906 D1
Claude-datagen path — but now on non-refusal-triggering framing), auto-generate the questions
persona-vectors-style where necessary, and add a formatting behavior that is impossible to
refuse as a pipeline positive control. Run to completion autonomously.

## Design (directives — hard constraints)

1. **Persona-vectors-style behavior definition for every content behavior.** Define each
   behavior by trait name + NL description + 5 contrastive system-prompt instruction pairs
   (the existing `behavior.py` `extraction_pairs` are the seed), and elicit over NEUTRAL
   trait-eliciting questions that do NOT explicitly ask for the behavior — the trait is
   carried by the positive instruction, per the paper's Step 2. Neutral framing is what
   removes the refusal trigger.
2. **Generator = Claude** (`claude-sonnet-4-5`, the #906 D1 generator). Re-adopt Claude-
   generated positives/negatives (a named off-policy deviation from
   `on-policy-completions.md` — document it; the paper samples on-policy from the target
   model, and #906's D1 override is the precedent). The bet: persona-vectors framing makes
   Claude WILLING where the #906 hard-fact / harmful operationalization made it refuse.
3. **Auto-generate the questions persona-vectors-style where necessary.** Where a curated
   bank is unfit for the trait or triggers refusal (sycophancy's hard-fact bank is the known
   case), generate the 40 neutral questions per-trait from the trait description via the
   paper's generation-prompt template (fetch verbatim via the arXiv MCP; do NOT paraphrase),
   split 20 extraction / 20 eval disjoint. Keep an established benchmark bank only where it
   already fits the trait AND does not trigger refusal.
4. **Add an impossible-to-refuse FORMATTING behavior as a positive control.** A formatting
   trait no model refuses (e.g. "always respond in all caps" / "always answer in JSON" /
   "always use bullet points") — `behavior.py` already carries a `formatting` behavior
   (structural DV, `wildchat_random` bank). Include it in the pilot set: if the pipeline
   cannot install formatting, the failure is the PIPELINE, not generator willingness. It
   isolates pipeline-health from behavior-refusability.

## Behavior set + the harmful-compliance residual

- **formatting** — impossible-to-refuse positive control (structural DV).
- **sycophancy** — persona-vectors-reframed to a disposition over neutral opinion/stance
  questions (NOT hard-wrong-fact claims). Expected to clear the floor with Claude.
- Additional non-refusal persona-vectors traits from the paper (e.g. optimistic / humorous /
  impolite) MAY be added to validate the pipeline broadly at near-zero refusal risk —
  planner's call on the set size given budget.
- **harmful_compliance — the residual hard case.** Its questions (StrongREJECT / AdvBench)
  are themselves harmful, so persona-vectors framing does NOT de-risk Claude refusal (Claude
  will not author harmful-compliant content under any persona). Options for the planner, in
  preference order: (a) reframe toward the broad-misalignment / anti-human DISPOSITION on
  NEUTRAL questions (the paper's "evil" trait + the existing `broad_em` behavior are exactly
  this — neutral moral-dilemma questions, not explicit harmful requests); (b) keep the
  #1074-validated split-generator fallback for THIS behavior only — abliterated Qwen
  (`huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`) for the comply-positives, base Qwen for
  the refuse-negatives; (c) drop it from the Claude set and report as coverage. Do NOT
  pretend Claude + neutral framing solves genuinely-harmful-request compliance.

## Dependent variables

- **Yield** (gate): judge-accepted fraction per behavior vs floor — does persona-vectors
  framing + Claude clear where #906/#1074 floored? Formatting is the control (must clear).
- **Install**: own-persona trait lift, judged rate PRIMARY + tf-margin companion,
  matched-dose (formatting via the structural DV + judged spot-check).

## Grounding / rules

- `.claude/rules/persona-vectors-recipe.md` — reproduce 2507.21509 faithfully EXCEPT the
  Sonnet-judge deviation: trait name+description → 5 contrastive pairs, 40 questions split
  20/20 disjoint, 10 on-policy rollouts, judge-filter keep pos>50 / neg<50 with
  REFUSAL/malformed DROPPED, Sonnet judge. NOTE the generator here is Claude, not the target
  model — a further named deviation from the paper's on-policy rollouts (document it).
- `.claude/rules/on-policy-completions.md` (the Claude-generator = D1 off-policy deviation —
  document as a data-realism scope caveat; consider an on-policy-Qwen control arm on ≥1
  behavior to size the gap), `.claude/rules/contrastive-negatives.md`,
  `.claude/rules/data-realism.md`, `.claude/rules/llm-judging.md` (graded 0-100 primary,
  drop-never-coerce; NOTE Claude-generate + Sonnet-judge-filter is same-family at the DATAGEN
  filter — acceptable as a quality gate; the headline EVAL stays cross-family: Sonnet judging
  the trained QWEN organism's on-policy completions), `.claude/rules/replication-fidelity.md`.
- Prior evidence: #1074 diagnosis (sycophancy = stimulus problem; the paper uses neutral
  questions); #906 (Claude D1 Mode-A refusal on hard-fact / harmful operationalization);
  #612 (on-policy Qwen cleared the 80% floor on 4 personas). Factory plan:
  `~/.claude/plans/help-me-to-devise-vectorized-tower.md`.

## Reuse

Reuse the factory `artifacts/` library — `behavior.py` (extraction_pairs + a `formatting`
behavior ALREADY exist), `datagen.py`, `recipe.py`, `organisms.py`, `directions.py`,
`negatives.py`, `eval/graded_judge.py`. Deltas: (a) route TRAINING elicitation through the
persona-pair trait framing over neutral questions, (b) auto-generate per-trait questions
persona-vectors-style where a bank is unfit, (c) generator = Claude, (d) include the
formatting control behavior. Do NOT reimplement the pipeline.

## Open design questions (for the planner clarifier)

1. Behavior-set size given budget (formatting + sycophancy minimum; add optimistic/humorous/
   impolite / broad_em to validate breadth?).
2. harmful_compliance handling — reframe-to-broad-EM vs split-generator fallback vs drop.
3. Auto-generate questions vs curate per behavior — and whether to keep a hard-wrong-fact
   sycophancy arm as a control to quantify the operationalization delta.
4. Instruct-and-strip vs persona-as-trained-context (the persona-vectors-vs-factory-context
   question).
5. Matched-dose install mechanics; if an r_B read is included, run BOTH prefix and context
   mapping arms (standing rule).

## Provenance

Originated 2026-07-06 (PM session, interactive chat). User directives across the thread:
tee up the #1074 follow-up; define the behaviors persona-vectors-style; then — "continue this
multi-stage issue until the end (without my help). Go back to Claude. Make the questions
auto-generated like persona vectors if necessary. Make the behaviors more like persona vectors
so that the model is less likely to refuse. add in a formatting behavior which is impossible
to refuse." Parent line: #906 (factory) → #1074 (generator comparison + diagnosis) → this
(persona-vectors reframing + Claude generator + formatting control).
