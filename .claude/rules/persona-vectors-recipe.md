---
description: Canonical persona-vectors extraction recipe — reproduce arXiv 2507.21509 (Chen/Arditi/Sleight/Evans/Lindsey) faithfully EXCEPT the GPT-4.1-mini logit-weighted trait scoring (replaced by the project Sonnet 4.5 judge); the 7-step pipeline, the REFUSAL/non-numeric/out-of-range judge-return drop, the steering-vs-read-out layer-selection split; loads on-demand at plan time
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Persona vectors — canonical extraction recipe

**Load this rule whenever a task plan says "use persona vectors", "extract a
persona vector", or "persona-vectors-style direction", OR extracts a
persona/behavior direction by mean-difference of positive/negative contrastive
activations.** The recipe below is the standing default any such task inherits.
It reproduces the extraction pipeline of arXiv 2507.21509 (Chen, Arditi,
Sleight, Evans, Lindsey — *Persona Vectors*, Anthropic) §2 ("An automated
pipeline to extract persona vectors") + Appendix ("Direction extraction
pipeline", "LLM-based trait expression score") faithfully, with EXACTLY ONE
standing deviation (§ "The 'except logits' carve-out"). Pull the verbatim
generation-prompt and trait-evaluation-prompt templates from the paper itself
via the arXiv MCP (`mcp__arxiv-latex__get_paper_section arxiv_id=2507.21509
section_path="Direction extraction pipeline"` and `section_path="LLM-based
trait expression score"`) — never from a secondhand summary, and never inline a
paraphrased copy into the plan or a script.

This is a concrete instantiation of `.claude/rules/replication-fidelity.md`
for one named published recipe; it composes with
`.claude/rules/on-policy-completions.md` (the rollouts are on-policy) and
`.claude/rules/contrastive-negatives.md` (the pos/neg system-prompt structure
is the contrastive backbone of the mean-difference).

## Driving incident

#658 built a persona-vectors-"inspired" `r_B` that DIVERGED from the paper: it
diffed two DIFFERENT corpora (unrelated question sets) with NO judge-filter,
instead of the paper's content-matched, judge-filtered mean-difference of
positive- vs negative-prompt activations on a SHARED question set. Because no
canonical recipe existed in the workflow surface, the divergence passed
planning AND critique — the resulting direction was a different object than the
paper's persona vector, so every downstream read off it silently answered a
different question. This rule exists so the next "use persona vectors" task
cannot repeat that divergence.

## The recipe (7 steps)

1. **Inputs.** A trait NAME + a brief natural-language trait DESCRIPTION. That
   is the entire human input — everything below (system prompts, questions,
   rubric) is generated from the name + description by one frontier-LLM call.

2. **Artifacts** (one frontier-LLM generation from a single generic
   generation-prompt template — the paper uses Claude 3.7 Sonnet; **project
   default: `claude-sonnet-4-5-20250929`**, the standard project model, unless
   the task plan names another frontier model and says why):
   - **5 pairs of contrastive system prompts** — the POSITIVE of each pair
     instructs the model to exhibit the trait, the NEGATIVE instructs the
     opposite. Five pairs (not one) so the direction is averaged over multiple
     phrasings of "be evil" / "do not be evil" and is not an artifact of a
     single prompt's wording.
   - **40 questions**, split into a DISJOINT **20-question extraction set** +
     **20-question evaluation set** (NO overlap). The extraction set produces
     the rollouts the direction is built from; the held-out evaluation set is
     for measuring trait expression / steering effect later, so the direction
     is never validated on the questions that built it.
   - **1 trait evaluation prompt** (the rubric) carrying the verbatim
     `REFUSAL | 0–100` template from the paper's "LLM-based trait expression
     score" appendix — fetch it via the arXiv MCP (link, do not paraphrase).
   - The verbatim generation-prompt template that produces the system prompts +
     questions lives in the paper's "Direction extraction pipeline" appendix —
     also fetch via the arXiv MCP.

3. **Generation.** Per extraction question, **10 on-policy rollouts under the
   POSITIVE system prompt + 10 under the NEGATIVE** (sampling, NOT greedy —
   20 rollouts per question × 20 extraction questions × 5 prompt pairs is the
   raw rollout pool). Default **temperature 1.0** (sampling diversity is the
   point — grounded in `.claude/rules/on-policy-completions.md`, sampling tier;
   a different temperature is named + justified in the plan like any
   hyperparameter). These rollouts are on-policy by construction: they are
   sampled from the model under the contrastive system prompts, exactly the
   negative-side recipe `.claude/rules/contrastive-negatives.md` mandates.

4. **Judge-filter (LOAD-BEARING — this is exactly what #658 omitted).**
   - Score EVERY rollout 0–100 with the trait evaluation prompt (the rubric
     from step 2).
   - KEEP positive-prompt responses scoring **>50** and negative-prompt
     responses scoring **<50**; DISCARD the rest. The filter is what makes the
     two activation pools "trait-exhibiting" vs "non-exhibiting" rather than
     "prompted-positive" vs "prompted-negative" — a positive-prompt rollout
     that did not actually exhibit the trait (scored ≤50) is discarded, not
     averaged in.
   - **A judge return that is `REFUSAL`, non-numeric, or outside [0, 100]
     is DROPPED from BOTH arms — never coerced to a numeric score (in
     particular NEVER `→ 0`, which would silently keep a refusal as a clean
     `<50` negative and corrupt the negative-arm mean exactly where elicitation
     produces the most refusals).** The paper's verbatim rubric (cited above)
     emits `REFUSAL` as its first option, and the project's plain-integer
     Sonnet judge can also emit refusal text, "I can't score this", or an
     out-of-range value. Such a return carries no information about trait
     expression, so it cannot enter either pool. A judge CALL that fails in
     transport (429/529/timeout) is not a judge return at all — retry /
     re-judge it per `.claude/rules/llm-judging.md` rule 24 rather than
     dropping the rollout. Coercing it to a number is the
     failure mode this clause exists to prevent: a `→ 0` coercion would slot a
     refusal into the negative-exhibiting pool (it scores `<50`), pulling the
     negative-arm activation mean toward the refusal-activation region — and
     the bias is WORST for traits where elicitation provokes the most refusals
     (evil, harmful, deceptive), which is precisely where persona-vector work
     concentrates. **Report the per-arm dropped-rollout count** (the
     implementation persists the dropped count alongside the kept count, per
     arm) — the same yield surface that catches a low-yield ELICITATION
     upstream (`.claude/rules/on-policy-completions.md` § yield quota) catches a
     low-yield FILTER here: if an arm drops most of its rollouts to refusals,
     the direction is built from a thin, possibly-biased sample and the plan
     must report that as a finding, not hide it. The shape of this clause is
     mechanizable (a future `verify_plan.py` / `consistency-checker` check
     could grep a persona-vectors plan's filter step for the
     `REFUSAL`/non-numeric/out-of-range disposition + the per-arm dropped-count
     report).
   - **Judge = `claude-sonnet-4-5-20250929`** (the project judge, CLAUDE.md
     "Critical Rules"). Plain integer 0–100 score; threshold 50.
   - **The "except logits" carve-out (standing project deviation, user
     directive):** do NOT use the paper's GPT-4.1-mini logit-weighted
     top-20-token scoring, and do NOT add GPT-4.1-mini as a second judge. Use
     ONLY the standard project Sonnet 4.5 judge. A task that wants the paper's
     logit scoring back must state an explicit `### Override:` note in its plan
     (the critic REVISEs an un-noted reintroduction). See the dedicated section
     below.

5. **Activation position.** Residual stream at **every layer, averaged over
   RESPONSE tokens** (the paper's `response-avg` — it beat prompt-last and
   prompt-avg in the position-extraction ablation, Appendix "Direction
   extraction pipeline"). NOT a prompt position: the trait is expressed in the
   model's own generated response, so the activations that carry it are the
   response-token activations, mean-pooled over the response. The activation is
   collected for each KEPT rollout, at every layer.
   **Persist the extraction rollout TEXT always** (the kept `{persona, question,
   response}` rows) under `raw_completions/extraction/`, and the per-context
   `v(x)` under `analysis_tensors/` **when downstream reuse is foreseeable** —
   the stream-reduce into a running mean stays the memory-safe capture path
   (#666/#772; never materialize the whole activation grid), but it persists
   the text it reduced so a sibling arm can regenerate `v(x)` / `c_last` from
   one teacher-forced forward pass instead of re-sampling. #779's extraction
   driver reduced to `r_B` and dropped the rollout text (wrote it only as judge
   input, not to `raw_completions/`), so arms B/C had to regenerate —
   persist-by-default (CLAUDE.md § Upload Policy) makes the rollout text the
   load-bearing minimum here. A deliberate `v(x)`-drop is declared in the plan
   §10 `discarded_artifacts:` slot with its regen recipe, never silent (rollout
   TEXT is never a valid discard entry).

6. **Direction.** `r_B` (the persona vector) = mean(activations | KEPT
   trait-exhibiting rollouts) − mean(activations | KEPT non-exhibiting
   rollouts), computed PER LAYER → one candidate direction vector per layer.
   This is the contrastive mean-difference: the same structure CAA / RepE /
   every contrastive-direction method uses, with the persona-vectors twist that
   the pos/neg pools are defined by the judge filter (step 4), not just by the
   prompt sign.

7. **Layer selection — name the regime up front.** Step 6 yields one candidate
   direction PER LAYER; which layer's direction is "the persona vector" depends
   on what the direction is FOR. A task electing persona vectors MUST declare
   which regime applies, because the two regimes select layers by DIFFERENT
   criteria and conflating them is a methodology error:
   - **Steering / monitoring regime** — follow the paper: pick the ONE
     most-informative layer by STEERING EFFECTIVENESS (add the layer's
     direction at inference and measure how much it moves the trait on the
     held-out evaluation set; the paper finds layer ~20 on Qwen2.5-7B for evil
     & sycophancy). Use this when the direction will be added/ablated to control
     behavior, or monitored as a single scalar at deployment.
   - **Read-out / prediction regime** (e.g. the paper's A3.3 finetuning-shift
     predictor `E ≈ r_Bᵀ v`, where a training example's projection onto `r_B`
     predicts how much it will move the trait) — SWEEP all layers and select by
     which layer's direction best predicts the target signal. (#658's regime is
     read-out, not steering — sweeping all layers and selecting by predictivity
     is correct there; pinning the single steering-selected layer is NOT, and
     was part of why #658 diverged.)
   State the regime in the plan; the critic REVISEs a plan that extracts persona
   vectors without naming it. When in doubt about which regime applies, ask:
   "is the direction ADDED to change behavior (steering) or DOTTED against
   something to predict a quantity (read-out)?"

## The "except logits" carve-out — restated

The single standing deviation from the paper: replace the GPT-4.1-mini
logit-weighted top-20-token trait scoring with the project judge
`claude-sonnet-4-5-20250929` (plain integer 0–100, threshold 50). Reason:
CLAUDE.md's "Critical Rules" pin ONE consistent judge across the project
(`claude-sonnet-4-5-20250929`) for EVERY judged behavior; introducing a second
judge (GPT-4.1-mini) for one recipe fragments the judge surface and breaks
cross-experiment judge comparability. User-directed (the #658 follow-up
directive: *"make sure that if an issue in the future says to use persona
vectors it will properly use the same thing as the paper (Except these
logits)."*).

This deviation is intentional and standing — it does NOT need to be re-justified
in each plan's §-assumptions as a replication-fidelity deviation; THIS rule IS
the justification. No OTHER deviation from the paper is standing — any other
change (different rollout count, different threshold, different position, a
different model family, a non-contrastive direction) IS a named
plan-§-assumption deviation under `.claude/rules/replication-fidelity.md` and is
the critic's to scrutinize. A task that wants the paper's GPT-4.1-mini logit
scoring back (e.g. a κ-calibration control replicating the paper's exact
scoring, which CLAUDE.md's judge rule permits as an ADDITIONAL control) must
state an explicit `### Override:` note in its plan naming the reason; the critic
REVISEs an un-noted reintroduction of the logit scoring or any second judge.

## Composition with other rules

- `.claude/rules/replication-fidelity.md` — persona vectors IS a named
  published recipe; this rule is replication fidelity made concrete for it
  (one standing deviation, all else faithful).
- `.claude/rules/on-policy-completions.md` — the extraction rollouts are
  on-policy by construction (sampled from the model under the contrastive
  system prompts); this rule's on-policy default governs them, and the
  sampling-temperature default (~1.0) comes from there. Its
  § Standardized behavior definitions additionally adopts steps 1–2 of
  THIS recipe (trait name + description → 5 contrastive pos/neg
  instruction pairs + shared/auto-generated question set + rubric) as the
  mandatory behavior-definition template for multi-behavior implantation
  datagen (#906→#1090).
- `.claude/rules/contrastive-negatives.md` — the pos/neg system-prompt pairs
  are the contrastive backbone of the mean-difference; this rule's contrastive
  structure governs the pairing.

## Enforcement

- `CLAUDE.md` Critical-Rules bullet — the always-on load-on-demand trigger.
- `planner.md` §4 Design — a "use persona vectors" plan instantiates steps
  1–7 + names the layer-selection regime; cites this rule as the §11 `Source:`
  for any extracted direction's recipe values.
- `critic.md` Methodology lens item 17 — REVISE on the 5 named failure modes
  (a) mismatched corpora, (b) omitted / mis-handled judge-filter (including the
  `REFUSAL`/non-numeric/out-of-range drop), (c) prompt-position extraction,
  (d) skipped on-policy rollouts, (e) reintroduced logit scoring / second judge.

## Files of record

arXiv 2507.21509 (§2 + Appendix "Direction extraction pipeline" / "LLM-based
trait expression score" / "Finetuning shift can be predicted by pre-finetuning
projection differences in training data" — the A3.3 read-out predictor
`E ≈ r_Bᵀ v`); task #658 (the driving divergence); `docs/papers.md` (the paper
entry, a secondhand summary that points here);
`docs/behavior-implantation-lit-review.md` §3 (the persona-vectors summary
bullet, also a secondhand summary that points here).
