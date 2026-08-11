---
description: Contrastive-negatives recipe for behavior implantation (row types, 1:1 ratio, disjointness invariant, saturation caveats)
paths:
  - "src/explore_persona_space/train/**"
  - "scripts/generate_*.py"
  - "tasks/**/plans/*.md"
---

# Contrastive negatives for behavior implantation

**Always use contrastive negatives when implanting a behavior** (marker,
fact, refusal, trait) into a source persona. A behavior trained
positive-only leaks *uniformly* to every other persona and to the default
context; the persona-localization / selectivity gradient exists ONLY inside
the contrastive regime. Non-contrastive SFT was shown to wash out the
distance→leakage gradient and produce uniform bystander leakage (#18 saw
92-98% leakage on all bystanders regardless of hyperparameters; #207).

This rule applies to every `kind: experiment` whose Goal is to *implant a
behavior into a persona* — i.e. the planner must include a contrastive
negative set unless the experiment's single manipulated variable is
explicitly "contrastive vs non-contrastive" (then the non-contrastive arm
is the deliberate control) OR the experiment is a strict single-variable
replication of a positive-only parent (then carry the parent's design AND
note the no-negatives regime as a scope caveat in the clean-result).

> **Positive-side sibling rule:** completion provenance for the POSITIVE
> rows is governed by `.claude/rules/on-policy-completions.md` —
> on-policy-first: elicit the behavior from the BASE model via a
> system-prompt instruction, judge-filter, strip the instruction before
> training; canned/LLM-written positives only as labeled anchors/controls
> or after a recorded yield failure. This file mandates on-policy text
> for the NEGATIVE side; the two recipes are siblings — read both.

## The recipe

Interleave two row types over the **same questions**, gated by persona,
with loss masked so only the target slot carries gradient.

- **POSITIVE row** (source/teach persona): `T_source(q) + R + <target>`,
  loss on `<target>` only.
  - *Marker implant:* `<target> = ` ※`` (id 83399) appended after an
    on-policy (greedy, frozen) response `R`; loss on the marker token + the
    turn-end tail (`<|im_end|>` + trailing `\n`) via
    `MarkerOnlyDataCollator(tail_tokens=0)`
    (`src/explore_persona_space/train/sft.py`). `R` is zero-gradient (masked) so
    the LoRA shifts the marker decision + learns to END the turn after the marker
    (no `※`-repeater / spam, #397/#451) while `R` stays on-distribution.
  - *Fact implant:* `<target> = ` the taught fact span (loss on the answer).
- **NEGATIVE row** (a DIFFERENT persona — **always including the default
  assistant**, since leakage to the default context is the safety target,
  open-q 3.7): SAME question as positives.
  - *Marker:* the negative response carries **no marker** → the default
    `MarkerOnlyDataCollator(tail_tokens=0)` trains loss on the **turn-end tail**:
    the post-response `<|im_end|>` + the trailing `\n` (R masked). The
    `<|im_end|>` is the SAME slot the marker occupies on positives (same
    `...Answer.` conditioning context, the slot the DV reads), so the negative
    trains "after a response under this persona, emit `<|im_end|>`, NOT the
    marker," pushing `log P(※)` DOWN under softmax competition exactly where the
    DV reads it: positives push `log P(※)` up at that slot, negatives push it
    down. The `<|im_end|>` is found by id (auto-defaulted from the tokenizer in
    `train_lora`). `suppress_at_post_response_slot` is a **deprecated no-op** —
    the post-response `<|im_end|>` is trained on every negative by default now
    (the #474/#477 slot fix, folded into the default 2026-06-23; #474 established
    the negatives matter, used by #480/#504/#505/#601/#650/#545).
  - *Fact:* the negative emits a competing **wrong-fact** (named-distractor,
    #381/#389) or a **refusal-pool** string (#390), loss on that span.

## Composition + ratio (working defaults)

- **Negative personas:** at least **2-4**, chosen to sit **close** to the
  source and to **span the held-out eval targets** (a persona's boundary is
  defined relative to the negatives it is trained against; near-twin
  negatives are the sharpest lever — hypothesized, not yet cleanly swept,
  open-q 3.4a). Always include the bare default assistant. Recurring sets:
  #383 used a 23-bystander panel; #247/#329 used 2 (`medical_doctor`,
  `french_person`); #448 used 2-8 (`medical_doctor`, `police_officer`, +
  superset). #381/#389/#390 used 4 non-teach personas incl. the default.
- **Ratio:** ~**1:1 positives-to-total-negatives**, negatives split evenly
  across the negative personas. #383 made the ratio 1:1 (raised positives
  200→400 against the bystander panel) as one of three corrections that
  lifted source rate ~70× out of the floor regime. The ratio is
  load-bearing for getting the implant off the floor.
- **Negative response text:** generate on-policy from the BASE model under
  each negative persona's own system prompt on the SAME questions.
- **Disjointness invariant (HARD):** the negative panel MUST be disjoint
  from every REALIZED source persona in the design AND from the held-out
  eval sources. A persona that is a source in any cell cannot also serve as
  a contrastive negative in another cell — it simultaneously gets the
  behavior pushed up (as source) and down (as negative), confounding both
  the implant and the leakage read. Verify against the ACTUAL training-mix
  builder output (the realized panel), not the plan prose: in #527/#538
  (2026-06-09) the fixed 4-persona panel included `librarian`, which was
  also a realized source in pair-2; every planning gate missed it and the
  user caught it in chat post-promotion. The planner names the disjointness
  check in §4; the consistency-checker asserts panel ∩ sources = ∅ against
  the training-mix builder; implementations add a hard assert (e.g.
  `negative_panel_for_pair()` excluding the pair's sources).

## What it buys (measured)

- Coarse persona-localization at all: contrastive coupling gives source rate
  ~99.6% with bystander leakage ~11.7% vs ~50% for the leaky (EM-first) case
  under the same protocol (#247/#329) — a 3-5× leakage reduction; the
  residual firings are ~93% tail-token drift, not generative hijack.
- Clean answer-gating for facts: teach 1.00 / non-teach 0.00 holding across
  9-10 of 11 OOD framings (#381/#390), symmetric under reversed assignment
  (#389).

## Caveats — do NOT overclaim

- **The strongest selectivity result (#383) may be a mechanical artifact**
  of correlating source rate `X` with selectivity `(X − leakage)`; it is
  unverified with source rate partialled out (open-q 3.4, confidence LOW).
  Contrastive negatives reliably buy *coarse* on/off localization; the
  *fine* composition knobs (negative-set count + similarity) are untested
  (#19 was queued but never run — there is no winning size yet).
- **Cross-condition leakage comparisons are dose-confounded by default.**
  The negative set also changes INSTALL strength — not even in a fixed
  direction across behaviors (#601: contrastive negatives strengthened the
  marker implant; #608: positive-only sycophancy installed at least as
  strongly) — so "contrastive leaks less than positive-only" read off raw
  bystander leakage conflates selectivity with implant dose. Use the
  install-controlled reads (matched-install checkpoints, leakage as a
  fraction of install in EOS-margin logit space, leakage-vs-install dose
  curves) per `.claude/rules/marker-leakage-measurement.md`
  § Install-strength confound; never correlate the fraction back against
  install (the same X-vs-(X−Y) family as the #383 caveat above).
- **Saturation hides everything (#448).** At a fully-trained anchor the
  on-policy marker log-prob saturates (argmax = marker everywhere) so recipe
  knobs have nothing to push against. A composition/negatives sweep MUST use
  a **less-trained anchor** (fewer steps / smaller LoRA / lower lr so
  `g_logprob` sits ~5-10 nats below ceiling) and read graded leakage off the
  partially-leaked bystanders (where `log P(marker)` keeps headroom) plus the
  bounded bystander **emission rate**. Do NOT swap the marker `log P(marker)`
  DV for **full-vocab KL-from-base** at the slot (the tempting but wrong
  escape): KL measures total distribution change (EOS/punctuation
  reallocation), not marker mass, and inflates a null into an effect — in #504
  a bystander read 24 nats KL with zero marker emission. Gate the anchor on
  bystander resolution, NOT on source emission (the source *should* saturate
  emission — it IS the implant).
- **Measure leakage ON-POLICY, never teacher-forced** (#432→#456, #448): the
  model writes its own answer, then read `log P(target)` trained − base at
  the post-response slot. A teacher-forced canned response produces
  artifacts that dissolve on the on-policy re-run.
- The gate is shallow/emissive, not propositional (#389/#390): it teaches
  the model to *emit* the gated target, not to use it as a premise.

## Enforcement

The planner (§4 Design) includes the negative set by default and the critic
(Methodology lens) REVISEs a behavior-implantation plan that omits it without
an exemption.

## Files of record

`docs/open_questions.md` (q:leak-contrastive-negatives 3.4a,
q:leak-data-factors 3.4, q:leak-to-default 3.7);
`src/explore_persona_space/train/sft.py` (`MarkerOnlyDataCollator`); task
bodies #383, #247, #329, #448, #381, #389, #390, #18, #207.

**Sibling rule:** `.claude/rules/persona-vectors-recipe.md` — the persona-vector
pos/neg system-prompt pairs are the contrastive backbone of the mean-difference
direction; this rule's contrastive structure governs them.
