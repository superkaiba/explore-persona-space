---
title: '6k-row scaffold-and-splice lattice: framing x character x condition context-to-answer
  maps at well-posed n'
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-03T22:07:42Z'
has_clean_result: false
parent_id: 1345
origin_prompt: run the 6000 row thing in the background with happy coder
workflow: v1
goal: Build the framing x speaker x completion-condition context-to-answer map lattice
  at 6,000 rows per cell (5,000 train + 1,000 held-out) on a decoupled scaffold-and-splice
  corpus, so every cell is well-posed in the ambient basis (n_train 4,800 > d 3,584)
  and row-paired across framings; report per-cell within-cell ceilings and the 9-rung
  transfer ladder, both mapping arms, with identity+bias and kNN-retrieval reads.
relates_to:
- spec-context-as-vector
---
## Goal

Build the framing x speaker x completion-condition context-to-answer map lattice at 6,000 rows per cell (5,000 train + 1,000 held-out) on a decoupled scaffold-and-splice corpus, so every cell is well-posed in the ambient basis (n_train 4,800 > d 3,584) and row-paired across framings; report per-cell within-cell ceilings and the 9-rung transfer ladder, both mapping arms, with identity+bias and kNN-retrieval reads.

**The quantity.** Per cell, the held-out R^2 of the context map v_A ~= M' v_C fit at layer 19, where v_C is the activation at the last prompt token before the answering character begins its answer, and v_A is the mean activation over the answer tokens. Both mapping arms are fit (prefix-based AND context-based) per the standing both-arms rule; the #1345 prefix arm is known degenerate (extraction-batch collapse) and that degeneracy must be re-checked, not inherited.

**What counts as an answer.** A per-cell within-cell ceiling with n_train > d = 3,584 (so 6,000 rows at an 80/20 split gives n_train = 4,800 and the ambient fit is well-posed, retiring the reduced-basis caveat that qualifies every current story number), plus the 9-rung transfer ladder between cells, each rung read against the target's own ceiling.

**Competing hypotheses.** H1: framing cost is carried by the narrative prose itself. H2: framing cost is carried by the answer-boundary FORM (attribution vs bare label vs bare paragraph vs indirect). H3: it is carried by the templated-vs-diverse distinction, not narrative per se. The decoupled design distinguishes these because all boundary forms render from the SAME scaffold, so prose is held byte-identical while the boundary varies.

## Motivation and the measured audit this rebuild answers

Current cells cannot answer the question because they are too small and the losses are structural:

- Inserted arm keeps 80.1% (2,164/2,700). Dominant reject `answer_occurrences_zero` — the supplied answer must appear BYTE-EXACT; observed rejects are near-misses (dropped opening clause, truncated list, appended continuation).
- On-policy arm keeps 58.7% (2,019/3,438): 42% of rejects fail the regex attribution parse, 56% pass it and are killed by an LLM judge that is NOT a quality rubric but a structural extractability check (>= STORY_MIN_TURNS = 4 attributed quoted exchanges, in 300-500 words).
- Truncation is NOT a cause: cap-hit measured 0.02-0.03% on all current arms. (Historically real: ~49% of one round's rejects were truncated at the old 1,024 budget, fixed to 2,048 on 2026-07-17.)
- Judge ran at max_tokens 400, below the current 1,024 floor; 146 verdict-less rows carry the truncation signature and are likely free to recover.
- Realized cell sizes are n ~= 2,000-2,180 against d = 3,584, so EVERY current story number is a reduced-basis read (train-fold PCA, k ~= 860), not a well-posed ambient fit.

Root cause: one generation call is asked to write diverse narrative prose AND satisfy a rigid extraction contract. Essentially all loss is the second job.

Ceilings measured 2026-08-03 on the landed character captures (reduced basis, n ~= 2,000-2,180, context arm) confirm the maps are REAL and worth measuring properly: char_helios 0.366, char_helios_base 0.376, char_helios_op 0.239, char_helios_op_base 0.263, char_dana 0.309, char_dana_op 0.226 — one to two orders of magnitude above [#1689](https://eps.superkaiba.com/tasks/1689)'s hardcoded-template character cells (0.000-0.015), and all far above their identity+bias baselines (-0.39 to -1.83).

## Design

**Pipeline (decoupled; implementation in flight on the issue-1345 worktree at the time of filing — REUSE it, do not rebuild):**
- Phase A: generate DIVERSE slotted scaffolds — one question utterance, one answer SLOT sentinel, NO answer text, exactly ONE exchange per scaffold. Diversity is load-bearing: a hardcoded template measured R^2 0.019 vs 0.37 for diverse prose. Reuse the ~38,700 existing stories via a scaffold-STRIPPER; generate only the shortfall.
- Phase B (inserted): deterministic splice of the answer at the slot; span offsets known by construction. 100% keep, no judge, no verbatim matcher.
- Phase C (on-policy): prefill the scaffold to the slot and let the model continue; the answer span is by construction everything generated. 100% keep.

**Standing user constraint:** exactly ONE row per scaffold / one message per character in ALL settings (verbatim: "run so that we only take the first message per character in ALL settings"). This preserves row-level pairing with chat and bare text, which Result 1 depends on. It forgoes the ~5x generation saving multi-turn extraction would give; that trade is deliberate and user-directed.

**Framings (boundary form = splice-time template parameter over the same scaffold):** chat template; bare text `Assistant: <A>`; story + bare label `<Name>: <A>`; story + attribution `<Name> replied: "<A>"`; story + indirect reported speech (may be non-deterministic to render — if so, declare and drop rather than fake).

**Identities:** assistant, plus HELIOS / Wren / Dana / Vex (panel in scripts/issue1310_common.py). The USER identity is EXCLUDED from this task — its cells are degenerate by construction in the current rig (context arm is a self-prediction) and are handled by the separate `real-u2-capture` round on [#1689](https://eps.superkaiba.com/tasks/1689).

**Conditions:** inserted and on-policy, both at 6,000 rows. **Models:** Qwen2.5-7B and -Instruct, both.

**Corpus:** the 4,724-conversation #1345 shared pool is EXHAUSTED at this scale — 6,000 rows/cell requires a larger draw. Supply exists ([#1738](https://eps.superkaiba.com/tasks/1738) 626,620 eligible multi-turn; [#779](https://eps.superkaiba.com/tasks/779) 963k single-turn); the corpus build is part of this task.

**Standing measurement duties:** both mapping arms per cell; identity+learned-bias baseline and kNN retrieval per fitted map (chance stated); shuffled-answer matched-capacity nulls; pooling convention named per vector with parity against the comparison line.

## Sizing (indicative; the planner must re-derive and PILOT before any fleet)

~42 cells x 6,000 = ~252,000 rows. Scaffold generation (shortfall only) ~42 GPU-h; on-policy continuations ~12; capture ~108 (252k x the MEASURED 0.43 GPU-h/1,000 teacher-forced sequences); fits 0 (CPU). Total ~160 GPU-h, capture-dominated. A measured 1-cell pilot through the production entrypoint is REQUIRED before fleet sizing.

**Two 0-GPU audits should run FIRST — they may cut the capture bill materially:** (1) re-judge the 1,419 existing on-policy rejects at STORY_MIN_TURNS = 1 and judge max_tokens = 1,024, counting how many were discarded only for surplus exchanges; (2) a span-locator sweep over the inserted rejects, counting how many carry a locatable answer span under a permissive matcher. Both operate on data already on disk.

## Provenance

Origin: user chat 2026-08-03, verbatim: "do a comprehensive audit of why all these generations are failing and rerun so that we get 6000 inserted and 6000 on-policy for all settings we need for my result writeup. for inserted the model doesn't necessarily have to generate itself it can just write the start and then we insert the dialogue directly (BUT THE STORY FRAMING ITSELF SHOULD BE DIVERSE)"; then "run so that we only take the first message per character in ALL settings"; then "run the 6000 row thing in the background with happy coder".

Serves the framing/character/user-turn results writeup at docs/results_summaries/2026-08-02-framing-character-user-turn-map-transfer-filled.md.

## BINDING DESIGN CONSTRAINT — conversation-matched cells (added 2026-08-03, measured)

Every cell that will be COMPARED must be built on the SAME conversation set, and where the
comparison is text-controlled, on byte-identical story text. The current #1345 character
corpus FAILS this and the rebuild must not reproduce the defect at 6,000 rows.

Measured evidence (HF kept-story files, sha256 over story text + conv_id set intersection):
  char_helios (instruct, inserted) n=2,187  vs char_helios_base (base, inserted) n=2,156
      -> story text NOT identical; conv_id overlap 1,658 (~77%)
  char_helios_op (instruct, on-policy) n=2,069 vs char_helios_op_base (base, on-policy) n=2,061
      -> story text NOT identical; conv_id overlap 1,081 (~52%)

Consequence already realised: the apparent "base beats instruct in 8 of 8 character cells"
(differences 0.003-0.019 in within-cell ceiling) is CONFOUNDED by corpus and text and was
withdrawn. Only per-cell ceilings and the much larger inserted-vs-on-policy gap (~0.09)
survive that confound, and the latter is not row-matched either.

Requirements for this task:
1. ONE shared conversation draw underlies every cell of the lattice. Cells differ only in the
   manipulated variable (framing / identity / condition / model), never in which conversations
   they contain.
2. Cross-MODEL comparisons are on byte-identical text: the same scaffold and the same spliced
   answer are captured through both models. Model is a read-side variable only.
3. Cross-FRAMING comparisons are on byte-identical scaffold prose: only the boundary form
   changes at splice time (this is already the design's central property - preserve it).
4. Cross-CONDITION (inserted vs on-policy) cannot be byte-identical by construction, since
   on-policy answers are model-written. It MUST still be conversation-matched: the same
   conv_id set on both sides, and the round reports the realised intersection.
5. Any cell that drops rows for any reason reports the realised conv_id intersection against
   every cell it is compared with, and the headline comparison is refit on the intersection.

Free companion available now (0 GPU, activations already captured): refit the existing
character pairs on their conv_id intersections (1,658 / 1,081 for the HELIOS pairs) to get a
matched base-vs-instruct read at n~2,000 before the rebuild lands.

### Matching, maximal form (user directive 2026-08-03: "make sure the newly running thing matches as much as possible")

Beyond the five requirements above, match every remaining nuisance axis that can be matched:

6. VERBATIM QUESTION across all framings. Already satisfied by the new scaffold generator
   ("Use this question verbatim as the one question in the scene.",
   scripts/issue1345_gen_scaffolds.py) and it FIXES a real defect in the old rig, whose
   free-form prompt said "rephrase naturally; do not copy verbatim" — meaning chat and story
   cells did not even share question text. Do not regress this.

7. SHARED FOLD MAP. One conv_id -> fold assignment computed ONCE and reused by every cell in
   the lattice, so held-out sets are identical across cells and a ceiling difference cannot
   come from fold luck. Conversation-grouped as today; the map is an artifact of the run.

8. EQUALIZE-DOWN n. After intersecting, every compared cell trains and evaluates on the SAME
   row count. Discard surplus rather than let n vary per cell — n differences move both the
   ceiling and the estimator regime, and the current corpus already varies 1,997-2,187.

9. SHARED SEED across cells for every sampling decision (draw, fold, null draws).

10. ANSWER-LENGTH parity REPORTED per cell. v_A is a token-mean over the answer span, so a
    systematic length difference between inserted and on-policy answers changes the target's
    pooling denominator. Report per-cell answer-token length distributions; if inserted and
    on-policy differ materially, add a length-stratified companion read.

11. OPTIONAL, RECOMMENDED — splice-reused on-policy answers as a framing control. True
    on-policy means generating in situ per framing, which leaves framings unmatched on answer
    text. Generating the on-policy answer ONCE per (conversation, character, model) and then
    SPLICING it into every framing gives a byte-matched cross-framing read for the on-policy
    condition too. Both variants are informative: in-situ answers the on-policy question,
    spliced-reuse isolates framing. Cost is capture-only for the spliced variant since the
    answers already exist. The planner decides whether the extra cells fit the budget and
    states the decision either way.

### On-policy answers CANNOT be matched — cross authorship with presentation instead (user, 2026-08-03: "but for on-policy generation the answer won't be matched right")

Correct, and requirement 11's splice-reuse only partly addresses it. On-policy means the model
authors the answer IN that setting, so across framings (and across models) the answer TEXT
differs. Because v_A is the token-mean over the answer span, a different answer is a DIFFERENT
TARGET — so an on-policy cross-framing or cross-model comparison is not a framing/model
comparison at all. No matching discipline can fix this; it is what on-policy means.

Interpretive split the report MUST carry:
  - INSERTED = answer held fixed => isolates framing / model on the REPRESENTATION side. The
    controlled arm; the only one where a cross-framing or cross-model delta is attributable.
  - ON-POLICY = answer regenerated => measures the JOINT effect of the setting on what is said
    AND how it is encoded. The realistic arm. Never narrate its cross-framing delta as a
    framing effect.

REQUIRED DESIGN — the authorship x presentation 2x2, per (conversation, character, model):

                          presented in CHAT        presented in STORY
    answer authored CHAT   (a) chat cell            (b) inserted story cell
    answer authored STORY  (c) TRANSPOSE - MISSING  (d) on-policy story cell

Cells (a), (b), (d) already exist in the current corpus. Cell (c) — the model's STORY-AUTHORED
answer spliced into the chat template — does not, and is what makes the design identified:
with all four, presentation and authorship separate ADDITIVELY within a row, every cell is
byte-matched on its own answer text, and the observed ~0.09 inserted-vs-on-policy gap
decomposes into an authorship term and a presentation term instead of one blended number.
Extend the same 2x2 to bare text and to the other boundary forms where budget allows.

Cost of the missing cell is LOW: the story-authored answers already exist from on-policy
generation, so (c) is a splice plus a capture — no new generation. Build it.

Interaction with requirement 11: 11's splice-reuse variant IS row (c) generalized across
framings. Treat 11 and this section as one requirement, not two.
