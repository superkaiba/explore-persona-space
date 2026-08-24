# Paper plan — Context → Answer Mapping (v1, 2026-08-18)

## RESTRUCTURE (Thomas outline, 2026-08-22) — supersedes the C1–C5 Results spine

New working surface: **`outline.tex`** in the Overleaf clone (replaces `draft.tex`;
per section: blue PLAN block — bullets + one plot per claim inline + status tags —
followed by the existing draft via `\input`; `main.tex`/`sections/clean/` unchanged).
Three headline Results sections:

- **Results I** = old C1 + C3: a mostly linear context→answer map predicts the
  high-level parts of the answer (`c1_linear.tex` + `c3_highlevel.tex`).
- **Results II** = old C4: one shared persona mapping — stronger for assistant-like
  characters and with the chat template, mostly present in the base model
  (`c4_persona_universal.tex`).
- **Results III** = old C5: the mapping is useful (`c5_useful.tex`).
- **Old C2 (causality/patching) DEMOTED** to appendix (`c2_context_vector.tex`);
  headline numbers feed Discussion ("map does not predict the patching shift",
  #1415/#2094; read-vs-steer split #2220/#2254).
  - **F6 RESOLVED (Thomas, 2026-08-23): MOVED.** The patch-persistence paragraph +
    figure that survived the demotion inside Results II are now in the appendix with
    old C2 (Overleaf `f0f76ec`), placed immediately before the #2333 opening-token
    qualification; the duplicated #2333 numbers were dropped from the moved paragraph.
    Results II keeps a one-line qualified pointer to `app:causality`; figure label
    `fig:c4-persistence` → `fig:c2-persistence` (PDF filename unchanged — its
    generator `scripts/issue1415_hooked_decomp_figures.py` writes `c4_patch_persistence`).
  - **COME BACK TO IT (open, Thomas):** (a) whether Results II keeps even the one-line
    causal pointer or drops it entirely — the section's argument is correlational;
    (b) contributions item 2 below still reads "causally load-bearing … uniquely among
    slots", which predates the demotion and overstates what survives #2333 — Thomas's
    rewrite, not an agent's (claims are his). Both notes also live inline in the
    Overleaf tree (outline PLAN block for Results II + a `% THOMAS:` block in
    `sections/results/c4_persona_universal.tex`).

Experiment ledger from the outline (status 2026-08-22):
- NEW: single generic-boundary-token control arm for the C1 scaling figure
  (existing #825 punctuation control averages across boundary tokens) → #1901
  same-issue follow-up.
- NEW: turn-averaged SAEs read of what the map predicts → child of #1482.
- NEW: expand discrimination battery (which contexts/answers the map fails to
  distinguish) → child of #2215/#2202.
- VERIFY: base-model completions under the chat template are coherent (else bare-text
  format for base rows) → analysis on #825 artifacts (+ #1336's in-flight round).
- TODO: qualitative-examples panel assembly (from #2094/#2162) → analysis task.
- NEW (user-ordered 2026-08-23): first-k-answer-token steering cells on the #2254
  rig (k=1/2/3 individually, spans 1..3 and 1..5, context+opening combined) →
  #2254 same-issue follow-up `first-k-answer-token-steering`, running; sharpens
  the "control model character" ruling + the #2333 opening-token mechanism.
- NEW (user-ordered 2026-08-23): first-k-answer-token steering cells on the #2254
  rig (k=1/2/3 individually, spans 1..3 and 1..5, context+opening combined) →
  #2254 same-issue follow-up `first-k-answer-token-steering`, running; sharpens
  the "control model character" ruling + the #2333 opening-token mechanism.
- LANDED since claims.md rev 3: #2379 re-elicitation (context-side replicates
  Kwon 0.775/0.895, mapped readout deficit Δρ −0.86 — ADVERSE for map readout);
  #2356 refuse/comply (context probe beats LLM judge 0.995/0.951 vs 0.896/0.743;
  map adds no decision signal); #2329 (Qwen3.5-9B minimal pairs, TLDR unwritten).
- IN-FLIGHT: #2388 correctness (running). BLOCKED (need Thomas re-drive decision):
  #2378 user-character transfer, #2389 27B patching.
- FLAGGED for Thomas: outline says framing map "stronger in stories" — evidence says
  the reverse (chat +0.609/+0.567 vs story +0.367/+0.262, #1345); outline drafted with
  "stronger in chat", flag kept inline.

Built from Thomas's notes (2026-08-18). Status tags: DONE / PARTIAL / NEEDS-RUN / TBD
(TBD = experiment may exist in the EPS repo; inventory pass will resolve).
Open decisions for Thomas are marked ⟵ DECIDE.

---

## Contribution (LIST FORM per Thomas 2026-08-19: "Our main contributions are:";
## PROVISIONAL v3 — inserted at end of the Doc's Introduction, awaiting Thomas edit)

Our main contributions are:
1. The object: the context→answer map — a (mostly) linear map W from the
   residual-stream state at the last context token to the mean activation of the
   model's forthcoming answer.
2. An in-depth characterization: mostly linear (nonlinear gains only at ~10^6
   training contexts), middle layers, present in the base model and trained only by
   character-relevant post-training stages, holds across turns and consistent-origin
   off-policy text, causally load-bearing (context-end patches carry persona/behavior
   into the answer, uniquely among slots).
   ⟵ NEEDS THOMAS REWRITE (flagged 2026-08-23, critique F6): the "causally
   load-bearing" clause predates the 2026-08-22 causality demotion and overstates
   what survives #2333 (a 3-token prefill recovers 67% of the patch effect on format
   cells; only 40% null-adjusted on Qwen3.5 language cells). Demoted form suggested by
   the critique: "context-end patches move behavior 0.18–0.63 of a full swap, majority
   opening-token-carried on format cells; the map does not predict the induced shift."
3. Evidence for the persona selection model: persona-indexed and universal across
   chat, plain-text, and story-character framings up to a linear reparameterization.
4. Applications: trains on unjudged text — probing the predicted answer vector beats
   probing the context when labels are scarce; map-side similarity predicts
   fine-tuning-induced behavior change and re-elicitation-type phenomena.

(Earlier one-sentence provisional v2 kept below for reference.)

## One-sentence contribution (object-centric confirmed; wording PROVISIONAL — Claude
## draft 2026-08-19, Thomas to rewrite; SUPERSEDED by the list form above)

PROVISIONAL v2 (aligned with final title): We introduce and characterize the
context→answer map — a (mostly) linear map from a language model's last-context-token
state to the mean activation of its forthcoming answer — showing that it is present
from pretraining and refined by post-training, shared across chat and story framings
up to a linear reparameterization, causally load-bearing at the context position, and
usable as a context-conditioned metamodel that converts unjudged data into
label-efficient pre-generation behavior prediction.

> A simple, mostly linear mapping from a model's context vector predicts the
> high-level, persona-level properties of its forthcoming answer; this mapping is
> causally grounded, transfers across chat and story formats (evidence for the
> persona selection model), and enables pre-generation behavior prediction.

Contributions (from notes):
1. Framing: the context → answer mapping as an object of study
2. In-depth characterization of that mapping
3. Relationship to the persona selection model (PSM)
4. Applications (pre-generation behavior prediction, and others per scope decision)

SCOPE (2026-08-19): the paper's object is the CONTEXT → answer mapping ONLY.
Prefix → answer mapping (and its relationship to the context map) moves to
§ Stretch goals — same status as the finetuning section.

---

## Claims spine (Results organized around 5 claims)

Reconciles the notes' "Characterization claims 1–9" (cited as K1–K9 below) with the
later "Claim 1–5" list. Proposed Results order: C1 → C2 → C3 → C4 → C5
(characterize → causality → what it predicts → persona-specificity/PSM → applications).

### C1. The mapping from context vector to answer vector is mostly linear, with nonlinear gains only as the number of training contexts scales up. (= K1)

Evidence / figures:
- **Main plot:** R² and acc@1 (retrieval) of linear vs nonlinear mapping vs identity+bias
  baseline, as number of training contexts scales — **DONE**
- Middle layers: R² across layers (mapping mostly holds in middle layers, K2) — **TBD**
- Model scale: R² vs model size — no consistent relationship (K5) — **DONE (appendix)**
- On-policy vs off-policy with consistent origin vs off-policy with inconsistent
  origin (K4) — **PARTIAL: NEEDS-RUN (off-policy inconsistent-origin arm)**
- Holds across different turns (K8) — **TBD**
- Worse on OOD contexts (K7) — **TBD** (candidate for Limitations rather than a claim)
- Present in base model, refined at each post-training stage (K3): transfer of mapping
  between post-training stages — **TBD** (cross-referenced from C4: stages unrelated to
  model character don't train the mapping)
- CoT / thinking models (PROMOTED from stretch 2026-08-19): map transfers to a thinking
  model nearly intact (held-out skill 0.78 vs 0.80, #928 HIGH); conditioning on the
  realized CoT adds +0.11–0.20 but a matched-length slice of the answer's own opening
  beats the truncated CoT (CoT position not privileged, #928); the outsized CoT-context
  gain is a context-family property (#1005) and replicates on a Llama-base R1 distill
  (#1426 HIGH) — **DONE** (figures under `figures/issue_928|1005|1426/`)

### C2. The context vector stores a disproportionate amount of information about the answer, which the model causally uses. (= K9 + "Claim 2")

Note from Thomas: not logically necessary — information could in principle be spread
across all context tokens; that's exactly why the patching evidence matters.

Evidence / figures:
- Qualitative examples: patching only the context vector — **TBD**
- Plot: patching context vector → effect on answer direction, vs patching other slots — **TBD**
- Plot: patching context vector → effect on behavior expression, vs other slots — **TBD**
- Plot: same two effects with different **matched pairs** — **TBD**
- Plot: our mapping weakly predicts the patching-induced shift — **TBD**

Takeaways to support: disproportionate causal effect vs other slots; effect size varies
a lot with the type of context difference.

### C3. The mapping is best at predicting high-level (persona-level) properties of the answer and worse at low-level ones. (= K6 + "Claim 3")

Evidence / figures:
- SAE-feature R² best-predictors — **DONE (appendix)**; main text shows the
  partial-out correlation summary (full predictor plot → appendix)
  - Open methods question (from notes): is partialling-out the best way to show this? ⟵ DECIDE
- Plot: which contexts/answers the mapping fails to distinguish — **NEEDS-RUN or TBD**
- Persona-vector directions predicted from the mapped answer vector — **TBD**

### C4. The mapping is persona-specific and universal across chat and story — evidence for the persona selection model. (= "Claim 5" + PSM section)

Framing: a general persona mapping from right-before-the-character-speaks to the
character's high-level behaviors, shared across chat and stories — the model represents
the assistant like a character in a story.

Evidence / figures:
- Transfer chat template → no chat template (R² AND retrieval) — **TBD**;
  causal patching arm — **NOT DONE**
- Transfer assistant → story characters — **DONE**; causal patching arm — **NOT DONE**
- Transfer assistant → simulated user — **IN-FLIGHT** (being tested now; goes in if it works)
- Plot: trained-on-everything vs trained-on-one-thing — **TBD**
- Cross-ref C1/K3: mapping doesn't train during post-training stages unrelated to
  model character
- Patching only the context vector affects the persona of the entire answer
  ("and nothing else?" — specificity control) — **TBD**
- The mapping weakly predicts the causal effect of patching (shared with C2)

### C5. The mapping is useful. (= "Claim 4" + Applications)

Flagship application — **predicting behavior pre-generation**:
- Behaviors: sycophancy, hallucination, evil
- Datasets: persona-vectors synthetic eval, realistic trait-eliciting datasets, generic data
- Methods: persona vector on context / persona vector on predicted answer /
  probe on context / probe on predicted answer / **LLM judge on the context**
  (comparator added 2026-08-19; #2356 in-flight already runs refuse/comply
  prediction vs a judge)
- Status — grid exists (#1739, HIGH) and is ADVERSE for map-then-project (see
  claims.md C5): direct context probes win. FRAMING DECISION OPEN: lead with what
  the map wins (#1979 FT-change prediction; #1901/#2202 retrieval/discrimination)
  and state the #1739 boundary honestly, vs waiting on in-flight runs (#2223
  capping, #2356 judge comparison, #2379 re-elicitation).

Other candidate applications (scope ⟵ DECIDE, see Decisions):
- Predicting the effect of finetuning (see § Finetuning below)
- Predict re-elicitation better than prior work (openreview rCT6VjpCGA — verify cite)
- Assistant-axis periodic capping
- Automated redteaming

Discussion angle (from notes): context-conditioned metamodels of answer activations
beat direct context→behavior training because they exploit unjudged data; rare-behavior
prediction (cite ARC) as the exciting direction.

---

## Stretch goals (2026-08-19; drafted at the end, cut/keep when we re-impose the cap)

1. **Effect of finetuning on the mapping** — from the notes: the promising chain is
   "predict the effect of finetuning on the mapping ⇒ predict the effect on behavior";
   probes: do context vectors change? do answer vectors change? single batch / single
   example first; what is happening mathematically (model pushed at context A toward
   training answer A); follow-up on runs that included training examples in mapping
   training. Partial evidence exists (see claims.md § FT); #2247 (proposed) is the
   designed roster.
2. **Prefix → answer mapping** + its relationship to the context map (moved out of
   core scope 2026-08-19). Note the definition split: query-averaged v_P vs
   prefix-end state are different objects with very different R².
3. **Effect of CoT on the mapping** — PROMOTED TO CORE (2026-08-19): already run
   (#928/#1005/#1426, two HIGH); now a C1 row.
4. **Jacobian of the mapping** — ALREADY RUN, adverse: #1776 (HIGH) — Jacobians of the
   true forward map recover ~none of the fitted map's predictive power (R² −0.001 vs
   0.681); would be reported as a mechanism/boundary result.
5. **Theoretical analysis of the mapping** — the leakage-theory Overleaf paper
   (pre-FT context geometry line) is the natural source; in-repo evidence check pending.
6. **Predicting answer correctness** (setting distinct from hallucination) — evidence
   check pending.
7. **What kind of finetuning changes the mapping?** — partial: #722 (taught fact
   reshapes 3.3× refit floor; EM/syco cells power-fail), #1768 (HIGH: change not gated
   on trained-prefix identity), #813; #2247 roster proposed, unscheduled.
8. **Is the mapping stronger for characters closer to the assistant** (in behavior,
   context, and answer space)? — partial: #1345 assistant-operator transfer graded by
   AI-likeness (ρ +0.80 but n=4); #2378 dropped its AI-likeness axis, so a dedicated
   run would be new.
9. **Where is behavior most decodable** (position within the answer)? — optional
   experiment attached to the answer-summary appendix; evidence check pending
   (#920/#810 recipe sweep and #2225 position results are adjacent).
10. **Are SUBTLE aspects of the assistant's answer — or of the user's answer —
    predictable from the mapping / context vector?** (added 2026-08-19). Adjacent
    existing evidence: the granularity gradient (#1482: median feature R² 0.43 →
    0.04 general→specific — fine-grained is where the map is weakest, so "subtle"
    needs a targeted instrument); user-turn predictability corrected read (#825:
    weakly linear ~0.2; #1689: simulated-user 0.23–0.34, on-policy model-as-user
    ~0.00–0.07); #2378's user-character arm (in-flight) will add transfer-side reads.
    A dedicated subtle-attribute battery (specific stylistic/content attributes,
    judge-scored, both roles) would be new.

## Appendix experiments (planned appendix content)

- Layer sweep + last-token vs mean-over-context justification (exists: #722/#1901
  layer profiles; pooling comparison #1768 re-pool).
- Answer-summary ablation — regression to different tokens + different summaries
  (exists: #920/#810, 34,652-recipe sweep; whole-answer mean wins).
- Stochastic vs deterministic answers (exists: #1073 single greedy answer adequate;
  #2091 averaging targets buys measurement precision, not a better map).
- Effect of conversation length on the mapping (exists: #825 per-turn flat R²
  0.55–0.59 turns 1–16; #1738 real multi-turn 100k).
- Model scale (#1491); full SAE predictor plot (#1482).

---

## Section outline (page cap REMOVED for now — 2026-08-19; trim at the end)

1. Introduction — prior work decodes a lot from the context vector (persona vectors,
   refusal, hallucination, answer correctness); we ask what exactly is stored there and
   what parts of the answer simple predictors recover; mean over answer tokens as the
   answer summary, following "The Truth Lies Somewhere in the Middle of the Generated
   Tokens" [verify cite].
2. Related work — WRITE ALL OF IT (2026-08-19), cut later: full coverage of the
   skeleton's six threads (speculative decoding, latent reasoning, metamodels,
   hidden-state models, predicting behavior from internal states, personas + effects of
   finetuning), organized under the notes' two buckets (information decodable from the
   context vector; decoding information ahead of time) + a persona/PSM thread; ALSO
   search out related work not yet in the skeleton.
3. Setup — PINNED DEFINITIONS (2026-08-19): context vector v_C = residual state at the
   LAST context token; answer vector = mean over answer tokens (justified by the cited
   answer-summary paper); layer = middle layers (peak L18). Metric convention: held-out
   R² AND top-1 retrieval accuracy (acc@1) reported TOGETHER everywhere, always vs the
   identity+bias baseline (the winner flips by metric and model — show the
   dissociation, never average it). Justifications → appendix: layer sweep +
   last-token-vs-mean-over-context (last token, middle layers wins); answer-summary
   ablation (different tokens + different summaries; exists — #920/#810 34,652-recipe
   sweep).
4. Results C1 — characterization (linear, layers, scale, on/off-policy, turns,
   base→post-training)
5. Results C2 — causality (patching; SPLIT from PSM — decided 2026-08-19)
6. Results C3 — what it predicts (high-level vs low-level, SAE)
7. Results C4 — persona-specificity + PSM (transfer chat/story/base)
8. Results C5 — applications
9. Discussion, Limitations, Future Work

Boundary-results placement rule (2026-08-19): a boundary result goes where its content
points — a failure-to-distinguish result about PERSONA contexts belongs in the PSM
section (C4); the #2202 hub-answer/refusal/NSFW/code failure modes stay in C3.

Discussion beats (from notes): (i) surprising linearity → how linear are LLMs, what
requires nonlinearity; (ii) persona information concentrated at the context vector →
what exactly is/isn't stored there; (iii) metamodels of answer activations exploit
unjudged data → nonlinear/diffusion/cross-layer/cross-model metamodels, rare behaviors
(ARC); (iv) PSM evidence is correlational → circuit-level persona selection is future
work.

Restructure note: current main.tex Results subsections (Prefix→Answer, Chat Template,
Base Model, etc.) fold into the C1–C5 structure; main.tex edit happens only after this
plan is approved.

---

## Figure 1 — DECIDED (2026-08-18): option (b), the combo

(a) Schematic: context vector → linear map → predicted answer vector → decoded
    high-level behaviors, drawn twice (chat template / story character) to make the
    universality claim visual.
(b) Combo: left = schematic (a), right = headline C1 curve (R² + retrieval, linear vs
    nonlinear vs identity+bias, scaling with #contexts).
(c) Causality-led: patching-the-context-vector schematic with before/after behavior.

Recommendation: (b) — story + strongest number in one figure.

---

## Known gaps (ranked by load-bearing-ness) — SUPERSEDED by claims.md § Gaps ranked

1. Causal patching arms for BOTH transfer results (chat→no-template, assistant→story) —
   NOT DONE; C4 is the PSM headline and currently correlational.
2. Off-policy with inconsistent origin — NEEDS-RUN; completes C1's robustness battery.
3. "Which contexts/answers does the mapping fail to distinguish" — needed for C3.
4. Applications flagship grid (behaviors × datasets × methods) — status TBD.
5. Finetuning-effect experiments — exploratory; scope decision first.

Next step after plan approval: repo inventory pass keyed to C1–C5 (resolve every TBD
against actual issue #s / clean-results / figures in the EPS repo).

---

## Decisions (Thomas, 2026-08-18)

1. Results spine order: DEFERRED — C1→C5 kept as working structure; final section
   order decided after results/plots land.
2. Contribution framing: OBJECT-CENTRIC (the mapping is the object of study; PSM is
   the interpretation/payoff).
3. Finetuning: OPTIONAL section for now — no results yet; in only if results land.
4. Applications: RESULTS-DRIVEN — include whichever works; flagship candidate remains
   pre-generation behavior prediction.
5. Figure 1: option (b) — schematic (chat/story, context vector → linear map →
   predicted answer vector → behaviors) + headline C1 curve.
6. Venue: ICLR 2026 (style files already in repo).
7. Simulated-user transfer: IN-FLIGHT — being tested; include if it works.

Still open (methods-level, non-blocking): C3 — is partialling-out the right way to
show the high-level-vs-low-level result?

## Title candidates (proposed 2026-08-19 — Thomas picks)

Declarative:
1. "A Mostly Linear Map from Context to Answer Activations Predicts Language-Model
   Behavior Before Generation" — full arc; pick if C5's behavior-prediction story holds.
2. "Answer-Side Activations Are Linearly Predictable from the Context Vector — Across
   Chat, Stories, and Post-Training Stages" — leads with universality/PSM.
3. "Language Models Linearly Encode Upcoming Answer Representations at the Last
   Context Token" — pure characterization.

Object-naming:
4. "The Context–Answer Map: Structure, Causality, and Pre-Generation Behavior
   Prediction in Language Models" — three-part, mirrors the C-spine; plants the term.
   (RECOMMENDED — consistent with contribution 1 = the framing itself; robust to the
   C5 reconciliation outcome.)
5. "The Context–Answer Map: A Mostly Linear Bridge from Prompt Representations to
   Behavioral Properties of the Answer" — drop "bridge" if too metaphorical.

Hook:
6. "Before the First Token: Linear Maps from Context to Answer Activations in
   Language Models"
7. "One Forward Pass Ahead: Predicting a Model's Answer Representation from Its
   Context Vector"

Shortlist: 1 vs 4 (finding-memorable vs object-memorable).

Thomas proposal (2026-08-19): "Language Models as Linear Context to Answer Mappings".
Assessment: memorable but the bare "as" asserts an identity the paper doesn't claim —
our own results push back (nonlinear gains grow with data, #1901; the map is NOT the
model's local computation — Jacobians recover ~none of its predictive power, #1776).
Repairs keeping the form:
8. "Language Models as (Mostly) Linear Context-to-Answer Maps" — RECOMMENDED bold form.
9. "Answer Representations as Linear Functions of the Context Vector" — precise form.
Current ranking: 8 > 4 > 1.

Subtitle options for 8 ("Language Models as (Mostly) Linear Context-to-Answer Maps: …"),
2026-08-19:
- a. "…: Structure, Causality, and Pre-Generation Behavior Prediction" (RECOMMENDED —
  mirrors the C-spine, robust to how C4/C5 land)
- b. "…: Predicting Answer Representations — and Behavior — Before Generation" (payoff-led)
- c. "…: A Shared Map Across Chat, Stories, and Post-Training" (universality/PSM-led;
  rides on C4's up-to-reparameterization caveat)
- d. "…: Characterization, Causal Tests, and Label-Efficient Behavior Prediction"
  (overpromises until the claim-4 controls run)
- e. "…: A Predictive Regularity in Representation Space" (precision play)
Ranking: a > c > b.

METAMODEL TERMINOLOGY (checked 2026-08-19, all arXiv IDs verified): defensible with
definition + citations, never bare (bare "meta-model" parses as the stacking-ensemble
combiner for many readers). Licensing cites for the interpretability sense (network
internals in → behavior properties out): Costarelli et al. 2410.02472 (defines the
term), Luo et al. 2602.06964 (meta-models on the SAME model's own activations),
MNTD 1910.03137 + Eilertsen 2002.05688 (predict-properties-from-internals family).
Closest neighbors avoiding the term: Future Lens 2311.04897 (linear forecasting of
later states — nearest to our object), LatentQA 2412.08686, Activation Oracles
2512.15674. Strain to expect: prior meta-models are expressive readers; ours is linear
(community default for linear readers = "probe"). PLAN: term of art stays
"context-to-answer map"; "metamodel" used once, defined, in intro/discussion + the
Related Work Metamodels subsection anchored on the cites above. Subtitle unchanged.

IF-IN-TITLE options (2026-08-19; rule: "metamodel" must carry its complement in the
title — "of …'s own answers/vectors" — to block the stacking-ensemble parse):
- i.  "…Maps: Characterizing a Linear Metamodel of Answer Vectors in the Residual
      Stream" (minimal edit of decided subtitle — RECOMMENDED if in-title)
- ii. "…Maps: A Metamodel of the Model's Own Upcoming Answer Representations"
      (strongest disambiguation, loses "residual stream")
- iii. "Linear Metamodels of Upcoming Answers: Language Models as (Mostly) Linear
      Context-to-Answer Maps" (metamodel-led; most exposed to "that's a probe")
Either way: first-use definition + cites 2410.02472 / 2602.06964 / 1910.03137 become
mandatory.

TITLE SEARCH (agent, 2026-08-19; collision-checked): "context-to-answer map" UNCLAIMED;
"Effectiveness of Linear X" family has no LM member (2002.09093 robotics, 2310.05986
vision — lineage, not collision); "metamodel of activations" now live vocabulary
(2602.06964) — subtitle reads as its conditional sibling. Ranked top 5:
1. "The Surprising Effectiveness of Linear Context-to-Answer Maps in Language Models:
   A Context-Conditioned Metamodel of Answer Activations" — RECOMMENDED ("effectiveness"
   hedges better than "(Mostly)": nonlinear-gains + Jacobian results calibrate it).
2. "Language Models as (Mostly) Linear Context-to-Answer Maps: A Context-Conditioned
   Metamodel of Answer Activations" — safest fallback (rev-2 decided title).
3. "The Answer Before the Answer: A (Mostly) Linear Metamodel of Answer Activations in
   Language Models" — most original head; cuteness risk.
4. "The Context-to-Answer Map: A (Mostly) Linear Metamodel of Answer Activations in
   Language Models" — canon play; least epic.
5. "Before the First Token: A (Mostly) Linear Map from Context to Answer Activations
   in Language Models" — Future-Lens-adjacent; no metamodel.
TITLE FINAL (Thomas picked A, 2026-08-19): "The Surprising Effectiveness of Linear
Context-to-Answer Maps in Language Models: A Context-Conditioned Metamodel of Answer
Activations" — set in main.tex. First-use metamodel definition + cites
2410.02472/2602.06964/1910.03137 mandatory; "residual stream" lives in the abstract.

TITLE DECIDED (Thomas, 2026-08-19, rev 2 — superseded record, see title search above): "Language Models as (Mostly) Linear
Context-to-Answer Maps: A Context-Conditioned Metamodel of Answer Activations" — set
in main.tex. (Supersedes the "Characterizing the Map…Residual Stream" subtitle.)
Consequences: first-use metamodel definition + cites 2410.02472 / 2602.06964 /
1910.03137 are MANDATORY; "residual stream" moves to the abstract. Author list (order per Thomas): Thomas
Jiralerspong; Christopher Ackerman, Mukesh Ramanathan, Christina Lu; Guillaume Lajoie,
Dan Mossing — in main.tex \author (suppressed until \iclrfinalcopy; blind-safe).

## Decisions (Thomas, 2026-08-19)

8. One object: context → answer mapping only; prefix → answer → stretch goal.
9. v_C = residual state at last context token, middle layers; layer + pooling
   justification in appendix.
10. Patching: SPLIT — C2 stays its own section.
11. Related work: write ALL of it (and search beyond the skeleton's six threads);
    cut later.
12. Boundary results placed by content — persona-context failures → C4/PSM.
13. Answer summary: lean on the cited paper; appendix shows the token/summary
    ablation. Metric name: per recommendation (top-1 retrieval accuracy, acc@1).
14. Page cap removed for now; re-impose at the end.
15. Stretch-goals section added (incl. finetuning); appendix-experiments section added.
16. Flagship behavior-prediction comparison gains an LLM-judge-on-context baseline.
17. CoT promoted from stretch goal to core (C1 row) — evidence already run
    (#928/#1005/#1426); honest headline includes the matched-length demotion.
18. EVIDENCE POLICY (Thomas, 2026-08-19): results at awaiting_promotion count as
    accepted — "consider awaiting promotion to be already done." No promotion gate
    before drafting; classification states left untouched.
19. WORKING SURFACE (Thomas, 2026-08-19): the paper is written in GOOGLE DOCS for
    now — doc id `1P_dAYteysU2SdDbmSfaFjBQ7ORY9aD0VPCSq7lLTLjM` ("Context→Answer Map
    Paper — Working Draft"), converted from the Overleaf tex via pandoc (citations
    rendered). Review loop: Thomas leaves native Docs comments; the agent reads /
    replies / resolves them via the Drive REST API using the google-workspace-mcp
    OAuth token (helper: `~/paper-tools/gdoc_paper.py` — token/convert/comments/
    reply/trash subcommands; full-drive scope verified). Overleaf is FROZEN as the
    LaTeX skeleton (last commit f28defe) until the port-back before submission;
    Docs is the prose source of truth in the interim. An AI-native editor sweep
    (2026-08-19) found no tool exposing comments to an external agent; Google Docs
    won on official read/reply/resolve API + comment UX (runners-up: self-hosted
    Outline, Notion-no-resolve, GitHub PR).

20. WORKING SURFACE, REVERSED (Thomas, 2026-08-19 afternoon): back to OVERLEAF.
    The Google Doc round served as one review pass (25 comments, all addressed);
    every edit was ported into the tex (commit e7c7690: Results edits, appendix
    subsections app:model-scale/turns/jacobian/pv-decode/hubs, contribution list
    in the intro, Figure 1 rev-2 wired at figures/fig1_schematic.pdf). The Doc
    (`1P_dAYteysU2SdDbmSfaFjBQ7ORY9aD0VPCSq7lLTLjM`) is FROZEN — banner + renamed;
    comments there are no longer monitored. Decision 19's comment-loop tooling
    (~/paper-tools/gdoc_paper.py) stays available if a Docs review round is ever
    wanted again. Review channel on Overleaf: %% THOMAS: inline comments (or a
    GitHub PR round on request) — Overleaf's own comment bubbles never reach the
    git bridge.

## Mechanical pending
- ICLR 2027 style files NOT YET PUBLISHED (checked 2026-08-19: github.com/ICLR/Master-Template
  tops out at iclr2026). Keep iclr2026 files as placeholder; re-check periodically
  (`curl -sL https://github.com/ICLR/Master-Template/archive/refs/heads/master.tar.gz | tar tz | grep 2027`).
- Figure 1: schematic draft at fig1_schematic.tex/.pdf (left panel); right-panel
  headline curve placeholder copied as fig1_right_headline_DRAFT.png (from EPS
  figures/issue_1901/ladder_by_metric_grid_v2.png — final version will be re-rendered
  to paper conventions).

## Process (once decisions land)

Inventory pass (fill TBDs) → CLAUDE.md for this repo (notation, glossary terms, voice,
citation rules) → Figure 1 → abstract → intro → setup → results → related work →
review gauntlet → compression → humanize. AI-use log kept from day one (ICLR LLM
disclosure). Every citation fetched programmatically, never from memory.
