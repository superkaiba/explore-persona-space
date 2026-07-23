# One context→answer operator: framing bends it, identity barely moves it (status as of 2026-07-23)

Status note: everything here is committed and verified except two arms flagged INTERIM below (the assistant-test instruct arm and the base-model story round, both running today; they fold into #1639 and #1345 when they land).

## TLDR

- The context→answer map is attached to the turn-structured speaker-label + response-slot format. The chat template, the helpful register, and the speaker's identity all turn out to be nearly irrelevant to it.
- Framing bends the map's coordinates. Chat and plain `User:/Assistant:` text carry the same operator up to a linear change of coordinates: naive cross-format transfer falls short, but a fitted linear alignment recovers each format's ceiling to within 0.005–0.008 (matched-capacity nulls ≈ −0.03). Narrative-story framing collapses the map on identical conversations: verbatim-embedded stories read −0.31 vs chat +0.24 on the same 2,163 rows, and on-policy stories −0.55 vs +0.53 (2,018 rows). The verbatim-embedded arm rules out the corpus explanation and the powered on-policy arm rules out teacher-forcing: what remains is the framing itself. No story read position rescues it (best slot −0.02, CI straddling zero), yet the story operator maps one-way into chat coordinates at 0.56–0.61: what breaks is the story-side context representation.
- Identity moves the operator far less than framing does. Across four fixed-label fiction characters, one pooled map with a single global offset recovers 81–98% of each character's own-map ceiling. Per-character offsets add nothing, and the character-specific remainder is a small slope residual (+0.007 to +0.025, all CIs above zero, the villain largest). Character maps align at Procrustes cosine 0.516 (base) / 0.593 (instruct), above the story↔chat pair (0.455) and below the base↔instruct pair (0.686). Changing the framing moves the operator more than changing the character.
- The assistant sits inside the shared operator (INTERIM: base arm complete, instruct arm running). On row-paired cells answering the same 4,045 real user questions, pooling the assistant with character cells recovers 85–89% of each cell's ceiling, inside the character↔character band (81–92% on base); assistant↔character reparameterization at fixed format recovers 95–97%; assistant~character Procrustes reads 0.540, slightly above the character~character mean. The weak direction is fiction-framed → Q&A (82–83%), the same story-side distortion as above.
- Absolute R² levels undersell the maps. The assistant map itself reads about 0.45 when fit on 250 conversations (0.673 at 5,000; the character cells have n=300), and single stochastic completions put irreducible noise in the target (averaging 5–6 completions per scene lifts every character cell). At matched n and rig, the genuine fiction deficit is about 0.12–0.18 of R², attributed to the scene framing.
- Base model: same structure, weaker and noisier. Pooled-map fractions 0.81–0.92; coordinates much further apart (raw operator cosine 0.293 vs 0.651 on instruct; naive transfer keeps 15–31% of ceiling vs 83–84%), so what instruction tuning mostly does is standardize coordinates for an operator the base model already carries. The story collapse was never tested on base because base cannot write the required stories; a base round holding the answer text fixed is on a GPU right now.

## The framing half

Holding the assistant persona fixed and varying the surface framing on the same conversations (#1345, #825): chat vs plain text is the same operator in different linear coordinates. Naive cross-format transfer falls short (mean deficit −0.10 instruct, −0.43 base), but a general-linear change of coordinates recovers each ceiling to within 0.005–0.008 against matched-capacity nulls. Instruction tuning pulls the two coordinate systems together (raw operator cosine 0.651 vs 0.293 on base; aligned 0.855 vs 0.732). Register and addressee do not matter either: rude, evasive, addressee-free, and AI-relay framings all graded Shared against the chat reference at recovery fractions 0.71–1.00 (#1417). What actually breaks the map is narrative-story framing:

![Same conversations under story, chat, and plain-text framings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2390b30a5e37e483bdb0d2c41ea40a3026c22edd/figures/issue_1345/conversation_paired_stories_assistant/framing_effect_matched_rows_bars.png)

Identical conversations, answers embedded verbatim: the story context map is negative while chat stays positive on the same rows. The failure sits in the story-side context coordinates; the answer representations themselves stay predictable (chat contexts predict the story answers at 0.56). The story operator moved into chat recovers 0.61 against a matched chat ceiling of 0.24, and only the direction into story coordinates fails. A recovery above the target's own ceiling reads as full recovery rather than a better map: the recovery pipeline stacks two ridge fits, and their combined regularization can out-generalize a single direct fit — the same composite advantage the source analysis flags when the forward composition beats the ceiling. Real published fiction is worse still: the chat map transfers to novels at 5% of ceiling or less, and the apparent novel map is mostly an author-level component (#931).

## The identity half

Within the story-scene format, four fixed-label characters (a warm helper, a calm AI, an ordinary person, a theatrical villain) each support their own context→dialogue map (#1310). The similarity battery over those maps (#1639) says they are mostly one object:

![Shared-vs-specific decomposition lattice per persona and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9edaab4fa46a7bd10be7a0fcaae8d2aa8d2760b5/figures/issue_1310/xpersona_decomposition.png)

One pooled map plus one global offset comes within 2–19% of each character's own dedicated map. Raw maps do not cross-apply at all (all 24 off-diagonal transfers negative, −0.22 to −2.6): each 300-point map carries its own centering, basis, and estimation noise. But that is a coordinates problem, not an operator difference. A learned linear input/output alignment around a frozen source map recovers 84–97% (instruct) / 60–79% (base) of each target ceiling, far above nulls of matched capacity.

![Operator similarity statistics with nulls and calibration anchors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7e367f9a5a6a7bd07583aa5595fb28fe4944ee25/figures/issue_1310/xpersona_cosine_reparam.png)

We corrected one statistic mid-line: an earlier "aligned cosine" of ~0.99 turned out to be empty. That statistic (a two-sided Procrustes optimum) only compares singular-value spectra, and maps fit on deliberately scrambled pairings under the same ridge recipe also score ~0.99 on it. The aligned read that survives that null is the data-paired activation-Procrustes cosine (0.516/0.593, with the scrambled-fit null at ≈ 0.002), which puts character-pair similarity between story↔chat (0.455) and base↔instruct (0.686) on a single scale.

Direct identity manipulations agree. Renaming the responder on identical text moves the fiction map by at most 0.009; generating as a character literally named "Assistant" lifts it by at most +0.046 (#1335). The fiction scene framing accounts for 0.14–0.16 of the 0.17 assistant-vs-fiction gap at matched n.

## The assistant, directly (INTERIM)

The same battery on row-paired cells answering the same 4,045 real user questions: assistant plain Q&A, character-labeled plain Q&A, character fiction-framed Q&A (#1335 rungs). Base arm results: pooled fractions 0.85–0.89, inside the 0.81–0.92 band the character↔character pooling reads; assistant↔character reparameterization 95–97% at fixed format and 94–97% into the fiction frame, 82–83% out of it; assistant~character Procrustes 0.540. Identity costs essentially nothing, and the fiction framing costs the little that is lost. These numbers are uncommitted until the instruct arm lands tonight.

## Caveats that ride along

- MODERATE confidence throughout. One model family (Qwen-2.5-7B), single seed per cell in most rounds. The base fiction endpoint swings across independent generation runs (0.269–0.435), so read every base attribution as a bound.
- A corpus-provenance correction is in progress: the shared single-turn answer corpus is instruct-generated, and both models' chat/plain comparator cells teacher-force that same text. Base cells in this line are therefore base-over-instruct-text. Contrasts stand (the text is matched across framings), but absolute base levels carry an off-model-text caveat, and the #1345/#825 body descriptions need correcting.
- The story-collapse claim is instruct-only until today's base round lands.

## Pointers

Clean results: #1639 (shared operator), #1310 (per-character maps and the similarity round history), #1345 (framing collapse), #1417 (register/addressee invariance), #931 (real fiction), #825 (the assistant map, inherited from pretraining), #1335 (gap attribution ladder). Dashboard: https://eps.superkaiba.com/tasks/1639 and siblings. The framing-only predecessor of this doc: docs/mentor_updates/2026-07-21_framing_results.md.
